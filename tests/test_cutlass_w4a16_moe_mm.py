# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test comparing Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows,
    quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types


def mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].to(torch.float32)
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].to(torch.float32)
    x_max = x_max * 8.0 / 7.0
    amax = torch.where(x_max > -x_min, x_max, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    ) + 8
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return x_int4.reshape(*x.shape[:-1], x.shape[-1] // 2), scales.reshape(
        -1, sf_vec_size
    )


def mxint4_dequantize(
    x: torch.Tensor, scales: torch.Tensor, sf_vec_size: int = 32
) -> torch.Tensor:
    original_shape = x.shape
    x = x.reshape(-1, sf_vec_size // 2)
    scales = scales.reshape(-1, 1)
    assert x.shape[0] == scales.shape[0]
    x_int8 = torch.empty(
        x.shape[0], sf_vec_size // 2, 2, dtype=torch.int8, device=x.device
    )
    x_int8[..., 0] = x & 0x0F
    x_int8[..., 1] = (x >> 4) & 0x0F
    x = x_int8.to(torch.float32)
    x_scaled = x.reshape(-1, sf_vec_size) * scales

    return x_scaled.reshape(original_shape[:-1] + (original_shape[-1] * 2,))


def cutlass_quantize(
    atype: torch.dtype,
    w: torch.Tensor,
    wtype: ScalarType,
    stype: torch.dtype | None,
    group_size: int | None,
    zero_points: bool = False,
):
    """
    Quantize weights into W4 and compute reference dequantized weights.

    Encoding/reordering of weights and packing of scales is deferred
    until after all experts are combined.
    """
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w, wtype, group_size=group_size, zero_points=zero_points
    )

    # Since scales are later cast to fp8, recompute w_ref in atype here.
    w_ref = (
        w_q.to(torch.float32)
        * w_s.to(atype).to(torch.float32).repeat_interleave(group_size, dim=0)
    ).to(atype)

    # Bit mask prevents sign extension of int4 when packing.
    w_q = pack_rows(w_q & 0x0F, wtype.size_bits, *w_q.shape)
    # Make weights row-major (N, K).
    w_q = w_q.t().contiguous()

    return w_ref, w_q, w_s.to(atype), w_zp


@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
@pytest.mark.parametrize("bs", [1, 64, 128, 384])
@pytest.mark.parametrize("M", [128, 1024, 2048])
@pytest.mark.parametrize("N", [128, 1024, 2048, 7168])
@pytest.mark.parametrize("K", [2048, 4096, 7168, 16384, 32768])
@pytest.mark.parametrize("group_size", [32])
@pytest.mark.parametrize("maybe_schedule", [None])
def test_cutlass_w4a16_moe_mm(
    bs: int,
    M: int,
    N: int,
    K: int,
    group_size: int,
    maybe_schedule: str | None,
):
    alignment = 16
    torch.random.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    Ms = torch.randint(0, M // alignment, (bs,)) * alignment
    print(f"{Ms=}")
    M_full = torch.sum(Ms).item()
    A = torch.randn(M_full, K, device=device, dtype=dtype)
    B = torch.randn(bs, K, N, device=device, dtype=dtype)

    B_ref, B_int4, B_scales = [], [], []
    for i in range(bs):
        B_ref_, B_int4_, B_scales_, _ = cutlass_quantize(
            torch.bfloat16,
            B[i],
            scalar_types.int4,
            torch.bfloat16,
            group_size,
            zero_points=False,
        )
        B_ref.append(B_ref_)
        B_int4.append(B_int4_.view(torch.int32))
        B_scales.append(B_scales_)
    B_ref = torch.stack(B_ref)
    B_int4 = torch.stack(B_int4)
    B_scales = torch.stack(B_scales)

    out_tensors = torch.empty(M_full, N, device=device, dtype=dtype)

    # swap AB
    problem_sizes = torch.zeros(bs, 3, device=device, dtype=torch.int32)
    problem_sizes[:, 0] = N
    problem_sizes[:, 2] = K
    for i in range(bs):
        problem_sizes[i, 1] = Ms[i]

    # Strides for memory layout
    # A strides: [K, 1] for row-major [M, K]
    a_strides = torch.full((bs,), K, device=device, dtype=torch.int64)

    B_int4_cutlass, b_strides = ops.cutlass_reorder_int4b_grouped(B_int4)

    # C strides: [N, 1] for row-major [M, N]
    c_strides = torch.full((bs,), N, device=device, dtype=torch.int64)

    # sizeof(StrideS) = 16 bytes, so we need to use 2xint64 to encode it
    group_scale_strides = torch.zeros((bs, 2), device=device, dtype=torch.int64)
    group_scale_strides[:, 0] = N

    offsets = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64),
            torch.cumsum(Ms, dim=0)[:-1],
        ]
    ).to(device=device)

    # Call the kernel
    ops.cutlass_w4a16_moe_mm(
        out_tensors,
        A,
        B_int4_cutlass,
        B_scales.to(torch.bfloat16),
        group_size,
        offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        group_scale_strides,
        maybe_schedule=maybe_schedule,
    )

    # ========reference implementation========
    out_tensors_ref = torch.empty(M_full, N, device=device, dtype=dtype)
    offset = 0
    for i in range(bs):
        out_tensors_ref[offset : offset + Ms[i]] = A[offset : offset + Ms[i]] @ B_ref[i]
        offset += Ms[i]
    # print(f"{out_tensors=}")
    # print(f"{out_tensors_ref=}")
    torch.testing.assert_close(out_tensors, out_tensors_ref, atol=5, rtol=1e-2)
