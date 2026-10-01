# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.models.kimi_k3.common.mtp import fused_mtp_input
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.platforms import current_platform

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5
attn_res_module = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")


def _on_rocm_below_10() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_rocm_version

    return (get_rocm_version() or (0,)) < (10,)


# The Triton bundled with ROCm < 10 (3.7.x) crashes in the AMD
# CanonicalizePointers pass on the kernel's tl.where over pointer tensors. Only
# the num_blocks > 0 loop contains it. ROCm serves AttnRes from
# vllm/models/kimi_k3/amd/ops/attn_res.py instead, so nothing real is lost.
_OLD_ROCM = _on_rocm_below_10()


def _skip_on_old_rocm(num_blocks: int) -> None:
    if _OLD_ROCM and num_blocks > 0:
        pytest.skip("Triton on ROCm < 10 cannot compile this kernel's pointer select")


def _randn_with_row_padding(*shape: int, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return storage[..., : shape[-1]]


def _reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if delta is not None:
        prefix = prefix + delta
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)
    probs = (keys @ qk_weight).softmax(dim=-1)
    output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
    if output_norm_weight is not None:
        output = F.rms_norm(output, (HIDDEN_SIZE,), output_norm_weight, EPS)
    return output, prefix


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "row_padding",
        "write_block",
        "has_delta",
        "backend",
    ),
    [
        pytest.param(1, 0, 0, True, False, "triton", id="triton-empty"),
        pytest.param(1, 0, 0, True, True, "triton", id="triton-empty-add"),
        pytest.param(17, 5, 7, True, False, "triton", id="triton-write"),
        pytest.param(17, 5, 7, True, True, "triton", id="triton-write-add"),
        pytest.param(3, 8, 0, False, False, "triton", id="triton-full"),
        pytest.param(3, 8, 0, False, True, "triton", id="triton-full-add"),
        pytest.param(1, 4, 0, False, True, "nvidia", id="nvidia-single-token"),
        pytest.param(320, 1, 0, False, True, "nvidia", id="nvidia-1"),
        pytest.param(320, 4, 0, False, True, "nvidia", id="nvidia-4"),
        pytest.param(320, 8, 0, False, True, "nvidia", id="nvidia-8"),
    ],
)
def test_attn_res(
    num_tokens: int,
    num_blocks: int,
    row_padding: int,
    write_block: bool,
    has_delta: bool,
    backend: str,
):
    if backend == "nvidia" and not current_platform.is_device_capability_family(100):
        pytest.skip("NVIDIA AttnRes requires the SM100 family")
    _skip_on_old_rocm(num_blocks)

    prefix = _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=row_padding)
    delta = (
        _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=row_padding)
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens, MAX_BLOCKS, HIDDEN_SIZE, padding=row_padding
    )
    norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    original_blocks = blocks.clone()
    expected, expected_prefix = _reference(
        prefix.clone(),
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )
    block_write_idx = num_blocks if write_block else -1

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    if write_block:
        original_blocks[:, block_write_idx].copy_(expected_prefix)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("num_blocks", range(MAX_BLOCKS + 1))
def test_attn_res_block_counts(num_blocks: int):
    _skip_on_old_rocm(num_blocks)
    prefix = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(
        1, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = torch.ones_like(norm_weight)
    expected, _ = _reference(
        prefix.clone(),
        None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )

    actual = attn_res(
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        -1,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def test_attn_res_without_output_norm():
    _skip_on_old_rocm(MAX_BLOCKS)
    prefix = torch.randn(7, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix)
    blocks = torch.randn(
        7, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    expected, _ = _reference(
        prefix.clone(), delta, blocks, norm_weight, qk_weight, None, MAX_BLOCKS
    )

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        None,
        MAX_BLOCKS,
        -1,
        EPS,
        0.0,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("num_blocks", "has_delta", "has_output_norm", "write_block"),
    [
        (0, False, True, True),
        (4, True, True, True),
        (4, False, True, False),
        (8, True, False, False),
    ],
)
def test_sm100_variants_do_not_fall_back_to_triton(
    monkeypatch,
    num_blocks: int,
    has_delta: bool,
    has_output_norm: bool,
    write_block: bool,
):
    if not current_platform.is_device_capability_family(100) or not hasattr(
        torch.ops._C, "kimi_k3_attn_res"
    ):
        pytest.skip("native AttnRes requires an SM100 build")

    class FailingKernel:
        def __getitem__(self, grid):
            raise AssertionError(f"unexpected Triton launch with grid {grid}")

    monkeypatch.setattr(attn_res_module, "_attn_res_kernel", FailingKernel())
    prefix = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    delta = torch.zeros_like(prefix) if has_delta else None
    blocks = torch.randn(
        1, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = torch.randn_like(norm_weight) / HIDDEN_SIZE**0.5
    output_norm_weight = torch.ones_like(norm_weight) if has_output_norm else None

    attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        num_blocks if write_block else -1,
        EPS,
        EPS,
    )


def _require_native_attn_res_fp8() -> None:
    if not current_platform.is_device_capability_family(100) or not hasattr(
        torch.ops._C, "kimi_k3_attn_res_fp8"
    ):
        pytest.skip("native FP8 AttnRes requires an SM100 build")


def _assert_packed_fp8_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_q, actual_scales = actual
    expected_q, expected_scales = expected
    torch.testing.assert_close(
        actual_q.view(torch.uint8), expected_q.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)
    assert actual_q.is_contiguous()
    assert actual_scales.stride() == expected_scales.stride()
    num_tokens, scale_words = actual_scales.shape
    padded_tokens = actual_scales.stride(1)
    # The last column's trailing padding is outside empty_strided's storage.
    for scales in (actual_scales, expected_scales):
        padding = scales.as_strided(
            (scale_words - 1, padded_tokens - num_tokens),
            (padded_tokens, 1),
            storage_offset=num_tokens,
        )
        assert torch.count_nonzero(padding).item() == 0


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "has_delta",
        "has_output_norm",
        "write_block",
        "capture_graph",
    ),
    [
        *[
            pytest.param(5, n, True, True, False, False, id=f"blocks-{n}")
            for n in range(MAX_BLOCKS + 1)
        ],
        pytest.param(1, 0, False, True, True, False, id="first-block-write"),
        pytest.param(2, 1, True, True, False, False, id="padding-two"),
        pytest.param(3, 1, True, True, False, False, id="padding-one"),
        pytest.param(4, 1, True, True, False, False, id="no-padding"),
        pytest.param(17, 4, True, True, True, False, id="block-write"),
        pytest.param(129, 8, False, True, False, True, id="graph-no-delta"),
        pytest.param(17, 8, True, False, False, False, id="without-output-norm"),
        pytest.param(16377, 6, True, True, True, True, id="prefill-block-write"),
    ],
)
def test_attn_res_fp8_matches_two_kernels(
    num_tokens: int,
    num_blocks: int,
    has_delta: bool,
    has_output_norm: bool,
    write_block: bool,
    capture_graph: bool,
):
    """Fusion preserves quantized outputs and all residual-state updates."""
    _require_native_attn_res_fp8()
    torch.manual_seed(0)
    prefix = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix) if has_delta else None
    capacity = max(1, num_blocks + int(write_block))
    blocks = torch.randn(
        num_tokens, capacity, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    expected_prefix, expected_blocks = prefix.clone(), blocks.clone()
    norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = torch.randn_like(norm_weight) / HIDDEN_SIZE**0.5
    output_norm_weight = (
        1 + 0.1 * torch.randn_like(norm_weight) if has_output_norm else None
    )
    block_write_idx = num_blocks if write_block else -1

    def unfused():
        output = ops.kimi_k3_attn_res(
            expected_prefix,
            delta,
            expected_blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            block_write_idx,
            EPS,
            EPS,
        )
        return per_token_group_quant_fp8_packed_for_deepgemm(
            output, 128, eps=1e-10, use_ue8m0=True
        )

    def fused():
        return ops.kimi_k3_attn_res_fp8(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            block_write_idx,
            EPS,
            EPS,
        )

    def check(actual, expected):
        _assert_packed_fp8_equal(actual, expected)
        torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
        torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)

    expected = unfused()
    actual = fused()
    check(actual, expected)
    if capture_graph:
        torch.accelerator.synchronize()
        unfused_graph, fused_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(unfused_graph):
            expected = unfused()
        with torch.cuda.graph(fused_graph):
            actual = fused()
        for _ in range(3):
            unfused_graph.replay()
            fused_graph.replay()
            check(actual, expected)


def test_attn_res_fp8_scale_rounding_boundaries():
    """Match scale clamping and upward exponent rounding at BF16 boundaries."""
    _require_native_attn_res_fp8()
    boundary = torch.ldexp(
        torch.full((4,), 448.0, device="cuda"),
        torch.tensor([-20, -8, 0, 8], device="cuda"),
    ).to(torch.bfloat16)
    group_maxima = torch.cat(
        (
            torch.tensor(
                [0, 2**-133, 2**-126, 1e-8], device="cuda", dtype=torch.bfloat16
            ),
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, float("inf"))),
        )
    ).repeat(4)[: HIDDEN_SIZE // 128]
    prefix = group_maxima.repeat_interleave(128).expand(5, -1).clone()
    prefix[:, 1::2].neg_()
    blocks = torch.zeros(5, 1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = torch.zeros_like(norm_weight)
    output = ops.kimi_k3_attn_res(
        prefix, None, blocks, norm_weight, qk_weight, None, 0, -1, EPS, EPS
    )
    expected = per_token_group_quant_fp8_packed_for_deepgemm(
        output, 128, eps=1e-10, use_ue8m0=True
    )
    actual = ops.kimi_k3_attn_res_fp8(
        prefix, None, blocks, norm_weight, qk_weight, None, 0, -1, EPS, EPS
    )
    _assert_packed_fp8_equal(actual, expected)


@pytest.mark.parametrize("num_tokens", [0, 1, 17])
def test_fused_mtp_input(num_tokens: int):
    positions = torch.arange(num_tokens, device="cuda")
    inputs_embeds = _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=7)
    previous_hidden_states = _randn_with_row_padding(
        num_tokens, HIDDEN_SIZE, padding=11
    )
    enorm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    masked_inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    expected = torch.cat(
        (
            F.rms_norm(masked_inputs_embeds, (HIDDEN_SIZE,), enorm_weight, EPS),
            F.rms_norm(previous_hidden_states, (HIDDEN_SIZE,), hnorm_weight, EPS),
        ),
        dim=-1,
    )
    actual = fused_mtp_input(
        positions,
        inputs_embeds,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert actual.shape == (num_tokens, 2 * HIDDEN_SIZE)
    assert actual.is_contiguous()
