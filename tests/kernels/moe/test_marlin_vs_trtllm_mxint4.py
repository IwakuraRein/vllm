# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test comparing Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE."""

import pytest
import torch
import numpy as np
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    int4_w4afp8_moe_quant_config,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
    prepare_static_weights_for_trtllm_mxint4_moe,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from flashinfer import cutlass_fused_moe
import triton
from vllm.model_executor.layers.fused_moe import fused_experts

import logging
from vllm import _custom_ops as ops

from flashinfer.testing.utils import bench_gpu_time_with_cupti

import nvtx

def mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to MXINT4 with block scaling (group_size=sf_vec_size).

    Returns:
        - uint8 packed (2 INT4/byte): [..., k//2] - stores SIGNED INT4 [-8, 7]
        - scales in BF16: [..., k//sf_vec_size]
    """
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].to(torch.float32)
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].to(torch.float32)
    x_max = x_max * 8.0 / 7.0
    amax = torch.where(x_max > -x_min, x_max, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    )
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return (
        x_int4.to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2),
        scales.to(x.dtype).reshape(*x.shape[:-1], x.shape[-1] // sf_vec_size),
    )


def mxint4_quantize_moe_weights(
    weights_bf16: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MoE weights [e, n, k] to MxInt4 format.

    Args:
        weights_bf16: BF16 weights of shape [num_experts, out_features, in_features]
        group_size: Quantization group size (default: 32)

    Returns:
        - weights_mxint4: Quantized weights [e, n, k//2] uint8
        - scales_mxint4: Quantization scales [e, n, k//group_size] bf16
    """
    e = weights_bf16.shape[0]
    weight_list = []
    scale_list = []

    for i in range(e):
        w_q, w_s = mxint4_quantize(weights_bf16[i], sf_vec_size=group_size)
        weight_list.append(w_q)
        scale_list.append(w_s)

    return torch.stack(weight_list), torch.stack(scale_list)


__all__ = [
    "mxint4_quantize",
    "mxint4_quantize_moe_weights",
    "marlin_quantize_moe_weights",
]


def marlin_quantize_moe_weights(
    weights_bf16: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MoE weights [e, n, k] to Marlin INT4 format.

    Args:
        weights_bf16: BF16 weights of shape [num_experts, out_features, in_features]
        group_size: Quantization group size (default: 32)

    Returns:
        - weights_marlin: Marlin quantized weights [e, k//8, n] int32
        - scales_marlin: Marlin quantization scales [e, k//group_size, n] bf16
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )

    e, n, k = weights_bf16.shape
    weight_list = []
    scale_list = []

    for i in range(e):
        # Transpose for Marlin: [n, k] → [k, n]
        w_t = weights_bf16[i].T.contiguous()
        _, w_q, w_s, _, _, _ = marlin_quantize(
            w_t, scalar_types.uint4b8, group_size, act_order=False
        )
        weight_list.append(w_q)
        scale_list.append(w_s)

    # Stack to get [e, ...] shape
    weights_marlin = torch.stack(weight_list)  # [e, k // 8, n]
    scales_marlin = torch.stack(scale_list)  # [e, k // group_size, n]

    return weights_marlin, scales_marlin


@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("n", [7168])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("e", [384])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("group_size", [32])
def test_marlin_vs_trtllm_mxint4_moe_kimik2(monkeypatch, m, n, k, e, topk, group_size):
    """Compare Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE.

    Uses mxint4_quantize() to generate common INT4 weights + BF16 scales,
    then runs both Marlin and TRT-LLM kernels and compares outputs.
    """
    pytest.importorskip("flashinfer")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_INT4", "1")

    torch.cuda.manual_seed(0)

    dtype = torch.bfloat16

    # DeepSeekV3 routing config (from Kimi-K2-Thinking config.json)
    n_group = 1  # n_group from model config
    topk_group = 1  # topk_group from model config
    routed_scaling = 2.827  # routed_scaling_factor from model config

    # Input - realistic activation range for LLM (after LayerNorm: mean~0, std~1)
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.5

    # Generate routing logits and bias (DeepSeekV3 expects float logits)
    # Realistic ranges: logits typically [-3, 3], bias [-2, 2]
    routing_logits = torch.randn((m, e), device="cuda", dtype=torch.float32) * 1.5
    routing_bias = torch.randn(e, device="cuda", dtype=torch.float32) * 0.8

    # 1. Generate BF16 weights (SHARED between both paths)
    # Realistic weight initialization: Xavier/Glorot uniform scaling
    # std = sqrt(2 / (fan_in + fan_out))
    std_w1 = (2.0 / (k + 2 * n)) ** 0.5
    std_w2 = (2.0 / (n + k)) ** 0.5
    w1_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) * std_w1
    w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) * std_w2

    # === Path 1: TRT-LLM FlashInfer MXINT4 MoE ===
    # Similar to: if self.use_flashinfer_mxint4_moe
    # Quantize using MXINT4 method (signed INT4)
    w1_int4, w1_scales = mxint4_quantize_moe_weights(w1_bf16, group_size)
    w2_int4, w2_scales = mxint4_quantize_moe_weights(w2_bf16, group_size)

    trtllm_weights = prepare_static_weights_for_trtllm_mxint4_moe(
        gemm1_weights=w1_int4,
        gemm1_scales=w1_scales,
        gemm2_weights=w2_int4,
        gemm2_scales=w2_scales,
    )

    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe

    # Routing handled internally by trtllm_mxint4_block_scale_moe
    trtllm_output = trtllm_mxint4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias.to(torch.bfloat16),
        hidden_states=a,
        gemm1_weights=trtllm_weights["gemm1_weights"],
        gemm1_weights_scale=trtllm_weights["gemm1_scales"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=trtllm_weights["gemm2_weights"],
        gemm2_weights_scale=trtllm_weights["gemm2_scales"],
        num_experts=e,
        top_k=topk,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=n,
        local_expert_offset=0,
        local_num_experts=e,
        routed_scaling_factor=routed_scaling,
        routing_method_type=RoutingMethodType.DeepSeekV3,
        enable_pdl=None,
        output=None,
        tune_max_num_tokens=8192,
    ).to(dtype)

    # === Path 2: Marlin INT4 MoE ===
    # Similar to: else (non-flashinfer path)
    # Quantize using Marlin's method (UINT4b8)
    w1_marlin, w1_scales_marlin = marlin_quantize_moe_weights(w1_bf16, group_size)
    w2_marlin, w2_scales_marlin = marlin_quantize_moe_weights(w2_bf16, group_size)

    # Use production routing kernel (same as router.select_experts internally uses)
    topk_weights, topk_ids = grouped_topk(
        hidden_states=a,
        gating_output=routing_logits,
        topk=topk,
        renormalize=False,  # DeepSeekV3 doesn't renormalize
        num_expert_group=n_group,
        topk_group=topk_group,
        scoring_func="sigmoid",  # DeepSeekV3 uses sigmoid
        routed_scaling_factor=routed_scaling,
        e_score_correction_bias=routing_bias,
    )

    marlin_output = fused_marlin_moe(
        a,
        w1_marlin,
        w2_marlin,
        None,
        None,
        w1_scales_marlin,
        w2_scales_marlin,
        None,  # gating_output not needed when topk_weights/ids provided
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=None,
        global_scale1=None,
        global_scale2=None,
        g_idx1=None,
        g_idx2=None,
        input_global_scale1=None,
        input_global_scale2=None,
        sort_indices1=None,
        sort_indices2=None,
        w1_zeros=None,
        w2_zeros=None,
        input_dtype=dtype,
        quant_type_id=scalar_types.uint4b8.id,
        is_k_full=True,
    )

    # Sanity check: manually compute BF16 reference for comparison
    # Use same routing as Marlin path for consistency
    bf16_output = torch.zeros((m, k), device="cuda", dtype=dtype)
    for token_idx in range(m):
        for expert_rank in range(topk):
            expert_id = topk_ids[token_idx, expert_rank].item()
            weight = topk_weights[token_idx, expert_rank].item()
            # w1: [2*n, k] @ [k] -> [2*n]
            up_gate = a[token_idx] @ w1_bf16[expert_id].T  # [2*n]
            gate, up = up_gate.chunk(2, dim=0)
            intermediate = torch.nn.functional.silu(gate) * up  # [n]
            # w2: [k, n] @ [n] -> [k]
            expert_out = intermediate @ w2_bf16[expert_id].T  # [k]
            bf16_output[token_idx] += weight * expert_out
    # Compare against BF16 reference.
    torch.testing.assert_close(marlin_output, bf16_output, atol=0.3, rtol=1.0)
    torch.testing.assert_close(trtllm_output, bf16_output, atol=0.3, rtol=1.0)

    # Compare against each other for sanity.
    # Note: Different quantization schemes (UINT4b8 vs signed MXINT4) cause
    # some differences
    torch.testing.assert_close(marlin_output, trtllm_output, atol=0.3, rtol=6.0)


def bench_marlin_vs_triton_mxint4_moe(bs, intermediate_size, hidden_size, expert_num, topk, cutlass_schedule=None):
    group_size = 32

    x = torch.randn((bs, hidden_size), device="cuda", dtype=torch.bfloat16)
    w13_bf16 = torch.randn((expert_num, 2 * intermediate_size, hidden_size), device="cuda", dtype=torch.bfloat16)
    w2_bf16 = torch.randn((expert_num, hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16)

    w13_int4, w13_scales = mxint4_quantize_moe_weights(w13_bf16, group_size)
    w2_int4, w2_scales = mxint4_quantize_moe_weights(w2_bf16, group_size)

    # DeepSeekV3 routing config (from Kimi-K2-Thinking config.json)
    # n_group = 1  # n_group from model config
    # topk_group = 1  # topk_group from model config
    # routed_scaling = 2.827  # routed_scaling_factor from model config
    routing_logits = torch.randn((bs, expert_num), device="cuda", dtype=torch.float32) * 1.5
    routing_bias = torch.randn(expert_num, device="cuda", dtype=torch.float32) * 0.8
    topk_weights, topk_ids = grouped_topk(
        hidden_states=x,
        gating_output=routing_logits,
        topk=topk,
        renormalize=False,  # DeepSeekV3 doesn't renormalize
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",  # DeepSeekV3 uses sigmoid
        routed_scaling_factor=2.827,
        e_score_correction_bias=routing_bias,
    )


    ## ================== Cutlass INT4 MoE ==================
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        cutlass_moe_w4a16_bf16,
    )
    device = "cuda"
    a_strides1_c_strides2 = torch.full(
        (expert_num,),
        hidden_size,
        device=device,
        dtype=torch.int64,
    )
    a_strides2 = torch.full(
        (expert_num,),
        intermediate_size,
        device=device,
        dtype=torch.int64,
    )
    c_strides1 = torch.full(
        (expert_num,),
        2 * intermediate_size,
        device=device,
        dtype=torch.int64,
    )

    # S (group-wise scales)
    # sizeof(StrideS) = 16 bytes, so we need to use 2xint64 to encode it
    s_strides1 = torch.zeros(
        (expert_num, 2), device=device, dtype=torch.int64
    )
    s_strides1[:, 0] = 2 * intermediate_size

    s_strides2 = torch.zeros(
        (expert_num, 2), device=device, dtype=torch.int64
    )
    s_strides2[:, 0] = hidden_size
    quant_config=int4_w4a16_moe_quant_config(
        w1_scale=w13_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )
    w13_cutlass, b_strides1 = ops.cutlass_reorder_int4b_grouped(w13_int4.view(torch.int32))
    w2_cutlass, b_strides2 = ops.cutlass_reorder_int4b_grouped(w2_int4.view(torch.int32))
    cutlass_fn = lambda: cutlass_moe_w4a16_bf16(
        x,
        w13_cutlass,
        w2_cutlass,
        topk_weights,
        topk_ids,
        quant_config=quant_config,
        activation="silu",
        global_num_experts=expert_num,
        expert_map=None,
        a_strides1=a_strides1_c_strides2,
        a_strides2=a_strides2,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        c_strides1=c_strides1,
        c_strides2=a_strides1_c_strides2,
        s_strides1=s_strides1,
        s_strides2=s_strides2,
        group_size=group_size,
        maybe_schedule=cutlass_schedule,
    )
    with nvtx.annotate("cutlass"):
        for _ in range(10):
            cutlass_output = cutlass_fn()
    # ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    #     cutlass_fn, rep=100, quantiles=quantiles
    # )
    ms = np.median(bench_gpu_time_with_cupti(cutlass_fn, l2_flush=True, dry_run_iters=10, repeat_iters=100))
    print(f"    Cutlass: {ms:.2f} ms")

    # ================== Default INT4 MoE ==================
    # triton_fn = lambda: fused_experts(
    #     x,
    #     w13_int4,
    #     w2_int4,
    #     topk_weights,
    #     topk_ids,
    #     inplace=False,
    #     activation="silu",
    #     apply_router_weight_on_input = False, # TODO: need double check
    #     global_num_experts=expert_num,
    #     expert_map=None,
    #     quant_config=int4_w4a16_moe_quant_config(
    #         w1_scale=w13_scales,
    #         w2_scale=w2_scales,
    #         w1_zp=None,
    #         w2_zp=None,
    #         block_shape=[0, group_size],
    #     ),
    # )
    # # warmup run
    # triton_output = triton_fn()
    # ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    #     triton_fn, rep=100, quantiles=quantiles
    # )
    # print(f"    Triton: {ms:.2f} ms")



    # ================== Marlin INT4 MoE ==================
    w1_marlin, w13_scales_marlin = marlin_quantize_moe_weights(w13_bf16, group_size)
    w2_marlin, w2_scales_marlin = marlin_quantize_moe_weights(w2_bf16, group_size)
    marlin_fn = lambda: fused_marlin_moe(
        x,
        w1_marlin,
        w2_marlin,
        None,
        None,
        w13_scales_marlin,
        w2_scales_marlin,
        None,  # gating_output not needed when topk_weights/ids provided
        topk_weights,
        topk_ids,
        global_num_experts=expert_num,
        expert_map=None,
        global_scale1=None,
        global_scale2=None,
        g_idx1=None,
        g_idx2=None,
        input_global_scale1=None,
        input_global_scale2=None,
        sort_indices1=None,
        sort_indices2=None,
        w1_zeros=None,
        w2_zeros=None,
        input_dtype=torch.bfloat16,
        quant_type_id=scalar_types.uint4b8.id,
        is_k_full=True,
    )
    # with nvtx.annotate("marlin"):
    marlin_output = marlin_fn()
    # ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
    #     marlin_fn, rep=100, quantiles=quantiles
    # )
    ms = np.median(bench_gpu_time_with_cupti(marlin_fn, l2_flush=True, dry_run_iters=10, repeat_iters=100))
    print(f"    Marlin: {ms:.2f} ms")

    def MSE(a:torch.Tensor, b:torch.Tensor) -> float:
        return ((a - b) ** 2).mean().item()
    bf16_output = torch.zeros((bs, hidden_size), device="cuda", dtype=torch.bfloat16)
    for token_idx in range(bs):
        for expert_rank in range(topk):
            expert_id = topk_ids[token_idx, expert_rank].item()
            weight = topk_weights[token_idx, expert_rank].item()
            # w1: [2*n, k] @ [k] -> [2*n]
            up_gate = x[token_idx] @ w13_bf16[expert_id].T  # [2*n]
            gate, up = up_gate.chunk(2, dim=0)
            intermediate = torch.nn.functional.silu(gate) * up  # [n]
            # w2: [k, n] @ [n] -> [k]
            expert_out = intermediate @ w2_bf16[expert_id].T  # [k]
            bf16_output[token_idx] += weight * expert_out
    # is_triton_close = torch.allclose(triton_output, bf16_output, atol=0.3, rtol=1.0)
    is_marlin_close = torch.allclose(marlin_output, bf16_output, atol=0.3, rtol=1.0)
    is_cutlass_close = torch.allclose(cutlass_output, bf16_output, atol=0.3, rtol=1.0)
    # print(f'    {is_triton_close=}')
    # print(f'    {is_marlin_close=}')
    print(f'    {is_cutlass_close=}')
    # if not is_triton_close:
    #     print(f'    {triton_output=}')
    #     print(f'    {bf16_output=}')
    #     print(f"    MSE: {MSE(triton_output, bf16_output)}")

    if not is_marlin_close:
        print(f'    {marlin_output=}')
        print(f'    {bf16_output=}')
        print(f"    MSE: {MSE(marlin_output, bf16_output)}")

    if not is_cutlass_close:
        print(f'    {cutlass_output=}')
        print(f'    {bf16_output=}')
        print(f"    MSE: {MSE(cutlass_output, bf16_output)}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.ERROR)
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(device="cuda")
    for expert_num, intermediate_size, hidden_size in [(48, 2048, 7168)]:
        print(f"expert_num={expert_num}, intermediate_size={intermediate_size}, hidden_size={hidden_size}:")
        for bs in [1]:
            print(f"  bs={bs}:")
            # for cutlass_schedule in ["Kernel_128x16_2x1x1_Coop", 'Kernel_256x32_1x1x1_Coop', 'Kernel_256x64_1x1x1_Coop', 'Kernel_256x128_2x1x1_Coop', 'Kernel_128x256_2x1x1_Coop']:
            for cutlass_schedule in ["Kernel_128x16_2x1x1_Coop"]:
                print(f"    cutlass_schedule={cutlass_schedule}:")
                bench_marlin_vs_triton_mxint4_moe(
                    intermediate_size=intermediate_size,
                    hidden_size=hidden_size,
                    bs=bs,
                    expert_num=expert_num,
                    topk=1,
                )