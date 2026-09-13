# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cute_dsl_fused_moe_nvfp4,
    has_flashinfer_cutedsl_moe_nvfp4,
    has_flashinfer_cutedsl_moe_nvfp4_w4a16,
)


class FlashInferCuteDSLExperts(mk.FusedMoEExpertsModular):
    """
    CuteDSL NvFP4 MoE experts using the FlashInfer functional API.

    Uses Standard activation format (non-batched). The kernel handles
    routing, expert computation, and reduction internally.
    Supports expert parallelism natively.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.weight_quant_dtype == "nvfp4"
        assert quant_config.quant_dtype in ("nvfp4", None)
        self.use_a16 = quant_config.quant_dtype is None
        self.out_dtype = moe_config.in_dtype
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.local_expert_offset = self.ep_rank * self.local_num_experts
        self.gemm1_alpha = quant_config.gemm1_alpha
        self.gemm1_beta = quant_config.gemm1_beta
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.situ_beta = moe_config.activation_situ_beta
        self.situ_linear_beta = moe_config.activation_situ_linear_beta

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_a16:
            return
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported or activation_key is not None:
            return supported, reason
        if moe_config.in_dtype != torch.bfloat16:
            return False, "FlashInfer CuTe DSL NVFP4 W4A16 requires BF16 activations"
        if not has_flashinfer_cutedsl_moe_nvfp4_w4a16():
            return False, (
                "FlashInfer CuTe DSL NVFP4 W4A16 requires a FlashInfer version "
                "with quant_mode support"
            )
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_moe_nvfp4()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
            MoEActivation.SITU,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        assert self.hidden_dim == (K if self.use_a16 else K * 2)
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        quant_kwargs: dict[str, str] = {}
        if self.use_a16:
            assert hidden_states.dtype == torch.bfloat16
            assert a1q_scale is None
            x_sf = None
            quant_kwargs["quant_mode"] = "w4a16"
        else:
            assert a1q_scale is not None
            # The functional API expects x_sf as (M, K//16, 1).
            x_sf = a1q_scale.unsqueeze(-1)

        # The kernel defaults swiglu_{alpha,beta,limit} to the plain-SwiGLU
        # values, so only forward the ones the model actually sets.
        swiglu_params: dict[str, float | None] = {}
        if activation == MoEActivation.SILU:
            swiglu_params = {"swiglu_limit": self.gemm1_clamp_limit}
        elif activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ):
            swiglu_params = {
                "swiglu_alpha": self.gemm1_alpha,
                "swiglu_beta": self.gemm1_beta,
                "swiglu_limit": self.gemm1_clamp_limit,
            }
        elif activation == MoEActivation.SITU:
            # The cute_dsl kernel keys SiTU on situ_beta and requires
            # activation_type to stay a base type (ActivationType.Situ is
            # rejected by normalize_cute_dsl_moe_activation_type), so the
            # Swiglu base type is passed below and SiTU rides the betas.
            if self.situ_beta is None:
                raise ValueError(
                    "SITU activation requires moe_config.activation_situ_beta"
                )
            swiglu_params = {
                "situ_beta": self.situ_beta,
                "situ_linear_beta": self.situ_linear_beta,
            }
        swiglu_kwargs = {k: v for k, v in swiglu_params.items() if v is not None}

        flashinfer_cute_dsl_fused_moe_nvfp4(
            x=hidden_states,
            x_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.float(),
            w1_weight=w1,
            w1_weight_sf=self.w1_scale,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=None if self.use_a16 else self.a2_gscale,
            w2_weight=w2,
            w2_weight_sf=self.w2_scale,
            w2_alpha=self.g2_alphas,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=activation_to_flashinfer_int(
                MoEActivation.SILU if activation == MoEActivation.SITU else activation
            ),
            **swiglu_kwargs,
            **quant_kwargs,
        )
