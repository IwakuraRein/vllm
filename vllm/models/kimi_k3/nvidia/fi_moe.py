# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer NVFP4 MegaMoE experts for Kimi K3."""

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI


class KimiK3MegaMoEExpertsFI(DeepseekV4MegaMoEExpertsFI):
    """Run SiTU on rank-local tokens with NVFP4 or requantized MXFP4 experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        activation: str,
        activation_beta: float | None,
        activation_linear_beta: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vllm_config,
            activation=activation,
            situ_beta=activation_beta or 1.0,
            situ_linear_beta=activation_linear_beta or None,
            **kwargs,
        )
        quant_config = vllm_config.quant_config
        self._reciprocal_global_scales = isinstance(
            quant_config, CompressedTensorsConfig
        )
        if not self._nvfp4_prequant and self._uses_nvfp4_weights(quant_config):
            self._nvfp4_prequant = True
            self._realloc_nvfp4_params()
        if self._nvfp4_prequant:
            for name in (
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            ):
                getattr(self, name).dummy_weight_value = 1.0

    def _uses_nvfp4_weights(self, quant_config) -> bool:
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            return not quant_config.is_layer_excluded(self.prefix) and (
                quant_config._resolve_quant_algo(self.prefix)
                in ("NVFP4", "W4A16_NVFP4")
            )
        if isinstance(quant_config, ModelOptNvFp4Config):
            return (
                quant_config.is_checkpoint_nvfp4_serialized
                and not quant_config.is_layer_excluded(self.prefix)
            )
        if isinstance(quant_config, CompressedTensorsConfig):
            # Match the checkpoint's expert target and broad Linear targets.
            probe = nn.Linear(1, 1, bias=False, device="meta")
            scheme = quant_config.get_scheme_dict(probe, self.prefix)
            return scheme is not None and quant_config._is_nvfp4_format(
                scheme.get("weights")
            )
        return False

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self._reciprocal_global_scales and (
            "weight_scale_2" in weight_name or "input_scale" in weight_name
        ):
            loaded_weight = loaded_weight.reciprocal()
        return super().weight_loader(
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success,
        )


KimiK3MegaMoEExpertsFI.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
