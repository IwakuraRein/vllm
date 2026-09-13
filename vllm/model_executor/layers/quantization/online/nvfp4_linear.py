# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.quantization.online.fp8 import (
    OnlineLinearBase,
    _is_tp_sharded,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_tp_weight_quant,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

NVFP4_BLOCK_SIZE = 16


class Nvfp4OnlineLinearMethod(OnlineLinearBase):
    """Quantize BF16/FP16 weights to NVFP4 on Blackwell, keeping A16 inputs."""

    supported_activation_quant = {None}

    def __init__(self):
        if not current_platform.is_cuda() or not (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
        ):
            raise ValueError(
                "nvfp4_weight_only online quantization requires a Blackwell "
                "(SM100/SM120 family) GPU for load-time weight quantization."
            )
        super().__init__()
        self.kernel = init_nvfp4_linear_kernel(
            use_a16=True, input_dtype=self.input_dtype
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"NVFP4 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{NVFP4_BLOCK_SIZE}."
            )

        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Fused projections share one global scale, including across TP shards.
        amax = weight_amax(layer.weight).to(torch.float32)
        amax = amax_for_tp_weight_quant(amax, _is_tp_sharded(layer)).clamp_min(1e-8)
        global_scale = (FLOAT4_E2M1_MAX * torch.finfo(torch.float8_e4m3fn).max) / amax
        weight, weight_scale = scaled_fp4_quant(
            layer.weight.contiguous(),
            global_scale,
            is_sf_swizzled_layout=False,
        )

        layer.input_scale = None
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", weight_scale)
        replace_parameter(layer, "weight_global_scale", global_scale.reciprocal())

        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
