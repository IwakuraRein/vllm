# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU expert smoke test; no model checkpoint or inference engine."""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import init_test_distributed_environment, multi_gpu_test
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _nvfp4_constant_qdq(value: torch.Tensor) -> torch.Tensor:
    """Each scalar represents a constant 16-element NVFP4 block."""
    scale = (value.abs() / 6).to(torch.float8_e4m3fn).float()
    scaled = torch.where(scale > 0, value.abs() / scale, 0)
    # Even E2M1 encodings precede odd ones to implement ties-to-even.
    levels = value.new_tensor([0, 1, 2, 4, 0.5, 1.5, 3, 6])
    nearest = (scaled[..., None] - levels).abs().argmin(dim=-1)
    return levels[nearest] * scale * value.sign()


def _run_experts(rank: int, port: str, checkpoint_recipe: str) -> None:
    from vllm.models.kimi_k3.nvidia.fi_moe import KimiK3MegaMoEExpertsFI
    from vllm.models.kimi_k3.nvidia.model import _use_sequence_parallel
    from vllm.utils.flashinfer_moe_ep import finalize_fi_moe_ep_runtime

    torch.accelerator.set_device_index(rank)
    device = torch.device(f"cuda:{rank}")
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
        ),
        kernel_config=KernelConfig(moe_backend="flashinfer_moe_ep_mega_cutedsl"),
    )
    config.scheduler_config.max_num_batched_tokens = 16
    nvfp4 = checkpoint_recipe == "nvfp4"
    if nvfp4:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptMixedPrecisionConfig,
        )

        config.quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_method": "modelopt_mixed",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "smoke.experts": {"quant_algo": "NVFP4", "group_size": 16}
                },
            }
        )
    assert _use_sequence_parallel(config)
    with set_current_vllm_config(config):
        init_test_distributed_environment(2, 1, rank, port, local_rank=rank)
        assert get_ep_group().world_size == 2
        experts = None
        try:
            with torch.device(device):
                experts = KimiK3MegaMoEExpertsFI(
                    config,
                    num_experts=4,
                    num_local_experts=2,
                    experts_start_idx=2 * rank,
                    top_k=2,
                    hidden_size=128,
                    intermediate_size=128,
                    activation="situ",
                    activation_beta=1.5,
                    activation_linear_beta=0.75,
                    prefix="smoke.experts",
                )
            # E2M1 nibbles 2/1 encode 1/.5 in both checkpoint recipes.
            experts.w13_weight[:, :128].fill_(0x22)
            experts.w13_weight[:, 128:].fill_(0x11)
            for local_expert in range(2):
                expert_id = 2 * rank + local_expert
                experts.w13_weight_scale[local_expert, :128].fill_(
                    2.0 ** (expert_id - 6) if nvfp4 else 121 + expert_id
                )
                if nvfp4:
                    fc1_alpha = 0.5 + 0.25 * expert_id
                    experts.w13_weight_scale_2[local_expert, 0] = fc1_alpha
                    experts.w13_weight_scale_2[local_expert, 1] = 2 * fc1_alpha
                    experts.w2_weight_scale_2[local_expert] = 1.5 + 0.25 * expert_id
            experts.w13_weight_scale[:, 128:].fill_(1 / 32 if nvfp4 else 122)
            experts.w2_weight.fill_(0x22)
            experts.w2_weight_scale.fill_(1 / 128 if nvfp4 else 120)
            if nvfp4:
                experts.w13_input_scale.fill_(7)
                experts.w2_input_scale.fill_(11)
            experts.finalize_weights()

            # Reuse the workspace after a padded batch and then an empty rank.
            for counts in ((3, 2), (0, 1), (2, 3)):
                n = counts[rank]
                rows = torch.arange(n, device=device)
                x = ((rows + rank + 1) / 8).to(torch.bfloat16)
                x = x[:, None].expand(n, 128).contiguous()
                ids = torch.stack((rows % 2, rows % 2 + 2), dim=1).to(torch.int64)
                weights = torch.tensor([0.25, 0.75], device=device).expand(n, 2)
                weights = weights.contiguous()
                padding = (rows == n - 1) & (rank == 1) & (n > 1)

                total = 128 * _nvfp4_constant_qdq(x[:, :1].float())
                if nvfp4:
                    fc1_alpha = 0.5 + 0.25 * ids.float()
                    gate_weight = torch.exp2(ids.float() - 6) * fc1_alpha
                    up_weight = fc1_alpha / 32
                    down = 1.5 + 0.25 * ids.float()
                else:
                    gate_weight = _nvfp4_constant_qdq(torch.exp2(ids.float() - 6))
                    up_weight = _nvfp4_constant_qdq(total.new_tensor(1 / 64))
                    down = 128 * _nvfp4_constant_qdq(total.new_tensor(1 / 128))
                gate = total * gate_weight
                up = total * up_weight
                activated = (1.5 * torch.tanh(gate / 1.5) * torch.sigmoid(gate)) * (
                    0.75 * torch.tanh(up / 0.75)
                )
                # The kernel applies routing weights before intermediate QDQ.
                intermediate = _nvfp4_constant_qdq(activated * weights)
                partials = (intermediate * down).to(torch.bfloat16).float()
                expected = partials.sum(dim=-1, keepdim=True)
                expected = expected.expand(n, 128).clone()
                expected[padding] = 0

                with set_forward_context(None, config, is_padding=padding):
                    actual = experts(x, weights, ids, activation_clamp=None)
                torch.accelerator.synchronize()
                assert actual.shape == x.shape
                assert actual.dtype == torch.bfloat16
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.01, atol=0.001
                )
        finally:
            if experts is not None and experts._mega_layer is not None:
                experts._mega_layer.destroy()
            finalize_fi_moe_ep_runtime()
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("checkpoint_recipe", ["mxfp4", "nvfp4"])
@multi_gpu_test(num_gpus=2)
def test_cutedsl_mega_moe_situ_with_sequence_parallel_tokens(
    monkeypatch, checkpoint_recipe
):
    """SiTU and cross-rank routing survive padded and empty local SP shards."""
    if not current_platform.is_cuda() or not (
        current_platform.is_device_capability_family(100)
    ):
        pytest.skip("FlashInfer NVFP4 MegaMoE requires SM100/SM103")
    flashinfer = pytest.importorskip("flashinfer.moe_ep")
    if "situ_beta" not in flashinfer.Nvfp4CutedslMegaMoeConfig.__dataclass_fields__:
        pytest.skip("SiTU MegaMoE requires FlashInfer 0.7.1rc1 or newer")
    pytest.importorskip("nvshmem.core", reason="MegaMoE requires nvshmem4py")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    spawn(
        _run_experts,
        args=(str(get_open_port()), checkpoint_recipe),
        nprocs=2,
        join=True,
    )
