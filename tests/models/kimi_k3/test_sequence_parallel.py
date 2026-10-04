# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.models.common.ops import sequence_parallel as sp_ops
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import mtp as kimi_mtp
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


class _IdentityNorm(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = 1e-5

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _Projection(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, hidden_size),
            requires_grad=False,
        )


class _SequenceParallelMTPBlock:
    use_sequence_parallel = True

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states * 2, None, hidden_states * 3


def _mock_sequence_parallel_collectives(monkeypatch):
    monkeypatch.setattr(
        kimi_model,
        "sp_reduce_scatter",
        lambda tensor: tensor.chunk(2, dim=0)[0],
    )
    monkeypatch.setattr(
        kimi_model,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_model,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


@pytest.mark.parametrize(
    ("num_tokens", "is_padding", "tp_rank", "expected"),
    [
        (1, None, 0, [False]),
        (1, None, 1, [True]),
        (5, None, 2, [False, True]),
        (5, None, 3, [True, True]),
        (5, [False, True, False, False, False], 0, [False, True]),
    ],
)
def test_sp_padding_mask_marks_added_rows(
    monkeypatch,
    num_tokens: int,
    is_padding: list[bool] | None,
    tp_rank: int,
    expected: list[bool],
):
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: tp_rank)

    hidden_states = torch.empty(num_tokens, 2)
    padding = torch.tensor(is_padding) if is_padding is not None else None
    actual = sp_ops.sp_padding_mask(padding, hidden_states)

    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize(
    ("data_parallel_size", "expected"),
    [
        (1, False),
        (2, True),
    ],
)
def test_moe_sequence_parallel_requires_data_parallel(
    monkeypatch,
    data_parallel_size: int,
    expected: bool,
):
    monkeypatch.setattr(current_platform, "device_count", lambda: 2)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )

    assert parallel_config.use_sequence_parallel_moe is expected


@pytest.mark.parametrize(
    "tp_size,enable_ep,pp_size,expected",
    [
        (4, True, 1, True),
        (1, True, 1, False),
        (4, False, 1, False),
        (4, True, 2, False),
    ],
)
def test_kimi_cutedsl_mega_moe_enables_sequence_parallel_without_dp(
    tp_size, enable_ep, pp_size, expected
):
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="flashinfer_moe_ep_mega_cutedsl"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            enable_expert_parallel=enable_ep,
            tensor_parallel_size=tp_size,
            data_parallel_size=1,
        ),
    )

    assert kimi_model._use_sequence_parallel(vllm_config) is expected


@pytest.fixture
def kimi_cutedsl_moe(monkeypatch):
    """Construct the real MoE adapter with CPU projections and no runtime."""
    from vllm.models.deepseek_v4.nvidia import fi_moe

    monkeypatch.setattr(fi_moe, "build_fi_mega_config", Mock())
    monkeypatch.setattr(kimi_model, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        kimi_model,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=1),
    )
    monkeypatch.setattr(current_platform, "get_device_capability", lambda: None)
    monkeypatch.setattr(kimi_model, "GateLinear", lambda **kwargs: _Projection())
    monkeypatch.setattr(
        kimi_model,
        "ReplicatedLinear",
        lambda in_features, out_features, **kwargs: nn.Linear(
            in_features, out_features, bias=False, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(kimi_model, "aux_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda: None)

    def build(
        moe_backend="flashinfer_moe_ep_mega_cutedsl",
        situ_beta=1.5,
        situ_linear_beta=0.25,
    ):
        config = KimiLinearConfig(
            hidden_size=128,
            moe_intermediate_size=128,
            num_experts=8,
            num_experts_per_token=2,
            num_shared_experts=None,
            routed_expert_hidden_size=64,
            hidden_act="situ",
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
        )
        vllm_config = SimpleNamespace(
            quant_config=None,
            kernel_config=SimpleNamespace(moe_backend=moe_backend),
            parallel_config=SimpleNamespace(
                enable_expert_parallel=True, enable_eplb=False
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        return kimi_model.KimiMoE(config, vllm_config, use_sequence_parallel=True)

    return build


def test_kimi_moe_selects_cutedsl_experts(kimi_cutedsl_moe):
    from vllm.models.kimi_k3.nvidia.fi_moe import KimiK3MegaMoEExpertsFI

    moe = kimi_cutedsl_moe()

    assert moe.use_mega_moe
    assert isinstance(moe.experts, KimiK3MegaMoEExpertsFI)
    assert moe.experts.num_local_experts == 2
    assert moe.experts.experts_start_idx == 2
    assert moe.experts._activation == "situ"
    assert moe.experts._situ_beta == 1.5
    assert moe.experts._situ_linear_beta == 0.25


def test_kimi_cutedsl_experts_use_default_situ_parameters(kimi_cutedsl_moe):
    moe = kimi_cutedsl_moe(situ_beta=None, situ_linear_beta=0.0)

    assert moe.experts._situ_beta == 1.0
    assert moe.experts._situ_linear_beta is None


@pytest.mark.parametrize("num_tokens,tp_rank", [(0, 0), (1, 0), (1, 3), (5, 2), (5, 3)])
def test_kimi_cutedsl_mega_moe_smoke_preserves_local_tokens(
    monkeypatch, kimi_cutedsl_moe, num_tokens, tp_rank
):
    """First and cached forwards mask padding and keep empty ranks participating."""
    import vllm.forward_context as forward_context
    from vllm.models.deepseek_v4.nvidia import fi_moe

    moe = kimi_cutedsl_moe()
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: tp_rank)
    full_input = torch.arange(num_tokens * 128, dtype=torch.float32).reshape(-1, 128)
    hidden_states = sp_ops.sp_shard(full_input.to(torch.bfloat16))
    is_padding = sp_ops.sp_padding_mask(None, full_input)
    monkeypatch.setattr(fi_moe, "_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=is_padding),
    )
    monkeypatch.setattr(fi_moe, "ensure_fi_moe_ep_runtime", lambda _: None)
    flashinfer_moe_ep = ModuleType("flashinfer.moe_ep")
    monkeypatch.setattr(
        flashinfer_moe_ep, "MoEEpTensors", SimpleNamespace, raising=False
    )
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", flashinfer_moe_ep)

    topk_ids = torch.zeros(hidden_states.shape[0], 2, dtype=torch.int32)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    router = Mock(
        return_value=(hidden_states[:, :64].contiguous(), topk_weights, topk_ids)
    )
    monkeypatch.setattr(moe, "_maybe_overlap_router_and_down_proj", router)
    with torch.no_grad():
        moe.routed_expert_up_proj.weight.copy_(torch.eye(64).repeat(2, 1))

    shared = _RecordingMoE()
    shared_forward = Mock(side_effect=lambda x: x + 3)
    monkeypatch.setattr(shared, "forward", shared_forward)
    moe.shared_experts = shared
    workspace = SimpleNamespace()
    staged = []

    def stage_inputs(tensors, workspace, *, quantize_input):
        assert quantize_input
        staged.append(tensors)

    kernel = SimpleNamespace(
        stage_inputs=stage_inputs,
        compute=lambda workspace, transformed, output: staged[-1].hidden_states + 1,
    )

    def forward(tensors):
        stage_inputs(tensors, workspace, quantize_input=True)
        return kernel.compute(workspace, None, output=None)

    layer_forward = Mock(side_effect=forward)
    moe.experts._mega_layer = SimpleNamespace(
        forward=layer_forward,
        _kernel=kernel,
        _ensure_workspace=lambda: workspace,
        _transformed=None,
        _fleet_params=SimpleNamespace(token_hidden_size=64),
    )
    alphas = (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

    def finalize_weights():
        moe.experts._epilogue_alphas = alphas

    monkeypatch.setattr(moe.experts, "finalize_weights", finalize_weights)
    expected = (hidden_states[:, :64] + 1).repeat(1, 2) + (hidden_states + 3)
    for _ in range(2):
        torch.testing.assert_close(moe(hidden_states), expected)

    assert layer_forward.call_count == 1
    assert len(staged) == router.call_count == shared_forward.call_count == 2
    for tensors in staged:
        assert tensors.fc1_alpha is alphas[0]
        assert tensors.fc2_alpha is alphas[1]
        torch.testing.assert_close(tensors.hidden_states, hidden_states[:, :64])
        torch.testing.assert_close(tensors.topk_weights, topk_weights)
        torch.testing.assert_close(
            tensors.topk_ids, topk_ids.masked_fill(is_padding[:, None], -1)
        )


def test_kimi_decoder_layer_keeps_moe_states_sequence_sharded(monkeypatch):
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = False
    layer.use_sequence_parallel = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.mlp = _RecordingMoE()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states,
        layer,
    )

    _mock_sequence_parallel_collectives(monkeypatch)

    positions = torch.arange(3)
    full_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states = kimi_model.sp_shard(full_hidden_states)
    hidden_states, prefix_sum, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )

    assert prefix_sum is None
    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.mlp.num_tokens == 2

    hidden_states, prefix_sum, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    assert prefix_sum is None
    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.mlp.num_tokens == 2


def test_kimi_attn_residual_states_stay_sequence_sharded(monkeypatch):
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.use_sequence_parallel = True
    layer.prev_valid_blocks = 0
    layer.block_write_idx = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attention_res_norm = _IdentityNorm()
    layer.mlp_res_norm = _IdentityNorm()
    layer.self_attention_res_proj = _Projection()
    layer.mlp_res_proj = _Projection()
    layer.mlp = _RecordingMoE()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states,
        layer,
    )

    _mock_sequence_parallel_collectives(monkeypatch)
    monkeypatch.setattr(
        kimi_model,
        "attn_res",
        lambda prefix_sum, hidden_states, *args, **kwargs: (
            prefix_sum if hidden_states is None else prefix_sum + hidden_states
        ),
    )

    prefix_sum = kimi_model.sp_shard(torch.arange(6, dtype=torch.float32).view(3, 2))
    block_residual = torch.zeros(2, 1, 2)
    hidden_states, prefix_sum, block_residual = layer(
        positions=torch.arange(3),
        hidden_states=None,
        prefix_sum=prefix_sum,
        residual=block_residual,
    )

    assert hidden_states.shape == prefix_sum.shape == (2, 2)
    assert block_residual.shape == (2, 1, 2)
    assert layer.mlp.num_tokens == 2


def test_kimi_mtp_restores_sequence_parallel_output(monkeypatch):
    layer = object.__new__(kimi_mtp.KimiK3MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = nn.Identity()
    object.__setattr__(layer, "mtp_block", _SequenceParallelMTPBlock())

    final_norm = Mock(side_effect=lambda hidden_states: hidden_states + 1)
    object.__setattr__(
        layer,
        "shared_head",
        SimpleNamespace(norm=final_norm),
    )

    monkeypatch.setattr(
        kimi_mtp,
        "fused_mtp_input",
        lambda positions, inputs_embeds, *args: inputs_embeds,
    )
    monkeypatch.setattr(
        kimi_mtp,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_mtp,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    logits_hidden_states, hidden_states = layer(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    sharded_states = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected_hidden_states = torch.cat(
        [sharded_states * 5, sharded_states * 5],
        dim=0,
    )[:3]
    torch.testing.assert_close(hidden_states, expected_hidden_states)
    torch.testing.assert_close(logits_hidden_states, expected_hidden_states + 1)
    final_norm.assert_called_once()
    torch.testing.assert_close(final_norm.call_args.args[0], expected_hidden_states)


@pytest.mark.parametrize(
    ("enabled", "use_sequence_parallel", "eligible", "tp_size", "expected"),
    [
        (True, True, True, 8, True),
        (False, True, True, 8, False),  # opt-in only
        (True, False, True, 8, False),  # replication only exists under SP
        (True, True, False, 8, False),  # FusedMoE path owns the reduction
        (True, True, True, 1, False),  # nothing to shard
        (True, True, True, 5, False),  # 6144 % 5 -- would fail divide()
    ],
)
def test_shard_sequence_parallel_mlp_gating(
    monkeypatch,
    enabled: bool,
    use_sequence_parallel: bool,
    eligible: bool,
    tp_size: int,
    expected: bool,
):
    monkeypatch.setattr(kimi_model.envs, "VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT", enabled)
    monkeypatch.setattr(
        kimi_model, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    assert (
        kimi_model.shard_sequence_parallel_mlp(
            hidden_size=7168,
            intermediate_size=6144,
            use_sequence_parallel=use_sequence_parallel,
            eligible=eligible,
        )
        is expected
    )


def test_sharded_sequence_parallel_mlp_matches_replicated(default_vllm_config):
    """Sharded SP MLP must reproduce the replicated result for every token.

    Each rank owns a *disjoint* token shard, so a weight shard alone cannot
    finish a rank's own tokens: the ranks must gather the full token set,
    compute partial sums over their intermediate shard, and reduce-scatter.
    Splicing per-rank feature slices together instead silently mixes different
    tokens and produces plausible-looking garbage.
    """
    tp_size, hidden, intermediate, tokens_per_rank = 4, 16, 12, 3
    torch.manual_seed(0)
    num_tokens = tp_size * tokens_per_rank
    x = torch.randn(num_tokens, hidden)
    gate_weight = torch.randn(intermediate, hidden)
    up_weight = torch.randn(intermediate, hidden)
    down_weight = torch.randn(hidden, intermediate)
    act_fn = SiluAndMul()

    replicated = act_fn(x @ torch.cat([gate_weight, up_weight]).T) @ down_weight.T

    shard = intermediate // tp_size
    # Every rank all-gathers the full token set, then computes its partial.
    partials = [
        act_fn(
            x
            @ torch.cat(
                [
                    gate_weight[r * shard : (r + 1) * shard],
                    up_weight[r * shard : (r + 1) * shard],
                ]
            ).T
        )
        @ down_weight[:, r * shard : (r + 1) * shard].T
        for r in range(tp_size)
    ]
    reduced = torch.stack(partials).sum(0)
    # Reduce-scatter: rank r keeps only its own token shard.
    for r in range(tp_size):
        mine = reduced[r * tokens_per_rank : (r + 1) * tokens_per_rank]
        expected = replicated[r * tokens_per_rank : (r + 1) * tokens_per_rank]
        torch.testing.assert_close(mine, expected, atol=1e-5, rtol=1e-5)


def test_sp_all_gather_uses_custom_kernel(monkeypatch):
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    expected = torch.cat([hidden_states, hidden_states])
    custom_all_gather = Mock(return_value=expected)
    device_communicator = SimpleNamespace(
        custom_all_gather=custom_all_gather,
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", fallback)

    output = sp_ops.sp_all_gather(hidden_states)

    torch.testing.assert_close(output, expected)
    custom_all_gather.assert_called_once_with(hidden_states)
    fallback.assert_not_called()


def test_sp_reduce_scatter_uses_custom_kernel_after_padding(monkeypatch):
    hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    expected = torch.arange(4, dtype=torch.float32).view(2, 2)
    custom_reduce_scatter = Mock(return_value=expected)
    device_communicator = SimpleNamespace(
        custom_reduce_scatter=custom_reduce_scatter,
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_reduce_scatter", fallback)

    output = sp_ops.sp_reduce_scatter(hidden_states)

    torch.testing.assert_close(output, expected)
    padded = custom_reduce_scatter.call_args.args[0]
    assert padded.shape == (4, 2)
    torch.testing.assert_close(padded[:3], hidden_states)
    torch.testing.assert_close(padded[3], torch.zeros(2))
    fallback.assert_not_called()


@pytest.mark.parametrize("shape", [(3,), (3, 2, 2)])
def test_sp_shard_pads_only_the_token_axis(monkeypatch, shape):
    hidden_states = torch.arange(math.prod(shape), dtype=torch.float32).view(shape)
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: 1)

    output = sp_ops.sp_shard(hidden_states)

    padding = hidden_states.new_zeros((1, *shape[1:]))
    expected = torch.cat([hidden_states, padding])[2:]
    torch.testing.assert_close(output, expected)


def test_sp_collectives_fall_back_without_custom_kernel(monkeypatch):
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=None),
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    all_gather = Mock(return_value=hidden_states)
    reduce_scatter = Mock(return_value=hidden_states)
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(
        sp_ops,
        "tensor_model_parallel_reduce_scatter",
        reduce_scatter,
    )

    torch.testing.assert_close(sp_ops.sp_all_gather(hidden_states), hidden_states)
    torch.testing.assert_close(sp_ops.sp_reduce_scatter(hidden_states), hidden_states)
    all_gather.assert_called_once_with(hidden_states, 0)
    reduce_scatter.assert_called_once_with(hidden_states, 0)
