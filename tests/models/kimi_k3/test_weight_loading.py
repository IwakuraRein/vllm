# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
    KimiMoE,
)

pytestmark = pytest.mark.cpu_test


class _FakeKimiLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            params[name].data.copy_(value)
            loaded.add(name)
        return loaded

    def finalize_mega_moe_weights(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))
        # MegaMoE finalization replaces its original weight parameters.
        self.tensor_a = None
        self.tensor_c = None


def test_interleaved_composite_weights_finalize_kimi_once_after_loading() -> None:
    language_model = object.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    language_model.model = _FakeKimiLinearModel()

    model = object.__new__(KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = language_model
    model.vision_tower = nn.Module()
    model.vision_tower.tensor_b = nn.Parameter(torch.zeros(1))

    loaded = model.load_weights(
        iter(
            [
                ("language_model.model.tensor_a", torch.tensor([1.0])),
                ("vision_tower.tensor_b", torch.tensor([2.0])),
                ("language_model.model.tensor_c", torch.tensor([3.0])),
            ]
        )
    )

    assert loaded == {
        "language_model.model.tensor_a",
        "vision_tower.tensor_b",
        "language_model.model.tensor_c",
    }
    assert language_model.model.finalized_values == []

    model.process_weights_after_loading()

    assert language_model.model.finalized_values == [(1.0, 3.0)]
    assert model.vision_tower.tensor_b.item() == 2.0


def test_mega_expert_dispatch_preserves_expert_ids_and_skips_other_stages():
    """Direct dispatch leaves EP ownership to the expert loader and skips PP/MTP."""
    model = object.__new__(KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config=None,
        is_moe=True,
        num_experts=896,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
    )
    moe = object.__new__(KimiMoE)
    nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.experts = nn.Module()
    param = nn.Parameter(torch.zeros(1), requires_grad=False)
    moe.experts.w13_weight_scale_2 = param
    calls = []

    def load_expert(param, value, name, *, expert_id, shard_id):
        calls.append((name, expert_id, shard_id))
        if expert_id == 895:
            param.data.copy_(value)

    param.weight_loader = load_expert
    model.layers = nn.ModuleList([moe, PPMissingLayer()])
    loaded = model.load_weights(
        [
            (
                f"layers.{layer}.experts.{expert}.w3.weight_scale_2",
                torch.tensor([value]),
            )
            for layer, expert, value in [
                (0, 0, 1.0),
                (0, 895, 2.0),
                (1, 895, 3.0),
                (2, 895, 4.0),
            ]
        ]
    )

    assert loaded == {"layers.0.experts.w13_weight_scale_2"}
    assert calls == [
        ("layers.0.experts.w13_weight_scale_2", 0, "w3"),
        ("layers.0.experts.w13_weight_scale_2", 895, "w3"),
    ]
    assert param.item() == 2.0
