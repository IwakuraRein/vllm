# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the flashinfer moe_ep backend plumbing.

Everything here runs without a GPU or a flashinfer install: the flashinfer
modules the helpers import lazily are replaced with capture fakes.
"""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    MEGA_MOE_BACKENDS,
    validate_flashinfer_moe_ep_model,
)
from vllm.utils.flashinfer_moe_ep import (
    _E2M1_LUT,
    FI_MOE_EP_BACKEND_SPECS,
    _dequant_fp4_ue8m0_gran32,
    build_fi_mega_config,
    fi_moe_ep_backend_spec,
    make_fi_moe_ep_bootstrap,
    megakernel_runtime_requirements,
)


@dataclass
class _FakeBootstrapConfig:
    world_size: int
    rank: int
    process_group: Any = None
    auto_bootstrap: bool = True
    device: int | None = field(default=None, kw_only=True)


@dataclass
class _FakeDeepGemmMegaMoeConfig:
    intermediate_size: int
    top_k: int
    activation_clamp: float | None
    fast_math: bool


@dataclass
class _FakeNvfp4CutedslMegaMoeConfig:
    intermediate_size: int
    top_k: int
    activation_clamp: float | None
    fast_math: bool
    activation: str = "swiglu"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None


@dataclass
class _FakeMegaConfig:
    megakernel: Any
    preprocess_weights: bool
    quantize_input: bool


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a minimal fake flashinfer.moe_ep for the lazy imports."""
    moe_ep = ModuleType("flashinfer.moe_ep")
    core = ModuleType("flashinfer.moe_ep.core")
    runtime = ModuleType("flashinfer.moe_ep.core.runtime")
    flashinfer = ModuleType("flashinfer")
    fake_attrs: dict[ModuleType, dict[str, Any]] = {
        moe_ep: {
            "BootstrapConfig": _FakeBootstrapConfig,
            "DeepGemmMegaMoeConfig": _FakeDeepGemmMegaMoeConfig,
            "Nvfp4CutedslMegaMoeConfig": _FakeNvfp4CutedslMegaMoeConfig,
            "MegaConfig": _FakeMegaConfig,
            "core": core,
        },
        runtime: {"TORCH_DIST": "torch_dist", "NVSHMEM": "nvshmem"},
        flashinfer: {"moe_ep": moe_ep},
        core: {"runtime": runtime},
    }
    for mod, attrs in fake_attrs.items():
        for attr, value in attrs.items():
            setattr(mod, attr, value)

    for name, mod in {
        "flashinfer": flashinfer,
        "flashinfer.moe_ep": moe_ep,
        "flashinfer.moe_ep.core": core,
        "flashinfer.moe_ep.core.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return moe_ep


def test_fi_backend_strings_are_registered_mega_moe_backends():
    assert set(FI_MOE_EP_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS
    assert FLASHINFER_MOE_EP_BACKENDS < MEGA_MOE_BACKENDS


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
def test_fi_moe_ep_backend_rejected_for_unsupported_model(moe_backend):
    """An FI moe_ep backend with an unsupported model must fail at config time
    instead of silently falling through to the generic FusedMoE path."""
    with pytest.raises(ValueError, match="only supported"):
        validate_flashinfer_moe_ep_model(moe_backend, ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(FLASHINFER_MOE_EP_BACKENDS))
def test_fi_moe_ep_backend_accepted_for_dsv4(moe_backend):
    validate_flashinfer_moe_ep_model(moe_backend, ["DeepseekV4ForCausalLM"])


@pytest.mark.parametrize(
    "architecture", ["KimiK3ForConditionalGeneration", "KimiK3MTPModel"]
)
def test_cutedsl_mega_moe_accepts_kimi_k3(architecture):
    moe_backend = "flashinfer_moe_ep_mega_cutedsl"
    validate_flashinfer_moe_ep_model(moe_backend, [architecture])
    assert fi_moe_ep_backend_spec(moe_backend).megakernel == "nvfp4_cutedsl"


def test_flashinfer_deep_gemm_mega_moe_rejects_kimi_k3():
    with pytest.raises(ValueError, match="only supported"):
        validate_flashinfer_moe_ep_model(
            "flashinfer_moe_ep_mega_deep_gemm", ["KimiK3ForConditionalGeneration"]
        )


@pytest.mark.parametrize(
    "architectures",
    [["KimiK3ForConditionalGeneration"], ["MixtralForCausalLM"]],
)
def test_native_deep_gemm_mega_moe_not_arch_gated(architectures):
    """VLLM's own deep_gemm mega path is not DSv4-only (Kimi K3 uses it);
    models validate their own constraints at construction time."""
    validate_flashinfer_moe_ep_model("deep_gemm_mega_moe", architectures)


def test_non_fi_backend_ignores_architectures():
    validate_flashinfer_moe_ep_model("auto", ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(MEGA_MOE_BACKENDS))
def test_all_mega_backends_get_sequence_parallel_moe(moe_backend):
    """Every mega backend must qualify for sequence-parallel MoE at
    TP>1/EP: the predicate once matched only the native backend string,
    which silently ran the fi backends full-batch with an all-reduce on
    every rank — 0.42-0.65x native e2e at TP8."""
    from vllm.models.deepseek_v4.nvidia.model import _use_sequence_parallel

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )
    assert _use_sequence_parallel(vllm_config)


def test_fi_moe_ep_backend_spec_kernel_and_nvshmem_contract():
    dg = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    assert dg.megakernel == "deep_gemm_mega"
    assert not dg.needs_nvshmem

    cd = fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    assert cd.megakernel == "nvfp4_cutedsl"
    assert cd.needs_nvshmem

    with pytest.raises(ValueError, match="not a flashinfer moe_ep backend"):
        fi_moe_ep_backend_spec("deep_gemm_mega_moe")


def test_megakernel_runtime_requirements(fake_flashinfer):
    dg = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_deep_gemm")
    )
    assert dg == frozenset({"torch_dist"})

    cd = megakernel_runtime_requirements(
        fi_moe_ep_backend_spec("flashinfer_moe_ep_mega_cutedsl")
    )
    assert cd == frozenset({"torch_dist", "nvshmem"})


def test_bootstrap_pins_the_device_vllm_bound(fake_flashinfer, monkeypatch):
    """The runtime must not rederive the device from LOCAL_RANK/rank: under a
    remapped CUDA_VISIBLE_DEVICES that ordinal points at the wrong GPU
    (CUDA_ERROR_ILLEGAL_ADDRESS in the weight transforms). vLLM passes the
    device it already bound via BootstrapConfig.device."""
    import vllm.utils.flashinfer_moe_ep as mod

    pg = object()
    monkeypatch.setattr(
        mod,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=2, device_group=pg),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    bootstrap = make_fi_moe_ep_bootstrap()

    assert bootstrap.world_size == 4
    assert bootstrap.rank == 2
    assert bootstrap.process_group is pg
    assert bootstrap.auto_bootstrap is False
    assert bootstrap.device == 3


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("flashinfer_moe_ep_mega_cutedsl", ["empty_cache", "bootstrap"]),
        ("flashinfer_moe_ep_mega_deep_gemm", ["bootstrap"]),
    ],
)
def test_runtime_releases_cached_memory_before_nvshmem_only_once(
    fake_flashinfer, monkeypatch, backend, expected
):
    from vllm.utils import flashinfer_moe_ep

    calls = []
    bootstrap = object()
    handle = object()
    monkeypatch.setattr(flashinfer_moe_ep, "_FI_RUNTIME_HANDLE", None)
    monkeypatch.setattr(
        flashinfer_moe_ep, "make_fi_moe_ep_bootstrap", lambda: bootstrap
    )
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: calls.append("empty_cache")
    )

    def initialize(actual_bootstrap, requirements):
        assert actual_bootstrap is bootstrap
        assert ("nvshmem" in requirements) == ("empty_cache" in calls)
        calls.append("bootstrap")
        return handle

    monkeypatch.setattr(
        fake_flashinfer, "bootstrap_moe_ep_runtime", initialize, raising=False
    )
    config = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend=backend))
    flashinfer_moe_ep.ensure_fi_moe_ep_runtime(config)
    flashinfer_moe_ep.ensure_fi_moe_ep_runtime(config)

    assert calls == expected
    assert flashinfer_moe_ep._FI_RUNTIME_HANDLE is handle


def test_build_fi_mega_config_selects_kernel_config(fake_flashinfer):
    dg = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=7.0,
        megakernel="deep_gemm_mega",
    )
    assert isinstance(dg.megakernel, _FakeDeepGemmMegaMoeConfig)
    assert dg.megakernel.intermediate_size == 2048
    assert dg.megakernel.top_k == 8
    assert dg.megakernel.activation_clamp == 7.0
    assert dg.preprocess_weights and dg.quantize_input

    cd = build_fi_mega_config(
        intermediate_size=2048,
        top_k=8,
        activation_clamp=None,
        megakernel="nvfp4_cutedsl",
    )
    assert isinstance(cd.megakernel, _FakeNvfp4CutedslMegaMoeConfig)

    with pytest.raises(ValueError, match="Unsupported fi_moe_ep megakernel"):
        build_fi_mega_config(
            intermediate_size=2048,
            top_k=8,
            activation_clamp=None,
            megakernel="deep_gemm",
        )


@pytest.mark.parametrize("situ_linear_beta", [None, 0.25])
def test_cutedsl_mega_config_preserves_situ_parameters(
    fake_flashinfer, situ_linear_beta
):
    config = build_fi_mega_config(
        intermediate_size=128,
        top_k=2,
        activation_clamp=None,
        megakernel="nvfp4_cutedsl",
        activation="situ",
        situ_beta=1.5,
        situ_linear_beta=situ_linear_beta,
    )

    assert config.megakernel.activation == "situ"
    assert config.megakernel.situ_beta == 1.5
    assert config.megakernel.situ_linear_beta == situ_linear_beta


def test_cutedsl_situ_requires_flashinfer_activation_support(
    fake_flashinfer, monkeypatch
):
    """Old FlashInfer must fail before silently selecting a SwiGLU kernel."""
    monkeypatch.setattr(
        fake_flashinfer, "Nvfp4CutedslMegaMoeConfig", _FakeDeepGemmMegaMoeConfig
    )
    with pytest.raises(RuntimeError, match=r"0\.7\.1rc1.*PR #5455"):
        build_fi_mega_config(
            intermediate_size=128,
            top_k=2,
            activation_clamp=None,
            megakernel="nvfp4_cutedsl",
            activation="situ",
            situ_beta=1.5,
        )


def test_ckpt_uses_nvfp4_experts_reads_moe_quant_algo():
    from vllm.models.deepseek_v4.nvidia.fi_moe import ckpt_uses_nvfp4_experts

    nvfp4 = SimpleNamespace(quant_config=SimpleNamespace(moe_quant_algo="NVFP4"))
    assert ckpt_uses_nvfp4_experts(nvfp4)

    mxfp4 = SimpleNamespace(quant_config=SimpleNamespace(moe_quant_algo=None))
    assert not ckpt_uses_nvfp4_experts(mxfp4)

    no_algo = SimpleNamespace(quant_config=SimpleNamespace())
    assert not ckpt_uses_nvfp4_experts(no_algo)


@pytest.mark.parametrize("checkpoint_format", ["modelopt_mixed", "compressed-tensors"])
def test_kimi_nvfp4_checkpoint_scales_and_dummy_initialization(
    fake_flashinfer, checkpoint_format
):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.model_executor.model_loader.weight_utils import (
        initialize_single_dummy_weight,
    )
    from vllm.models.kimi_k3.nvidia.fi_moe import KimiK3MegaMoEExpertsFI

    prefix = "layer.experts"
    if checkpoint_format == "modelopt_mixed":
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_method": "modelopt_mixed",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {prefix: {"quant_algo": "NVFP4", "group_size": 16}},
            }
        )
    else:
        quant_config = CompressedTensorsConfig.from_config(
            {
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "experts": {
                        "targets": [prefix],
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                            "strategy": "tensor_group",
                            "group_size": 16,
                        },
                    }
                },
            }
        )
    config = SimpleNamespace(
        quant_config=quant_config,
        kernel_config=SimpleNamespace(moe_backend="flashinfer_moe_ep_mega_cutedsl"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = KimiK3MegaMoEExpertsFI(
        config,
        num_experts=2,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        prefix=prefix,
        activation="situ",
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )

    assert experts._nvfp4_prequant
    assert experts.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert experts.w13_weight_scale.shape == (1, 256, 8)
    for name in ("w13_weight_scale", "w13_weight_scale_2", "w2_weight_scale_2"):
        param = getattr(experts, name)
        initialize_single_dummy_weight(param)
        torch.testing.assert_close(param.float(), torch.ones_like(param.float()))

    experts.weight_loader(
        experts.w13_weight_scale_2,
        torch.tensor(0.25),
        "layer.experts.w13_weight_scale_2",
        "w3",
        0,
    )
    expected = 0.25 if checkpoint_format == "modelopt_mixed" else 4.0
    assert experts.w13_weight_scale_2[0, 1].item() == expected


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("weight", "weight"),
        ("weight_packed", "weight"),
        ("weight_scale", "weight_scale"),
        ("weight_scale_2", "weight_scale_2"),
        ("weight_global_scale", "weight_scale_2"),
        ("input_scale", "input_scale"),
        ("input_global_scale", "input_scale"),
    ],
)
def test_kimi_mega_weight_mapping_preserves_scale_suffixes(suffix, expected):
    from vllm.models.kimi_k3.nvidia.model import (
        make_kimi_k3_mega_moe_expert_params_mapping,
        map_kimi_k3_mega_moe_expert_weight,
    )

    checkpoint_name = f"layer.experts.0.w1.{suffix}"
    assert map_kimi_k3_mega_moe_expert_weight(checkpoint_name, 1) == (
        f"layer.experts.w13_{expected}",
        0,
        "w1",
    )
    for (
        param_name,
        weight_name,
        expert_id,
        shard_id,
    ) in make_kimi_k3_mega_moe_expert_params_mapping(1):
        if weight_name in checkpoint_name:
            assert checkpoint_name.replace(weight_name, param_name) == (
                f"layer.experts.w13_{expected}"
            )
            assert expert_id == 0 and shard_id == "w1"
            break
    else:
        pytest.fail(f"No mapping for {checkpoint_name}")


def test_dequant_fp4_ue8m0_gran32_decodes_lut_and_scales():
    """One 32-element scale group per row: low nibble is the even element,
    high nibble the odd one, ue8m0 scale applies to the whole group."""
    packed = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    sf = torch.tensor([[127], [128]], dtype=torch.uint8)  # 2**0, 2**1

    out = _dequant_fp4_ue8m0_gran32(packed, sf)

    assert out.shape == (2, 32)
    assert out.dtype == torch.bfloat16
    expected = torch.empty(2, 32)
    for row in range(2):
        for col in range(16):
            byte = int(packed[row, col])
            expected[row, 2 * col] = _E2M1_LUT[byte & 0x0F]
            expected[row, 2 * col + 1] = _E2M1_LUT[byte >> 4]
        expected[row] *= 2.0**row
    assert torch.equal(out, expected.to(torch.bfloat16))
