# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA allocation and warmup must agree on physical addressing."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _canonicalize_sparse_mla_kv_cache_dtype,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashinfer_mla import FlashInferMLABackend
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.mla.flashmla_windowed import (
    FlashMLAWindowedBackend,
    FlashMLAWindowedFP8Backend,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.sparse_utils import (
    _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL,
    flat_kv_row_view,
)
from vllm.v1.attention.backends.mla.tokenspeed_mla import TokenspeedMLABackend
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache, select_common_block_size

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="CUDA only"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "layout", [KVCacheLayout.BLHNC, KVCacheLayout.BLNHC, KVCacheLayout.LBNHC]
)
@pytest.mark.parametrize(
    "backend,cache_dtype,sm100,stride_alignment",
    [
        (FlashInferMLASparseTRTLLMBackend, "auto", True, 1152),
        (FlashInferMLASparseTRTLLMBackend, "fp8", True, 576),
        (FlashMLASparseBackend, "auto", True, 1152),
        (FlashMLASparseBackend, "fp8_ds_mla", False, None),
        (FlashMLASparseBackend, "fp8_ds_mla", True, 656),
        (FlashMLASparseBackend, "nvfp4_ds_mla", True, 352),
    ],
)
def test_allocation_and_warmup_follow_addressing_mode(
    monkeypatch, layout, backend, cache_dtype, sm100, stride_alignment, device
):
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: sm100
    )
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=SimpleNamespace(hisparse_config=None),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )
    config.cache_config.kv_cache_layout = layout.name
    layer = SimpleNamespace(
        _vllm_config=config,
        attn_backend=backend,
        kv_cache_dtype=cache_dtype,
        head_size=576,
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
    )
    layer._uses_flat_kv_cache = MethodType(MLAAttention._uses_flat_kv_cache, layer)
    spec = MLAAttention.get_kv_cache_spec(layer, config)
    indexer = SimpleNamespace(
        cache_config=config.cache_config, head_dim=132, dtype=torch.uint8
    )
    index_spec = DeepseekV32IndexerCache.get_kv_cache_spec(indexer, config)
    assert spec.block_stride_alignment == stride_alignment
    assert index_spec.block_stride_alignment is None
    assert layout in get_supported_kv_cache_layouts(
        [backend, DeepseekV32IndexerBackend]
    )
    specs = {"mla": spec, "indexer": index_spec}
    group = KVCacheGroupSpec(
        list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs)
    )
    cache = get_kv_cache_config_from_groups(config, [group], 2**20)
    views = allocate_kv_cache(cache, torch.device(device), layout)
    register = MagicMock()
    monkeypatch.setattr(
        _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL, "register_warmup", register
    )
    MLAAttention.bind_kv_cache(layer, views["mla"])
    DeepseekV32IndexerCache.bind_kv_cache(indexer, views["indexer"])
    assert indexer.kv_cache.stride(0) == views["indexer"].stride(0)
    if stride_alignment is not None:
        assert (
            views["mla"].stride(0) * views["mla"].element_size() % stride_alignment == 0
        )
    if not layer._uses_flat_kv_cache():
        register.assert_not_called()
    else:
        rows, stride = flat_kv_row_view(layer.kv_cache, 64)
        register.assert_called_once_with(config, block_stride_rows=stride)
        layer.kv_cache[1, 0].fill_(3)
        torch.testing.assert_close(rows[stride], layer.kv_cache[1, 0])
    if stride_alignment is None:
        assert cache.kv_cache_tensors[0].size == cache.num_blocks * sum(
            item.page_size_bytes for item in specs.values()
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_ds_mla"])
@pytest.mark.parametrize("block_size", [32, 64])
def test_windowed_fp8_packs_mixed_pages_without_changing_block_sizes(
    monkeypatch, cache_dtype, block_size
):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 100
    )
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=block_size, cache_dtype="fp8"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_transfer_config=None,
    )
    layer = SimpleNamespace(
        attn_backend=FlashMLAWindowedFP8Backend,
        kv_cache_dtype=_canonicalize_sparse_mla_kv_cache_dtype(
            FlashMLAWindowedFP8Backend, cache_dtype
        ),
        head_size=576,
        sliding_window=4096,
        indexer=None,
        non_causal_multi_token_decode=True,
    )
    layer._uses_flat_kv_cache = MethodType(MLAAttention._uses_flat_kv_cache, layer)
    window = MLAAttention.get_kv_cache_spec(layer, config)
    assert isinstance(window, SlidingWindowMLASpec)
    assert window.block_stride_alignment == 656
    assert window.page_size_bytes == block_size * 656
    assert window.cache_dtype_str == "fp8_ds_mla"
    specs = {
        "full": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
        ),
        "window.0": window,
        "window.1": window,
        "linear": MambaSpec(
            block_size=block_size,
            shapes=((10, 4608), (12, 128, 128)),
            dtypes=(torch.bfloat16, torch.bfloat16),
            page_size_padded=516096,
            block_stride_alignment=32,
        ),
    }
    full_backends = [TokenspeedMLABackend, FlashInferMLABackend]
    supported = get_supported_kv_cache_layouts(
        [*full_backends, FlashMLAWindowedFP8Backend]
    )
    layout = resolve_kv_cache_layout(
        config, [[item.name for item in supported]], specs.values()
    )
    assert layout == KVCacheLayout.BLNHC
    groups = get_kv_cache_groups(config, specs)
    grouped = {
        name: spec
        for group in groups
        for name, spec in group.kv_cache_spec.kv_cache_specs.items()
    }
    assert grouped == specs
    cache_config = get_kv_cache_config_from_groups(config, groups, 4 * 2**20)
    kernel_block_sizes = [
        group.kv_cache_spec.block_size
        if "linear" in group.layer_names
        else select_common_block_size(
            group.kv_cache_spec.block_size,
            full_backends
            if "full" in group.layer_names
            else [FlashMLAWindowedFP8Backend],
        )
        for group in groups
    ]
    views = allocate_kv_cache(
        cache_config, torch.device("cpu"), layout, kernel_block_sizes
    )
    full, window0, window1 = (views[name] for name in specs if name != "linear")
    assert full.shape[1:] == (1, block_size, 576)
    assert window0.shape[1:] == (1, block_size, 656)
    assert window0.stride(0) % 656 == 0
    assert window0.stride(0) % 32 == 0
    widest = max(group.kv_cache_spec.page_size_bytes for group in groups)
    assert widest <= window0.stride(0) < widest + 1312
    assert window0.stride(-2) == 656
    full[2].fill_(3)
    window0[1].fill_(7)
    window1[3].fill_(11)
    assert torch.all(full[2] == 3)
    assert torch.all(window0[1] == 7)
    assert torch.all(window1[3] == 11)
    state = SimpleNamespace(
        get_state_shape=lambda: specs["linear"].shapes,
        get_state_dtype=lambda: specs["linear"].dtypes,
    )
    MambaBase.bind_kv_cache(state, views["linear"])
    for tensor in state.kv_cache:
        assert tensor.stride(0) % 16 == 0
        assert tensor.data_ptr() % 16 == 0
    assert config.cache_config.cache_dtype == "fp8"


@pytest.mark.cpu_test
def test_bf16_window_does_not_require_packed_layout():
    assert FlashMLAWindowedBackend.supported_kv_cache_layouts() is None
    assert (
        _canonicalize_sparse_mla_kv_cache_dtype(FlashMLAWindowedBackend, "auto")
        == "auto"
    )
