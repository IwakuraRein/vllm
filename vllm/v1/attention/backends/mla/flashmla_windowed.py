# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionLayer, KVCacheLayout
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_sparse_supported,
)


@triton.jit
def _window_indices_kernel(
    QueryStart,
    SeqLens,
    BlockTable,
    Indices,
    Lengths,
    cp_rank,
    table_stride: tl.constexpr,
    page_size: tl.constexpr,
    window: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
    CP_WORLD: tl.constexpr,
):
    request = tl.program_id(0)
    query = tl.program_id(1)
    start = tl.load(QueryStart + request)
    query_len = tl.load(QueryStart + request + 1) - start
    # SeqLens are global: the window is over global positions.
    seq_len = tl.load(SeqLens + request)
    position = seq_len - query_len + query
    left = tl.maximum(0, position - window + 1)
    right = tl.minimum(seq_len, position + window)
    if CP_WORLD > 1:
        # Under DCP (interleave 1) this rank holds global positions cp_rank,
        # cp_rank + CP_WORLD, ... at local positions 0, 1, ..., which the block
        # table maps as it maps a rank's own tokens.
        left = (tl.maximum(left - cp_rank, 0) + CP_WORLD - 1) // CP_WORLD
        right = (tl.maximum(right - cp_rank, 0) + CP_WORLD - 1) // CP_WORLD
    length = tl.maximum(0, right - left)
    offsets = tl.arange(0, BLOCK)
    logical = left + offsets
    valid = (query < query_len) & (offsets < length) & (offsets < width)
    page = tl.load(
        BlockTable + request.to(tl.int64) * table_stride + logical // page_size,
        mask=valid,
        other=0,
    )
    physical = page * page_size + logical % page_size
    row = (start + query).to(tl.int64)
    tl.store(
        Indices + row * width + offsets,
        tl.where(valid, physical, -1),
        mask=(query < query_len) & (offsets < width),
    )
    tl.store(Lengths + row, length, mask=query < query_len)


class FlashMLAWindowedMetadataBuilder(TritonMLAMetadataBuilder):
    # The window kernel reads every query token's window from the rank's own slots.
    supports_dcp_with_varlen: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        parallel_config = vllm_config.parallel_config
        interleave_size = parallel_config.cp_kv_cache_interleave_size
        if parallel_config.decode_context_parallel_size > 1 and interleave_size != 1:
            raise ValueError(
                "Windowed FlashMLA with DCP requires "
                f"cp_kv_cache_interleave_size=1; got {interleave_size}."
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)


class FlashMLAWindowedBackend(TritonMLABackend):
    """Model-selected non-causal windowed MLA with a Triton fallback."""

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_WINDOWED"

    @staticmethod
    def get_impl_cls() -> type["FlashMLAWindowedImpl"]:
        return FlashMLAWindowedImpl

    @staticmethod
    def get_builder_cls() -> type["FlashMLAWindowedMetadataBuilder"]:
        return FlashMLAWindowedMetadataBuilder

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False


class FlashMLAWindowedImpl(TritonMLAImpl):
    supports_windowed_dcp: bool = True

    def _forward_windowed_mqa(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dcp = self.dcp_world_size > 1
        packed_fp8 = self.kv_cache_dtype == "fp8_ds_mla"
        cache_supported = (
            kv_cache.dtype == torch.uint8
            and kv_cache.shape[-1] == 656
            and kv_cache.stride(-1) == 1
            and kv_cache.stride(-2) == 656
            and (
                kv_cache.stride(0) % 656 == 0
                or not current_platform.is_device_capability_family(100)
            )
            if packed_fp8
            else kv_cache.dtype == torch.bfloat16
            and kv_cache.shape[-1] == 576
            and kv_cache.is_contiguous()
        )
        if not (
            is_flashmla_sparse_supported()[0]
            and q.dtype == torch.bfloat16
            and q.shape[-1] == 576
            and self.kv_lora_rank == 512
            and cache_supported
        ):
            if packed_fp8:
                raise NotImplementedError(
                    "Windowed FP8 FlashMLA requires BF16 queries, a 512 + 64 "
                    "latent and a row-aligned fp8_ds_mla cache."
                )
            if dcp:
                raise NotImplementedError(
                    "Windowed MLA with DCP needs the FlashMLA path: a BF16 KV "
                    "cache with a 512 + 64 latent."
                )
            return super()._forward_windowed_mqa(q, kv_cache, metadata, layer)

        assert not metadata.causal
        assert self.sliding_window is not None
        assert metadata.decode is not None
        # Future draft tokens extend the first query's visible range beyond W.
        max_window = min(
            2 * self.sliding_window - 1,
            self.sliding_window + metadata.max_query_len - 1,
        )
        if dcp:
            # A rank holds every dcp_world_size-th position of the window.
            assert metadata.decode.dcp_tot_seq_lens is not None
            max_keys = triton.cdiv(max_window, self.dcp_world_size)
            seq_lens = metadata.decode.dcp_tot_seq_lens
        else:
            max_keys = min(metadata.max_seq_len, max_window)
            seq_lens = metadata.decode.seq_lens
        width = triton.cdiv(max(1, max_keys), 128) * 128
        indices = torch.full(
            (q.shape[0], 1, width), -1, dtype=torch.int32, device=q.device
        )
        lengths = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
        _window_indices_kernel[(metadata.num_decodes, metadata.max_query_len)](
            metadata.query_start_loc,
            seq_lens,
            metadata.decode.block_table,
            indices,
            lengths,
            self.dcp_rank,
            metadata.decode.block_table.stride(0),
            kv_cache.shape[1],
            self.sliding_window,
            width,
            triton.next_power_of_2(width),
            CP_WORLD=self.dcp_world_size,
        )
        num_heads = q.shape[1]
        alignment = (
            64
            if packed_fp8 or current_platform.is_device_capability_family(90)
            else 128
        )
        padded_heads = triton.cdiv(num_heads, alignment) * alignment
        if packed_fp8 and padded_heads not in (64, 128):
            raise NotImplementedError("FP8 FlashMLA supports at most 128 query heads.")
        if num_heads != padded_heads:
            padded_q = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_q[:, :num_heads] = q
            q = padded_q
        if packed_fp8:
            # Record scheduler generation in every graph: window lengths change
            # on replay, so initialized scheduling metadata cannot be reused.
            scheduler, splits = get_mla_metadata()
            output, lse = flash_mla_with_kvcache(
                q.unsqueeze(1),
                kv_cache.unsqueeze(2),
                None,
                None,
                self.kv_lora_rank,
                scheduler,
                splits,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
                indices=indices,
                topk_length=(
                    lengths
                    if current_platform.is_device_capability_family(100)
                    else None
                ),
            )
            output = output.squeeze(1)
            lse = lse.squeeze(-1)
        else:
            output, _, lse = flash_mla_sparse_fwd(
                q,
                kv_cache.view(-1, 1, 576),
                indices,
                self.scale,
                topk_length=lengths,
            )
        output = output[:, :num_heads].contiguous()
        if not dcp:
            return output, None
        # A rank can hold none of a short sequence's window; its share must carry
        # no weight when the ranks' partial outputs are merged.
        empty = lengths == 0
        output.masked_fill_(empty.view(-1, 1, 1), 0.0)
        lse = lse[:, :num_heads].contiguous()
        lse.masked_fill_(empty.view(-1, 1), float("-inf"))
        return output, lse


class FlashMLAWindowedFP8Backend(FlashMLAWindowedBackend):
    """Packed FP8 windows sharing an allocation with other cache formats."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)
