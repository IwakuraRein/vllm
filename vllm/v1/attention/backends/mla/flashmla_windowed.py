# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    is_flashmla_sparse_supported,
)


@triton.jit
def _window_indices_kernel(
    QueryStart,
    SeqLens,
    BlockTable,
    Indices,
    Lengths,
    table_stride: tl.constexpr,
    page_size: tl.constexpr,
    window: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    query = tl.program_id(1)
    start = tl.load(QueryStart + request)
    query_len = tl.load(QueryStart + request + 1) - start
    seq_len = tl.load(SeqLens + request)
    position = seq_len - query_len + query
    left = tl.maximum(0, position - window + 1)
    right = tl.minimum(seq_len, position + window)
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


class FlashMLAWindowedBackend(TritonMLABackend):
    """Model-selected non-causal windowed MLA with a Triton fallback."""

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_WINDOWED"

    @staticmethod
    def get_impl_cls() -> type["FlashMLAWindowedImpl"]:
        return FlashMLAWindowedImpl

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False


class FlashMLAWindowedImpl(TritonMLAImpl):
    def _forward_windowed_mqa(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, None]:
        if not (
            is_flashmla_sparse_supported()[0]
            and q.dtype == kv_cache.dtype == torch.bfloat16
            and q.shape[-1] == kv_cache.shape[-1] == 576
            and self.kv_lora_rank == 512
            and kv_cache.is_contiguous()
        ):
            return super()._forward_windowed_mqa(q, kv_cache, metadata, layer)

        assert not metadata.causal
        assert self.sliding_window is not None
        assert metadata.decode is not None
        # Future draft tokens extend the first query's visible range beyond W.
        max_keys = min(
            metadata.max_seq_len,
            2 * self.sliding_window - 1,
            self.sliding_window + metadata.max_query_len - 1,
        )
        width = triton.cdiv(max(1, max_keys), 128) * 128
        indices = torch.empty(
            (q.shape[0], 1, width), dtype=torch.int32, device=q.device
        )
        lengths = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
        _window_indices_kernel[(metadata.num_decodes, metadata.max_query_len)](
            metadata.query_start_loc,
            metadata.decode.seq_lens,
            metadata.decode.block_table,
            indices,
            lengths,
            metadata.decode.block_table.stride(0),
            kv_cache.shape[1],
            self.sliding_window,
            width,
            triton.next_power_of_2(width),
        )
        num_heads = q.shape[1]
        alignment = 64 if current_platform.is_device_capability_family(90) else 128
        padded_heads = triton.cdiv(num_heads, alignment) * alignment
        if num_heads != padded_heads:
            padded_q = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_q[:, :num_heads] = q
            q = padded_q
        output, _, _ = flash_mla_sparse_fwd(
            q,
            kv_cache.view(-1, 1, 576),
            indices,
            self.scale,
            topk_length=lengths,
        )
        return output[:, :num_heads].contiguous(), None
