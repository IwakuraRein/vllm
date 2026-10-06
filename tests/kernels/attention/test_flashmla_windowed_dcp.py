# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Windowed FlashMLA under decode context parallelism (DCP).

Each simulated DCP rank holds its share of the cache as vLLM lays it out with
cp_kv_cache_interleave_size=1: global position g on rank g % D at local position
g // D, in pages addressed through a block table. The ranks' partial outputs,
merged by their LSEs, must match one rank holding every token, which in turn
must match dense attention over each query's window.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.flashmla_windowed import FlashMLAWindowedImpl
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

DEVICE = "cuda"
NUM_HEADS = 64
KV_LORA_RANK = 512
HEAD_SIZE = 576
SCALE = 192**-0.5
WINDOW = 4096
PAGE_SIZE = 32
QUERY_LEN = 4
DCP_WORLD_SIZE = 8
# Global lengths, the query block included. At DCP 8, some ranks hold none of
# the 5- and 13-token sequences' windows.
SEQ_LENS = [5, 13, 3000, 9000, 20000]


def _dequantize_cache(cache: torch.Tensor) -> torch.Tensor:
    latent = cache[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = cache[..., 512:528].contiguous().view(torch.float32)
    rope = cache[..., 528:].contiguous().view(torch.bfloat16)
    latent = latent * scales.repeat_interleave(128, dim=-1)
    return torch.cat((latent.to(torch.bfloat16), rope), dim=-1)


def _prepare_rank(
    kv: list[torch.Tensor],
    q: torch.Tensor,
    rank: int,
    world_size: int,
    *,
    cache_dtype: str = "auto",
    padded_pages: bool = False,
):
    local = [x[rank::world_size] for x in kv]
    blocks = [triton.cdiv(x.shape[0], PAGE_SIZE) for x in local]
    num_pages = sum(blocks) + 7
    pages = torch.randperm(num_pages).tolist()
    if cache_dtype == "fp8_ds_mla":
        stride = (PAGE_SIZE + (4 if padded_pages else 0)) * 656
        offset = 656 if padded_pages else 0
        storage = torch.zeros(
            num_pages * stride + offset, dtype=torch.uint8, device=q.device
        )
        cache = storage.as_strided(
            (num_pages, PAGE_SIZE, 656), (stride, 656, 1), offset
        )
    else:
        cache = q.new_zeros(num_pages, PAGE_SIZE, HEAD_SIZE)
    block_table = torch.zeros(
        len(local), max(blocks) + 1, dtype=torch.int32, device=q.device
    )
    for r, x in enumerate(local):
        for b in range(blocks[r]):
            page = pages.pop()
            block_table[r, b] = page
            chunk = x[b * PAGE_SIZE : (b + 1) * PAGE_SIZE]
            if cache_dtype == "fp8_ds_mla":
                slots = torch.arange(
                    page * PAGE_SIZE,
                    page * PAGE_SIZE + chunk.shape[0],
                    dtype=torch.int64,
                    device=q.device,
                )
                ops.concat_and_cache_mla(
                    chunk[:, :KV_LORA_RANK].contiguous(),
                    chunk[:, KV_LORA_RANK:].contiguous(),
                    cache,
                    slots,
                    cache_dtype,
                    torch.ones(1, dtype=torch.float32, device=q.device),
                )
            else:
                cache[page, : chunk.shape[0]] = chunk

    # Only the attributes _forward_windowed_mqa reads; __init__ wants a full config.
    impl = object.__new__(FlashMLAWindowedImpl)
    impl.dcp_world_size = world_size
    impl.dcp_rank = rank
    impl.kv_lora_rank = KV_LORA_RANK
    impl.kv_cache_dtype = cache_dtype
    impl.sliding_window = WINDOW
    impl.scale = SCALE

    def lens(xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [x.shape[0] for x in xs], dtype=torch.int32, device=q.device
        )

    query_len = q.shape[0] // len(kv)
    metadata: Any = SimpleNamespace(
        causal=False,
        num_decodes=len(kv),
        max_query_len=query_len,
        max_seq_len=max(x.shape[0] for x in local),
        query_start_loc=torch.arange(
            0,
            (len(kv) + 1) * query_len,
            query_len,
            dtype=torch.int32,
            device=q.device,
        ),
        decode=SimpleNamespace(
            seq_lens=lens(local),
            dcp_tot_seq_lens=lens(kv) if world_size > 1 else None,
            block_table=block_table,
        ),
    )
    return impl, cache, metadata


def _forward_rank(kv, q, rank, world_size, **kwargs):
    impl, cache, metadata = _prepare_rank(kv, q, rank, world_size, **kwargs)
    return impl._forward_windowed_mqa(q, cache, metadata, None)


def _cached_sequences(cache, block_table, seq_lens):
    decoded = _dequantize_cache(cache)
    return [
        decoded[pages.long()].flatten(0, 1)[:length]
        for pages, length in zip(block_table, seq_lens)
    ]


def _dense(
    kv: list[torch.Tensor], q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(q.shape[0], q.shape[1], KV_LORA_RANK, device=q.device)
    lse = torch.empty(q.shape[0], q.shape[1], device=q.device)
    query_len = q.shape[0] // len(kv)
    for r, x in enumerate(kv):
        seq_len = x.shape[0]
        for j in range(query_len):
            row = r * query_len + j
            position = seq_len - query_len + j
            keys = x[max(0, position - WINDOW + 1) : position + WINDOW].float()
            scores = q[row].float() @ keys.T * SCALE
            lse[row] = torch.logsumexp(scores, dim=-1)
            out[row] = torch.softmax(scores, dim=-1) @ keys[:, :KV_LORA_RANK]
    return out, lse


@pytest.mark.skipif(
    not is_flashmla_sparse_supported()[0], reason="FlashMLA sparse is not supported"
)
@torch.inference_mode()
def test_windowed_flashmla_dcp_merge_matches_single_rank():
    torch.manual_seed(0)
    kv = [
        torch.randn(s, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE) for s in SEQ_LENS
    ]
    q = torch.randn(
        len(SEQ_LENS) * QUERY_LEN,
        NUM_HEADS,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    ref_out, ref_lse = _forward_rank(kv, q, rank=0, world_size=1)
    assert ref_lse is None
    dense_out, dense_lse = _dense(kv, q)
    torch.testing.assert_close(ref_out.float(), dense_out, atol=5e-2, rtol=0)

    parts = [
        _forward_rank(kv, q, rank, DCP_WORLD_SIZE) for rank in range(DCP_WORLD_SIZE)
    ]
    assert all(lse is not None for _, lse in parts)
    outs = torch.stack([out.float() for out, _ in parts])
    lses = torch.stack([lse for _, lse in parts if lse is not None])
    # Rows where a rank holds none of the window carry zero weight in the merge.
    assert torch.isneginf(lses).all(dim=-1).any()
    peak = lses.max(dim=0).values
    weights = torch.exp(lses - peak)
    merged = (outs * weights.unsqueeze(-1)).sum(0) / weights.sum(0).unsqueeze(-1)
    merged_lse = peak + torch.log(weights.sum(0))

    assert torch.isfinite(merged).all()
    torch.testing.assert_close(merged, ref_out.float(), atol=5e-2, rtol=0)
    torch.testing.assert_close(merged_lse, dense_lse, atol=1e-2, rtol=0)


@pytest.mark.skipif(
    not is_flashmla_sparse_supported()[0], reason="FlashMLA sparse is not supported"
)
@pytest.mark.parametrize("num_heads", [8, 64])
@torch.inference_mode()
def test_fp8_windowed_dcp_matches_packed_cache_reference(num_heads):
    """Packed writes, offset/padded pages and empty DCP shards keep their meaning."""
    torch.manual_seed(1)
    kv = [
        torch.randn(s, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE) for s in SEQ_LENS
    ]
    q = torch.randn(
        len(kv) * QUERY_LEN,
        num_heads,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    kwargs = dict(cache_dtype="fp8_ds_mla", padded_pages=True)
    impl, cache, metadata = _prepare_rank(kv, q, 0, 1, **kwargs)
    assert cache.storage_offset() != 0 and not cache.is_contiguous()
    reference_kv = _cached_sequences(cache, metadata.decode.block_table, SEQ_LENS)
    expected, expected_lse = _dense(reference_kv, q)
    actual, lse = impl._forward_windowed_mqa(q, cache, metadata, None)
    assert lse is None
    torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=0)

    parts = [
        _forward_rank(kv, q, rank, DCP_WORLD_SIZE, **kwargs)
        for rank in range(DCP_WORLD_SIZE)
    ]
    outputs = torch.stack([out.float() for out, _ in parts])
    lses = torch.stack([lse for _, lse in parts])
    assert torch.isneginf(lses).all(dim=-1).any()
    peak = lses.max(dim=0).values
    weights = torch.exp(lses - peak)
    merged = (outputs * weights.unsqueeze(-1)).sum(0) / weights.sum(0).unsqueeze(-1)
    merged_lse = peak + torch.log(weights.sum(0))
    torch.testing.assert_close(merged, expected, atol=5e-2, rtol=0)
    torch.testing.assert_close(merged_lse, expected_lse, atol=1e-2, rtol=0)


@pytest.mark.skipif(
    not is_flashmla_sparse_supported()[0], reason="FlashMLA sparse is not supported"
)
@torch.inference_mode()
def test_fp8_windowed_graph_replay_updates_sequence_lengths_and_pages():
    """A replay must reschedule changing windows instead of retaining warmup lengths."""
    torch.manual_seed(2)
    lengths = [17, 129, 4353]
    kv = [
        torch.randn(s, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE) for s in lengths
    ]
    q = torch.randn(len(kv) * 8, 8, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE)
    impl, cache, metadata = _prepare_rank(
        kv, q, 0, 1, cache_dtype="fp8_ds_mla", padded_pages=True
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        impl._forward_windowed_mqa(q, cache, metadata, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, _ = impl._forward_windowed_mqa(q, cache, metadata, None)

    lengths = [9, 65, 4100]
    metadata.decode.seq_lens.copy_(torch.tensor(lengths, device=DEVICE))
    for r, length in enumerate(lengths):
        count = triton.cdiv(length, PAGE_SIZE)
        metadata.decode.block_table[r, :count] = metadata.decode.block_table[
            r, :count
        ].flip(0)
    q.normal_()
    graph.replay()
    torch.accelerator.synchronize()
    reference_kv = _cached_sequences(cache, metadata.decode.block_table, lengths)
    expected, _ = _dense(reference_kv, q)
    eager, _ = impl._forward_windowed_mqa(q, cache, metadata, None)
    torch.testing.assert_close(captured.float(), expected, atol=5e-2, rtol=0)
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
