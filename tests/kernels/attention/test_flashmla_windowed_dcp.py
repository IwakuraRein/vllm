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


def _forward_rank(
    kv: list[torch.Tensor], q: torch.Tensor, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    local = [x[rank::world_size] for x in kv]
    blocks = [triton.cdiv(x.shape[0], PAGE_SIZE) for x in local]
    num_pages = sum(blocks) + 7
    pages = torch.randperm(num_pages).tolist()
    cache = q.new_zeros(num_pages, PAGE_SIZE, HEAD_SIZE)
    block_table = torch.zeros(
        len(local), max(blocks) + 1, dtype=torch.int32, device=q.device
    )
    for r, x in enumerate(local):
        for b in range(blocks[r]):
            page = pages.pop()
            block_table[r, b] = page
            chunk = x[b * PAGE_SIZE : (b + 1) * PAGE_SIZE]
            cache[page, : chunk.shape[0]] = chunk

    # Only the attributes _forward_windowed_mqa reads; __init__ wants a full config.
    impl = object.__new__(FlashMLAWindowedImpl)
    impl.dcp_world_size = world_size
    impl.dcp_rank = rank
    impl.kv_lora_rank = KV_LORA_RANK
    impl.sliding_window = WINDOW
    impl.scale = SCALE

    def lens(xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [x.shape[0] for x in xs], dtype=torch.int32, device=q.device
        )

    metadata: Any = SimpleNamespace(
        causal=False,
        num_decodes=len(kv),
        max_query_len=QUERY_LEN,
        max_seq_len=max(x.shape[0] for x in local),
        query_start_loc=torch.arange(
            0,
            (len(kv) + 1) * QUERY_LEN,
            QUERY_LEN,
            dtype=torch.int32,
            device=q.device,
        ),
        decode=SimpleNamespace(
            seq_lens=lens(local),
            dcp_tot_seq_lens=lens(kv) if world_size > 1 else None,
            block_table=block_table,
        ),
    )
    layer: Any = None  # The windowed path doesn't read the layer.
    return impl._forward_windowed_mqa(q, cache, metadata, layer)


def _dense(
    kv: list[torch.Tensor], q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(q.shape[0], NUM_HEADS, KV_LORA_RANK, device=q.device)
    lse = torch.empty(q.shape[0], NUM_HEADS, device=q.device)
    for r, x in enumerate(kv):
        seq_len = x.shape[0]
        for j in range(QUERY_LEN):
            row = r * QUERY_LEN + j
            position = seq_len - QUERY_LEN + j
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
