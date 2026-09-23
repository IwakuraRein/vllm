# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class DFlashMLAAttention(nn.Module):
    """RoPE MLA with a latent cache populated from target context and draft queries."""

    def __init__(
        self,
        config,
        *,
        sliding_window: int | None,
        causal: bool,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if causal:
            raise ValueError("MLA DFlash requires non-causal draft attention.")
        if any(
            getattr(config, name, False)
            for name in ("mla_use_nope", "mla_use_output_gate", "mla_use_qk_norm")
        ):
            raise ValueError(
                "MLA DFlash requires RoPE without output gating or QK norm."
            )
        if config.q_lora_rank is None:
            raise ValueError("MLA DFlash requires q_lora_rank.")
        parallel_config = get_current_vllm_config().parallel_config
        if (
            parallel_config.decode_context_parallel_size > 1
            or parallel_config.prefill_context_parallel_size > 1
        ):
            raise ValueError("MLA DFlash does not yet support context parallelism.")

        self.causal = causal
        self.sliding_window = sliding_window
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        self.num_heads = config.num_attention_heads // tp_size

        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            config.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters = dict(config.rope_parameters)
        self.scaling = self.qk_head_dim**-0.5
        if rope_parameters["rope_type"] in ("yarn", "deepseek_yarn"):
            rope_parameters["rope_type"] = "deepseek_yarn"
            self.scaling *= (
                yarn_get_mscale(
                    rope_parameters["factor"], rope_parameters.get("mscale_all_dim", 0)
                )
                ** 2
            )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )
        attn_backend = None
        if sliding_window is not None:
            from vllm.v1.attention.backends.mla.flashmla_windowed import (
                FlashMLAWindowedBackend,
            )
            from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

            vllm_config = get_current_vllm_config()
            attn_backend = TritonMLABackend
            if (
                vllm_config.attention_config.backend != AttentionBackendEnum.TRITON_MLA
                and vllm_config.model_config.dtype == torch.bfloat16
                and (
                    cache_config is None
                    or cache_config.cache_dtype in ("auto", "bfloat16")
                )
                and self.kv_lora_rank == 512
                and self.qk_rope_head_dim == 64
                and is_flashmla_sparse_supported()[0]
            ):
                attn_backend = FlashMLAWindowedBackend

        self.attn = MLAAttention(
            self.num_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            config.q_lora_rank,
            self.kv_lora_rank,
            self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_backend=attn_backend,
            sliding_window=sliding_window,
            non_causal_multi_token_decode=True,
        )

    def project_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return self.kv_a_layernorm(kv_c.contiguous()), k_pe.contiguous()

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q, _ = self.q_a_proj(hidden_states)
        q, _ = self.q_b_proj(self.q_a_layernorm(q))
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_c, k_pe = self.project_kv(hidden_states)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat((q_nope, q_pe), dim=-1)
        output = self.attn(
            q,
            kv_c,
            k_pe,
            output_shape=torch.Size(
                (hidden_states.shape[0], self.num_heads * self.v_head_dim)
            ),
        )
        return self.o_proj(output)[0]

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        kv_c, k_pe = self.project_kv(hidden_states)
        # The DeepSeek YaRN FlashInfer wrapper requires both Q and K. Rotate
        # this standalone context key with the same cache via the CUDA op.
        ops.rotary_embedding(
            positions,
            k_pe,
            None,
            self.qk_rope_head_dim,
            self.rotary_emb.cos_sin_cache,
            self.rotary_emb.is_neox_style,
        )
        if slot_mapping is not None:
            self.attn.update_kv_cache(
                kv_c,
                k_pe.unsqueeze(1),
                self.attn.kv_cache,
                slot_mapping,
                None,
                self.attn.kv_cache_dtype,
                self.attn._k_scale,
            )
