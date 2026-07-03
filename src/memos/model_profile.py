"""Model architecture profiling for roofline analysis.

Provides accurate flops_per_token and bytes_per_token calculations
for any vLLM-supported model, derived from HuggingFace config fields.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields

from memos.types import InferenceConfig

_GATED_FFN = frozenset(
    {
        "llama",
        "codellama",
        "mistral",
        "mixtral",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma4",
        "qwen",
        "qwen2",
        "qwen3",
        "qwen2_moe",
        "qwen3_moe",
        "cohere",
        "cohere2",
        "cohere2_moe",
        "phi3",
        "phi4",
        "phimoe",
        "internlm2",
        "internlm3",
        "yi",
        "olmo",
        "olmo2",
        "olmo3",
        "starcoder2",
        "granite",
        "granitemoe",
        "granite_moe_hybrid",
        "solar",
        "exaone",
        "exaone4",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v4",
        "jamba",
        "zamba2",
        "dbrx",
        "arctic",
        "olmoe",
        "grok",
        "grok1",
        "glm4",
        "glm4_moe",
        "nemotron",
        "nemotron_h",
        "falcon_h1",
        "bamba",
        "minicpm",
        "minicpm3",
    }
)

_PURE_SSM = frozenset({"mamba", "mamba2", "falcon_mamba"})

_HYBRID_SSM = frozenset(
    {
        "jamba",
        "zamba2",
        "bamba",
        "falcon_h1",
        "nemotron_h",
        "granite_moe_hybrid",
        "olmo_hybrid",
    }
)


@dataclass(slots=True)
class ModelProfile:
    """Minimal architecture descriptor. All methods are O(1) arithmetic."""

    # Universal
    num_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    dtype_bytes: int = 2
    tie_word_embeddings: bool = True

    # Attention
    head_dim: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    sliding_window: int = 0

    # MLA (DeepSeek)
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0

    # FFN
    intermediate_size: int = 0
    ffn_multiplier: int = 3

    # MoE
    num_experts: int = 1
    top_k_experts: int = 1
    expert_intermediate_size: int = 0
    n_shared_experts: int = 0
    shared_expert_intermediate_size: int = 0

    # SSM
    ssm_state_size: int = 0
    ssm_expand: int = 2
    ssm_conv_kernel: int = 4

    # KV cache precision
    kv_dtype_bytes: int = 0

    # Pre-computed layer counts (two independent axes)
    n_standard_attn: int = 0
    n_mla: int = 0
    n_ssm: int = 0
    n_dense_ffn: int = 0
    n_moe: int = 0
    n_sliding_window_attn: int = 0
    ssm_has_ffn: bool = False

    # --- Derived properties ---

    @property
    def _kv_dt(self) -> int:
        return self.kv_dtype_bytes or self.dtype_bytes

    @property
    def _expert_inter(self) -> int:
        return self.expert_intermediate_size or self.intermediate_size

    @property
    def _n_full_attn(self) -> int:
        return self.n_standard_attn - self.n_sliding_window_attn

    @property
    def _ssm_inner(self) -> int:
        return self.ssm_expand * self.hidden_size

    # --- Public methods ---

    def weight_bytes(self) -> int:
        """Total model weight footprint in bytes (all MoE experts stored)."""
        h = self.hidden_size
        d = self.head_dim
        nh = self.num_heads
        nkv = self.num_kv_heads
        dt = self.dtype_bytes

        total = 0

        # Standard attention layers
        if self.n_standard_attn > 0:
            attn_per = h * d * (2 * nh + 2 * nkv)
            total += self.n_standard_attn * attn_per

        # MLA layers
        if self.n_mla > 0:
            kv_down = h * self.kv_lora_rank
            kv_rope = h * self.qk_rope_head_dim
            if self.q_lora_rank > 0:
                q_proj = h * self.q_lora_rank + self.q_lora_rank * nh * (
                    self.qk_nope_head_dim + self.qk_rope_head_dim
                )
            else:
                q_proj = h * nh * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            o_proj = nh * self.v_head_dim * h
            total += self.n_mla * (kv_down + kv_rope + q_proj + o_proj)

        # SSM layers
        if self.n_ssm > 0:
            inner = self._ssm_inner
            ssm_per = h * 2 * inner + inner * self.ssm_conv_kernel + inner * h
            total += self.n_ssm * ssm_per

        # Dense FFN layers
        if self.n_dense_ffn > 0:
            total += self.n_dense_ffn * self.ffn_multiplier * h * self.intermediate_size

        # MoE FFN layers
        if self.n_moe > 0:
            ei = self._expert_inter
            experts = self.num_experts * self.ffn_multiplier * h * ei
            shared = (
                self.n_shared_experts
                * self.ffn_multiplier
                * h
                * self.shared_expert_intermediate_size
            )
            router = h * self.num_experts
            total += self.n_moe * (experts + shared + router)

        # Norms (2 per layer: pre-attn + pre-ffn)
        total += self.num_layers * 2 * h

        # Embeddings
        total += h * self.vocab_size
        if not self.tie_word_embeddings:
            total += h * self.vocab_size

        return total * dt

    def flops_per_token(
        self,
        seq: int,
        batch: int = 1,
        inference_config: InferenceConfig | None = None,
    ) -> float:
        """Decode FLOPs for generating 1 token with `seq` cached tokens."""
        h = self.hidden_size
        d = self.head_dim
        nh = self.num_heads
        nkv = self.num_kv_heads
        total = 0.0

        # Standard attention
        if self.n_standard_attn > 0:
            proj = 2 * h * d * (2 * nh + 2 * nkv)
            attn = 4 * nh * d * seq
            total += self.n_standard_attn * (proj + attn)

        # MLA attention
        if self.n_mla > 0:
            nope = self.qk_nope_head_dim
            rope = self.qk_rope_head_dim
            klr = self.kv_lora_rank
            vd = self.v_head_dim

            if self.q_lora_rank > 0:
                q_proj = 2 * h * self.q_lora_rank + 2 * self.q_lora_rank * nh * (
                    nope + rope
                )
            else:
                q_proj = 2 * h * nh * (nope + rope)
            kv_proj = 2 * h * klr + 2 * h * rope
            o_proj = 2 * nh * vd * h
            q_absorb = 2 * nh * nope * klr
            qk = 2 * nh * (klr + rope) * seq
            sv = 2 * nh * klr * seq
            v_decomp = 2 * nh * klr * vd
            total += self.n_mla * (
                q_proj + kv_proj + o_proj + q_absorb + qk + sv + v_decomp
            )

        # SSM layers
        if self.n_ssm > 0:
            inner = self._ssm_inner
            state = self.ssm_state_size
            in_proj = 2 * h * 2 * inner
            conv = 2 * inner * self.ssm_conv_kernel
            scan = 6 * inner * state
            out_proj = 2 * inner * h
            total += self.n_ssm * (in_proj + conv + scan + out_proj)

        # Dense FFN
        if self.n_dense_ffn > 0:
            total += (
                self.n_dense_ffn * 2 * self.ffn_multiplier * h * self.intermediate_size
            )

        # MoE FFN
        if self.n_moe > 0:
            ei = self._expert_inter
            router = 2 * h * self.num_experts
            active = self.top_k_experts * 2 * self.ffn_multiplier * h * ei
            shared = (
                self.n_shared_experts
                * 2
                * self.ffn_multiplier
                * h
                * self.shared_expert_intermediate_size
            )
            total += self.n_moe * (router + active + shared)

        # LM head
        total += 2 * h * self.vocab_size

        tp = max((inference_config.tp if inference_config else 1), 1)
        return total / tp

    def bytes_per_token(
        self,
        seq: int,
        batch: int = 1,
        inference_config: InferenceConfig | None = None,
    ) -> float:
        """Memory traffic per decode token (weight reads + KV cache reads)."""
        cfg_batch = max((inference_config.batch_size if inference_config else batch), 1)
        tp = max((inference_config.tp if inference_config else 1), 1)
        w = self._effective_active_weight_bytes(inference_config) / (cfg_batch * tp)
        kv = self._kv_read_bytes(seq, inference_config=inference_config) / tp
        return w + kv

    def kv_cache_bytes(self, seq: int, batch: int = 1) -> int:
        """Total KV cache capacity needed for `batch` sessions at `seq` length."""
        kv_dt = self._kv_dt
        d = self.head_dim
        nkv = self.num_kv_heads
        total = 0

        # Full-context standard attention
        n_full = self._n_full_attn
        if n_full > 0:
            total += batch * n_full * seq * 2 * nkv * d * kv_dt

        # Sliding-window standard attention
        n_sw = self.n_sliding_window_attn
        if n_sw > 0:
            eff_seq = min(seq, self.sliding_window) if self.sliding_window else seq
            total += batch * n_sw * eff_seq * 2 * nkv * d * kv_dt

        # MLA
        if self.n_mla > 0:
            total += (
                batch
                * self.n_mla
                * seq
                * (self.kv_lora_rank + self.qk_rope_head_dim)
                * kv_dt
            )

        # SSM state (fixed per session, does not grow with seq)
        if self.n_ssm > 0:
            total += batch * self.n_ssm * self._ssm_inner * self.ssm_state_size * kv_dt

        return total

    def _active_weight_bytes(self) -> int:
        """Weight bytes actually READ per decode step (excludes inactive MoE experts)."""
        total = self.weight_bytes()
        if self.n_moe > 0 and self.num_experts > self.top_k_experts:
            inactive = self.num_experts - self.top_k_experts
            per_expert = (
                self.ffn_multiplier
                * self.hidden_size
                * self._expert_inter
                * self.dtype_bytes
            )
            total -= self.n_moe * inactive * per_expert
        return total

    def _effective_active_weight_bytes(
        self, inference_config: InferenceConfig | None
    ) -> float:
        total = float(self._active_weight_bytes())
        if not inference_config:
            return total

        # Adjust for effective storage precision (e.g. INT8/FP8/INT4 weights).
        base_dt = float(self.dtype_bytes)
        effective_dt = max(float(inference_config.weight_dtype_bytes), 1e-9)
        total *= effective_dt / base_dt

        # Group-quantized weights carry scale metadata overhead.
        group_size = max(int(inference_config.weight_group_size), 0)
        if group_size > 0:
            total *= 1.0 + (4.0 / (group_size * effective_dt))
        return total

    def _kv_read_bytes(
        self, seq: int, inference_config: InferenceConfig | None = None
    ) -> float:
        """Bytes read from KV cache per decode step."""
        kv_dt = float(
            inference_config.kv_dtype_bytes if inference_config else self._kv_dt
        )
        d = self.head_dim
        nkv = self.num_kv_heads
        total = 0.0

        # Full-context standard attention
        n_full = self._n_full_attn
        if n_full > 0:
            total += n_full * seq * 2 * nkv * d * kv_dt

        # Sliding-window standard attention
        n_sw = self.n_sliding_window_attn
        if n_sw > 0:
            eff_seq = min(seq, self.sliding_window) if self.sliding_window else seq
            total += n_sw * eff_seq * 2 * nkv * d * kv_dt

        # MLA
        if self.n_mla > 0:
            total += (
                self.n_mla * seq * (self.kv_lora_rank + self.qk_rope_head_dim) * kv_dt
            )

        # SSM state (fixed, tiny)
        if self.n_ssm > 0:
            total += self.n_ssm * self._ssm_inner * self.ssm_state_size * kv_dt

        return total

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Emit only non-default values for compact JSON."""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val != f.default:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, d: dict) -> ModelProfile:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def from_pretrained(
    model_name: str,
    dtype_bytes: int = 2,
    kv_dtype_bytes: int = 0,
) -> ModelProfile:
    """Build a ModelProfile from a HuggingFace model config.

    Downloads config.json (small) from HF hub. Does NOT download model weights.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", 0))
    vocab_size = getattr(config, "vocab_size", 0)
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 0))
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_heads if num_heads else 0)
    tie = getattr(config, "tie_word_embeddings", True)

    # FFN type
    if model_type in _GATED_FFN:
        ffn_mult = 3
    elif model_type in frozenset(
        {
            "gpt2",
            "gpt_j",
            "gpt_neox",
            "opt",
            "bloom",
            "falcon",
            "mpt",
            "phi",
            "persimmon",
            "stablelm",
        }
    ):
        ffn_mult = 2
    else:
        ffn_mult = 3 if getattr(config, "hidden_act", "") == "silu" else 2
        if model_type:
            print(
                f"[memos] Unknown model_type '{model_type}', assuming ffn_multiplier={ffn_mult}",
                file=sys.stderr,
            )

    intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)

    # MoE detection
    n_experts = getattr(
        config,
        "num_local_experts",
        getattr(config, "n_routed_experts", getattr(config, "num_experts", 1)),
    )
    top_k = getattr(
        config, "num_experts_per_tok", getattr(config, "num_experts_per_topk", 1)
    )
    expert_inter = getattr(config, "moe_intermediate_size", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    shared_inter = getattr(config, "shared_expert_intermediate_size", 0)

    # MLA detection
    kv_lora_rank = getattr(config, "kv_lora_rank", 0)
    q_lora_rank = getattr(config, "q_lora_rank", 0)
    qk_rope = getattr(config, "qk_rope_head_dim", 0)
    qk_nope = getattr(config, "qk_nope_head_dim", 0)
    v_head_dim = getattr(config, "v_head_dim", 0)

    # SSM detection
    ssm_state = 0
    ssm_expand = 2
    ssm_conv = 4
    ssm_has_ffn = False

    if model_type in _PURE_SSM:
        ssm_state = getattr(config, "state_size", 16)
        ssm_expand = getattr(config, "expand", 2)
        ssm_conv = getattr(config, "conv_kernel", 4)
        intermediate_size = 0  # pure Mamba has no FFN
        num_heads = 0
        num_kv_heads = 0
        head_dim = 0
    elif model_type in _HYBRID_SSM:
        ssm_state = getattr(config, "mamba_d_state", getattr(config, "state_size", 16))
        ssm_expand = getattr(config, "mamba_expand", getattr(config, "expand", 2))
        ssm_conv = getattr(config, "mamba_d_conv", getattr(config, "conv_kernel", 4))
        ssm_has_ffn = True

    # Sliding window
    sliding_window = getattr(config, "sliding_window", None) or 0

    # --- Layer count computation ---
    n_standard_attn = 0
    n_mla = 0
    n_ssm_layers = 0
    n_dense_ffn = 0
    n_moe_layers = 0
    n_sliding_window = 0

    if model_type in _PURE_SSM:
        n_ssm_layers = num_layers
        # Pure Mamba: no FFN, no attention
        n_dense_ffn = 0
        n_moe_layers = 0
    elif hasattr(config, "layer_types") and config.layer_types:
        # Gemma 2, Falcon-H1, etc: explicit per-layer type list
        lt = config.layer_types
        for t in lt:
            if t == "sliding_attention":
                n_standard_attn += 1
                n_sliding_window += 1
            elif "attention" in t or t == "full_attention":
                n_standard_attn += 1
            elif "mamba" in t:
                n_ssm_layers += 1
            else:
                n_standard_attn += 1  # unknown -> assume attention
        # All layers have FFN in layer_types models
        if n_experts > 1:
            # Check for expert_layer_period
            if hasattr(config, "expert_layer_period"):
                period = config.expert_layer_period
                offset = getattr(config, "expert_layer_offset", 0)
                n_moe_layers = sum(
                    1
                    for i in range(num_layers)
                    if (i - offset) % period == 0 and i >= offset
                )
                n_dense_ffn = num_layers - n_moe_layers
            else:
                n_moe_layers = num_layers
                n_dense_ffn = 0
        else:
            n_dense_ffn = num_layers
            n_moe_layers = 0
    elif hasattr(config, "attn_layer_period"):
        # Jamba/Zamba: period-based hybrid
        attn_period = config.attn_layer_period
        attn_offset = getattr(config, "attn_layer_offset", 0)
        n_standard_attn = sum(
            1
            for i in range(num_layers)
            if (i - attn_offset) % attn_period == 0 and i >= attn_offset
        )
        n_ssm_layers = num_layers - n_standard_attn

        if n_experts > 1 and hasattr(config, "expert_layer_period"):
            period = config.expert_layer_period
            offset = getattr(config, "expert_layer_offset", 0)
            n_moe_layers = sum(
                1
                for i in range(num_layers)
                if (i - offset) % period == 0 and i >= offset
            )
            n_dense_ffn = num_layers - n_moe_layers
        elif n_experts > 1:
            n_moe_layers = num_layers
            n_dense_ffn = 0
        else:
            n_dense_ffn = num_layers
            n_moe_layers = 0
    else:
        # DeepSeek or uniform model
        first_k = getattr(config, "first_k_dense_replace", 0)

        if kv_lora_rank > 0:
            # MLA model (DeepSeek)
            n_standard_attn = first_k
            n_mla = num_layers - first_k
        else:
            # Standard transformer
            n_standard_attn = num_layers
            n_mla = 0

        # Sliding window: if set and no layer_types, ALL standard attn layers use it
        if sliding_window > 0 and n_standard_attn > 0:
            n_sliding_window = n_standard_attn

        # FFN layer counts
        if n_experts > 1:
            moe_freq = getattr(config, "moe_layer_freq", 1)
            if first_k > 0:
                n_dense_ffn = first_k
                remaining = num_layers - first_k
                n_moe_layers = sum(1 for i in range(remaining) if i % moe_freq == 0)
                n_dense_ffn += remaining - n_moe_layers
            else:
                n_moe_layers = sum(1 for i in range(num_layers) if i % moe_freq == 0)
                n_dense_ffn = num_layers - n_moe_layers
        else:
            n_dense_ffn = num_layers
            n_moe_layers = 0

    return ModelProfile(
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtype_bytes=dtype_bytes,
        tie_word_embeddings=tie,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        sliding_window=sliding_window,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=qk_rope,
        qk_nope_head_dim=qk_nope,
        v_head_dim=v_head_dim,
        intermediate_size=intermediate_size,
        ffn_multiplier=ffn_mult,
        num_experts=n_experts,
        top_k_experts=top_k,
        expert_intermediate_size=expert_inter,
        n_shared_experts=n_shared,
        shared_expert_intermediate_size=shared_inter,
        ssm_state_size=ssm_state,
        ssm_expand=ssm_expand,
        ssm_conv_kernel=ssm_conv,
        kv_dtype_bytes=kv_dtype_bytes,
        n_standard_attn=n_standard_attn,
        n_mla=n_mla,
        n_ssm=n_ssm_layers,
        n_dense_ffn=n_dense_ffn,
        n_moe=n_moe_layers,
        n_sliding_window_attn=n_sliding_window,
        ssm_has_ffn=ssm_has_ffn,
    )
