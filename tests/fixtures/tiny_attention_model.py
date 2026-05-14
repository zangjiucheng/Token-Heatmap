"""Synthetic HuggingFace-style causal LM used by AttentionProbe tests.

These modules mimic the structure of HF decoder-only models (Qwen2, Llama 3, ...)
closely enough for `AttentionProbe` to exercise its hook lifecycle, GQA awareness,
and RoPE pre/post capture, without requiring any model download or network access.

The shapes follow the HF convention:
- `q_proj` output is reshaped to `[batch, H_q, seq, head_dim]`
- `k_proj` / `v_proj` outputs are reshaped to `[batch, H_kv, seq, head_dim]`
- attention forward returns `(attn_output, attn_weights)` where `attn_weights`
  has shape `[batch, H_q, q_seq, k_seq]`
- when the probe sets `_probe_capture_rope_active = True`, the attention
  module stashes its post-rotation Q and K as `_probe_q_last_post_rope` and
  `_probe_k_last_post_rope`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TinyConfig:
    """Minimal HF-style config exposing the attributes AttentionProbe reads."""

    hidden_size: int = 16
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 4
    vocab_size: int = 32
    use_rope: bool = False
    rope_base: float = 10000.0
    attn_implementation: str = "eager"
    _attn_implementation: str = "eager"
    # Optional extension fields, kept for completeness; unused by the probe.
    extra: dict[str, Any] = field(default_factory=dict)


def _apply_rope(t: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply a basic rotary embedding to a `[batch, heads, seq, head_dim]` tensor.

    Mirrors the canonical RoPE formulation: split the last dim into two halves
    and rotate them by the position-dependent angle. Deterministic and
    self-contained so tests don't need HF's rotary implementation.
    """
    batch, heads, seq, dim = t.shape
    half = dim // 2
    positions = torch.arange(seq, dtype=t.dtype, device=t.device).unsqueeze(-1)
    freqs = torch.arange(half, dtype=t.dtype, device=t.device)
    inv_freq = 1.0 / (base ** (freqs / half))
    angles = positions * inv_freq  # [seq, half]
    cos = angles.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half]
    sin = angles.sin().unsqueeze(0).unsqueeze(0)

    x1 = t[..., :half]
    x2 = t[..., half:]
    rotated_first = x1 * cos - x2 * sin
    rotated_second = x1 * sin + x2 * cos
    return torch.cat([rotated_first, rotated_second], dim=-1)


class TinyAttention(nn.Module):
    """A minimal eager attention module with optional GQA and optional RoPE."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)  # [batch, H_q, seq, d]
        k = k.transpose(1, 2)  # [batch, H_kv, seq, d]
        v = v.transpose(1, 2)

        if self.config.use_rope:
            q_rot = _apply_rope(q, base=self.config.rope_base)
            k_rot = _apply_rope(k, base=self.config.rope_base)
            if getattr(self, "_probe_capture_rope_active", False):
                self._probe_q_last_post_rope = q_rot[:, :, -1:, :]
                self._probe_k_last_post_rope = k_rot[:, :, -1:, :]
            q = q_rot
            k = k_rot

        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_full = k.repeat_interleave(repeat, dim=1)
            v_full = v.repeat_interleave(repeat, dim=1)
        else:
            k_full = k
            v_full = v

        scaling = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * scaling

        # Causal mask
        mask = torch.full((seq, seq), float("-inf"), device=q.device, dtype=q.dtype)
        mask = torch.triu(mask, diagonal=1)
        attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v_full)  # [batch, H_q, seq, d]
        context = context.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        output = self.o_proj(context)
        return output, attn_weights


class TinyDecoderLayer(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.self_attn = TinyAttention(config)
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(hidden_states)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


class TinyDecoder(nn.Module):
    """`model.model` in HF terms: the inner stack that owns `layers`."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class TinyCausalLM(nn.Module):
    """`AutoModelForCausalLM` analogue for tests."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TinyDecoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids)
        return self.lm_head(hidden)


def build_tiny_model(
    *,
    seed: int = 0,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int | None = None,
    head_dim: int = 4,
    vocab_size: int = 32,
    use_rope: bool = False,
    attn_implementation: str = "eager",
) -> TinyCausalLM:
    """Construct a deterministic tiny causal LM for tests.

    `num_key_value_heads=None` defaults to `num_attention_heads` (standard MHA).
    """

    torch.manual_seed(seed)
    cfg = TinyConfig(
        hidden_size=num_attention_heads * head_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads or num_attention_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        use_rope=use_rope,
        attn_implementation=attn_implementation,
        _attn_implementation=attn_implementation,
    )
    model = TinyCausalLM(cfg)
    model.eval()
    return model
