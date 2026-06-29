"""Attention weight and Q/K/V capture via HuggingFace forward hooks.

`AttentionProbe` is a sibling to `AdaptiveTokenProbe`: opt-in, eager-attention-only,
architecture-aware. It owns forward hooks on each attention layer's `q_proj`,
`k_proj`, `v_proj`, and the attention layer itself, draining captures into a
per-step `AttentionStats` payload that downstream tickets reshape into trace
schema entries, statistics, etc.

The probe captures the last token's row of the attention distribution and the
last token's Q / K / V vectors. KV-cache historical entries are intentionally
not duplicated. Use `capture_full_distribution=True` to retain the full
`[H, S]` weight matrix; otherwise the matrix is sparsified to `top_k_positions`.

Post-RoPE Q/K capture relies on the attention module exposing
`_probe_q_last_post_rope` / `_probe_k_last_post_rope` attributes during its
forward pass (the probe sets `_probe_capture_rope_active = True` to opt-in).
The synthetic fixtures used by the test suite follow this convention; wiring
the same hook into HuggingFace's rotary kernels is left to downstream tickets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


class AttentionProbeError(RuntimeError):
    """Raised when `AttentionProbe` cannot operate on the given model."""


@dataclass
class AttentionProbeConfig:
    """Configuration for `AttentionProbe`.

    Attributes:
        layers: Either ``"all"`` (hook every decoder layer) or a list of layer
            indices to hook. Out-of-range indices raise on `attach`.
        capture_qkv: When True, capture per-step Q / K / V vectors for the last
            position. When False, only attention weights are captured.
        capture_full_distribution: When True, retain the full `[H, S]` attention
            weight matrix per layer. When False, retain a sparse representation
            of the top-``top_k_positions`` attended source positions per head.
        top_k_positions: How many top-attended positions to keep in sparse mode.
        capture_pre_rope_qk: When True, capture both pre-RoPE Q/K (from the
            projection outputs) and post-RoPE Q/K (from attributes the
            attention module stashes during forward).
    """

    layers: str | list[int] = "all"
    capture_qkv: bool = True
    capture_full_distribution: bool = False
    top_k_positions: int = 8
    capture_pre_rope_qk: bool = False


@dataclass
class AttentionLayerStats:
    """Per-layer captures for a single generation step."""

    layer_idx: int
    attention_weights: torch.Tensor | dict[str, torch.Tensor]
    q_last: torch.Tensor | None = None
    k_last: torch.Tensor | None = None
    v_last: torch.Tensor | None = None
    q_last_pre_rope: torch.Tensor | None = None
    k_last_pre_rope: torch.Tensor | None = None


@dataclass
class AttentionStats:
    """All per-layer captures plus the architecture metadata needed to read them.

    `head_to_kv_group[h]` gives the KV-head index that query head `h` shares
    keys and values with under Grouped Query Attention. Under standard MHA
    (no GQA), this is the identity mapping.
    """

    layers: dict[int, AttentionLayerStats]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    head_to_kv_group: list[int] = field(default_factory=list)


def _resolve_decoder_layers(model: Any) -> list[nn.Module]:
    """Locate the decoder layer list on a HuggingFace-style causal LM."""

    candidates: list[Any] = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(model)

    for parent in candidates:
        layers = getattr(parent, "layers", None)
        if layers is None:
            continue
        try:
            return list(layers)
        except TypeError:
            continue
    return []


def _get_self_attn(layer_module: nn.Module) -> nn.Module | None:
    return getattr(layer_module, "self_attn", None)


class AttentionProbe(nn.Module):
    """Capture per-layer attention weights and Q/K/V via forward hooks.

    The probe is opt-in: until `attach` is called, importing or instantiating
    it has no effect on generation. `attach` is idempotent, as is `detach`.

    Example:
        >>> probe = AttentionProbe(AttentionProbeConfig(layers=[0, 2]))
        >>> probe.attach(model)
        >>> model(input_ids)  # run a step
        >>> stats = probe.capture_step()
        >>> probe.detach()
    """

    def __init__(self, config: AttentionProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or AttentionProbeConfig()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._model: Any = None
        self._original_attn_impl: str | None = None
        self._original_attn_impl_underscore: str | None = None
        self._had_attn_impl: bool = False
        self._had_attn_impl_underscore: bool = False
        self._target_layers: list[int] = []
        self._num_attention_heads: int = 0
        self._num_key_value_heads: int = 0
        self._head_dim: int = 0
        self._captured_attn_modules: list[nn.Module] = []
        self._buf_attn_weights: dict[int, torch.Tensor] = {}
        self._buf_q_pre: dict[int, torch.Tensor] = {}
        self._buf_k_pre: dict[int, torch.Tensor] = {}
        self._buf_v_pre: dict[int, torch.Tensor] = {}
        self._buf_q_post: dict[int, torch.Tensor] = {}
        self._buf_k_post: dict[int, torch.Tensor] = {}

    @property
    def is_attached(self) -> bool:
        return self._model is not None

    @property
    def target_layers(self) -> list[int]:
        return list(self._target_layers)

    def attach(self, model: Any) -> AttentionProbe:
        """Install forward hooks on the model's attention layers.

        Forces `config.attn_implementation = "eager"` so that the eager kernel
        returns attention weights as the second tuple element. The original
        value (if any) is restored on `detach`. Calling `attach` while already
        attached is a no-op.
        """

        if self.is_attached:
            return self

        decoder_layers = _resolve_decoder_layers(model)
        if not decoder_layers:
            raise AttentionProbeError(
                "Could not locate a decoder layer list on the model. Expected "
                "`model.model.layers` or `model.layers` (HuggingFace causal-LM convention)."
            )

        model_config = getattr(model, "config", None)
        if model_config is None:
            raise AttentionProbeError("Model has no `config` attribute.")

        num_heads = getattr(model_config, "num_attention_heads", None)
        if num_heads is None:
            raise AttentionProbeError("`config.num_attention_heads` is required.")
        num_kv_heads = int(getattr(model_config, "num_key_value_heads", num_heads))
        num_heads = int(num_heads)

        if hasattr(model_config, "head_dim") and model_config.head_dim is not None:
            head_dim = int(model_config.head_dim)
        else:
            hidden_size = getattr(model_config, "hidden_size", None)
            if hidden_size is None:
                raise AttentionProbeError(
                    "Cannot infer head_dim: config exposes neither `head_dim` nor `hidden_size`."
                )
            head_dim = int(hidden_size) // num_heads

        if num_heads % num_kv_heads != 0:
            raise AttentionProbeError(
                f"num_attention_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({num_kv_heads})."
            )

        if isinstance(self.config.layers, str):
            if self.config.layers != "all":
                raise AttentionProbeError(
                    f"config.layers must be 'all' or a list of ints; got {self.config.layers!r}."
                )
            target = list(range(len(decoder_layers)))
        elif isinstance(self.config.layers, list):
            target = sorted({int(i) for i in self.config.layers})
            for idx in target:
                if idx < 0 or idx >= len(decoder_layers):
                    raise AttentionProbeError(
                        f"layer index {idx} out of range for model with {len(decoder_layers)} layers."
                    )
        else:
            raise AttentionProbeError(
                "config.layers must be 'all' or a list of ints; "
                f"got {type(self.config.layers).__name__}."
            )

        attn_modules: list[nn.Module] = []
        for idx in target:
            attn = _get_self_attn(decoder_layers[idx])
            if attn is None:
                raise AttentionProbeError(
                    f"Layer {idx} has no `self_attn` submodule; model architecture is unsupported."
                )
            missing = [name for name in ("q_proj", "k_proj", "v_proj") if not hasattr(attn, name)]
            if missing:
                raise AttentionProbeError(
                    f"Layer {idx} self_attn is missing required projection(s): "
                    f"{', '.join(missing)}. AttentionProbe only supports models that expose "
                    "explicit q_proj/k_proj/v_proj submodules."
                )
            attn_modules.append(attn)

        self._num_attention_heads = num_heads
        self._num_key_value_heads = num_kv_heads
        self._head_dim = head_dim
        self._target_layers = target

        self._had_attn_impl = hasattr(model_config, "attn_implementation")
        self._had_attn_impl_underscore = hasattr(model_config, "_attn_implementation")
        if self._had_attn_impl:
            self._original_attn_impl = model_config.attn_implementation
            model_config.attn_implementation = "eager"
        if self._had_attn_impl_underscore:
            self._original_attn_impl_underscore = model_config._attn_implementation
            model_config._attn_implementation = "eager"

        for layer_idx, attn in zip(target, attn_modules, strict=True):
            self._handles.append(
                attn.q_proj.register_forward_hook(self._make_proj_hook(layer_idx, "q"))
            )
            self._handles.append(
                attn.k_proj.register_forward_hook(self._make_proj_hook(layer_idx, "k"))
            )
            self._handles.append(
                attn.v_proj.register_forward_hook(self._make_proj_hook(layer_idx, "v"))
            )
            self._handles.append(attn.register_forward_hook(self._make_attn_hook(layer_idx)))
            if self.config.capture_pre_rope_qk:
                attn._probe_capture_rope_active = True
            self._captured_attn_modules.append(attn)

        self._model = model
        return self

    def detach(self) -> AttentionProbe:
        """Remove hooks, restore attention implementation, clear buffers."""

        if not self.is_attached:
            return self

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        for attn in self._captured_attn_modules:
            if hasattr(attn, "_probe_capture_rope_active"):
                delattr(attn, "_probe_capture_rope_active")
            for attr in ("_probe_q_last_post_rope", "_probe_k_last_post_rope"):
                if hasattr(attn, attr):
                    delattr(attn, attr)
        self._captured_attn_modules.clear()

        model_config = getattr(self._model, "config", None)
        if model_config is not None:
            if self._had_attn_impl:
                model_config.attn_implementation = self._original_attn_impl
            if self._had_attn_impl_underscore:
                model_config._attn_implementation = self._original_attn_impl_underscore

        self._original_attn_impl = None
        self._original_attn_impl_underscore = None
        self._had_attn_impl = False
        self._had_attn_impl_underscore = False
        self._model = None
        self._target_layers = []
        self._clear_buffers()
        return self

    def _clear_buffers(self) -> None:
        self._buf_attn_weights.clear()
        self._buf_q_pre.clear()
        self._buf_k_pre.clear()
        self._buf_v_pre.clear()
        self._buf_q_post.clear()
        self._buf_k_post.clear()

    def _make_proj_hook(self, layer_idx: int, which: str):
        target = {"q": self._buf_q_pre, "k": self._buf_k_pre, "v": self._buf_v_pre}[which]

        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            target[layer_idx] = output.detach()

        return hook

    def _make_attn_hook(self, layer_idx: int):
        def hook(module: nn.Module, _inputs: Any, output: Any) -> None:
            attn_weights: torch.Tensor | None = None
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1].detach()
            elif (cached := getattr(module, "_probe_last_attn_weights", None)) is not None:
                attn_weights = cached.detach()
            if attn_weights is not None:
                self._buf_attn_weights[layer_idx] = attn_weights

            q_post = getattr(module, "_probe_q_last_post_rope", None)
            k_post = getattr(module, "_probe_k_last_post_rope", None)
            if q_post is not None:
                self._buf_q_post[layer_idx] = q_post.detach()
            if k_post is not None:
                self._buf_k_post[layer_idx] = k_post.detach()

        return hook

    def capture_step(self) -> AttentionStats:
        """Drain the per-step buffers into an `AttentionStats` payload.

        Buffers are cleared after every call so each step yields fresh data.
        """

        num_q = self._num_attention_heads
        num_kv = self._num_key_value_heads
        d = self._head_dim
        n_rep = num_q // num_kv if num_kv else 1
        head_to_kv_group = [h // n_rep for h in range(num_q)]

        layer_stats: dict[int, AttentionLayerStats] = {}
        for layer_idx in self._target_layers:
            layer_stats[layer_idx] = self._build_layer_stats(layer_idx, num_q, num_kv, d)

        self._clear_buffers()
        return AttentionStats(
            layers=layer_stats,
            num_attention_heads=num_q,
            num_key_value_heads=num_kv,
            head_dim=d,
            head_to_kv_group=head_to_kv_group,
        )

    def _build_layer_stats(
        self, layer_idx: int, num_q: int, num_kv: int, d: int
    ) -> AttentionLayerStats:
        attn_weights = self._buf_attn_weights.get(layer_idx)
        weights_payload = self._format_attention_weights(attn_weights, num_q)

        q_pre = self._slice_last_position(self._buf_q_pre.get(layer_idx), num_q, d)
        k_pre = self._slice_last_position(self._buf_k_pre.get(layer_idx), num_kv, d)
        v_pre = self._slice_last_position(self._buf_v_pre.get(layer_idx), num_kv, d)
        q_post = self._slice_last_position(self._buf_q_post.get(layer_idx), num_q, d)
        k_post = self._slice_last_position(self._buf_k_post.get(layer_idx), num_kv, d)

        if not self.config.capture_qkv:
            return AttentionLayerStats(layer_idx=layer_idx, attention_weights=weights_payload)

        if self.config.capture_pre_rope_qk and q_post is not None and k_post is not None:
            return AttentionLayerStats(
                layer_idx=layer_idx,
                attention_weights=weights_payload,
                q_last=q_post,
                k_last=k_post,
                v_last=v_pre,
                q_last_pre_rope=q_pre,
                k_last_pre_rope=k_pre,
            )

        primary_q = q_post if q_post is not None else q_pre
        primary_k = k_post if k_post is not None else k_pre
        return AttentionLayerStats(
            layer_idx=layer_idx,
            attention_weights=weights_payload,
            q_last=primary_q,
            k_last=primary_k,
            v_last=v_pre,
        )

    def _format_attention_weights(
        self, attn_weights: torch.Tensor | None, num_q: int
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if attn_weights is None:
            if self.config.capture_full_distribution:
                return torch.zeros(num_q, 0)
            return {
                "positions": torch.zeros(num_q, 0, dtype=torch.long),
                "weights": torch.zeros(num_q, 0),
            }

        # Eager attention: [batch, H_q, q_seq, k_seq] -> last query position [H, S]
        last_row = attn_weights[0, :, -1, :].detach().cpu()
        if self.config.capture_full_distribution:
            return last_row

        k = max(1, min(int(self.config.top_k_positions), last_row.shape[-1]))
        top_w, top_p = torch.topk(last_row, k=k, dim=-1)
        return {"positions": top_p, "weights": top_w}

    @staticmethod
    def _slice_last_position(
        tensor: torch.Tensor | None, num_heads: int, head_dim: int
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim == 3:
            # Linear hook output: [batch, seq, num_heads * head_dim]
            return tensor[0, -1].reshape(num_heads, head_dim).detach().cpu()
        if tensor.ndim == 4:
            # Pre-reshaped (e.g., post-RoPE) [batch, num_heads, seq, head_dim]
            return tensor[0, :, -1, :].detach().cpu()
        if tensor.ndim == 2:
            # Already [num_heads, head_dim]
            return tensor.detach().cpu()
        raise AttentionProbeError(
            f"Unexpected captured tensor rank {tensor.ndim} for last-position slice."
        )
