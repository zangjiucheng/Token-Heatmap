"""Classic logit-lens analysis via HuggingFace forward hooks.

`LogitLens` is a sibling to `AttentionProbe`: opt-in, architecture-aware, hook-
based. For each generation step it captures the residual-stream hidden state at
the last position from every selected decoder layer, then projects each through
(optionally) the model's final `LayerNorm` and the tied `lm_head` to expose
what each layer "thinks" the next token should be.

Architecture handling is duck-typed:

- Llama-style (Qwen, Mistral, …): final norm at ``model.model.norm``,
  output head at ``model.lm_head``.
- GPT-style: final norm at ``model.transformer.ln_f``, output head at
  ``model.lm_head``.

The probe never mutates hidden states; hooks observe outputs only. Token-for-
token generation is therefore identical with or without the probe attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitLensError(RuntimeError):
    """Raised when `LogitLens` cannot operate on the given model."""


@dataclass
class LogitLensConfig:
    """Configuration for `LogitLens`.

    Attributes:
        layers: Either ``"all"`` (project every decoder layer) or a list of
            layer indices to project. Out-of-range indices raise on `attach`.
        top_k: Number of top-probability tokens retained per layer.
        apply_final_layernorm: When True, apply the model's final LayerNorm
            before projection (standard logit lens). When False, skip the
            normalization (raw lens, useful for early-layer analysis).
    """

    layers: str | list[int] = "all"
    top_k: int = 8
    apply_final_layernorm: bool = True


@dataclass
class LogitLensLayerStats:
    """Per-layer logit-lens captures for a single generation step."""

    layer_idx: int
    top_k_token_ids: torch.Tensor
    top_k_probs: torch.Tensor
    top_k_logprobs: torch.Tensor
    entropy: float
    selected_token_rank: int
    selected_token_prob: float


@dataclass
class LogitLensStats:
    """All per-layer logit-lens captures for a single generation step."""

    layers: dict[int, LogitLensLayerStats]


def _resolve_decoder_layers(model: Any) -> list[nn.Module]:
    """Locate the decoder layer list on a HuggingFace-style causal LM."""

    candidates: list[Any] = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        candidates.append(transformer)
    candidates.append(model)

    for parent in candidates:
        layers = getattr(parent, "layers", None)
        if layers is None:
            layers = getattr(parent, "h", None)
        if layers is None:
            continue
        try:
            return list(layers)
        except TypeError:
            continue
    return []


def _resolve_final_norm(model: Any) -> nn.Module | None:
    """Locate the model's final LayerNorm/RMSNorm.

    Returns ``None`` if no recognizable final-norm module is found. Callers
    that require the norm (``apply_final_layernorm=True``) should treat
    ``None`` as a hard error.
    """

    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("norm", "final_layernorm", "ln_f"):
            module = getattr(inner, attr, None)
            if isinstance(module, nn.Module):
                return module

    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for attr in ("ln_f", "norm", "final_layernorm"):
            module = getattr(transformer, attr, None)
            if isinstance(module, nn.Module):
                return module

    for attr in ("norm", "ln_f", "final_layernorm"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            return module
    return None


def _resolve_lm_head(model: Any) -> nn.Module | None:
    """Locate the model's output projection head."""

    for attr in ("lm_head", "embed_out", "output_projection"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            return module
    return None


class LogitLens(nn.Module):
    """Capture per-layer next-token distributions via forward hooks.

    The probe is opt-in: until `attach` is called, importing or instantiating
    it has no effect on generation. `attach` is idempotent, as is `detach`.

    Example:
        >>> lens = LogitLens(LogitLensConfig(layers="all", top_k=5))
        >>> lens.attach(model)
        >>> model(input_ids)
        >>> stats = lens.capture_step(selected_token_id=next_token)
        >>> lens.detach()
    """

    def __init__(self, config: LogitLensConfig | None = None) -> None:
        super().__init__()
        self.config = config or LogitLensConfig()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._model: Any = None
        self._target_layers: list[int] = []
        self._final_norm: nn.Module | None = None
        self._lm_head: nn.Module | None = None
        self._buf_hidden: dict[int, torch.Tensor] = {}

    @property
    def is_attached(self) -> bool:
        return self._model is not None

    @property
    def target_layers(self) -> list[int]:
        return list(self._target_layers)

    def attach(self, model: Any) -> LogitLens:
        """Install forward hooks on each selected decoder layer.

        Resolves the model's final LayerNorm and `lm_head` up front. Raises
        `LogitLensError` with a clear message if the architecture cannot be
        recognized. Calling `attach` while already attached is a no-op.
        """

        if self.is_attached:
            return self

        decoder_layers = _resolve_decoder_layers(model)
        if not decoder_layers:
            raise LogitLensError(
                "Could not locate a decoder layer list on the model. Expected "
                "`model.model.layers`, `model.transformer.h`, or `model.layers` "
                "(HuggingFace causal-LM convention)."
            )

        lm_head = _resolve_lm_head(model)
        if lm_head is None:
            raise LogitLensError(
                "Could not locate an `lm_head` (or `embed_out`) projection on "
                "the model. LogitLens requires an explicit output head module."
            )

        final_norm = _resolve_final_norm(model)
        if self.config.apply_final_layernorm and final_norm is None:
            raise LogitLensError(
                "apply_final_layernorm=True but no final LayerNorm/RMSNorm was "
                "found. Expected `model.model.norm` (Llama-style) or "
                "`model.transformer.ln_f` (GPT-style). Pass "
                "`apply_final_layernorm=False` for a raw lens if your "
                "architecture has no recognizable final norm."
            )

        if isinstance(self.config.layers, str):
            if self.config.layers != "all":
                raise LogitLensError(
                    f"config.layers must be 'all' or a list of ints; got {self.config.layers!r}."
                )
            target = list(range(len(decoder_layers)))
        elif isinstance(self.config.layers, list):
            target = sorted({int(i) for i in self.config.layers})
            for idx in target:
                if idx < 0 or idx >= len(decoder_layers):
                    raise LogitLensError(
                        f"layer index {idx} out of range for model with "
                        f"{len(decoder_layers)} layers."
                    )
        else:
            raise LogitLensError(
                "config.layers must be 'all' or a list of ints; "
                f"got {type(self.config.layers).__name__}."
            )

        if self.config.top_k < 1:
            raise LogitLensError(f"top_k must be >= 1; got {self.config.top_k}.")

        self._target_layers = target
        self._final_norm = final_norm
        self._lm_head = lm_head

        for layer_idx in target:
            layer = decoder_layers[layer_idx]
            self._handles.append(layer.register_forward_hook(self._make_layer_hook(layer_idx)))

        self._model = model
        return self

    def detach(self) -> LogitLens:
        """Remove hooks and clear per-step buffers."""

        if not self.is_attached:
            return self

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        self._model = None
        self._target_layers = []
        self._final_norm = None
        self._lm_head = None
        self._buf_hidden.clear()
        return self

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return
            # hidden: [batch, seq, hidden_size] -> last position
            self._buf_hidden[layer_idx] = hidden[:, -1, :].detach()

        return hook

    @torch.no_grad()
    def capture_step(self, selected_token_id: int | torch.Tensor | None = None) -> LogitLensStats:
        """Project captured hidden states through the output head per layer.

        Args:
            selected_token_id: The token actually selected at this step. If
                provided, `selected_token_rank` and `selected_token_prob` are
                populated; otherwise they default to ``0`` and ``0.0``.

        Returns:
            `LogitLensStats` containing per-layer next-token distributions.
            Per-step buffers are cleared after each call.
        """

        if self._lm_head is None:
            raise LogitLensError("capture_step called while detached.")

        selected_id: int | None = None
        if selected_token_id is not None:
            if isinstance(selected_token_id, torch.Tensor):
                selected_id = int(selected_token_id.flatten()[0])
            else:
                selected_id = int(selected_token_id)

        layer_stats: dict[int, LogitLensLayerStats] = {}
        for layer_idx in self._target_layers:
            hidden = self._buf_hidden.get(layer_idx)
            if hidden is None:
                continue
            layer_stats[layer_idx] = self._project_layer(layer_idx, hidden, selected_id)

        self._buf_hidden.clear()
        return LogitLensStats(layers=layer_stats)

    def _project_layer(
        self, layer_idx: int, hidden: torch.Tensor, selected_id: int | None
    ) -> LogitLensLayerStats:
        x = hidden
        if self.config.apply_final_layernorm and self._final_norm is not None:
            x = self._final_norm(x)
        # lm_head accepts [batch, hidden_size] -> [batch, vocab_size]
        logits = self._lm_head(x)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        k = min(int(self.config.top_k), int(probs.shape[-1]))
        top_probs, top_ids = torch.topk(probs[0], k=k, dim=-1, largest=True, sorted=True)
        top_logprobs = log_probs[0].gather(0, top_ids)

        plogp = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
        entropy = float((-plogp.sum(dim=-1))[0].item())

        if selected_id is not None:
            sel_prob = float(probs[0, selected_id].item())
            sel_rank = int((probs[0] > probs[0, selected_id]).sum().item()) + 1
        else:
            sel_prob = 0.0
            sel_rank = 0

        return LogitLensLayerStats(
            layer_idx=layer_idx,
            top_k_token_ids=top_ids.detach().cpu(),
            top_k_probs=top_probs.detach().cpu(),
            top_k_logprobs=top_logprobs.detach().cpu(),
            entropy=entropy,
            selected_token_rank=sel_rank,
            selected_token_prob=sel_prob,
        )
