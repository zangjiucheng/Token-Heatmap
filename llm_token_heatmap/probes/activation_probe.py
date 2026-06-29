"""Activation capture via HuggingFace forward hooks.

`ActivationProbe` is a sibling to `AttentionProbe` and `LogitLens`: opt-in,
architecture-aware, hook-based. It owns one forward hook per configured
``(layer, submodule)`` pair and drains every capture into a per-step list of
summary statistics whose field names match the ``ActivationLayerEntry``
definition in ``docs/web/activation.schema.json``.

Supported submodule keys (aliases collapse to a single canonical name):

- ``resid_pre`` / ``residual_pre`` — input to each decoder layer (pre-hook).
- ``resid_post`` / ``residual_post`` — output of each decoder layer.
- ``mlp_out`` / ``mlp.down_proj`` — output of ``layer.mlp.down_proj`` when
  present, falling back to ``layer.mlp`` for compact architectures.
- ``o_proj`` — output of ``layer.self_attn.o_proj``.

Two capture modes are exposed:

- :meth:`capture_step` drains the per-step buffer and reduces every captured
  tensor to its last position. The :func:`generate_with_adaptive_probe` loop
  calls this once per generated token.
- :meth:`capture_along_sequence` runs a single forward pass on a fixed
  ``input_ids`` tensor and returns one record per token position along the
  sequence. This is the "force-prefix" path that lets two probes attached to
  different models capture honest, position-aligned diffs without any
  sampling differences.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Maps every user-visible submodule alias to the canonical name the probe
# emits in `ActivationLayerEntry.submodule`. Canonical names are also used as
# the metadata key set so they must be stable.
_SUBMODULE_ALIASES: dict[str, str] = {
    "resid_pre": "resid_pre",
    "residual_pre": "resid_pre",
    "resid_post": "resid_post",
    "residual_post": "resid_post",
    "mlp_out": "mlp_out",
    "mlp.down_proj": "mlp_out",
    "o_proj": "o_proj",
}


class ActivationProbeError(RuntimeError):
    """Raised when `ActivationProbe` cannot operate on the given model."""


@dataclass
class ActivationProbeConfig:
    """Configuration for `ActivationProbe`.

    Attributes:
        layers: Either ``"all"`` (hook every decoder layer) or a list of layer
            indices to hook. Out-of-range indices raise on `attach`.
        submodules: Submodule keys to capture per layer; see the module
            docstring for the supported set. Duplicate aliases collapse to a
            single canonical entry. Must be non-empty.
        top_k: Number of highest-magnitude neuron indices retained per
            ``(layer, submodule)`` in each captured entry's ``top_neurons``
            list. Clipped to ``hidden_dim``.
        sparsity_threshold: Absolute-value threshold below which a neuron is
            counted as "near zero" for the ``sparsity`` summary stat.
        capture_full: When True, retain the raw per-``(layer, submodule)``
            activation tensors alongside the summary stats so the Tier 2
            sidecar serializer (``activation_serializer.write_sidecar``) can
            persist them. Off by default — adds memory + copy overhead, only
            opt in when ``--capture-full-activations`` is set on the CLI.
            Retained tensors are accessible via ``last_full_stats`` after
            ``capture_step`` and ``full_stats_per_position`` after
            ``capture_along_sequence``.
    """

    layers: str | list[int] = "all"
    submodules: list[str] = field(default_factory=lambda: ["resid_post", "mlp_out"])
    top_k: int = 8
    sparsity_threshold: float = 1e-6
    capture_full: bool = False


@dataclass
class TopNeuron:
    """One entry in `ActivationLayerEntry.top_neurons` — matches the schema."""

    index: int
    value: float


@dataclass
class ActivationLayerEntry:
    """Summary stats for a single ``(step|position, layer, submodule)`` capture.

    Field names mirror the ``ActivationLayerEntry`` definition in
    ``docs/web/activation.schema.json`` so ``dataclasses.asdict(entry)`` is
    schema-shaped on the wire.
    """

    layer: int
    submodule: str
    l2_norm: float
    mean_abs: float
    sparsity: float
    top_neurons: list[TopNeuron]


@dataclass
class ActivationFullStats:
    """Raw per-``(layer, submodule)`` activation tensors retained when
    ``ActivationProbeConfig.capture_full`` is True.

    Consumed by :func:`llm_token_heatmap.serialize.activation_serializer.write_sidecar`
    to produce the Tier 2 ``.npz`` sidecar. Each tensor is a 1-D vector along
    the hidden dimension corresponding to a single token position (the last
    position after ``capture_step``, or one particular position after
    ``capture_along_sequence``).
    """

    layer_tensors: dict[tuple[int, str], torch.Tensor]
    num_layers: int
    hidden_dim: int
    captured_layers: list[int]
    captured_submodules: list[str]
    # Per-layer o_proj *input* (concatenated per-head attention outputs `z`,
    # shape [num_heads*head_dim]) at the last position. Captured alongside the
    # o_proj output when full-capturing, and used for per-head DLA. Empty when
    # o_proj wasn't among the captured submodules.
    attn_z: dict[int, torch.Tensor] = field(default_factory=dict)


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


def _resolve_submodule_target(layer: nn.Module, canonical: str) -> nn.Module | None:
    """Return the module to hook for ``canonical`` on a single decoder layer.

    Residual keys hook the layer module itself (pre-hook for ``resid_pre``,
    post-hook for ``resid_post``). MLP and o_proj keys descend into the
    layer's submodule tree.
    """

    if canonical in ("resid_pre", "resid_post"):
        return layer
    if canonical == "mlp_out":
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None
        down_proj = getattr(mlp, "down_proj", None)
        return down_proj if down_proj is not None else mlp
    if canonical == "o_proj":
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return None
        return getattr(attn, "o_proj", None)
    return None


class ActivationProbe(nn.Module):
    """Capture per-layer activations via forward hooks.

    The probe is opt-in: until `attach` is called, importing or instantiating
    it has no effect on generation. Both `attach` and `detach` are idempotent.

    Example:
        >>> probe = ActivationProbe(
        ...     ActivationProbeConfig(layers="all", submodules=["resid_post"])
        ... )
        >>> probe.attach(model)
        >>> model(input_ids)
        >>> entries = probe.capture_step()
        >>> probe.detach()
    """

    def __init__(self, config: ActivationProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActivationProbeConfig()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._model: Any = None
        self._target_layers: list[int] = []
        self._submodule_keys: list[str] = []
        self._buffers: dict[tuple[int, str], torch.Tensor] = {}
        self._num_layers: int = 0
        self._hidden_dim: int = 0
        self._last_full_tensors: dict[tuple[int, str], torch.Tensor] = {}
        self._full_tensors_per_position: list[dict[tuple[int, str], torch.Tensor]] = []
        # o_proj input (`z`) buffers for per-head DLA — captured only under
        # full-capture, keyed by layer index.
        self._attn_z_buffers: dict[int, torch.Tensor] = {}
        self._last_attn_z: dict[int, torch.Tensor] = {}

    @property
    def is_attached(self) -> bool:
        return self._model is not None

    @property
    def target_layers(self) -> list[int]:
        return list(self._target_layers)

    @property
    def submodule_keys(self) -> list[str]:
        return list(self._submodule_keys)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def attach(self, model: Any) -> ActivationProbe:
        """Install one forward hook per ``(layer, submodule)`` pair.

        Resolves the decoder layer list, normalizes submodule aliases, and
        validates every target submodule exists before installing any hooks
        so a failure leaves the model untouched. Calling `attach` while
        already attached is a no-op.
        """

        if self.is_attached:
            return self

        decoder_layers = _resolve_decoder_layers(model)
        if not decoder_layers:
            raise ActivationProbeError(
                "Could not locate a decoder layer list on the model. Expected "
                "`model.model.layers`, `model.transformer.h`, or `model.layers` "
                "(HuggingFace causal-LM convention)."
            )

        model_config = getattr(model, "config", None)
        hidden_size = getattr(model_config, "hidden_size", None) if model_config else None
        if hidden_size is None:
            raise ActivationProbeError(
                "Model config must expose `hidden_size`; activation summary "
                "stats are reduced along the hidden dimension."
            )

        if isinstance(self.config.layers, str):
            if self.config.layers != "all":
                raise ActivationProbeError(
                    f"config.layers must be 'all' or a list of ints; "
                    f"got {self.config.layers!r}."
                )
            target = list(range(len(decoder_layers)))
        elif isinstance(self.config.layers, list):
            target = sorted({int(i) for i in self.config.layers})
            for idx in target:
                if idx < 0 or idx >= len(decoder_layers):
                    raise ActivationProbeError(
                        f"layer index {idx} out of range for model with "
                        f"{len(decoder_layers)} layers."
                    )
        else:
            raise ActivationProbeError(
                "config.layers must be 'all' or a list of ints; got "
                f"{type(self.config.layers).__name__}."
            )

        canonical_keys: list[str] = []
        seen: set[str] = set()
        for raw in self.config.submodules:
            if raw not in _SUBMODULE_ALIASES:
                raise ActivationProbeError(
                    f"Unsupported submodule key: {raw!r}. Supported keys: "
                    f"{sorted(_SUBMODULE_ALIASES)}"
                )
            canonical = _SUBMODULE_ALIASES[raw]
            if canonical in seen:
                continue
            seen.add(canonical)
            canonical_keys.append(canonical)

        if not canonical_keys:
            raise ActivationProbeError(
                "config.submodules must contain at least one submodule key."
            )

        resolved: list[tuple[int, str, nn.Module]] = []
        for layer_idx in target:
            for canonical in canonical_keys:
                module = _resolve_submodule_target(decoder_layers[layer_idx], canonical)
                if module is None:
                    raise ActivationProbeError(
                        f"Layer {layer_idx} does not expose submodule "
                        f"{canonical!r}; model architecture is unsupported."
                    )
                resolved.append((layer_idx, canonical, module))

        for layer_idx, canonical, module in resolved:
            if canonical == "resid_pre":
                handle = module.register_forward_pre_hook(
                    self._make_pre_hook(layer_idx, canonical)
                )
            else:
                handle = module.register_forward_hook(
                    self._make_hook(layer_idx, canonical)
                )
            self._handles.append(handle)

        self._model = model
        self._target_layers = target
        self._submodule_keys = canonical_keys
        self._num_layers = len(decoder_layers)
        self._hidden_dim = int(hidden_size)
        return self

    def detach(self) -> ActivationProbe:
        """Remove every hook and return the probe to idle state."""

        if not self.is_attached:
            return self

        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._buffers.clear()
        self._last_full_tensors.clear()
        self._full_tensors_per_position.clear()
        self._model = None
        self._target_layers = []
        self._submodule_keys = []
        return self

    def _make_pre_hook(self, layer_idx: int, canonical: str):
        key = (layer_idx, canonical)

        def hook(_module: nn.Module, inputs: tuple) -> None:
            if not inputs:
                return
            tensor = inputs[0]
            if not torch.is_tensor(tensor):
                return
            self._buffers[key] = tensor.detach()

        return hook

    def _make_hook(self, layer_idx: int, canonical: str):
        key = (layer_idx, canonical)

        def hook(_module: nn.Module, inputs: Any, output: Any) -> None:
            tensor: torch.Tensor | None = None
            if torch.is_tensor(output):
                tensor = output
            elif isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                tensor = output[0]
            if tensor is not None:
                self._buffers[key] = tensor.detach()
            # The o_proj input is the concatenated per-head attention output `z`
            # (pre-W_O), which per-head DLA needs. Capture it only under
            # full-capture so the default summary path is unaffected.
            if canonical == "o_proj" and self.config.capture_full and inputs:
                z = inputs[0] if isinstance(inputs, tuple) else inputs
                if torch.is_tensor(z):
                    self._attn_z_buffers[layer_idx] = z.detach()

        return hook

    def _reduce_vector(
        self, vec: torch.Tensor, layer_idx: int, canonical: str
    ) -> ActivationLayerEntry:
        vec = vec.detach().to(torch.float32).cpu().flatten()
        n = vec.numel()
        l2 = float(torch.linalg.vector_norm(vec).item()) if n else 0.0
        mean_abs = float(vec.abs().mean().item()) if n else 0.0
        if n:
            sparsity = float(
                (vec.abs() < self.config.sparsity_threshold)
                .to(torch.float32)
                .mean()
                .item()
            )
        else:
            sparsity = 1.0

        k = max(0, min(int(self.config.top_k), n))
        top_neurons: list[TopNeuron] = []
        if k > 0:
            _, top_idx = torch.topk(vec.abs(), k=k)
            for idx in top_idx.tolist():
                top_neurons.append(TopNeuron(index=int(idx), value=float(vec[idx].item())))
        return ActivationLayerEntry(
            layer=int(layer_idx),
            submodule=canonical,
            l2_norm=l2,
            mean_abs=mean_abs,
            sparsity=sparsity,
            top_neurons=top_neurons,
        )

    @staticmethod
    def _slice_position(tensor: torch.Tensor, position: int) -> torch.Tensor:
        if tensor.ndim >= 3:
            return tensor[0, position]
        if tensor.ndim == 2:
            return tensor[position]
        return tensor

    def capture_step(self) -> list[ActivationLayerEntry]:
        """Drain per-step buffers into one entry per ``(layer, submodule)``.

        Each captured tensor is reduced to its **last** position before
        summary stats are computed (the new token under KV-cache generation
        is always at index ``-1``). Buffers are cleared after every call so
        the next forward pass starts fresh.

        When ``config.capture_full`` is True, the raw last-position vector is
        also stashed in :attr:`last_full_stats` so the Tier 2 sidecar
        serializer can persist it.
        """

        if self.config.capture_full:
            self._last_full_tensors.clear()
            self._last_attn_z.clear()
        entries: list[ActivationLayerEntry] = []
        for layer_idx in self._target_layers:
            for canonical in self._submodule_keys:
                key = (layer_idx, canonical)
                tensor = self._buffers.get(key)
                if tensor is None:
                    raise ActivationProbeError(
                        f"No captured activation for layer {layer_idx} "
                        f"submodule {canonical!r}. Did the forward pass run "
                        "after the probe was attached?"
                    )
                vec = self._slice_position(tensor, -1)
                if self.config.capture_full:
                    self._last_full_tensors[key] = (
                        vec.detach().to(torch.float32).cpu().flatten().clone()
                    )
                entries.append(self._reduce_vector(vec, layer_idx, canonical))
        if self.config.capture_full and self._attn_z_buffers:
            for layer_idx, ztensor in self._attn_z_buffers.items():
                zvec = self._slice_position(ztensor, -1)
                self._last_attn_z[int(layer_idx)] = (
                    zvec.detach().to(torch.float32).cpu().flatten().clone()
                )
        self._attn_z_buffers.clear()
        self._buffers.clear()
        return entries

    @torch.no_grad()
    def capture_along_sequence(
        self, model: Any, input_ids: torch.Tensor
    ) -> list[list[ActivationLayerEntry]]:
        """Run a single forward pass and capture activations per token position.

        The model is invoked once with the fixed ``input_ids`` — no sampling,
        no KV-cache reuse — and the hooks fire on the full sequence. The
        return value is a list of length ``input_ids.shape[-1]``; entry
        ``i`` is the per-``(layer, submodule)`` records for token ``i``.

        This is the "force-prefix" path: two probes attached to different
        models can capture honest position-aligned diffs along an identical
        token sequence without any per-step sampling divergence.
        """

        if not self.is_attached:
            raise ActivationProbeError(
                "capture_along_sequence requires the probe to be attached first."
            )

        self._buffers.clear()
        if self.config.capture_full:
            self._full_tensors_per_position.clear()
        model(input_ids)

        seq_len = int(input_ids.shape[-1])
        per_position: list[list[ActivationLayerEntry]] = []
        for pos in range(seq_len):
            position_entries: list[ActivationLayerEntry] = []
            position_tensors: dict[tuple[int, str], torch.Tensor] = {}
            for layer_idx in self._target_layers:
                for canonical in self._submodule_keys:
                    key = (layer_idx, canonical)
                    tensor = self._buffers.get(key)
                    if tensor is None:
                        raise ActivationProbeError(
                            f"No captured activation for layer {layer_idx} "
                            f"submodule {canonical!r}."
                        )
                    vec = self._slice_position(tensor, pos)
                    if self.config.capture_full:
                        position_tensors[key] = (
                            vec.detach().to(torch.float32).cpu().flatten().clone()
                        )
                    position_entries.append(self._reduce_vector(vec, layer_idx, canonical))
            per_position.append(position_entries)
            if self.config.capture_full:
                self._full_tensors_per_position.append(position_tensors)
        self._buffers.clear()
        return per_position

    @property
    def last_full_stats(self) -> ActivationFullStats | None:
        """Raw last-position tensors captured by the most recent
        :meth:`capture_step`. ``None`` when ``config.capture_full`` is False or
        no capture has happened yet.
        """

        if not self.config.capture_full or not self._last_full_tensors:
            return None
        captured_layers = sorted({layer for layer, _ in self._last_full_tensors})
        return ActivationFullStats(
            layer_tensors=dict(self._last_full_tensors),
            num_layers=self._num_layers,
            hidden_dim=self._hidden_dim,
            captured_layers=captured_layers,
            captured_submodules=list(self._submodule_keys),
            attn_z=dict(self._last_attn_z),
        )

    @property
    def full_stats_per_position(self) -> list[ActivationFullStats]:
        """Raw per-position tensors captured by the most recent
        :meth:`capture_along_sequence`. Empty list when
        ``config.capture_full`` is False or no capture has happened yet.
        """

        if not self.config.capture_full or not self._full_tensors_per_position:
            return []
        out: list[ActivationFullStats] = []
        for layer_tensors in self._full_tensors_per_position:
            captured_layers = sorted({layer for layer, _ in layer_tensors})
            out.append(
                ActivationFullStats(
                    layer_tensors=dict(layer_tensors),
                    num_layers=self._num_layers,
                    hidden_dim=self._hidden_dim,
                    captured_layers=captured_layers,
                    captured_submodules=list(self._submodule_keys),
                )
            )
        return out


def entry_to_dict(entry: ActivationLayerEntry) -> dict[str, Any]:
    """Convert an `ActivationLayerEntry` to a schema-shaped dict."""

    return asdict(entry)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Compute a stable SHA-256 fingerprint of a HuggingFace tokenizer.

    The fingerprint hashes the tokenizer's class name plus the canonical
    sorted ``(token_string, token_id)`` pairs returned by
    ``tokenizer.get_vocab()``. BPE merges (when exposed via
    ``backend_tokenizer.model``) are folded in as a secondary digest so two
    BPE tokenizers that happen to agree on vocab but disagree on merge
    priority still produce different fingerprints. The result is
    deterministic across processes and stable across CPU / GPU loads —
    `compare_activations` relies on equal fingerprints to safely zip two
    traces by ``token_id`` instead of falling back to position alignment.
    """

    hasher = hashlib.sha256()
    class_name = type(tokenizer).__name__
    hasher.update(class_name.encode("utf-8"))
    hasher.update(b"\0")

    vocab = getattr(tokenizer, "get_vocab", None)
    vocab_dict: dict[str, int] = vocab() if callable(vocab) else {}
    sorted_items = sorted(vocab_dict.items(), key=lambda kv: (kv[1], kv[0]))
    hasher.update(json.dumps(sorted_items, ensure_ascii=False).encode("utf-8"))
    hasher.update(b"\0")

    merges_repr: str = ""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None) if backend is not None else None
    if model is not None:
        for attr in ("merges", "get_merges"):
            candidate = getattr(model, attr, None)
            if candidate is None:
                continue
            try:
                merges = candidate() if callable(candidate) else candidate
                merges_repr = json.dumps(list(merges), ensure_ascii=False)
                break
            except Exception:  # noqa: BLE001 — merges are best-effort.
                continue
    hasher.update(merges_repr.encode("utf-8"))

    return f"sha256:{hasher.hexdigest()}"
