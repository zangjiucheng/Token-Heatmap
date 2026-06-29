"""Serialize `AttentionStats` into trace-schema-shaped payloads.

This module is the plumbing layer between the in-memory ``AttentionStats``
emitted by :class:`llm_token_heatmap.AttentionProbe` and the JSON trace schema
defined in ``docs/web/trace.schema.json``.

Two payloads are produced:

* **Tier 1 (inline)** -- :func:`attention_stats_to_payload` returns a small
  per-layer dict carrying head-aggregated scalars (entropy, self/BOS weight,
  Q/K/V norms, the QK alignment angle, plus a sparse top-positions list).
  This rides inline in the main trace JSON and is cheap enough to load eagerly.
* **Tier 2 (sidecar)** -- :func:`write_sidecar` writes a ``.npz`` archive
  containing the full attention distribution and the raw Q/K/V tensors so the
  expensive data only loads on demand. :func:`read_sidecar` is the inverse
  helper used by tests and downstream readers.

The shape of each payload mirrors the JSON Schema definitions
``AttentionMetadata`` / ``AttentionLayerEntry`` (trace) and the
``attention-sidecar`` schema (sidecar). The serializer does **not** compute the
statistics themselves -- that lives in a downstream ticket. What it computes
here is the minimal head aggregation needed for the inline summary; raw
distributions and Q/K/V tensors are passed through to the sidecar unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_token_heatmap.analysis.attention_stats import (
    AttentionLayerDerivedStats,
    compute_attention_stats,
)
from llm_token_heatmap.probes.attention_probe import AttentionLayerStats, AttentionStats

SIDECAR_SCHEMA_VERSION = "2.0.0"


def attention_stats_to_payload(
    stats: AttentionStats,
    *,
    capture_full: bool = False,
    top_k_positions: int = 8,
    token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Build the Tier 1 inline trace payload from an :class:`AttentionStats`.

    Args:
        stats: Per-step captures from :meth:`AttentionProbe.capture_step`.
        capture_full: When True, ``stats.layers[*].attention_weights`` is a
            dense ``[H, S]`` tensor; when False it is the sparse
            ``{positions, weights}`` dict the probe emits in top-k mode.
        top_k_positions: How many aggregate top positions to keep in the inline
            summary. Independent of the probe's own ``top_k_positions``.
        token_ids: The full token-id sequence at this step (prompt + generated
            so far), last element being the current query token. When given,
            each head gets a per-head **induction score** — the attention it
            places on the token that followed the current token's most recent
            earlier occurrence (the textbook induction-head signature). Most
            accurate under full-distribution capture; in sparse (top-k) mode it
            only registers when the induction target made the head's top-k.

    Returns:
        A dict with two keys:

        * ``attention_metadata`` -- matches the schema ``AttentionMetadata``.
        * ``attention`` -- list of per-layer entries matching
          ``AttentionLayerEntry``, sorted by ``layer`` ascending.
    """

    induction_target = _induction_target_position(token_ids)
    captured_layers = sorted(stats.layers.keys())
    metadata = {
        "num_layers": len(captured_layers) if not captured_layers else max(captured_layers) + 1,
        "num_attention_heads": int(stats.num_attention_heads),
        "num_key_value_heads": int(stats.num_key_value_heads),
        "head_dim": int(stats.head_dim),
        "captured_layers": [int(i) for i in captured_layers],
    }

    derived = compute_attention_stats(stats, top_k=top_k_positions)
    layer_entries: list[dict[str, Any]] = []
    for layer_idx in captured_layers:
        layer_entries.append(
            _layer_stats_to_entry(
                stats.layers[layer_idx],
                derived.layers[layer_idx],
                capture_full=capture_full,
                top_k_positions=top_k_positions,
                induction_target=induction_target,
            )
        )

    return {
        "attention_metadata": metadata,
        "attention": layer_entries,
    }


def _layer_stats_to_entry(
    layer: AttentionLayerStats,
    derived: AttentionLayerDerivedStats,
    *,
    capture_full: bool,
    top_k_positions: int,
    induction_target: int | None = None,
) -> dict[str, Any]:
    weights_dense = _to_dense_weights(layer.attention_weights, capture_full)
    top_positions = _aggregate_top_positions(weights_dense, top_k_positions)

    heads = derived.heads
    n = max(1, len(heads))
    # Per-head scalars, stored COLUMNAR (parallel arrays keyed by metric, ordered
    # by head index) instead of a list-of-dicts: the repeated JSON key names
    # otherwise dominate the file (28 heads x 28 layers x N steps). The Layer x
    # Head grid reads these columns. q/k/v norms are intentionally dropped — they
    # measure activation magnitude, not function, and are anti-correlated with a
    # head's actual contribution (see head_roles); use DLA to rank importance.
    # top1_weight is the head's single largest source weight; induction is the
    # attention it puts on the induction target (token after the current token's
    # last occurrence) — high in induction heads.
    per_head = {
        "entropy": [float(h.entropy) for h in heads],
        "self_weight": [float(h.self_weight) for h in heads],
        "bos_weight": [float(h.bos_weight) for h in heads],
        "top1_weight": [float(max((w for _, w in h.top_k_positions), default=0.0)) for h in heads],
        "induction": [
            _head_induction(weights_dense, head_idx, induction_target)
            for head_idx in range(len(heads))
        ],
    }
    return {
        "layer": int(layer.layer_idx),
        "entropy": float(sum(h.entropy for h in heads) / n),
        "self_weight": float(sum(h.self_weight for h in heads) / n),
        "bos_weight": float(sum(h.bos_weight for h in heads) / n),
        "top_positions": top_positions,
        "per_head": per_head,
    }


def _induction_target_position(token_ids: list[int] | None) -> int | None:
    """Source position the textbook induction pattern points at for this query.

    The query is the sequence's last token. An induction head, having seen this
    token before at position ``j``, attends to ``j + 1`` — the token that
    followed it last time — and copies it. We return that ``j + 1`` for the most
    recent earlier occurrence, or ``None`` when the current token is novel (or
    only repeats immediately before the query, where the target collapses onto
    the query itself) so the induction score is undefined and reported as 0.
    """

    if not token_ids or len(token_ids) < 2:
        return None
    query_pos = len(token_ids) - 1
    current = token_ids[query_pos]
    for j in range(query_pos - 1, -1, -1):
        if token_ids[j] == current:
            target = j + 1
            return target if target != query_pos else None
    return None


def _head_induction(weights_dense: torch.Tensor, head_idx: int, target: int | None) -> float:
    """Attention this head places on the induction target position (0 when N/A)."""

    if target is None or head_idx >= weights_dense.shape[0]:
        return 0.0
    if target < 0 or target >= weights_dense.shape[-1]:
        return 0.0
    return float(weights_dense[head_idx, target].item())


def _to_dense_weights(
    weights: torch.Tensor | dict[str, torch.Tensor],
    capture_full: bool,
) -> torch.Tensor:
    """Return weights as a dense ``[H, S]`` tensor regardless of capture mode."""

    if isinstance(weights, torch.Tensor):
        return weights.detach().cpu().float()

    positions = weights["positions"]
    values = weights["weights"]
    if positions.numel() == 0:
        return torch.zeros(positions.shape[0], 0)

    num_heads = positions.shape[0]
    max_pos = int(positions.max().item()) + 1
    dense = torch.zeros(num_heads, max_pos)
    for h in range(num_heads):
        dense[h].scatter_(0, positions[h].long(), values[h].float())
    if capture_full:
        # Caller asserted full capture but probe handed back sparse -- still safe.
        return dense
    return dense


def _aggregate_top_positions(weights: torch.Tensor, k: int) -> list[dict[str, Any]]:
    if weights.numel() == 0 or weights.shape[-1] == 0 or k <= 0:
        return []
    head_mean = weights.mean(dim=0)
    effective_k = min(int(k), head_mean.shape[-1])
    top_w, top_idx = torch.topk(head_mean, k=effective_k)
    return [
        {"position": int(top_idx[i].item()), "weight": float(top_w[i].item())}
        for i in range(effective_k)
        if float(top_w[i].item()) > 0.0
    ]


def write_sidecar(
    stats: AttentionStats,
    path: str | Path,
    *,
    step: int,
) -> Path:
    """Write the Tier 2 sidecar archive for one (trace, step).

    The file format is ``numpy.savez_compressed``. Per-layer arrays live under
    keys ``layer_<idx>_attention_weights`` / ``_q_last`` / ``_k_last`` /
    ``_v_last`` (the latter three are absent when the probe didn't capture
    QKV); top-level keys hold the architecture metadata. :func:`read_sidecar`
    is the inverse helper.

    Returns:
        The :class:`Path` that was written (``.npz`` suffix added if missing).
    """

    out_path = Path(path)
    if out_path.suffix != ".npz":
        out_path = (
            out_path.with_suffix(out_path.suffix + ".npz")
            if out_path.suffix
            else out_path.with_suffix(".npz")
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    captured_layers = sorted(stats.layers.keys())
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.array(SIDECAR_SCHEMA_VERSION),
        "step": np.array(int(step), dtype=np.int64),
        "num_attention_heads": np.array(int(stats.num_attention_heads), dtype=np.int64),
        "num_key_value_heads": np.array(int(stats.num_key_value_heads), dtype=np.int64),
        "head_dim": np.array(int(stats.head_dim), dtype=np.int64),
        "captured_layers": np.array(captured_layers, dtype=np.int64),
    }

    for layer_idx in captured_layers:
        layer = stats.layers[layer_idx]
        weights = _to_dense_weights(layer.attention_weights, capture_full=True)
        # Cast to float32 *before* .numpy(): numpy has no bfloat16 dtype, so a
        # bf16 tensor (the default on modern bf16-native models) would raise
        # "unsupported ScalarType BFloat16". `.float()` upcasts bf16/fp16 → f32.
        arrays[f"layer_{layer_idx}_attention_weights"] = weights.detach().float().cpu().numpy()
        if layer.q_last is not None:
            arrays[f"layer_{layer_idx}_q_last"] = layer.q_last.detach().float().cpu().numpy()
        if layer.k_last is not None:
            arrays[f"layer_{layer_idx}_k_last"] = layer.k_last.detach().float().cpu().numpy()
        if layer.v_last is not None:
            arrays[f"layer_{layer_idx}_v_last"] = layer.v_last.detach().float().cpu().numpy()

    np.savez_compressed(out_path, **arrays)
    return out_path


def read_sidecar(path: str | Path) -> dict[str, Any]:
    """Inverse of :func:`write_sidecar`: load the sidecar into a JSON-shaped dict.

    The returned shape matches ``docs/web/attention-sidecar.schema.json``.
    Arrays are returned as nested Python lists so the result is directly
    JSON-serializable (and validatable against the sidecar schema).
    """

    with np.load(Path(path), allow_pickle=False) as data:
        captured_layers = [int(i) for i in data["captured_layers"].tolist()]
        layers: list[dict[str, Any]] = []
        for layer_idx in captured_layers:
            entry: dict[str, Any] = {
                "layer": int(layer_idx),
                "attention_weights": data[f"layer_{layer_idx}_attention_weights"].tolist(),
            }
            for field in ("q_last", "k_last", "v_last"):
                key = f"layer_{layer_idx}_{field}"
                entry[field] = data[key].tolist() if key in data.files else None
            layers.append(entry)

        return {
            "schema_version": str(data["schema_version"]),
            "step": int(data["step"]),
            "num_attention_heads": int(data["num_attention_heads"]),
            "num_key_value_heads": int(data["num_key_value_heads"]),
            "head_dim": int(data["head_dim"]),
            "layers": layers,
        }
