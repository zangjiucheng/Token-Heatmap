"""Per-(step, layer, head) attention statistics derived from ``AttentionStats``.

Pure functions over :class:`AttentionStats` payloads emitted by
:class:`AttentionProbe`. No model calls, no I/O. The output structure carries
two parallel views:

* Per-head detail used by the SPA (entropy, self/BOS weight, top-k positions,
  Q/K/V norms, QK alignment angle, effective span, and a 16-bin specialization
  fingerprint).
* Per-layer aggregates used by the inline trace summary (mean / max entropy,
  copy- and sink-head fractions).

All entropies are reported in **nats** (natural log). Effective span is
``2 ** (entropy / ln 2)`` -- equivalently ``exp(entropy_nats)`` -- so a head
that uniformly attends to ``n`` previous positions has span ``n``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from llm_token_heatmap.attention_probe import AttentionLayerStats, AttentionStats

DEFAULT_TOP_K = 8
DEFAULT_FINGERPRINT_BINS = 16
COPY_HEAD_THRESHOLD = 0.5
SINK_HEAD_THRESHOLD = 0.5


@dataclass
class AttentionHeadStats:
    """Per-head attention summary for a single (step, layer, head)."""

    entropy: float
    self_weight: float
    bos_weight: float
    top_k_positions: list[tuple[int, float]]
    q_norm: float
    k_norm: float
    v_norm: float
    qk_alignment_angle_deg: float
    effective_attention_span: float
    specialization_fingerprint: list[float]


@dataclass
class AttentionLayerAggregates:
    """Per-layer aggregates over the head dimension."""

    mean_entropy: float
    max_entropy: float
    copy_head_fraction: float
    sink_head_fraction: float


@dataclass
class AttentionLayerDerivedStats:
    """All per-head stats for one layer plus that layer's aggregates."""

    layer_idx: int
    heads: list[AttentionHeadStats]
    aggregates: AttentionLayerAggregates


@dataclass
class AttentionDerivedStats:
    """Top-level container returned by :func:`compute_attention_stats`."""

    layers: dict[int, AttentionLayerDerivedStats]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    top_k: int
    num_fingerprint_bins: int
    head_to_kv_group: list[int] = field(default_factory=list)


def compute_attention_stats(
    stats: AttentionStats,
    *,
    top_k: int = DEFAULT_TOP_K,
    fingerprint_bins: int = DEFAULT_FINGERPRINT_BINS,
) -> AttentionDerivedStats:
    """Compute derived per-head statistics from an :class:`AttentionStats`.

    Pure function: the input is not mutated and the output depends only on the
    captured tensors.

    Args:
        stats: Per-step captures from :meth:`AttentionProbe.capture_step`.
        top_k: How many top-attended source positions to keep per head.
        fingerprint_bins: Number of bins in the specialization fingerprint.
    """

    head_to_kv_group = list(stats.head_to_kv_group) or [
        h * stats.num_key_value_heads // max(1, stats.num_attention_heads)
        for h in range(stats.num_attention_heads)
    ]

    layer_results: dict[int, AttentionLayerDerivedStats] = {}
    for layer_idx in sorted(stats.layers.keys()):
        layer_results[layer_idx] = _compute_layer_stats(
            stats.layers[layer_idx],
            head_to_kv_group=head_to_kv_group,
            top_k=top_k,
            fingerprint_bins=fingerprint_bins,
        )

    return AttentionDerivedStats(
        layers=layer_results,
        num_attention_heads=int(stats.num_attention_heads),
        num_key_value_heads=int(stats.num_key_value_heads),
        head_dim=int(stats.head_dim),
        top_k=int(top_k),
        num_fingerprint_bins=int(fingerprint_bins),
        head_to_kv_group=head_to_kv_group,
    )


def _compute_layer_stats(
    layer: AttentionLayerStats,
    *,
    head_to_kv_group: list[int],
    top_k: int,
    fingerprint_bins: int,
) -> AttentionLayerDerivedStats:
    weights = _to_dense_weights(layer.attention_weights)
    num_heads = weights.shape[0]
    seq_len = weights.shape[-1]
    current_pos = max(0, seq_len - 1)

    q = _ensure_2d(layer.q_last, num_heads)
    k = _ensure_2d(layer.k_last, max(1, len(head_to_kv_group))) if layer.k_last is not None else None
    v = _ensure_2d(layer.v_last, max(1, len(head_to_kv_group))) if layer.v_last is not None else None

    head_stats: list[AttentionHeadStats] = []
    for head_idx in range(num_heads):
        row = weights[head_idx]
        entropy = _entropy_nats(row)
        self_weight = float(row[-1].item()) if seq_len > 0 else 0.0
        bos_weight = float(row[0].item()) if seq_len > 0 else 0.0
        top_pairs = _top_k_pairs(row, top_k)
        q_norm = _vector_norm(q, head_idx)
        kv_idx = head_to_kv_group[head_idx] if head_to_kv_group else head_idx
        k_norm = _vector_norm(k, kv_idx)
        v_norm = _vector_norm(v, kv_idx)
        angle = _qk_alignment_angle_deg(q, k, head_idx, kv_idx)
        effective_span = math.exp(entropy) if entropy > 0 else (1.0 if seq_len > 0 else 0.0)
        fingerprint = _specialization_fingerprint(row, current_pos, fingerprint_bins)
        head_stats.append(
            AttentionHeadStats(
                entropy=entropy,
                self_weight=self_weight,
                bos_weight=bos_weight,
                top_k_positions=top_pairs,
                q_norm=q_norm,
                k_norm=k_norm,
                v_norm=v_norm,
                qk_alignment_angle_deg=angle,
                effective_attention_span=effective_span,
                specialization_fingerprint=fingerprint,
            )
        )

    aggregates = _layer_aggregates(head_stats)
    return AttentionLayerDerivedStats(
        layer_idx=int(layer.layer_idx),
        heads=head_stats,
        aggregates=aggregates,
    )


def _layer_aggregates(heads: list[AttentionHeadStats]) -> AttentionLayerAggregates:
    if not heads:
        return AttentionLayerAggregates(0.0, 0.0, 0.0, 0.0)
    entropies = [h.entropy for h in heads]
    copy_count = sum(1 for h in heads if h.self_weight > COPY_HEAD_THRESHOLD)
    sink_count = sum(1 for h in heads if h.bos_weight > SINK_HEAD_THRESHOLD)
    return AttentionLayerAggregates(
        mean_entropy=float(sum(entropies) / len(entropies)),
        max_entropy=float(max(entropies)),
        copy_head_fraction=copy_count / len(heads),
        sink_head_fraction=sink_count / len(heads),
    )


def _entropy_nats(row: torch.Tensor) -> float:
    if row.numel() == 0:
        return 0.0
    eps = 1e-12
    positive = row.clamp(min=0.0)
    nonzero_mask = positive > 0
    if not bool(nonzero_mask.any().item()):
        return 0.0
    logw = torch.where(nonzero_mask, torch.log(positive + eps), torch.zeros_like(positive))
    entropy = -(positive * logw).sum()
    return float(max(0.0, entropy.item()))


def _top_k_pairs(row: torch.Tensor, k: int) -> list[tuple[int, float]]:
    if row.numel() == 0 or k <= 0:
        return []
    effective_k = min(int(k), row.shape[-1])
    top_w, top_idx = torch.topk(row, k=effective_k)
    return [
        (int(top_idx[i].item()), float(top_w[i].item()))
        for i in range(effective_k)
    ]


def _vector_norm(tensor: torch.Tensor | None, index: int) -> float:
    if tensor is None:
        return 0.0
    if index < 0 or index >= tensor.shape[0]:
        return 0.0
    return float(torch.linalg.norm(tensor[index].float()).item())


def _qk_alignment_angle_deg(
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    head_idx: int,
    kv_idx: int,
) -> float:
    if q is None or k is None:
        return 0.0
    if head_idx < 0 or head_idx >= q.shape[0]:
        return 0.0
    if kv_idx < 0 or kv_idx >= k.shape[0]:
        return 0.0
    qv = q[head_idx].float()
    kv = k[kv_idx].float()
    denom = float((torch.linalg.norm(qv) * torch.linalg.norm(kv)).item())
    if denom <= 0.0:
        return 0.0
    cos = float(torch.dot(qv, kv).item()) / denom
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _specialization_fingerprint(
    row: torch.Tensor,
    current_pos: int,
    bins: int,
) -> list[float]:
    """Normalized 16-bin histogram of weight by relative position.

    Relative position is ``(target_pos - current_pos) / current_pos`` and lies
    in ``[-1, 0]``. When ``current_pos == 0`` (no previous tokens), the
    fingerprint is the uniform distribution so downstream consumers can still
    treat every entry as a normalized distribution.
    """

    bins = max(1, int(bins))
    if current_pos <= 0 or row.shape[-1] == 0:
        return [1.0 / bins] * bins

    seq_len = row.shape[-1]
    positive = row.clamp(min=0.0).float()
    total = float(positive.sum().item())
    if total <= 0.0:
        return [1.0 / bins] * bins

    hist = [0.0] * bins
    for pos in range(seq_len):
        rel = (pos - current_pos) / current_pos
        # rel in [-1, 0]; map to bin index in [0, bins-1].
        normalized = rel + 1.0  # now in [0, 1]
        if normalized >= 1.0:
            bin_idx = bins - 1
        elif normalized <= 0.0:
            bin_idx = 0
        else:
            bin_idx = min(bins - 1, int(normalized * bins))
        hist[bin_idx] += float(positive[pos].item())

    return [v / total for v in hist]


def _to_dense_weights(weights: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
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
    return dense


def _ensure_2d(tensor: torch.Tensor | None, _expected_rows: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    t = tensor.detach().cpu().float()
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t
