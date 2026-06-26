import type { AttentionLayerEntry } from '@/types/trace';

export type AttentionMetric =
  | 'entropy'
  | 'self_weight'
  | 'bos_weight'
  | 'induction'
  | 'top1_weight'
  | 'q_norm'
  | 'k_norm'
  | 'v_norm';

export const ATTENTION_METRICS: ReadonlyArray<AttentionMetric> = [
  'entropy',
  'self_weight',
  'bos_weight',
  'induction',
  'top1_weight',
  'q_norm',
  'k_norm',
  'v_norm',
];

export const ATTENTION_METRIC_LABELS: Record<AttentionMetric, string> = {
  entropy: 'Entropy',
  self_weight: 'Self-weight',
  bos_weight: 'BOS weight',
  induction: 'Induction',
  top1_weight: 'Top-1 weight',
  q_norm: 'Q norm',
  k_norm: 'K norm',
  v_norm: 'V norm',
};

/**
 * Per-head Tier 1 scalars. Carried under the optional `per_head` field on
 * AttentionLayerEntry when the foundational schema extension lands; until
 * then the grid degrades to broadcasting the layer-mean across all head
 * columns (Q1 fallback).
 */
export interface PerHeadAttentionScalars {
  entropy: number;
  self_weight: number;
  bos_weight: number;
  /**
   * Induction score — attention this head puts on the token after the current
   * token's most recent earlier occurrence. High in induction heads; 0 in the
   * broadcast fallback (older traces carry no per-head induction at the layer
   * level).
   */
  induction: number;
  top1_weight: number;
  q_norm: number;
  k_norm: number;
  v_norm: number;
}

export interface AttentionLayerEntryWithPerHead extends AttentionLayerEntry {
  per_head?: PerHeadAttentionScalars[];
}

/** Compute the scalar value for a (metric, scalars) pair. */
export function getMetricValue(
  metric: AttentionMetric,
  scalars: PerHeadAttentionScalars,
): number {
  return scalars[metric];
}

/**
 * Derive per-head scalars for a given layer entry. When the entry has a
 * `per_head` array we return it directly; otherwise we broadcast the
 * layer-mean scalars across `numHeads` synthetic head slots. `top1_weight`
 * is approximated from the entry's `top_positions[0].weight` (the
 * highest-weight source position across heads).
 */
export function derivePerHeadScalars(
  entry: AttentionLayerEntry,
  numHeads: number,
): PerHeadAttentionScalars[] {
  const ext = entry as AttentionLayerEntryWithPerHead;
  if (ext.per_head && ext.per_head.length > 0) {
    return ext.per_head;
  }
  const top1 =
    entry.top_positions && entry.top_positions.length > 0
      ? entry.top_positions[0].weight
      : 0;
  const broadcast: PerHeadAttentionScalars = {
    entropy: entry.entropy,
    self_weight: entry.self_weight,
    bos_weight: entry.bos_weight,
    // Induction is a per-head-only scalar (no layer-mean to broadcast); older
    // traces without `per_head` fall back to 0 here.
    induction: 0,
    top1_weight: top1,
    q_norm: entry.q_norm,
    k_norm: entry.k_norm,
    v_norm: entry.v_norm,
  };
  return Array.from({ length: numHeads }, () => broadcast);
}

/**
 * Sidecar JSON payload schema (Tier 2). The serializer rewrite is scoped
 * to a follow-up foundational ticket; this type captures the consumer-side
 * contract. One file per trace, indexed by step.
 */
export interface AttentionSidecar {
  step?: number;
  num_layers: number;
  num_heads: number;
  /**
   * Ordered by `captured_layers` from `attention_metadata`. Each entry
   * carries the full `[num_heads, key_seq_len]` attention distribution and
   * (optionally) per-head Q/K/V norms.
   */
  layers: AttentionSidecarLayer[];
}

export interface AttentionSidecarLayer {
  layer: number;
  /** Shape: [num_heads, key_seq_len]. */
  attention_distribution: number[][];
  per_head?: PerHeadAttentionScalars[];
}
