import type { AttentionLayerEntry } from '@/types/trace';

// Ordered functional-first: BOS-weight (sink signature) and induction lead,
// because a head's role is what it does, not how big its vectors are. Q/K/V
// norms were removed — they measure activation magnitude, which is
// anti-correlated with a head's actual contribution.
export type AttentionMetric =
  | 'bos_weight'
  | 'induction'
  | 'self_weight'
  | 'top1_weight'
  | 'entropy';

export const ATTENTION_METRICS: ReadonlyArray<AttentionMetric> = [
  'bos_weight',
  'induction',
  'self_weight',
  'top1_weight',
  'entropy',
];

export const ATTENTION_METRIC_LABELS: Record<AttentionMetric, string> = {
  bos_weight: 'BOS weight (sink)',
  induction: 'Induction',
  self_weight: 'Self-weight',
  top1_weight: 'Top-1 weight',
  entropy: 'Entropy',
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
   * broadcast fallback (older traces carry no per-head induction).
   */
  induction: number;
  top1_weight: number;
}

/**
 * Per-head scalars in COLUMNAR form — one parallel array per metric, ordered by
 * head index (matches the trace schema). Columnar keeps the repeated JSON keys
 * from bloating the file. Q/K/V norms are intentionally absent.
 */
export interface PerHeadAttentionColumns {
  entropy?: number[];
  self_weight?: number[];
  bos_weight?: number[];
  induction?: number[];
  top1_weight?: number[];
}

export interface AttentionLayerEntryWithPerHead extends AttentionLayerEntry {
  per_head?: PerHeadAttentionColumns;
}

/** Compute the scalar value for a (metric, scalars) pair. */
export function getMetricValue(
  metric: AttentionMetric,
  scalars: PerHeadAttentionScalars,
): number {
  return scalars[metric];
}

/**
 * Derive per-head scalars for a layer entry. When the entry carries COLUMNAR
 * `per_head` data we read each metric's array by head index; otherwise we
 * broadcast the layer-mean scalars across `numHeads` slots (graceful fallback
 * for older traces / file-drop mode). `top1_weight` falls back to the layer's
 * highest-weight source position.
 */
export function derivePerHeadScalars(
  entry: AttentionLayerEntry,
  numHeads: number,
): PerHeadAttentionScalars[] {
  const cols = (entry as AttentionLayerEntryWithPerHead).per_head;
  const top1 =
    entry.top_positions && entry.top_positions.length > 0
      ? entry.top_positions[0].weight
      : 0;
  if (cols && cols.bos_weight && cols.bos_weight.length > 0) {
    return Array.from({ length: cols.bos_weight.length }, (_, i) => ({
      entropy: cols.entropy?.[i] ?? entry.entropy,
      self_weight: cols.self_weight?.[i] ?? entry.self_weight,
      bos_weight: cols.bos_weight?.[i] ?? entry.bos_weight,
      induction: cols.induction?.[i] ?? 0,
      top1_weight: cols.top1_weight?.[i] ?? top1,
    }));
  }
  const broadcast: PerHeadAttentionScalars = {
    entropy: entry.entropy,
    self_weight: entry.self_weight,
    bos_weight: entry.bos_weight,
    // Induction has no layer-mean to broadcast; falls back to 0.
    induction: 0,
    top1_weight: top1,
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
   * (optionally) columnar per-head scalars.
   */
  layers: AttentionSidecarLayer[];
}

export interface AttentionSidecarLayer {
  layer: number;
  /** Shape: [num_heads, key_seq_len]. */
  attention_distribution: number[][];
  per_head?: PerHeadAttentionColumns;
}
