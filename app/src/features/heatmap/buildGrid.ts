import type { Trace } from '@/types/trace';

export type ValueCol = 'logprob' | 'prob';
export type DistributionSource = 'raw' | 'processed';

export interface HeatmapGrid {
  /** Number of generation steps (columns). */
  steps: number;
  /** Max k_used across the trace; height of the rendered matrix. */
  ranks: number;
  /**
   * Cell values in row-major order: `values[rank * steps + step]`.
   * Cells with `rank >= k_used` for a step are `NaN`.
   */
  values: Float32Array;
  /**
   * Candidate token strings, same indexing as `values`. Empty string for
   * invalid cells.
   */
  tokens: string[];
  /** Candidate probability, same indexing — used by tooltip regardless of valueCol. */
  probs: Float32Array;
  /** Candidate logprob, same indexing. */
  logprobs: Float32Array;
  /** Per-step k_used (length = steps). */
  kUsed: Int32Array;
  /** Per-step entropy (length = steps). */
  entropy: Float32Array;
  /** Finite min over `values` (NaN if no finite values). */
  valueMin: number;
  /** Finite max over `values` (NaN if no finite values). */
  valueMax: number;
}

/**
 * Pure projection of a `Trace` into a 2D candidate matrix suitable for
 * direct canvas rendering. The `source` parameter selects between the raw
 * temperature-scaled distribution and the post-sampling-filter distribution
 * distribution. `processed` is the historical default used by the single
 * heatmap view.
 */
export function buildGrid(
  trace: Trace,
  valueCol: ValueCol,
  source: DistributionSource = 'processed',
): HeatmapGrid {
  const steps = trace.steps.length;
  let ranks = 0;
  for (const s of trace.steps) {
    const dist = source === 'raw' ? s.raw : s.processed;
    if (dist.k_used > ranks) ranks = dist.k_used;
  }
  if (ranks < 1) ranks = 1;

  const size = ranks * steps;
  const values = new Float32Array(size);
  const probs = new Float32Array(size);
  const logprobs = new Float32Array(size);
  const tokens: string[] = new Array(size).fill('');
  const kUsed = new Int32Array(steps);
  const entropy = new Float32Array(steps);

  values.fill(NaN);
  probs.fill(NaN);
  logprobs.fill(NaN);

  let min = Infinity;
  let max = -Infinity;

  for (let step = 0; step < steps; step += 1) {
    const dist =
      source === 'raw' ? trace.steps[step].raw : trace.steps[step].processed;
    kUsed[step] = dist.k_used;
    entropy[step] = dist.entropy;
    for (const cand of dist.candidates) {
      const rank = cand.rank - 1; // 1-indexed in payload, 0-indexed in matrix
      if (rank < 0 || rank >= ranks) continue;
      const idx = rank * steps + step;
      const v = valueCol === 'prob' ? cand.prob : cand.logprob;
      values[idx] = v;
      probs[idx] = cand.prob;
      logprobs[idx] = cand.logprob;
      tokens[idx] = cand.token;
      if (Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }

  return {
    steps,
    ranks,
    values,
    tokens,
    probs,
    logprobs,
    kUsed,
    entropy,
    valueMin: Number.isFinite(min) ? min : NaN,
    valueMax: Number.isFinite(max) ? max : NaN,
  };
}

/** Normalize `v` to `[0, 1]` over `[min, max]`. Returns NaN when v is NaN. */
export function normalize(v: number, min: number, max: number): number {
  if (!Number.isFinite(v)) return NaN;
  if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) return 0.5;
  return (v - min) / (max - min);
}
