/**
 * Pure-TS port of `llm_token_heatmap.compare_activations`.
 *
 * Takes two activation-bearing traces and emits an ActivationDiff record
 * matching `docs/web/activation-diff.schema.json`. The math operates on the
 * inline summary stats: for each aligned (step, layer, submodule) the
 * activation vector is reconstructed as a sparse vector keyed by the union
 * of `top_neurons` indices from both traces (missing dims treated as zero).
 * The fixture in `testFixtures.makeTwoActivationTraces` is constructed so
 * `top_neurons` covers every nonzero dimension, making the result exact for
 * parity testing.
 */
import type {
  ActivationDiff,
  ActivationLayerEntry,
  ActivationMetadata,
  Alignment,
  AlignmentMismatch,
  DiffStep,
  LayerDelta,
  StepWithActivations,
  TopChangedNeuron,
  TraceWithActivations,
} from '@/types/activation';

export type DiffAlignmentMode = 'token_id' | 'position' | 'auto';

export interface CompareActivationsOptions {
  /** Alignment mode; defaults to 'auto'. */
  align?: DiffAlignmentMode;
  /** Top-K changed neurons per (step, layer, submodule). Defaults to 8. */
  topK?: number;
}

export const DIFF_SCHEMA_VERSION = '1.0.0';

export class CompareActivationsError extends Error {
  override readonly name = 'CompareActivationsError';
}

function requireActivationMetadata(
  trace: TraceWithActivations,
  label: 'A' | 'B',
): ActivationMetadata {
  if (!trace.activation_metadata) {
    throw new CompareActivationsError(
      `Trace ${label} has no activation_metadata; cannot diff.`,
    );
  }
  return trace.activation_metadata;
}

function resolveAlignmentMode(
  requested: DiffAlignmentMode,
  metaA: ActivationMetadata,
  metaB: ActivationMetadata,
): 'token_id' | 'position' {
  if (requested === 'token_id') return 'token_id';
  if (requested === 'position') return 'position';
  return metaA.tokenizer_fingerprint === metaB.tokenizer_fingerprint
    ? 'token_id'
    : 'position';
}

interface AlignedPair {
  step: number;
  a: StepWithActivations;
  b: StepWithActivations;
  stepA: number;
  stepB: number;
}

interface AlignmentResult {
  pairs: AlignedPair[];
  mismatches: AlignmentMismatch[];
}

function alignByTokenId(
  stepsA: StepWithActivations[],
  stepsB: StepWithActivations[],
): AlignmentResult {
  const pairs: AlignedPair[] = [];
  const mismatches: AlignmentMismatch[] = [];
  const minLen = Math.min(stepsA.length, stepsB.length);
  let aligned = 0;
  for (let i = 0; i < minLen; i += 1) {
    const a = stepsA[i];
    const b = stepsB[i];
    if (a.selected.token_id === b.selected.token_id) {
      pairs.push({ step: aligned, a, b, stepA: i, stepB: i });
      aligned += 1;
    } else {
      mismatches.push({
        step_a: i,
        step_b: i,
        reason: 'token_id_divergence',
      });
    }
  }
  for (let i = minLen; i < stepsA.length; i += 1) {
    mismatches.push({ step_a: i, step_b: null, reason: 'trailing_steps_in_a' });
  }
  for (let i = minLen; i < stepsB.length; i += 1) {
    mismatches.push({ step_a: null, step_b: i, reason: 'trailing_steps_in_b' });
  }
  return { pairs, mismatches };
}

function getOffset(step: StepWithActivations, fallback: number): number {
  return step.decoded_text_offset != null ? step.decoded_text_offset : fallback;
}

function alignByPosition(
  stepsA: StepWithActivations[],
  stepsB: StepWithActivations[],
): AlignmentResult {
  // Match steps with equal decoded_text_offset; if a step's offset is missing
  // we fall back to its index. Two pointers walk both sequences in offset
  // order; non-matching offsets are recorded as mismatches.
  const pairs: AlignedPair[] = [];
  const mismatches: AlignmentMismatch[] = [];
  let i = 0;
  let j = 0;
  let aligned = 0;
  while (i < stepsA.length && j < stepsB.length) {
    const oa = getOffset(stepsA[i], i);
    const ob = getOffset(stepsB[j], j);
    if (oa === ob) {
      pairs.push({
        step: aligned,
        a: stepsA[i],
        b: stepsB[j],
        stepA: i,
        stepB: j,
      });
      aligned += 1;
      i += 1;
      j += 1;
    } else if (oa < ob) {
      mismatches.push({ step_a: i, step_b: null, reason: 'offset_gap' });
      i += 1;
    } else {
      mismatches.push({ step_a: null, step_b: j, reason: 'offset_gap' });
      j += 1;
    }
  }
  for (; i < stepsA.length; i += 1) {
    mismatches.push({ step_a: i, step_b: null, reason: 'trailing_steps_in_a' });
  }
  for (; j < stepsB.length; j += 1) {
    mismatches.push({ step_a: null, step_b: j, reason: 'trailing_steps_in_b' });
  }
  return { pairs, mismatches };
}

function intersectSubmodules(
  metaA: ActivationMetadata,
  metaB: ActivationMetadata,
): string[] {
  const setB = new Set(metaB.captured_submodules);
  return metaA.captured_submodules.filter((s) => setB.has(s));
}

function intersectLayers(
  metaA: ActivationMetadata,
  metaB: ActivationMetadata,
): number[] {
  const layersA =
    metaA.captured_layers ??
    Array.from({ length: metaA.num_layers }, (_, i) => i);
  const layersB = new Set(
    metaB.captured_layers ??
      Array.from({ length: metaB.num_layers }, (_, i) => i),
  );
  return [...layersA].filter((l) => layersB.has(l)).sort((a, b) => a - b);
}

function findEntry(
  activations: ActivationLayerEntry[] | undefined,
  layer: number,
  submodule: string,
): ActivationLayerEntry | undefined {
  return activations?.find(
    (e) => e.layer === layer && e.submodule === submodule,
  );
}

interface LayerMath {
  l2: number;
  cosine: number;
  topChanged: TopChangedNeuron[];
}

/**
 * Compute L2 of (a-b), cosine(a,b), and the top-K |delta| neurons from a
 * pair of `top_neurons` lists. Neurons not present in either list are
 * treated as zero. `l2NormA`/`l2NormB` are passed in so we can fall back
 * to the inline norms for cosine when a vector has no captured neurons.
 */
function computeLayerMath(
  a: ActivationLayerEntry | undefined,
  b: ActivationLayerEntry | undefined,
  topK: number,
): LayerMath {
  const neuronsA = new Map<number, number>();
  for (const n of a?.top_neurons ?? []) neuronsA.set(n.index, n.value);
  const neuronsB = new Map<number, number>();
  for (const n of b?.top_neurons ?? []) neuronsB.set(n.index, n.value);

  const indices = new Set<number>();
  neuronsA.forEach((_, k) => indices.add(k));
  neuronsB.forEach((_, k) => indices.add(k));

  let sumSqDelta = 0;
  let dot = 0;
  const deltas: TopChangedNeuron[] = [];
  for (const idx of indices) {
    const va = neuronsA.get(idx) ?? 0;
    const vb = neuronsB.get(idx) ?? 0;
    const d = va - vb;
    sumSqDelta += d * d;
    dot += va * vb;
    deltas.push({ index: idx, delta: d });
  }

  const l2 = Math.sqrt(sumSqDelta);

  // Use the producer's inline l2_norm when available so cosine reflects the
  // full vector (top_neurons may not capture every nonzero dim in general).
  // For the bundled fixture the inline norm equals the sparse-vector norm,
  // so cosine is exact.
  const normA = a?.l2_norm ?? Math.sqrt(sumOfSquares(neuronsA));
  const normB = b?.l2_norm ?? Math.sqrt(sumOfSquares(neuronsB));
  let cosine: number;
  if (normA < 1e-12 || normB < 1e-12) {
    cosine = 0;
  } else {
    cosine = dot / (normA * normB);
    if (cosine > 1) cosine = 1;
    if (cosine < -1) cosine = -1;
  }

  deltas.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
  const topChanged = deltas.slice(0, topK);

  return { l2, cosine, topChanged };
}

function sumOfSquares(m: Map<number, number>): number {
  let s = 0;
  m.forEach((v) => {
    s += v * v;
  });
  return s;
}

/**
 * Run the diff. Returns a payload conforming to
 * `docs/web/activation-diff.schema.json`.
 */
export function compareActivations(
  traceA: TraceWithActivations,
  traceB: TraceWithActivations,
  options: CompareActivationsOptions = {},
): ActivationDiff {
  const metaA = requireActivationMetadata(traceA, 'A');
  const metaB = requireActivationMetadata(traceB, 'B');

  const requested = options.align ?? 'auto';
  const resolved = resolveAlignmentMode(requested, metaA, metaB);
  const topK = options.topK ?? 8;

  const alignment: Alignment = {
    mode: requested,
    tokenizer_a_fingerprint: metaA.tokenizer_fingerprint,
    tokenizer_b_fingerprint: metaB.tokenizer_fingerprint,
    mismatches: [],
  };

  const { pairs, mismatches } =
    resolved === 'token_id'
      ? alignByTokenId(traceA.steps, traceB.steps)
      : alignByPosition(traceA.steps, traceB.steps);
  alignment.mismatches = mismatches;

  const submodules = intersectSubmodules(metaA, metaB);
  const layers = intersectLayers(metaA, metaB);

  const steps: DiffStep[] = pairs.map(({ step, a, b }) => {
    const delta: LayerDelta[] = [];
    for (const layer of layers) {
      for (const submodule of submodules) {
        const entryA = findEntry(a.activations, layer, submodule);
        const entryB = findEntry(b.activations, layer, submodule);
        const { l2, cosine, topChanged } = computeLayerMath(
          entryA,
          entryB,
          topK,
        );
        delta.push({
          layer,
          submodule,
          l2,
          cosine,
          top_changed_neurons: topChanged,
        });
      }
    }
    return {
      step,
      token_id_a: a.selected.token_id,
      token_id_b: b.selected.token_id,
      decoded_text_offset_a: getOffset(a, step),
      decoded_text_offset_b: getOffset(b, step),
      delta,
    };
  });

  return {
    schema_version: DIFF_SCHEMA_VERSION,
    alignment,
    steps,
  };
}

/**
 * Diff metric exposed by the UI's L2 ↔ cosine toggle. Stored separately
 * from the trace metric type so it can grow independently.
 */
export type DiffMetric = 'l2' | 'cosine';

export const DIFF_METRICS: DiffMetric[] = ['l2', 'cosine'];

export const DIFF_METRIC_LABELS: Record<DiffMetric, string> = {
  l2: 'L2 of (A − B)',
  cosine: 'Cosine(A, B)',
};
