import type { Trace, Step } from '@/types/trace';
import type {
  ActivationLayerEntry,
  TraceWithActivations,
} from '@/types/activation';

function makeDistribution(): Step['raw'] {
  return {
    k_used: 2,
    entropy: 1.5,
    top_mass_used: 0.9,
    selected_prob: 0.5,
    selected_logprob: -0.7,
    selected_rank: 1,
    candidates: [
      { rank: 1, token_id: 7, token: ' foo', prob: 0.5, logprob: -0.7 },
      { rank: 2, token_id: 8, token: ' bar', prob: 0.3, logprob: -1.2 },
    ],
  };
}

const SUBMODULES = ['resid_post', 'mlp.down_proj'] as const;
const NUM_LAYERS = 4;
const NUM_STEPS = 3;

function makeLayerEntry(
  step: number,
  layer: number,
  submodule: string,
): ActivationLayerEntry {
  // Magnitudes vary across step/layer/submodule so colors are visibly
  // different in tests and the eye.
  const base = (step + 1) * (layer + 1);
  const submoduleOffset = submodule === 'resid_post' ? 0 : 0.5;
  return {
    layer,
    submodule,
    l2_norm: base + submoduleOffset,
    mean_abs: base * 0.1 + submoduleOffset,
    sparsity: Math.min(1, (layer + 1) / NUM_LAYERS - submoduleOffset * 0.1),
    top_neurons: [
      { index: layer * 10 + 0, value: base + submoduleOffset },
      { index: layer * 10 + 1, value: -(base + submoduleOffset) * 0.7 },
      { index: layer * 10 + 2, value: (base + submoduleOffset) * 0.4 },
    ],
  };
}

export function makeActivationTrace(): TraceWithActivations {
  const steps = Array.from({ length: NUM_STEPS }, (_, step) => {
    const activations: ActivationLayerEntry[] = [];
    for (let layer = 0; layer < NUM_LAYERS; layer += 1) {
      for (const submodule of SUBMODULES) {
        activations.push(makeLayerEntry(step, layer, submodule));
      }
    }
    return {
      step,
      selected: { token_id: 7 + step, token: ` tok${step}` },
      raw: makeDistribution(),
      processed: makeDistribution(),
      decoded_text_offset: step * 4,
      activations,
      attention_sidecar_ref: null,
    };
  });

  return {
    schema_version: '1.0.0',
    activation_metadata: {
      captured_submodules: [SUBMODULES[0], ...SUBMODULES.slice(1)] as [
        string,
        ...string[],
      ],
      num_layers: NUM_LAYERS,
      hidden_dim: 32,
      tokenizer_fingerprint: 'fixture-tokenizer',
      captured_layers: Array.from({ length: NUM_LAYERS }, (_, i) => i),
    },
    metadata: {
      model: 'test/model',
      prompt: 'hi',
      generated_text: 'hi world',
      generated_at: '2026-05-13T00:00:00Z',
      generation_params: {
        max_new_tokens: NUM_STEPS,
        temperature: 1,
        top_p: 1,
        sample_top_k: 0,
      },
      probe_config: { min_k: 1, max_k: 5, mass_threshold: 0.9 },
    },
    tokens: {
      prompt_token_ids: [1, 2],
      prompt_tokens: ['hi', ' '],
    },
    steps,
  };
}

/** An activation trace augmented with a whole-trace TWERA neuron ranking, as
 * produced by `token-heatmap trace --capture-full-activations`. */
export function makeActivationTraceWithTwera(): TraceWithActivations {
  const trace = makeActivationTrace();
  return {
    ...trace,
    neuron_attribution: {
      method: 'twera_approx',
      n_steps: NUM_STEPS,
      note: 'fixture',
      layers: [
        {
          layer: 0,
          submodule: 'resid_post',
          neurons: [
            { index: 12, twera: 0.74, mean_activation: 1.1 },
            { index: 3, twera: 0.41, mean_activation: 0.9 },
            { index: 27, twera: 0.18, mean_activation: 0.2 },
          ],
        },
        {
          layer: 1,
          submodule: 'resid_post',
          neurons: [{ index: 5, twera: 0.6, mean_activation: 0.8 }],
        },
      ],
    },
  };
}

export function makeTraceWithoutActivations(): Trace {
  const trace = makeActivationTrace();
  return {
    ...trace,
    activation_metadata: undefined,
    steps: trace.steps.map((s) => ({
      step: s.step,
      selected: s.selected,
      raw: s.raw,
      processed: s.processed,
      attention_sidecar_ref: null,
    })),
  } as Trace;
}

export const FIXTURE_SUBMODULES = SUBMODULES;
export const FIXTURE_NUM_LAYERS = NUM_LAYERS;
export const FIXTURE_NUM_STEPS = NUM_STEPS;

const DIFF_HIDDEN_DIM = 4;
const DIFF_NUM_LAYERS = 2;
const DIFF_NUM_STEPS = 2;
const DIFF_SUBMODULES = ['resid_post'] as const;

/**
 * Build a (layer, step, submodule)-dependent activation vector of length
 * DIFF_HIDDEN_DIM. The vector is fully captured in `top_neurons` so the
 * sparse-vector math in compareActivations is exact for this fixture.
 */
function diffVector(side: 'A' | 'B', step: number, layer: number): number[] {
  const base = step * 0.5 + layer * 0.25 + (side === 'A' ? 0 : 0.5);
  const v = [base, base + 0.1, -base, base * 2 - 0.3];
  return v;
}

function l2(v: number[]): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function makeDiffLayerEntry(
  side: 'A' | 'B',
  step: number,
  layer: number,
): ActivationLayerEntry {
  const v = diffVector(side, step, layer);
  const l2Norm = l2(v);
  const meanAbs = v.reduce((s, x) => s + Math.abs(x), 0) / v.length;
  const sparsity = v.filter((x) => Math.abs(x) < 1e-6).length / v.length;
  const top_neurons = v
    .map((value, index) => ({ index, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return {
    layer,
    submodule: DIFF_SUBMODULES[0],
    l2_norm: l2Norm,
    mean_abs: meanAbs,
    sparsity,
    top_neurons,
  };
}

function makeDiffTrace(
  side: 'A' | 'B',
  tokenizerFingerprint = 'diff-fixture-tokenizer',
): TraceWithActivations {
  const steps = Array.from({ length: DIFF_NUM_STEPS }, (_, step) => {
    const activations: ActivationLayerEntry[] = [];
    for (let layer = 0; layer < DIFF_NUM_LAYERS; layer += 1) {
      activations.push(makeDiffLayerEntry(side, step, layer));
    }
    return {
      step,
      selected: {
        token_id: 10 + step,
        token: ` tok${step}`,
      },
      raw: makeDistribution(),
      processed: makeDistribution(),
      decoded_text_offset: step * 4,
      activations,
      attention_sidecar_ref: null,
    };
  });
  return {
    schema_version: '1.0.0',
    activation_metadata: {
      captured_submodules: [DIFF_SUBMODULES[0]] as [string, ...string[]],
      num_layers: DIFF_NUM_LAYERS,
      hidden_dim: DIFF_HIDDEN_DIM,
      tokenizer_fingerprint: tokenizerFingerprint,
      captured_layers: Array.from({ length: DIFF_NUM_LAYERS }, (_, i) => i),
    },
    metadata: {
      model: 'test/model',
      prompt: side === 'A' ? 'hi A' : 'hi B',
      generated_text: side === 'A' ? 'A out' : 'B out',
      generated_at: '2026-05-13T00:00:00Z',
      generation_params: {
        max_new_tokens: DIFF_NUM_STEPS,
        temperature: 1,
        top_p: 1,
        sample_top_k: 0,
      },
      probe_config: { min_k: 1, max_k: 5, mass_threshold: 0.9 },
    },
    tokens: {
      prompt_token_ids: [1, 2],
      prompt_tokens: [side === 'A' ? 'hi' : 'yo', ' '],
    },
    steps,
  };
}

/**
 * Two synthetic activation traces designed for parity testing. The
 * `top_neurons` for every (step, layer) covers all DIFF_HIDDEN_DIM
 * dimensions, so the sparse-vector math used by `compareActivations` is
 * numerically exact against the documented compare semantics.
 */
export function makeTwoActivationTraces(): {
  traceA: TraceWithActivations;
  traceB: TraceWithActivations;
} {
  return { traceA: makeDiffTrace('A'), traceB: makeDiffTrace('B') };
}

export const FIXTURE_DIFF_HIDDEN_DIM = DIFF_HIDDEN_DIM;
export const FIXTURE_DIFF_NUM_LAYERS = DIFF_NUM_LAYERS;
export const FIXTURE_DIFF_NUM_STEPS = DIFF_NUM_STEPS;
export const FIXTURE_DIFF_SUBMODULES = DIFF_SUBMODULES;

/**
 * Hand-computed oracle: max per-cell |TS result − oracle| MUST be < 1e-5.
 * Computed analytically from the documented L2 / cosine semantics applied
 * to `diffVector` and stored inline so tests don't depend on the comparator
 * implementation.
 */
export function makeDiffOracle(): {
  cells: Array<{
    step: number;
    layer: number;
    submodule: string;
    l2: number;
    cosine: number;
    topChangedDeltas: number[];
  }>;
} {
  const cells = [];
  for (let step = 0; step < DIFF_NUM_STEPS; step += 1) {
    for (let layer = 0; layer < DIFF_NUM_LAYERS; layer += 1) {
      const a = diffVector('A', step, layer);
      const b = diffVector('B', step, layer);
      const diff = a.map((va, i) => va - b[i]);
      const oracleL2 = l2(diff);
      const dot = a.reduce((s, va, i) => s + va * b[i], 0);
      const normA = l2(a);
      const normB = l2(b);
      const oracleCosine =
        normA < 1e-12 || normB < 1e-12 ? 0 : dot / (normA * normB);
      const topChangedDeltas = diff
        .slice()
        .sort((x, y) => Math.abs(y) - Math.abs(x));
      cells.push({
        step,
        layer,
        submodule: DIFF_SUBMODULES[0],
        l2: oracleL2,
        cosine: oracleCosine,
        topChangedDeltas,
      });
    }
  }
  return { cells };
}
