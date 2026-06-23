import type { ManifoldLayer, Step, Trace } from '@/types/trace';

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

const N_POSITIONS = 5;

function makeManifoldLayer(layer: number): ManifoldLayer {
  const coords = Array.from({ length: N_POSITIONS }, (_, i) => [
    Math.cos(i),
    Math.sin(i),
    0.3 * i,
  ]);
  return {
    layer,
    submodule: 'resid_post',
    n_positions: N_POSITIONS,
    hidden_dim: 8,
    positions: Array.from({ length: N_POSITIONS }, (_, i) => i),
    pca: {
      eigenvalues: [3, 1.5, 0.5],
      explained_variance_ratio: [0.6, 0.3, 0.1],
      cumulative_variance_ratio: [0.6, 0.9, 1.0],
    },
    // Distinct per layer so a selector change is observable in tests.
    participation_ratio: 2.1 + layer * 0.5,
    intrinsic_dimension: { twonn: 1.2 + layer * 0.3 },
    projection: { n_components: 3, coords },
    trajectory_curvature: {
      mean: 0.4,
      per_position: [null, 0.41, 0.42, 0.4, null],
    },
    periodicity: { dominant_period: 6, power: 0.7, peak_frequency: 1 / 6 },
  };
}

export function makeManifoldTrace(): Trace {
  const steps: Step[] = Array.from({ length: N_POSITIONS }, (_, step) => ({
    step,
    selected: { token_id: 7 + step, token: ` tok${step}` },
    raw: makeDistribution(),
    processed: makeDistribution(),
    attention_sidecar_ref: null,
  }));
  return {
    schema_version: '2.0.0',
    manifold: {
      schema_version: '1.0.0',
      method: 'pca',
      n_components: 3,
      layers: [makeManifoldLayer(0), makeManifoldLayer(1)],
    },
    metadata: {
      model: 'test/model',
      prompt: 'hi',
      generated_text: 'hi world',
      generated_at: '2026-05-13T00:00:00Z',
      generation_params: {
        max_new_tokens: N_POSITIONS,
        temperature: 1,
        top_p: 1,
        sample_top_k: 0,
      },
      probe_config: { min_k: 1, max_k: 5, mass_threshold: 0.9 },
    },
    tokens: { prompt_token_ids: [1, 2], prompt_tokens: ['hi', ' '] },
    steps,
  };
}

export function makeTraceWithoutManifold(): Trace {
  const trace = makeManifoldTrace();
  return { ...trace, manifold: undefined };
}

export const FIXTURE_MANIFOLD_POSITIONS = N_POSITIONS;
