import type { Trace, AttentionLayerEntry, Step } from '@/types/trace';

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

function makeAttentionEntry(layer: number): AttentionLayerEntry {
  // Intentionally inverse trends across metrics so the layer-mean
  // broadcasting still yields per-metric color variation that tests can
  // detect (e.g. layer 0 is the min for `entropy` but the max for `q_norm`).
  return {
    layer,
    entropy: 1.0 + layer * 0.5,
    self_weight: 0.3 + layer * 0.05,
    bos_weight: 0.2 - layer * 0.02,
    top_positions: [
      { position: 0, weight: 0.4 - layer * 0.05 },
      { position: 1, weight: 0.25 },
    ],
    q_norm: 2.0 - layer * 0.5,
    k_norm: 0.9 + layer * 0.1,
    v_norm: 1.1,
    qk_alignment_angle: 45,
  };
}

export function makeAttentionTrace(): Trace {
  return {
    schema_version: '1.0.0',
    attention_metadata: {
      num_layers: 4,
      num_attention_heads: 4,
      num_key_value_heads: 4,
      head_dim: 64,
      captured_layers: [0, 1, 2, 3],
    },
    metadata: {
      model: 'test/model',
      prompt: 'hi',
      generated_text: 'hi world',
      generated_at: '2026-05-13T00:00:00Z',
      generation_params: { max_new_tokens: 2, temperature: 1, top_p: 1, sample_top_k: 0 },
      probe_config: { min_k: 1, max_k: 5, mass_threshold: 0.9 },
    },
    tokens: {
      prompt_token_ids: [1, 2],
      prompt_tokens: ['hi', ' '],
    },
    steps: [
      {
        step: 0,
        selected: { token_id: 7, token: ' world' },
        raw: makeDistribution(),
        processed: makeDistribution(),
        logit_lens: [
          {
            layer_idx: 0,
            top_k: [
              { rank: 1, token_id: 9, token: ' the', prob: 0.2, logprob: -1.6 },
              { rank: 2, token_id: 7, token: ' world', prob: 0.1, logprob: -2.3 },
            ],
            entropy: 4.0,
            selected_token_rank: 2,
            selected_token_prob: 0.1,
          },
          {
            layer_idx: 3,
            top_k: [
              { rank: 1, token_id: 7, token: ' world', prob: 0.6, logprob: -0.5 },
              { rank: 2, token_id: 9, token: ' the', prob: 0.2, logprob: -1.6 },
            ],
            entropy: 1.2,
            selected_token_rank: 1,
            selected_token_prob: 0.6,
          },
        ],
        attention: [0, 1, 2, 3].map(makeAttentionEntry),
        attention_sidecar_ref: null,
      },
    ],
  };
}

export function makeTraceWithoutAttention(): Trace {
  const base = makeAttentionTrace();
  return {
    ...base,
    attention_metadata: undefined,
    steps: base.steps.map((s) => ({
      ...s,
      attention: undefined,
      attention_sidecar_ref: null,
    })),
  };
}
