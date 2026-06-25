import type { Step, Trace } from '@/types/trace';

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

function baseTrace(): Trace {
  const steps: Step[] = [0, 1].map((step) => ({
    step,
    selected: { token_id: 7 + step, token: ` tok${step}` },
    raw: makeDistribution(),
    processed: makeDistribution(),
    attention_sidecar_ref: null,
  }));
  return {
    schema_version: '2.0.0',
    metadata: {
      model: 'Qwen/Qwen2.5-7B-Instruct',
      prompt: 'hi',
      generated_text: 'hi world',
      generated_at: '2026-05-13T00:00:00Z',
      device: 'cuda',
      generation_params: {
        max_new_tokens: 2,
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

/** A trace carrying a full self-contained `model_architecture` block. */
export function makeTraceWithArchitecture(): Trace {
  return {
    ...baseTrace(),
    model_architecture: {
      architecture: 'Qwen2ForCausalLM',
      model_type: 'qwen2',
      num_layers: 28,
      hidden_size: 3584,
      num_attention_heads: 28,
      num_key_value_heads: 4,
      head_dim: 128,
      intermediate_size: 18944,
      vocab_size: 152064,
      max_position_embeddings: 32768,
      rope_theta: 1000000,
      tie_word_embeddings: false,
      num_parameters: 7615616512,
      dtype: 'float16',
    },
  };
}

/** A trace with no `model_architecture`, only attention metadata — exercises
 * the graceful fallback path. */
export function makeTraceArchitectureFallback(): Trace {
  return {
    ...baseTrace(),
    attention_metadata: {
      num_layers: 28,
      num_attention_heads: 28,
      num_key_value_heads: 4,
      head_dim: 128,
      captured_layers: [0, 1, 2],
    },
  };
}

/** A trace with neither architecture nor probe metadata — only the model name. */
export function makeTraceArchitectureBare(): Trace {
  return baseTrace();
}
