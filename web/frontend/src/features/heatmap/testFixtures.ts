import type { Trace, Step } from '@/types/trace';

/**
 * Build a synthetic `Trace` with a configurable number of steps and ranks.
 * Used by perf tests, min/max cell-sizing tests, and scroll tests where
 * the sample trace's 5×5 grid is too small to exercise the relevant code.
 */
export function syntheticTrace(steps: number, ranks: number): Trace {
  const stepObjs: Step[] = [];
  for (let i = 0; i < steps; i += 1) {
    const candidates = [];
    for (let r = 0; r < ranks; r += 1) {
      const prob = 1 / (r + 1) / ranks;
      candidates.push({
        rank: r + 1,
        token_id: i * 1000 + r,
        token: `t${i}-${r}`,
        prob,
        logprob: Math.log(prob + 1e-9),
      });
    }
    stepObjs.push({
      step: i,
      selected: { token_id: candidates[0].token_id, token: candidates[0].token },
      raw: {
        k_used: ranks,
        entropy: 1.0,
        top_mass_used: 0.95,
        selected_prob: candidates[0].prob,
        selected_logprob: candidates[0].logprob,
        selected_rank: 1,
        candidates,
      },
      processed: {
        k_used: ranks,
        entropy: 1.0,
        top_mass_used: 0.95,
        selected_prob: candidates[0].prob,
        selected_logprob: candidates[0].logprob,
        selected_rank: 1,
        candidates,
      },
    });
  }
  return {
    schema_version: '2.0.0',
    metadata: {
      model: 'synthetic',
      prompt: '',
      generated_text: '',
      generated_at: '2026-05-12T00:00:00Z',
      generation_params: {
        max_new_tokens: steps,
        temperature: 1.0,
        top_p: 1.0,
        sample_top_k: 0,
      },
      probe_config: { min_k: 1, max_k: ranks, mass_threshold: 0.9 },
    },
    tokens: { prompt_token_ids: [], prompt_tokens: [] },
    steps: stepObjs,
  };
}
