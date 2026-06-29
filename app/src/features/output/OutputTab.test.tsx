import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import type { Trace } from '@/types/trace';
import { OutputTab } from './OutputTab';

function makeTrace(overrides: Partial<Trace> = {}): Trace {
  const dist = {
    k_used: 1,
    entropy: 0,
    top_mass_used: 1,
    selected_prob: 1,
    selected_logprob: 0,
    selected_rank: 1,
    candidates: [{ rank: 1, token_id: 1, token: 'x', prob: 1, logprob: 0 }],
  };
  // Two lines of output so whitespace preservation is observable.
  const tokens = ['1 2 3', '\n', '4 5 6'];
  return {
    schema_version: '2.0.0',
    metadata: {
      model: 'test/model',
      prompt: 'Count.',
      generated_text: 'Count.1 2 3\n4 5 6',
      generated_at: '2026-06-25T00:00:00Z',
      generation_params: {
        max_new_tokens: 3,
        temperature: 1,
        top_p: 1,
        sample_top_k: 0,
      },
      probe_config: { min_k: 1, max_k: 1, mass_threshold: 1 },
    },
    tokens: { prompt_token_ids: [1], prompt_tokens: ['Count.'] },
    steps: tokens.map((tok, i) => ({
      step: i,
      selected: { token_id: i, token: tok },
      raw: dist,
      processed: dist,
      attention_sidecar_ref: null,
    })),
    ...overrides,
  };
}

describe('OutputTab', () => {
  it('renders the reconstructed completion with newlines preserved', () => {
    render(<OutputTab trace={makeTrace()} />);
    // Completion = concatenated selected tokens, including the newline.
    expect(screen.getByTestId('output-text').textContent).toBe('1 2 3\n4 5 6');
    expect(screen.getByText(/3 tokens/)).toBeInTheDocument();
    expect(screen.getByText(/2 lines/)).toBeInTheDocument();
  });

  it('switches to the full text (with prompt)', async () => {
    const user = userEvent.setup();
    render(<OutputTab trace={makeTrace()} />);
    await user.click(screen.getByTestId('output-view-full'));
    expect(screen.getByTestId('output-text').textContent).toBe(
      'Count.1 2 3\n4 5 6',
    );
  });

  it('disables the Full toggle when there is no generated_text', () => {
    const trace = makeTrace();
    trace.metadata.generated_text = '';
    render(<OutputTab trace={trace} />);
    expect(screen.getByTestId('output-view-full')).toBeDisabled();
  });

  it('shows the empty state when there is no output at all', () => {
    const trace = makeTrace({ steps: [] });
    trace.metadata.generated_text = '';
    render(<OutputTab trace={trace} />);
    expect(screen.getByTestId('output-empty')).toBeInTheDocument();
  });
});
