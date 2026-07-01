import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { Trace } from '@/types/trace';
import { LogitLensConvergence } from './LogitLensConvergence';

function step(probs: number[]) {
  return {
    selected: { token: ' x' },
    logit_lens: probs.map((p, i) => ({
      layer_idx: i,
      top_k: [],
      entropy: 0,
      selected_token_rank: 1,
      selected_token_prob: p,
    })),
  };
}

function makeTrace(withTuned = false): Trace {
  const base = {
    steps: [step([0.01, 0.2, 0.9]), step([0.0, 0.3, 0.8])],
  } as Record<string, unknown>;
  if (withTuned) {
    base.tuned_logit_lens = {
      steps: [
        {
          layers: [
            { layer_idx: 0, selected_token_prob: 0.3 },
            { layer_idx: 1, selected_token_prob: 0.6 },
            { layer_idx: 2, selected_token_prob: 0.95 },
          ],
        },
        {
          layers: [
            { layer_idx: 0, selected_token_prob: 0.25 },
            { layer_idx: 1, selected_token_prob: 0.55 },
            { layer_idx: 2, selected_token_prob: 0.9 },
          ],
        },
      ],
    };
  }
  return base as unknown as Trace;
}

describe('LogitLensConvergence', () => {
  it('draws the vanilla line from per-layer probs', () => {
    render(<LogitLensConvergence trace={makeTrace()} />);
    expect(screen.getByTestId('logit-lens-convergence')).toBeInTheDocument();
    expect(screen.getByTestId('ll-converge-line-vanilla')).toBeInTheDocument();
    expect(
      screen.queryByTestId('ll-converge-line-tuned'),
    ).not.toBeInTheDocument();
  });

  it('adds a tuned line when tuned_logit_lens is present', () => {
    render(<LogitLensConvergence trace={makeTrace(true)} />);
    expect(screen.getByTestId('ll-converge-line-vanilla')).toBeInTheDocument();
    expect(screen.getByTestId('ll-converge-line-tuned')).toBeInTheDocument();
  });

  it('renders nothing with fewer than two layers', () => {
    const trace = { steps: [step([0.5])] } as unknown as Trace;
    const { container } = render(<LogitLensConvergence trace={trace} />);
    expect(container.firstChild).toBeNull();
  });
});
