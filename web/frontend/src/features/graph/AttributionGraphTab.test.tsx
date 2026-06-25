import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { setApiClientForTests, type ApiClient } from '@/api/client';
import type { Trace } from '@/types/trace';
import { AttributionGraphTab } from './AttributionGraphTab';

// The graph reuses the InterventionPanel, which probes backend health — stub it.
beforeEach(() => {
  setApiClientForTests({
    health: vi.fn().mockResolvedValue(false),
  } as unknown as ApiClient);
});
afterEach(() => setApiClientForTests(null));

function makeTrace(withDla = true): Trace {
  const base = {
    metadata: { model: 'm', prompt: 'p' },
    steps: [{ step: 0, selected: { token: ' Paris', token_id: 5 } }],
  } as unknown as Trace;
  if (!withDla) return base;
  return {
    ...base,
    direct_logit_attribution: {
      method: 'dla_fold_norm',
      n_steps: 1,
      steps: [
        {
          step: 0,
          target_token_id: 5,
          total_logit: 3.2,
          embed: 0.5,
          bias: 0,
          error: 0.01,
          layers: [
            { layer: 0, attn: 0.2, mlp: -0.1 },
            {
              layer: 5,
              attn: 1.2,
              mlp: 0.3,
              heads: [
                { head: 7, attn: 1.0 },
                { head: 2, attn: 0.2 },
              ],
            },
          ],
        },
      ],
    },
  } as unknown as Trace;
}

describe('AttributionGraphTab', () => {
  it('renders a graph of the top contributors with per-head nodes', () => {
    render(<AttributionGraphTab trace={makeTrace()} selectedStep={0} />);
    expect(
      screen.getByTestId('attribution-graph-tab-content'),
    ).toBeInTheDocument();
    expect(screen.getByTestId('attribution-graph-svg')).toBeInTheDocument();
    // embed + L0(attn,mlp) + L5(head 7, head 2, mlp) = 6 nodes.
    expect(screen.getByTestId('graph-node-embed')).toBeInTheDocument();
    expect(screen.getByTestId('graph-node-L5h7')).toBeInTheDocument();
    expect(screen.getByTestId('graph-node-L5mlp')).toBeInTheDocument();
    expect(screen.getByText(/Top 6 of 6 contributors/i)).toBeInTheDocument();
  });

  it('makes component nodes ablatable but not the embedding node', () => {
    render(<AttributionGraphTab trace={makeTrace()} selectedStep={0} />);
    expect(screen.getByTestId('graph-node-L5h7')).toHaveAttribute(
      'role',
      'button',
    );
    expect(screen.getByTestId('graph-node-embed')).not.toHaveAttribute('role');
  });

  it('shows an empty state when the trace has no attribution', () => {
    render(<AttributionGraphTab trace={makeTrace(false)} selectedStep={0} />);
    expect(
      screen.getByTestId('attribution-graph-tab-empty'),
    ).toBeInTheDocument();
  });
});
