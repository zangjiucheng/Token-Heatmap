import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { Trace } from '@/types/trace';
import { LogitLensDecisionMap } from './LogitLensDecisionMap';

function makeTrace(): Trace {
  const layer = (idx: number, rank: number, prob: number) => ({
    layer_idx: idx,
    top_k: [],
    entropy: 0,
    selected_token_rank: rank,
    selected_token_prob: prob,
  });
  const step = (token: string, per: Array<[number, number]>) => ({
    selected: { token },
    logit_lens: per.map(([rank, prob], i) => layer(i, rank, prob)),
  });
  return {
    steps: [
      step(' A', [
        [100, 0.01],
        [5, 0.3],
        [1, 0.9],
      ]),
      step(' B', [
        [200, 0.0],
        [50, 0.05],
        [2, 0.4],
      ]),
    ],
  } as unknown as Trace;
}

describe('LogitLensDecisionMap', () => {
  it('renders a cell for every (step, layer) pair', () => {
    render(<LogitLensDecisionMap trace={makeTrace()} selectedStep={null} />);
    expect(screen.getByTestId('logit-lens-decision-map')).toBeInTheDocument();
    // 2 steps × 3 layers
    expect(screen.getByTestId('decision-cell-0-0')).toBeInTheDocument();
    expect(screen.getByTestId('decision-cell-0-2')).toBeInTheDocument();
    expect(screen.getByTestId('decision-cell-1-2')).toBeInTheDocument();
  });

  it('clicking a cell selects that step (linked brushing)', () => {
    const onSelectStep = vi.fn();
    render(
      <LogitLensDecisionMap
        trace={makeTrace()}
        selectedStep={null}
        onSelectStep={onSelectStep}
      />,
    );
    fireEvent.click(screen.getByTestId('decision-cell-1-0'));
    expect(onSelectStep).toHaveBeenCalledWith(1);
  });

  it('toggles the metric between rank and prob', () => {
    render(<LogitLensDecisionMap trace={makeTrace()} selectedStep={null} />);
    // default = rank: legend shows "rank 1" at the bright end
    expect(screen.getByText('rank 1')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('decision-map-metric-prob'));
    expect(screen.getByText('1.0')).toBeInTheDocument();
  });

  it('renders nothing when no step has logit_lens data', () => {
    const trace = { steps: [{ selected: { token: ' X' } }] } as unknown as Trace;
    const { container } = render(
      <LogitLensDecisionMap trace={trace} selectedStep={null} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('hides the vanilla/tuned toggle without tuned_logit_lens', () => {
    render(<LogitLensDecisionMap trace={makeTrace()} selectedStep={null} />);
    expect(
      screen.queryByTestId('decision-map-lens-tuned'),
    ).not.toBeInTheDocument();
  });

  it('shows and switches the vanilla/tuned lens when tuned data is present', () => {
    const base = makeTrace() as unknown as { steps: unknown[] };
    const trace = {
      ...base,
      tuned_logit_lens: {
        steps: [
          {
            layers: [
              { layer_idx: 0, selected_token_rank: 2, selected_token_prob: 0.4 },
              { layer_idx: 1, selected_token_rank: 1, selected_token_prob: 0.8 },
              { layer_idx: 2, selected_token_rank: 1, selected_token_prob: 0.95 },
            ],
          },
          {
            layers: [
              { layer_idx: 0, selected_token_rank: 3, selected_token_prob: 0.2 },
              { layer_idx: 1, selected_token_rank: 1, selected_token_prob: 0.6 },
              { layer_idx: 2, selected_token_rank: 1, selected_token_prob: 0.9 },
            ],
          },
        ],
      },
    } as unknown as Trace;

    render(<LogitLensDecisionMap trace={trace} selectedStep={null} />);
    const tuned = screen.getByTestId('decision-map-lens-tuned');
    expect(screen.getByTestId('decision-map-lens-vanilla')).toHaveAttribute(
      'data-selected',
      'true',
    );
    fireEvent.click(tuned);
    expect(tuned).toHaveAttribute('data-selected', 'true');
    // still renders the grid from the tuned source
    expect(screen.getByTestId('decision-cell-0-0')).toBeInTheDocument();
  });
});
