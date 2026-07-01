import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { Trace } from '@/types/trace';
import { CausalHeatmap } from './CausalHeatmap';

function makeTrace(): Trace {
  const layer = (l: number, adp: number, mdp: number) => ({
    layer: l,
    attn: { kl: 0.1, dp: adp },
    mlp: { kl: 0.5, dp: mdp },
  });
  const step = (i: number, token: string) => ({
    step: i,
    token,
    base_prob: 0.5,
    layers: [layer(0, -0.2, -0.9), layer(1, 0.05, -0.4)],
  });
  return {
    causal_grid: {
      components: ['attn', 'mlp'],
      num_layers: 2,
      steps: [step(0, ' A'), step(1, ' B')],
    },
  } as unknown as Trace;
}

describe('CausalHeatmap', () => {
  it('renders a cell per (step, layer)', () => {
    render(<CausalHeatmap trace={makeTrace()} selectedStep={null} />);
    expect(screen.getByTestId('causal-heatmap')).toBeInTheDocument();
    expect(screen.getByTestId('causal-cell-0-0')).toBeInTheDocument();
    expect(screen.getByTestId('causal-cell-1-1')).toBeInTheDocument();
  });

  it('toggles between attn and mlp', () => {
    render(<CausalHeatmap trace={makeTrace()} selectedStep={null} />);
    const mlp = screen.getByTestId('causal-component-mlp');
    expect(mlp).toHaveAttribute('data-selected', 'false');
    fireEvent.click(mlp);
    expect(mlp).toHaveAttribute('data-selected', 'true');
    expect(screen.getByTestId('causal-component-attn')).toHaveAttribute(
      'data-selected',
      'false',
    );
  });

  it('clicking a cell selects that step', () => {
    const onSelectStep = vi.fn();
    render(
      <CausalHeatmap
        trace={makeTrace()}
        selectedStep={null}
        onSelectStep={onSelectStep}
      />,
    );
    fireEvent.click(screen.getByTestId('causal-cell-1-0'));
    expect(onSelectStep).toHaveBeenCalledWith(1);
  });

  it('renders nothing without a causal_grid', () => {
    const { container } = render(
      <CausalHeatmap trace={{ steps: [] } as unknown as Trace} selectedStep={null} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
