import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { DiffView } from './DiffView';
import {
  FIXTURE_DIFF_NUM_LAYERS,
  FIXTURE_DIFF_NUM_STEPS,
  makeTwoActivationTraces,
} from './testFixtures';

describe('DiffView', () => {
  it('renders the delta heatmap and detail-panel placeholder', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    render(<DiffView traceA={traceA} traceB={traceB} />);
    expect(screen.getByTestId('diff-view-content')).toBeInTheDocument();
    const heatmap = screen.getByTestId('diff-heatmap');
    expect(heatmap.getAttribute('data-num-layers')).toBe(
      String(FIXTURE_DIFF_NUM_LAYERS),
    );
    expect(heatmap.getAttribute('data-num-steps')).toBe(
      String(FIXTURE_DIFF_NUM_STEPS),
    );
    expect(heatmap.getAttribute('data-metric')).toBe('l2');
    expect(screen.getByTestId('diff-detail-panel')).toBeInTheDocument();
    expect(screen.getByTestId('diff-alignment-status')).toHaveTextContent(
      /auto/i,
    );
  });

  it('clicking a cell updates the top-K changed-neurons panel', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    render(<DiffView traceA={traceA} traceB={traceB} />);
    fireEvent.click(screen.getByTestId('diff-cell-1-1'));
    expect(
      screen.getByTestId('diff-detail-panel-title'),
    ).toHaveTextContent('Step 1 · L1');
    expect(screen.getByTestId('diff-top-neuron-0')).toBeInTheDocument();
  });

  it('toggles metric between L2 and cosine without remount', async () => {
    const user = userEvent.setup();
    const { traceA, traceB } = makeTwoActivationTraces();
    render(<DiffView traceA={traceA} traceB={traceB} />);
    const heatmap = screen.getByTestId('diff-heatmap');
    expect(heatmap.getAttribute('data-metric')).toBe('l2');

    await user.selectOptions(
      screen.getByTestId('diff-metric-select'),
      'cosine',
    );
    // Same heatmap DOM node should be reused.
    expect(screen.getByTestId('diff-heatmap')).toBe(heatmap);
    expect(heatmap.getAttribute('data-metric')).toBe('cosine');
  });

  it('token strip toggles between trace A and trace B tokens', async () => {
    const user = userEvent.setup();
    const { traceA, traceB } = makeTwoActivationTraces();
    // Make B's tokens visibly different so we can assert on the switch.
    for (const s of traceB.steps) {
      s.selected = { ...s.selected, token: ` B${s.step}` };
    }
    render(<DiffView traceA={traceA} traceB={traceB} />);
    // Default = trace A tokens.
    expect(screen.getByTestId('generated-token-0')).toHaveTextContent(/tok0/);
    await user.click(screen.getByTestId('diff-token-side-b'));
    expect(screen.getByTestId('generated-token-0')).toHaveTextContent(/B0/);
    await user.click(screen.getByTestId('diff-token-side-a'));
    expect(screen.getByTestId('generated-token-0')).toHaveTextContent(/tok0/);
  });
});
