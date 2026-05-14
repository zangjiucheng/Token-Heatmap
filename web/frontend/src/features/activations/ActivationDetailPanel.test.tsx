import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ActivationDetailPanel } from './ActivationDetailPanel';
import { FIXTURE_SUBMODULES, makeActivationTrace } from './testFixtures';

describe('ActivationDetailPanel', () => {
  it('renders the empty state when no cell is selected', () => {
    render(
      <ActivationDetailPanel
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={null}
        selectedLayer={null}
      />,
    );
    const panel = screen.getByTestId('activation-detail-panel');
    expect(panel.className).toContain('activation-detail-panel--empty');
  });

  it('shows the top-k neurons for the selected (step, layer, submodule)', () => {
    render(
      <ActivationDetailPanel
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={1}
        selectedLayer={2}
      />,
    );
    expect(
      screen.getByTestId('activation-detail-panel-title'),
    ).toHaveTextContent(`Step 1 · L2 · ${FIXTURE_SUBMODULES[0]}`);
    // The fixture seeds three top neurons per entry with indices
    // [layer*10, layer*10+1, layer*10+2].
    const row0 = screen.getByTestId('activation-top-neuron-0');
    expect(row0.getAttribute('data-neuron-index')).toBe('20');
    const row1 = screen.getByTestId('activation-top-neuron-1');
    expect(row1.getAttribute('data-neuron-index')).toBe('21');
    const row2 = screen.getByTestId('activation-top-neuron-2');
    expect(row2.getAttribute('data-neuron-index')).toBe('22');
  });

  it('switches contents when the selected step changes', () => {
    const trace = makeActivationTrace();
    const { rerender } = render(
      <ActivationDetailPanel
        trace={trace}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={0}
        selectedLayer={1}
      />,
    );
    expect(
      screen.getByTestId('activation-detail-panel-title'),
    ).toHaveTextContent('Step 0 · L1');
    rerender(
      <ActivationDetailPanel
        trace={trace}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={2}
        selectedLayer={1}
      />,
    );
    expect(
      screen.getByTestId('activation-detail-panel-title'),
    ).toHaveTextContent('Step 2 · L1');
  });
});
