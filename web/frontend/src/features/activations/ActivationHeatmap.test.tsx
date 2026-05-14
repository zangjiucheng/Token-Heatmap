import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ActivationHeatmap } from './ActivationHeatmap';
import {
  FIXTURE_NUM_LAYERS,
  FIXTURE_NUM_STEPS,
  FIXTURE_SUBMODULES,
  makeActivationTrace,
} from './testFixtures';

describe('ActivationHeatmap', () => {
  it('renders one cell per (step, captured_layer)', () => {
    render(
      <ActivationHeatmap
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        metric="l2_norm"
        selectedStep={null}
        selectedLayer={null}
        onSelectCell={() => {}}
      />,
    );
    const heatmap = screen.getByTestId('activation-heatmap');
    expect(heatmap.getAttribute('data-num-layers')).toBe(
      String(FIXTURE_NUM_LAYERS),
    );
    expect(heatmap.getAttribute('data-num-steps')).toBe(
      String(FIXTURE_NUM_STEPS),
    );

    let count = 0;
    for (let step = 0; step < FIXTURE_NUM_STEPS; step += 1) {
      for (let layer = 0; layer < FIXTURE_NUM_LAYERS; layer += 1) {
        expect(
          screen.getByTestId(`activation-cell-${step}-${layer}`),
        ).toBeInTheDocument();
        count += 1;
      }
    }
    expect(count).toBe(FIXTURE_NUM_STEPS * FIXTURE_NUM_LAYERS);
  });

  it('clicking a cell invokes onSelectCell with the (step, layer)', () => {
    const onSelectCell = vi.fn();
    render(
      <ActivationHeatmap
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        metric="l2_norm"
        selectedStep={null}
        selectedLayer={null}
        onSelectCell={onSelectCell}
      />,
    );
    fireEvent.click(screen.getByTestId('activation-cell-2-3'));
    expect(onSelectCell).toHaveBeenCalledWith(2, 3);
  });

  it('reflects the hovered step on the heatmap data attribute', () => {
    render(
      <ActivationHeatmap
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        metric="l2_norm"
        selectedStep={null}
        selectedLayer={null}
        onSelectCell={() => {}}
        hoveredStep={1}
      />,
    );
    expect(
      screen
        .getByTestId('activation-heatmap')
        .getAttribute('data-hovered-step'),
    ).toBe('1');
  });
});
