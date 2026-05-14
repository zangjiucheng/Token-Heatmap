import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ActivationsTab } from './ActivationsTab';
import {
  FIXTURE_NUM_LAYERS,
  FIXTURE_NUM_STEPS,
  FIXTURE_SUBMODULES,
  makeActivationTrace,
  makeTraceWithoutActivations,
} from './testFixtures';
import type { TraceWithActivations } from '@/types/activation';

describe('ActivationsTab', () => {
  it('renders the empty state when the trace has no activation_metadata', () => {
    render(
      <ActivationsTab
        trace={makeTraceWithoutActivations() as TraceWithActivations}
        selectedStep={null}
        onSelectStep={() => {}}
        hoveredStep={null}
        onHoverStep={() => {}}
      />,
    );
    expect(screen.getByTestId('activations-tab-empty')).toBeInTheDocument();
    expect(
      screen.queryByTestId('activations-tab-content'),
    ).not.toBeInTheDocument();
  });

  it('renders the heatmap, pickers, and detail panel when metadata is present', () => {
    render(
      <ActivationsTab
        trace={makeActivationTrace()}
        selectedStep={null}
        onSelectStep={() => {}}
        hoveredStep={null}
        onHoverStep={() => {}}
      />,
    );
    expect(screen.getByTestId('activations-tab-content')).toBeInTheDocument();
    expect(
      screen.getByTestId('activation-submodule-select'),
    ).toBeInTheDocument();
    expect(screen.getByTestId('activation-metric-select')).toBeInTheDocument();
    expect(screen.getByTestId('activation-heatmap')).toBeInTheDocument();
    expect(screen.getByTestId('activation-detail-panel')).toBeInTheDocument();
  });

  it('changes the submodule without remounting the page', () => {
    render(
      <ActivationsTab
        trace={makeActivationTrace()}
        selectedStep={null}
        onSelectStep={() => {}}
        hoveredStep={null}
        onHoverStep={() => {}}
      />,
    );
    const heatmap = screen.getByTestId('activation-heatmap');
    const select = screen.getByTestId(
      'activation-submodule-select',
    ) as HTMLSelectElement;
    expect(heatmap.getAttribute('data-num-layers')).toBe(
      String(FIXTURE_NUM_LAYERS),
    );
    expect(heatmap.getAttribute('data-num-steps')).toBe(
      String(FIXTURE_NUM_STEPS),
    );

    fireEvent.change(select, { target: { value: FIXTURE_SUBMODULES[1] } });
    expect(select.value).toBe(FIXTURE_SUBMODULES[1]);
    // The same heatmap node is reused — proves no remount.
    expect(screen.getByTestId('activation-heatmap')).toBe(heatmap);
  });

  it('cell click updates selectedStep (shared state) and detail panel', () => {
    const onSelectStep = vi.fn();
    const trace = makeActivationTrace();
    const { rerender } = render(
      <ActivationsTab
        trace={trace}
        selectedStep={null}
        onSelectStep={onSelectStep}
        hoveredStep={null}
        onHoverStep={() => {}}
      />,
    );
    fireEvent.click(screen.getByTestId('activation-cell-1-2'));
    expect(onSelectStep).toHaveBeenCalledWith(1);

    // Simulate the parent lifting selectedStep into props.
    rerender(
      <ActivationsTab
        trace={trace}
        selectedStep={1}
        onSelectStep={onSelectStep}
        hoveredStep={null}
        onHoverStep={() => {}}
      />,
    );
    expect(
      screen.getByTestId('activation-detail-panel-title'),
    ).toHaveTextContent('Step 1 · L2');
  });

  it('hovering a cell drives onHoverStep', () => {
    const onHoverStep = vi.fn();
    render(
      <ActivationsTab
        trace={makeActivationTrace()}
        selectedStep={null}
        onSelectStep={() => {}}
        hoveredStep={null}
        onHoverStep={onHoverStep}
      />,
    );
    fireEvent.mouseEnter(screen.getByTestId('activation-cell-2-1'));
    expect(onHoverStep).toHaveBeenCalledWith(2);
    fireEvent.mouseLeave(screen.getByTestId('activation-cell-2-1'));
    expect(onHoverStep).toHaveBeenCalledWith(null);
  });
});
