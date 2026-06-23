import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ManifoldTab } from './ManifoldTab';
import {
  FIXTURE_MANIFOLD_POSITIONS,
  makeManifoldTrace,
  makeTraceWithoutManifold,
} from './testFixtures';

function noop() {}

describe('ManifoldTab', () => {
  it('shows the empty state when the trace has no manifold analysis', () => {
    render(
      <ManifoldTab
        trace={makeTraceWithoutManifold()}
        selectedStep={null}
        onSelectStep={noop}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    expect(screen.getByTestId('manifold-tab-empty')).toBeInTheDocument();
    expect(screen.getByText(/token-heatmap manifold/i)).toBeInTheDocument();
  });

  it('renders the scatter, scree plot, and metrics when manifold data is present', () => {
    render(
      <ManifoldTab
        trace={makeManifoldTrace()}
        selectedStep={null}
        onSelectStep={noop}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    expect(screen.getByTestId('manifold-tab-content')).toBeInTheDocument();
    expect(screen.getByTestId('manifold-scatter')).toBeInTheDocument();
    expect(screen.getByTestId('manifold-scree')).toBeInTheDocument();
    expect(screen.getByTestId('manifold-metrics')).toBeInTheDocument();
  });

  it('draws one point per token position', () => {
    render(
      <ManifoldTab
        trace={makeManifoldTrace()}
        selectedStep={null}
        onSelectStep={noop}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    for (let i = 0; i < FIXTURE_MANIFOLD_POSITIONS; i += 1) {
      expect(screen.getByTestId(`manifold-point-${i}`)).toBeInTheDocument();
    }
  });

  it('shows the geometry metrics for the active layer', () => {
    render(
      <ManifoldTab
        trace={makeManifoldTrace()}
        selectedStep={null}
        onSelectStep={noop}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    // Layer 0 participation ratio is 2.1 in the fixture.
    expect(screen.getByTestId('manifold-metric-pr')).toHaveTextContent('2.10');
    expect(screen.getByTestId('manifold-metric-twonn')).toHaveTextContent(
      '1.20',
    );
  });

  it('switches the active cloud when the layer selector changes', async () => {
    render(
      <ManifoldTab
        trace={makeManifoldTrace()}
        selectedStep={null}
        onSelectStep={noop}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    expect(screen.getByTestId('manifold-metric-pr')).toHaveTextContent('2.10');
    await userEvent.selectOptions(
      screen.getByTestId('manifold-layer-select'),
      '1:resid_post',
    );
    // Layer 1 participation ratio is 2.6 in the fixture.
    expect(screen.getByTestId('manifold-metric-pr')).toHaveTextContent('2.60');
  });

  it('calls onSelectStep with the clicked point step', async () => {
    const onSelectStep = vi.fn();
    render(
      <ManifoldTab
        trace={makeManifoldTrace()}
        selectedStep={null}
        onSelectStep={onSelectStep}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    await userEvent.click(screen.getByTestId('manifold-point-3'));
    expect(onSelectStep).toHaveBeenCalledWith(3);
  });

  it('marks the selected step point', () => {
    render(
      <ManifoldTab
        trace={makeManifoldTrace()}
        selectedStep={2}
        onSelectStep={noop}
        hoveredStep={null}
        onHoverStep={noop}
      />,
    );
    expect(screen.getByTestId('manifold-point-2')).toHaveAttribute(
      'data-selected',
      'true',
    );
    expect(screen.getByTestId('manifold-point-0')).toHaveAttribute(
      'data-selected',
      'false',
    );
  });
});
