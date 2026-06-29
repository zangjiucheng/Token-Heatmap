import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { TokenHeatmap, AXIS_GUTTER_X } from './TokenHeatmap';
import { syntheticTrace } from './testFixtures';

const trace = sampleTrace as unknown as Trace;

function patchRect(el: HTMLElement, left: number, top: number, w: number, h: number) {
  el.getBoundingClientRect = () =>
    ({
      x: left,
      y: top,
      left,
      top,
      right: left + w,
      bottom: top + h,
      width: w,
      height: h,
      toJSON: () => ({}),
    }) as DOMRect;
}

function renderHeatmap(
  overrides: Partial<React.ComponentProps<typeof TokenHeatmap>> = {},
) {
  const onSelectStep = overrides.onSelectStep ?? vi.fn();
  const utils = render(
    <TokenHeatmap
      trace={trace}
      valueCol="logprob"
      selectedStep={null}
      onSelectStep={onSelectStep}
      width={800}
      height={400}
      {...overrides}
    />,
  );
  return { ...utils, onSelectStep };
}

describe('TokenHeatmap', () => {
  it('renders a canvas with the descriptive aria-label', () => {
    renderHeatmap();
    const plot = screen.getByRole('application');
    expect(plot.getAttribute('aria-label')).toMatch(/Token probability heatmap/i);
    expect(plot.getAttribute('aria-label')).toMatch(/logprob/);
    expect(screen.getByTestId('token-heatmap-canvas')).toBeInTheDocument();
  });

  it('calls onSelectStep with the column index when a cell is clicked', () => {
    const onSelectStep = vi.fn();
    renderHeatmap({ onSelectStep });
    const plot = screen.getByTestId('token-heatmap-plot');
    patchRect(plot, 0, 0, 800, 400);

    const steps = trace.steps.length;
    // 5 steps in a 800px container with min/max defaults clamp cellW to maxCellWidth=48.
    const cellW = 48;
    const targetStep = Math.min(2, steps - 1);
    const canvas = screen.getByTestId('token-heatmap-canvas');
    // Canvas sits in grid-column 2, so its viewport left edge is the left-rail width.
    patchRect(canvas, AXIS_GUTTER_X, 0, steps * cellW, 400);

    const targetX = AXIS_GUTTER_X + cellW * targetStep + cellW / 2;
    fireEvent.click(plot, { clientX: targetX, clientY: 80 });
    expect(onSelectStep).toHaveBeenCalledWith(targetStep);
  });

  it('renders the legend with the current value column label', () => {
    renderHeatmap();
    expect(screen.getByTestId('heatmap-legend-title')).toHaveTextContent('logprob');
  });

  it('updates legend min/max when valueCol changes', () => {
    const { rerender } = renderHeatmap();
    const minLogprob = screen.getByTestId('heatmap-legend-min').textContent;
    rerender(
      <TokenHeatmap
        trace={trace}
        valueCol="prob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={800}
        height={400}
      />,
    );
    expect(screen.getByTestId('heatmap-legend-title')).toHaveTextContent('prob');
    const minProb = screen.getByTestId('heatmap-legend-min').textContent;
    expect(minProb).not.toBe(minLogprob);
  });

  it('exposes a reset button', () => {
    renderHeatmap();
    expect(screen.getByTestId('token-heatmap-reset')).toBeInTheDocument();
  });

  it('renders the legend inside a sticky right rail', () => {
    renderHeatmap();
    const rail = screen.getByTestId('token-heatmap-legend-rail');
    expect(rail).toBeInTheDocument();
    expect(rail.querySelector('[data-testid="heatmap-legend"]')).not.toBeNull();
  });

  it('test_cells_clamped_to_min_when_many_steps: 200 steps in 600px → cellW=16', () => {
    const big = syntheticTrace(200, 5);
    render(
      <TokenHeatmap
        trace={big}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={600}
        height={300}
      />,
    );
    const plot = screen.getByTestId('token-heatmap-plot');
    const cellW = Number(plot.getAttribute('data-cell-width'));
    expect(cellW).toBe(16);
  });

  it('test_cells_clamped_to_max_when_few_steps: 10 steps in 1200px → cellW=48, no stretch', () => {
    const small = syntheticTrace(10, 5);
    render(
      <TokenHeatmap
        trace={small}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={1200}
        height={400}
      />,
    );
    const plot = screen.getByTestId('token-heatmap-plot');
    const cellW = Number(plot.getAttribute('data-cell-width'));
    const contentW = Number(plot.getAttribute('data-content-width'));
    const scrollW = Number(plot.getAttribute('data-scroll-width'));
    expect(cellW).toBe(48);
    // Grid stays at its natural size; it does not stretch to fill the scroll area.
    expect(contentW).toBeLessThan(scrollW);
    expect(contentW).toBe(10 * 48);
  });

  it('test_horizontal_scroll_appears_when_grid_overflows', () => {
    const big = syntheticTrace(200, 5);
    render(
      <TokenHeatmap
        trace={big}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={600}
        height={300}
      />,
    );
    const plot = screen.getByTestId('token-heatmap-plot');
    const contentW = Number(plot.getAttribute('data-content-width'));
    const scrollW = Number(plot.getAttribute('data-scroll-width'));
    const scrollEl = screen.getByTestId('token-heatmap-scroll');
    expect(contentW).toBeGreaterThan(scrollW);
    // The scrollable wrapper itself opts in to horizontal overflow.
    expect(scrollEl).toBeInTheDocument();
  });

  it('renders without console errors on 10 successive resize events', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    renderHeatmap();
    for (let i = 0; i < 10; i += 1) {
      fireEvent(window, new Event('resize'));
    }
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
    errorSpy.mockRestore();
    warnSpy.mockRestore();
  });
});
