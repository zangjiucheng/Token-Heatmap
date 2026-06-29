import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { TokenHeatmap } from './TokenHeatmap';
import { syntheticTrace } from './testFixtures';

describe('TokenHeatmap performance', () => {
  it('renders a 200 step × 64 rank trace in under 1000 ms', () => {
    const trace = syntheticTrace(200, 64);
    // jsdom does not implement a 2D canvas context, so the internal paint
    // hook is skipped. We assert wall-clock render time only: the value the
    // ticket cares about is end-to-end render budget, and React + buildGrid
    // dominate even when the canvas paint runs in a real browser.
    const t0 = performance.now();
    render(
      <TokenHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={800}
        height={400}
      />,
    );
    const wallClock = performance.now() - t0;
    expect(wallClock).toBeLessThan(1000);
  });

  it('200-step render activates min-cell mode (cellW = minCellWidth)', () => {
    const trace = syntheticTrace(200, 64);
    render(
      <TokenHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={800}
        height={400}
      />,
    );
    const plot = screen.getByTestId('token-heatmap-plot');
    const cellW = Number(plot.getAttribute('data-cell-width'));
    const contentW = Number(plot.getAttribute('data-content-width'));
    const scrollW = Number(plot.getAttribute('data-scroll-width'));
    // Min-cell mode: width clamped to default minCellWidth (16) and content
    // overflows the scroll viewport, so the user gets a horizontal scrollbar.
    expect(cellW).toBe(16);
    expect(contentW).toBeGreaterThan(scrollW);
  });
});
