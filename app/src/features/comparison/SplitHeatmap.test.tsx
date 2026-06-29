import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { SplitHeatmap } from './SplitHeatmap';
import { AXIS_GUTTER_X } from '@/features/heatmap/TokenHeatmap';
import { syntheticTrace } from '@/features/heatmap/testFixtures';

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

describe('SplitHeatmap', () => {
  it('renders two panes labelled raw and processed', () => {
    render(
      <SplitHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
      />,
    );
    expect(screen.getByTestId('split-heatmap-raw')).toBeInTheDocument();
    expect(screen.getByTestId('split-heatmap-processed')).toBeInTheDocument();
  });

  it('shares a single legend reflecting the combined color range', () => {
    render(
      <SplitHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
      />,
    );
    // The legend lives at the SplitHeatmap level; per-pane legend rails are
    // hidden along with the toolbar.
    const legends = screen.getAllByTestId('heatmap-legend');
    expect(legends).toHaveLength(1);
  });

  it('syncs hover across both panes: hovering the left sets hoveredStep on the right', () => {
    render(
      <SplitHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
      />,
    );
    const plots = screen.getAllByTestId('token-heatmap-plot');
    expect(plots).toHaveLength(2);
    const [leftPlot, rightPlot] = plots;
    // TokenHeatmap auto-sizes off the container's rect at mount; in jsdom that
    // resolves to its (200, 150) floor. Patch both rects so the mouse-move
    // translation lines up with that size.
    patchRect(leftPlot, 0, 0, 200, 150);
    patchRect(rightPlot, 0, 0, 200, 150);

    const steps = trace.steps.length;
    // In split mode hideToolbar=true so no right rail. scrollW = 200-36 = 164.
    // baseCellW = 164/5 = 32.8, within [16, 48] so cellW = 32.8.
    const cellW = (200 - AXIS_GUTTER_X) / steps;
    const canvases = screen.getAllByTestId('token-heatmap-canvas');
    patchRect(canvases[0], AXIS_GUTTER_X, 0, steps * cellW, 150);
    patchRect(canvases[1], AXIS_GUTTER_X, 0, steps * cellW, 150);

    // Hover column 1 on the left pane. SplitHeatmap should lift this to shared
    // state and re-render the right pane with the cross-pane hover step.
    fireEvent.mouseMove(leftPlot, {
      clientX: AXIS_GUTTER_X + cellW * 1.5,
      clientY: 5,
    });
    expect(rightPlot.getAttribute('data-external-hovered-step')).toBe('1');
  });

  it('test_synchronised_horizontal_scroll: scrolling one pane scrolls the other', () => {
    // Use a wide trace so the natural grid overflows the (200px) scroll
    // wrapper auto-sized by jsdom; with the sample 5-step trace there's
    // nothing to scroll.
    const wide = syntheticTrace(200, 5);
    render(
      <SplitHeatmap
        trace={wide}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
      />,
    );
    const scrolls = screen.getAllByTestId('token-heatmap-scroll');
    expect(scrolls).toHaveLength(2);
    const [leftScroll, rightScroll] = scrolls;

    // Simulate the user scrolling the left pane. In jsdom we set scrollLeft
    // directly and dispatch a scroll event so the onScroll handler fires.
    Object.defineProperty(leftScroll, 'scrollLeft', {
      configurable: true,
      get() {
        return this._sl ?? 0;
      },
      set(v: number) {
        this._sl = v;
      },
    });
    Object.defineProperty(rightScroll, 'scrollLeft', {
      configurable: true,
      get() {
        return this._sl ?? 0;
      },
      set(v: number) {
        this._sl = v;
      },
    });

    (leftScroll as unknown as { scrollLeft: number }).scrollLeft = 75;
    fireEvent.scroll(leftScroll, { target: { scrollLeft: 75 } });

    // SplitHeatmap should propagate the new scrollLeft back into the right
    // pane via its `scrollLeft` prop and the useLayoutEffect sync.
    expect(
      (rightScroll as unknown as { scrollLeft: number }).scrollLeft,
    ).toBe(75);
  });
});
