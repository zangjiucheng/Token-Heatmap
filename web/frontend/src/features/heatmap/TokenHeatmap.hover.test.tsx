import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { TokenHeatmap, AXIS_GUTTER_X } from './TokenHeatmap';

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

// Sample trace = 5 steps × 5 ranks. With width=800/height=400 and default
// min/max sizing, cellW clamps to maxCellWidth (48) and cellH to maxCellHeight (32).
const STEPS = trace.steps.length;
const CELL_W = 48;
const CELL_H = 32;
const CONTENT_W = STEPS * CELL_W;

function setupRects() {
  const plot = screen.getByTestId('token-heatmap-plot');
  patchRect(plot, 0, 0, 800, 400);
  const canvas = screen.getByTestId('token-heatmap-canvas');
  patchRect(canvas, AXIS_GUTTER_X, 0, CONTENT_W, 400);
  return { plot, canvas };
}

describe('TokenHeatmap hover', () => {
  it('surfaces a tooltip with the expected fields and HTML-safe token text', () => {
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

    const { plot } = setupRects();

    // Hover the (step 0, rank 0) cell — that's the selected token at step 0.
    const x = AXIS_GUTTER_X + CELL_W / 2;
    const y = CELL_H / 2;
    fireEvent.mouseMove(plot, { clientX: x, clientY: y });

    const tooltip = screen.getByTestId('heatmap-tooltip');
    expect(tooltip).toBeInTheDocument();
    expect(screen.getByTestId('heatmap-tooltip-step')).toHaveTextContent('0');
    expect(screen.getByTestId('heatmap-tooltip-rank')).toHaveTextContent('1');
    expect(screen.getByTestId('heatmap-tooltip-kused')).toHaveTextContent(
      String(trace.steps[0].processed.k_used),
    );
    expect(screen.getByTestId('heatmap-tooltip-prob')).toBeInTheDocument();
    expect(screen.getByTestId('heatmap-tooltip-logprob')).toBeInTheDocument();
    expect(screen.getByTestId('heatmap-tooltip-entropy')).toBeInTheDocument();

    const tokenEl = screen.getByTestId('heatmap-tooltip-token');
    expect(tokenEl.querySelector('script')).toBeNull();
    expect(tokenEl.innerHTML).not.toMatch(/<script/i);
  });

  it('renders an HTML-unsafe token verbatim as text', () => {
    const malicious: Trace = JSON.parse(JSON.stringify(trace));
    malicious.steps[0].processed.candidates[0].token = '<script>x</script>';

    render(
      <TokenHeatmap
        trace={malicious}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={vi.fn()}
        width={800}
        height={400}
      />,
    );

    const { plot } = setupRects();
    fireEvent.mouseMove(plot, {
      clientX: AXIS_GUTTER_X + CELL_W / 2,
      clientY: CELL_H / 2,
    });

    const tokenEl = screen.getByTestId('heatmap-tooltip-token');
    expect(tokenEl.textContent).toBe('<script>x</script>');
    expect(tokenEl.querySelector('script')).toBeNull();
  });

  it('dismisses the tooltip when Escape is pressed', () => {
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

    const { plot } = setupRects();
    fireEvent.mouseMove(plot, { clientX: AXIS_GUTTER_X + CELL_W / 2, clientY: 20 });
    expect(screen.getByTestId('heatmap-tooltip')).toBeInTheDocument();
    fireEvent.keyDown(plot, { key: 'Escape' });
    expect(screen.queryByTestId('heatmap-tooltip')).toBeNull();
  });

  it('test_hover_after_scroll_reports_correct_step', () => {
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
    patchRect(plot, 0, 0, 800, 400);

    // Simulate a scrolled state by patching the canvas's bounding rect.
    // Native horizontal scroll shifts the inner div left by `scrollLeft`,
    // which the browser reflects in the canvas's getBoundingClientRect.
    // The component's hit-testing reads that rect directly, so faking the
    // shifted left edge here is equivalent to a real scroll event in jsdom.
    const SCROLL = CELL_W * 2;
    const canvas = screen.getByTestId('token-heatmap-canvas');
    patchRect(canvas, AXIS_GUTTER_X - SCROLL, 0, CONTENT_W, 400);

    // A hover at screen-x = AXIS_GUTTER_X + CELL_W/2 now lands inside the
    // canvas at x = SCROLL + CELL_W/2 → column 2 → step 2.
    fireEvent.mouseMove(plot, {
      clientX: AXIS_GUTTER_X + CELL_W / 2,
      clientY: CELL_H / 2,
    });

    expect(screen.getByTestId('heatmap-tooltip-step')).toHaveTextContent('2');
  });

  it('clicking after scroll selects the correct step', () => {
    const onSelectStep = vi.fn();
    render(
      <TokenHeatmap
        trace={trace}
        valueCol="logprob"
        selectedStep={null}
        onSelectStep={onSelectStep}
        width={800}
        height={400}
      />,
    );

    const plot = screen.getByTestId('token-heatmap-plot');
    patchRect(plot, 0, 0, 800, 400);
    const SCROLL = CELL_W * 3;
    const canvas = screen.getByTestId('token-heatmap-canvas');
    patchRect(canvas, AXIS_GUTTER_X - SCROLL, 0, CONTENT_W, 400);

    fireEvent.click(plot, {
      clientX: AXIS_GUTTER_X + CELL_W / 2,
      clientY: CELL_H / 2,
    });
    expect(onSelectStep).toHaveBeenCalledWith(3);
  });

});
