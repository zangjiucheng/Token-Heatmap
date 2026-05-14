import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { EntropyTimeline } from './EntropyTimeline';

const trace = sampleTrace as unknown as Trace;
const WIDTH = 400;
const HEIGHT = 120;
// Must match constants in TimelineChart.tsx.
const PADDING_LEFT = 32;
const PADDING_RIGHT = 8;

function mockRect(el: HTMLElement) {
  el.getBoundingClientRect = () =>
    ({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: WIDTH,
      bottom: HEIGHT,
      width: WIDTH,
      height: HEIGHT,
      toJSON: () => ({}),
    }) as DOMRect;
}

describe('EntropyTimeline', () => {
  it('renders empty state when trace is null', () => {
    render(
      <EntropyTimeline
        trace={null}
        selectedStep={null}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    expect(screen.getByTestId('entropy-timeline')).toHaveTextContent(
      /No trace loaded/i,
    );
  });

  it('renders one point per step', () => {
    render(
      <EntropyTimeline
        trace={trace}
        selectedStep={1}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    expect(screen.getByTestId('entropy-timeline-line')).toBeInTheDocument();
    expect(
      screen.getByTestId('entropy-timeline-selected-point'),
    ).toBeInTheDocument();
  });

  it('calls onSelectStep with the closest x when clicked', () => {
    const onSelectStep = vi.fn();
    render(
      <EntropyTimeline
        trace={trace}
        selectedStep={null}
        hoveredStep={null}
        onSelectStep={onSelectStep}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    const container = screen.getByTestId('entropy-timeline');
    mockRect(container);
    const totalSteps = trace.steps.length;
    const plotWidth = WIDTH - PADDING_LEFT - PADDING_RIGHT;
    const stepWidth = plotWidth / totalSteps;
    const target = Math.min(2, totalSteps - 1);
    const x = PADDING_LEFT + stepWidth * target + stepWidth / 2;

    fireEvent.click(container, { clientX: x, clientY: 40 });
    expect(onSelectStep).toHaveBeenCalledWith(target);
  });

  it('moves selectedStep on arrow key when focused', () => {
    const onSelectStep = vi.fn();
    render(
      <EntropyTimeline
        trace={trace}
        selectedStep={1}
        hoveredStep={null}
        onSelectStep={onSelectStep}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    const container = screen.getByTestId('entropy-timeline');
    container.focus();
    fireEvent.keyDown(container, { key: 'ArrowRight' });
    expect(onSelectStep).toHaveBeenCalledWith(2);
    fireEvent.keyDown(container, { key: 'ArrowLeft' });
    expect(onSelectStep).toHaveBeenCalledWith(0);
  });

  it('moves the cursor when selectedStep prop changes', () => {
    const { rerender } = render(
      <EntropyTimeline
        trace={trace}
        selectedStep={0}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    const x1 = screen
      .getByTestId('entropy-timeline-cursor')
      .getAttribute('x1');
    rerender(
      <EntropyTimeline
        trace={trace}
        selectedStep={trace.steps.length - 1}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    const x2 = screen
      .getByTestId('entropy-timeline-cursor')
      .getAttribute('x1');
    expect(x1).not.toBe(x2);
  });
});
