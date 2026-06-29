import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { SelectedProbabilityTimeline } from './SelectedProbabilityTimeline';

const trace = sampleTrace as unknown as Trace;
const WIDTH = 400;
const HEIGHT = 120;
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

describe('SelectedProbabilityTimeline', () => {
  it('renders empty state when trace is null', () => {
    render(
      <SelectedProbabilityTimeline
        trace={null}
        selectedStep={null}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    expect(
      screen.getByTestId('selected-probability-timeline'),
    ).toHaveTextContent(/No trace loaded/i);
  });

  it('renders a line for the sample trace', () => {
    render(
      <SelectedProbabilityTimeline
        trace={trace}
        selectedStep={null}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    expect(
      screen.getByTestId('selected-probability-timeline-line'),
    ).toBeInTheDocument();
  });

  it('calls onSelectStep with the closest x when clicked', () => {
    const onSelectStep = vi.fn();
    render(
      <SelectedProbabilityTimeline
        trace={trace}
        selectedStep={null}
        hoveredStep={null}
        onSelectStep={onSelectStep}
        onHoverStep={vi.fn()}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    const container = screen.getByTestId('selected-probability-timeline');
    mockRect(container);
    const totalSteps = trace.steps.length;
    const plotWidth = WIDTH - PADDING_LEFT - PADDING_RIGHT;
    const stepWidth = plotWidth / totalSteps;
    const target = Math.min(1, totalSteps - 1);
    const x = PADDING_LEFT + stepWidth * target + stepWidth / 2;

    fireEvent.click(container, { clientX: x, clientY: 40 });
    expect(onSelectStep).toHaveBeenCalledWith(target);
  });

  it('reports hovered step via onHoverStep on mouse move', () => {
    const onHoverStep = vi.fn();
    render(
      <SelectedProbabilityTimeline
        trace={trace}
        selectedStep={null}
        hoveredStep={null}
        onSelectStep={vi.fn()}
        onHoverStep={onHoverStep}
        width={WIDTH}
        height={HEIGHT}
      />,
    );
    const container = screen.getByTestId('selected-probability-timeline');
    mockRect(container);
    const totalSteps = trace.steps.length;
    const plotWidth = WIDTH - PADDING_LEFT - PADDING_RIGHT;
    const stepWidth = plotWidth / totalSteps;
    const target = Math.min(2, totalSteps - 1);
    const x = PADDING_LEFT + stepWidth * target + stepWidth / 2;
    fireEvent.mouseMove(container, { clientX: x, clientY: 40 });
    expect(onHoverStep).toHaveBeenLastCalledWith(target);
  });
});
