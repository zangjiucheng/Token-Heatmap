import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TimelineCursor } from './TimelineCursor';

function renderInSvg(node: React.ReactNode) {
  return render(<svg>{node}</svg>);
}

describe('TimelineCursor', () => {
  it('renders nothing when step is null', () => {
    const { queryByTestId } = renderInSvg(
      <TimelineCursor
        step={null}
        totalSteps={10}
        plotWidth={100}
        plotHeight={50}
      />,
    );
    expect(queryByTestId('timeline-cursor')).toBeNull();
  });

  it('renders nothing when totalSteps is 0', () => {
    const { queryByTestId } = renderInSvg(
      <TimelineCursor
        step={0}
        totalSteps={0}
        plotWidth={100}
        plotHeight={50}
      />,
    );
    expect(queryByTestId('timeline-cursor')).toBeNull();
  });

  it('positions x at the centre of the requested step', () => {
    const { getByTestId } = renderInSvg(
      <TimelineCursor
        step={2}
        totalSteps={10}
        plotWidth={100}
        plotHeight={50}
        plotX={0}
      />,
    );
    const line = getByTestId('timeline-cursor');
    // stepWidth = 10; centre of step 2 = 25
    expect(line.getAttribute('x1')).toBe('25');
    expect(line.getAttribute('x2')).toBe('25');
    expect(line.getAttribute('data-step')).toBe('2');
  });

  it('moves x when step prop changes', () => {
    const { rerender, getByTestId } = renderInSvg(
      <TimelineCursor
        step={1}
        totalSteps={10}
        plotWidth={100}
        plotHeight={50}
        plotX={0}
      />,
    );
    const x1 = getByTestId('timeline-cursor').getAttribute('x1');
    rerender(
      <svg>
        <TimelineCursor
          step={7}
          totalSteps={10}
          plotWidth={100}
          plotHeight={50}
          plotX={0}
        />
      </svg>,
    );
    const x2 = getByTestId('timeline-cursor').getAttribute('x1');
    expect(x1).not.toBe(x2);
    expect(x2).toBe('75');
  });

  it('uses dashed stroke for hover variant', () => {
    const { getByTestId } = renderInSvg(
      <TimelineCursor
        step={3}
        totalSteps={10}
        plotWidth={100}
        plotHeight={50}
        variant="hover"
      />,
    );
    expect(getByTestId('timeline-cursor').getAttribute('stroke-dasharray')).toBe(
      '4 3',
    );
  });
});
