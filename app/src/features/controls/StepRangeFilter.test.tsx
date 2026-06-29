import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { StepRangeFilter } from './StepRangeFilter';

describe('StepRangeFilter', () => {
  it('renders with accessible labels and shows the current range readout', () => {
    render(
      <StepRangeFilter
        min={0}
        max={9}
        value={[2, 6]}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByLabelText('Step range start')).toBeInTheDocument();
    expect(screen.getByLabelText('Step range end')).toBeInTheDocument();
    expect(screen.getByTestId('step-range-filter-readout')).toHaveTextContent(
      '2 – 6',
    );
  });

  it('clamps the start thumb to <= end', () => {
    const onChange = vi.fn();
    render(
      <StepRangeFilter min={0} max={9} value={[2, 6]} onChange={onChange} />,
    );
    const start = screen.getByLabelText('Step range start');
    fireEvent.change(start, { target: { value: '8' } });
    expect(onChange).toHaveBeenCalledWith([6, 6]);
  });

  it('clamps the end thumb to >= start', () => {
    const onChange = vi.fn();
    render(
      <StepRangeFilter min={0} max={9} value={[2, 6]} onChange={onChange} />,
    );
    const end = screen.getByLabelText('Step range end');
    fireEvent.change(end, { target: { value: '0' } });
    expect(onChange).toHaveBeenCalledWith([2, 2]);
  });

  it('clamps inputs to the trace bounds', () => {
    const onChange = vi.fn();
    render(
      <StepRangeFilter min={0} max={9} value={[2, 6]} onChange={onChange} />,
    );
    fireEvent.change(screen.getByLabelText('Step range end'), {
      target: { value: '999' },
    });
    expect(onChange).toHaveBeenCalledWith([2, 9]);
  });

  it('supports keyboard arrow nudges via the native range input', () => {
    const onChange = vi.fn();
    render(
      <StepRangeFilter min={0} max={9} value={[2, 6]} onChange={onChange} />,
    );
    const start = screen.getByLabelText('Step range start') as HTMLInputElement;
    // ArrowRight on a native <input type=range step=1> dispatches a change to
    // value + 1. jsdom does not implement this natively, so we simulate the
    // resulting change event directly.
    fireEvent.change(start, { target: { value: '3' } });
    expect(onChange).toHaveBeenLastCalledWith([3, 6]);
  });
});
