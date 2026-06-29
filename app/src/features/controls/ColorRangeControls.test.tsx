import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ColorRangeControls, type ColorRangeValue } from './ColorRangeControls';

const autoValue: ColorRangeValue = { mode: 'auto', min: null, max: null };
const manualValue: ColorRangeValue = { mode: 'manual', min: -5, max: 0 };

describe('ColorRangeControls', () => {
  it('renders auto and manual radios with accessible names', () => {
    render(
      <ColorRangeControls
        value={autoValue}
        onChange={vi.fn()}
        autoMin={-10}
        autoMax={0}
      />,
    );
    expect(screen.getByLabelText('Auto')).toBeChecked();
    expect(screen.getByLabelText('Manual')).not.toBeChecked();
  });

  it('seeds manual mode from the auto bounds and notifies onChange', async () => {
    const onChange = vi.fn();
    render(
      <ColorRangeControls
        value={autoValue}
        onChange={onChange}
        autoMin={-4}
        autoMax={2}
      />,
    );
    await userEvent.click(screen.getByLabelText('Manual'));
    expect(onChange).toHaveBeenCalledWith({
      mode: 'manual',
      min: -4,
      max: 2,
    });
  });

  it('disables the numeric inputs in auto mode', () => {
    render(
      <ColorRangeControls
        value={autoValue}
        onChange={vi.fn()}
        autoMin={-1}
        autoMax={1}
      />,
    );
    expect(screen.getByTestId('color-range-min')).toBeDisabled();
    expect(screen.getByTestId('color-range-max')).toBeDisabled();
  });

  it('shows an inline error and suppresses onChange when min > max', () => {
    const onChange = vi.fn();
    render(
      <ColorRangeControls
        value={manualValue}
        onChange={onChange}
        autoMin={-10}
        autoMax={0}
      />,
    );
    onChange.mockClear();
    fireEvent.change(screen.getByTestId('color-range-min'), {
      target: { value: '5' },
    });
    expect(screen.getByTestId('color-range-error')).toHaveTextContent(
      /min must be ≤ max/i,
    );
    expect(onChange).not.toHaveBeenCalled();
  });

  it('shows an inline error and suppresses onChange when the input is not a number', () => {
    const onChange = vi.fn();
    render(
      <ColorRangeControls
        value={manualValue}
        onChange={onChange}
        autoMin={-10}
        autoMax={0}
      />,
    );
    onChange.mockClear();
    fireEvent.change(screen.getByTestId('color-range-min'), {
      target: { value: 'abc' },
    });
    expect(screen.getByTestId('color-range-error')).toHaveTextContent(
      /min must be a number/i,
    );
    expect(onChange).not.toHaveBeenCalled();
  });

  it('fires onChange with the new range for valid edits', () => {
    const onChange = vi.fn();
    render(
      <ColorRangeControls
        value={manualValue}
        onChange={onChange}
        autoMin={-10}
        autoMax={0}
      />,
    );
    onChange.mockClear();
    fireEvent.change(screen.getByTestId('color-range-max'), {
      target: { value: '1' },
    });
    expect(onChange).toHaveBeenCalledWith({
      mode: 'manual',
      min: -5,
      max: 1,
    });
  });
});
