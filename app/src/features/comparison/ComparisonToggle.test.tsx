import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ComparisonToggle, type ComparisonMode } from './ComparisonToggle';

describe('ComparisonToggle', () => {
  it('renders three options with accessible names', () => {
    render(<ComparisonToggle value="raw" onChange={vi.fn()} />);
    expect(screen.getByLabelText('Raw')).toBeInTheDocument();
    expect(screen.getByLabelText('Processed')).toBeInTheDocument();
    expect(screen.getByLabelText('Split')).toBeInTheDocument();
  });

  it('marks the current mode as checked', () => {
    render(<ComparisonToggle value="processed" onChange={vi.fn()} />);
    expect(screen.getByLabelText('Raw')).not.toBeChecked();
    expect(screen.getByLabelText('Processed')).toBeChecked();
    expect(screen.getByLabelText('Split')).not.toBeChecked();
  });

  // Starting from `split` so each click is a genuine selection change — a
  // radio's onChange does not fire when clicking an already-checked option.
  it.each<{ mode: ComparisonMode; testId: string }>([
    { mode: 'raw', testId: 'comparison-toggle-raw' },
    { mode: 'processed', testId: 'comparison-toggle-processed' },
  ])('fires onChange with $mode when its option is clicked', async ({ mode, testId }) => {
    const onChange = vi.fn();
    render(<ComparisonToggle value="split" onChange={onChange} />);
    const input = screen.getByTestId(testId);
    await userEvent.click(input);
    expect(onChange).toHaveBeenCalledWith(mode);
  });

  it('fires onChange with split when split is clicked from another mode', async () => {
    const onChange = vi.fn();
    render(<ComparisonToggle value="raw" onChange={onChange} />);
    await userEvent.click(screen.getByTestId('comparison-toggle-split'));
    expect(onChange).toHaveBeenCalledWith('split');
  });

  it('exposes a radiogroup role with the legend as accessible name', () => {
    render(<ComparisonToggle value="raw" onChange={vi.fn()} label="Distribution" />);
    const group = screen.getByRole('radiogroup', { name: 'Distribution' });
    expect(group).toBeInTheDocument();
  });
});
