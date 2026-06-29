import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { EmptyState } from '@/components/feedback/EmptyState';

describe('EmptyState', () => {
  it('invokes onLoadSample when the sample button is clicked', async () => {
    const onLoadSample = vi.fn();
    render(<EmptyState onLoadSample={onLoadSample} />);
    await userEvent.click(
      screen.getByRole('button', { name: /try sample data/i }),
    );
    expect(onLoadSample).toHaveBeenCalledTimes(1);
  });

  it('renders the heading and description', () => {
    render(<EmptyState onLoadSample={() => undefined} />);
    expect(
      screen.getByRole('heading', { name: /no trace loaded/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/drop a json trace file/i)).toBeInTheDocument();
  });

  it('offers only local loading — no remote URL field', () => {
    render(<EmptyState onLoadSample={() => undefined} />);
    expect(screen.queryByLabelText(/trace url/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /^load$/i }),
    ).not.toBeInTheDocument();
  });
});
