import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ErrorState } from '@/components/feedback/ErrorState';

describe('ErrorState', () => {
  it('renders the message and calls onRetry / onReset', async () => {
    const onRetry = vi.fn();
    const onReset = vi.fn();
    render(
      <ErrorState
        message="something broke"
        onRetry={onRetry}
        onReset={onReset}
      />,
    );
    expect(screen.getByText(/something broke/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /retry/i }));
    await userEvent.click(screen.getByRole('button', { name: /reset/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it('hides retry/reset buttons when no handlers are passed', () => {
    render(<ErrorState message="x" />);
    expect(screen.queryByRole('button', { name: /retry/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /reset/i })).toBeNull();
  });

  it('renders validation issues with their JSON pointers when provided', () => {
    render(
      <ErrorState
        title="Trace failed schema validation"
        message="The trace JSON did not match the expected shape."
        issues={[
          { pointer: '/steps/0/logit_lens/0/top_k', message: "must have required property 'top_k'" },
          { pointer: '/steps/0/logit_lens/0', message: 'must NOT have additional properties' },
        ]}
      />,
    );
    expect(screen.getByText(/show 2 validation issues/i)).toBeInTheDocument();
    expect(screen.getByText('/steps/0/logit_lens/0/top_k')).toBeInTheDocument();
    expect(screen.getByText(/must have required property 'top_k'/i)).toBeInTheDocument();
    expect(screen.getByText('/steps/0/logit_lens/0')).toBeInTheDocument();
    expect(screen.getByText(/must NOT have additional properties/i)).toBeInTheDocument();
  });

  it('omits the issues block when none are passed', () => {
    render(<ErrorState message="x" />);
    expect(screen.queryByText(/validation issues?/i)).toBeNull();
  });
});
