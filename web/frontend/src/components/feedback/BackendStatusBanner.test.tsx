import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { BackendStatusBanner } from './BackendStatusBanner';

describe('BackendStatusBanner', () => {
  it('renders nothing when status is healthy', () => {
    const { container } = render(<BackendStatusBanner status="healthy" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when status is unknown', () => {
    const { container } = render(<BackendStatusBanner status="unknown" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the banner when status is unhealthy', () => {
    render(<BackendStatusBanner status="unhealthy" />);
    expect(screen.getByText(/backend unreachable/i)).toBeInTheDocument();
  });

  it('invokes onRetry when the retry button is clicked', async () => {
    const onRetry = vi.fn();
    render(<BackendStatusBanner status="unhealthy" onRetry={onRetry} />);
    await userEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
