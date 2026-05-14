import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { LoadingState } from '@/components/feedback/LoadingState';

describe('LoadingState', () => {
  it('exposes the label to assistive tech via role=status', () => {
    render(<LoadingState label="Loading sample" />);
    expect(screen.getByRole('status')).toHaveAttribute(
      'aria-label',
      'Loading sample',
    );
  });
});
