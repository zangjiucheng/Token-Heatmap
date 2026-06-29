import { render, screen, fireEvent } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ErrorBoundary } from './ErrorBoundary';

function Boom({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error('Boom!');
  return <div data-testid="child">child rendered</div>;
}

describe('ErrorBoundary', () => {
  let originalConsoleError: typeof console.error;

  beforeEach(() => {
    // Silence React's error logging during these tests.
    originalConsoleError = console.error;
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <Boom shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it('renders the fallback with a Reload button when a child throws', () => {
    render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^reload$/i }),
    ).toBeInTheDocument();
    expect(screen.getByText('Boom!')).toBeInTheDocument();
  });

  it('calls location.reload when Reload is clicked', () => {
    const reloadSpy = vi.fn();
    Object.defineProperty(window, 'location', {
      writable: true,
      value: { ...window.location, reload: reloadSpy },
    });

    render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>,
    );

    fireEvent.click(screen.getByTestId('error-boundary-reload'));
    expect(reloadSpy).toHaveBeenCalledTimes(1);
  });

  it('lets the user dismiss the error and re-render children', () => {
    function Wrapper() {
      return (
        <ErrorBoundary>
          <Boom shouldThrow={false} />
        </ErrorBoundary>
      );
    }
    const { rerender } = render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();

    // Click dismiss, then re-render with non-throwing child.
    fireEvent.click(screen.getByTestId('error-boundary-dismiss'));
    rerender(<Wrapper />);
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });
});
