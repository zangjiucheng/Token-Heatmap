import { Component, type ErrorInfo, type ReactNode } from 'react';
import './ErrorBoundary.css';

export interface ErrorBoundaryProps {
  children: ReactNode;
  /** Optional custom fallback renderer. */
  fallback?: (
    error: Error,
    reset: () => void,
    reload: () => void,
  ) => ReactNode;
  /** Notified whenever a child throws — useful for telemetry hooks. */
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.props.onError?.(error, info);
    // eslint-disable-next-line no-console -- surface for diagnostics
    console.error('ErrorBoundary caught', error, info);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  reload = (): void => {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) {
      return this.props.fallback(error, this.reset, this.reload);
    }

    return (
      <section
        className="error-boundary"
        role="alert"
        aria-live="assertive"
        data-testid="error-boundary-fallback"
      >
        <h2 className="error-boundary__title">Something went wrong</h2>
        <p className="error-boundary__message">
          The view failed to render. You can reload the app or dismiss the
          error and continue.
        </p>
        <pre className="error-boundary__detail" aria-label="Error detail">
          {error.message}
        </pre>
        <div className="error-boundary__actions">
          <button
            type="button"
            className="error-boundary__button error-boundary__button--primary"
            onClick={this.reload}
            data-testid="error-boundary-reload"
          >
            Reload
          </button>
          <button
            type="button"
            className="error-boundary__button"
            onClick={this.reset}
            data-testid="error-boundary-dismiss"
          >
            Dismiss
          </button>
        </div>
      </section>
    );
  }
}
