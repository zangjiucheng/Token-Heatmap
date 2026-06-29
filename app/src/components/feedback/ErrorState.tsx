import type { ValidationIssue } from '@/lib/trace/errors';
import './ErrorState.css';

export interface ErrorStateProps {
  title?: string;
  message: string;
  issues?: ValidationIssue[];
  onRetry?: () => void;
  onReset?: () => void;
}

export function ErrorState({
  title = 'Something went wrong',
  message,
  issues,
  onRetry,
  onReset,
}: ErrorStateProps) {
  const hasIssues = !!issues && issues.length > 0;
  return (
    <section className="error-state" role="alert" aria-live="assertive">
      <h2 className="error-state__title">{title}</h2>
      <p className="error-state__message">{message}</p>
      {hasIssues && (
        <details className="error-state__issues">
          <summary className="error-state__issues-summary">
            {`Show ${issues.length} validation ${issues.length === 1 ? 'issue' : 'issues'}`}
          </summary>
          <ul className="error-state__issues-list">
            {issues.map((issue, idx) => (
              <li key={`${issue.pointer}-${idx}`} className="error-state__issue">
                <code className="error-state__issue-pointer">{issue.pointer}</code>
                <span className="error-state__issue-message">{issue.message}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <div className="error-state__actions">
        {onRetry && (
          <button
            type="button"
            className="error-state__button error-state__button--primary"
            onClick={onRetry}
          >
            Retry
          </button>
        )}
        {onReset && (
          <button
            type="button"
            className="error-state__button"
            onClick={onReset}
          >
            Reset
          </button>
        )}
      </div>
    </section>
  );
}
