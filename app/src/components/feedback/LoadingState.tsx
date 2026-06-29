import './LoadingState.css';

export interface LoadingStateProps {
  label?: string;
  rows?: number;
}

export function LoadingState({
  label = 'Loading…',
  rows = 6,
}: LoadingStateProps) {
  return (
    <div
      className="loading-state"
      role="status"
      aria-live="polite"
      aria-label={label}
    >
      <span className="loading-state__sr-only">{label}</span>
      <div className="loading-state__skeleton-grid" aria-hidden="true">
        {Array.from({ length: rows }, (_, i) => (
          <div key={i} className="loading-state__skeleton-row" />
        ))}
      </div>
    </div>
  );
}
