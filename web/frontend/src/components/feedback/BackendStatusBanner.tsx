import type { BackendHealthStatus } from '@/hooks/useBackendHealth';
import './BackendStatusBanner.css';

export interface BackendStatusBannerProps {
  status: BackendHealthStatus;
  onRetry?: () => void;
}

/**
 * Renders a dismissible banner when the backend is unreachable so users
 * know the "Generate" path is offline and can fall back to file upload.
 */
export function BackendStatusBanner({ status, onRetry }: BackendStatusBannerProps) {
  if (status !== 'unhealthy') return null;
  return (
    <div className="backend-status-banner" role="alert">
      <div className="backend-status-banner__copy">
        <strong>Backend unreachable.</strong> Trace generation is unavailable.
        Use the file upload or sample data instead.
      </div>
      {onRetry && (
        <button
          type="button"
          className="backend-status-banner__retry"
          onClick={onRetry}
        >
          Retry
        </button>
      )}
    </div>
  );
}
