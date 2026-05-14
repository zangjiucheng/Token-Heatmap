import { useCallback, useEffect, useRef, useState } from 'react';

import { getApiClient } from '@/api/client';

export type BackendHealthStatus = 'unknown' | 'healthy' | 'unhealthy';

export interface UseBackendHealthOptions {
  /** Polling interval in milliseconds. Set to 0 to disable polling. */
  intervalMs?: number;
  /**
   * Number of consecutive failed probes required before transitioning to
   * `unhealthy`. Debounces transient network blips.
   */
  failureThreshold?: number;
}

export interface UseBackendHealthResult {
  status: BackendHealthStatus;
  /** Manually trigger a probe. Useful for "Retry" buttons. */
  probe: () => Promise<void>;
}

/**
 * Polls `GET /health` on a configurable cadence and exposes a debounced
 * status. Failures within the threshold are not reported so a single
 * dropped packet doesn't flash a "backend down" banner.
 */
export function useBackendHealth(
  options: UseBackendHealthOptions = {},
): UseBackendHealthResult {
  const { intervalMs = 30_000, failureThreshold = 2 } = options;
  const [status, setStatus] = useState<BackendHealthStatus>('unknown');
  const failureCount = useRef(0);
  const cancelled = useRef(false);

  const probe = useCallback(async (): Promise<void> => {
    try {
      const ok = await getApiClient().health();
      if (cancelled.current) return;
      if (ok) {
        failureCount.current = 0;
        setStatus('healthy');
      } else {
        failureCount.current += 1;
        if (failureCount.current >= failureThreshold) {
          setStatus('unhealthy');
        }
      }
    } catch {
      if (cancelled.current) return;
      failureCount.current += 1;
      if (failureCount.current >= failureThreshold) {
        setStatus('unhealthy');
      }
    }
  }, [failureThreshold]);

  useEffect(() => {
    cancelled.current = false;
    void probe();
    if (intervalMs <= 0) {
      return () => {
        cancelled.current = true;
      };
    }
    const id = setInterval(() => {
      void probe();
    }, intervalMs);
    return () => {
      cancelled.current = true;
      clearInterval(id);
    };
  }, [intervalMs, probe]);

  return { status, probe };
}
