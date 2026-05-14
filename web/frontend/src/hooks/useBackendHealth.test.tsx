import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiClient, setApiClientForTests } from '@/api/client';
import { useBackendHealth } from './useBackendHealth';

function ok() {
  return new Response(JSON.stringify({ status: 'ok' }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}
function down() {
  return new Response('', { status: 503 });
}

describe('useBackendHealth', () => {
  afterEach(() => {
    setApiClientForTests(null);
    vi.useRealTimers();
  });

  it('transitions to healthy on the first ok probe', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(ok());
    setApiClientForTests(new ApiClient({ baseUrl: 'http://api', fetchImpl }));
    const { result } = renderHook(() => useBackendHealth({ intervalMs: 0 }));
    await waitFor(() => expect(result.current.status).toBe('healthy'));
  });

  it('debounces a single failure before flipping to unhealthy', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError('network'));
    setApiClientForTests(new ApiClient({ baseUrl: 'http://api', fetchImpl }));
    const { result } = renderHook(() =>
      useBackendHealth({ intervalMs: 0, failureThreshold: 2 }),
    );
    await waitFor(() => expect(fetchImpl).toHaveBeenCalled());
    // One failed probe is not enough.
    expect(result.current.status).not.toBe('unhealthy');
    await act(async () => {
      await result.current.probe();
    });
    await waitFor(() => expect(result.current.status).toBe('unhealthy'));
  });

  it('recovers to healthy after a successful probe', async () => {
    let calls = 0;
    const fetchImpl = vi.fn(async () => {
      calls += 1;
      if (calls <= 2) throw new TypeError('network');
      return ok();
    });
    setApiClientForTests(
      new ApiClient({ baseUrl: 'http://api', fetchImpl: fetchImpl as unknown as typeof fetch }),
    );
    const { result } = renderHook(() =>
      useBackendHealth({ intervalMs: 0, failureThreshold: 2 }),
    );
    await act(async () => {
      await result.current.probe();
    });
    await waitFor(() => expect(result.current.status).toBe('unhealthy'));
    await act(async () => {
      await result.current.probe();
    });
    await waitFor(() => expect(result.current.status).toBe('healthy'));
  });

  it('polls on the configured interval', async () => {
    vi.useFakeTimers();
    const fetchImpl = vi.fn().mockResolvedValue(ok());
    setApiClientForTests(new ApiClient({ baseUrl: 'http://api', fetchImpl }));
    renderHook(() => useBackendHealth({ intervalMs: 1000 }));
    // Initial probe fires immediately.
    await vi.waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(1));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500);
    });
    expect(fetchImpl.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it('uses down responses (non-2xx) as failures', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(down());
    setApiClientForTests(new ApiClient({ baseUrl: 'http://api', fetchImpl }));
    const { result } = renderHook(() =>
      useBackendHealth({ intervalMs: 0, failureThreshold: 1 }),
    );
    await waitFor(() => expect(result.current.status).toBe('unhealthy'));
  });
});
