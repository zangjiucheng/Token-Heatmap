import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import sampleTrace from '@/lib/sample/trace.json';
import { ApiClient, setApiClientForTests } from '@/api/client';
import { useTrace } from './useTrace';

describe('useTrace', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setApiClientForTests(null);
  });

  it('starts idle', () => {
    const { result } = renderHook(() => useTrace());
    expect(result.current.status).toBe('idle');
    expect(result.current.trace).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('useTrace_transitions: idle -> loading -> ready on successful load', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(sampleTrace), { status: 200 }),
    );

    const { result } = renderHook(() => useTrace());
    const states: string[] = [result.current.status];

    let pending: Promise<void>;
    act(() => {
      pending = result.current.load({ type: 'url', url: '/sample.json' });
    });

    await waitFor(() => {
      if (result.current.status !== states[states.length - 1]) {
        states.push(result.current.status);
      }
      expect(result.current.status).toBe('ready');
    });
    await act(async () => {
      await pending;
    });

    expect(result.current.trace?.steps).toHaveLength(5);
    expect(result.current.error).toBeNull();
    expect(states).toContain('loading');
  });

  it('useTrace_transitions: idle -> loading -> error on failure', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('boom', { status: 500 }),
    );

    const { result } = renderHook(() => useTrace());
    let pending: Promise<void>;
    act(() => {
      pending = result.current.load({ type: 'url', url: '/sample.json' });
    });

    await act(async () => {
      await pending;
    });

    expect(result.current.status).toBe('error');
    expect(result.current.trace).toBeNull();
    expect(result.current.error?.kind).toBe('network');
    expect(result.current.error?.status).toBe(500);
  });

  it('load with sample source transitions to ready', async () => {
    const { result } = renderHook(() => useTrace());
    await act(async () => {
      await result.current.load({ type: 'sample' });
    });
    expect(result.current.status).toBe('ready');
    expect(result.current.trace?.schema_version).toBe('2.0.0');
  });

  it('load with generate source POSTs to /trace/generate and transitions to ready', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(sampleTrace), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    setApiClientForTests(new ApiClient({ baseUrl: 'http://api', fetchImpl }));

    const { result } = renderHook(() => useTrace());
    await act(async () => {
      await result.current.load({
        type: 'generate',
        params: {
          model: 'tiny',
          prompt: 'hello',
          max_new_tokens: 8,
          temperature: 0.8,
          top_p: 0.95,
          min_k: 8,
          max_k: 64,
          mass_threshold: 0.95,
          capture_attention: false,
          capture_logit_lens: false,
          capture_activations: false,
        },
      });
    });

    expect(result.current.status).toBe('ready');
    expect(result.current.trace?.steps).toHaveLength(5);
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api/trace/generate');
    expect(init.method).toBe('POST');
  });
});
