import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import sampleTrace from '@/lib/sample/trace.json';
import { useTrace } from './useTrace';

describe('useTrace', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('starts idle', () => {
    const { result } = renderHook(() => useTrace());
    expect(result.current.status).toBe('idle');
    expect(result.current.trace).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('useTrace_transitions: idle -> loading -> ready on successful load', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(
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
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(new Response('boom', { status: 500 }));

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
});
