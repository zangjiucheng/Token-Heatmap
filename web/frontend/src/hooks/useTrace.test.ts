import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { useTrace } from './useTrace';

describe('useTrace', () => {
  it('starts idle', () => {
    const { result } = renderHook(() => useTrace());
    expect(result.current.status).toBe('idle');
    expect(result.current.trace).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('useTrace_transitions: idle -> loading -> ready on successful load', async () => {
    const { result } = renderHook(() => useTrace());
    const states: string[] = [result.current.status];

    let pending: Promise<void>;
    act(() => {
      pending = result.current.load({ type: 'sample' });
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
    const { result } = renderHook(() => useTrace());
    let pending: Promise<void>;
    act(() => {
      // A missing cached id is the simplest loader failure now that there is
      // no network source.
      pending = result.current.load({ type: 'cached', id: 'missing' });
    });

    await act(async () => {
      await pending;
    });

    expect(result.current.status).toBe('error');
    expect(result.current.trace).toBeNull();
    expect(result.current.error?.kind).toBe('network');
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
