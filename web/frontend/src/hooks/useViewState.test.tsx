import { describe, expect, it } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import { useViewState } from './useViewState';

function withRouter(initialEntries: string[] = ['/']) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <MemoryRouter initialEntries={initialEntries}>{children}</MemoryRouter>;
  };
}

function useViewStateWithLocation() {
  return { vs: useViewState(), location: useLocation() };
}

describe('useViewState', () => {
  it('returns defaults when no search params are present', () => {
    const { result } = renderHook(() => useViewState(), {
      wrapper: withRouter(['/trace']),
    });
    expect(result.current.state.mode).toBe('raw');
    expect(result.current.state.valueCol).toBe('logprob');
    expect(result.current.state.stepRange).toBeNull();
    expect(result.current.state.colorRange).toEqual({
      mode: 'auto',
      min: null,
      max: null,
    });
  });

  it('parses mode, valueCol, stepRange, and color bounds from the URL', () => {
    const { result } = renderHook(() => useViewState(), {
      wrapper: withRouter([
        '/trace?mode=split&valueCol=prob&stepRange=1-4&colorMin=-3&colorMax=0',
      ]),
    });
    expect(result.current.state.mode).toBe('split');
    expect(result.current.state.valueCol).toBe('prob');
    expect(result.current.state.stepRange).toEqual([1, 4]);
    expect(result.current.state.colorRange).toEqual({
      mode: 'manual',
      min: -3,
      max: 0,
    });
  });

  it('ignores invalid mode and stepRange values', () => {
    const { result } = renderHook(() => useViewState(), {
      wrapper: withRouter(['/trace?mode=bogus&stepRange=8-2']),
    });
    expect(result.current.state.mode).toBe('raw');
    expect(result.current.state.stepRange).toBeNull();
  });

  it('writes setMode / setValueCol / setStepRange / setColorRange into the URL', () => {
    const { result } = renderHook(useViewStateWithLocation, {
      wrapper: withRouter(['/trace']),
    });
    act(() => result.current.vs.setMode('split'));
    act(() => result.current.vs.setValueCol('prob'));
    act(() => result.current.vs.setStepRange([2, 5]));
    act(() =>
      result.current.vs.setColorRange({ mode: 'manual', min: -2, max: 1 }),
    );
    const search = result.current.location.search;
    expect(search).toContain('mode=split');
    expect(search).toContain('valueCol=prob');
    expect(search).toContain('stepRange=2-5');
    expect(search).toContain('colorMin=-2');
    expect(search).toContain('colorMax=1');
  });

  it('round-trips state through the URL: set then re-read produces the same state', () => {
    const { result, rerender } = renderHook(() => useViewState(), {
      wrapper: withRouter(['/trace']),
    });
    act(() => result.current.setMode('processed'));
    act(() => result.current.setStepRange([0, 3]));
    rerender();
    expect(result.current.state.mode).toBe('processed');
    expect(result.current.state.stepRange).toEqual([0, 3]);
  });

  it('omits default values from the URL to keep it tidy', () => {
    const { result } = renderHook(useViewStateWithLocation, {
      wrapper: withRouter(['/trace?mode=split']),
    });
    act(() => result.current.vs.setMode('raw'));
    expect(result.current.location.search).not.toContain('mode=raw');
  });

  it('defaults leftOpen and rightOpen to true when params are absent', () => {
    const { result } = renderHook(() => useViewState(), {
      wrapper: withRouter(['/trace']),
    });
    expect(result.current.state.leftOpen).toBe(true);
    expect(result.current.state.rightOpen).toBe(true);
  });

  it('parses ?left=0&right=0 as collapsed panels', () => {
    const { result } = renderHook(() => useViewState(), {
      wrapper: withRouter(['/trace?left=0&right=0']),
    });
    expect(result.current.state.leftOpen).toBe(false);
    expect(result.current.state.rightOpen).toBe(false);
  });

  it('left_open round-trips through the URL', () => {
    const { result } = renderHook(useViewStateWithLocation, {
      wrapper: withRouter(['/trace']),
    });
    act(() => result.current.vs.setLeftOpen(false));
    expect(result.current.location.search).toContain('left=0');
    act(() => result.current.vs.setLeftOpen(true));
    expect(result.current.location.search).not.toContain('left=');
  });

  it('right_open round-trips through the URL', () => {
    const { result } = renderHook(useViewStateWithLocation, {
      wrapper: withRouter(['/trace']),
    });
    act(() => result.current.vs.setRightOpen(false));
    expect(result.current.location.search).toContain('right=0');
    act(() => result.current.vs.setRightOpen(true));
    expect(result.current.location.search).not.toContain('right=');
  });

});
