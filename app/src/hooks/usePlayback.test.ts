import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { usePlayback } from './usePlayback';

/**
 * Drives the hook with fake timers and a controlled `selectedStep` so we can
 * assert the exact advance / loop behaviour without a real clock.
 */
function setup(range: [number, number], initial: number | null = null) {
  let step = initial;
  const setSelectedStep = vi.fn((s: number) => {
    step = s;
  });
  const view = renderHook(
    ({ selectedStep }) =>
      usePlayback({ selectedStep, setSelectedStep, range, enabled: true }),
    { initialProps: { selectedStep: step } },
  );
  // Re-render with the latest step after each tick so the hook's ref tracks it,
  // mirroring how the parent component re-renders on selectedStep changes.
  const tick = (ms: number) => {
    act(() => {
      vi.advanceTimersByTime(ms);
    });
    view.rerender({ selectedStep: step });
  };
  return {
    view,
    tick,
    get step() {
      return step;
    },
    setSelectedStep,
  };
}

describe('usePlayback', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('does not advance until playing', () => {
    const h = setup([0, 4]);
    h.tick(2000);
    expect(h.setSelectedStep).not.toHaveBeenCalled();
  });

  it('advances one step per tick and loops back to the window start', () => {
    const h = setup([0, 3], null);
    act(() => h.view.result.current.play());
    // null -> start, then 1, 2, 3, then wrap to 0.
    h.tick(700);
    expect(h.step).toBe(0);
    h.tick(700);
    expect(h.step).toBe(1);
    h.tick(700);
    expect(h.step).toBe(2);
    h.tick(700);
    expect(h.step).toBe(3);
    h.tick(700);
    expect(h.step).toBe(0); // looped
  });

  it('pause stops advancing', () => {
    const h = setup([0, 4], 0);
    act(() => h.view.result.current.play());
    h.tick(700);
    expect(h.step).toBe(1);
    act(() => h.view.result.current.pause());
    h.tick(5000);
    expect(h.step).toBe(1);
  });

  it('respects a sub-window and loops within it', () => {
    const h = setup([2, 4], 3);
    act(() => h.view.result.current.play());
    h.tick(700);
    expect(h.step).toBe(4);
    h.tick(700);
    expect(h.step).toBe(2); // loops to window start, not 0
  });

  it('cycleSpeed shortens the interval', () => {
    const h = setup([0, 9], 0);
    act(() => {
      h.view.result.current.play();
      h.view.result.current.cycleSpeed(); // 1x -> 2x (350ms)
    });
    h.tick(350);
    expect(h.step).toBe(1);
  });
});
