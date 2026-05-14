import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_PANE_WIDTHS,
  MIN_LEFT_WIDTH,
  MIN_RIGHT_WIDTH,
  PANE_WIDTHS_STORAGE_KEY,
  usePaneWidths,
} from '@/hooks/usePaneWidths';

beforeEach(() => {
  window.localStorage.clear();
});

describe('usePaneWidths', () => {
  it('test_default_widths_when_no_localStorage', () => {
    const { result } = renderHook(() => usePaneWidths());
    expect(result.current.widths.left).toBe(DEFAULT_PANE_WIDTHS.left);
    expect(result.current.widths.right).toBe(DEFAULT_PANE_WIDTHS.right);
  });

  it('reads previously stored widths on mount', () => {
    window.localStorage.setItem(
      PANE_WIDTHS_STORAGE_KEY,
      JSON.stringify({ left: 305, right: 410 }),
    );
    const { result } = renderHook(() => usePaneWidths());
    expect(result.current.widths.left).toBe(305);
    expect(result.current.widths.right).toBe(410);
  });

  it('test_widths_persisted_to_localStorage', () => {
    const { result } = renderHook(() => usePaneWidths());
    act(() => {
      result.current.setContainerWidth(2000);
    });
    act(() => {
      result.current.setWidth('left', 320);
    });
    const stored = JSON.parse(
      window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY) ?? '{}',
    ) as { left?: number; right?: number };
    expect(stored.left).toBe(320);
    expect(result.current.widths.left).toBe(320);
  });

  it('test_clamps_to_min_width_constraints', () => {
    const { result } = renderHook(() => usePaneWidths());
    act(() => {
      result.current.setContainerWidth(2000);
    });
    act(() => {
      result.current.setWidth('left', 50);
    });
    expect(result.current.widths.left).toBe(MIN_LEFT_WIDTH);
    act(() => {
      result.current.setWidth('right', 10);
    });
    expect(result.current.widths.right).toBe(MIN_RIGHT_WIDTH);
  });

  it('clamps a side that would shrink the center column below 360 px', () => {
    const { result } = renderHook(() => usePaneWidths());
    // A 1000 px container with right=320 leaves at most 1000 - 320 - 360 - 12
    // = 308 px for the left pane.
    act(() => {
      result.current.setContainerWidth(1000);
    });
    act(() => {
      result.current.setWidth('left', 700);
    });
    expect(result.current.widths.left).toBeLessThanOrEqual(308);
    expect(result.current.widths.left).toBeGreaterThanOrEqual(MIN_LEFT_WIDTH);
  });

  it('test_reset_clears_localStorage_entry', () => {
    const { result } = renderHook(() => usePaneWidths());
    act(() => {
      result.current.setContainerWidth(2000);
    });
    act(() => {
      result.current.setWidth('left', 305);
    });
    expect(
      JSON.parse(window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY) ?? '{}'),
    ).toEqual({ left: 305 });

    act(() => {
      result.current.reset('left');
    });
    // The 'left' entry is gone from storage; the hook returns the default.
    const stored = JSON.parse(
      window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY) ?? '{}',
    ) as Record<string, number>;
    expect(stored.left).toBeUndefined();
    expect(result.current.widths.left).toBe(DEFAULT_PANE_WIDTHS.left);
  });

  it('removes the storage entry entirely when both sides are reset', () => {
    const { result } = renderHook(() => usePaneWidths());
    act(() => result.current.setContainerWidth(2000));
    act(() => result.current.setWidth('left', 305));
    act(() => result.current.setWidth('right', 410));
    act(() => result.current.reset('left'));
    act(() => result.current.reset('right'));
    expect(window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY)).toBeNull();
  });
});
