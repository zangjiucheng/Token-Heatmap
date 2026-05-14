import { useCallback, useEffect, useState } from 'react';

export type PaneSide = 'left' | 'right';

export interface PaneWidths {
  left: number;
  right: number;
}

export interface StoredPaneWidths {
  left?: number;
  right?: number;
}

export const PANE_WIDTHS_STORAGE_KEY = 'token-heatmap.layout.paneWidths';

export const DEFAULT_PANE_WIDTHS: PaneWidths = {
  left: 280,
  right: 320,
};

export const MIN_LEFT_WIDTH = 220;
export const MIN_RIGHT_WIDTH = 260;
export const MIN_CENTER_WIDTH = 360;
export const MAX_PANE_WIDTH = 720;

function minForSide(side: PaneSide): number {
  return side === 'left' ? MIN_LEFT_WIDTH : MIN_RIGHT_WIDTH;
}

function readStored(): StoredPaneWidths {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object') return {};
    const result: StoredPaneWidths = {};
    const obj = parsed as Record<string, unknown>;
    if (typeof obj.left === 'number' && Number.isFinite(obj.left)) {
      result.left = obj.left;
    }
    if (typeof obj.right === 'number' && Number.isFinite(obj.right)) {
      result.right = obj.right;
    }
    return result;
  } catch {
    return {};
  }
}

function writeStored(stored: StoredPaneWidths): void {
  if (typeof window === 'undefined') return;
  try {
    if (stored.left == null && stored.right == null) {
      window.localStorage.removeItem(PANE_WIDTHS_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(
      PANE_WIDTHS_STORAGE_KEY,
      JSON.stringify(stored),
    );
  } catch {
    // Ignore — private mode or storage unavailable.
  }
}

/**
 * Clamp a requested pane width so that:
 *   - it is at least the side's own minimum,
 *   - it is no larger than MAX_PANE_WIDTH, and
 *   - the resulting center column is at least MIN_CENTER_WIDTH px wide,
 *     given the current container width and the opposite pane's width.
 */
const MIN_VIABLE_CONTAINER_WIDTH =
  MIN_LEFT_WIDTH + MIN_RIGHT_WIDTH + MIN_CENTER_WIDTH + 12;

export function clampWidth(
  side: PaneSide,
  requested: number,
  otherWidth: number,
  containerWidth: number | null,
): number {
  const min = minForSide(side);
  let next = Math.max(min, Math.min(MAX_PANE_WIDTH, requested));
  if (
    containerWidth != null &&
    Number.isFinite(containerWidth) &&
    containerWidth >= MIN_VIABLE_CONTAINER_WIDTH
  ) {
    // Account for the two 6px gutter tracks that always sit in the grid
    // template when both panes are open. When one side is collapsed its
    // gutter is display:none and the grid template drops it, but `otherWidth`
    // already reflects the actual rendered width, so this approximation is
    // close enough for clamping purposes.
    const maxForCenter = Math.max(
      min,
      containerWidth - otherWidth - MIN_CENTER_WIDTH - 12,
    );
    next = Math.min(next, maxForCenter);
  }
  return Math.max(min, Math.round(next));
}

export interface UsePaneWidthsResult {
  widths: PaneWidths;
  /** Set a single side's width (clamped + persisted). */
  setWidth: (side: PaneSide, px: number) => void;
  /** Reset a single side back to its default and clear its stored entry. */
  reset: (side: PaneSide) => void;
  /** Latest container width measured by the caller (px); `null` until known. */
  containerWidth: number | null;
  setContainerWidth: (px: number | null) => void;
}

/**
 * Persists user-chosen widths for the left and right panes in localStorage,
 * clamping each side so the center column never collapses below
 * `MIN_CENTER_WIDTH` pixels.
 */
export function usePaneWidths(): UsePaneWidthsResult {
  const [stored, setStored] = useState<StoredPaneWidths>(() => readStored());
  const [containerWidth, setContainerWidth] = useState<number | null>(null);

  useEffect(() => {
    writeStored(stored);
  }, [stored]);

  const widths: PaneWidths = {
    left: stored.left ?? DEFAULT_PANE_WIDTHS.left,
    right: stored.right ?? DEFAULT_PANE_WIDTHS.right,
  };

  const setWidth = useCallback(
    (side: PaneSide, px: number) => {
      setStored((prev) => {
        const other =
          side === 'left'
            ? (prev.right ?? DEFAULT_PANE_WIDTHS.right)
            : (prev.left ?? DEFAULT_PANE_WIDTHS.left);
        const clamped = clampWidth(side, px, other, containerWidth);
        return { ...prev, [side]: clamped };
      });
    },
    [containerWidth],
  );

  const reset = useCallback((side: PaneSide) => {
    setStored((prev) => {
      const next = { ...prev };
      delete next[side];
      return next;
    });
  }, []);

  return {
    widths,
    setWidth,
    reset,
    containerWidth,
    setContainerWidth,
  };
}
