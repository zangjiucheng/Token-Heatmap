import { useCallback, useRef, type KeyboardEvent, type PointerEvent } from 'react';
import type { PaneSide } from '@/hooks/usePaneWidths';
import './PaneGutter.css';

export interface PaneGutterProps {
  /** Which pane the gutter resizes. Controls the sign of the drag delta. */
  side: PaneSide;
  /** Current width of the controlled pane in pixels. */
  width: number;
  minWidth: number;
  maxWidth: number;
  /** Accessible name, e.g. "Resize controls panel". */
  label: string;
  /**
   * Called with a signed pixel delta on each pointer move and each keyboard
   * nudge. Positive values widen the controlled side.
   */
  onResize: (delta: number) => void;
  /**
   * Called after a drag ends (pointerup) or after a keyboard nudge so callers
   * can persist the new width. Not called during a drag-in-progress.
   */
  onCommit?: () => void;
  /** Called on double-click — caller should restore the default width. */
  onReset: () => void;
  /** Visible label / data-side for tests. */
  testId?: string;
}

const KEYBOARD_STEP = 8;
const KEYBOARD_SHIFT_STEP = 32;

/**
 * A focusable vertical separator between two grid columns. Pointer drag
 * adjusts the controlled pane's width incrementally; keyboard ←/→ nudge by
 * 8 px (32 px with Shift); double-click resets to the default width.
 */
export function PaneGutter({
  side,
  width,
  minWidth,
  maxWidth,
  label,
  onResize,
  onCommit,
  onReset,
  testId,
}: PaneGutterProps) {
  const lastClientXRef = useRef<number | null>(null);
  const isDraggingRef = useRef(false);

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      // Only react to primary button.
      if (event.button !== 0 && event.pointerType === 'mouse') return;
      event.preventDefault();
      isDraggingRef.current = true;
      lastClientXRef.current = event.clientX;
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // jsdom doesn't implement setPointerCapture; harmless.
      }
    },
    [],
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!isDraggingRef.current) return;
      const last = lastClientXRef.current;
      if (last == null) return;
      const rawDelta = event.clientX - last;
      if (rawDelta === 0) return;
      lastClientXRef.current = event.clientX;
      const sideDelta = side === 'left' ? rawDelta : -rawDelta;
      onResize(sideDelta);
    },
    [side, onResize],
  );

  const endDrag = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      lastClientXRef.current = null;
      try {
        event.currentTarget.releasePointerCapture(event.pointerId);
      } catch {
        // ignore
      }
      onCommit?.();
    },
    [onCommit],
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const magnitude = event.shiftKey ? KEYBOARD_SHIFT_STEP : KEYBOARD_STEP;
      // Per the acceptance criteria: → widens the side, ← narrows it.
      const sign = event.key === 'ArrowRight' ? 1 : -1;
      onResize(sign * magnitude);
      onCommit?.();
    },
    [onResize, onCommit],
  );

  const handleDoubleClick = useCallback(() => {
    onReset();
  }, [onReset]);

  return (
    // A focusable separator with pointer + keyboard handlers is the
    // recommended ARIA pattern for a resize gutter; the
    // no-noninteractive-tabindex / -interactions rules don't apply.
    /* eslint-disable jsx-a11y/no-noninteractive-tabindex, jsx-a11y/no-noninteractive-element-interactions */
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(width)}
      aria-valuemin={Math.round(minWidth)}
      aria-valuemax={Math.round(maxWidth)}
      tabIndex={0}
      className="pane-gutter"
      data-side={side}
      data-testid={testId ?? `pane-gutter-${side}`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onKeyDown={handleKeyDown}
      onDoubleClick={handleDoubleClick}
    />
    /* eslint-enable jsx-a11y/no-noninteractive-tabindex, jsx-a11y/no-noninteractive-element-interactions */
  );
}
