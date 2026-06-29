import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { announceLiveRegion } from '@/lib/a11y/announceLiveRegion';
import {
  clampWidth,
  DEFAULT_PANE_WIDTHS,
  MAX_PANE_WIDTH,
  MIN_LEFT_WIDTH,
  MIN_RIGHT_WIDTH,
  usePaneWidths,
  type PaneSide,
} from '@/hooks/usePaneWidths';
import { PaneGutter } from './PaneGutter';
import './ThreePaneLayout.css';

const NARROW_QUERY = '(max-width: 1023px)';

export interface ThreePaneLayoutProps {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
  leftLabel?: string;
  rightLabel?: string;
  /**
   * Whether each side panel is expanded. When omitted the layout falls back
   * to local state — useful for tests / stories that don't lift state.
   */
  leftOpen?: boolean;
  rightOpen?: boolean;
  onToggleLeft?: (open: boolean) => void;
  onToggleRight?: (open: boolean) => void;
}

function useIsNarrow(): boolean {
  const [isNarrow, setIsNarrow] = useState(() => {
    if (
      typeof window === 'undefined' ||
      typeof window.matchMedia !== 'function'
    ) {
      return false;
    }
    return window.matchMedia(NARROW_QUERY).matches;
  });

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return undefined;
    const mql = window.matchMedia(NARROW_QUERY);
    const handler = (event: MediaQueryListEvent) => setIsNarrow(event.matches);
    setIsNarrow(mql.matches);
    if (typeof mql.addEventListener === 'function') {
      mql.addEventListener('change', handler);
      return () => mql.removeEventListener('change', handler);
    }
    mql.addListener(handler);
    return () => mql.removeListener(handler);
  }, []);

  return isNarrow;
}

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function focusFirstInside(container: HTMLElement | null): void {
  if (!container) return;
  const first = container.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
  if (first) {
    first.focus();
  } else {
    // Fall back to focusing the container itself so screen readers land
    // somewhere inside the newly expanded pane.
    if (container.getAttribute('tabindex') == null) {
      container.setAttribute('tabindex', '-1');
    }
    container.focus();
  }
}

export function ThreePaneLayout({
  left,
  center,
  right,
  leftLabel = 'View settings',
  rightLabel = 'Inspector',
  leftOpen: leftOpenProp,
  rightOpen: rightOpenProp,
  onToggleLeft,
  onToggleRight,
}: ThreePaneLayoutProps) {
  const isNarrow = useIsNarrow();

  const [internalLeftOpen, setInternalLeftOpen] = useState(true);
  const [internalRightOpen, setInternalRightOpen] = useState(true);

  const leftOpen = leftOpenProp ?? internalLeftOpen;
  const rightOpen = rightOpenProp ?? internalRightOpen;

  const leftBodyRef = useRef<HTMLDivElement | null>(null);
  const rightBodyRef = useRef<HTMLDivElement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  // Track whether a transition was user-initiated so we don't move focus or
  // announce on the initial render.
  const pendingLeftFocus = useRef(false);
  const pendingRightFocus = useRef(false);

  const {
    widths,
    setWidth,
    reset: resetWidth,
    setContainerWidth,
  } = usePaneWidths();
  // Transient deltas accumulated during an in-progress drag so the gutter
  // can update the layout responsively without persisting on every pointermove.
  const [pendingDelta, setPendingDelta] = useState<{
    left: number;
    right: number;
  }>(() => ({ left: 0, right: 0 }));
  const widthsRef = useRef(widths);
  widthsRef.current = widths;
  const containerRef = useRef<number | null>(null);

  const effectiveWidths = {
    left: clampWidth(
      'left',
      widths.left + pendingDelta.left,
      widths.right + pendingDelta.right,
      containerRef.current,
    ),
    right: clampWidth(
      'right',
      widths.right + pendingDelta.right,
      widths.left + pendingDelta.left,
      containerRef.current,
    ),
  };

  const handleToggleLeft = useCallback(() => {
    const next = !leftOpen;
    if (next) pendingLeftFocus.current = true;
    if (onToggleLeft) {
      onToggleLeft(next);
    } else {
      setInternalLeftOpen(next);
    }
    announceLiveRegion(
      next ? `${leftLabel} panel expanded` : `${leftLabel} panel collapsed`,
    );
  }, [leftOpen, leftLabel, onToggleLeft]);

  const handleToggleRight = useCallback(() => {
    const next = !rightOpen;
    if (next) pendingRightFocus.current = true;
    if (onToggleRight) {
      onToggleRight(next);
    } else {
      setInternalRightOpen(next);
    }
    announceLiveRegion(
      next ? `${rightLabel} panel expanded` : `${rightLabel} panel collapsed`,
    );
  }, [rightOpen, rightLabel, onToggleRight]);

  useEffect(() => {
    if (leftOpen && pendingLeftFocus.current) {
      pendingLeftFocus.current = false;
      focusFirstInside(leftBodyRef.current);
    }
  }, [leftOpen]);

  useEffect(() => {
    if (rightOpen && pendingRightFocus.current) {
      pendingRightFocus.current = false;
      focusFirstInside(rightBodyRef.current);
    }
  }, [rightOpen]);

  // Observe the root width so usePaneWidths can clamp against the live
  // viewport size (keeps the center column above its 360 px floor).
  useEffect(() => {
    const el = rootRef.current;
    const apply = (w: number) => {
      containerRef.current = w;
      setContainerWidth(w);
    };
    if (!el) return undefined;
    apply(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      apply(entry.contentRect.width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [setContainerWidth]);

  const handleResize = useCallback((side: PaneSide, delta: number) => {
    setPendingDelta((prev) => ({ ...prev, [side]: prev[side] + delta }));
  }, []);

  const handleCommit = useCallback(
    (side: PaneSide) => {
      setPendingDelta((prev) => {
        if (prev[side] === 0) return prev;
        const target = widthsRef.current[side] + prev[side];
        setWidth(side, target);
        return { ...prev, [side]: 0 };
      });
    },
    [setWidth],
  );

  const handleReset = useCallback(
    (side: PaneSide) => {
      setPendingDelta((prev) => ({ ...prev, [side]: 0 }));
      resetWidth(side);
    },
    [resetWidth],
  );

  const showLeftGutter = leftOpen && !isNarrow;
  const showRightGutter = rightOpen && !isNarrow;

  const rootStyle = useMemo<CSSProperties>(
    () =>
      ({
        '--pane-left': `${Math.round(effectiveWidths.left)}px`,
        '--pane-right': `${Math.round(effectiveWidths.right)}px`,
      }) as CSSProperties,
    [effectiveWidths.left, effectiveWidths.right],
  );

  return (
    <div
      ref={rootRef}
      className="three-pane"
      data-narrow={isNarrow ? 'true' : 'false'}
      data-left-open={leftOpen ? 'true' : 'false'}
      data-right-open={rightOpen ? 'true' : 'false'}
      style={rootStyle}
    >
      <aside
        className="three-pane__left"
        aria-label={leftLabel}
        data-testid="three-pane-left"
      >
        {leftOpen ? (
          <>
            <div className="three-pane__pane-header">
              <span>{leftLabel}</span>
              <button
                type="button"
                className="three-pane__toggle"
                aria-expanded
                aria-controls="three-pane-left-content"
                aria-label={`Hide ${leftLabel.toLowerCase()}`}
                onClick={handleToggleLeft}
                data-testid="three-pane-left-collapse"
              >
                <span aria-hidden="true">‹</span>
              </button>
            </div>
            <div
              id="three-pane-left-content"
              className="three-pane__pane-body"
              ref={leftBodyRef}
            >
              {left}
            </div>
          </>
        ) : (
          <div className="three-pane__rail">
            <button
              type="button"
              className="three-pane__rail-button"
              aria-expanded={false}
              aria-controls="three-pane-left-content"
              aria-label={`Show ${leftLabel.toLowerCase()}`}
              onClick={handleToggleLeft}
              data-testid="three-pane-left-expand"
            >
              <span aria-hidden="true">›</span>
            </button>
          </div>
        )}
      </aside>

      {showLeftGutter ? (
        <PaneGutter
          side="left"
          width={effectiveWidths.left}
          minWidth={MIN_LEFT_WIDTH}
          maxWidth={MAX_PANE_WIDTH}
          label={`Resize ${leftLabel.toLowerCase()} panel`}
          onResize={(d) => handleResize('left', d)}
          onCommit={() => handleCommit('left')}
          onReset={() => handleReset('left')}
          testId="three-pane-gutter-left"
        />
      ) : null}

      <section
        className="three-pane__center"
        aria-label="Main viewport"
        data-testid="three-pane-center"
      >
        {center}
      </section>

      {showRightGutter ? (
        <PaneGutter
          side="right"
          width={effectiveWidths.right}
          minWidth={MIN_RIGHT_WIDTH}
          maxWidth={MAX_PANE_WIDTH}
          label={`Resize ${rightLabel.toLowerCase()} panel`}
          onResize={(d) => handleResize('right', d)}
          onCommit={() => handleCommit('right')}
          onReset={() => handleReset('right')}
          testId="three-pane-gutter-right"
        />
      ) : null}

      <aside
        className="three-pane__right"
        aria-label={rightLabel}
        data-testid="three-pane-right"
      >
        {rightOpen ? (
          <>
            <div className="three-pane__pane-header">
              <span>{rightLabel}</span>
              <button
                type="button"
                className="three-pane__toggle"
                aria-expanded
                aria-controls="three-pane-right-content"
                aria-label={`Hide ${rightLabel.toLowerCase()}`}
                onClick={handleToggleRight}
                data-testid="three-pane-right-collapse"
              >
                <span aria-hidden="true">›</span>
              </button>
            </div>
            <div
              id="three-pane-right-content"
              className="three-pane__pane-body"
              ref={rightBodyRef}
            >
              {right}
            </div>
          </>
        ) : (
          <div className="three-pane__rail">
            <button
              type="button"
              className="three-pane__rail-button"
              aria-expanded={false}
              aria-controls="three-pane-right-content"
              aria-label={`Show ${rightLabel.toLowerCase()}`}
              onClick={handleToggleRight}
              data-testid="three-pane-right-expand"
            >
              <span aria-hidden="true">‹</span>
            </button>
          </div>
        )}
      </aside>
    </div>
  );
}

export { DEFAULT_PANE_WIDTHS };
