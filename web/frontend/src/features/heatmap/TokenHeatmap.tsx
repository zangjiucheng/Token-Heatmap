import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
  type PointerEvent as ReactPointerEvent,
  type UIEvent as ReactUIEvent,
} from 'react';
import type { Trace } from '@/types/trace';
import {
  buildGrid,
  normalize,
  type DistributionSource,
  type ValueCol,
} from './buildGrid';
import { sampleColor } from './colormap';
import { HeatmapTooltip, type HeatmapTooltipDatum } from './HeatmapTooltip';
import { HeatmapLegend } from './HeatmapLegend';
import { useThemeTokens } from '@/hooks/useThemeTokens';
import './TokenHeatmap.css';

export interface TokenHeatmapProps {
  trace: Trace;
  valueCol: ValueCol;
  selectedStep: number | null;
  onSelectStep: (step: number) => void;
  /**
   * Transient step from an external source — either a timeline hover or a
   * sibling heatmap in `SplitHeatmap`. Painted as a soft column highlight,
   * distinct from the stronger `selectedStep` indicator.
   */
  externalHoveredStep?: number | null;
  /** Optional callback fired after each repaint with the elapsed ms. */
  onRenderTime?: (ms: number) => void;
  /** Optional fixed size for tests; otherwise the component fills its container. */
  width?: number;
  height?: number;
  /** Debounce delay for resize listener in ms. */
  resizeDebounceMs?: number;
  /**
   * Which distribution to render. Defaults to `'processed'` to match the
   * historical behaviour; `'raw'` and the side-by-side comparison are used
   * by `SplitHeatmap`.
   */
  source?: DistributionSource;
  /**
   * Inclusive step range `[startStep, endStep]` to render. Cells outside the
   * range are hidden from both the matrix and the axis. Defaults to the full
   * trace range.
   */
  stepRange?: [number, number];
  /**
   * Optional override for the color scale. When omitted the scale is computed
   * from the grid's finite min/max (auto mode); when provided the scale is
   * clamped to this range and out-of-range values clip to the bottom/top
   * colors.
   */
  valueRange?: { min: number; max: number };
  /** Notified when the locally hovered step changes (or becomes null). */
  onHoverStep?: (step: number | null) => void;
  /** Optional label suffix appended to the accessible name. */
  ariaLabelSuffix?: string;
  /** Hides the toolbar (reset + legend rail) when rendering inside SplitHeatmap. */
  hideToolbar?: boolean;
  /** Minimum cell width in CSS pixels; cells never shrink below this floor. */
  minCellWidth?: number;
  /** Minimum cell height in CSS pixels; cells never shrink below this floor. */
  minCellHeight?: number;
  /** Maximum cell width in CSS pixels (null = unbounded). */
  maxCellWidth?: number | null;
  /** Maximum cell height in CSS pixels (null = unbounded). */
  maxCellHeight?: number | null;
  /**
   * Controlled horizontal scroll position. When provided, the heatmap's
   * scroll wrapper is kept in sync with this value (used by `SplitHeatmap`
   * to lock the two panes together).
   */
  scrollLeft?: number;
  /** Notified when the user scrolls the heatmap horizontally. */
  onScrollChange?: (scrollLeft: number) => void;
}

interface CellDescriptor {
  step: number;
  rank: number;
}

interface ViewState {
  zoom: number;
  panY: number;
}

const INITIAL_VIEW: ViewState = { zoom: 1, panY: 0 };
const MIN_ZOOM = 1;
const MAX_ZOOM = 16;
export const AXIS_GUTTER_X = 36;
export const AXIS_GUTTER_Y = 28;
export const LEGEND_RAIL_W = 120;
const DEFAULT_MIN_CELL_W = 16;
const DEFAULT_MIN_CELL_H = 14;
const DEFAULT_MAX_CELL_W = 48;
const DEFAULT_MAX_CELL_H = 32;

function clampSize(value: number, min: number, max: number | null): number {
  const lo = Math.max(1, min);
  const capped = max == null ? value : Math.min(value, max);
  return Math.max(lo, capped);
}

function pickTextColor(rNorm: number): string {
  // Light text over dark cells, dark text over bright yellow cells.
  return rNorm > 0.65 ? '#1a1d21' : '#ffffff';
}

function truncateForCell(token: string, cellWidth: number): string {
  // ~7px per char at 11px font; leave a 4px padding either side.
  const maxChars = Math.max(0, Math.floor((cellWidth - 4) / 7));
  if (maxChars <= 0) return '';
  if (token.length <= maxChars) return token;
  if (maxChars <= 1) return token.slice(0, 1);
  return `${token.slice(0, maxChars - 1)}…`;
}

function visibleTokenText(token: string): string {
  if (token === '\n') return '\\n';
  if (token === '\t') return '\\t';
  return token;
}

export function TokenHeatmap({
  trace,
  valueCol,
  selectedStep,
  onSelectStep,
  externalHoveredStep = null,
  onRenderTime,
  width: widthProp,
  height: heightProp,
  resizeDebounceMs = 50,
  source = 'processed',
  stepRange,
  valueRange,
  onHoverStep,
  ariaLabelSuffix,
  hideToolbar = false,
  minCellWidth = DEFAULT_MIN_CELL_W,
  minCellHeight = DEFAULT_MIN_CELL_H,
  maxCellWidth = DEFAULT_MAX_CELL_W,
  maxCellHeight = DEFAULT_MAX_CELL_H,
  scrollLeft: scrollLeftProp,
  onScrollChange,
}: TokenHeatmapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const axisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dataCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Canvas can't read CSS vars; pull resolved theme colors and repaint on
  // theme change (tk is in the draw effects' deps below).
  const tk = useThemeTokens();

  const [size, setSize] = useState<{ w: number; h: number }>({
    w: widthProp ?? 800,
    h: heightProp ?? 400,
  });
  const [view, setView] = useState<ViewState>(INITIAL_VIEW);
  const [hover, setHover] = useState<{
    cell: CellDescriptor;
    x: number;
    y: number;
  } | null>(null);
  const [tooltipDismissed, setTooltipDismissed] = useState(false);

  const grid = useMemo(
    () => buildGrid(trace, valueCol, source),
    [trace, valueCol, source],
  );

  // Resolved step window: clamped to grid bounds, defaults to the full range.
  const [stepStart, stepEnd] = useMemo<[number, number]>(() => {
    if (grid.steps === 0) return [0, -1];
    const defaultStart = 0;
    const defaultEnd = grid.steps - 1;
    if (!stepRange) return [defaultStart, defaultEnd];
    const rawStart = Math.max(0, Math.min(grid.steps - 1, stepRange[0]));
    const rawEnd = Math.max(0, Math.min(grid.steps - 1, stepRange[1]));
    return rawStart <= rawEnd ? [rawStart, rawEnd] : [rawEnd, rawStart];
  }, [grid.steps, stepRange]);
  const visibleStepCount = Math.max(1, stepEnd - stepStart + 1);

  // Effective color-scale bounds: auto from the grid unless explicitly overridden.
  const effectiveMin = valueRange ? valueRange.min : grid.valueMin;
  const effectiveMax = valueRange ? valueRange.max : grid.valueMax;

  // Re-enable tooltip when trace or valueCol changes.
  useEffect(() => {
    setTooltipDismissed(false);
  }, [trace, valueCol]);

  // Auto-size from the container (unless explicit width/height are given).
  useLayoutEffect(() => {
    if (widthProp != null && heightProp != null) return undefined;
    const container = containerRef.current;
    if (!container) return undefined;

    const measure = () => {
      const rect = container.getBoundingClientRect();
      setSize({
        w: widthProp ?? Math.max(200, Math.floor(rect.width)),
        h: heightProp ?? Math.max(150, Math.floor(rect.height)),
      });
    };
    measure();

    if (typeof ResizeObserver === 'undefined') return undefined;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const ro = new ResizeObserver(() => {
      if (timer != null) clearTimeout(timer);
      timer = setTimeout(measure, resizeDebounceMs);
    });
    ro.observe(container);
    return () => {
      if (timer != null) clearTimeout(timer);
      ro.disconnect();
    };
  }, [widthProp, heightProp, resizeDebounceMs]);

  // Compute layout geometry. Cell width/height are clamped to the configured
  // floor/ceiling so cells never become unreadable on long traces or
  // awkwardly stretched on short ones. When the natural grid exceeds the
  // scroll wrapper, the inner content is wider than the wrapper and the
  // browser supplies a horizontal scrollbar.
  const rightRailW = hideToolbar ? 0 : LEGEND_RAIL_W;
  const plot = useMemo(() => {
    const scrollW = Math.max(1, size.w - AXIS_GUTTER_X - rightRailW);
    const plotH = Math.max(1, size.h);
    const cellAreaH = Math.max(1, plotH - AXIS_GUTTER_Y);
    const ranksVisible = grid.ranks > 0 ? grid.ranks : 1;
    const baseCellW = clampSize(
      scrollW / visibleStepCount,
      minCellWidth,
      maxCellWidth,
    );
    const baseCellH = clampSize(
      cellAreaH / ranksVisible,
      minCellHeight,
      maxCellHeight,
    );
    const cellW = baseCellW * view.zoom;
    const cellH = baseCellH * view.zoom;
    // The inner content is sized to the natural grid width. When this is
    // smaller than the scroll wrapper the grid stays left-aligned with empty
    // space to its right (no stretching); when it is larger the browser
    // supplies a horizontal scrollbar.
    const contentW = visibleStepCount * cellW;
    return {
      scrollW,
      plotH,
      cellAreaH,
      cellW,
      cellH,
      contentW,
      originY: view.panY,
    };
  }, [
    size,
    rightRailW,
    grid.ranks,
    visibleStepCount,
    view,
    minCellWidth,
    minCellHeight,
    maxCellWidth,
    maxCellHeight,
  ]);

  // Sync controlled `scrollLeft` prop down to the DOM. The check guards
  // against feedback loops when our own `onScroll` lifts the value up.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (scrollLeftProp == null) return;
    const target = Math.max(
      0,
      Math.min(scrollLeftProp, plot.contentW - plot.scrollW),
    );
    if (Math.abs(el.scrollLeft - target) > 0.5) {
      el.scrollLeft = target;
    }
  }, [scrollLeftProp, plot.contentW, plot.scrollW]);

  // Paint the data canvas whenever inputs change.
  useLayoutEffect(() => {
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr =
      typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const w = plot.contentW;
    const h = plot.plotH;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const start =
      typeof performance !== 'undefined' &&
      typeof performance.now === 'function'
        ? performance.now()
        : 0;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Background for the cell area.
    ctx.fillStyle = tk.surface;
    ctx.fillRect(0, 0, w, plot.cellAreaH);

    const { cellW, cellH, originY } = plot;
    const min = effectiveMin;
    const max = effectiveMax;

    // Hatched pattern for invalid cells (drawn once, reused via fillStyle).
    const hatchSize = 6;
    const hatch = document.createElement('canvas');
    hatch.width = hatchSize;
    hatch.height = hatchSize;
    const hctx = hatch.getContext('2d');
    if (hctx) {
      hctx.fillStyle = 'rgba(0, 0, 0, 0)';
      hctx.fillRect(0, 0, hatchSize, hatchSize);
      hctx.strokeStyle = tk.borderStrong;
      hctx.lineWidth = 1;
      hctx.beginPath();
      hctx.moveTo(-1, hatchSize + 1);
      hctx.lineTo(hatchSize + 1, -1);
      hctx.stroke();
    }
    const hatchPattern = ctx.createPattern(hatch, 'repeat');

    // Clip to the cell area so vertical pan and zoom can spill off-screen
    // safely without painting over the x-axis tick row.
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, w, plot.cellAreaH);
    ctx.clip();

    const fontSize = Math.max(8, Math.min(12, Math.floor(cellH * 0.6)));
    ctx.font = `${fontSize}px 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let step = stepStart; step <= stepEnd; step += 1) {
      const col = step - stepStart;
      const x = col * cellW;
      if (x + cellW < 0) continue;
      if (x > w) break;

      const k = grid.kUsed[step];

      for (let rank = 0; rank < grid.ranks; rank += 1) {
        const y = originY + rank * cellH;
        if (y + cellH < 0) continue;
        if (y > plot.cellAreaH) break;

        const idx = rank * grid.steps + step;
        const v = grid.values[idx];

        if (rank >= k || !Number.isFinite(v)) {
          if (hatchPattern) {
            ctx.fillStyle = hatchPattern;
            ctx.fillRect(x, y, cellW, cellH);
          }
          continue;
        }

        // In manual color-range mode, clip to [0, 1] so out-of-range values
        // saturate at the bottom/top color.
        const tRaw = normalize(v, min, max);
        const t = Math.max(0, Math.min(1, tRaw));
        const [r, g, b] = sampleColor(t);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, y, cellW, cellH);

        if (cellW >= 18 && cellH >= 12) {
          const token = visibleTokenText(grid.tokens[idx]);
          const display = truncateForCell(token, cellW);
          if (display.length > 0) {
            ctx.fillStyle = pickTextColor(t);
            ctx.fillText(display, x + cellW / 2, y + cellH / 2);
          }
        }
      }
    }

    ctx.restore();

    // X-axis ticks below the cell area (scrolls with the content).
    ctx.fillStyle = tk.textMuted;
    ctx.font = "11px 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace";
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xStride =
      visibleStepCount > 20 ? Math.ceil(visibleStepCount / 12) : 1;
    for (let step = stepStart; step <= stepEnd; step += xStride) {
      const col = step - stepStart;
      const x = col * cellW + cellW / 2;
      ctx.fillText(String(step), x, plot.cellAreaH + 4);
    }

    if (onRenderTime && typeof performance !== 'undefined') {
      onRenderTime(performance.now() - start);
    }
  }, [
    grid,
    plot,
    stepStart,
    stepEnd,
    visibleStepCount,
    effectiveMin,
    effectiveMax,
    onRenderTime,
    tk,
  ]);

  // Paint the left-rail axis canvas (y-axis tick labels + rotated title).
  useLayoutEffect(() => {
    const canvas = axisCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr =
      typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const w = AXIS_GUTTER_X;
    const h = plot.plotH;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = tk.textMuted;
    ctx.font = "11px 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace";
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const { cellH, originY } = plot;
    const yStride = grid.ranks > 16 ? Math.ceil(grid.ranks / 12) : 1;
    for (let rank = 0; rank < grid.ranks; rank += yStride) {
      const y = originY + rank * cellH + cellH / 2;
      if (y < 0 || y > plot.cellAreaH) continue;
      ctx.fillText(String(rank + 1), w - 4, y);
    }

    // Rotated "Adaptive rank" label.
    ctx.fillStyle = tk.text;
    ctx.font = "600 11px 'Space Grotesk', system-ui, sans-serif";
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.save();
    ctx.translate(10, plot.cellAreaH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Adaptive rank', 0, 0);
    ctx.restore();
  }, [plot, grid.ranks, tk]);

  // Paint the overlay canvas (selection column + hover crosshair). It lives
  // inside the scroll wrapper so its painted positions move with the content;
  // no scroll-offset arithmetic is needed.
  useLayoutEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr =
      typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const w = plot.contentW;
    const h = plot.plotH;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, w, plot.cellAreaH);
    ctx.clip();

    // Soft highlight for a step driven from an external source — either a
    // timeline hover or a sibling heatmap in SplitHeatmap. Painted first so
    // a stronger selection overlay can still cover it.
    if (
      externalHoveredStep != null &&
      externalHoveredStep >= stepStart &&
      externalHoveredStep <= stepEnd &&
      externalHoveredStep !== selectedStep
    ) {
      const col = externalHoveredStep - stepStart;
      const x = col * plot.cellW;
      ctx.fillStyle = 'rgba(213, 94, 0, 0.12)';
      ctx.fillRect(x, 0, plot.cellW, plot.cellAreaH);
      ctx.strokeStyle = '#d55e00';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x + 0.5, 0.5, plot.cellW - 1, plot.cellAreaH - 1);
      ctx.setLineDash([]);
    }

    if (
      selectedStep != null &&
      selectedStep >= stepStart &&
      selectedStep <= stepEnd
    ) {
      const col = selectedStep - stepStart;
      const x = col * plot.cellW;
      ctx.fillStyle = 'rgba(0, 114, 178, 0.18)';
      ctx.fillRect(x, 0, plot.cellW, plot.cellAreaH);
      ctx.strokeStyle = 'rgb(0, 114, 178)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, 1, plot.cellW - 2, plot.cellAreaH - 2);
    }

    if (hover && !tooltipDismissed) {
      const col = hover.cell.step - stepStart;
      const hx = col * plot.cellW;
      const hy = plot.originY + hover.cell.rank * plot.cellH;
      ctx.strokeStyle = '#1a1d21';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(hx, hy, plot.cellW, plot.cellH);
    }

    ctx.restore();
  }, [
    selectedStep,
    externalHoveredStep,
    hover,
    tooltipDismissed,
    stepStart,
    stepEnd,
    plot,
  ]);

  // Translate a pointer event into a grid cell. Hit-testing uses the data
  // canvas's bounding rect, which moves with horizontal scroll, so the
  // scroll offset is already baked into `clientX - canvasRect.left`.
  const cellAtPointer = useCallback(
    (clientX: number, clientY: number): CellDescriptor | null => {
      const canvas = dataCanvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      if (x < 0 || x > plot.contentW) return null;
      if (y < 0 || y > plot.cellAreaH) return null;
      const col = Math.floor(x / plot.cellW);
      const rank = Math.floor((y - plot.originY) / plot.cellH);
      if (col < 0 || col >= visibleStepCount) return null;
      if (rank < 0 || rank >= grid.ranks) return null;
      const step = stepStart + col;
      if (step < stepStart || step > stepEnd) return null;
      return { step, rank };
    },
    [grid.ranks, plot, stepStart, stepEnd, visibleStepCount],
  );

  const tooltipDatum = useMemo<HeatmapTooltipDatum | null>(() => {
    if (!hover || tooltipDismissed) return null;
    const { step, rank } = hover.cell;
    const idx = rank * grid.steps + step;
    const k = grid.kUsed[step];
    if (rank >= k) return null;
    return {
      step,
      rank: rank + 1,
      token: grid.tokens[idx],
      prob: grid.probs[idx],
      logprob: grid.logprobs[idx],
      kUsed: k,
      entropy: grid.entropy[step],
    };
  }, [hover, tooltipDismissed, grid]);

  const handleScroll = useCallback(
    (event: ReactUIEvent<HTMLDivElement>) => {
      onScrollChange?.(event.currentTarget.scrollLeft);
    },
    [onScrollChange],
  );

  const handleMouseMove = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      const cell = cellAtPointer(event.clientX, event.clientY);
      if (!cell) {
        setHover(null);
        onHoverStep?.(null);
        return;
      }
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      setHover({
        cell,
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      });
      setTooltipDismissed(false);
      onHoverStep?.(cell.step);
    },
    [cellAtPointer, onHoverStep],
  );

  const handleMouseLeave = useCallback(() => {
    setHover(null);
    onHoverStep?.(null);
  }, [onHoverStep]);

  const handleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      const cell = cellAtPointer(event.clientX, event.clientY);
      if (!cell) return;
      onSelectStep(cell.step);
      containerRef.current?.focus();
    },
    [cellAtPointer, onSelectStep],
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (grid.steps === 0) return;
      const current = selectedStep ?? stepStart;
      switch (event.key) {
        case 'ArrowLeft': {
          event.preventDefault();
          onSelectStep(Math.max(stepStart, current - 1));
          return;
        }
        case 'ArrowRight': {
          event.preventDefault();
          onSelectStep(Math.min(stepEnd, current + 1));
          return;
        }
        case 'Home': {
          event.preventDefault();
          onSelectStep(stepStart);
          return;
        }
        case 'End': {
          event.preventDefault();
          onSelectStep(stepEnd);
          return;
        }
        case 'Enter': {
          event.preventDefault();
          if (selectedStep != null) onSelectStep(selectedStep);
          return;
        }
        case 'Escape': {
          event.preventDefault();
          setTooltipDismissed(true);
          setHover(null);
          return;
        }
        default:
          return;
      }
    },
    [grid.steps, selectedStep, onSelectStep, stepStart, stepEnd],
  );

  const handleWheel = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const delta = -event.deltaY * 0.001;
    setView((prev) => {
      const nextZoom = Math.min(
        MAX_ZOOM,
        Math.max(MIN_ZOOM, prev.zoom * (1 + delta)),
      );
      if (nextZoom === prev.zoom) return prev;
      return { ...prev, zoom: nextZoom };
    });
  }, []);

  // Pan via pointer drag with the middle button (or shift+drag). Horizontal
  // drag translates to scrollLeft updates so the heatmap follows naturally
  // alongside the native scrollbar; vertical drag updates panY.
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 1 && !event.shiftKey) return;
      event.preventDefault();
      dragRef.current = { x: event.clientX, y: event.clientY };
      (event.currentTarget as HTMLDivElement).setPointerCapture(
        event.pointerId,
      );
    },
    [],
  );
  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const start = dragRef.current;
      if (!start) return;
      const dx = event.clientX - start.x;
      const dy = event.clientY - start.y;
      dragRef.current = { x: event.clientX, y: event.clientY };
      if (scrollRef.current && dx !== 0) {
        scrollRef.current.scrollLeft -= dx;
      }
      if (dy !== 0) {
        setView((prev) => ({ ...prev, panY: prev.panY + dy }));
      }
    },
    [],
  );
  const handlePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!dragRef.current) return;
      dragRef.current = null;
      (event.currentTarget as HTMLDivElement).releasePointerCapture?.(
        event.pointerId,
      );
    },
    [],
  );

  const handleReset = useCallback(() => {
    setView(INITIAL_VIEW);
    if (scrollRef.current) scrollRef.current.scrollLeft = 0;
    onScrollChange?.(0);
  }, [onScrollChange]);

  const ariaLabel = [
    `Token probability heatmap, ${grid.steps} generation steps by ${grid.ranks} adaptive ranks, colored by ${valueCol}`,
    ariaLabelSuffix,
  ]
    .filter(Boolean)
    .join(' — ');

  return (
    <div className="token-heatmap" data-testid="token-heatmap">
      {!hideToolbar && (
        <div className="token-heatmap__toolbar">
          <button
            type="button"
            className="token-heatmap__reset"
            onClick={handleReset}
            data-testid="token-heatmap-reset"
          >
            Reset view
          </button>
        </div>
      )}
      {/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex --
          role="application" is the correct ARIA pattern for this composite
          widget: interactions are custom, not native.
          jsx-a11y treats application as non-interactive, so both rules are
          intentionally disabled for this element. */}
      <div
        ref={containerRef}
        className="token-heatmap__plot"
        role="application"
        aria-label={ariaLabel}
        tabIndex={0}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        style={
          widthProp != null && heightProp != null
            ? { width: widthProp, height: heightProp }
            : undefined
        }
        data-testid="token-heatmap-plot"
        data-selected-step={selectedStep ?? ''}
        data-external-hovered-step={externalHoveredStep ?? ''}
        data-cell-width={plot.cellW}
        data-cell-height={plot.cellH}
        data-content-width={plot.contentW}
        data-scroll-width={plot.scrollW}
      >
        <canvas
          ref={axisCanvasRef}
          className="token-heatmap__axis-canvas"
          aria-hidden="true"
          data-testid="token-heatmap-axis"
        />
        <div
          ref={scrollRef}
          className="token-heatmap__scroll"
          onScroll={handleScroll}
          data-testid="token-heatmap-scroll"
        >
          <div
            className="token-heatmap__inner"
            style={{ width: plot.contentW, height: plot.plotH }}
            data-testid="token-heatmap-inner"
          >
            <canvas
              ref={dataCanvasRef}
              className="token-heatmap__canvas token-heatmap__canvas--data"
              data-testid="token-heatmap-canvas"
            />
            <canvas
              ref={overlayCanvasRef}
              className="token-heatmap__canvas token-heatmap__canvas--overlay"
              aria-hidden="true"
            />
          </div>
        </div>
        {!hideToolbar && (
          <div
            className="token-heatmap__legend-rail"
            data-testid="token-heatmap-legend-rail"
          >
            <HeatmapLegend
              min={effectiveMin}
              max={effectiveMax}
              valueCol={valueCol}
            />
          </div>
        )}
        {tooltipDatum && hover && (
          <HeatmapTooltip
            datum={tooltipDatum}
            x={hover.x}
            y={hover.y}
            containerWidth={size.w}
            containerHeight={size.h}
            valueCol={valueCol}
          />
        )}
      </div>
      {/* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
    </div>
  );
}
