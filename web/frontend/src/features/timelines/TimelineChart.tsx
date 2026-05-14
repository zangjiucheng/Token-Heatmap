import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from 'react';
import { TimelineCursor } from './TimelineCursor';
import './TimelineChart.css';

export interface TimelineChartProps {
  title: string;
  /** Per-step values. Length defines the number of steps. */
  values: number[];
  selectedStep: number | null;
  hoveredStep: number | null;
  onSelectStep: (step: number) => void;
  onHoverStep: (step: number | null) => void;
  /** Optional y-axis label (e.g. "entropy"). */
  yLabel?: string;
  /** Fixed width for tests; otherwise auto-fills container. */
  width?: number;
  height?: number;
  /** Optional aria-label override. */
  ariaLabel?: string;
  /** Test id for the root container. */
  testId?: string;
  /**
   * Inclusive step window `[startStep, endStep]` to render. Steps outside the
   * range are dropped from the rendered line, points, and x-axis. Defaults to
   * the full series.
   */
  stepRange?: [number, number];
}

const PADDING_LEFT = 32;
const PADDING_RIGHT = 8;
const PADDING_TOP = 6;
const PADDING_BOTTOM = 18;

function formatTickValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 100 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) {
    return v.toExponential(1);
  }
  return v.toFixed(2);
}

/**
 * Compute the step index whose centre is closest to a given x coordinate.
 * Returns null if x is outside the plot area or there are no steps.
 */
function stepAtX(
  x: number,
  plotX: number,
  plotWidth: number,
  stepStart: number,
  visibleStepCount: number,
): number | null {
  if (visibleStepCount <= 0) return null;
  if (x < plotX || x > plotX + plotWidth) return null;
  const relX = x - plotX;
  const stepWidth = plotWidth / visibleStepCount;
  const col = Math.floor(relX / stepWidth);
  const step = stepStart + col;
  return Math.max(stepStart, Math.min(stepStart + visibleStepCount - 1, step));
}

export function TimelineChart({
  title,
  values,
  selectedStep,
  hoveredStep,
  onSelectStep,
  onHoverStep,
  yLabel,
  width: widthProp,
  height: heightProp,
  ariaLabel,
  testId,
  stepRange,
}: TimelineChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({
    w: widthProp ?? 400,
    h: heightProp ?? 120,
  });

  useLayoutEffect(() => {
    if (widthProp != null && heightProp != null) return undefined;
    const container = containerRef.current;
    if (!container) return undefined;
    const measure = () => {
      const rect = container.getBoundingClientRect();
      setSize({
        w: widthProp ?? Math.max(120, Math.floor(rect.width)),
        h: heightProp ?? Math.max(80, Math.floor(rect.height)),
      });
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(container);
    return () => ro.disconnect();
  }, [widthProp, heightProp]);

  const totalSteps = values.length;
  const [stepStart, stepEnd] = useMemo<[number, number]>(() => {
    if (totalSteps === 0) return [0, -1];
    if (!stepRange) return [0, totalSteps - 1];
    const rawStart = Math.max(0, Math.min(totalSteps - 1, stepRange[0]));
    const rawEnd = Math.max(0, Math.min(totalSteps - 1, stepRange[1]));
    return rawStart <= rawEnd ? [rawStart, rawEnd] : [rawEnd, rawStart];
  }, [totalSteps, stepRange]);
  const visibleStepCount = Math.max(1, stepEnd - stepStart + 1);

  const {
    plotX,
    plotY,
    plotWidth,
    plotHeight,
    yMin,
    yMax,
    pointsPath,
    points,
    pointByStep,
  } = useMemo(() => {
      const pw = Math.max(1, size.w - PADDING_LEFT - PADDING_RIGHT);
      const ph = Math.max(1, size.h - PADDING_TOP - PADDING_BOTTOM);
      let min = Infinity;
      let max = -Infinity;
      for (let i = stepStart; i <= stepEnd; i += 1) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        min = 0;
        max = 1;
      } else if (min === max) {
        const eps = Math.abs(min) > 1 ? Math.abs(min) * 0.1 : 0.1;
        min -= eps;
        max += eps;
      }
      const denom = max - min;
      const stepWidth = pw / visibleStepCount;
      const computedPoints: Array<{ x: number; y: number; step: number }> = [];
      const byStep = new Map<number, { x: number; y: number }>();
      for (let step = stepStart; step <= stepEnd; step += 1) {
        const v = values[step];
        const col = step - stepStart;
        const x = PADDING_LEFT + stepWidth * col + stepWidth / 2;
        const y = !Number.isFinite(v)
          ? PADDING_TOP + ph
          : PADDING_TOP + ph - ((v - min) / denom) * ph;
        computedPoints.push({ x, y, step });
        byStep.set(step, { x, y });
      }
      const path = computedPoints
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
        .join(' ');
      return {
        plotX: PADDING_LEFT,
        plotY: PADDING_TOP,
        plotWidth: pw,
        plotHeight: ph,
        yMin: min,
        yMax: max,
        pointsPath: path,
        points: computedPoints,
        pointByStep: byStep,
      };
    }, [size, values, stepStart, stepEnd, visibleStepCount]);

  const handleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const step = stepAtX(x, plotX, plotWidth, stepStart, visibleStepCount);
      if (step != null) {
        onSelectStep(step);
        container.focus();
      }
    },
    [plotX, plotWidth, stepStart, visibleStepCount, onSelectStep],
  );

  const handleMouseMove = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const step = stepAtX(x, plotX, plotWidth, stepStart, visibleStepCount);
      if (step !== hoveredStep) onHoverStep(step);
    },
    [plotX, plotWidth, stepStart, visibleStepCount, hoveredStep, onHoverStep],
  );

  const handleMouseLeave = useCallback(() => {
    onHoverStep(null);
  }, [onHoverStep]);

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (totalSteps === 0) return;
      const current = selectedStep ?? stepStart;
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          onSelectStep(Math.max(stepStart, current - 1));
          return;
        case 'ArrowRight':
          event.preventDefault();
          onSelectStep(Math.min(stepEnd, current + 1));
          return;
        case 'Home':
          event.preventDefault();
          onSelectStep(stepStart);
          return;
        case 'End':
          event.preventDefault();
          onSelectStep(stepEnd);
          return;
        default:
          return;
      }
    },
    [totalSteps, selectedStep, onSelectStep, stepStart, stepEnd],
  );

  const xTickStride = visibleStepCount > 20 ? Math.ceil(visibleStepCount / 8) : 1;

  if (totalSteps === 0) {
    return (
      <div
        ref={containerRef}
        className="timeline-chart"
        data-testid={testId}
      >
        <h4 className="timeline-chart__title">{title}</h4>
        <div className="timeline-chart__empty">No data</div>
      </div>
    );
  }

  return (
    /* eslint-disable jsx-a11y/no-noninteractive-element-interactions,
                      jsx-a11y/no-noninteractive-tabindex */
    <div
      ref={containerRef}
      className="timeline-chart"
      role="figure"
      aria-label={ariaLabel ?? `${title} timeline`}
      tabIndex={0}
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onKeyDown={handleKeyDown}
      data-testid={testId}
      style={
        widthProp != null && heightProp != null
          ? { width: widthProp, height: heightProp }
          : undefined
      }
    >
      <h4 className="timeline-chart__title">{title}</h4>
      <svg
        className="timeline-chart__svg"
        width={size.w}
        height={size.h - 18}
        viewBox={`0 0 ${size.w} ${size.h - 18}`}
        data-testid={testId ? `${testId}-svg` : undefined}
        aria-hidden="true"
      >
        <line
          className="timeline-chart__axis"
          x1={plotX}
          x2={plotX}
          y1={plotY}
          y2={plotY + plotHeight}
        />
        <line
          className="timeline-chart__axis"
          x1={plotX}
          x2={plotX + plotWidth}
          y1={plotY + plotHeight}
          y2={plotY + plotHeight}
        />

        <text
          className="timeline-chart__axis-label"
          x={4}
          y={plotY + 4}
          dominantBaseline="hanging"
          fontSize={10}
        >
          {formatTickValue(yMax)}
        </text>
        <text
          className="timeline-chart__axis-label"
          x={4}
          y={plotY + plotHeight}
          dominantBaseline="ideographic"
          fontSize={10}
        >
          {formatTickValue(yMin)}
        </text>
        {yLabel && (
          <text
            className="timeline-chart__axis-label"
            x={4}
            y={plotY + plotHeight / 2}
            fontSize={10}
            transform={`rotate(-90 4 ${plotY + plotHeight / 2})`}
          >
            {yLabel}
          </text>
        )}

        {points.length > 1 && (
          <path
            className="timeline-chart__line"
            d={pointsPath}
            data-testid={testId ? `${testId}-line` : undefined}
          />
        )}

        {hoveredStep != null && pointByStep.has(hoveredStep) && (
          <circle
            cx={pointByStep.get(hoveredStep)!.x}
            cy={pointByStep.get(hoveredStep)!.y}
            r={3}
            className="timeline-chart__point timeline-chart__point--hover"
            data-testid={testId ? `${testId}-hover-point` : undefined}
          />
        )}

        {selectedStep != null && pointByStep.has(selectedStep) && (
          <circle
            cx={pointByStep.get(selectedStep)!.x}
            cy={pointByStep.get(selectedStep)!.y}
            r={3.5}
            className="timeline-chart__point"
            data-testid={testId ? `${testId}-selected-point` : undefined}
          />
        )}

        <TimelineCursor
          step={hoveredStep}
          totalSteps={totalSteps}
          stepStart={stepStart}
          visibleStepCount={visibleStepCount}
          plotWidth={plotWidth}
          plotHeight={plotHeight}
          plotX={plotX}
          plotY={plotY}
          variant="hover"
          stroke="#d55e00"
          testId={testId ? `${testId}-hover-cursor` : 'timeline-hover-cursor'}
        />
        <TimelineCursor
          step={selectedStep}
          totalSteps={totalSteps}
          stepStart={stepStart}
          visibleStepCount={visibleStepCount}
          plotWidth={plotWidth}
          plotHeight={plotHeight}
          plotX={plotX}
          plotY={plotY}
          variant="selected"
          testId={testId ? `${testId}-cursor` : 'timeline-cursor'}
        />

        {Array.from({ length: Math.ceil(visibleStepCount / xTickStride) }).map(
          (_, i) => {
            const col = i * xTickStride;
            if (col >= visibleStepCount) return null;
            const step = stepStart + col;
            const stepWidth = plotWidth / visibleStepCount;
            const x = plotX + stepWidth * col + stepWidth / 2;
            return (
              <text
                key={step}
                className="timeline-chart__axis-label"
                x={x}
                y={plotY + plotHeight + 12}
                textAnchor="middle"
                fontSize={10}
              >
                {step}
              </text>
            );
          },
        )}
      </svg>
    </div>
    /* eslint-enable jsx-a11y/no-noninteractive-element-interactions,
                     jsx-a11y/no-noninteractive-tabindex */
  );
}
