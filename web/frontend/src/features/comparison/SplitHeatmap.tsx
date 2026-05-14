import { useCallback, useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import { TokenHeatmap } from '@/features/heatmap';
import { buildGrid, type ValueCol } from '@/features/heatmap';
import { HeatmapLegend } from '@/features/heatmap';
import './SplitHeatmap.css';

export interface SplitHeatmapProps {
  trace: Trace;
  valueCol: ValueCol;
  selectedStep: number | null;
  onSelectStep: (step: number) => void;
  /**
   * Inclusive step window forwarded to both panes. Defaults to the full range.
   */
  stepRange?: [number, number];
  /**
   * Optional color-range override forwarded to both panes. When omitted, the
   * combined min/max across the raw + processed grids is used so both panes
   * share a single color scale.
   */
  valueRange?: { min: number; max: number };
  /** Min/max cell sizing forwarded to both panes; both share the same constraints. */
  minCellWidth?: number;
  minCellHeight?: number;
  maxCellWidth?: number | null;
  maxCellHeight?: number | null;
}

function combinedRange(
  rawMin: number,
  rawMax: number,
  procMin: number,
  procMax: number,
): { min: number; max: number } {
  const candidates = [rawMin, rawMax, procMin, procMax].filter((v) =>
    Number.isFinite(v),
  );
  if (candidates.length === 0) return { min: NaN, max: NaN };
  return {
    min: Math.min(...candidates),
    max: Math.max(...candidates),
  };
}

export function SplitHeatmap({
  trace,
  valueCol,
  selectedStep,
  onSelectStep,
  stepRange,
  valueRange,
  minCellWidth,
  minCellHeight,
  maxCellWidth,
  maxCellHeight,
}: SplitHeatmapProps) {
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  // Lifted scroll state: both panes are driven by, and report back to, this
  // single value so scrolling one scrolls the other in lockstep.
  const [sharedScrollLeft, setSharedScrollLeft] = useState(0);

  const handleScrollChange = useCallback((next: number) => {
    setSharedScrollLeft((prev) => (Math.abs(prev - next) < 0.5 ? prev : next));
  }, []);

  // Compute a shared color scale spanning both distributions so the legend is
  // consistent across the two panes.
  const sharedRange = useMemo(() => {
    if (valueRange) return valueRange;
    const rawGrid = buildGrid(trace, valueCol, 'raw');
    const procGrid = buildGrid(trace, valueCol, 'processed');
    return combinedRange(
      rawGrid.valueMin,
      rawGrid.valueMax,
      procGrid.valueMin,
      procGrid.valueMax,
    );
  }, [trace, valueCol, valueRange]);

  return (
    <div className="split-heatmap" data-testid="split-heatmap">
      <div className="split-heatmap__legend">
        <HeatmapLegend
          min={sharedRange.min}
          max={sharedRange.max}
          valueCol={valueCol}
        />
      </div>
      <div className="split-heatmap__panes">
        <section
          className="split-heatmap__pane"
          aria-label="Raw distribution"
          data-testid="split-heatmap-raw"
        >
          <header className="split-heatmap__pane-title">Raw</header>
          <TokenHeatmap
            trace={trace}
            valueCol={valueCol}
            source="raw"
            selectedStep={selectedStep}
            onSelectStep={onSelectStep}
            externalHoveredStep={hoveredStep}
            onHoverStep={setHoveredStep}
            stepRange={stepRange}
            valueRange={sharedRange}
            hideToolbar
            ariaLabelSuffix="raw distribution"
            minCellWidth={minCellWidth}
            minCellHeight={minCellHeight}
            maxCellWidth={maxCellWidth}
            maxCellHeight={maxCellHeight}
            scrollLeft={sharedScrollLeft}
            onScrollChange={handleScrollChange}
          />
        </section>
        <section
          className="split-heatmap__pane"
          aria-label="Processed distribution"
          data-testid="split-heatmap-processed"
        >
          <header className="split-heatmap__pane-title">Processed</header>
          <TokenHeatmap
            trace={trace}
            valueCol={valueCol}
            source="processed"
            selectedStep={selectedStep}
            onSelectStep={onSelectStep}
            externalHoveredStep={hoveredStep}
            onHoverStep={setHoveredStep}
            stepRange={stepRange}
            valueRange={sharedRange}
            hideToolbar
            ariaLabelSuffix="processed distribution"
            minCellWidth={minCellWidth}
            minCellHeight={minCellHeight}
            maxCellWidth={maxCellWidth}
            maxCellHeight={maxCellHeight}
            scrollLeft={sharedScrollLeft}
            onScrollChange={handleScrollChange}
          />
        </section>
      </div>
    </div>
  );
}
