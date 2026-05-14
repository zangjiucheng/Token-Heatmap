import { useMemo } from 'react';
import type { Trace } from '@/types/trace';
import { TimelineChart } from './TimelineChart';

export interface SelectedProbabilityTimelineProps {
  trace: Trace | null;
  selectedStep: number | null;
  hoveredStep: number | null;
  onSelectStep: (step: number) => void;
  onHoverStep: (step: number | null) => void;
  width?: number;
  height?: number;
  stepRange?: [number, number];
}

export function SelectedProbabilityTimeline({
  trace,
  selectedStep,
  hoveredStep,
  onSelectStep,
  onHoverStep,
  width,
  height,
  stepRange,
}: SelectedProbabilityTimelineProps) {
  const values = useMemo(() => {
    if (!trace) return [];
    return trace.steps.map((s) => s.processed.selected_prob);
  }, [trace]);

  if (!trace) {
    return (
      <div
        className="timeline-chart"
        data-testid="selected-probability-timeline"
      >
        <h4 className="timeline-chart__title">Selected token probability</h4>
        <div className="timeline-chart__empty">No trace loaded</div>
      </div>
    );
  }

  return (
    <TimelineChart
      title="Selected token probability"
      values={values}
      selectedStep={selectedStep}
      hoveredStep={hoveredStep}
      onSelectStep={onSelectStep}
      onHoverStep={onHoverStep}
      yLabel="P(selected)"
      ariaLabel="Per-step selected token probability timeline"
      width={width}
      height={height}
      stepRange={stepRange}
      testId="selected-probability-timeline"
    />
  );
}
