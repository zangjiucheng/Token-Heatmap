import { useMemo } from 'react';
import type { Trace } from '@/types/trace';
import { TimelineChart } from './TimelineChart';

export interface EntropyTimelineProps {
  trace: Trace | null;
  selectedStep: number | null;
  hoveredStep: number | null;
  onSelectStep: (step: number) => void;
  onHoverStep: (step: number | null) => void;
  width?: number;
  height?: number;
  stepRange?: [number, number];
}

export function EntropyTimeline({
  trace,
  selectedStep,
  hoveredStep,
  onSelectStep,
  onHoverStep,
  width,
  height,
  stepRange,
}: EntropyTimelineProps) {
  const values = useMemo(() => {
    if (!trace) return [];
    return trace.steps.map((s) => s.processed.entropy);
  }, [trace]);

  if (!trace) {
    return (
      <div className="timeline-chart" data-testid="entropy-timeline">
        <h4 className="timeline-chart__title">Entropy</h4>
        <div className="timeline-chart__empty">No trace loaded</div>
      </div>
    );
  }

  return (
    <TimelineChart
      title="Entropy"
      values={values}
      selectedStep={selectedStep}
      hoveredStep={hoveredStep}
      onSelectStep={onSelectStep}
      onHoverStep={onHoverStep}
      yLabel="entropy (nats)"
      ariaLabel="Per-step entropy timeline"
      width={width}
      height={height}
      stepRange={stepRange}
      testId="entropy-timeline"
    />
  );
}
