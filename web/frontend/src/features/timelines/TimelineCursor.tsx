export interface TimelineCursorProps {
  /** The step the cursor should snap to, or null to hide. */
  step: number | null;
  /** Total number of steps in the trace (used to compute x position). */
  totalSteps: number;
  /** Inner plot width in SVG user units. */
  plotWidth: number;
  /** Inner plot height (cursor spans this vertically). */
  plotHeight: number;
  /** Plot origin x offset (gutter for y-axis labels). */
  plotX?: number;
  /** Plot origin y offset. */
  plotY?: number;
  /** Stroke colour. */
  stroke?: string;
  /** Visual variant: 'selected' (solid) or 'hover' (dashed, dimmer). */
  variant?: 'selected' | 'hover';
  /** Optional test id. */
  testId?: string;
  /** First visible step (defaults to 0). */
  stepStart?: number;
  /** Number of visible steps (defaults to totalSteps). */
  visibleStepCount?: number;
}

/**
 * Vertical cursor for the entropy / selected-probability timelines.
 * Renders as SVG inside a parent `<svg>` and snaps to the centre of step `step`.
 */
export function TimelineCursor({
  step,
  totalSteps,
  plotWidth,
  plotHeight,
  plotX = 0,
  plotY = 0,
  stroke = 'rgb(0, 114, 178)',
  variant = 'selected',
  testId = 'timeline-cursor',
  stepStart = 0,
  visibleStepCount,
}: TimelineCursorProps) {
  if (step == null || totalSteps <= 0) return null;
  const count = visibleStepCount ?? totalSteps;
  if (count <= 0) return null;
  if (step < stepStart || step >= stepStart + count) return null;

  const stepWidth = plotWidth / count;
  const x = plotX + stepWidth * (step - stepStart) + stepWidth / 2;

  const isHover = variant === 'hover';
  return (
    <line
      x1={x}
      x2={x}
      y1={plotY}
      y2={plotY + plotHeight}
      stroke={stroke}
      strokeWidth={isHover ? 1 : 2}
      strokeDasharray={isHover ? '4 3' : undefined}
      opacity={isHover ? 0.5 : 1}
      data-testid={testId}
      data-step={step}
    />
  );
}
