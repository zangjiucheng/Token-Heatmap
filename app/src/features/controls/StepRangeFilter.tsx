import { useCallback } from 'react';
import './StepRangeFilter.css';

export interface StepRangeFilterProps {
  /** Inclusive lower bound of the trace (typically 0). */
  min: number;
  /** Inclusive upper bound of the trace (`steps - 1`). */
  max: number;
  /** Current `[start, end]` selection (inclusive). */
  value: [number, number];
  onChange: (next: [number, number]) => void;
  label?: string;
}

function clamp(v: number, min: number, max: number): number {
  if (Number.isNaN(v)) return min;
  return Math.max(min, Math.min(max, v));
}

export function StepRangeFilter({
  min,
  max,
  value,
  onChange,
  label = 'Step range',
}: StepRangeFilterProps) {
  const [start, end] = value;
  const handleStart = useCallback(
    (next: number) => {
      const clamped = clamp(next, min, end);
      onChange([clamped, end]);
    },
    [min, end, onChange],
  );
  const handleEnd = useCallback(
    (next: number) => {
      const clamped = clamp(next, start, max);
      onChange([start, clamped]);
    },
    [max, start, onChange],
  );

  const disabled = max <= min;

  return (
    <div className="step-range-filter" data-testid="step-range-filter">
      <div className="step-range-filter__header">
        <span className="step-range-filter__label">{label}</span>
        <span
          className="step-range-filter__readout"
          data-testid="step-range-filter-readout"
        >
          {start} – {end}
        </span>
      </div>
      <div className="step-range-filter__inputs">
        <label className="step-range-filter__thumb">
          <span className="step-range-filter__thumb-label">Start</span>
          <input
            type="range"
            min={min}
            max={max}
            step={1}
            value={start}
            disabled={disabled}
            onChange={(e) => handleStart(Number(e.target.value))}
            aria-label={`${label} start`}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={start}
            data-testid="step-range-filter-start"
          />
        </label>
        <label className="step-range-filter__thumb">
          <span className="step-range-filter__thumb-label">End</span>
          <input
            type="range"
            min={min}
            max={max}
            step={1}
            value={end}
            disabled={disabled}
            onChange={(e) => handleEnd(Number(e.target.value))}
            aria-label={`${label} end`}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={end}
            data-testid="step-range-filter-end"
          />
        </label>
      </div>
    </div>
  );
}
