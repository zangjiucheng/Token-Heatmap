import { useEffect, useId, useState } from 'react';
import './ColorRangeControls.css';

export type ColorRangeMode = 'auto' | 'manual';

export interface ColorRangeValue {
  mode: ColorRangeMode;
  /** Used only when mode === 'manual'. */
  min: number | null;
  /** Used only when mode === 'manual'. */
  max: number | null;
}

export interface ColorRangeControlsProps {
  value: ColorRangeValue;
  onChange: (next: ColorRangeValue) => void;
  /** Auto-mode bounds, surfaced so the user has a reference when switching to manual. */
  autoMin: number;
  autoMax: number;
  label?: string;
}

function parseInput(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === '') return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function isInvalid(
  rawMin: string,
  rawMax: string,
): { invalid: boolean; message: string } {
  const minParsed = parseInput(rawMin);
  const maxParsed = parseInput(rawMax);
  if (rawMin.trim() !== '' && minParsed === null) {
    return { invalid: true, message: 'Min must be a number' };
  }
  if (rawMax.trim() !== '' && maxParsed === null) {
    return { invalid: true, message: 'Max must be a number' };
  }
  if (minParsed !== null && maxParsed !== null && minParsed > maxParsed) {
    return { invalid: true, message: 'Min must be ≤ max' };
  }
  return { invalid: false, message: '' };
}

export function ColorRangeControls({
  value,
  onChange,
  autoMin,
  autoMax,
  label = 'Color range',
}: ColorRangeControlsProps) {
  const id = useId();
  const [draftMin, setDraftMin] = useState<string>(
    value.min == null ? '' : String(value.min),
  );
  const [draftMax, setDraftMax] = useState<string>(
    value.max == null ? '' : String(value.max),
  );

  // Sync drafts when the controlled value changes from outside (e.g. URL state).
  useEffect(() => {
    setDraftMin(value.min == null ? '' : String(value.min));
    setDraftMax(value.max == null ? '' : String(value.max));
  }, [value.min, value.max]);

  const handleModeChange = (mode: ColorRangeMode) => {
    if (mode === 'auto') {
      onChange({ mode: 'auto', min: null, max: null });
      return;
    }
    // Seed manual mode from the current auto bounds for a sensible default.
    const seedMin = Number.isFinite(autoMin) ? autoMin : null;
    const seedMax = Number.isFinite(autoMax) ? autoMax : null;
    setDraftMin(seedMin == null ? '' : String(seedMin));
    setDraftMax(seedMax == null ? '' : String(seedMax));
    onChange({ mode: 'manual', min: seedMin, max: seedMax });
  };

  const validity = isInvalid(draftMin, draftMax);

  const commitDrafts = (nextMin: string, nextMax: string) => {
    const v = isInvalid(nextMin, nextMax);
    if (v.invalid) return;
    const minParsed = parseInput(nextMin);
    const maxParsed = parseInput(nextMax);
    onChange({ mode: 'manual', min: minParsed, max: maxParsed });
  };

  return (
    <fieldset
      className="color-range-controls"
      data-testid="color-range-controls"
      aria-label={label}
    >
      <legend className="color-range-controls__legend">{label}</legend>
      <div className="color-range-controls__modes" role="radiogroup">
        <label className="color-range-controls__mode">
          <input
            type="radio"
            name={`${id}-mode`}
            value="auto"
            checked={value.mode === 'auto'}
            onChange={() => handleModeChange('auto')}
            data-testid="color-range-mode-auto"
          />
          <span>Auto</span>
        </label>
        <label className="color-range-controls__mode">
          <input
            type="radio"
            name={`${id}-mode`}
            value="manual"
            checked={value.mode === 'manual'}
            onChange={() => handleModeChange('manual')}
            data-testid="color-range-mode-manual"
          />
          <span>Manual</span>
        </label>
      </div>
      <div className="color-range-controls__inputs">
        <label className="color-range-controls__input">
          <span>Min</span>
          <input
            type="text"
            inputMode="decimal"
            disabled={value.mode !== 'manual'}
            value={draftMin}
            aria-invalid={validity.invalid || undefined}
            aria-label="Color range min"
            onChange={(e) => {
              const next = e.target.value;
              setDraftMin(next);
              commitDrafts(next, draftMax);
            }}
            data-testid="color-range-min"
          />
        </label>
        <label className="color-range-controls__input">
          <span>Max</span>
          <input
            type="text"
            inputMode="decimal"
            disabled={value.mode !== 'manual'}
            value={draftMax}
            aria-invalid={validity.invalid || undefined}
            aria-label="Color range max"
            onChange={(e) => {
              const next = e.target.value;
              setDraftMax(next);
              commitDrafts(draftMin, next);
            }}
            data-testid="color-range-max"
          />
        </label>
      </div>
      {validity.invalid && value.mode === 'manual' && (
        <p
          className="color-range-controls__error"
          role="alert"
          data-testid="color-range-error"
        >
          {validity.message}
        </p>
      )}
    </fieldset>
  );
}
