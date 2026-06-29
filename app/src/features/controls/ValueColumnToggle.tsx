import type { ValueCol } from '@/features/heatmap';
import './ValueColumnToggle.css';

export interface ValueColumnToggleProps {
  value: ValueCol;
  onChange: (next: ValueCol) => void;
  label?: string;
}

const OPTIONS: ReadonlyArray<{ value: ValueCol; label: string }> = [
  { value: 'logprob', label: 'logprob' },
  { value: 'prob', label: 'prob' },
];

export function ValueColumnToggle({
  value,
  onChange,
  label = 'Value column',
}: ValueColumnToggleProps) {
  return (
    <label className="value-column-toggle" data-testid="value-column-toggle">
      <span className="value-column-toggle__label">{label}</span>
      <select
        className="value-column-toggle__select"
        value={value}
        onChange={(e) => onChange(e.target.value as ValueCol)}
        aria-label={label}
        data-testid="value-column-select"
      >
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}
