import './DistributionToggle.css';

/**
 * Which of the two per-step distributions the heatmap reads. The probe records
 * both at every step: `raw` is the temperature-scaled logits, `processed` is
 * what is left after the top-p / top-k sampling filters.
 */
export type DistributionMode = 'raw' | 'processed';

export interface DistributionToggleProps {
  value: DistributionMode;
  onChange: (mode: DistributionMode) => void;
  /** Optional label override for the fieldset legend. */
  label?: string;
}

const OPTIONS: ReadonlyArray<{ value: DistributionMode; label: string; hint: string }> = [
  { value: 'raw', label: 'Raw', hint: 'Temperature-scaled logits before sampling filters' },
  { value: 'processed', label: 'Processed', hint: 'After top-p / top-k / repetition penalty' },
];

export function DistributionToggle({
  value,
  onChange,
  label = 'Distribution',
}: DistributionToggleProps) {
  return (
    <fieldset
      className="distribution-toggle"
      data-testid="distribution-toggle"
      role="radiogroup"
      aria-label={label}
    >
      <legend className="distribution-toggle__legend">{label}</legend>
      <div className="distribution-toggle__options">
        {OPTIONS.map((opt) => {
          const id = `distribution-toggle-${opt.value}`;
          const selected = value === opt.value;
          return (
            <label
              key={opt.value}
              htmlFor={id}
              className="distribution-toggle__option"
              data-selected={selected ? 'true' : 'false'}
              title={opt.hint}
            >
              <input
                id={id}
                type="radio"
                name="distribution-mode"
                value={opt.value}
                checked={selected}
                onChange={() => onChange(opt.value)}
                data-testid={`distribution-toggle-${opt.value}`}
              />
              <span>{opt.label}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
