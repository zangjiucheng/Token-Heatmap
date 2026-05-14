import './ComparisonToggle.css';

export type ComparisonMode = 'raw' | 'processed' | 'split';

export interface ComparisonToggleProps {
  value: ComparisonMode;
  onChange: (mode: ComparisonMode) => void;
  /** Optional label override for the fieldset legend. */
  label?: string;
}

const OPTIONS: ReadonlyArray<{ value: ComparisonMode; label: string; hint: string }> = [
  { value: 'raw', label: 'Raw', hint: 'Temperature-scaled logits before sampling filters' },
  { value: 'processed', label: 'Processed', hint: 'After top-p / top-k / repetition penalty' },
  { value: 'split', label: 'Split', hint: 'Raw and processed side by side' },
];

export function ComparisonToggle({
  value,
  onChange,
  label = 'Distribution',
}: ComparisonToggleProps) {
  return (
    <fieldset
      className="comparison-toggle"
      data-testid="comparison-toggle"
      role="radiogroup"
      aria-label={label}
    >
      <legend className="comparison-toggle__legend">{label}</legend>
      <div className="comparison-toggle__options">
        {OPTIONS.map((opt) => {
          const id = `comparison-toggle-${opt.value}`;
          const selected = value === opt.value;
          return (
            <label
              key={opt.value}
              htmlFor={id}
              className="comparison-toggle__option"
              data-selected={selected ? 'true' : 'false'}
              title={opt.hint}
            >
              <input
                id={id}
                type="radio"
                name="comparison-mode"
                value={opt.value}
                checked={selected}
                onChange={() => onChange(opt.value)}
                data-testid={`comparison-toggle-${opt.value}`}
              />
              <span>{opt.label}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
