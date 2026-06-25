import { useMemo } from 'react';
import type { Trace, DirectLogitAttributionStep } from '@/types/trace';
import { InterventionPanel } from './InterventionPanel';
import './DirectLogitAttributionTab.css';
import './InterventionPanel.css';

export interface DirectLogitAttributionTabProps {
  trace: Trace;
  selectedStep: number | null;
}

interface Component {
  key: string;
  label: string;
  value: number;
}

function pickStep(
  steps: DirectLogitAttributionStep[],
  selectedStep: number | null,
): DirectLogitAttributionStep {
  if (selectedStep != null) {
    const match = steps.find((s) => s.step === selectedStep);
    if (match) return match;
  }
  return steps[0];
}

/**
 * Direct Logit Attribution lens — "why this token?". Decomposes the selected
 * step's realized next-token logit into per-layer attention/MLP contributions
 * (+ the residual input), sorted by impact, with an explicit unexplained-error
 * bar so the faithfulness of the decomposition is visible (epic 01 / 04).
 */
export function DirectLogitAttributionTab({
  trace,
  selectedStep,
}: DirectLogitAttributionTabProps) {
  const dla = trace.direct_logit_attribution;

  const view = useMemo(() => {
    const steps = dla?.steps ?? [];
    if (steps.length === 0) return null;
    const step = pickStep(steps, selectedStep);
    const components: Component[] = [
      { key: 'embed', label: 'embed', value: step.embed ?? 0 },
      ...step.layers.flatMap((l) => [
        { key: `L${l.layer}.attn`, label: `L${l.layer} · attn`, value: l.attn },
        { key: `L${l.layer}.mlp`, label: `L${l.layer} · mlp`, value: l.mlp },
      ]),
    ];
    if ((step.bias ?? 0) !== 0) {
      components.push({ key: 'bias', label: 'norm bias', value: step.bias ?? 0 });
    }
    const error = step.error ?? 0;
    const maxAbs = Math.max(
      1e-9,
      ...components.map((c) => Math.abs(c.value)),
      Math.abs(error),
    );
    const sorted = [...components].sort(
      (a, b) => Math.abs(b.value) - Math.abs(a.value),
    );
    const total = step.total_logit;
    const explained = total - error;
    const errorPct =
      Math.abs(total) > 1e-9 ? (Math.abs(error) / Math.abs(total)) * 100 : 0;
    const token =
      trace.steps.find((s) => s.step === step.step)?.selected.token ?? '';
    return { step, sorted, error, maxAbs, total, explained, errorPct, token };
  }, [dla, selectedStep, trace.steps]);

  if (!view) {
    return (
      <div
        className="dla-tab dla-tab--empty"
        data-testid="direct-logit-attribution-tab-empty"
      >
        <p>
          No direct logit attribution in this trace. Re-run the CLI with{' '}
          <code>--capture-full-activations</code> to decompose each token’s logit
          by layer.
        </p>
      </div>
    );
  }

  const { step, sorted, error, maxAbs, total, explained, errorPct, token } =
    view;

  return (
    <div className="dla-tab" data-testid="direct-logit-attribution-tab-content">
      <header className="dla-tab__header">
        <div>
          <p className="eyebrow">Direct logit attribution</p>
          <h3 className="dla-tab__title">
            Why <code className="dla-tab__token">{JSON.stringify(token)}</code>?
          </h3>
        </div>
        <dl className="dla-tab__summary" aria-label="Attribution summary">
          <div>
            <dt>Logit</dt>
            <dd data-numeric>{total.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Explained</dt>
            <dd data-numeric>{explained.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Error</dt>
            <dd data-numeric>
              {error.toFixed(3)} ({errorPct.toFixed(1)}%)
            </dd>
          </div>
        </dl>
      </header>

      <p className="dla-tab__note">
        Each bar is a component’s direct contribution to the token’s logit
        (orange promotes, blue suppresses), folding the final norm. Sorted by
        impact. Direct/OV path only — it doesn’t explain how attention patterns
        form.
      </p>

      <ul className="dla-tab__bars" data-testid="dla-bars">
        {sorted.map((c) => (
          <Bar key={c.key} label={c.label} value={c.value} maxAbs={maxAbs} />
        ))}
        <Bar
          label="unexplained (error)"
          value={error}
          maxAbs={maxAbs}
          muted
        />
      </ul>

      <InterventionPanel trace={trace} step={step} />
    </div>
  );
}

function Bar({
  label,
  value,
  maxAbs,
  muted = false,
}: {
  label: string;
  value: number;
  maxAbs: number;
  muted?: boolean;
}) {
  const pct = Math.min(100, (Math.abs(value) / maxAbs) * 100);
  const positive = value >= 0;
  return (
    <li className="dla-bar" data-muted={muted ? 'true' : 'false'}>
      <span className="dla-bar__label" title={label}>
        {label}
      </span>
      <span className="dla-bar__track">
        <span
          className="dla-bar__fill"
          data-sign={positive ? 'pos' : 'neg'}
          style={{ width: `${pct / 2}%` }}
        />
      </span>
      <span className="dla-bar__value" data-numeric>
        {value >= 0 ? '+' : ''}
        {value.toFixed(3)}
      </span>
    </li>
  );
}
