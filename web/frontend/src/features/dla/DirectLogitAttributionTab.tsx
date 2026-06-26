import { Fragment, useMemo, useState } from 'react';
import type {
  Trace,
  DirectLogitAttributionStep,
  DirectLogitAttributionHead,
} from '@/types/trace';
import './DirectLogitAttributionTab.css';

export interface DirectLogitAttributionTabProps {
  trace: Trace;
  selectedStep: number | null;
}

interface Component {
  key: string;
  label: string;
  value: number;
  layer?: number;
  heads?: DirectLogitAttributionHead[];
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
 * (+ residual input), sorted by impact, with an explicit unexplained-error bar.
 * Attention bars expand into per-head contributions (when captured).
 */
export function DirectLogitAttributionTab({
  trace,
  selectedStep,
}: DirectLogitAttributionTabProps) {
  const dla = trace.direct_logit_attribution;
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  const view = useMemo(() => {
    const steps = dla?.steps ?? [];
    if (steps.length === 0) return null;
    const step = pickStep(steps, selectedStep);
    const components: Component[] = [
      { key: 'embed', label: 'embed', value: step.embed ?? 0 },
      ...step.layers.flatMap((l) => [
        {
          key: `L${l.layer}.attn`,
          label: `L${l.layer} · attn`,
          value: l.attn,
          layer: l.layer,
          heads: l.heads,
        },
        { key: `L${l.layer}.mlp`, label: `L${l.layer} · mlp`, value: l.mlp },
      ]),
    ];
    if ((step.bias ?? 0) !== 0) {
      components.push({
        key: 'bias',
        label: 'norm bias',
        value: step.bias ?? 0,
      });
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
          <code>--capture-full-activations</code> to decompose each token’s
          logit by layer.
        </p>
      </div>
    );
  }

  const { sorted, error, maxAbs, total, explained, errorPct, token } = view;

  const toggleExpand = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

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
        (orange promotes, blue suppresses), folding the final norm. Expand an
        attention bar to see per-head contributions. Direct/OV path only — it
        doesn’t explain how attention patterns form.
      </p>

      <ul className="dla-tab__bars" data-testid="dla-bars">
        {sorted.map((c) => {
          const heads = c.heads ?? [];
          const expandable = heads.length > 0 && c.layer != null;
          const isOpen = expanded.has(c.key);
          return (
            <Fragment key={c.key}>
              <Bar
                label={c.label}
                value={c.value}
                maxAbs={maxAbs}
                expandable={expandable}
                expanded={isOpen}
                onToggle={expandable ? () => toggleExpand(c.key) : undefined}
              />
              {expandable && isOpen ? (
                <li className="dla-heads" data-testid={`dla-heads-${c.layer}`}>
                  <ul className="dla-heads__list">
                    {[...heads]
                      .sort((a, b) => Math.abs(b.attn) - Math.abs(a.attn))
                      .map((h) => (
                        <HeadBar
                          key={h.head}
                          head={h.head}
                          value={h.attn}
                          maxAbs={maxAbs}
                        />
                      ))}
                  </ul>
                </li>
              ) : null}
            </Fragment>
          );
        })}
        <Bar label="unexplained (error)" value={error} maxAbs={maxAbs} muted />
      </ul>

      <p className="dla-tab__ablation-note">
        Interactive ablation returns via the CLI precomputing ablations into the
        trace.
      </p>
    </div>
  );
}

function Bar({
  label,
  value,
  maxAbs,
  muted = false,
  expandable = false,
  expanded = false,
  onToggle,
}: {
  label: string;
  value: number;
  maxAbs: number;
  muted?: boolean;
  expandable?: boolean;
  expanded?: boolean;
  onToggle?: () => void;
}) {
  const pct = Math.min(100, (Math.abs(value) / maxAbs) * 100);
  const positive = value >= 0;
  return (
    <li className="dla-bar" data-muted={muted ? 'true' : 'false'}>
      {expandable ? (
        <button
          type="button"
          className="dla-bar__label dla-bar__label--toggle"
          onClick={onToggle}
          aria-expanded={expanded}
          title={label}
        >
          <span aria-hidden="true">{expanded ? '▾' : '▸'}</span> {label}
        </button>
      ) : (
        <span className="dla-bar__label" title={label}>
          {label}
        </span>
      )}
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

function HeadBar({
  head,
  value,
  maxAbs,
}: {
  head: number;
  value: number;
  maxAbs: number;
}) {
  const pct = Math.min(100, (Math.abs(value) / maxAbs) * 100);
  const positive = value >= 0;
  return (
    <li className="dla-head-bar" data-testid={`dla-head-${head}`}>
      <span className="dla-bar__label">head {head}</span>
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
