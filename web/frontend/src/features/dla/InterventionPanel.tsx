import { useEffect, useMemo, useState } from 'react';
import {
  getApiClient,
  type InterventionResult,
  type InterventionSpec,
} from '@/api/client';
import { useBackendHealth } from '@/hooks/useBackendHealth';
import type { Trace, DirectLogitAttributionStep } from '@/types/trace';

export interface PresetTarget {
  layer: number;
  component: 'attn' | 'mlp' | 'head';
  head?: number;
}

export interface InterventionPanelProps {
  trace: Trace;
  step: DirectLogitAttributionStep;
  /** When set, the panel selects this target and runs it immediately. */
  preset?: PresetTarget | null;
  onPresetConsumed?: () => void;
}

interface ComponentChoice {
  key: string;
  layer: number;
  component: 'attn' | 'mlp' | 'head';
  head?: number;
  label: string;
  value: number;
}

function choiceKey(layer: number, component: string, head?: number): string {
  return component === 'head' ? `${layer}:head:${head}` : `${layer}:${component}`;
}

/**
 * Causal validation for the DLA lens: pick a component (block or head) the
 * attribution ranks, ablate / scale its write to the final residual on the live
 * backend, and see the next-token distribution move. Gated on backend health.
 */
export function InterventionPanel({
  trace,
  step,
  preset,
  onPresetConsumed,
}: InterventionPanelProps) {
  const health = useBackendHealth();
  const online = health.status === 'healthy';

  const choices = useMemo<ComponentChoice[]>(() => {
    const out: ComponentChoice[] = [];
    for (const l of step.layers) {
      out.push({
        key: choiceKey(l.layer, 'attn'),
        layer: l.layer,
        component: 'attn',
        label: `L${l.layer} · attn`,
        value: l.attn,
      });
      out.push({
        key: choiceKey(l.layer, 'mlp'),
        layer: l.layer,
        component: 'mlp',
        label: `L${l.layer} · mlp`,
        value: l.mlp,
      });
      for (const h of l.heads ?? []) {
        out.push({
          key: choiceKey(l.layer, 'head', h.head),
          layer: l.layer,
          component: 'head',
          head: h.head,
          label: `L${l.layer} · head ${h.head}`,
          value: h.attn,
        });
      }
    }
    return out.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }, [step.layers]);

  const [selKey, setSelKey] = useState<string>(() => choices[0]?.key ?? '');
  const [op, setOp] = useState<'zero' | 'scale'>('zero');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InterventionResult | null>(null);

  const sel = choices.find((c) => c.key === selKey) ?? choices[0];

  const run = async (choice?: ComponentChoice) => {
    const c = choice ?? sel;
    if (!c) return;
    setBusy(true);
    setError(null);
    try {
      const continuation = trace.steps
        .filter((s) => s.step < step.step)
        .map((s) => s.selected.token_id);
      const intervention: InterventionSpec = {
        layer: c.layer,
        component: c.component,
        op,
        factor: op === 'scale' ? 2 : 0,
        ...(c.component === 'head' ? { head: c.head } : {}),
      };
      const res = await getApiClient().intervene({
        model: trace.metadata.model,
        prompt: trace.metadata.prompt ?? '',
        continuation_token_ids: continuation,
        interventions: [intervention],
        target_token_id: step.target_token_id ?? null,
        top_k: 8,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Intervention failed');
    } finally {
      setBusy(false);
    }
  };

  // A per-head "ablate" click from the bars selects + runs that target.
  useEffect(() => {
    if (!preset) return;
    const key = choiceKey(preset.layer, preset.component, preset.head);
    const found = choices.find((c) => c.key === key);
    if (found) {
      setSelKey(found.key);
      setOp('zero');
      if (online) void run({ ...found });
    }
    onPresetConsumed?.();
    // Run once per preset change; `run`/`choices` are intentionally omitted.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  return (
    <section
      className="intervention-panel"
      data-testid="intervention-panel"
      aria-label="Intervention"
    >
      <div className="intervention-panel__head">
        <p className="eyebrow">Intervene · causal check</p>
        <p className="intervention-panel__lead">
          Ablate a component and watch the next-token distribution move — the DLA
          bar is a hypothesis; this tests it.
        </p>
      </div>

      {!online ? (
        <p className="intervention-panel__offline" data-testid="intervention-offline">
          Interventions need the live backend (the model is loaded server-side).
          Start it with <code>./scripts/dev.sh</code>, then{' '}
          <button
            type="button"
            className="intervention-panel__link"
            onClick={() => void health.probe()}
          >
            retry
          </button>
          .
        </p>
      ) : (
        <div className="intervention-panel__controls">
          <label className="intervention-panel__field">
            <span>Component</span>
            <select
              value={sel?.key ?? ''}
              onChange={(e) => setSelKey(e.target.value)}
              data-testid="intervention-component"
            >
              {choices.map((c) => (
                <option key={c.key} value={c.key}>
                  {c.label} ({c.value >= 0 ? '+' : ''}
                  {c.value.toFixed(2)})
                </option>
              ))}
            </select>
          </label>
          <label className="intervention-panel__field">
            <span>Op</span>
            <select
              value={op}
              onChange={(e) => setOp(e.target.value as 'zero' | 'scale')}
              data-testid="intervention-op"
            >
              <option value="zero">Ablate (0×)</option>
              <option value="scale">Amplify (2×)</option>
            </select>
          </label>
          <button
            type="button"
            className="intervention-panel__run"
            onClick={() => void run()}
            disabled={busy || !sel}
            data-testid="intervention-run"
          >
            {busy ? 'Running…' : 'Run'}
          </button>
        </div>
      )}

      {error ? (
        <p className="intervention-panel__error" role="alert" data-testid="intervention-error">
          {error}
        </p>
      ) : null}

      {result ? <InterventionResultView result={result} /> : null}
    </section>
  );
}

function fmtPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function InterventionResultView({ result }: { result: InterventionResult }) {
  const { baseline, patched, diff, target_token } = result;
  const dropped = diff.target_prob_delta < 0;
  return (
    <div className="intervention-result" data-testid="intervention-result">
      <div className="intervention-result__headline">
        <span className="intervention-result__kl" data-numeric>
          KL {diff.kl.toFixed(3)} nats
        </span>
        <span
          className="intervention-result__target"
          data-dir={dropped ? 'down' : 'up'}
          data-testid="intervention-target-delta"
        >
          P(<code>{JSON.stringify(target_token)}</code>){' '}
          <span data-numeric>{fmtPct(baseline.target_prob)}</span> →{' '}
          <span data-numeric>{fmtPct(patched.target_prob)}</span>
        </span>
      </div>

      <div className="intervention-result__cols">
        <DistColumn title="Baseline" dist={baseline} />
        <DistColumn title="Patched" dist={patched} />
      </div>

      {diff.top_flips.length > 0 ? (
        <p className="intervention-result__flips">
          Top-token flips:{' '}
          {diff.top_flips
            .map(
              (f) =>
                `#${f.rank} ${JSON.stringify(f.from_token)}→${JSON.stringify(
                  f.to_token,
                )}`,
            )
            .join(', ')}
        </p>
      ) : (
        <p className="intervention-result__flips">No top-token order change.</p>
      )}
    </div>
  );
}

function DistColumn({
  title,
  dist,
}: {
  title: string;
  dist: InterventionResult['baseline'];
}) {
  const max = Math.max(1e-9, ...dist.top.map((t) => t.prob));
  return (
    <div className="intervention-result__col">
      <p className="eyebrow">{title}</p>
      <ul className="intervention-result__list">
        {dist.top.map((t) => (
          <li key={t.token_id} className="intervention-result__row">
            <span className="intervention-result__tok" title={t.token}>
              {JSON.stringify(t.token)}
            </span>
            <span className="intervention-result__bar">
              <span
                className="intervention-result__bar-fill"
                style={{ width: `${(t.prob / max) * 100}%` }}
              />
            </span>
            <span className="intervention-result__p" data-numeric>
              {fmtPct(t.prob)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
