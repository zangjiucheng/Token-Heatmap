import { useState } from 'react';
import type { LogitLensCandidate, Trace } from '@/types/trace';
import { escapeToken } from '@/features/detail/escapeToken';
import { LogitLensTable, type LogitLensTokenizer } from './LogitLensTable';
import './AttentionTab.css';
import './LogitLensTable.css';

export interface LogitLensTabProps {
  trace: Trace;
  selectedStep: number | null;
  tokenizer?: LogitLensTokenizer;
}

interface PromptLensLayer {
  layer_idx: number;
  top_k: LogitLensCandidate[];
}
interface PromptLensPosition {
  position: number;
  token: string;
  token_id: number;
  layers: PromptLensLayer[];
}
interface PromptLogitLens {
  top_k?: number;
  num_layers?: number;
  positions: PromptLensPosition[];
}

function tokenText(
  cand: LogitLensCandidate,
  tokenizer: LogitLensTokenizer | undefined,
): string {
  if (cand.token) return escapeToken(cand.token);
  if (tokenizer) return escapeToken(tokenizer.decode(cand.token_id));
  return `#${cand.token_id}`;
}

function formatProb(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (v >= 0.001 || v === 0) return v.toFixed(4);
  return v.toExponential(2);
}

/** Per-layer top-k decode of the residual over one PROMPT position. */
function PromptPositionTable({
  position,
  topK = 3,
  tokenizer,
}: {
  position: PromptLensPosition;
  topK?: number;
  tokenizer?: LogitLensTokenizer;
}) {
  const cols = Array.from({ length: topK }, (_, i) => i);
  const sorted = position.layers.slice().sort((a, b) => a.layer_idx - b.layer_idx);
  return (
    <div className="logit-lens-table" data-testid="prompt-logit-lens-table">
      <table aria-label="Prompt-position logit lens predictions per layer">
        <thead>
          <tr>
            <th scope="col">Layer</th>
            {cols.map((i) => (
              <th key={i} scope="col">
                Top {i + 1}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((layer) => (
            <tr key={layer.layer_idx} data-testid={`prompt-logit-lens-row-${layer.layer_idx}`}>
              <th scope="row">L{layer.layer_idx}</th>
              {cols.map((i) => {
                const cand = layer.top_k[i];
                if (!cand) {
                  return (
                    <td key={i} className="logit-lens-table__cell--empty">
                      —
                    </td>
                  );
                }
                const widthPct = Math.max(2, Math.min(100, cand.prob * 100));
                return (
                  <td
                    key={i}
                    data-testid={`prompt-logit-lens-cell-${layer.layer_idx}-${i}`}
                    data-token-id={cand.token_id}
                  >
                    <div className="logit-lens-table__cell-inner">
                      <span
                        className="logit-lens-table__heat-bar"
                        style={{ width: `${widthPct}%` }}
                        aria-hidden="true"
                      />
                      <span className="logit-lens-table__token" title={cand.token}>
                        {tokenText(cand, tokenizer)}
                      </span>
                      <span className="logit-lens-table__prob">{formatProb(cand.prob)}</span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LogitLensTab({ trace, selectedStep, tokenizer }: LogitLensTabProps) {
  const pll = (trace as { prompt_logit_lens?: PromptLogitLens }).prompt_logit_lens;
  const promptPositions = pll?.positions ?? [];
  const hasStepLens = trace.steps.some(
    (s) => Array.isArray(s.logit_lens) && s.logit_lens.length > 0,
  );
  const hasPromptLens = promptPositions.length > 0;

  // 'answer' = the per-step (generated) lens; a number = a prompt position.
  const [view, setView] = useState<'answer' | number>('answer');

  if (!hasStepLens && !hasPromptLens) {
    return (
      <div
        className="attention-tab attention-tab--empty"
        data-testid="logit-lens-tab-empty"
        role="region"
        aria-label="Logit lens tab empty state"
      >
        <p>
          This trace was generated without <code>--capture-logit-lens</code>.
          Re-run the CLI with that flag to inspect per-layer predictions.
        </p>
      </div>
    );
  }

  const promptPos =
    typeof view === 'number'
      ? (promptPositions.find((p) => p.position === view) ?? null)
      : null;

  return (
    <div
      className="attention-tab"
      data-testid="logit-lens-tab-content"
      role="region"
      aria-label="Logit lens tab"
    >
      {hasPromptLens && (
        <div className="logit-lens-tab__controls">
          <label htmlFor="logit-lens-position">Position</label>
          <select
            id="logit-lens-position"
            data-testid="logit-lens-position-select"
            value={typeof view === 'number' ? String(view) : 'answer'}
            onChange={(e) =>
              setView(e.target.value === 'answer' ? 'answer' : Number(e.target.value))
            }
          >
            <option value="answer">Answer (generated)</option>
            {promptPositions.map((p) => (
              <option key={p.position} value={p.position}>
                {p.position}: {escapeToken(p.token)}
              </option>
            ))}
          </select>
          {promptPos && (
            <p className="logit-lens-tab__hint">
              Decoding the residual over <code>{escapeToken(promptPos.token)}</code> at
              each layer — an intermediate token can surface in the middle layers
              (e.g. a bridging entity in multi-hop recall).
            </p>
          )}
        </div>
      )}
      {promptPos ? (
        <PromptPositionTable position={promptPos} tokenizer={tokenizer} />
      ) : (
        <LogitLensTable trace={trace} selectedStep={selectedStep} tokenizer={tokenizer} />
      )}
    </div>
  );
}
