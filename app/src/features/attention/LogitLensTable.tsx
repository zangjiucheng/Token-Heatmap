import { useMemo } from 'react';
import type { LogitLensCandidate, Trace } from '@/types/trace';
import { escapeToken } from '@/features/detail/escapeToken';
import './LogitLensTable.css';

export interface LogitLensTokenizer {
  /** Decode a token id to a string. When absent the table falls back to the
   *  numeric id. */
  decode(tokenId: number): string;
}

export interface LogitLensTableProps {
  trace: Trace;
  selectedStep: number | null;
  tokenizer?: LogitLensTokenizer;
  /** Number of top-k columns to render. Defaults to 3. */
  topK?: number;
}

function renderTokenText(
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

export function LogitLensTable({
  trace,
  selectedStep,
  tokenizer,
  topK = 3,
}: LogitLensTableProps) {
  const step =
    selectedStep != null &&
    selectedStep >= 0 &&
    selectedStep < trace.steps.length
      ? trace.steps[selectedStep]
      : null;
  const layers = step?.logit_lens ?? [];
  const selectedTokenId = step?.selected.token_id ?? null;

  const cols = useMemo(() => Array.from({ length: topK }, (_, i) => i), [topK]);

  if (!step) {
    return (
      <div
        className="logit-lens-table logit-lens-table--empty"
        data-testid="logit-lens-table"
      >
        <p>
          Select a generation step from the token strip, heatmap, or overview
          timelines to inspect logit-lens predictions.
        </p>
      </div>
    );
  }
  if (layers.length === 0) {
    return (
      <div
        className="logit-lens-table logit-lens-table--empty"
        data-testid="logit-lens-table"
      >
        <p>This trace has no logit-lens captures.</p>
      </div>
    );
  }

  // Sort ascending by layer_idx so the bottom row is the deepest captured
  // layer. The AC-relaxed assertion (Q6) becomes: if the deepest captured
  // layer equals num_layers - 1 then its top-1 should equal selected_token.
  const sortedLayers = layers.slice().sort((a, b) => a.layer_idx - b.layer_idx);

  return (
    <div className="logit-lens-table" data-testid="logit-lens-table">
      <table aria-label="Logit lens predictions per captured layer">
        <thead>
          <tr>
            <th scope="col">Layer</th>
            {cols.map((i) => (
              <th key={i} scope="col">
                Top {i + 1}
              </th>
            ))}
            <th scope="col">Entropy</th>
            <th scope="col">Selected rank</th>
          </tr>
        </thead>
        <tbody>
          {sortedLayers.map((layer, rowIdx) => {
            const isFinalRow = rowIdx === sortedLayers.length - 1;
            return (
              <tr
                key={layer.layer_idx}
                data-testid={`logit-lens-row-${layer.layer_idx}`}
                data-final-row={isFinalRow ? 'true' : 'false'}
              >
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
                  const isSelected = cand.token_id === selectedTokenId;
                  const widthPct = Math.max(2, Math.min(100, cand.prob * 100));
                  return (
                    <td
                      key={i}
                      className={
                        isSelected
                          ? 'logit-lens-table__cell--selected'
                          : undefined
                      }
                      data-testid={`logit-lens-cell-${layer.layer_idx}-${i}`}
                      data-token-id={cand.token_id}
                    >
                      <div className="logit-lens-table__cell-inner">
                        <span
                          className="logit-lens-table__heat-bar"
                          style={{ width: `${widthPct}%` }}
                          aria-hidden="true"
                        />
                        <span
                          className="logit-lens-table__token"
                          title={cand.token}
                        >
                          {renderTokenText(cand, tokenizer)}
                        </span>
                        <span className="logit-lens-table__prob">
                          {formatProb(cand.prob)}
                        </span>
                      </div>
                    </td>
                  );
                })}
                <td>{layer.entropy.toFixed(3)}</td>
                <td>{layer.selected_token_rank}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
