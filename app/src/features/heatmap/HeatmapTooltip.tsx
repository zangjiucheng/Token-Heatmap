import type { ValueCol } from './buildGrid';
import './HeatmapTooltip.css';

export interface HeatmapTooltipDatum {
  step: number;
  rank: number;
  /** Candidate token text, raw (not HTML-escaped); the component escapes on render. */
  token: string;
  prob: number;
  logprob: number;
  kUsed: number;
  entropy: number;
}

export interface HeatmapTooltipProps {
  datum: HeatmapTooltipDatum;
  /** Cursor position in container-relative pixels. */
  x: number;
  y: number;
  /** Container size so we can flip the tooltip when near an edge. */
  containerWidth: number;
  containerHeight: number;
  valueCol: ValueCol;
}

const OFFSET = 12;
const ESTIMATED_WIDTH = 240;
const ESTIMATED_HEIGHT = 140;

/**
 * Render token text safely. We display the raw bytes verbatim — React's text
 * interpolation already HTML-escapes the value, so no manual escaping is
 * required. We do replace whitespace-only tokens with a visible glyph so the
 * tooltip remains informative.
 */
function renderTokenDisplay(token: string): string {
  if (token === '') return '∅';
  if (token === '\n') return '\\n';
  if (token === '\t') return '\\t';
  if (token === '\r') return '\\r';
  return token;
}

export function HeatmapTooltip({
  datum,
  x,
  y,
  containerWidth,
  containerHeight,
  valueCol,
}: HeatmapTooltipProps) {
  const flipX = x + ESTIMATED_WIDTH + OFFSET > containerWidth;
  const flipY = y + ESTIMATED_HEIGHT + OFFSET > containerHeight;
  const left = flipX ? x - ESTIMATED_WIDTH - OFFSET : x + OFFSET;
  const top = flipY ? y - ESTIMATED_HEIGHT - OFFSET : y + OFFSET;

  return (
    <div
      className="heatmap-tooltip"
      role="tooltip"
      data-testid="heatmap-tooltip"
      style={{ left: Math.max(0, left), top: Math.max(0, top) }}
    >
      <div className="heatmap-tooltip__token" data-testid="heatmap-tooltip-token">
        {renderTokenDisplay(datum.token)}
      </div>
      <dl className="heatmap-tooltip__grid">
        <dt>Step</dt>
        <dd data-testid="heatmap-tooltip-step">{datum.step}</dd>
        <dt>Rank</dt>
        <dd data-testid="heatmap-tooltip-rank">{datum.rank}</dd>
        <dt>Prob</dt>
        <dd data-testid="heatmap-tooltip-prob">{datum.prob.toFixed(4)}</dd>
        <dt>Logprob</dt>
        <dd data-testid="heatmap-tooltip-logprob">{datum.logprob.toFixed(4)}</dd>
        <dt>k_used</dt>
        <dd data-testid="heatmap-tooltip-kused">{datum.kUsed}</dd>
        <dt>Entropy</dt>
        <dd data-testid="heatmap-tooltip-entropy">{datum.entropy.toFixed(4)}</dd>
      </dl>
      <div className="heatmap-tooltip__footer" aria-hidden="true">
        coloring: {valueCol}
      </div>
    </div>
  );
}
