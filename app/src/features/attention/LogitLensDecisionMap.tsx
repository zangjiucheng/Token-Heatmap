import { useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import { escapeToken } from '@/features/detail/escapeToken';
import { rampColor } from '@/features/manifold/colorRamp';
import './LogitLensDecisionMap.css';

export interface LogitLensDecisionMapProps {
  trace: Trace;
  selectedStep: number | null;
  /** Click a column to jump to that generation step (linked brushing). */
  onSelectStep?: (step: number) => void;
}

type Metric = 'rank' | 'prob';

interface Cell {
  step: number;
  layer: number;
  /** selected-token rank (1-indexed) at this layer, or null if absent. */
  rank: number | null;
  /** probability the layer's lens assigns the selected token, or null. */
  prob: number | null;
}

const MAX_CELL = 18;
const GAP = 1;
const PAD_LEFT = 34;
const PAD_TOP = 60;
const LEGEND_W = 64;

function truncateToken(token: string, max = 6): string {
  const t = escapeToken(token);
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

/**
 * Decision map — a layer × generated-token heatmap of the *selected* token's
 * logit-lens rank (or probability) at each layer. Reading a column top→bottom
 * (shallow→deep) shows *where* in the network that token gets decided: the rank
 * collapses to 1 (bright) at the layer that commits to it. Built entirely from
 * the already-captured `logit_lens[*].selected_token_rank / _prob` — no extra
 * capture. Clicking a column selects that step (drives the per-layer table).
 */
export function LogitLensDecisionMap({
  trace,
  selectedStep,
  onSelectStep,
}: LogitLensDecisionMapProps) {
  const [metric, setMetric] = useState<Metric>('rank');
  const [lens, setLens] = useState<'vanilla' | 'tuned'>('vanilla');
  const [hover, setHover] = useState<Cell | null>(null);

  const tuned = (
    trace as {
      tuned_logit_lens?: {
        steps?: {
          layers: {
            layer_idx: number;
            selected_token_rank: number;
            selected_token_prob: number;
          }[];
        }[];
      };
    }
  ).tuned_logit_lens;
  const hasTuned = (tuned?.steps?.length ?? 0) > 0;

  // Per-step per-layer source, switched by the vanilla/tuned toggle. Both carry
  // selected_token_rank / _prob, which is all the map reads.
  const perLayer = (
    s: (typeof trace.steps)[number],
    i: number,
  ): {
    layer_idx: number;
    selected_token_rank: number;
    selected_token_prob: number;
  }[] =>
    lens === 'tuned' && tuned?.steps
      ? (tuned.steps[i]?.layers ?? [])
      : (s.logit_lens ?? []);

  // Layer axis = the union of layer_idx across every step.
  const layers = useMemo(() => {
    const set = new Set<number>();
    trace.steps.forEach((s, i) => {
      for (const l of perLayer(s, i)) set.add(l.layer_idx);
    });
    return [...set].sort((a, b) => a - b);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace.steps, lens, tuned]);

  const { cells, maxRank } = useMemo(() => {
    const out: Cell[] = [];
    let hiRank = 1;
    trace.steps.forEach((s, step) => {
      const byLayer = new Map(perLayer(s, step).map((l) => [l.layer_idx, l]));
      for (const layer of layers) {
        const l = byLayer.get(layer);
        const rank = l ? l.selected_token_rank : null;
        if (rank != null && rank > hiRank) hiRank = rank;
        out.push({
          step,
          layer,
          rank,
          prob: l ? l.selected_token_prob : null,
        });
      }
    });
    return { cells: out, maxRank: hiRank };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace.steps, layers, lens, tuned]);

  const numSteps = trace.steps.length;
  const numLayers = layers.length;

  if (numLayers === 0 || numSteps === 0) return null;

  // Shrink cells so the whole layer stack fits ~one glance (the deep layers,
  // where tokens actually get decided, would otherwise scroll off the bottom).
  const CELL = Math.max(9, Math.min(MAX_CELL, Math.floor(320 / numLayers)));

  // Colour: bright (viridis high) = "decided" for both metrics.
  //   rank → t = 1 - log(rank)/log(maxRank)  (rank 1 → 1; deep rank → 0)
  //   prob → t = prob                        (1 → 1; 0 → 0)
  const logMax = Math.log(Math.max(2, maxRank));
  const colorFor = (cell: Cell): string | null => {
    if (metric === 'rank') {
      if (cell.rank == null) return null;
      return rampColor(1 - Math.log(cell.rank) / logMax);
    }
    if (cell.prob == null) return null;
    return rampColor(cell.prob);
  };

  const width = PAD_LEFT + numSteps * (CELL + GAP) + LEGEND_W;
  const height = PAD_TOP + numLayers * (CELL + GAP) + 8;
  const gridBottom = PAD_TOP + numLayers * (CELL + GAP);

  return (
    <section
      className="decision-map"
      data-testid="logit-lens-decision-map"
      aria-label="Logit-lens decision map"
    >
      <div className="decision-map__head">
        <h3 className="decision-map__title">
          Decision map <span>· layer × token</span>
        </h3>
        <div className="decision-map__toggles">
          {hasTuned && (
            <div
              className="decision-map__toggle"
              role="group"
              aria-label="Lens"
            >
              <button
                type="button"
                data-selected={lens === 'vanilla'}
                data-testid="decision-map-lens-vanilla"
                onClick={() => setLens('vanilla')}
              >
                Vanilla
              </button>
              <button
                type="button"
                data-selected={lens === 'tuned'}
                data-testid="decision-map-lens-tuned"
                onClick={() => setLens('tuned')}
              >
                Tuned
              </button>
            </div>
          )}
          <div
            className="decision-map__toggle"
            role="group"
            aria-label="Decision map metric"
          >
            <button
              type="button"
              data-selected={metric === 'rank'}
              data-testid="decision-map-metric-rank"
              onClick={() => setMetric('rank')}
            >
              Rank
            </button>
            <button
              type="button"
              data-selected={metric === 'prob'}
              data-testid="decision-map-metric-prob"
              onClick={() => setMetric('prob')}
            >
              Prob
            </button>
          </div>
        </div>
      </div>
      <p className="decision-map__hint">
        {lens === 'tuned'
          ? 'Tuned lens: each layer’s residual is mapped through a fitted affine probe before decoding, fixing the vanilla lens’s early-layer basis mismatch. Compare with Vanilla — early layers are far more decodable.'
          : 'Each column is a generated token; each row a layer (shallow → deep). Bright = the token is already the layer’s top prediction — so a column brightening low down shows a token decided late.'}
      </p>

      <div className="decision-map__plot">
        <svg
          role="img"
          aria-label={`Layer by token logit-lens decision map, coloured by selected-token ${metric}`}
          className="decision-map__svg"
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
        >
          {/* generated-token column labels (rotated) */}
          {trace.steps.map((s, step) => {
            const cx = PAD_LEFT + step * (CELL + GAP) + CELL / 2;
            return (
              <text
                key={`t-${step}`}
                x={cx}
                y={PAD_TOP - 8}
                className="decision-map__token-label"
                transform={`rotate(-55 ${cx} ${PAD_TOP - 8})`}
                data-selected={selectedStep === step}
              >
                {truncateToken(s.selected.token)}
              </text>
            );
          })}
          {/* layer row labels */}
          {layers.map((layer, row) => (
            <text
              key={`l-${layer}`}
              x={PAD_LEFT - 6}
              y={PAD_TOP + row * (CELL + GAP) + CELL / 2 + 3}
              textAnchor="end"
              className="decision-map__axis-label"
            >
              {`L${layer}`}
            </text>
          ))}
          {/* cells */}
          {cells.map((cell) => {
            const row = layers.indexOf(cell.layer);
            const x = PAD_LEFT + cell.step * (CELL + GAP);
            const y = PAD_TOP + row * (CELL + GAP);
            const fill = colorFor(cell);
            const label =
              metric === 'rank'
                ? cell.rank != null
                  ? `rank ${cell.rank}`
                  : 'n/a'
                : cell.prob != null
                  ? `p ${cell.prob.toFixed(3)}`
                  : 'n/a';
            return (
              <rect
                key={`c-${cell.step}-${cell.layer}`}
                x={x}
                y={y}
                width={CELL}
                height={CELL}
                fill={fill ?? 'var(--color-bg-muted)'}
                data-testid={`decision-cell-${cell.step}-${cell.layer}`}
                data-step={cell.step}
                data-layer={cell.layer}
                onMouseEnter={() => setHover(cell)}
                onMouseLeave={() => setHover((h) => (h === cell ? null : h))}
                onClick={() => onSelectStep?.(cell.step)}
                tabIndex={0}
                role="button"
                aria-label={`Step ${cell.step} layer ${cell.layer}, selected-token ${label}`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectStep?.(cell.step);
                  }
                }}
                style={{ cursor: onSelectStep ? 'pointer' : 'default' }}
              />
            );
          })}
          {/* selected-column outline */}
          {selectedStep != null && selectedStep < numSteps && (
            <rect
              className="decision-map__selected"
              x={PAD_LEFT + selectedStep * (CELL + GAP) - 1}
              y={PAD_TOP - 1}
              width={CELL + 2}
              height={numLayers * (CELL + GAP) + 1}
              fill="none"
              pointerEvents="none"
            />
          )}
          {/* legend */}
          <g transform={`translate(${PAD_LEFT + numSteps * (CELL + GAP) + 14}, ${PAD_TOP})`}>
            <defs>
              <linearGradient id="decision-legend" x1="0" x2="0" y1="1" y2="0">
                {Array.from({ length: 9 }, (_, i) => (
                  <stop
                    key={i}
                    offset={`${(i / 8) * 100}%`}
                    stopColor={rampColor(i / 8)}
                  />
                ))}
              </linearGradient>
            </defs>
            <rect
              x={0}
              y={0}
              width={14}
              height={numLayers * (CELL + GAP)}
              fill="url(#decision-legend)"
            />
            <text x={20} y={10} className="decision-map__legend-label">
              {metric === 'rank' ? 'rank 1' : '1.0'}
            </text>
            <text x={20} y={gridBottom - PAD_TOP} className="decision-map__legend-label">
              {metric === 'rank' ? `rank ${maxRank}` : '0'}
            </text>
          </g>
        </svg>
      </div>

      {hover && (
        <div className="decision-map__tooltip" data-testid="decision-map-tooltip">
          <strong>
            {truncateToken(trace.steps[hover.step].selected.token, 12)} · L
            {hover.layer}
          </strong>
          <br />
          rank {hover.rank ?? '—'} · p{' '}
          {hover.prob != null ? hover.prob.toFixed(4) : '—'}
        </div>
      )}
    </section>
  );
}
