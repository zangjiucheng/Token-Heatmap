import { useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import { escapeToken } from '@/features/detail/escapeToken';
import './CausalHeatmap.css';

interface CausalCell {
  kl: number;
  dp: number;
}
interface CausalLayer {
  layer: number;
  attn?: CausalCell;
  mlp?: CausalCell;
}
interface CausalStep {
  step: number;
  token: string;
  base_prob: number;
  layers: CausalLayer[];
}
interface CausalGrid {
  components?: string[];
  num_layers?: number;
  steps?: CausalStep[];
}

export interface CausalHeatmapProps {
  trace: Trace;
  selectedStep: number | null;
  onSelectStep?: (step: number) => void;
}

type Component = 'attn' | 'mlp';

const MAX_CELL = 18;
const GAP = 1;
const PAD_LEFT = 34;
const PAD_TOP = 60;
const LEGEND_W = 68;

function truncateToken(token: string, max = 6): string {
  const t = escapeToken(token);
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}
// Diverging ramp: ablation DROPS the token (dp<0, block was load-bearing) → warm
// clay; ablation RAISES it (dp>0) → cool teal; ~0 → muted.
function divergingColor(dp: number, maxAbs: number): string {
  const t = Math.max(-1, Math.min(1, maxAbs > 0 ? dp / maxAbs : 0));
  const mid = [42, 45, 51]; // muted
  const warm = [217, 119, 87]; // clay
  const cool = [49, 136, 158]; // teal
  const end = t < 0 ? warm : cool;
  const k = Math.abs(t);
  return `rgb(${lerp(mid[0], end[0], k)}, ${lerp(mid[1], end[1], k)}, ${lerp(mid[2], end[2], k)})`;
}

/**
 * Causal-effect heatmap — the causal counterpart to Direct Logit Attribution.
 * For every generated token (column) and every layer (row), it shows how much
 * ablating that layer's attention (or MLP) block moves the token's probability.
 * Deep warm = the block was load-bearing (removing it collapses the answer),
 * cool = removing it *helps*. Same layer × token coordinate system as the token
 * heatmap. Reads trace.causal_grid (written by `token-heatmap causal`).
 */
export function CausalHeatmap({
  trace,
  selectedStep,
  onSelectStep,
}: CausalHeatmapProps) {
  const [component, setComponent] = useState<Component>('attn');
  const [hover, setHover] = useState<{
    step: number;
    layer: number;
    cell: CausalCell;
  } | null>(null);

  const grid = (trace as { causal_grid?: CausalGrid }).causal_grid;
  const steps = useMemo(() => grid?.steps ?? [], [grid]);

  const layers = useMemo(() => {
    const set = new Set<number>();
    for (const s of steps) for (const l of s.layers) set.add(l.layer);
    return [...set].sort((a, b) => a - b);
  }, [steps]);

  const maxAbs = useMemo(() => {
    let m = 0;
    for (const s of steps) {
      for (const l of s.layers) {
        const c = l[component];
        if (c && Math.abs(c.dp) > m) m = Math.abs(c.dp);
      }
    }
    return m || 1;
  }, [steps, component]);

  const numSteps = steps.length;
  const numLayers = layers.length;
  if (numSteps === 0 || numLayers === 0) return null;

  const CELL = Math.max(9, Math.min(MAX_CELL, Math.floor(320 / numLayers)));
  const width = PAD_LEFT + numSteps * (CELL + GAP) + LEGEND_W;
  const height = PAD_TOP + numLayers * (CELL + GAP) + 8;
  const componentsPresent = grid?.components ?? ['attn', 'mlp'];

  return (
    <section
      className="causal-map"
      data-testid="causal-heatmap"
      aria-label="Causal-effect heatmap"
    >
      <div className="causal-map__head">
        <h3 className="causal-map__title">
          Causal effect <span>· ablate layer block</span>
        </h3>
        <div
          className="causal-map__toggle"
          role="group"
          aria-label="Ablated block"
        >
          {(['attn', 'mlp'] as Component[])
            .filter((c) => componentsPresent.includes(c))
            .map((c) => (
              <button
                key={c}
                type="button"
                data-selected={component === c}
                data-testid={`causal-component-${c}`}
                onClick={() => setComponent(c)}
              >
                {c === 'attn' ? 'Attn' : 'MLP'}
              </button>
            ))}
        </div>
      </div>
      <p className="causal-map__hint">
        Ablate each layer&rsquo;s {component === 'attn' ? 'attention' : 'MLP'}{' '}
        block at each token; colour = change in that token&rsquo;s probability.
        Deep warm = load-bearing (removing it collapses the answer); cool =
        removing it helps.
      </p>

      <div className="causal-map__plot">
        <svg
          role="img"
          aria-label={`Layer by token causal-effect heatmap for the ${component} block`}
          className="causal-map__svg"
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
        >
          {steps.map((s) => {
            const cx = PAD_LEFT + s.step * (CELL + GAP) + CELL / 2;
            return (
              <text
                key={`t-${s.step}`}
                x={cx}
                y={PAD_TOP - 8}
                className="causal-map__token-label"
                transform={`rotate(-55 ${cx} ${PAD_TOP - 8})`}
                data-selected={selectedStep === s.step}
              >
                {truncateToken(s.token)}
              </text>
            );
          })}
          {layers.map((layer, row) => (
            <text
              key={`l-${layer}`}
              x={PAD_LEFT - 6}
              y={PAD_TOP + row * (CELL + GAP) + CELL / 2 + 3}
              textAnchor="end"
              className="causal-map__axis-label"
            >
              {`L${layer}`}
            </text>
          ))}
          {steps.map((s) =>
            s.layers.map((l) => {
              const row = layers.indexOf(l.layer);
              const cell = l[component];
              const x = PAD_LEFT + s.step * (CELL + GAP);
              const y = PAD_TOP + row * (CELL + GAP);
              return (
                <rect
                  key={`c-${s.step}-${l.layer}`}
                  x={x}
                  y={y}
                  width={CELL}
                  height={CELL}
                  fill={cell ? divergingColor(cell.dp, maxAbs) : 'var(--color-bg-muted)'}
                  data-testid={`causal-cell-${s.step}-${l.layer}`}
                  onMouseEnter={() =>
                    cell && setHover({ step: s.step, layer: l.layer, cell })
                  }
                  onMouseLeave={() => setHover(null)}
                  onClick={() => onSelectStep?.(s.step)}
                  tabIndex={0}
                  role="button"
                  aria-label={`Step ${s.step} layer ${l.layer}, probability change ${cell ? cell.dp.toFixed(3) : 'n/a'}`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelectStep?.(s.step);
                    }
                  }}
                  style={{ cursor: onSelectStep ? 'pointer' : 'default' }}
                />
              );
            }),
          )}
          {selectedStep != null && selectedStep < numSteps && (
            <rect
              className="causal-map__selected"
              x={PAD_LEFT + selectedStep * (CELL + GAP) - 1}
              y={PAD_TOP - 1}
              width={CELL + 2}
              height={numLayers * (CELL + GAP) + 1}
              fill="none"
              pointerEvents="none"
            />
          )}
          {/* diverging legend */}
          <g transform={`translate(${PAD_LEFT + numSteps * (CELL + GAP) + 14}, ${PAD_TOP})`}>
            <defs>
              <linearGradient id="causal-legend" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={divergingColor(maxAbs, maxAbs)} />
                <stop offset="50%" stopColor={divergingColor(0, maxAbs)} />
                <stop offset="100%" stopColor={divergingColor(-maxAbs, maxAbs)} />
              </linearGradient>
            </defs>
            <rect
              x={0}
              y={0}
              width={14}
              height={numLayers * (CELL + GAP)}
              fill="url(#causal-legend)"
            />
            <text x={20} y={9} className="causal-map__legend-label">
              +{maxAbs.toFixed(2)}
            </text>
            <text
              x={20}
              y={numLayers * (CELL + GAP)}
              className="causal-map__legend-label"
            >
              −{maxAbs.toFixed(2)}
            </text>
          </g>
        </svg>
      </div>

      {hover && (
        <div className="causal-map__tooltip" data-testid="causal-heatmap-tooltip">
          <strong>
            {truncateToken(steps[hover.step]?.token ?? '', 12)} · L{hover.layer} ·{' '}
            {component}
          </strong>
          <br />
          ΔP {hover.cell.dp >= 0 ? '+' : ''}
          {hover.cell.dp.toFixed(4)} · KL {hover.cell.kl.toFixed(3)}
        </div>
      )}
    </section>
  );
}
