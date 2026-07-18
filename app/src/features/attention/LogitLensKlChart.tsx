import { useMemo, useState } from 'react';
import { escapeToken } from '@/features/detail/escapeToken';
import './LogitLensKlChart.css';

/**
 * Structural layer shape shared by the per-step lens (`LogitLensLayer`) and
 * the prompt-position lens payload: only the fields the chart reads.
 */
export interface KlLensLayer {
  layer_idx: number;
  kl_to_final?: number;
  kl_from_prev?: number;
  top_k: { token_id: number; token?: string; prob: number }[];
}

export interface LogitLensKlChartProps {
  layers: KlLensLayer[];
  /** Caption naming what the curves describe (e.g. "step 3 · ' Paris'"). */
  caption?: string;
}

const W = 640;
const H = 240;
const M = { left: 48, right: 14, top: 16, bottom: 32 } as const;

interface LayerPoint {
  layerIdx: number;
  x: number;
  klToFinal: number | null;
  klFromPrev: number | null;
  top1: string;
  isFlip: boolean;
}

function formatNats(v: number | null): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return v >= 100 ? v.toFixed(0) : v.toFixed(2);
}

function top1Text(layer: KlLensLayer): string {
  const cand = layer.top_k[0];
  if (!cand) return '';
  return cand.token ? escapeToken(cand.token) : `#${cand.token_id}`;
}

/**
 * Prediction movement through depth: `kl_to_final` (distance still to travel
 * to the final layer's distribution — collapses to 0 at the prediction depth)
 * and `kl_from_prev` (how much each block moved the prediction), with the
 * layers where the top-1 token flipped identity marked along the top edge.
 *
 * Renders nothing when the trace predates the KL metrics.
 */
export function LogitLensKlChart({ layers, caption }: LogitLensKlChartProps) {
  const [hovered, setHovered] = useState<number | null>(null);

  const sorted = useMemo(
    () => layers.slice().sort((a, b) => a.layer_idx - b.layer_idx),
    [layers],
  );
  const hasKl = sorted.some((l) => l.kl_to_final != null);

  const points = useMemo<LayerPoint[]>(() => {
    if (!hasKl || sorted.length === 0) return [];
    const minLayer = sorted[0].layer_idx;
    const maxLayer = sorted[sorted.length - 1].layer_idx;
    const span = Math.max(1, maxLayer - minLayer);
    return sorted.map((layer, i) => {
      const prev = i > 0 ? sorted[i - 1] : null;
      const top1Id = layer.top_k[0]?.token_id;
      const prevTop1Id = prev?.top_k[0]?.token_id;
      return {
        layerIdx: layer.layer_idx,
        x:
          M.left +
          ((layer.layer_idx - minLayer) / span) * (W - M.left - M.right),
        klToFinal: layer.kl_to_final ?? null,
        klFromPrev: layer.kl_from_prev ?? null,
        top1: top1Text(layer),
        isFlip:
          prev != null && top1Id != null && prevTop1Id != null
            ? top1Id !== prevTop1Id
            : false,
      };
    });
  }, [sorted, hasKl]);

  if (points.length === 0) return null;

  const yMax = Math.max(
    1e-6,
    ...points.map((p) => Math.max(p.klToFinal ?? 0, p.klFromPrev ?? 0)),
  );
  const yOf = (v: number) =>
    H - M.bottom - (v / yMax) * (H - M.top - M.bottom);

  const linePath = (pick: (p: LayerPoint) => number | null) =>
    points
      .filter((p) => pick(p) != null)
      .map(
        (p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${yOf(pick(p) as number).toFixed(1)}`,
      )
      .join(' ');

  const flips = points.filter((p) => p.isFlip);
  const hoveredPoint =
    hovered != null ? points.find((p) => p.layerIdx === hovered) : undefined;

  const handleMove = (evt: React.MouseEvent<SVGSVGElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * W;
    let nearest: LayerPoint | null = null;
    for (const p of points) {
      if (nearest == null || Math.abs(p.x - x) < Math.abs(nearest.x - x)) {
        nearest = p;
      }
    }
    setHovered(nearest?.layerIdx ?? null);
  };

  const yTicks = [0, yMax / 2, yMax];

  return (
    <figure className="lens-kl-chart" data-testid="logit-lens-kl-chart">
      <figcaption className="lens-kl-chart__caption">
        Prediction movement through depth{caption ? ` — ${caption}` : ''}
      </figcaption>
      <div className="lens-kl-chart__legend" aria-hidden="true">
        <span className="lens-kl-chart__legend-item">
          <span className="lens-kl-chart__chip lens-kl-chart__chip--final" />
          KL(final ‖ layer)
        </span>
        <span className="lens-kl-chart__legend-item">
          <span className="lens-kl-chart__chip lens-kl-chart__chip--prev" />
          KL(layer ‖ previous)
        </span>
        {flips.length > 0 && (
          <span className="lens-kl-chart__legend-item">
            <span className="lens-kl-chart__chip lens-kl-chart__chip--flip" />
            top-1 flip
          </span>
        )}
      </div>
      <div className="lens-kl-chart__plot">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="KL divergence per layer: distance to the final prediction and movement per block"
          onMouseMove={handleMove}
          onMouseLeave={() => setHovered(null)}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line
                className="lens-kl-chart__grid"
                x1={M.left}
                x2={W - M.right}
                y1={yOf(t)}
                y2={yOf(t)}
              />
              <text className="lens-kl-chart__tick" x={M.left - 6} y={yOf(t) + 3} textAnchor="end">
                {formatNats(t)}
              </text>
            </g>
          ))}
          {points.map((p) => (
            <text
              key={p.layerIdx}
              className="lens-kl-chart__tick"
              x={p.x}
              y={H - M.bottom + 14}
              textAnchor="middle"
            >
              {points.length <= 12 || p.layerIdx % 4 === 0 ? p.layerIdx : ''}
            </text>
          ))}
          <text
            className="lens-kl-chart__tick"
            x={(M.left + W - M.right) / 2}
            y={H - 4}
            textAnchor="middle"
          >
            Layer
          </text>
          <path
            className="lens-kl-chart__line lens-kl-chart__line--final"
            d={linePath((p) => p.klToFinal)}
            data-testid="kl-to-final-path"
          />
          <path
            className="lens-kl-chart__line lens-kl-chart__line--prev"
            d={linePath((p) => p.klFromPrev)}
            data-testid="kl-from-prev-path"
          />
          {flips.map((p) => (
            <path
              key={p.layerIdx}
              className="lens-kl-chart__flip"
              d={`M${p.x},${M.top - 4} l4,6 l-8,0 Z`}
              data-testid={`kl-flip-marker-${p.layerIdx}`}
            >
              <title>{`Top-1 flipped to ${p.top1} at layer ${p.layerIdx}`}</title>
            </path>
          ))}
          {hoveredPoint && (
            <line
              className="lens-kl-chart__cursor"
              x1={hoveredPoint.x}
              x2={hoveredPoint.x}
              y1={M.top}
              y2={H - M.bottom}
            />
          )}
        </svg>
        {hoveredPoint && (
          <div
            className="lens-kl-chart__tooltip"
            data-testid="kl-tooltip"
            style={{ left: `${(hoveredPoint.x / W) * 100}%` }}
          >
            <div className="lens-kl-chart__tooltip-title">
              L{hoveredPoint.layerIdx} · top-1 {hoveredPoint.top1}
              {hoveredPoint.isFlip ? ' (flip)' : ''}
            </div>
            <div>KL→final {formatNats(hoveredPoint.klToFinal)} nats</div>
            <div>KL Δprev {formatNats(hoveredPoint.klFromPrev)} nats</div>
          </div>
        )}
      </div>
    </figure>
  );
}
