import { useMemo, useState } from 'react';
import type { ManifoldLayer } from '@/types/trace';
import './ManifoldDepthCurve.css';

export interface ManifoldDepthCurveProps {
  layers: ManifoldLayer[];
}

type Metric = 'id' | 'pr';

const W = 560;
const H = 220;
const PAD_L = 40;
const PAD_B = 30;
const PAD_T = 12;
const PAD_R = 118;

// Distinct, palette-matched colours per submodule (assigned by first-seen order).
const SERIES_COLORS = ['#d97757', '#31688e', '#35b779', '#b072d0', '#e0a740'];

function valueOf(layer: ManifoldLayer, metric: Metric): number | null {
  const v =
    metric === 'id' ? layer.intrinsic_dimension?.twonn : layer.participation_ratio;
  return v != null && Number.isFinite(v) ? v : null;
}

/**
 * Geometry-across-depth curve — intrinsic dimension (TwoNN) or participation
 * ratio plotted against layer index, one line per submodule. Surfaces the
 * "hunchback" profile (dimension swelling in the middle, contracting late) and
 * lets you compare submodules / (by loading two traces) correct-vs-wrong runs.
 * Built from the manifold metrics already in the trace — no new capture.
 */
export function ManifoldDepthCurve({ layers }: ManifoldDepthCurveProps) {
  const [metric, setMetric] = useState<Metric>('id');

  const series = useMemo(() => {
    const bySub = new Map<string, { layer: number; value: number }[]>();
    for (const l of layers) {
      const v = valueOf(l, metric);
      if (v == null) continue;
      if (!bySub.has(l.submodule)) bySub.set(l.submodule, []);
      bySub.get(l.submodule)!.push({ layer: l.layer, value: v });
    }
    return [...bySub.entries()].map(([submodule, pts], i) => ({
      submodule,
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      points: pts.sort((a, b) => a.layer - b.layer),
    }));
  }, [layers, metric]);

  const { minLayer, maxLayer, maxVal } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    let vmax = 0;
    for (const s of series) {
      for (const p of s.points) {
        if (p.layer < lo) lo = p.layer;
        if (p.layer > hi) hi = p.layer;
        if (p.value > vmax) vmax = p.value;
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    return { minLayer: lo, maxLayer: hi, maxVal: vmax || 1 };
  }, [series]);

  const distinctLayers = maxLayer - minLayer;
  if (series.length === 0 || distinctLayers < 1) return null;

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xOf = (layer: number) =>
    PAD_L + ((layer - minLayer) / distinctLayers) * plotW;
  const yOf = (value: number) => PAD_T + (1 - value / maxVal) * plotH;

  return (
    <section
      className="depth-curve"
      data-testid="manifold-depth-curve"
      aria-label="Geometry across depth"
    >
      <div className="depth-curve__head">
        <h3 className="depth-curve__title">
          Geometry across depth <span>· by layer</span>
        </h3>
        <div
          className="depth-curve__toggle"
          role="group"
          aria-label="Depth-curve metric"
        >
          <button
            type="button"
            data-selected={metric === 'id'}
            data-testid="depth-curve-metric-id"
            onClick={() => setMetric('id')}
          >
            Intrinsic dim
          </button>
          <button
            type="button"
            data-selected={metric === 'pr'}
            data-testid="depth-curve-metric-pr"
            onClick={() => setMetric('pr')}
          >
            Participation
          </button>
        </div>
      </div>

      <svg
        className="depth-curve__svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${metric === 'id' ? 'Intrinsic dimension' : 'Participation ratio'} across layers, per submodule`}
      >
        {/* axes */}
        <line
          x1={PAD_L}
          y1={PAD_T}
          x2={PAD_L}
          y2={PAD_T + plotH}
          className="depth-curve__axis"
        />
        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={PAD_L + plotW}
          y2={PAD_T + plotH}
          className="depth-curve__axis"
        />
        {/* y ticks: 0 and max */}
        <text x={PAD_L - 6} y={PAD_T + plotH + 3} className="depth-curve__tick" textAnchor="end">
          0
        </text>
        <text x={PAD_L - 6} y={PAD_T + 4} className="depth-curve__tick" textAnchor="end">
          {maxVal.toFixed(maxVal >= 10 ? 0 : 1)}
        </text>
        {/* x ticks: min and max layer */}
        <text x={PAD_L} y={PAD_T + plotH + 18} className="depth-curve__tick" textAnchor="middle">
          L{minLayer}
        </text>
        <text
          x={PAD_L + plotW}
          y={PAD_T + plotH + 18}
          className="depth-curve__tick"
          textAnchor="middle"
        >
          L{maxLayer}
        </text>

        {/* one line + dots per submodule */}
        {series.map((s) => (
          <g key={s.submodule}>
            <path
              d={s.points
                .map(
                  (p, i) => `${i === 0 ? 'M' : 'L'}${xOf(p.layer).toFixed(2)},${yOf(p.value).toFixed(2)}`,
                )
                .join(' ')}
              fill="none"
              stroke={s.color}
              strokeWidth={1.6}
              data-testid={`depth-curve-line-${s.submodule}`}
            />
            {s.points.map((p) => (
              <circle
                key={p.layer}
                cx={xOf(p.layer)}
                cy={yOf(p.value)}
                r={2.4}
                fill={s.color}
              >
                <title>{`${s.submodule} · L${p.layer}: ${p.value.toFixed(2)}`}</title>
              </circle>
            ))}
          </g>
        ))}

        {/* legend */}
        <g transform={`translate(${PAD_L + plotW + 14}, ${PAD_T + 2})`}>
          {series.map((s, i) => (
            <g key={s.submodule} transform={`translate(0, ${i * 16})`}>
              <rect x={0} y={-7} width={10} height={10} rx={2} fill={s.color} />
              <text x={15} y={2} className="depth-curve__legend">
                {s.submodule}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </section>
  );
}
