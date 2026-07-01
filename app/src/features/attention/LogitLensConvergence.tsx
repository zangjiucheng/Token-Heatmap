import { useMemo } from 'react';
import type { Trace } from '@/types/trace';
import './LogitLensConvergence.css';

export interface LogitLensConvergenceProps {
  trace: Trace;
}

interface PerLayer {
  layer_idx: number;
  selected_token_prob: number;
}

const W = 520;
const H = 200;
const PAD_L = 40;
const PAD_B = 30;
const PAD_T = 12;
const PAD_R = 108;

const VANILLA = '#8a8f98';
const TUNED = '#d97757';

/**
 * Prediction-convergence curve — the mean probability the lens assigns the
 * *realized* answer token at each layer, averaged over generated tokens. It
 * rises from ~0 to 1 as the network commits; the tuned lens (when present)
 * rises earlier, the aggregate view of its early-layer decodability fix.
 * Built from the per-layer selected_token_prob already in the trace.
 */
export function LogitLensConvergence({ trace }: LogitLensConvergenceProps) {
  const tuned = (
    trace as { tuned_logit_lens?: { steps?: { layers: PerLayer[] }[] } }
  ).tuned_logit_lens;

  const { layers, vanillaMean, tunedMean, hasTuned } = useMemo(() => {
    const vSum = new Map<number, { sum: number; n: number }>();
    const tSum = new Map<number, { sum: number; n: number }>();
    const add = (
      m: Map<number, { sum: number; n: number }>,
      per: PerLayer[],
    ) => {
      for (const l of per) {
        const e = m.get(l.layer_idx) ?? { sum: 0, n: 0 };
        e.sum += l.selected_token_prob;
        e.n += 1;
        m.set(l.layer_idx, e);
      }
    };
    trace.steps.forEach((s, i) => {
      add(vSum, (s.logit_lens as PerLayer[] | undefined) ?? []);
      if (tuned?.steps?.[i]) add(tSum, tuned.steps[i].layers);
    });
    const layerSet = [...vSum.keys()].sort((a, b) => a - b);
    const mean = (m: Map<number, { sum: number; n: number }>) =>
      layerSet.map((l) => {
        const e = m.get(l);
        return e && e.n ? e.sum / e.n : 0;
      });
    return {
      layers: layerSet,
      vanillaMean: mean(vSum),
      tunedMean: mean(tSum),
      hasTuned: (tuned?.steps?.length ?? 0) > 0,
    };
  }, [trace.steps, tuned]);

  if (layers.length < 2) return null;

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const lo = layers[0];
  const span = layers[layers.length - 1] - lo || 1;
  const xOf = (layer: number) => PAD_L + ((layer - lo) / span) * plotW;
  const yOf = (v: number) => PAD_T + (1 - v) * plotH; // prob in [0,1]

  const path = (vals: number[]) =>
    vals
      .map(
        (v, i) =>
          `${i === 0 ? 'M' : 'L'}${xOf(layers[i]).toFixed(2)},${yOf(v).toFixed(2)}`,
      )
      .join(' ');

  const series = [
    { label: 'vanilla', color: VANILLA, vals: vanillaMean },
    ...(hasTuned ? [{ label: 'tuned', color: TUNED, vals: tunedMean }] : []),
  ];

  return (
    <section
      className="ll-converge"
      data-testid="logit-lens-convergence"
      aria-label="Prediction convergence"
    >
      <h3 className="ll-converge__title">
        Prediction convergence <span>· P(answer) by layer</span>
      </h3>
      <svg
        className="ll-converge__svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Mean probability of the answer token per layer"
      >
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} className="ll-converge__axis" />
        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={PAD_L + plotW}
          y2={PAD_T + plotH}
          className="ll-converge__axis"
        />
        <text x={PAD_L - 6} y={PAD_T + 4} className="ll-converge__tick" textAnchor="end">
          1
        </text>
        <text x={PAD_L - 6} y={PAD_T + plotH + 3} className="ll-converge__tick" textAnchor="end">
          0
        </text>
        <text x={PAD_L} y={PAD_T + plotH + 18} className="ll-converge__tick" textAnchor="middle">
          L{lo}
        </text>
        <text
          x={PAD_L + plotW}
          y={PAD_T + plotH + 18}
          className="ll-converge__tick"
          textAnchor="middle"
        >
          L{layers[layers.length - 1]}
        </text>
        {series.map((s) => (
          <path
            key={s.label}
            d={path(s.vals)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.8}
            data-testid={`ll-converge-line-${s.label}`}
          />
        ))}
        <g transform={`translate(${PAD_L + plotW + 14}, ${PAD_T + 2})`}>
          {series.map((s, i) => (
            <g key={s.label} transform={`translate(0, ${i * 16})`}>
              <rect x={0} y={-7} width={10} height={10} rx={2} fill={s.color} />
              <text x={15} y={2} className="ll-converge__legend">
                {s.label}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </section>
  );
}
