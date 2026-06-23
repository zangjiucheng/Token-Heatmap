import { useMemo } from 'react';

export interface ManifoldScreePlotProps {
  explainedVarianceRatio: number[];
  cumulativeVarianceRatio: number[];
  /** How many leading components to draw (default 12). */
  maxBars?: number;
}

const W = 320;
const H = 180;
const PAD_L = 28;
const PAD_B = 22;
const PAD_T = 10;
const PAD_R = 8;

/**
 * Scree plot: a bar per principal component showing its explained-variance
 * fraction, with the cumulative curve overlaid. A spectrum that collapses into
 * the first few bars is the visual signature of a low-dimensional manifold.
 */
export function ManifoldScreePlot({
  explainedVarianceRatio,
  cumulativeVarianceRatio,
  maxBars = 12,
}: ManifoldScreePlotProps) {
  const bars = useMemo(
    () => explainedVarianceRatio.slice(0, maxBars),
    [explainedVarianceRatio, maxBars],
  );
  const cumulative = useMemo(
    () => cumulativeVarianceRatio.slice(0, maxBars),
    [cumulativeVarianceRatio, maxBars],
  );

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const n = bars.length;
  const slot = n > 0 ? plotW / n : plotW;
  const barW = Math.max(2, slot * 0.7);

  const cumulativePath = cumulative
    .map((c, i) => {
      const x = PAD_L + slot * i + slot / 2;
      const y = PAD_T + (1 - c) * plotH;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');

  return (
    <svg
      className="manifold-scree"
      data-testid="manifold-scree"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Scree plot of explained variance for ${n} principal components`}
    >
      {/* y axis baseline at 0 and top at 1 */}
      <line
        x1={PAD_L}
        y1={PAD_T}
        x2={PAD_L}
        y2={PAD_T + plotH}
        className="manifold-scree__axis"
      />
      <line
        x1={PAD_L}
        y1={PAD_T + plotH}
        x2={W - PAD_R}
        y2={PAD_T + plotH}
        className="manifold-scree__axis"
      />
      {bars.map((v, i) => {
        const h = Math.max(0, v) * plotH;
        const x = PAD_L + slot * i + (slot - barW) / 2;
        const y = PAD_T + plotH - h;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barW}
            height={h}
            className="manifold-scree__bar"
            data-testid={`manifold-scree-bar-${i}`}
          >
            <title>{`PC${i + 1}: ${(v * 100).toFixed(1)}%`}</title>
          </rect>
        );
      })}
      {cumulative.length > 1 && (
        <path
          className="manifold-scree__cumulative"
          d={cumulativePath}
          fill="none"
          data-testid="manifold-scree-cumulative"
        />
      )}
    </svg>
  );
}
