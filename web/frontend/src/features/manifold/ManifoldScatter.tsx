import { useMemo } from 'react';
import { rampColor } from './colorRamp';

export interface ManifoldScatterProps {
  /** PCA scores, one row per position: `coords[i][component]`. */
  coords: number[][];
  /** Generation-step index for each row, aligned with `coords`. */
  positions: number[];
  /** Column of `coords` plotted on the x axis. */
  xComponent: number;
  /** Column of `coords` plotted on the y axis. */
  yComponent: number;
  selectedStep: number | null;
  hoveredStep: number | null;
  onSelectStep: (step: number) => void;
  onHoverStep: (step: number | null) => void;
}

// Fixed internal coordinate system; the SVG scales to its container via viewBox.
const W = 520;
const H = 360;
const PAD = 28;

function extent(values: number[]): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (max - min < 1e-12) return [min - 0.5, max + 0.5];
  return [min, max];
}

/**
 * Scatter of the activation cloud projected onto two principal components, one
 * point per token position coloured by step, with the position-ordered path
 * drawn faintly behind so the manifold *trajectory* (the curve / helix the
 * line-break paper describes) is visible. Click or hover a point to drive the
 * shared selected/hovered step.
 */
export function ManifoldScatter({
  coords,
  positions,
  xComponent,
  yComponent,
  selectedStep,
  hoveredStep,
  onSelectStep,
  onHoverStep,
}: ManifoldScatterProps) {
  const points = useMemo(() => {
    const xs = coords.map((row) => row[xComponent] ?? 0);
    const ys = coords.map((row) => row[yComponent] ?? 0);
    const [xmin, xmax] = extent(xs);
    const [ymin, ymax] = extent(ys);
    const tMin = positions.length ? Math.min(...positions) : 0;
    const tMax = positions.length ? Math.max(...positions) : 1;
    const tSpan = tMax - tMin || 1;
    return coords.map((row, i) => {
      const xv = row[xComponent] ?? 0;
      const yv = row[yComponent] ?? 0;
      const px = PAD + ((xv - xmin) / (xmax - xmin)) * (W - 2 * PAD);
      // SVG y grows downward; flip so larger values sit higher.
      const py = H - PAD - ((yv - ymin) / (ymax - ymin)) * (H - 2 * PAD);
      return {
        step: positions[i] ?? i,
        x: px,
        y: py,
        color: rampColor((positions[i] - tMin) / tSpan),
      };
    });
  }, [coords, positions, xComponent, yComponent]);

  const pathD = useMemo(
    () =>
      points
        .map(
          (p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${p.y.toFixed(2)}`,
        )
        .join(' '),
    [points],
  );

  return (
    <svg
      className="manifold-scatter"
      data-testid="manifold-scatter"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Activation cloud projected onto principal components ${xComponent + 1} and ${yComponent + 1}, ${points.length} token positions`}
    >
      <path
        className="manifold-scatter__trajectory"
        d={pathD}
        fill="none"
        data-testid="manifold-scatter-path"
      />
      {points.map((p) => {
        const isSelected = p.step === selectedStep;
        const isHovered = p.step === hoveredStep;
        return (
          <circle
            key={p.step}
            cx={p.x}
            cy={p.y}
            r={isSelected ? 7 : isHovered ? 6 : 4}
            fill={p.color}
            stroke={isSelected ? '#000' : isHovered ? '#333' : 'none'}
            strokeWidth={isSelected ? 2 : 1}
            className="manifold-scatter__point"
            data-testid={`manifold-point-${p.step}`}
            data-selected={isSelected ? 'true' : 'false'}
            onClick={() => onSelectStep(p.step)}
            onMouseEnter={() => onHoverStep(p.step)}
            onMouseLeave={() => onHoverStep(null)}
          >
            <title>{`step ${p.step}`}</title>
          </circle>
        );
      })}
    </svg>
  );
}
