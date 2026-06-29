import { useMemo, useRef, useState, type PointerEvent } from 'react';
import { rampColor } from './colorRamp';
import { useThemeTokens } from '@/hooks/useThemeTokens';
import {
  projectManifold,
  VIEW_W,
  VIEW_H,
  type ProjectedPoint,
} from './manifold3d';

export interface ManifoldScatter3DProps {
  /** PCA scores, one row per position; columns 0-2 are PC1-PC3. */
  coords: number[][];
  /** Generation-step index for each row, aligned with `coords`. */
  positions: number[];
  /** Optional per-row values to colour by; defaults to `positions`. */
  colorValues?: number[];
  selectedStep: number | null;
  hoveredStep: number | null;
  onSelectStep: (step: number) => void;
  onHoverStep: (step: number | null) => void;
}

const ROTATE_SPEED = 0.01;
const PITCH_LIMIT = 1.45;

/**
 * 3-D activation cloud: the (PC1, PC2, PC3) projection rendered as a rotatable
 * orthographic scatter so the manifold's true geometry — the helix / curve the
 * line-break paper describes — is visible rather than flattened. Drag to
 * rotate; points are depth-sorted and depth-cued (nearer = larger, more
 * opaque). Shares the step colour ramp and selection contract with the 2-D
 * view.
 */
export function ManifoldScatter3D({
  coords,
  positions,
  colorValues,
  selectedStep,
  hoveredStep,
  onSelectStep,
  onHoverStep,
}: ManifoldScatter3DProps) {
  const tk = useThemeTokens();
  const [rotation, setRotation] = useState({ yaw: 0.7, pitch: 0.42 });
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const { proj, colors, pathD, tMin, tSpan } = useMemo(() => {
    const projection = projectManifold(coords, rotation.yaw, rotation.pitch);
    const cvals = colorValues ?? positions;
    const min = cvals.length ? Math.min(...cvals) : 0;
    const max = cvals.length ? Math.max(...cvals) : 1;
    const span = max - min || 1;
    const cols = cvals.map((v) => rampColor((v - min) / span));
    const d = projection.points
      .map(
        (p, i) =>
          `${i === 0 ? 'M' : 'L'}${p.px.toFixed(1)},${p.py.toFixed(1)}`,
      )
      .join(' ');
    return { proj: projection, colors: cols, pathD: d, tMin: min, tSpan: span };
  }, [coords, positions, colorValues, rotation]);

  // Far-to-near draw order so nearer points overpaint farther ones.
  const ordered = useMemo(
    () => proj.points.map((p, i) => ({ p, i })).sort((a, b) => a.p.depth - b.p.depth),
    [proj],
  );

  const handlePointerDown = (e: PointerEvent<SVGSVGElement>) => {
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
    drag.current.x = e.clientX;
    drag.current.y = e.clientY;
    setRotation((r) => ({
      yaw: r.yaw + dx * ROTATE_SPEED,
      pitch: Math.max(
        -PITCH_LIMIT,
        Math.min(PITCH_LIMIT, r.pitch + dy * ROTATE_SPEED),
      ),
    }));
  };

  const handlePointerUp = (e: PointerEvent<SVGSVGElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    drag.current = null;
  };

  const radiusFor = (point: ProjectedPoint, big: boolean) => {
    const factor = (point.depth + 1) / 2; // ~[0,1], near = 1
    return (big ? 5.5 : 3.4) + 2.6 * factor;
  };

  return (
    <svg
      className="manifold-scatter manifold-scatter--3d"
      data-testid="manifold-scatter-3d"
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      role="img"
      aria-label={`3-D activation cloud (PC1, PC2, PC3), ${proj.points.length} token positions; drag to rotate`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {/* Axes from the centroid along PC1/PC2/PC3. */}
      <g className="manifold-scatter__axes" stroke={tk.borderStrong}>
        {proj.axes.map((axis) => (
          <g key={axis.label}>
            <line
              x1={axis.x1}
              y1={axis.y1}
              x2={axis.x2}
              y2={axis.y2}
              strokeWidth={1}
              opacity={0.55}
            />
            <text
              x={axis.x2}
              y={axis.y2}
              dx={4}
              dy={4}
              className="manifold-scatter__axis-label"
              fill={tk.textMuted}
            >
              {axis.label}
            </text>
          </g>
        ))}
      </g>

      <path
        className="manifold-scatter__trajectory"
        d={pathD}
        fill="none"
        data-testid="manifold-scatter-path"
      />

      {ordered.map(({ p, i }) => {
        const step = positions[i] ?? i;
        const isSelected = step === selectedStep;
        const isHovered = step === hoveredStep;
        const factor = (p.depth + 1) / 2;
        return (
          <circle
            key={step}
            cx={p.px}
            cy={p.py}
            r={radiusFor(p, isSelected || isHovered)}
            fill={colors[i] ?? rampColor((step - tMin) / tSpan)}
            fillOpacity={0.5 + 0.5 * factor}
            stroke={isSelected ? tk.selected : isHovered ? tk.text : 'none'}
            strokeWidth={isSelected ? 2 : 1}
            className="manifold-scatter__point"
            data-testid={`manifold-point-${step}`}
            data-selected={isSelected ? 'true' : 'false'}
            onClick={() => {
              if (!drag.current?.moved) onSelectStep(step);
            }}
            onMouseEnter={() => onHoverStep(step)}
            onMouseLeave={() => onHoverStep(null)}
          >
            <title>{`step ${step}`}</title>
          </circle>
        );
      })}
    </svg>
  );
}
