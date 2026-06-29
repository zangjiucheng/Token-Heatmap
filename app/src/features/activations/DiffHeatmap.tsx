import { useMemo, useState } from 'react';
import type { ActivationDiff } from '@/types/activation';
import { DIFF_METRIC_LABELS, type DiffMetric } from './compareActivations';
import './ActivationHeatmap.css';

export interface DiffHeatmapProps {
  diff: ActivationDiff;
  submodule: string;
  metric: DiffMetric;
  selectedStep: number | null;
  selectedLayer: number | null;
  onSelectCell: (step: number, layer: number) => void;
  hoveredStep?: number | null;
  onHoverStep?: (step: number | null) => void;
}

interface GridCell {
  step: number;
  layer: number;
  value: number;
}

const CELL_SIZE = 22;
const CELL_GAP = 1;
const LABEL_PAD_LEFT = 40;
const LABEL_PAD_TOP = 24;
const LEGEND_WIDTH = 56;

/** Sequential reds ramp for L2 deltas (always ≥ 0). */
const RAMP_L2 = [
  '#fff5f0',
  '#fee0d2',
  '#fcbba1',
  '#fc9272',
  '#fb6a4a',
  '#ef3b2c',
  '#cb181d',
  '#67000d',
];

/** Diverging blue–white–red ramp for cosine deltas (range −1..1). */
const RAMP_COSINE = [
  '#053061',
  'rgb(33, 102, 172)',
  'rgb(67, 147, 195)',
  '#92c5de',
  '#f7f7f7',
  '#f4a582',
  '#d6604d',
  '#b2182b',
];

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function colorForL2(value: number, min: number, max: number): string {
  if (max - min < 1e-12) return RAMP_L2[0];
  const t = clamp01((value - min) / (max - min));
  const idx = Math.min(RAMP_L2.length - 1, Math.floor(t * RAMP_L2.length));
  return RAMP_L2[idx];
}

function colorForCosine(value: number): string {
  // Map [-1, 1] → [0, RAMP_COSINE.length).
  const t = clamp01((value + 1) / 2);
  const idx = Math.min(
    RAMP_COSINE.length - 1,
    Math.floor(t * RAMP_COSINE.length),
  );
  return RAMP_COSINE[idx];
}

function formatScalar(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(3);
}

export function DiffHeatmap({
  diff,
  submodule,
  metric,
  selectedStep,
  selectedLayer,
  onSelectCell,
  hoveredStep = null,
  onHoverStep,
}: DiffHeatmapProps) {
  const [hover, setHover] = useState<GridCell | null>(null);

  const capturedLayers = useMemo(() => {
    const set = new Set<number>();
    for (const step of diff.steps) {
      for (const d of step.delta) {
        if (d.submodule === submodule) set.add(d.layer);
      }
    }
    return [...set].sort((a, b) => a - b);
  }, [diff.steps, submodule]);

  const { cells, min, max } = useMemo(() => {
    const out: GridCell[] = [];
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let step = 0; step < diff.steps.length; step += 1) {
      const stepRec = diff.steps[step];
      for (const layer of capturedLayers) {
        const d = stepRec.delta.find(
          (x) => x.layer === layer && x.submodule === submodule,
        );
        const value = d ? (metric === 'l2' ? d.l2 : d.cosine) : Number.NaN;
        if (Number.isFinite(value)) {
          if (value < lo) lo = value;
          if (value > hi) hi = value;
        }
        out.push({ step, layer, value });
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      lo = 0;
      hi = 1;
    }
    return { cells: out, min: lo, max: hi };
  }, [diff.steps, capturedLayers, submodule, metric]);

  const numLayers = capturedLayers.length;
  const numSteps = diff.steps.length;

  if (numLayers === 0 || numSteps === 0) {
    return (
      <div
        className="activation-heatmap activation-heatmap--empty"
        data-testid="diff-heatmap"
      >
        <p>No diff data to display.</p>
      </div>
    );
  }

  const width =
    LABEL_PAD_LEFT + numSteps * (CELL_SIZE + CELL_GAP) + LEGEND_WIDTH;
  const height = LABEL_PAD_TOP + numLayers * (CELL_SIZE + CELL_GAP) + 16;

  const colorFn =
    metric === 'l2'
      ? (v: number) => colorForL2(v, min, max)
      : (v: number) => colorForCosine(v);
  const ramp = metric === 'l2' ? RAMP_L2 : RAMP_COSINE;
  const legendMax = metric === 'cosine' ? 1 : max;
  const legendMin = metric === 'cosine' ? -1 : min;

  return (
    <div
      className="activation-heatmap"
      data-testid="diff-heatmap"
      data-num-layers={numLayers}
      data-num-steps={numSteps}
      data-metric={metric}
      data-selected-step={selectedStep ?? ''}
      data-hovered-step={hoveredStep ?? ''}
    >
      <div className="viz-frame">
        <svg
          role="img"
          aria-label={`Layer by step diff heatmap, colored by ${DIFF_METRIC_LABELS[metric]} for submodule ${submodule}`}
          className="activation-heatmap__svg"
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMinYMid meet"
          data-testid="diff-heatmap-svg"
        >
          {Array.from({ length: numSteps }, (_, step) => (
            <text
              key={`s-${step}`}
              x={LABEL_PAD_LEFT + step * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2}
              y={LABEL_PAD_TOP - 8}
              textAnchor="middle"
              className="activation-heatmap__axis-label"
            >
              {step}
            </text>
          ))}
          {capturedLayers.map((layer, row) => (
            <text
              key={`l-${layer}`}
              x={LABEL_PAD_LEFT - 8}
              y={
                LABEL_PAD_TOP + row * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2 + 4
              }
              textAnchor="end"
              className="activation-heatmap__axis-label"
            >
              L{layer}
            </text>
          ))}
          {cells.map((cell) => {
            const row = capturedLayers.indexOf(cell.layer);
            const x = LABEL_PAD_LEFT + cell.step * (CELL_SIZE + CELL_GAP);
            const y = LABEL_PAD_TOP + row * (CELL_SIZE + CELL_GAP);
            const isSelected =
              selectedStep === cell.step && selectedLayer === cell.layer;
            const isHoveredColumn =
              hoveredStep != null && hoveredStep === cell.step;
            const valid = Number.isFinite(cell.value);
            return (
              <rect
                key={`c-${cell.step}-${cell.layer}`}
                x={x}
                y={y}
                width={CELL_SIZE}
                height={CELL_SIZE}
                fill={valid ? colorFn(cell.value) : '#e5e5e5'}
                stroke={
                  isSelected
                    ? '#ffffff'
                    : isHoveredColumn
                      ? '#d55e00'
                      : 'transparent'
                }
                strokeWidth={isSelected ? 2 : isHoveredColumn ? 1 : 0}
                data-testid={`diff-cell-${cell.step}-${cell.layer}`}
                data-step={cell.step}
                data-layer={cell.layer}
                data-value={cell.value}
                onMouseEnter={() => {
                  setHover(cell);
                  onHoverStep?.(cell.step);
                }}
                onMouseLeave={() => {
                  setHover((h) => (h === cell ? null : h));
                  onHoverStep?.(null);
                }}
                onClick={() => onSelectCell(cell.step, cell.layer)}
                tabIndex={0}
                role="button"
                aria-label={`Step ${cell.step} layer ${cell.layer}, ${DIFF_METRIC_LABELS[metric]} ${formatScalar(cell.value)}`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectCell(cell.step, cell.layer);
                  }
                }}
                style={{ cursor: 'pointer' }}
              />
            );
          })}
          <g
            transform={`translate(${LABEL_PAD_LEFT + numSteps * (CELL_SIZE + CELL_GAP) + 12}, ${LABEL_PAD_TOP})`}
          >
            <defs>
              <linearGradient
                id={`diff-legend-gradient-${metric}`}
                x1="0"
                x2="0"
                y1="1"
                y2="0"
              >
                {ramp.map((c, i) => (
                  <stop
                    key={c}
                    offset={`${(i / (ramp.length - 1)) * 100}%`}
                    stopColor={c}
                  />
                ))}
              </linearGradient>
            </defs>
            <rect
              x={0}
              y={0}
              width={16}
              height={numLayers * (CELL_SIZE + CELL_GAP)}
              fill={`url(#diff-legend-gradient-${metric})`}
              data-testid="diff-heatmap-legend"
            />
            <text x={22} y={12} className="activation-heatmap__legend-label">
              {formatScalar(legendMax)}
            </text>
            <text
              x={22}
              y={numLayers * (CELL_SIZE + CELL_GAP)}
              className="activation-heatmap__legend-label"
            >
              {formatScalar(legendMin)}
            </text>
          </g>
        </svg>
      </div>
      {hover && (
        <div
          className="activation-heatmap__tooltip"
          data-testid="diff-heatmap-tooltip"
        >
          <strong>
            Step {hover.step} · L{hover.layer}
          </strong>
          <br />
          {DIFF_METRIC_LABELS[metric]}: {formatScalar(hover.value)}
        </div>
      )}
    </div>
  );
}
