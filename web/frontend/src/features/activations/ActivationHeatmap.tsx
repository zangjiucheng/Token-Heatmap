import { useMemo, useState } from 'react';
import type {
  ActivationLayerEntry,
  TraceWithActivations,
} from '@/types/activation';
import {
  ACTIVATION_METRIC_LABELS,
  type ActivationMetric,
} from './activation-types';
import './ActivationHeatmap.css';

export interface ActivationHeatmapProps {
  trace: TraceWithActivations;
  submodule: string;
  metric: ActivationMetric;
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

/** Perceptually-uniform viridis-like ramp, eight stops, matched to AttentionLayerHeadGrid. */
const COLOR_RAMP = [
  '#440154',
  '#482878',
  '#3e4989',
  '#31688e',
  '#26828e',
  '#1f9e89',
  '#35b779',
  '#fde725',
];

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function colorFor(value: number, min: number, max: number): string {
  if (max - min < 1e-12) return COLOR_RAMP[0];
  const t = clamp01((value - min) / (max - min));
  const idx = Math.min(
    COLOR_RAMP.length - 1,
    Math.floor(t * COLOR_RAMP.length),
  );
  return COLOR_RAMP[idx];
}

function formatScalar(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(3);
}

function findEntry(
  activations: ActivationLayerEntry[] | undefined,
  layer: number,
  submodule: string,
): ActivationLayerEntry | undefined {
  return activations?.find(
    (entry) => entry.layer === layer && entry.submodule === submodule,
  );
}

export function ActivationHeatmap({
  trace,
  submodule,
  metric,
  selectedStep,
  selectedLayer,
  onSelectCell,
  hoveredStep = null,
  onHoverStep,
}: ActivationHeatmapProps) {
  const [hover, setHover] = useState<GridCell | null>(null);

  const capturedLayers = useMemo(() => {
    const meta = trace.activation_metadata;
    if (!meta) return [] as number[];
    if (meta.captured_layers && meta.captured_layers.length > 0) {
      return [...meta.captured_layers].sort((a, b) => a - b);
    }
    return Array.from({ length: meta.num_layers }, (_, i) => i);
  }, [trace.activation_metadata]);

  const { cells, min, max } = useMemo(() => {
    const out: GridCell[] = [];
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let step = 0; step < trace.steps.length; step += 1) {
      const stepRec = trace.steps[step];
      for (const layer of capturedLayers) {
        const entry = findEntry(stepRec.activations, layer, submodule);
        const value = entry ? entry[metric] : Number.NaN;
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
  }, [trace.steps, capturedLayers, submodule, metric]);

  const numLayers = capturedLayers.length;
  const numSteps = trace.steps.length;

  if (numLayers === 0 || numSteps === 0) {
    return (
      <div
        className="activation-heatmap activation-heatmap--empty"
        data-testid="activation-heatmap"
      >
        <p>No activation data to display.</p>
      </div>
    );
  }

  const width =
    LABEL_PAD_LEFT + numSteps * (CELL_SIZE + CELL_GAP) + LEGEND_WIDTH;
  const height = LABEL_PAD_TOP + numLayers * (CELL_SIZE + CELL_GAP) + 16;

  return (
    <div
      className="activation-heatmap"
      data-testid="activation-heatmap"
      data-num-layers={numLayers}
      data-num-steps={numSteps}
      data-selected-step={selectedStep ?? ''}
      data-hovered-step={hoveredStep ?? ''}
    >
      <svg
        role="img"
        aria-label={`Layer by step activation heatmap, colored by ${ACTIVATION_METRIC_LABELS[metric]} for submodule ${submodule}`}
        className="activation-heatmap__svg"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        data-testid="activation-heatmap-svg"
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
            y={LABEL_PAD_TOP + row * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2 + 4}
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
              fill={valid ? colorFor(cell.value, min, max) : '#e5e5e5'}
              stroke={
                isSelected
                  ? '#ffffff'
                  : isHoveredColumn
                    ? '#d55e00'
                    : 'transparent'
              }
              strokeWidth={isSelected ? 2 : isHoveredColumn ? 1 : 0}
              data-testid={`activation-cell-${cell.step}-${cell.layer}`}
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
              aria-label={`Step ${cell.step} layer ${cell.layer}, ${ACTIVATION_METRIC_LABELS[metric]} ${formatScalar(cell.value)}`}
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
              id="activation-legend-gradient"
              x1="0"
              x2="0"
              y1="1"
              y2="0"
            >
              {COLOR_RAMP.map((c, i) => (
                <stop
                  key={c}
                  offset={`${(i / (COLOR_RAMP.length - 1)) * 100}%`}
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
            fill="url(#activation-legend-gradient)"
            data-testid="activation-heatmap-legend"
          />
          <text x={22} y={12} className="activation-heatmap__legend-label">
            {formatScalar(max)}
          </text>
          <text
            x={22}
            y={numLayers * (CELL_SIZE + CELL_GAP)}
            className="activation-heatmap__legend-label"
          >
            {formatScalar(min)}
          </text>
        </g>
      </svg>
      {hover && (
        <div
          className="activation-heatmap__tooltip"
          data-testid="activation-heatmap-tooltip"
        >
          <strong>
            Step {hover.step} · L{hover.layer}
          </strong>
          <br />
          {ACTIVATION_METRIC_LABELS[metric]}: {formatScalar(hover.value)}
        </div>
      )}
    </div>
  );
}
