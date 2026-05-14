import { useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import {
  ATTENTION_METRICS,
  ATTENTION_METRIC_LABELS,
  derivePerHeadScalars,
  type AttentionMetric,
  type PerHeadAttentionScalars,
} from './attention-types';
import './AttentionLayerHeadGrid.css';

export interface AttentionLayerHeadGridProps {
  trace: Trace;
  selectedStep: number | null;
  /** Currently selected (layer, head) — used to draw the selection ring. */
  selectedHead: { layer: number; head: number } | null;
  /** Initial metric. Component owns the metric state via a selector control. */
  initialMetric?: AttentionMetric;
  onSelectHead: (layer: number, head: number) => void;
}

interface GridCell {
  layer: number;
  head: number;
  value: number;
  scalars: PerHeadAttentionScalars;
}

const CELL_SIZE = 28;
const CELL_GAP = 2;
const LABEL_PAD_LEFT = 48;
const LABEL_PAD_TOP = 28;
const LEGEND_WIDTH = 56;

/** Perceptually-uniform-ish viridis-like ramp, eight stops, hex for SSR safety. */
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

export function AttentionLayerHeadGrid({
  trace,
  selectedStep,
  selectedHead,
  initialMetric = 'entropy',
  onSelectHead,
}: AttentionLayerHeadGridProps) {
  const [metric, setMetric] = useState<AttentionMetric>(initialMetric);
  const [hover, setHover] = useState<GridCell | null>(null);

  const numHeads = trace.attention_metadata?.num_attention_heads ?? 0;
  const step =
    selectedStep != null &&
    selectedStep >= 0 &&
    selectedStep < trace.steps.length
      ? trace.steps[selectedStep]
      : null;
  const attention = useMemo(() => step?.attention ?? [], [step]);

  const { cells, min, max } = useMemo(() => {
    const out: GridCell[] = [];
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (const entry of attention) {
      const perHead = derivePerHeadScalars(entry, numHeads);
      for (let h = 0; h < perHead.length; h += 1) {
        const scalars = perHead[h];
        const value = scalars[metric];
        if (Number.isFinite(value)) {
          if (value < lo) lo = value;
          if (value > hi) hi = value;
        }
        out.push({ layer: entry.layer, head: h, value, scalars });
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      lo = 0;
      hi = 1;
    }
    return { cells: out, min: lo, max: hi };
  }, [attention, metric, numHeads]);

  const numLayers = attention.length;
  const width =
    LABEL_PAD_LEFT + numHeads * (CELL_SIZE + CELL_GAP) + LEGEND_WIDTH;
  const height = LABEL_PAD_TOP + numLayers * (CELL_SIZE + CELL_GAP) + 16;

  if (numLayers === 0 || numHeads === 0) {
    const message =
      selectedStep == null
        ? 'Select a generation step from the token strip, heatmap, or overview timelines to inspect attention heads.'
        : 'No attention data is available for the selected step.';
    return (
      <div
        className="attention-grid attention-grid--empty"
        data-testid="attention-layer-head-grid"
      >
        <p>{message}</p>
      </div>
    );
  }

  return (
    <div className="attention-grid" data-testid="attention-layer-head-grid">
      <div className="attention-grid__toolbar">
        <label
          className="attention-grid__metric-label"
          htmlFor="attention-metric-select"
        >
          Metric
        </label>
        <select
          id="attention-metric-select"
          className="attention-grid__metric-select"
          value={metric}
          onChange={(e) => setMetric(e.target.value as AttentionMetric)}
          data-testid="attention-metric-select"
        >
          {ATTENTION_METRICS.map((m) => (
            <option key={m} value={m}>
              {ATTENTION_METRIC_LABELS[m]}
            </option>
          ))}
        </select>
      </div>
      <svg
        role="img"
        aria-label={`Layer by head attention grid, colored by ${ATTENTION_METRIC_LABELS[metric]}`}
        className="attention-grid__svg"
        width={width}
        height={height}
        data-testid="attention-grid-svg"
      >
        {/* Head column labels */}
        {Array.from({ length: numHeads }, (_, h) => (
          <text
            key={`h-${h}`}
            x={LABEL_PAD_LEFT + h * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2}
            y={LABEL_PAD_TOP - 8}
            textAnchor="middle"
            className="attention-grid__axis-label"
          >
            H{h}
          </text>
        ))}
        {/* Layer row labels (use the entry's layer index, not row index) */}
        {attention.map((entry, row) => (
          <text
            key={`l-${entry.layer}`}
            x={LABEL_PAD_LEFT - 8}
            y={LABEL_PAD_TOP + row * (CELL_SIZE + CELL_GAP) + CELL_SIZE / 2 + 4}
            textAnchor="end"
            className="attention-grid__axis-label"
          >
            L{entry.layer}
          </text>
        ))}
        {cells.map((cell) => {
          const row = attention.findIndex((e) => e.layer === cell.layer);
          const x = LABEL_PAD_LEFT + cell.head * (CELL_SIZE + CELL_GAP);
          const y = LABEL_PAD_TOP + row * (CELL_SIZE + CELL_GAP);
          const isSelected =
            selectedHead != null &&
            selectedHead.layer === cell.layer &&
            selectedHead.head === cell.head;
          return (
            <rect
              key={`c-${cell.layer}-${cell.head}`}
              x={x}
              y={y}
              width={CELL_SIZE}
              height={CELL_SIZE}
              fill={colorFor(cell.value, min, max)}
              stroke={isSelected ? '#ffffff' : 'transparent'}
              strokeWidth={isSelected ? 2 : 0}
              data-testid={`attention-cell-${cell.layer}-${cell.head}`}
              data-layer={cell.layer}
              data-head={cell.head}
              data-value={cell.value}
              onMouseEnter={() => setHover(cell)}
              onMouseLeave={() => setHover((h) => (h === cell ? null : h))}
              onClick={() => onSelectHead(cell.layer, cell.head)}
              tabIndex={0}
              role="button"
              aria-label={`Layer ${cell.layer} head ${cell.head}, ${ATTENTION_METRIC_LABELS[metric]} ${formatScalar(cell.value)}`}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelectHead(cell.layer, cell.head);
                }
              }}
              style={{ cursor: 'pointer' }}
            />
          );
        })}
        {/* Legend: vertical gradient bar */}
        <g
          transform={`translate(${LABEL_PAD_LEFT + numHeads * (CELL_SIZE + CELL_GAP) + 12}, ${LABEL_PAD_TOP})`}
        >
          <defs>
            <linearGradient
              id="attn-legend-gradient"
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
            fill="url(#attn-legend-gradient)"
            data-testid="attention-grid-legend"
          />
          <text x={22} y={12} className="attention-grid__legend-label">
            {formatScalar(max)}
          </text>
          <text
            x={22}
            y={numLayers * (CELL_SIZE + CELL_GAP)}
            className="attention-grid__legend-label"
          >
            {formatScalar(min)}
          </text>
        </g>
      </svg>
      {hover && (
        <div
          className="attention-grid__tooltip"
          data-testid="attention-grid-tooltip"
        >
          <strong>
            L{hover.layer} · H{hover.head}
          </strong>
          <br />
          {ATTENTION_METRIC_LABELS[metric]}: {formatScalar(hover.value)}
        </div>
      )}
    </div>
  );
}
