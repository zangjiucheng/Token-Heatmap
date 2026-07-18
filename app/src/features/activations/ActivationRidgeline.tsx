import { useMemo, useState } from 'react';
import type {
  ActivationLayerEntry,
  TraceWithActivations,
} from '@/types/activation';
import './ActivationRidgeline.css';

export interface ActivationRidgelineProps {
  trace: TraceWithActivations;
  submodule: string;
  selectedStep: number | null;
}

const W = 640;
const M = { left: 44, right: 14, top: 12, bottom: 34 } as const;
const ROW_H = 26;

function formatValue(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

/**
 * Per-layer activation value distributions for one step and submodule, drawn
 * as ridgeline silhouettes from the inline histograms the `ActivationProbe`
 * records (layer 0 at the top). The long flat baselines are real: a tiny
 * dense bulk plus a far-away outlier neuron — the "massive activations"
 * signature that scalar norms hide. Hover a row for its quantiles, std, and
 * excess kurtosis.
 */
export function ActivationRidgeline({
  trace,
  submodule,
  selectedStep,
}: ActivationRidgelineProps) {
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

  const step = useMemo(() => {
    if (
      selectedStep != null &&
      selectedStep >= 0 &&
      selectedStep < trace.steps.length
    ) {
      return trace.steps[selectedStep];
    }
    return trace.steps.find((s) => (s.activations ?? []).length > 0) ?? null;
  }, [trace, selectedStep]);

  const entries = useMemo<ActivationLayerEntry[]>(() => {
    const all = step?.activations ?? [];
    return all
      .filter((a) => a.submodule === submodule && a.histogram)
      .sort((a, b) => a.layer - b.layer);
  }, [step, submodule]);

  if (!step || entries.length === 0) {
    return (
      <div
        className="activation-ridgeline activation-ridgeline--empty"
        data-testid="activation-ridgeline-empty"
      >
        <p>
          This trace has no activation histograms. Re-capture it with the
          current <code>ActivationProbe</code> to see per-layer distributions.
        </p>
      </div>
    );
  }

  const xMin = Math.min(...entries.map((e) => e.histogram!.bin_min));
  const xMax = Math.max(...entries.map((e) => e.histogram!.bin_max));
  const span = xMax - xMin || 1;
  const xOf = (v: number) =>
    M.left + ((v - xMin) / span) * (W - M.left - M.right);

  const height = M.top + entries.length * ROW_H + M.bottom;

  const silhouette = (entry: ActivationLayerEntry, row: number): string => {
    const hist = entry.histogram!;
    const counts = hist.counts;
    const peak = Math.max(1, ...counts);
    const baseY = M.top + (row + 1) * ROW_H;
    const binW = (hist.bin_max - hist.bin_min) / counts.length;
    const pts: string[] = [`M${xOf(hist.bin_min).toFixed(1)},${baseY}`];
    counts.forEach((c, i) => {
      const cx = hist.bin_min + (i + 0.5) * binW;
      const h = (c / peak) * ROW_H * 1.55;
      pts.push(`L${xOf(cx).toFixed(1)},${(baseY - h).toFixed(1)}`);
    });
    pts.push(`L${xOf(hist.bin_max).toFixed(1)},${baseY} Z`);
    return pts.join(' ');
  };

  const xTicks = Array.from({ length: 5 }, (_, i) => xMin + (span * i) / 4);
  const hoveredEntry =
    hoveredLayer != null
      ? entries.find((e) => e.layer === hoveredLayer)
      : undefined;

  return (
    <figure
      className="activation-ridgeline"
      data-testid="activation-ridgeline"
    >
      <figcaption className="activation-ridgeline__caption">
        Value distribution per layer — step {step.step} ·{' '}
        <code>{submodule}</code>
      </figcaption>
      <div className="activation-ridgeline__plot">
        <svg
          viewBox={`0 0 ${W} ${height}`}
          role="img"
          aria-label={`Activation value distributions per layer for ${submodule}`}
          onMouseLeave={() => setHoveredLayer(null)}
        >
          {xTicks.map((t) => (
            <g key={t}>
              <line
                className="activation-ridgeline__grid"
                x1={xOf(t)}
                x2={xOf(t)}
                y1={M.top}
                y2={height - M.bottom}
              />
              <text
                className="activation-ridgeline__tick"
                x={xOf(t)}
                y={height - M.bottom + 14}
                textAnchor="middle"
              >
                {formatValue(t)}
              </text>
            </g>
          ))}
          <text
            className="activation-ridgeline__tick"
            x={(M.left + W - M.right) / 2}
            y={height - 4}
            textAnchor="middle"
          >
            Activation value
          </text>
          {entries.map((entry, row) => {
            const baseY = M.top + (row + 1) * ROW_H;
            const isHovered = entry.layer === hoveredLayer;
            return (
              <g key={entry.layer}>
                <text
                  className="activation-ridgeline__tick"
                  x={M.left - 6}
                  y={baseY}
                  textAnchor="end"
                >
                  L{entry.layer}
                </text>
                <path
                  className={
                    isHovered
                      ? 'activation-ridgeline__silhouette activation-ridgeline__silhouette--hovered'
                      : 'activation-ridgeline__silhouette'
                  }
                  d={silhouette(entry, row)}
                  data-testid={`ridgeline-row-${entry.layer}`}
                />
                <rect
                  className="activation-ridgeline__hit"
                  x={M.left}
                  y={baseY - ROW_H}
                  width={W - M.left - M.right}
                  height={ROW_H}
                  onMouseEnter={() => setHoveredLayer(entry.layer)}
                />
              </g>
            );
          })}
        </svg>
        {hoveredEntry && (
          <div className="activation-ridgeline__tooltip" data-testid="ridgeline-tooltip">
            <div className="activation-ridgeline__tooltip-title">
              Layer {hoveredEntry.layer}
            </div>
            <div>
              p01–p99: {formatValue(hoveredEntry.quantiles?.p01 ?? NaN)} …{' '}
              {formatValue(hoveredEntry.quantiles?.p99 ?? NaN)}
            </div>
            <div>
              range: {formatValue(hoveredEntry.histogram!.bin_min)} …{' '}
              {formatValue(hoveredEntry.histogram!.bin_max)}
            </div>
            <div>std {formatValue(hoveredEntry.std ?? NaN)}</div>
            <div>excess kurtosis {formatValue(hoveredEntry.kurtosis ?? NaN)}</div>
          </div>
        )}
      </div>
    </figure>
  );
}
