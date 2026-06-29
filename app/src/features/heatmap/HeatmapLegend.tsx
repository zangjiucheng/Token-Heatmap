import { useMemo } from 'react';
import { sampleColor, rgbToCss } from './colormap';
import type { ValueCol } from './buildGrid';
import './HeatmapLegend.css';

export interface HeatmapLegendProps {
  min: number;
  max: number;
  valueCol: ValueCol;
  /** Number of color stops to use in the gradient. */
  stops?: number;
}

function formatValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 100 || Math.abs(v) < 0.001) return v.toExponential(2);
  return v.toFixed(3);
}

export function HeatmapLegend({
  min,
  max,
  valueCol,
  stops = 16,
}: HeatmapLegendProps) {
  const gradient = useMemo(() => {
    const parts: string[] = [];
    for (let i = 0; i < stops; i += 1) {
      const t = stops === 1 ? 0 : i / (stops - 1);
      parts.push(`${rgbToCss(sampleColor(t))} ${(t * 100).toFixed(1)}%`);
    }
    return `linear-gradient(to right, ${parts.join(', ')})`;
  }, [stops]);

  return (
    <div
      className="heatmap-legend"
      role="img"
      aria-label={`Color scale for ${valueCol} from ${formatValue(min)} to ${formatValue(max)}`}
      data-testid="heatmap-legend"
    >
      <span
        className="heatmap-legend__label heatmap-legend__label--min"
        data-testid="heatmap-legend-min"
      >
        {formatValue(min)}
      </span>
      <span className="heatmap-legend__bar" style={{ backgroundImage: gradient }} />
      <span
        className="heatmap-legend__label heatmap-legend__label--max"
        data-testid="heatmap-legend-max"
      >
        {formatValue(max)}
      </span>
      <span
        className="heatmap-legend__title"
        data-testid="heatmap-legend-title"
      >
        {valueCol}
      </span>
    </div>
  );
}
