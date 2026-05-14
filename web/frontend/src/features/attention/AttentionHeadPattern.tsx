import { useEffect, useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import { loadAttentionSidecar } from './loadAttentionSidecar';
import type { AttentionSidecar } from './attention-types';
import './AttentionHeadPattern.css';

export interface AttentionHeadPatternProps {
  trace: Trace;
  selectedStep: number | null;
  selectedHead: { layer: number; head: number } | null;
  /** Override the global fetch — used by tests. */
  fetchImpl?: typeof fetch;
}

interface PointBar {
  position: number;
  weight: number;
  sourceToken: string;
}

const CHART_WIDTH = 480;
const CHART_HEIGHT = 160;
const PADDING_X = 32;
const PADDING_Y = 16;

function tokenAtPosition(trace: Trace, position: number, stepIdx: number): string {
  const promptLen = trace.tokens.prompt_tokens.length;
  if (position < promptLen) {
    return trace.tokens.prompt_tokens[position] ?? `pos ${position}`;
  }
  const generatedIdx = position - promptLen;
  if (generatedIdx < 0 || generatedIdx >= trace.steps.length) {
    return `pos ${position}`;
  }
  // The position refers to the source token that the current step is
  // attending to; for generated tokens that means the token chosen at step
  // `generatedIdx` (always < stepIdx given causal masking).
  if (generatedIdx > stepIdx) return `pos ${position}`;
  return trace.steps[generatedIdx].selected.token;
}

export function AttentionHeadPattern({
  trace,
  selectedStep,
  selectedHead,
  fetchImpl,
}: AttentionHeadPatternProps) {
  const [sidecar, setSidecar] = useState<AttentionSidecar | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const step =
    selectedStep != null && selectedStep >= 0 && selectedStep < trace.steps.length
      ? trace.steps[selectedStep]
      : null;
  const ref = step?.attention_sidecar_ref ?? null;

  useEffect(() => {
    setSidecar(null);
    setLoadError(null);
    if (!ref || selectedStep == null) return;
    setLoading(true);
    let cancelled = false;
    loadAttentionSidecar(selectedStep, ref, fetchImpl)
      .then((payload) => {
        if (cancelled) return;
        setSidecar(payload);
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setLoadError(err.message);
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ref, selectedStep, fetchImpl]);

  const bars: PointBar[] = useMemo(() => {
    if (!step || !selectedHead || selectedStep == null) return [];
    // Prefer sidecar (full distribution) when available.
    if (sidecar) {
      const layerEntry = sidecar.layers.find((l) => l.layer === selectedHead.layer);
      if (layerEntry && layerEntry.attention_distribution[selectedHead.head]) {
        const row = layerEntry.attention_distribution[selectedHead.head];
        return row.map((weight, position) => ({
          position,
          weight,
          sourceToken: tokenAtPosition(trace, position, selectedStep),
        }));
      }
    }
    // Fallback: sparse top_positions from the inline layer entry. This is
    // the layer-mean aggregation rather than a single head, but it is the
    // best we can do without a sidecar.
    const entry = step.attention?.find((e) => e.layer === selectedHead.layer);
    if (!entry) return [];
    return entry.top_positions.map((p) => ({
      position: p.position,
      weight: p.weight,
      sourceToken: tokenAtPosition(trace, p.position, selectedStep),
    }));
  }, [sidecar, step, selectedHead, selectedStep, trace]);

  if (!selectedHead) {
    return (
      <div
        className="attention-head-pattern attention-head-pattern--empty"
        data-testid="attention-head-pattern"
      >
        <p>Click a (layer, head) cell to inspect its attention pattern.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div
        className="attention-head-pattern"
        data-testid="attention-head-pattern"
      >
        <p>Loading attention sidecar…</p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div
        className="attention-head-pattern attention-head-pattern--error"
        role="alert"
        data-testid="attention-head-pattern"
      >
        <p>Could not load sidecar: {loadError}</p>
      </div>
    );
  }

  const maxWeight = bars.reduce((m, b) => Math.max(m, b.weight), 0) || 1;
  const innerW = CHART_WIDTH - PADDING_X * 2;
  const innerH = CHART_HEIGHT - PADDING_Y * 2;
  const barW = bars.length > 0 ? innerW / bars.length : 0;
  const source = sidecar ? 'sidecar' : 'inline-top-positions';

  return (
    <div
      className="attention-head-pattern"
      data-testid="attention-head-pattern"
      data-source={source}
    >
      <header className="attention-head-pattern__header">
        <span data-testid="attention-head-pattern-title">
          Layer {selectedHead.layer} · Head {selectedHead.head}
        </span>
      </header>
      {bars.length === 0 ? (
        <p className="attention-head-pattern__empty">
          No attention weights to display for this head.
        </p>
      ) : (
        <svg
          role="img"
          aria-label={`Attention pattern for layer ${selectedHead.layer} head ${selectedHead.head}`}
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          preserveAspectRatio="xMidYMid meet"
          className="attention-head-pattern__svg"
          data-testid="attention-head-pattern-svg"
        >
          {/* Axis */}
          <line
            x1={PADDING_X}
            x2={CHART_WIDTH - PADDING_X}
            y1={CHART_HEIGHT - PADDING_Y}
            y2={CHART_HEIGHT - PADDING_Y}
            stroke="currentColor"
            strokeOpacity={0.4}
          />
          {bars.map((b, i) => {
            const h = (b.weight / maxWeight) * innerH;
            const x = PADDING_X + i * barW;
            const y = CHART_HEIGHT - PADDING_Y - h;
            return (
              <g key={`${b.position}-${i}`}>
                <rect
                  x={x + 1}
                  y={y}
                  width={Math.max(1, barW - 2)}
                  height={h}
                  fill="#26828e"
                  data-testid={`attention-pattern-bar-${b.position}`}
                  data-position={b.position}
                  data-weight={b.weight}
                >
                  <title>{`Position ${b.position} (${b.sourceToken}): ${b.weight.toFixed(4)}`}</title>
                </rect>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
