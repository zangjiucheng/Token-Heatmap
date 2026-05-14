import { useEffect, useMemo, useState } from 'react';
import { GeneratedTokenStrip } from '@/features/detail';
import type { ActivationDiff, TraceWithActivations } from '@/types/activation';
import {
  DIFF_METRICS,
  DIFF_METRIC_LABELS,
  compareActivations,
  type DiffAlignmentMode,
  type DiffMetric,
} from './compareActivations';
import { DiffHeatmap } from './DiffHeatmap';
import { DiffDetailPanel } from './DiffDetailPanel';
import './ActivationsTab.css';

export interface DiffViewProps {
  traceA: TraceWithActivations;
  traceB: TraceWithActivations;
  /** Alignment mode override; defaults to 'auto'. */
  alignment?: DiffAlignmentMode;
  /** Top-K changed neurons per (step, layer, submodule). Defaults to 8. */
  topK?: number;
}

type TokenSide = 'A' | 'B';

export function DiffView({
  traceA,
  traceB,
  alignment = 'auto',
  topK,
}: DiffViewProps) {
  const diff: ActivationDiff = useMemo(
    () => compareActivations(traceA, traceB, { align: alignment, topK }),
    [traceA, traceB, alignment, topK],
  );

  const submodules = useMemo<string[]>(() => {
    const setB = new Set(traceB.activation_metadata?.captured_submodules ?? []);
    return (traceA.activation_metadata?.captured_submodules ?? []).filter((s) =>
      setB.has(s),
    );
  }, [traceA.activation_metadata, traceB.activation_metadata]);

  const [submodule, setSubmodule] = useState<string>(submodules[0] ?? '');
  const [metric, setMetric] = useState<DiffMetric>('l2');
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const [tokenSide, setTokenSide] = useState<TokenSide>('A');

  useEffect(() => {
    if (submodules.length === 0) return;
    if (!submodules.includes(submodule)) {
      setSubmodule(submodules[0]);
    }
  }, [submodules, submodule]);

  const handleSelectCell = (step: number, layer: number) => {
    setSelectedStep(step);
    setSelectedLayer(layer);
  };

  const displayTrace = tokenSide === 'A' ? traceA : traceB;
  const mismatches = diff.alignment.mismatches;

  return (
    <div
      className="activations-tab"
      data-testid="diff-view-content"
      role="region"
      aria-label="Activation diff view"
    >
      <div className="activations-tab__controls">
        <div className="activations-tab__control">
          <label htmlFor="diff-submodule-select">Submodule</label>
          <select
            id="diff-submodule-select"
            value={submodule}
            onChange={(e) => setSubmodule(e.target.value)}
            data-testid="diff-submodule-select"
          >
            {submodules.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="activations-tab__control">
          <label htmlFor="diff-metric-select">Metric</label>
          <select
            id="diff-metric-select"
            value={metric}
            onChange={(e) => setMetric(e.target.value as DiffMetric)}
            data-testid="diff-metric-select"
          >
            {DIFF_METRICS.map((m) => (
              <option key={m} value={m}>
                {DIFF_METRIC_LABELS[m]}
              </option>
            ))}
          </select>
        </div>
        <div
          className="activations-tab__control"
          role="group"
          aria-label="Token strip side"
        >
          <label htmlFor="diff-token-side-a">Tokens</label>
          <div className="diff-view__token-toggle" id="diff-token-side-a">
            <button
              type="button"
              data-testid="diff-token-side-a"
              aria-pressed={tokenSide === 'A'}
              onClick={() => setTokenSide('A')}
            >
              Trace A
            </button>
            <button
              type="button"
              data-testid="diff-token-side-b"
              aria-pressed={tokenSide === 'B'}
              onClick={() => setTokenSide('B')}
            >
              Trace B
            </button>
          </div>
        </div>
      </div>
      <div data-testid="diff-token-strip">
        <GeneratedTokenStrip
          trace={displayTrace}
          selectedStep={selectedStep}
          onSelectStep={(s) => setSelectedStep(s)}
          hoveredStep={hoveredStep}
          onHoverStep={setHoveredStep}
        />
      </div>
      <div
        className="diff-view__alignment-status"
        data-testid="diff-alignment-status"
      >
        Alignment: <code>{diff.alignment.mode}</code> · resolved on{' '}
        {diff.steps.length} step{diff.steps.length === 1 ? '' : 's'}
        {mismatches.length > 0
          ? ` · ${mismatches.length} mismatch${mismatches.length === 1 ? '' : 'es'}`
          : ''}
      </div>
      <div className="activations-tab__heatmap">
        <DiffHeatmap
          diff={diff}
          submodule={submodule}
          metric={metric}
          selectedStep={selectedStep}
          selectedLayer={selectedLayer}
          onSelectCell={handleSelectCell}
          hoveredStep={hoveredStep}
          onHoverStep={setHoveredStep}
        />
      </div>
      <div className="activations-tab__detail">
        <DiffDetailPanel
          diff={diff}
          submodule={submodule}
          selectedStep={selectedStep}
          selectedLayer={selectedLayer}
        />
      </div>
    </div>
  );
}
