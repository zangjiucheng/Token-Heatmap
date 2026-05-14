import { useEffect, useMemo, useState } from 'react';
import type { TraceWithActivations } from '@/types/activation';
import { ActivationHeatmap } from './ActivationHeatmap';
import { ActivationDetailPanel } from './ActivationDetailPanel';
import {
  ACTIVATION_METRICS,
  ACTIVATION_METRIC_LABELS,
  type ActivationMetric,
} from './activation-types';
import './ActivationsTab.css';

export interface ActivationsTabProps {
  trace: TraceWithActivations;
  selectedStep: number | null;
  onSelectStep: (step: number) => void;
  hoveredStep: number | null;
  onHoverStep: (step: number | null) => void;
}

export function ActivationsTab({
  trace,
  selectedStep,
  onSelectStep,
  hoveredStep,
  onHoverStep,
}: ActivationsTabProps) {
  const meta = trace.activation_metadata;
  const submodules = useMemo<readonly string[]>(
    () => meta?.captured_submodules ?? [],
    [meta],
  );
  const [submodule, setSubmodule] = useState<string>(submodules[0] ?? '');
  const [metric, setMetric] = useState<ActivationMetric>('l2_norm');
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);

  // If the producer changes which submodules are captured (extremely unlikely
  // within a single trace, but cheap to guard) keep the selector valid.
  useEffect(() => {
    if (submodules.length === 0) return;
    if (!submodules.includes(submodule)) {
      setSubmodule(submodules[0]);
    }
  }, [submodules, submodule]);

  const capturedLayers = meta?.captured_layers;
  const fallbackLayer = useMemo(() => {
    if (capturedLayers && capturedLayers.length > 0) {
      return capturedLayers[0];
    }
    return 0;
  }, [capturedLayers]);

  if (!meta) {
    return (
      <div
        className="activations-tab activations-tab--empty"
        data-testid="activations-tab-empty"
        role="region"
        aria-label="Activations tab empty state"
      >
        <p>
          This trace was generated without an <code>ActivationProbe</code>.
          Re-run the CLI with <code>--capture-activations</code> to inspect
          activations.
        </p>
      </div>
    );
  }

  const handleSelectCell = (step: number, layer: number) => {
    onSelectStep(step);
    setSelectedLayer(layer);
  };

  const effectiveSelectedLayer =
    selectedLayer ?? (selectedStep != null ? fallbackLayer : null);

  return (
    <div
      className="activations-tab"
      data-testid="activations-tab-content"
      role="region"
      aria-label="Activations tab"
    >
      <div className="activations-tab__controls">
        <div className="activations-tab__control">
          <label htmlFor="activation-submodule-select">Submodule</label>
          <select
            id="activation-submodule-select"
            value={submodule}
            onChange={(e) => setSubmodule(e.target.value)}
            data-testid="activation-submodule-select"
          >
            {submodules.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="activations-tab__control">
          <label htmlFor="activation-metric-select">Metric</label>
          <select
            id="activation-metric-select"
            value={metric}
            onChange={(e) => setMetric(e.target.value as ActivationMetric)}
            data-testid="activation-metric-select"
          >
            {ACTIVATION_METRICS.map((m) => (
              <option key={m} value={m}>
                {ACTIVATION_METRIC_LABELS[m]}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="activations-tab__heatmap">
        <ActivationHeatmap
          trace={trace}
          submodule={submodule}
          metric={metric}
          selectedStep={selectedStep}
          selectedLayer={effectiveSelectedLayer}
          onSelectCell={handleSelectCell}
          hoveredStep={hoveredStep}
          onHoverStep={onHoverStep}
        />
      </div>
      <div className="activations-tab__detail">
        <ActivationDetailPanel
          trace={trace}
          submodule={submodule}
          selectedStep={selectedStep}
          selectedLayer={effectiveSelectedLayer}
        />
      </div>
    </div>
  );
}
