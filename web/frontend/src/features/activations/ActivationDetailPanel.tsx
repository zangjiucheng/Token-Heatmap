import type { TraceWithActivations } from '@/types/activation';
import './ActivationDetailPanel.css';

export interface ActivationDetailPanelProps {
  trace: TraceWithActivations;
  submodule: string;
  selectedStep: number | null;
  selectedLayer: number | null;
}

function formatValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(2);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

export function ActivationDetailPanel({
  trace,
  submodule,
  selectedStep,
  selectedLayer,
}: ActivationDetailPanelProps) {
  if (
    selectedStep == null ||
    selectedLayer == null ||
    selectedStep < 0 ||
    selectedStep >= trace.steps.length
  ) {
    return (
      <div
        className="activation-detail-panel activation-detail-panel--empty"
        data-testid="activation-detail-panel"
        role="region"
        aria-label="Activation detail"
      >
        <p>Click a cell in the heatmap to see its top-k neurons.</p>
      </div>
    );
  }

  const step = trace.steps[selectedStep];
  const entry = step.activations?.find(
    (e) => e.layer === selectedLayer && e.submodule === submodule,
  );

  if (!entry) {
    return (
      <div
        className="activation-detail-panel activation-detail-panel--empty"
        data-testid="activation-detail-panel"
        role="region"
        aria-label="Activation detail"
      >
        <p>
          No activation data for step {selectedStep}, layer {selectedLayer},
          submodule {submodule}.
        </p>
      </div>
    );
  }

  return (
    <div
      className="activation-detail-panel"
      data-testid="activation-detail-panel"
      role="region"
      aria-label={`Activation detail for step ${selectedStep} layer ${selectedLayer}`}
    >
      <header className="activation-detail-panel__header">
        <h3
          className="activation-detail-panel__title"
          data-testid="activation-detail-panel-title"
        >
          Step {selectedStep} · L{selectedLayer} · {submodule}
        </h3>
        <span className="activation-detail-panel__subtitle">
          L2 {formatValue(entry.l2_norm)} · mean|x|{' '}
          {formatValue(entry.mean_abs)} · sparsity {formatValue(entry.sparsity)}
        </span>
      </header>
      <table
        className="activation-detail-panel__table"
        data-testid="activation-detail-panel-table"
      >
        <thead>
          <tr>
            <th scope="col">Rank</th>
            <th scope="col">Neuron</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {entry.top_neurons.map((n, i) => (
            <tr
              key={`${n.index}-${i}`}
              data-testid={`activation-top-neuron-${i}`}
              data-neuron-index={n.index}
              data-neuron-value={n.value}
            >
              <td>{i + 1}</td>
              <td>#{n.index}</td>
              <td>{formatValue(n.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
