import type { ActivationDiff } from '@/types/activation';
import './ActivationDetailPanel.css';

export interface DiffDetailPanelProps {
  diff: ActivationDiff;
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

export function DiffDetailPanel({
  diff,
  submodule,
  selectedStep,
  selectedLayer,
}: DiffDetailPanelProps) {
  if (
    selectedStep == null ||
    selectedLayer == null ||
    selectedStep < 0 ||
    selectedStep >= diff.steps.length
  ) {
    return (
      <div
        className="activation-detail-panel activation-detail-panel--empty"
        data-testid="diff-detail-panel"
        role="region"
        aria-label="Diff detail"
      >
        <p>Click a cell in the heatmap to see its top-k changed neurons.</p>
      </div>
    );
  }

  const step = diff.steps[selectedStep];
  const entry = step.delta.find(
    (e) => e.layer === selectedLayer && e.submodule === submodule,
  );

  if (!entry) {
    return (
      <div
        className="activation-detail-panel activation-detail-panel--empty"
        data-testid="diff-detail-panel"
        role="region"
        aria-label="Diff detail"
      >
        <p>
          No diff data for step {selectedStep}, layer {selectedLayer}, submodule
          {' '}
          {submodule}.
        </p>
      </div>
    );
  }

  return (
    <div
      className="activation-detail-panel"
      data-testid="diff-detail-panel"
      role="region"
      aria-label={`Diff detail for step ${selectedStep} layer ${selectedLayer}`}
    >
      <header className="activation-detail-panel__header">
        <h3
          className="activation-detail-panel__title"
          data-testid="diff-detail-panel-title"
        >
          Step {selectedStep} · L{selectedLayer} · {submodule}
        </h3>
        <span className="activation-detail-panel__subtitle">
          L2 {formatValue(entry.l2)} · cos {formatValue(entry.cosine)}
        </span>
      </header>
      <table
        className="activation-detail-panel__table"
        data-testid="diff-detail-panel-table"
      >
        <thead>
          <tr>
            <th scope="col">Rank</th>
            <th scope="col">Neuron</th>
            <th scope="col">Δ (A − B)</th>
          </tr>
        </thead>
        <tbody>
          {entry.top_changed_neurons.map((n, i) => (
            <tr
              key={`${n.index}-${i}`}
              data-testid={`diff-top-neuron-${i}`}
              data-neuron-index={n.index}
              data-neuron-delta={n.delta}
            >
              <td>{i + 1}</td>
              <td>#{n.index}</td>
              <td>{formatValue(n.delta)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
