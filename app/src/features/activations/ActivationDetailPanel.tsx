import type { TraceWithActivations } from '@/types/activation';
import './ActivationDetailPanel.css';

export type NeuronRankingMode = 'step' | 'twera';

export interface ActivationDetailPanelProps {
  trace: TraceWithActivations;
  submodule: string;
  selectedStep: number | null;
  selectedLayer: number | null;
  /** 'step' = per-step top-|value| neurons; 'twera' = whole-trace TWERA ranking. */
  rankingMode?: NeuronRankingMode;
}

function formatValue(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(2);
  if (Math.abs(v) >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

function EmptyPanel({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="activation-detail-panel activation-detail-panel--empty"
      data-testid="activation-detail-panel"
      role="region"
      aria-label="Activation detail"
    >
      <p>{children}</p>
    </div>
  );
}

/**
 * Whole-trace TWERA ranking for the selected (layer, submodule): neurons ranked
 * by their expected residual-direct attribution to the realized next token. A
 * single-trace approximation of Target-Weighted Expected Residual Attribution
 * (transformer-circuits.pub/2025/attribution-graphs). Independent of the
 * selected step.
 */
function TweraPanel({
  trace,
  submodule,
  selectedLayer,
}: {
  trace: TraceWithActivations;
  submodule: string;
  selectedLayer: number | null;
}) {
  const attribution = trace.neuron_attribution;
  if (!attribution) {
    return (
      <EmptyPanel>
        No TWERA data. Re-run the CLI with{' '}
        <code>--capture-full-activations</code> to compute the neuron
        attribution.
      </EmptyPanel>
    );
  }
  if (selectedLayer == null) {
    return <EmptyPanel>Click a heatmap cell to pick a layer.</EmptyPanel>;
  }
  const layer = attribution.layers?.find(
    (l) => l.layer === selectedLayer && l.submodule === submodule,
  );
  if (!layer || !layer.neurons?.length) {
    return (
      <EmptyPanel>
        No TWERA ranking for L{selectedLayer} · {submodule}.
      </EmptyPanel>
    );
  }

  return (
    <div
      className="activation-detail-panel"
      data-testid="activation-detail-panel"
      role="region"
      aria-label={`TWERA ranking for layer ${selectedLayer}`}
    >
      <header className="activation-detail-panel__header">
        <h3
          className="activation-detail-panel__title"
          data-testid="activation-detail-panel-title"
        >
          Whole trace · TWERA · L{selectedLayer} · {submodule}
        </h3>
        <span className="activation-detail-panel__subtitle">
          mean over {attribution.n_steps} steps · effect×activation toward the
          realized token
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
            <th
              scope="col"
              title="Target-Weighted Expected Residual Attribution"
            >
              TWERA
            </th>
            <th scope="col">mean act</th>
          </tr>
        </thead>
        <tbody>
          {layer.neurons.map((n, i) => (
            <tr
              key={`${n.index}-${i}`}
              data-testid={`activation-twera-neuron-${i}`}
              data-neuron-index={n.index}
              data-neuron-twera={n.twera}
            >
              <td>{i + 1}</td>
              <td>#{n.index}</td>
              <td>{formatValue(n.twera)}</td>
              <td>
                {n.mean_activation == null
                  ? '—'
                  : formatValue(n.mean_activation)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ActivationDetailPanel({
  trace,
  submodule,
  selectedStep,
  selectedLayer,
  rankingMode = 'step',
}: ActivationDetailPanelProps) {
  if (rankingMode === 'twera') {
    return (
      <TweraPanel
        trace={trace}
        submodule={submodule}
        selectedLayer={selectedLayer}
      />
    );
  }

  if (
    selectedStep == null ||
    selectedLayer == null ||
    selectedStep < 0 ||
    selectedStep >= trace.steps.length
  ) {
    return (
      <EmptyPanel>Click a cell in the heatmap to see its top-k neurons.</EmptyPanel>
    );
  }

  const step = trace.steps[selectedStep];
  const entry = step.activations?.find(
    (e) => e.layer === selectedLayer && e.submodule === submodule,
  );

  if (!entry) {
    return (
      <EmptyPanel>
        No activation data for step {selectedStep}, layer {selectedLayer},
        submodule {submodule}.
      </EmptyPanel>
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
