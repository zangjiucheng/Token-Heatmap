import type { Trace } from '@/types/trace';
import { CandidateTable } from './CandidateTable';
import { escapeToken } from './escapeToken';
import './StepDetailPanel.css';

export interface StepDetailPanelProps {
  trace: Trace | null;
  selectedStep: number | null;
}

function formatProb(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (v >= 0.001 || v === 0) return v.toFixed(4);
  return v.toExponential(2);
}

function formatNumber(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}

export function StepDetailPanel({ trace, selectedStep }: StepDetailPanelProps) {
  if (trace === null) {
    return (
      <div
        className="step-detail-panel step-detail-panel--empty"
        data-testid="step-detail-panel"
        role="region"
        aria-label="Step detail"
        tabIndex={-1}
      >
        <p className="step-detail-panel__empty">
          No trace loaded — load a trace to see step details.
        </p>
      </div>
    );
  }

  if (
    selectedStep == null ||
    selectedStep < 0 ||
    selectedStep >= trace.steps.length
  ) {
    return (
      <div
        className="step-detail-panel step-detail-panel--empty"
        data-testid="step-detail-panel"
        role="region"
        aria-label="Step detail"
        tabIndex={-1}
      >
        <p className="step-detail-panel__empty">
          Select a generation step from the token strip, heatmap, or overview
          timelines to inspect token details and top candidates.
        </p>
      </div>
    );
  }

  const step = trace.steps[selectedStep];
  const dist = step.processed;
  const selected = step.selected;

  return (
    <div
      className="step-detail-panel"
      data-testid="step-detail-panel"
      role="region"
      aria-label={`Step detail for step ${step.step}`}
      tabIndex={-1}
    >
      <header className="step-detail-panel__header">
        <h3 data-testid="step-detail-panel-step">Step {step.step}</h3>
        <div className="step-detail-panel__stats">
          <span className="step-detail-panel__stat-label">Selected token</span>
          <span
            className="step-detail-panel__stat-value"
            data-testid="step-detail-panel-selected-token"
            title={selected.token}
          >
            "{escapeToken(selected.token)}"
          </span>

          <span className="step-detail-panel__stat-label">Prob (selected)</span>
          <span
            className="step-detail-panel__stat-value"
            data-testid="step-detail-panel-selected-prob"
          >
            {formatProb(dist.selected_prob)}
          </span>

          <span className="step-detail-panel__stat-label">Rank (selected)</span>
          <span
            className="step-detail-panel__stat-value"
            data-testid="step-detail-panel-selected-rank"
          >
            {dist.selected_rank}
          </span>

          <span className="step-detail-panel__stat-label">Entropy</span>
          <span
            className="step-detail-panel__stat-value"
            data-testid="step-detail-panel-entropy"
          >
            {formatNumber(dist.entropy)}
          </span>

          <span className="step-detail-panel__stat-label">k_used</span>
          <span
            className="step-detail-panel__stat-value"
            data-testid="step-detail-panel-k-used"
          >
            {dist.k_used}
          </span>
        </div>
      </header>

      <div>
        <h4 className="step-detail-panel__section-title">
          Top {dist.candidates.length} candidates
        </h4>
        <CandidateTable
          candidates={dist.candidates}
          selectedTokenId={selected.token_id}
        />
      </div>
    </div>
  );
}
