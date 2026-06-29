import type { Trace } from '@/types/trace';
import type { SelectedHead } from '@/hooks/useViewState';
import { AttentionLayerHeadGrid } from './AttentionLayerHeadGrid';
import './AttentionTab.css';

export interface AttentionTabProps {
  trace: Trace;
  selectedStep: number | null;
  selectedHead: SelectedHead | null;
  onSelectHead: (head: SelectedHead | null) => void;
}

export function AttentionTab({
  trace,
  selectedStep,
  selectedHead,
  onSelectHead,
}: AttentionTabProps) {
  if (!trace.attention_metadata) {
    return (
      <div
        className="attention-tab attention-tab--empty"
        data-testid="attention-tab-empty"
        role="region"
        aria-label="Attention tab empty state"
      >
        <p>
          This trace was generated without <code>--capture-attention</code>.
          Re-run the CLI with that flag to inspect attention.
        </p>
      </div>
    );
  }

  if (selectedStep == null) {
    return (
      <div
        className="attention-tab attention-tab--empty"
        data-testid="attention-tab-step-empty"
        role="region"
        aria-label="Attention tab step selection prompt"
      >
        <p>
          Select a generation step from the token strip, heatmap, or overview
          timelines to inspect attention heads and logit-lens predictions.
        </p>
      </div>
    );
  }

  return (
    <div
      className="attention-tab"
      data-testid="attention-tab-content"
      role="region"
      aria-label="Attention tab"
    >
      <div className="attention-tab__grid">
        <AttentionLayerHeadGrid
          trace={trace}
          selectedStep={selectedStep}
          selectedHead={selectedHead}
          onSelectHead={(layer, head) => onSelectHead({ layer, head })}
        />
      </div>
    </div>
  );
}
