import type { Trace } from '@/types/trace';
import { LogitLensTable, type LogitLensTokenizer } from './LogitLensTable';
import './AttentionTab.css';

export interface LogitLensTabProps {
  trace: Trace;
  selectedStep: number | null;
  tokenizer?: LogitLensTokenizer;
}

export function LogitLensTab({
  trace,
  selectedStep,
  tokenizer,
}: LogitLensTabProps) {
  const hasLogitLens = trace.steps.some(
    (s) => Array.isArray(s.logit_lens) && s.logit_lens.length > 0,
  );

  if (!hasLogitLens) {
    return (
      <div
        className="attention-tab attention-tab--empty"
        data-testid="logit-lens-tab-empty"
        role="region"
        aria-label="Logit lens tab empty state"
      >
        <p>
          This trace was generated without <code>--capture-logit-lens</code>.
          Re-run the CLI with that flag to inspect per-layer predictions.
        </p>
      </div>
    );
  }

  return (
    <div
      className="attention-tab"
      data-testid="logit-lens-tab-content"
      role="region"
      aria-label="Logit lens tab"
    >
      <LogitLensTable
        trace={trace}
        selectedStep={selectedStep}
        tokenizer={tokenizer}
      />
    </div>
  );
}
