import { useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import './OutputTab.css';

export interface OutputTabProps {
  trace: Trace;
}

type OutputView = 'completion' | 'full';

/**
 * Output tab — a complete, readable render of the generation, whitespace
 * preserved (so multi-line tasks like counting / hard-wrapped text show their
 * real structure, unlike the single-line token strip).
 *
 * - "Completion" reconstructs exactly what the model produced by concatenating
 *   each step's selected token text.
 * - "Full" shows `metadata.generated_text` verbatim (prompt + completion,
 *   including any chat-template wrapping).
 */
export function OutputTab({ trace }: OutputTabProps) {
  const [view, setView] = useState<OutputView>('completion');
  const [copied, setCopied] = useState(false);

  const completion = useMemo(
    () => trace.steps.map((s) => s.selected?.token ?? '').join(''),
    [trace.steps],
  );
  const fullText = trace.metadata?.generated_text ?? '';
  const text = view === 'completion' ? completion : fullText;

  const tokenCount = trace.steps.length;
  const charCount = text.length;
  const lineCount = text ? text.split('\n').length : 0;

  const handleCopy = () => {
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    };
    try {
      void navigator.clipboard?.writeText(text).then(done, () => {});
    } catch {
      /* clipboard unavailable (e.g. insecure context) — no-op */
    }
  };

  return (
    <div className="output-tab" data-testid="output-tab">
      <div className="output-tab__controls">
        <div
          className="output-tab__toggle"
          role="tablist"
          aria-label="Output view"
        >
          <button
            type="button"
            role="tab"
            aria-selected={view === 'completion'}
            className={
              view === 'completion'
                ? 'output-tab__toggle-btn output-tab__toggle-btn--active'
                : 'output-tab__toggle-btn'
            }
            onClick={() => setView('completion')}
            data-testid="output-view-completion"
          >
            Completion
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={view === 'full'}
            className={
              view === 'full'
                ? 'output-tab__toggle-btn output-tab__toggle-btn--active'
                : 'output-tab__toggle-btn'
            }
            onClick={() => setView('full')}
            data-testid="output-view-full"
            disabled={!fullText}
            title={
              fullText
                ? 'Full decoded text including the prompt'
                : 'No generated_text on this trace'
            }
          >
            Full (with prompt)
          </button>
        </div>
        <div className="output-tab__meta" aria-label="Output stats">
          <span>{tokenCount} tokens</span>
          <span>·</span>
          <span>{lineCount} lines</span>
          <span>·</span>
          <span>{charCount} chars</span>
          <button
            type="button"
            className="output-tab__copy"
            onClick={handleCopy}
            data-testid="output-copy"
          >
            {copied ? 'Copied ✓' : 'Copy'}
          </button>
        </div>
      </div>
      {text ? (
        <pre className="output-tab__text" data-testid="output-text">
          {text}
        </pre>
      ) : (
        <p className="output-tab__empty" data-testid="output-empty">
          This trace has no generated text.
        </p>
      )}
    </div>
  );
}
