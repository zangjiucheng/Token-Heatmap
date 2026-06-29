import { useLayoutEffect, useRef } from 'react';
import type { Trace } from '@/types/trace';
import { escapeToken } from './escapeToken';
import './GeneratedTokenStrip.css';

export interface GeneratedTokenStripProps {
  trace: Trace;
  selectedStep: number | null;
  onSelectStep: (step: number) => void;
  hoveredStep?: number | null;
  onHoverStep?: (step: number | null) => void;
}

export function GeneratedTokenStrip({
  trace,
  selectedStep,
  onSelectStep,
  hoveredStep,
  onHoverStep,
}: GeneratedTokenStripProps) {
  const stripRef = useRef<HTMLDivElement | null>(null);

  // Scroll the selected token into view horizontally without disturbing any
  // outer scroll container (manual scrollLeft adjustment rather than
  // scrollIntoView, which can scroll ancestors).
  useLayoutEffect(() => {
    const strip = stripRef.current;
    if (!strip || selectedStep == null) return;
    const btn = strip.querySelector<HTMLElement>(`[data-step="${selectedStep}"]`);
    if (!btn) return;
    const stripRect = strip.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    const margin = 12;
    if (btnRect.left < stripRect.left + margin) {
      strip.scrollLeft -= stripRect.left + margin - btnRect.left;
    } else if (btnRect.right > stripRect.right - margin) {
      strip.scrollLeft += btnRect.right - (stripRect.right - margin);
    }
  }, [selectedStep]);

  if (trace.steps.length === 0) {
    return null;
  }

  return (
    <div
      ref={stripRef}
      className="generated-token-strip"
      role="toolbar"
      aria-label="Generated tokens; click a token to jump to its step"
      data-testid="generated-token-strip"
    >
      {trace.steps.map((step) => {
        const isSelected = selectedStep === step.step;
        const isHovered = hoveredStep === step.step;
        const display = escapeToken(step.selected.token);
        const className = [
          'generated-token-strip__token',
          isSelected ? 'generated-token-strip__token--selected' : '',
          isHovered && !isSelected ? 'generated-token-strip__token--hovered' : '',
        ]
          .filter(Boolean)
          .join(' ');
        return (
          <button
            type="button"
            key={step.step}
            data-step={step.step}
            data-testid={`generated-token-${step.step}`}
            className={className}
            onClick={() => onSelectStep(step.step)}
            onMouseEnter={onHoverStep ? () => onHoverStep(step.step) : undefined}
            onMouseLeave={onHoverStep ? () => onHoverStep(null) : undefined}
            aria-pressed={isSelected}
            title={`Step ${step.step}: ${display}`}
          >
            {display}
          </button>
        );
      })}
    </div>
  );
}
