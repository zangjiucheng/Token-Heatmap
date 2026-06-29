import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { StepDetailPanel } from './StepDetailPanel';

const trace = sampleTrace as unknown as Trace;

describe('StepDetailPanel', () => {
  it('renders empty state when trace is null', () => {
    render(<StepDetailPanel trace={null} selectedStep={null} />);
    expect(screen.getByTestId('step-detail-panel').className).toMatch(
      /step-detail-panel--empty/,
    );
    expect(screen.getByText(/No trace loaded/i)).toBeInTheDocument();
  });

  it('renders prompt-style empty state when no step is selected', () => {
    render(<StepDetailPanel trace={trace} selectedStep={null} />);
    expect(screen.getByText(/select a generation step/i)).toBeInTheDocument();
    expect(screen.getByTestId('step-detail-panel').className).toMatch(
      /step-detail-panel--empty/,
    );
  });

  it('renders header stats for the selected step', () => {
    render(<StepDetailPanel trace={trace} selectedStep={0} />);
    const step = trace.steps[0];
    expect(screen.getByTestId('step-detail-panel-step')).toHaveTextContent(
      `Step ${step.step}`,
    );
    expect(screen.getByTestId('step-detail-panel-k-used')).toHaveTextContent(
      String(step.processed.k_used),
    );
    expect(
      screen.getByTestId('step-detail-panel-selected-rank'),
    ).toHaveTextContent(String(step.processed.selected_rank));
    expect(
      screen.getByTestId('step-detail-panel-selected-prob'),
    ).toHaveTextContent(step.processed.selected_prob.toFixed(4));
  });

  it('marks the selected token row with a badge', () => {
    render(<StepDetailPanel trace={trace} selectedStep={0} />);
    const step = trace.steps[0];
    const selectedRank = step.processed.candidates.find(
      (c) => c.token_id === step.selected.token_id,
    )?.rank;
    expect(selectedRank).toBeDefined();
    expect(screen.getByTestId(`candidate-row-${selectedRank}`)).toHaveAttribute(
      'data-selected',
      'true',
    );
    expect(
      screen.getByTestId(`candidate-row-${selectedRank}-badge`),
    ).toBeInTheDocument();
  });
});
