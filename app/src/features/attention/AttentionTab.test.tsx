import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AttentionTab } from './AttentionTab';
import { makeTraceWithoutAttention, makeAttentionTrace } from './testFixtures';

describe('AttentionTab', () => {
  it('renders the empty state when the trace has no attention_metadata', () => {
    render(
      <AttentionTab
        trace={makeTraceWithoutAttention()}
        selectedStep={0}
        selectedHead={null}
        onSelectHead={() => {}}
      />,
    );
    expect(screen.getByTestId('attention-tab-empty')).toBeInTheDocument();
    expect(
      screen.queryByTestId('attention-layer-head-grid'),
    ).not.toBeInTheDocument();
  });

  it('renders the attention grid when attention_metadata is present', () => {
    render(
      <AttentionTab
        trace={makeAttentionTrace()}
        selectedStep={0}
        selectedHead={null}
        onSelectHead={() => {}}
      />,
    );
    expect(screen.getByTestId('attention-layer-head-grid')).toBeInTheDocument();
    // LogitLensTable lives in the dedicated Logit Lens tab now.
    expect(
      screen.queryByTestId('logit-lens-table'),
    ).not.toBeInTheDocument();
    // AttentionHeadPattern lives in the right pane (TraceViewerPage).
    expect(
      screen.queryByTestId('attention-head-pattern'),
    ).not.toBeInTheDocument();
  });

  it('renders a single centered prompt when no generation step is selected', () => {
    render(
      <AttentionTab
        trace={makeAttentionTrace()}
        selectedStep={null}
        selectedHead={null}
        onSelectHead={() => {}}
      />,
    );

    expect(screen.getByTestId('attention-tab-step-empty')).toHaveTextContent(
      /select a generation step/i,
    );
    expect(
      screen.queryByTestId('attention-layer-head-grid'),
    ).not.toBeInTheDocument();
  });
});
