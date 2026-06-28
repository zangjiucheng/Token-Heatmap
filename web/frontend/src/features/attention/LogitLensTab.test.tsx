import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { Trace } from '@/types/trace';
import { LogitLensTab } from './LogitLensTab';
import { makeAttentionTrace } from './testFixtures';

function makeTraceWithoutLogitLens() {
  const base = makeAttentionTrace();
  return {
    ...base,
    steps: base.steps.map((s) => ({ ...s, logit_lens: undefined })),
  };
}

function makeTraceWithPromptLens(): Trace {
  return {
    ...makeAttentionTrace(),
    prompt_logit_lens: {
      top_k: 3,
      num_layers: 4,
      positions: [
        {
          position: 0,
          token_id: 1,
          token: ' Dallas',
          layers: [
            { layer_idx: 0, top_k: [{ rank: 1, token_id: 5, token: ' the', prob: 0.2 }] },
            { layer_idx: 2, top_k: [{ rank: 1, token_id: 9, token: ' Texas', prob: 0.6 }] },
          ],
        },
        {
          position: 1,
          token_id: 2,
          token: ' is',
          layers: [
            { layer_idx: 2, top_k: [{ rank: 1, token_id: 7, token: ' Austin', prob: 0.7 }] },
          ],
        },
      ],
    },
  } as unknown as Trace;
}

describe('LogitLensTab', () => {
  it('renders the empty state when the trace has no logit_lens data', () => {
    render(
      <LogitLensTab trace={makeTraceWithoutLogitLens()} selectedStep={0} />,
    );
    expect(screen.getByTestId('logit-lens-tab-empty')).toBeInTheDocument();
    expect(
      screen.queryByTestId('logit-lens-tab-content'),
    ).not.toBeInTheDocument();
  });

  it('renders the LogitLensTable when logit_lens data is present', () => {
    render(
      <LogitLensTab trace={makeAttentionTrace()} selectedStep={0} />,
    );
    expect(screen.getByTestId('logit-lens-tab-content')).toBeInTheDocument();
    expect(screen.getByTestId('logit-lens-table')).toBeInTheDocument();
  });

  it('passes selectedStep=null prompt down to LogitLensTable', () => {
    render(
      <LogitLensTab trace={makeAttentionTrace()} selectedStep={null} />,
    );
    expect(screen.getByTestId('logit-lens-tab-content')).toBeInTheDocument();
    expect(screen.getByTestId('logit-lens-table')).toHaveTextContent(
      /select a generation step/i,
    );
  });
});
