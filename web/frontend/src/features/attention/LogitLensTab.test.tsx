import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { LogitLensTab } from './LogitLensTab';
import { makeAttentionTrace } from './testFixtures';

function makeTraceWithoutLogitLens() {
  const base = makeAttentionTrace();
  return {
    ...base,
    steps: base.steps.map((s) => ({ ...s, logit_lens: undefined })),
  };
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
