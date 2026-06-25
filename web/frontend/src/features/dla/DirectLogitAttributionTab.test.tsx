import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { setApiClientForTests, type ApiClient } from '@/api/client';
import type { Trace } from '@/types/trace';
import { DirectLogitAttributionTab } from './DirectLogitAttributionTab';

// The lens renders the InterventionPanel, which probes backend health. Stub the
// client so tests don't touch the network (offline hint is shown instead).
beforeEach(() => {
  setApiClientForTests({
    health: vi.fn().mockResolvedValue(false),
  } as unknown as ApiClient);
});
afterEach(() => setApiClientForTests(null));

function makeTrace(withDla = true): Trace {
  const base = {
    schema_version: '2.0.0',
    metadata: { model: 'm', prompt: '', generated_text: '' },
    tokens: { prompt_token_ids: [], prompt_tokens: [] },
    steps: [
      { step: 0, selected: { token: ' the', token_id: 5 }, raw: {}, processed: {} },
    ],
  } as unknown as Trace;
  if (!withDla) return base;
  return {
    ...base,
    direct_logit_attribution: {
      method: 'dla_fold_norm',
      n_steps: 1,
      num_layers: 2,
      steps: [
        {
          step: 0,
          target_token_id: 5,
          total_logit: 3.2,
          embed: 0.5,
          bias: 0,
          error: 0.01,
          layers: [
            { layer: 0, attn: 1.0, mlp: -0.5 },
            { layer: 1, attn: 0.8, mlp: 0.4 },
          ],
        },
      ],
    },
  } as unknown as Trace;
}

describe('DirectLogitAttributionTab', () => {
  it('renders the decomposition for the selected step', () => {
    const { container } = render(
      <DirectLogitAttributionTab trace={makeTrace()} selectedStep={0} />,
    );
    expect(
      screen.getByTestId('direct-logit-attribution-tab-content'),
    ).toBeInTheDocument();
    // The target token's logit is surfaced.
    expect(screen.getByText('3.200')).toBeInTheDocument();
    // embed + (2 layers × attn/mlp) + the unexplained-error bar = 6 bars.
    expect(container.querySelectorAll('.dla-bar')).toHaveLength(6);
  });

  it('sorts components by absolute impact (largest first)', () => {
    const { container } = render(
      <DirectLogitAttributionTab trace={makeTrace()} selectedStep={0} />,
    );
    const labels = Array.from(
      container.querySelectorAll('.dla-bar__label'),
    ).map((el) => el.textContent);
    // |L0.attn|=1.0 is the largest contributor; error (0.01) is the smallest.
    expect(labels[0]).toBe('L0 · attn');
    expect(labels[labels.length - 1]).toBe('unexplained (error)');
  });

  it('shows an empty state when the trace has no attribution', () => {
    render(
      <DirectLogitAttributionTab trace={makeTrace(false)} selectedStep={0} />,
    );
    expect(
      screen.getByTestId('direct-logit-attribution-tab-empty'),
    ).toBeInTheDocument();
  });
});
