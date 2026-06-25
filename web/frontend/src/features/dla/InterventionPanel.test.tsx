import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  setApiClientForTests,
  type ApiClient,
  type InterventionResult,
} from '@/api/client';
import type { Trace, DirectLogitAttributionStep } from '@/types/trace';
import { InterventionPanel } from './InterventionPanel';

const STEP: DirectLogitAttributionStep = {
  step: 1,
  target_token_id: 5,
  total_logit: 3.2,
  embed: 0.5,
  bias: 0,
  error: 0.01,
  layers: [
    { layer: 0, attn: 0.2, mlp: -0.1 },
    { layer: 5, attn: 1.4, mlp: 0.3 },
  ],
};

const TRACE = {
  metadata: { model: 'Qwen/Qwen2.5-0.5B-Instruct', prompt: 'The capital of France is' },
  steps: [
    { step: 0, selected: { token: ' the', token_id: 9 } },
    { step: 1, selected: { token: ' Paris', token_id: 5 } },
  ],
} as unknown as Trace;

const RESULT: InterventionResult = {
  target_token_id: 5,
  target_token: ' Paris',
  baseline: {
    top: [{ token: ' Paris', token_id: 5, prob: 0.6, logit: 3.2 }],
    target_prob: 0.6,
    target_logit: 3.2,
  },
  patched: {
    top: [{ token: ' London', token_id: 8, prob: 0.3, logit: 1.1 }],
    target_prob: 0.04,
    target_logit: 0.2,
  },
  diff: {
    kl: 2.3,
    target_prob_delta: -0.56,
    target_logit_delta: -3.0,
    top_flips: [{ rank: 1, from_token: ' Paris', to_token: ' London' }],
  },
  interventions: [],
};

function stubClient(overrides: Partial<ApiClient> = {}) {
  const intervene = vi.fn().mockResolvedValue(RESULT);
  setApiClientForTests({
    health: vi.fn().mockResolvedValue(true),
    intervene,
    ...overrides,
  } as unknown as ApiClient);
  return { intervene };
}

afterEach(() => setApiClientForTests(null));

describe('InterventionPanel', () => {
  it('runs an ablation and shows the baseline-vs-patched diff', async () => {
    const { intervene } = stubClient();
    const user = userEvent.setup();
    render(<InterventionPanel trace={TRACE} step={STEP} />);

    // Becomes enabled once the health probe resolves healthy.
    const run = await screen.findByTestId('intervention-run');
    await user.click(run);

    await waitFor(() =>
      expect(screen.getByTestId('intervention-result')).toBeInTheDocument(),
    );
    // Called for the top contributor (L5.attn) with the right context.
    expect(intervene).toHaveBeenCalledTimes(1);
    const arg = intervene.mock.calls[0][0];
    expect(arg.interventions[0]).toMatchObject({ layer: 5, component: 'attn', op: 'zero' });
    expect(arg.continuation_token_ids).toEqual([9]);
    expect(arg.target_token_id).toBe(5);
    expect(arg.model).toBe('Qwen/Qwen2.5-0.5B-Instruct');
    // KL + target-prob move are surfaced.
    expect(screen.getByText(/KL 2\.300 nats/)).toBeInTheDocument();
    expect(screen.getByTestId('intervention-target-delta')).toHaveAttribute(
      'data-dir',
      'down',
    );
  });

  it('runs a per-head ablation from a preset', async () => {
    const { intervene } = stubClient();
    const stepWithHeads = {
      ...STEP,
      layers: [
        { layer: 0, attn: 0.2, mlp: -0.1 },
        {
          layer: 5,
          attn: 1.4,
          mlp: 0.3,
          heads: [
            { head: 7, attn: 1.0 },
            { head: 2, attn: 0.4 },
          ],
        },
      ],
    } as typeof STEP;
    const { rerender } = render(
      <InterventionPanel trace={TRACE} step={stepWithHeads} preset={null} />,
    );
    // Wait until the backend probe reports healthy (controls visible).
    await screen.findByTestId('intervention-run');
    // A per-head "ablate" click arrives as a preset → auto-runs that head.
    rerender(
      <InterventionPanel
        trace={TRACE}
        step={stepWithHeads}
        preset={{ layer: 5, component: 'head', head: 7 }}
      />,
    );
    await waitFor(() => expect(intervene).toHaveBeenCalled());
    expect(intervene.mock.calls[0][0].interventions[0]).toMatchObject({
      layer: 5,
      component: 'head',
      head: 7,
      op: 'zero',
    });
  });

  it('shows an offline hint when the backend is unhealthy', async () => {
    stubClient({ health: vi.fn().mockResolvedValue(false) } as Partial<ApiClient>);
    render(<InterventionPanel trace={TRACE} step={STEP} />);
    expect(await screen.findByTestId('intervention-offline')).toBeInTheDocument();
    expect(screen.queryByTestId('intervention-run')).not.toBeInTheDocument();
  });
});
