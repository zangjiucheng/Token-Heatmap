import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, beforeEach, vi } from 'vitest';
import { AttentionHeadPattern } from './AttentionHeadPattern';
import {
  clearAttentionSidecarCache,
} from './loadAttentionSidecar';
import { makeAttentionTrace } from './testFixtures';
import type { AttentionSidecar } from './attention-types';

beforeEach(() => {
  clearAttentionSidecarCache();
});

describe('AttentionHeadPattern', () => {
  it('renders from the inline top_positions when no sidecar ref is present', () => {
    render(
      <AttentionHeadPattern
        trace={makeAttentionTrace()}
        selectedStep={0}
        selectedHead={{ layer: 0, head: 0 }}
      />,
    );
    expect(screen.getByTestId('attention-head-pattern')).toHaveAttribute(
      'data-source',
      'inline-top-positions',
    );
    // top_positions has 2 entries in the fixture.
    expect(screen.getByTestId('attention-pattern-bar-0')).toBeInTheDocument();
    expect(screen.getByTestId('attention-pattern-bar-1')).toBeInTheDocument();
  });

  it('renders from the sidecar when attention_sidecar_ref is set', async () => {
    const trace = makeAttentionTrace();
    trace.steps[0].attention_sidecar_ref = 'http://example.test/sidecar-0.json';
    const sidecar: AttentionSidecar = {
      num_layers: 4,
      num_heads: 4,
      layers: [
        {
          layer: 0,
          attention_distribution: [
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
            [0.5, 0.2, 0.2, 0.1],
          ],
        },
      ],
    };
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => sidecar,
    });

    render(
      <AttentionHeadPattern
        trace={trace}
        selectedStep={0}
        selectedHead={{ layer: 0, head: 0 }}
        fetchImpl={fetchImpl as unknown as typeof fetch}
      />,
    );

    await waitFor(() =>
      expect(screen.getByTestId('attention-head-pattern')).toHaveAttribute(
        'data-source',
        'sidecar',
      ),
    );
    // Sidecar row [0.1, 0.2, 0.3, 0.4] -> 4 bars at positions 0..3.
    expect(screen.getByTestId('attention-pattern-bar-0')).toBeInTheDocument();
    expect(screen.getByTestId('attention-pattern-bar-3')).toBeInTheDocument();
    expect(fetchImpl).toHaveBeenCalledWith('http://example.test/sidecar-0.json');
  });
});
