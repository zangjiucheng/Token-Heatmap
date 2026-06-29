import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearAttentionSidecarCache,
  loadAttentionSidecar,
} from './loadAttentionSidecar';
import type { AttentionSidecar } from './attention-types';

const SIDECAR: AttentionSidecar = {
  num_layers: 1,
  num_heads: 1,
  layers: [
    { layer: 0, attention_distribution: [[1]] },
  ],
};

beforeEach(() => {
  clearAttentionSidecarCache();
});

describe('loadAttentionSidecar', () => {
  it('caches by (step, ref) so a second call does not hit the network', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => SIDECAR,
    });

    const first = await loadAttentionSidecar(0, 'a.json', fetchImpl as unknown as typeof fetch);
    const second = await loadAttentionSidecar(0, 'a.json', fetchImpl as unknown as typeof fetch);

    expect(first).toEqual(SIDECAR);
    expect(second).toEqual(SIDECAR);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('propagates network failures and evicts the cache so retries can succeed', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => SIDECAR });

    await expect(
      loadAttentionSidecar(0, 'b.json', fetchImpl as unknown as typeof fetch),
    ).rejects.toThrow(/HTTP 503/);

    // Cache evicted on failure, so the second call hits fetch again.
    const second = await loadAttentionSidecar(
      0,
      'b.json',
      fetchImpl as unknown as typeof fetch,
    );
    expect(second).toEqual(SIDECAR);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
