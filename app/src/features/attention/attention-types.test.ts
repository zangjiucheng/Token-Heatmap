import { describe, expect, it } from 'vitest';
import type { AttentionLayerEntry } from '@/types/trace';
import {
  ATTENTION_METRICS,
  derivePerHeadScalars,
  type AttentionLayerEntryWithPerHead,
} from './attention-types';

const base: AttentionLayerEntry = {
  layer: 0,
  entropy: 1.5,
  self_weight: 0.3,
  bos_weight: 0.2,
  top_positions: [{ position: 0, weight: 0.7 }],
};

describe('derivePerHeadScalars', () => {
  it('reads COLUMNAR per_head by head index', () => {
    const entry: AttentionLayerEntryWithPerHead = {
      ...base,
      per_head: {
        bos_weight: [0.9, 0.1, 0.5],
        self_weight: [0.0, 0.8, 0.2],
        induction: [0.0, 0.0, 0.6],
        top1_weight: [0.95, 0.3, 0.5],
        entropy: [0.4, 1.9, 1.0],
      },
    };
    const heads = derivePerHeadScalars(entry, 3);
    expect(heads).toHaveLength(3);
    expect(heads[0].bos_weight).toBe(0.9); // a sink head
    expect(heads[2].induction).toBe(0.6); // an induction head
    expect(heads[1].self_weight).toBe(0.8);
  });

  it('broadcasts the layer mean when per_head is absent (graceful fallback)', () => {
    const heads = derivePerHeadScalars(base, 4);
    expect(heads).toHaveLength(4);
    expect(heads.every((h) => h.bos_weight === 0.2)).toBe(true);
    expect(heads.every((h) => h.induction === 0)).toBe(true);
    // top1_weight falls back to the layer's top source-position weight.
    expect(heads[0].top1_weight).toBe(0.7);
  });

  it('no longer exposes q/k/v norm metrics', () => {
    expect(ATTENTION_METRICS).not.toContain('q_norm');
    expect(ATTENTION_METRICS).not.toContain('k_norm');
    expect(ATTENTION_METRICS).not.toContain('v_norm');
    // bos_weight (sink signature) leads the functional-first ordering.
    expect(ATTENTION_METRICS[0]).toBe('bos_weight');
  });
});
