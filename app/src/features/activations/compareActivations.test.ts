import { describe, expect, it } from 'vitest';
import {
  CompareActivationsError,
  DIFF_SCHEMA_VERSION,
  compareActivations,
} from './compareActivations';
import {
  FIXTURE_DIFF_NUM_LAYERS,
  FIXTURE_DIFF_NUM_STEPS,
  FIXTURE_DIFF_SUBMODULES,
  makeDiffOracle,
  makeTraceWithoutActivations,
  makeTwoActivationTraces,
} from './testFixtures';
import type { TraceWithActivations } from '@/types/activation';

const TOL = 1e-5;

describe('compareActivations', () => {
  it('emits an ActivationDiff conforming to the documented shape', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    const diff = compareActivations(traceA, traceB);
    expect(diff.schema_version).toBe(DIFF_SCHEMA_VERSION);
    expect(diff.alignment.mode).toBe('auto');
    expect(diff.alignment.tokenizer_a_fingerprint).toBe(
      traceA.activation_metadata!.tokenizer_fingerprint,
    );
    expect(diff.alignment.tokenizer_b_fingerprint).toBe(
      traceB.activation_metadata!.tokenizer_fingerprint,
    );
    expect(diff.steps).toHaveLength(FIXTURE_DIFF_NUM_STEPS);
    for (const s of diff.steps) {
      expect(s.delta).toHaveLength(
        FIXTURE_DIFF_NUM_LAYERS * FIXTURE_DIFF_SUBMODULES.length,
      );
    }
  });

  it('agrees with the hand-computed oracle within 1e-5 on every cell', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    const diff = compareActivations(traceA, traceB);
    const oracle = makeDiffOracle();

    for (const expected of oracle.cells) {
      const stepRec = diff.steps.find((s) => s.step === expected.step);
      expect(stepRec, `step ${expected.step}`).toBeDefined();
      const entry = stepRec!.delta.find(
        (d) =>
          d.layer === expected.layer && d.submodule === expected.submodule,
      );
      expect(
        entry,
        `cell (step ${expected.step}, layer ${expected.layer})`,
      ).toBeDefined();
      expect(Math.abs(entry!.l2 - expected.l2)).toBeLessThan(TOL);
      expect(Math.abs(entry!.cosine - expected.cosine)).toBeLessThan(TOL);

      // top_changed_neurons must be sorted by descending |delta| and the
      // absolute-delta sequence must match the oracle's sorted-by-magnitude
      // delta sequence.
      const tcn = entry!.top_changed_neurons;
      for (let i = 1; i < tcn.length; i += 1) {
        expect(Math.abs(tcn[i].delta)).toBeLessThanOrEqual(
          Math.abs(tcn[i - 1].delta) + TOL,
        );
      }
      for (let i = 0; i < tcn.length; i += 1) {
        expect(
          Math.abs(Math.abs(tcn[i].delta) - Math.abs(expected.topChangedDeltas[i])),
        ).toBeLessThan(TOL);
      }
    }
  });

  it('resolves auto to token_id when fingerprints match', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    const diff = compareActivations(traceA, traceB, { align: 'auto' });
    // Same fingerprint and same token_ids → no mismatches and full coverage.
    expect(diff.alignment.mismatches).toEqual([]);
    expect(diff.steps).toHaveLength(traceA.steps.length);
  });

  it('falls back to position alignment when fingerprints differ', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    traceB.activation_metadata!.tokenizer_fingerprint = 'other-tokenizer';
    // Make B's token ids diverge so token_id alignment would fail; position
    // alignment via decoded_text_offset still succeeds.
    for (const s of traceB.steps) {
      s.selected = { ...s.selected, token_id: s.selected.token_id + 1000 };
    }
    const diff = compareActivations(traceA, traceB, { align: 'auto' });
    expect(diff.steps).toHaveLength(traceA.steps.length);
    expect(diff.alignment.mismatches).toEqual([]);
  });

  it('flags divergent token_ids as mismatches under explicit token_id mode', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    traceB.steps[0].selected = {
      ...traceB.steps[0].selected,
      token_id: 9999,
    };
    const diff = compareActivations(traceA, traceB, { align: 'token_id' });
    expect(diff.alignment.mismatches.length).toBeGreaterThanOrEqual(1);
    expect(diff.alignment.mismatches[0].reason).toBe('token_id_divergence');
    expect(diff.steps).toHaveLength(traceA.steps.length - 1);
  });

  it('throws when a trace has no activation_metadata', () => {
    const empty = makeTraceWithoutActivations() as TraceWithActivations;
    const { traceA } = makeTwoActivationTraces();
    expect(() => compareActivations(empty, traceA)).toThrow(
      CompareActivationsError,
    );
    expect(() => compareActivations(traceA, empty)).toThrow(
      CompareActivationsError,
    );
  });

  it('respects topK', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    const diff = compareActivations(traceA, traceB, { topK: 2 });
    for (const step of diff.steps) {
      for (const d of step.delta) {
        expect(d.top_changed_neurons.length).toBeLessThanOrEqual(2);
      }
    }
  });
});
