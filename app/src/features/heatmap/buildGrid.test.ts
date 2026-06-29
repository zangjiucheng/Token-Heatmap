import { describe, expect, it } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { buildGrid, normalize } from './buildGrid';

const trace = sampleTrace as unknown as Trace;

describe('buildGrid', () => {
  it('produces a matrix with correct dimensions for the sample trace', () => {
    const grid = buildGrid(trace, 'logprob');
    expect(grid.steps).toBe(trace.steps.length);
    expect(grid.ranks).toBeGreaterThan(0);
    expect(grid.values.length).toBe(grid.ranks * grid.steps);
    expect(grid.tokens.length).toBe(grid.ranks * grid.steps);
  });

  it('writes NaN for cells outside the per-step k_used', () => {
    const grid = buildGrid(trace, 'logprob');
    for (let step = 0; step < grid.steps; step += 1) {
      const k = grid.kUsed[step];
      for (let rank = k; rank < grid.ranks; rank += 1) {
        const idx = rank * grid.steps + step;
        expect(Number.isNaN(grid.values[idx])).toBe(true);
        expect(grid.tokens[idx]).toBe('');
      }
    }
  });

  it('writes the candidate token text for valid cells', () => {
    const grid = buildGrid(trace, 'logprob');
    for (let step = 0; step < grid.steps; step += 1) {
      const candidates = trace.steps[step].processed.candidates;
      for (const cand of candidates) {
        const idx = (cand.rank - 1) * grid.steps + step;
        expect(grid.tokens[idx]).toBe(cand.token);
        expect(grid.logprobs[idx]).toBeCloseTo(cand.logprob, 4);
        expect(grid.probs[idx]).toBeCloseTo(cand.prob, 4);
      }
    }
  });

  it('uses logprob values when valueCol is logprob', () => {
    const grid = buildGrid(trace, 'logprob');
    const first = trace.steps[0].processed.candidates[0];
    expect(grid.values[0]).toBeCloseTo(first.logprob, 4);
  });

  it('uses prob values when valueCol is prob', () => {
    const grid = buildGrid(trace, 'prob');
    const first = trace.steps[0].processed.candidates[0];
    expect(grid.values[0]).toBeCloseTo(first.prob, 4);
    expect(grid.valueMin).toBeGreaterThanOrEqual(0);
    expect(grid.valueMax).toBeLessThanOrEqual(1);
  });

  it('computes finite min and max over valid cells', () => {
    const grid = buildGrid(trace, 'logprob');
    expect(Number.isFinite(grid.valueMin)).toBe(true);
    expect(Number.isFinite(grid.valueMax)).toBe(true);
    expect(grid.valueMax).toBeGreaterThanOrEqual(grid.valueMin);
  });
});

describe('buildGrid sources', () => {
  it('reads from the raw distribution when source="raw"', () => {
    const grid = buildGrid(trace, 'logprob', 'raw');
    const first = trace.steps[0].raw.candidates[0];
    expect(grid.values[0]).toBeCloseTo(first.logprob, 4);
    expect(grid.tokens[0]).toBe(first.token);
  });

  it('reads from the processed distribution when source="processed"', () => {
    const grid = buildGrid(trace, 'logprob', 'processed');
    const first = trace.steps[0].processed.candidates[0];
    expect(grid.values[0]).toBeCloseTo(first.logprob, 4);
  });

  it('defaults to the processed distribution when source is omitted', () => {
    const a = buildGrid(trace, 'logprob');
    const b = buildGrid(trace, 'logprob', 'processed');
    expect(a.valueMin).toBeCloseTo(b.valueMin, 6);
    expect(a.valueMax).toBeCloseTo(b.valueMax, 6);
  });
});

describe('normalize', () => {
  it('maps min to 0 and max to 1', () => {
    expect(normalize(0, 0, 10)).toBe(0);
    expect(normalize(10, 0, 10)).toBe(1);
  });

  it('returns NaN for NaN input', () => {
    expect(Number.isNaN(normalize(NaN, 0, 1))).toBe(true);
  });

  it('returns 0.5 when min equals max', () => {
    expect(normalize(3, 3, 3)).toBe(0.5);
  });
});
