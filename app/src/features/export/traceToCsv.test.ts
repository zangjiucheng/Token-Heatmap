import { describe, expect, it } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { CSV_COLUMNS, traceToCsv } from './traceToCsv';

const trace = sampleTrace as unknown as Trace;

function expectedRowCount(t: Trace): number {
  let n = 0;
  for (const s of t.steps) n += s.raw.k_used + s.processed.k_used;
  return n;
}

describe('traceToCsv', () => {
  it('emits the documented column order from README.md plus source', () => {
    const csv = traceToCsv(trace);
    const header = csv.split('\n', 1)[0];
    expect(header).toBe(CSV_COLUMNS.join(','));
    // README order, in literal form, so a future README rename surfaces here.
    expect(header.split(',')).toEqual([
      'step',
      'rank',
      'token_id',
      'token',
      'prob',
      'logprob',
      'selected_token_id',
      'selected_token',
      'selected_prob',
      'selected_logprob',
      'selected_rank',
      'entropy',
      'k_used',
      'source',
    ]);
  });

  it('produces one row per (step, source, candidate)', () => {
    const csv = traceToCsv(trace);
    const lines = csv.trimEnd().split('\n');
    expect(lines.length - 1).toBe(expectedRowCount(trace));
  });

  it('escapes tokens containing commas, quotes, and newlines', () => {
    const fixture: Trace = JSON.parse(JSON.stringify(trace));
    fixture.steps[0].processed.candidates[0].token = 'a,b';
    fixture.steps[0].processed.candidates[1].token = 'q"q';
    if (fixture.steps[0].processed.candidates.length > 2) {
      fixture.steps[0].processed.candidates[2].token = 'line\nbreak';
    }
    const csv = traceToCsv(fixture);
    expect(csv).toContain('"a,b"');
    expect(csv).toContain('"q""q"');
    expect(csv).toContain('"line\nbreak"');
  });

  it('emits both raw and processed sources in order', () => {
    const csv = traceToCsv(trace);
    const lines = csv.trimEnd().split('\n').slice(1);
    const firstStepRawRows = trace.steps[0].raw.k_used;
    const firstStepProcessedRows = trace.steps[0].processed.k_used;
    for (let i = 0; i < firstStepRawRows; i += 1) {
      expect(lines[i].endsWith(',raw')).toBe(true);
    }
    for (
      let i = firstStepRawRows;
      i < firstStepRawRows + firstStepProcessedRows;
      i += 1
    ) {
      expect(lines[i].endsWith(',processed')).toBe(true);
    }
  });

  it('writes numeric step indices and 1-indexed ranks', () => {
    const csv = traceToCsv(trace);
    const firstRow = csv.split('\n')[1].split(',');
    expect(firstRow[0]).toBe('0'); // step
    expect(firstRow[1]).toBe('1'); // rank
  });
});
