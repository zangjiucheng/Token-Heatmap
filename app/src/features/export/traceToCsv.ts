import type { Trace, Distribution } from '@/types/trace';

/**
 * Column order matches the table in README.md (Main columns) with the `source`
 * column appended for raw/processed comparison. One row per
 * (step, source, candidate) tuple — each step contributes
 * `raw.k_used + processed.k_used` rows.
 */
export const CSV_COLUMNS = [
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
] as const;

export type CsvColumn = (typeof CSV_COLUMNS)[number];

const NEEDS_QUOTING = /[",\n\r]/;

function escapeCell(value: string | number): string {
  const text = typeof value === 'number' ? String(value) : value;
  if (NEEDS_QUOTING.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function emitRows(
  step: number,
  dist: Distribution,
  source: 'raw' | 'processed',
  selectedTokenId: number,
  selectedToken: string,
  rows: string[],
): void {
  for (const cand of dist.candidates) {
    const cells: Array<string | number> = [
      step,
      cand.rank,
      cand.token_id,
      cand.token,
      cand.prob,
      cand.logprob,
      selectedTokenId,
      selectedToken,
      dist.selected_prob,
      dist.selected_logprob,
      dist.selected_rank,
      dist.entropy,
      dist.k_used,
      source,
    ];
    rows.push(cells.map(escapeCell).join(','));
  }
}

/**
 * Serialize a `Trace` to a CSV string. Pure: no DOM access, no side effects.
 */
export function traceToCsv(trace: Trace): string {
  const lines: string[] = [];
  lines.push(CSV_COLUMNS.join(','));
  for (const step of trace.steps) {
    emitRows(
      step.step,
      step.raw,
      'raw',
      step.selected.token_id,
      step.selected.token,
      lines,
    );
    emitRows(
      step.step,
      step.processed,
      'processed',
      step.selected.token_id,
      step.selected.token,
      lines,
    );
  }
  return `${lines.join('\n')}\n`;
}
