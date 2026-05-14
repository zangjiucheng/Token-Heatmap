import { useMemo, useState } from 'react';
import type { Candidate } from '@/types/trace';
import { escapeToken } from './escapeToken';
import './CandidateTable.css';

export type SortKey = 'rank' | 'prob' | 'logprob';
export type SortDir = 'asc' | 'desc';

export interface CandidateTableProps {
  candidates: Candidate[];
  /** token_id of the selected candidate; used to mark the selected row. */
  selectedTokenId: number | null;
}

const COLUMNS: ReadonlyArray<{
  key: SortKey;
  label: string;
  align?: 'left' | 'right';
}> = [
  { key: 'rank', label: 'Rank', align: 'right' },
  { key: 'prob', label: 'Prob', align: 'right' },
  { key: 'logprob', label: 'Logprob', align: 'right' },
];

function compareCandidates(
  a: Candidate,
  b: Candidate,
  key: SortKey,
  dir: SortDir,
): number {
  const av = a[key];
  const bv = b[key];
  const sign = dir === 'asc' ? 1 : -1;
  if (av < bv) return -1 * sign;
  if (av > bv) return 1 * sign;
  return 0;
}

function formatProb(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (v >= 0.001 || v === 0) return v.toFixed(4);
  return v.toExponential(2);
}

function formatLogprob(v: number): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(3);
}

export function CandidateTable({
  candidates,
  selectedTokenId,
}: CandidateTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('rank');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const sorted = useMemo(() => {
    const copy = candidates.slice();
    copy.sort((a, b) => compareCandidates(a, b, sortKey, sortDir));
    return copy;
  }, [candidates, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      // Rank defaults ASC (1 first), prob/logprob default DESC (high first).
      setSortDir(key === 'rank' ? 'asc' : 'desc');
    }
  };

  return (
    <table
      className="candidate-table"
      data-testid="candidate-table"
      aria-label="Top candidate tokens"
    >
      <thead>
        <tr>
          {COLUMNS.map((col) => {
            const isActive = sortKey === col.key;
            const ariaSort = isActive
              ? sortDir === 'asc'
                ? 'ascending'
                : 'descending'
              : 'none';
            return (
              <th
                key={col.key}
                scope="col"
                aria-sort={ariaSort}
                style={col.align === 'right' ? { textAlign: 'right' } : undefined}
                data-testid={`candidate-table-header-${col.key}`}
              >
                <button
                  type="button"
                  className="is-sortable"
                  onClick={() => handleSort(col.key)}
                  style={{
                    background: 'none',
                    border: 0,
                    padding: 0,
                    font: 'inherit',
                    color: 'inherit',
                    cursor: 'pointer',
                  }}
                >
                  {col.label}
                  {isActive && (
                    <span
                      className="candidate-table__sort-indicator"
                      aria-hidden="true"
                    >
                      {sortDir === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </button>
              </th>
            );
          })}
          <th scope="col" style={{ textAlign: 'left' }}>
            Token
          </th>
          <th scope="col" style={{ textAlign: 'left' }}>
            Selected
          </th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((cand) => {
          const isSelected = cand.token_id === selectedTokenId;
          return (
            <tr
              key={`${cand.rank}-${cand.token_id}`}
              data-testid={`candidate-row-${cand.rank}`}
              data-selected={isSelected ? 'true' : 'false'}
              className={
                isSelected ? 'candidate-table__row--selected' : undefined
              }
            >
              <td style={{ textAlign: 'right' }}>{cand.rank}</td>
              <td style={{ textAlign: 'right' }}>{formatProb(cand.prob)}</td>
              <td style={{ textAlign: 'right' }}>
                {formatLogprob(cand.logprob)}
              </td>
              <td>
                <span
                  className="candidate-table__token"
                  title={cand.token}
                  data-testid={`candidate-row-${cand.rank}-token`}
                >
                  {escapeToken(cand.token)}
                </span>
              </td>
              <td>
                {isSelected && (
                  <span
                    className="candidate-table__badge"
                    aria-label="selected token"
                    data-testid={`candidate-row-${cand.rank}-badge`}
                  >
                    ✓
                  </span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
