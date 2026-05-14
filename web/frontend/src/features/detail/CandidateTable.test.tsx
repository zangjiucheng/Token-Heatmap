import { render, screen, fireEvent, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { Candidate } from '@/types/trace';
import { CandidateTable } from './CandidateTable';

function makeCandidates(): Candidate[] {
  return [
    { rank: 1, token_id: 11, token: ',', prob: 0.62, logprob: -0.478 },
    { rank: 2, token_id: 13, token: '.', prob: 0.22, logprob: -1.5141 },
    { rank: 3, token_id: 220, token: ' ', prob: 0.11, logprob: -2.2073 },
  ];
}

function ranksInRenderOrder(): number[] {
  const table = screen.getByTestId('candidate-table');
  const rows = within(table).getAllByRole('row').slice(1); // skip header
  return rows.map((row) => Number(within(row).getAllByRole('cell')[0].textContent));
}

describe('CandidateTable', () => {
  it('sorts by rank ASC by default', () => {
    render(
      <CandidateTable candidates={makeCandidates()} selectedTokenId={11} />,
    );
    expect(ranksInRenderOrder()).toEqual([1, 2, 3]);
  });

  it('toggles to DESC then ASC when clicking the Prob header', () => {
    render(
      <CandidateTable candidates={makeCandidates()} selectedTokenId={11} />,
    );
    const header = within(
      screen.getByTestId('candidate-table-header-prob'),
    ).getByRole('button');

    // First click on prob defaults to DESC (highest prob first → rank 1).
    fireEvent.click(header);
    expect(ranksInRenderOrder()).toEqual([1, 2, 3]);
    expect(
      screen.getByTestId('candidate-table-header-prob'),
    ).toHaveAttribute('aria-sort', 'descending');

    // Second click toggles to ASC (lowest prob first → rank 3).
    fireEvent.click(header);
    expect(ranksInRenderOrder()).toEqual([3, 2, 1]);
    expect(
      screen.getByTestId('candidate-table-header-prob'),
    ).toHaveAttribute('aria-sort', 'ascending');
  });

  it('toggles rank ASC ↔ DESC on repeated clicks', () => {
    render(
      <CandidateTable candidates={makeCandidates()} selectedTokenId={11} />,
    );
    const header = within(
      screen.getByTestId('candidate-table-header-rank'),
    ).getByRole('button');
    fireEvent.click(header);
    expect(ranksInRenderOrder()).toEqual([3, 2, 1]);
    fireEvent.click(header);
    expect(ranksInRenderOrder()).toEqual([1, 2, 3]);
  });

  it('marks the selected token row with a badge and selected attribute', () => {
    render(
      <CandidateTable candidates={makeCandidates()} selectedTokenId={13} />,
    );
    expect(screen.getByTestId('candidate-row-2')).toHaveAttribute(
      'data-selected',
      'true',
    );
    expect(screen.getByTestId('candidate-row-2-badge')).toBeInTheDocument();
    expect(screen.getByTestId('candidate-row-1')).toHaveAttribute(
      'data-selected',
      'false',
    );
    expect(screen.queryByTestId('candidate-row-1-badge')).toBeNull();
  });
});
