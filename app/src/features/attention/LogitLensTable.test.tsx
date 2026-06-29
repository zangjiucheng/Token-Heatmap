import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { LogitLensTable } from './LogitLensTable';
import { makeAttentionTrace } from './testFixtures';

describe('LogitLensTable', () => {
  it('marks the deepest captured layer top-1 as the selected token when it matches', () => {
    // Fixture: deepest captured layer (L3) has top_k[0].token_id == selected.token_id (7).
    render(<LogitLensTable trace={makeAttentionTrace()} selectedStep={0} />);
    // L3 is the deepest captured layer in the fixture; its top-1 cell is
    // selected (token_id 7 == selected.token_id).
    const row = screen.getByTestId('logit-lens-row-3');
    expect(row.getAttribute('data-final-row')).toBe('true');
    const top1Cell = within(row).getByTestId('logit-lens-cell-3-0');
    expect(top1Cell.getAttribute('data-token-id')).toBe('7');
    expect(top1Cell.className).toContain('logit-lens-table__cell--selected');
  });

  it('falls back to displaying token IDs when no tokenizer is provided and the candidate has no text', () => {
    const trace = makeAttentionTrace();
    // Strip token text to simulate a tokenizer-less trace.
    trace.steps[0].logit_lens = trace.steps[0].logit_lens!.map((layer) => ({
      ...layer,
      top_k: layer.top_k.map((c) => ({ ...c, token: '' })),
    }));
    render(<LogitLensTable trace={trace} selectedStep={0} />);
    // Token id 9 appears in top_k for layer 0.
    const cell = screen.getByTestId('logit-lens-cell-0-0');
    expect(cell.textContent).toContain('#9');
  });

  it('prompts the user to choose a generation step before rendering predictions', () => {
    render(<LogitLensTable trace={makeAttentionTrace()} selectedStep={null} />);

    expect(screen.getByTestId('logit-lens-table')).toHaveTextContent(
      /select a generation step/i,
    );
    expect(screen.queryByRole('table')).toBeNull();
  });
});
