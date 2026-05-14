import { expect, test } from '@playwright/test';
import {
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

/**
 * The CSV exporter ships a fixed column order (see src/features/export/
 * traceToCsv.ts). Keep this list in sync — a divergence here means the
 * exporter contract has changed and downstream consumers must be notified.
 */
const EXPECTED_CSV_HEADER = [
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
].join(',');

test.describe('export', () => {
  test('CSV download produces a file with the expected columns', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByTestId('export-csv').click(),
    ]);

    expect(download.suggestedFilename()).toMatch(/\.csv$/);

    const stream = await download.createReadStream();
    expect(stream).not.toBeNull();
    if (!stream) return;

    const chunks: Buffer[] = [];
    for await (const chunk of stream) {
      chunks.push(Buffer.from(chunk));
    }
    const text = Buffer.concat(chunks).toString('utf-8');
    const firstLine = text.split('\n')[0];
    expect(firstLine).toBe(EXPECTED_CSV_HEADER);

    // Every data row should declare its source as raw or processed.
    const dataRows = text.split('\n').slice(1).filter((row) => row.length > 0);
    expect(dataRows.length).toBeGreaterThan(0);
    for (const row of dataRows) {
      const source = row.split(',').pop();
      expect(source === 'raw' || source === 'processed').toBe(true);
    }
  });
});
