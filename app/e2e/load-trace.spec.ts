import { expect, test } from '@playwright/test';
import {
  dropTraceFile,
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

test.describe('load trace', () => {
  test('loads bundled sample → viewer renders all panels', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    // Detail panel is in its empty/waiting state until a step is selected.
    await expect(page.getByTestId('step-detail-panel')).toBeVisible();
  });

  test('dropping a local JSON file renders the viewer', async ({ page }) => {
    await gotoLanding(page);
    await dropTraceFile(page, 'sample-trace.json');

    // The data-layer routes valid file drops to the viewer at /trace/local
    // (id is implementation-defined; we just assert we ended up on a viewer
    // route with a heatmap on screen).
    await expect(page).toHaveURL(/\/trace(\/|$)/);
    await waitForViewerReady(page);
  });

  test('invalid trace → error state with Retry', async ({ page }) => {
    await gotoLanding(page);
    await dropTraceFile(page, 'invalid-trace.json');

    const error = page.getByRole('alert');
    await expect(error).toBeVisible();
    await expect(error.getByRole('heading', { name: /something went wrong/i })).toBeVisible();
    await expect(error.getByRole('button', { name: /retry/i })).toBeVisible();
  });
});
