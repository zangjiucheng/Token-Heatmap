import { expect, test } from '@playwright/test';
import {
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

test.describe('view state persistence', () => {
  test('non-default view state survives a full reload via the URL', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    // Set a non-default state: split comparison + prob value column.
    await page.getByTestId('comparison-toggle-split').check();
    await page.getByTestId('value-column-select').selectOption('prob');

    await expect(page).toHaveURL(/[?&]mode=split/);
    await expect(page).toHaveURL(/[?&]valueCol=prob/);

    const urlBefore = page.url();

    await page.reload();
    await waitForViewerReady(page);

    // URL is preserved verbatim and controls re-hydrate from it.
    expect(page.url()).toBe(urlBefore);
    await expect(page.getByTestId('comparison-toggle-split')).toBeChecked();
    await expect(page.getByTestId('value-column-select')).toHaveValue('prob');
    await expect(page.getByTestId('split-heatmap')).toBeVisible();
  });

  test('inspector collapse persists across a reload via the URL', async ({
    page,
  }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    // Collapse the inspector — the URL records right=0.
    await page.getByTestId('inspector-collapse').click();
    await expect(page).toHaveURL(/[?&]right=0/);
    await expect(page.getByTestId('inspector-expand')).toBeVisible();

    const urlBefore = page.url();

    await page.reload();
    // The inspector is collapsed, so wait on the plot rather than the full
    // viewer-ready helper (which expects the step-detail panel to be present).
    await expect(page.getByTestId('token-heatmap-plot')).toBeVisible();

    // URL is preserved verbatim and the inspector stays collapsed.
    expect(page.url()).toBe(urlBefore);
    await expect(page.getByTestId('inspector-expand')).toBeVisible();
  });
});
