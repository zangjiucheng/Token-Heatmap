import { expect, test } from '@playwright/test';
import {
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

test.describe('comparison view', () => {
  test.beforeEach(async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
  });

  test('split mode shows two heatmaps and syncs hover between them', async ({ page }) => {
    await page.getByTestId('comparison-toggle-split').check();

    const split = page.getByTestId('split-heatmap');
    await expect(split).toBeVisible();
    await expect(page.getByTestId('split-heatmap-raw')).toBeVisible();
    await expect(page.getByTestId('split-heatmap-processed')).toBeVisible();

    const rawPlot = page
      .getByTestId('split-heatmap-raw')
      .getByTestId('token-heatmap-plot');
    const procPlot = page
      .getByTestId('split-heatmap-processed')
      .getByTestId('token-heatmap-plot');

    const box = await rawPlot.boundingBox();
    expect(box).not.toBeNull();
    if (!box) return;

    // Hover the centre of the raw pane and assert the processed pane picks
    // up the same step in its external-hover overlay.
    await rawPlot.hover({
      position: { x: Math.round(box.width / 2), y: Math.round(box.height / 2) },
    });

    await expect
      .poll(() => procPlot.getAttribute('data-external-hovered-step'))
      .toMatch(/^\d+$/);
  });

  test('switching to split mode keeps the page root from scrolling', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 720 });

    await page.getByTestId('comparison-toggle-split').check();

    await expect(page.getByTestId('split-heatmap-raw')).toBeVisible();
    await expect(page.getByTestId('split-heatmap-processed')).toBeVisible();

    // The center pane handles its own overflow via .three-pane__center;
    // the html/body root should never scroll, even if internal content
    // is taller than the viewport.
    const rootOverflows = await page.evaluate(() => {
      const html = document.documentElement;
      const body = document.body;
      return (
        html.scrollHeight > html.clientHeight ||
        body.scrollHeight > body.clientHeight
      );
    });
    expect(rootOverflows).toBe(false);
  });

  test('switching value column updates colors without remounting the heatmap', async ({
    page,
  }) => {
    const plot = page.getByTestId('token-heatmap-plot');

    // Tag the canvas with a sentinel that survives only if React reuses the
    // same DOM node. Re-querying a fresh element handle after the change
    // and checking the sentinel proves no remount happened.
    const canvas = plot.getByTestId('token-heatmap-canvas');
    await canvas.evaluate((el) => {
      el.setAttribute('data-e2e-sentinel', 'before-toggle');
    });

    await page.getByTestId('value-column-select').selectOption('prob');

    // Same canvas node should still be there with our sentinel.
    await expect(canvas).toHaveAttribute('data-e2e-sentinel', 'before-toggle');

    // And the URL should reflect the new value column.
    await expect(page).toHaveURL(/[?&]valueCol=prob/);
  });
});
