import { expect, test } from '@playwright/test';
import {
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

/**
 * Visual regression sweep. Baselines are authoritative on Linux only
 * (committed from CI). Local runs on macOS / Windows write per-platform
 * copies via the snapshotPathTemplate so they don't fight the committed
 * Linux pixels.
 *
 * If a UI change is intentional, regenerate baselines with:
 *   npm run e2e:update-snapshots
 */

test.describe('visual regression', () => {
  test('landing', async ({ page }) => {
    await gotoLanding(page);
    await expect(page).toHaveScreenshot('landing.png', { fullPage: true });
  });

  test('viewer-empty', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
    // Force the detail panel into its "no step selected" state for a stable
    // baseline regardless of any default-selection behaviour the viewer
    // grows in future.
    await page.locator('body').click();
    await page.keyboard.press('Escape');
    await expect(page).toHaveScreenshot('viewer-empty.png', { fullPage: true });
  });

  test('viewer-with-trace', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
    await page.getByTestId('token-heatmap-plot').focus();
    await page.keyboard.press('Home');
    await expect(page.getByTestId('step-detail-panel-step')).toHaveText(/Step 0/);
    await expect(page).toHaveScreenshot('viewer-with-trace.png', { fullPage: true });
  });

  test('split-mode', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
    await page.getByTestId('comparison-toggle-split').check();
    await expect(page.getByTestId('split-heatmap')).toBeVisible();
    await expect(page).toHaveScreenshot('split-mode.png', { fullPage: true });
  });
});
