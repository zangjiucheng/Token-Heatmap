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

  test('left gutter drag persists pane widths across a reload', async ({
    page,
  }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    const gutter = page.getByTestId('three-pane-gutter-left');
    await gutter.waitFor({ state: 'visible' });

    // Use the keyboard to nudge the gutter so the test is independent of
    // exact pixel coordinates. Five Shift+ArrowRight nudges widen the left
    // pane by 5 * 32 = 160 px (default 280 → 440).
    await gutter.focus();
    for (let i = 0; i < 5; i += 1) {
      await page.keyboard.press('Shift+ArrowRight');
    }

    const widthAfter = await gutter.evaluate(
      (el) => el.getAttribute('aria-valuenow'),
    );
    expect(Number(widthAfter)).toBeGreaterThanOrEqual(440);

    const stored = await page.evaluate(() =>
      window.localStorage.getItem('token-heatmap.layout.paneWidths'),
    );
    expect(stored).not.toBeNull();
    const parsed = JSON.parse(stored ?? '{}') as { left?: number };
    expect(parsed.left).toBeGreaterThanOrEqual(440);

    await page.reload();
    await waitForViewerReady(page);

    const restored = page.getByTestId('three-pane-gutter-left');
    await restored.waitFor({ state: 'visible' });
    expect(
      Number(await restored.evaluate((el) => el.getAttribute('aria-valuenow'))),
    ).toBeGreaterThanOrEqual(440);
  });
});
