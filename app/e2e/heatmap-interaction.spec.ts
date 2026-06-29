import { expect, test } from '@playwright/test';
import {
  focusHeatmap,
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

test.describe('heatmap interaction', () => {
  test.beforeEach(async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
  });

  test('clicking a cell selects a step and moves the timeline cursors', async ({ page }) => {
    const plot = page.getByTestId('token-heatmap-plot');
    const box = await plot.boundingBox();
    expect(box, 'heatmap should have a rendered bounding box').not.toBeNull();
    if (!box) return;

    // Click roughly in the middle of the plot, which is well inside the data
    // grid for any non-trivial trace.
    await plot.click({
      position: { x: Math.round(box.width / 2), y: Math.round(box.height / 2) },
    });

    // Detail panel transitions to "Step detail for step N" when a step is
    // selected. Wait for the heading to appear.
    await expect(page.getByTestId('step-detail-panel-step')).toBeVisible();
    const heading = await page.getByTestId('step-detail-panel-step').textContent();
    expect(heading).toMatch(/Step \d+/);

    // Both timelines paint a selected-point marker on the same step.
    await expect(page.getByTestId('entropy-timeline-selected-point')).toBeVisible();
    await expect(
      page.getByTestId('selected-probability-timeline-selected-point'),
    ).toBeVisible();
  });

  test('arrow keys advance the selected step', async ({ page }) => {
    await focusHeatmap(page);

    // Home → step 0, ArrowRight → step 1, ArrowRight → step 2.
    await page.keyboard.press('Home');
    await expect(page.getByTestId('step-detail-panel-step')).toHaveText(/Step 0/);

    await page.keyboard.press('ArrowRight');
    await expect(page.getByTestId('step-detail-panel-step')).toHaveText(/Step 1/);

    await page.keyboard.press('ArrowRight');
    await expect(page.getByTestId('step-detail-panel-step')).toHaveText(/Step 2/);

    // ArrowLeft walks back.
    await page.keyboard.press('ArrowLeft');
    await expect(page.getByTestId('step-detail-panel-step')).toHaveText(/Step 1/);
  });

  test('collapsing the rail and inspector persists across reload; `]` re-opens the inspector', async ({ page }) => {
    // Collapse the lens rail and the inspector.
    await page.getByTestId('lens-rail-collapse').click();
    await page.getByTestId('inspector-collapse').click();

    await expect(page.getByTestId('lens-rail')).toHaveAttribute(
      'data-collapsed',
      'true',
    );
    await expect(page.getByTestId('inspector-expand')).toBeVisible();
    await expect(page).toHaveURL(/[?&]left=0/);
    await expect(page).toHaveURL(/[?&]right=0/);

    await page.reload();
    await expect(page.getByTestId('token-heatmap-plot')).toBeVisible();
    // State re-hydrates from the URL after reload.
    await expect(page.getByTestId('lens-rail')).toHaveAttribute(
      'data-collapsed',
      'true',
    );
    await expect(page.getByTestId('inspector-expand')).toBeVisible();

    // Press `]` and the inspector re-opens.
    await page.locator('body').click();
    await page.keyboard.press(']');
    await expect(page.getByTestId('inspector-collapse')).toBeVisible();
    await expect(page.getByTestId('inspector-expand')).toBeHidden();
  });

  test('? opens the keyboard help dialog', async ({ page }) => {
    // Help shortcut is a top-level binding; focus the body so the keydown
    // isn't swallowed by a form input.
    await page.locator('body').click();
    await page.keyboard.press('?');

    const dialog = page.getByTestId('keymap-help-dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole('heading', { name: /keyboard shortcuts/i })).toBeVisible();

    // Esc closes again so subsequent tests in this file (none here, but for
    // future maintainers) start from a known state.
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
  });
});
