import { expect, test } from '@playwright/test';
import { fixturePath, gotoLanding } from './helpers/loadTrace';

/**
 * End-to-end coverage of the diff-mode UI:
 * drop two activation-trace JSON files onto the landing-page diff dropzone,
 * confirm the SPA routes to ``/diff/<id>`` and renders the heatmap, switch
 * the metric, and click a delta cell to confirm the right-rail top-K panel
 * updates with that (step, layer)'s top changed neurons.
 *
 * The fixtures are tiny hand-crafted activation traces (3 layers × 2 submodules
 * × 3 steps) with a shared tokenizer fingerprint so ``compare_activations``
 * resolves under ``token_id`` and the resulting heatmap has 9 cells per
 * submodule for click targeting.
 */

async function dropTwoTraces(
  page: import('@playwright/test').Page,
): Promise<void> {
  await page
    .getByTestId('empty-state-diff-input')
    .setInputFiles([
      fixturePath('activation-trace-a.json'),
      fixturePath('activation-trace-b.json'),
    ]);
  await expect(page).toHaveURL(/\/diff\//);
  await expect(page.getByTestId('diff-view-content')).toBeVisible();
  await expect(page.getByTestId('diff-heatmap')).toBeVisible();
}

test.describe('activations diff mode', () => {
  test('loads two traces and renders the diff heatmap', async ({ page }) => {
    await gotoLanding(page);
    await dropTwoTraces(page);

    const heatmap = page.getByTestId('diff-heatmap');
    await expect(heatmap).toHaveAttribute('data-metric', 'l2');
    // The fixtures expose 3 layers (rows) × 3 steps (columns).
    await expect(heatmap).toHaveAttribute('data-num-layers', '3');
    await expect(heatmap).toHaveAttribute('data-num-steps', '3');

    // Switching the metric updates the heatmap's data-metric attribute
    // without remounting (sequential reds vs diverging cosine ramp).
    await page.getByTestId('diff-metric-select').selectOption('cosine');
    await expect(heatmap).toHaveAttribute('data-metric', 'cosine');

    await page.getByTestId('diff-metric-select').selectOption('l2');
    await expect(heatmap).toHaveAttribute('data-metric', 'l2');
  });

  test('clicking a delta cell updates the right-rail top-K', async ({
    page,
  }) => {
    await gotoLanding(page);
    await dropTwoTraces(page);

    // Empty state of the detail panel before any cell is clicked.
    const detailPanel = page.getByTestId('diff-detail-panel');
    await expect(detailPanel).toBeVisible();
    await expect(detailPanel).toContainText(/click a cell/i);

    // Click step 1, layer 2. The DiffHeatmap emits one rect per (step, layer)
    // with data-testid="diff-cell-<step>-<layer>".
    await page.getByTestId('diff-cell-1-2').click();

    const title = page.getByTestId('diff-detail-panel-title');
    await expect(title).toContainText('Step 1');
    await expect(title).toContainText('L2');

    // At least one top-changed neuron row appears (top_changed_neurons is
    // populated for every layer-delta with nonzero movement).
    await expect(page.getByTestId('diff-top-neuron-0')).toBeVisible();

    // Clicking a different cell moves the selection.
    await page.getByTestId('diff-cell-2-0').click();
    await expect(title).toContainText('Step 2');
    await expect(title).toContainText('L0');
  });
});
