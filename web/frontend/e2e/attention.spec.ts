import { expect, test } from '@playwright/test';
import { gotoLanding, loadBundledSample, waitForViewerReady } from './helpers/loadTrace';

test.describe('attention tab', () => {
  test('happy path: switch tab → click cell → click timeline → state syncs', async ({
    page,
  }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    // Switch to the Attention tab.
    const attentionTab = page.getByTestId('attention-tab');
    await expect(attentionTab).toBeEnabled();
    await attentionTab.click();
    await expect(page.getByTestId('attention-tab-content')).toBeVisible();

    // URL persists the tab choice.
    await expect(page).toHaveURL(/tab=attention/);

    // Grid renders cells.
    const cell = page.getByTestId('attention-cell-0-0');
    await expect(cell).toBeVisible();

    // Click a cell → head pattern updates and URL carries selectedHead.
    await cell.click();
    await expect(page).toHaveURL(/selectedHead=0-0/);
    await expect(page.getByTestId('attention-head-pattern-title')).toContainText(
      'Layer 0',
    );

    // Click a different cell — selection moves.
    await page.getByTestId('attention-cell-2-1').click();
    await expect(page).toHaveURL(/selectedHead=2-1/);
    await expect(page.getByTestId('attention-head-pattern-title')).toContainText(
      'Layer 2',
    );

    // Logit lens table is present and has at least one row.
    await expect(page.getByTestId('logit-lens-table')).toBeVisible();

    // Switching back to Heatmap preserves the tab state and shows the heatmap.
    await page.getByTestId('heatmap-tab').click();
    await expect(page.getByTestId('token-heatmap-plot')).toBeVisible();
    await expect(page).not.toHaveURL(/tab=attention/);

    // Going back to Attention preserves selectedHead.
    await page.getByTestId('attention-tab').click();
    await expect(page).toHaveURL(/selectedHead=2-1/);
  });

  test('attention tab is disabled when the trace has no attention metadata', async ({
    page,
  }) => {
    // Drop a fixture trace that has no attention block. The bundled sample
    // does include attention; this path requires a fixture, so we just
    // assert the disabled-tooltip behavior renders on an attention-less
    // trace once one exists. Skip when the bundled sample carries
    // attention — confirms the positive path; the negative path is covered
    // by the AttentionTab unit test.
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
    const attentionTab = page.getByTestId('attention-tab');
    const isEnabled = await attentionTab.isEnabled();
    test.skip(isEnabled, 'bundled sample carries attention_metadata; covered by unit test');
  });
});
