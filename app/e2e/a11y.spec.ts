import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import {
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

const SERIOUS_OR_CRITICAL = ['serious', 'critical'] as const;

async function expectNoSeriousViolations(page: Page, label: string): Promise<void> {
  const results = await new AxeBuilder({ page })
    // WCAG 2.1 AA scope; canvas-only widgets are exercised by dedicated
    // keyboard/aria coverage in src/a11y/*, so we trust the axe rules for
    // contrast, landmarks, labels, and structure here.
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  const serious = results.violations.filter((v) =>
    (SERIOUS_OR_CRITICAL as readonly string[]).includes(v.impact ?? ''),
  );
  expect(
    serious,
    `${label} should have zero serious/critical axe violations.\n${JSON.stringify(
      serious,
      null,
      2,
    )}`,
  ).toEqual([]);
}

test.describe('accessibility', () => {
  test('landing page is axe-clean', async ({ page }) => {
    await gotoLanding(page);
    await expectNoSeriousViolations(page, 'landing');
  });

  test('viewer with a loaded trace is axe-clean', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);
    await expectNoSeriousViolations(page, 'viewer-with-trace');
  });

  test('empty viewer (no step selected) is axe-clean', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    // The viewer's "empty" state here is: trace loaded, no step selected.
    // Detail panel renders its placeholder copy ("Click a step in the
    // heatmap…"). Use Esc to make sure no step is selected.
    await page.locator('body').click();
    await page.keyboard.press('Escape');
    await expectNoSeriousViolations(page, 'viewer-empty-selection');
  });
});
