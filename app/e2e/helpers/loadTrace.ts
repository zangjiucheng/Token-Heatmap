import { expect, type Page } from '@playwright/test';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const FIXTURES_DIR = path.resolve(__dirname, '..', 'fixtures');

export function fixturePath(name: string): string {
  return path.join(FIXTURES_DIR, name);
}

/**
 * Visit the landing page and wait for the empty-state UI to be ready so
 * subsequent interactions are deterministic.
 */
export async function gotoLanding(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /no trace loaded/i })).toBeVisible();
}

/**
 * Click the "Try sample data" button on the landing page and wait for the
 * trace viewer to render.
 */
export async function loadBundledSample(page: Page): Promise<void> {
  await page.getByRole('button', { name: /try sample data/i }).click();
  await expect(page).toHaveURL(/\/trace\/sample/);
  await expect(page.getByTestId('token-heatmap-plot')).toBeVisible();
}

/**
 * Drop a JSON fixture onto the landing page via the hidden file input. The
 * input is `display: none` but Playwright's setInputFiles bypasses that.
 */
export async function dropTraceFile(page: Page, fixtureName: string): Promise<void> {
  const input = page.locator('input[type="file"][aria-label="Trace file"]');
  await input.setInputFiles(fixturePath(fixtureName));
}

/**
 * Wait for the trace viewer to be fully ready: heatmap, detail panel, and
 * both timelines have rendered against a loaded trace.
 */
export async function waitForViewerReady(page: Page): Promise<void> {
  await expect(page.getByTestId('token-heatmap-plot')).toBeVisible();
  await expect(page.getByTestId('step-detail-panel')).toBeVisible();
  await expect(page.getByTestId('entropy-timeline')).toBeVisible();
  await expect(page.getByTestId('selected-probability-timeline')).toBeVisible();
}

/**
 * Focus the heatmap plot so keyboard shortcuts targeting selection are
 * received by the canvas application widget.
 */
export async function focusHeatmap(page: Page): Promise<void> {
  await page.getByTestId('token-heatmap-plot').focus();
}
