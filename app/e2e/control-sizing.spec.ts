import { expect, test } from '@playwright/test';
import {
  gotoLanding,
  loadBundledSample,
  waitForViewerReady,
} from './helpers/loadTrace';

/**
 * Interactive chrome must land on one of the three control-height tiers defined
 * in styles/theme.css (--control-h-sm / md / lg). This is measured rather than
 * asserted against the CSS source, because what matters is the height a control
 * actually renders at — padding, borders and line-height all move it, and a
 * `min-height` silently loses to content that is already taller.
 *
 * These heights had drifted to nine distinct values (23/24/26/27/30/31/32/35/36)
 * before the tiers existed, which reads as untidy wherever two controls sit side
 * by side. A new control that misses the scale fails here with its own selector
 * named, so the fix is obvious.
 */
const TIERS = [24, 32, 36];

const INTERACTIVE = [
  'button',
  'select',
  'input[type=text]',
  'input[type=number]',
  'a.app-header__icon-button',
].join(', ');

test.describe('control sizing', () => {
  test('every interactive control renders on a height tier', async ({
    page,
  }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    const offenders = await page.evaluate((selector) => {
      const seen = new Map<string, Set<number>>();
      document.querySelectorAll<HTMLElement>(selector).forEach((el) => {
        const rect = el.getBoundingClientRect();
        // Hidden or zero-size controls (collapsed panels, visually-hidden
        // radios backing a styled label) have no rendered height to check.
        if (rect.height < 8) return;
        const key = `${el.tagName.toLowerCase()}.${el.className.split(' ')[0] || '(unclassed)'}`;
        const set = seen.get(key) ?? new Set<number>();
        set.add(Math.round(rect.height));
        seen.set(key, set);
      });
      return [...seen.entries()]
        .map(([key, set]) => ({ key, heights: [...set].sort((a, b) => a - b) }))
        .filter(({ heights }) => heights.some((h) => ![24, 32, 36].includes(h)));
    }, INTERACTIVE);

    expect(
      offenders,
      `controls off the ${TIERS.join('/')}px scale — give them a ` +
        `var(--control-h-*) height:\n` +
        offenders.map((o) => `  ${o.key}: ${o.heights.join(', ')}px`).join('\n'),
    ).toEqual([]);
  });

  test('controls of the same kind agree with each other', async ({ page }) => {
    await gotoLanding(page);
    await loadBundledSample(page);
    await waitForViewerReady(page);

    // A single class rendering at two heights is the more visible defect: the
    // lens rail had items at 31px and 36px sitting directly on top of one
    // another.
    const inconsistent = await page.evaluate((selector) => {
      const seen = new Map<string, Set<number>>();
      document.querySelectorAll<HTMLElement>(selector).forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.height < 8) return;
        const cls = el.className.split(' ')[0];
        if (!cls) return;
        const set = seen.get(cls) ?? new Set<number>();
        set.add(Math.round(rect.height));
        seen.set(cls, set);
      });
      return [...seen.entries()]
        .filter(([, set]) => set.size > 1)
        .map(([cls, set]) => `${cls}: ${[...set].sort((a, b) => a - b).join(', ')}px`);
    }, INTERACTIVE);

    expect(
      inconsistent,
      `these classes render at more than one height:\n  ${inconsistent.join('\n  ')}`,
    ).toEqual([]);
  });
});
