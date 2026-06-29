/**
 * Polite ARIA live-region helper. Used to announce transient changes like
 * "selected step 17 of 64" to assistive tech without yanking focus.
 *
 * A single hidden region is appended to <body> on first use and reused for
 * every subsequent announcement. Re-announcing the same string is forced
 * by toggling textContent through an empty value so SRs treat it as new.
 */

const REGION_ID = 'a11y-live-region-polite';
const ASSERTIVE_REGION_ID = 'a11y-live-region-assertive';

export type LiveRegionPoliteness = 'polite' | 'assertive';

function getRegion(politeness: LiveRegionPoliteness): HTMLElement | null {
  if (typeof document === 'undefined') return null;
  const id = politeness === 'assertive' ? ASSERTIVE_REGION_ID : REGION_ID;
  let region = document.getElementById(id);
  if (!region) {
    region = document.createElement('div');
    region.id = id;
    region.setAttribute('role', 'status');
    region.setAttribute('aria-live', politeness);
    region.setAttribute('aria-atomic', 'true');
    // Visually hidden but readable by assistive tech.
    Object.assign(region.style, {
      position: 'absolute',
      width: '1px',
      height: '1px',
      padding: '0',
      margin: '-1px',
      overflow: 'hidden',
      clip: 'rect(0 0 0 0)',
      whiteSpace: 'nowrap',
      border: '0',
    } as Partial<CSSStyleDeclaration>);
    document.body.appendChild(region);
  }
  return region;
}

export function announceLiveRegion(
  message: string,
  politeness: LiveRegionPoliteness = 'polite',
): void {
  const region = getRegion(politeness);
  if (!region) return;
  // Clear first so identical messages re-announce.
  region.textContent = '';
  // Setting text in a microtask lets jsdom reflect the change reliably.
  window.setTimeout(() => {
    region.textContent = message;
  }, 0);
}

export function clearLiveRegion(
  politeness: LiveRegionPoliteness = 'polite',
): void {
  const region = getRegion(politeness);
  if (region) region.textContent = '';
}
