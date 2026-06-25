import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import App from '@/App';
import { BackendStatusBanner } from '@/components/feedback/BackendStatusBanner';

/**
 * Compute the accessible name for an element. We do not pull in a full
 * accessible-name algorithm — for the elements we render today, a name is
 * present iff one of the following is non-empty: aria-label, aria-labelledby
 * target, visible text content, or (for inputs) an associated <label>.
 */
function accessibleName(el: HTMLElement): string {
  const ariaLabel = el.getAttribute('aria-label');
  if (ariaLabel && ariaLabel.trim() !== '') return ariaLabel.trim();
  const labelledby = el.getAttribute('aria-labelledby');
  if (labelledby) {
    const labelEl = document.getElementById(labelledby);
    if (labelEl?.textContent && labelEl.textContent.trim() !== '') {
      return labelEl.textContent.trim();
    }
  }
  const title = el.getAttribute('title');
  if (title && title.trim() !== '') return title.trim();
  const text = el.textContent?.trim() ?? '';
  if (text !== '') return text;
  if (el instanceof HTMLInputElement) {
    if (el.labels && el.labels.length > 0) {
      const labelText = Array.from(el.labels)
        .map((l) => l.textContent?.trim() ?? '')
        .join(' ')
        .trim();
      if (labelText !== '') return labelText;
    }
    if (el.placeholder && el.placeholder.trim() !== '') {
      return el.placeholder.trim();
    }
  }
  return '';
}

function assertAllHaveNames(role: string, container: HTMLElement) {
  const elements = screen.queryAllByRole(role, { hidden: true });
  // Restrict to nodes inside the rendered container so we don't pick up
  // dialog portals from other tests.
  const scoped = elements.filter((el) => container.contains(el));
  for (const el of scoped) {
    const name = accessibleName(el);
    expect
      .soft(name, `${role} missing accessible name: ${el.outerHTML}`)
      .not.toBe('');
  }
}

describe('accessible names', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    window.history.replaceState(null, '', '/');
  });

  it('every button on the landing page has an accessible name', async () => {
    const { container } = render(<App />);
    await screen.findByRole('heading', { name: /no trace loaded/i });
    assertAllHaveNames('button', container);
    assertAllHaveNames('link', container);
  });

  it('BackendStatusBanner retry button has an accessible name when unhealthy', () => {
    const { container } = render(
      <BackendStatusBanner status="unhealthy" onRetry={() => {}} />,
    );
    assertAllHaveNames('button', container);
  });

  it('every interactive element on the trace viewer has an accessible name', async () => {
    window.history.replaceState(null, '', '/trace/sample');
    const { container } = render(<App />);
    await waitFor(
      () => {
        const heatmap = container.querySelector(
          '[data-testid="token-heatmap"]',
        );
        const empty = container.querySelector(
          '[data-testid="step-detail-panel"]',
        );
        expect(heatmap ?? empty).not.toBeNull();
      },
      { timeout: 4000 },
    );
    for (const role of ['button', 'link', 'checkbox', 'radio', 'slider']) {
      assertAllHaveNames(role, container);
    }
  });

  it('rail and inspector toggles have accessible names when both are collapsed', async () => {
    window.history.replaceState(null, '', '/trace/sample?left=0&right=0');
    const { container } = render(<App />);
    await waitFor(
      () => {
        const expandInspector = container.querySelector(
          '[data-testid="inspector-expand"]',
        );
        expect(expandInspector).not.toBeNull();
      },
      { timeout: 4000 },
    );
    assertAllHaveNames('button', container);
    // Sanity check: the collapse/expand affordances resolve by accessible name.
    expect(
      screen.getByRole('button', { name: /expand lens rail/i }),
    ).toBeTruthy();
    expect(
      screen.getByRole('button', { name: /show inspector/i }),
    ).toBeTruthy();
  });
});
