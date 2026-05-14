import { render, screen, waitFor, act } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import axe from 'axe-core';
import App from '@/App';
import { BackendStatusBanner } from '@/components/feedback/BackendStatusBanner';

const AXE_OPTIONS: axe.RunOptions = {
  // jsdom cannot compute real colors or layout; skip rules that depend on
  // actually rendered visuals so we focus on structural violations only.
  rules: {
    'color-contrast': { enabled: false },
    region: { enabled: false },
  },
  resultTypes: ['violations'],
};

async function runAxe(container: HTMLElement) {
  return axe.run(container, AXE_OPTIONS);
}

function filterSerious(results: axe.AxeResults) {
  return results.violations.filter(
    (v) => v.impact === 'serious' || v.impact === 'critical',
  );
}

describe('axe-core sweep', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    window.history.replaceState(null, '', '/');
  });

  it('landing page has no serious or critical violations', async () => {
    const { container } = render(<App />);
    await screen.findByRole('heading', { name: /no trace loaded/i });
    const results = await runAxe(container);
    expect(filterSerious(results)).toEqual([]);
  });

  it('trace viewer (loading state) has no serious or critical violations', async () => {
    window.history.replaceState(null, '', '/trace');
    const { container } = render(<App />);
    // Wait for any heading or loading state to appear.
    await waitFor(() => {
      const main = container.querySelector('main');
      expect(main).not.toBeNull();
    });
    const results = await runAxe(container);
    expect(filterSerious(results)).toEqual([]);
  });

  it('unhealthy BackendStatusBanner has no serious or critical violations', async () => {
    const { container } = render(
      <BackendStatusBanner status="unhealthy" onRetry={() => {}} />,
    );
    const results = await runAxe(container);
    expect(filterSerious(results)).toEqual([]);
  });

  it('trace viewer with sample trace has no serious or critical violations', async () => {
    window.history.replaceState(null, '', '/trace/sample');
    const { container } = render(<App />);
    await waitFor(
      async () => {
        // Either the heatmap renders or the page settles into another state.
        const heatmap = container.querySelector('[data-testid="token-heatmap"]');
        const empty = container.querySelector('[data-testid="step-detail-panel"]');
        expect(heatmap ?? empty).not.toBeNull();
      },
      { timeout: 4000 },
    );
    // Yield a tick for effects to settle before scanning.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    const results = await runAxe(container);
    expect(filterSerious(results)).toEqual([]);
  });
});
