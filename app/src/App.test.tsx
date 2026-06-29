import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import axe from 'axe-core';
import App from '@/App';

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
  window.history.replaceState(null, '', '/');
});

describe('App', () => {
  it('renders the landing page at /', async () => {
    render(<App />);
    expect(
      await screen.findByRole('heading', { name: /no trace loaded/i }),
    ).toBeInTheDocument();
  });

  it('renders the skip-to-content link', () => {
    render(<App />);
    const skipLink = screen.getByText(/skip to content/i);
    expect(skipLink).toBeInTheDocument();
    expect(skipLink.getAttribute('href')).toBe('#main-content');
  });

  it('has zero axe-core violations on the landing page', async () => {
    const { container } = render(<App />);
    const results = await axe.run(container, {
      // jsdom can't compute real colors/layout, so skip rules that
      // depend on rendered visuals.
      rules: {
        'color-contrast': { enabled: false },
        region: { enabled: false },
      },
    });
    expect(results.violations).toEqual([]);
  });
});
