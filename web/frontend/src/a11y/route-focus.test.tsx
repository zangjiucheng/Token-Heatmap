import { act, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import App from '@/App';

describe('route focus management', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    window.history.replaceState(null, '', '/');
  });

  it('focuses the h1 inside <main> on initial render', async () => {
    render(<App />);
    const h1 = await screen.findByRole('heading', {
      name: /no trace loaded/i,
    });
    await waitFor(() => {
      expect(document.activeElement).toBe(h1);
    });
  });

  it('moves focus to the new page heading on route transition', async () => {
    render(<App />);
    await screen.findByRole('heading', { name: /no trace loaded/i });

    // Push the new route — RouteFocusManager runs on location change.
    act(() => {
      window.history.pushState(null, '', '/trace/sample');
      window.dispatchEvent(new PopStateEvent('popstate'));
    });

    await waitFor(() => {
      const main = document.getElementById('main-content');
      expect(main).not.toBeNull();
      const active = document.activeElement as HTMLElement | null;
      // Focus should land on an h1 inside <main> (or <main> itself as fallback).
      expect(active && main?.contains(active)).toBe(true);
    });
  });
});
