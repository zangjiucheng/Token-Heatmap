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

    // Navigate to the new route — assigning the hash fires `hashchange`, which
    // HashRouter listens for; RouteFocusManager then runs on the location change.
    act(() => {
      window.location.hash = '#/trace/sample';
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
