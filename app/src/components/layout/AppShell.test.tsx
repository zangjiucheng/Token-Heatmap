import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppShell } from '@/components/layout/AppShell';

function mockMatchMedia(matches: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

function renderShell(initialEntry = '/') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<div>inner content</div>} />
          <Route path="trace/:id" element={<div>viewer content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
  mockMatchMedia(false);
});

describe('AppShell', () => {
  it('renders the header, skip link, and outlet content', () => {
    renderShell();
    expect(screen.getByRole('banner')).toBeInTheDocument();
    expect(screen.getByText(/skip to content/i)).toBeInTheDocument();
    expect(screen.getByText(/inner content/i)).toBeInTheDocument();
  });

  it('offers a way out of the viewer back to trace selection', async () => {
    // The desktop app has no browser Back button, so this control is the only
    // discoverable route from an open trace back to picking a different one.
    renderShell('/trace/uploaded');
    expect(screen.getByText(/viewer content/i)).toBeInTheDocument();

    const open = screen.getByRole('link', { name: /open another trace/i });
    await userEvent.click(open);

    expect(screen.getByText(/inner content/i)).toBeInTheDocument();
  });

  it('hides the open-trace control on the landing page', () => {
    renderShell('/');
    expect(
      screen.queryByRole('link', { name: /open another trace/i }),
    ).not.toBeInTheDocument();
  });

  it('theme toggle flips the data-theme attribute and writes to localStorage', async () => {
    renderShell();
    // Dark is the shipped default.
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');

    const toggle = screen.getByRole('button', {
      name: /switch to light theme/i,
    });
    await userEvent.click(toggle);

    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(window.localStorage.getItem('llm-heatmap-theme')).toBe('light');
  });
});
