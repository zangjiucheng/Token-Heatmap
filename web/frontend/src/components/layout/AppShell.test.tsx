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

function renderShell() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<div>inner content</div>} />
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

  it('theme toggle flips the data-theme attribute and writes to localStorage', async () => {
    renderShell();
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');

    const toggle = screen.getByRole('button', {
      name: /switch to dark theme/i,
    });
    await userEvent.click(toggle);

    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(window.localStorage.getItem('llm-heatmap-theme')).toBe('dark');
  });
});
