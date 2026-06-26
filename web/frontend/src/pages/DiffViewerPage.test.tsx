import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LandingPage } from '@/pages/LandingPage';
import { DiffViewerPage } from '@/pages/DiffViewerPage';
import { clearTraceCache, putDiffPair } from '@/lib/trace/store';
import { makeTwoActivationTraces } from '@/features/activations/testFixtures';

// The trace schema bundled in the frontend doesn't yet carry activation
// fields, so we mock the loader to bypass validation for the routing test
// (validation correctness is exercised elsewhere).
vi.mock('@/lib/trace/load', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/trace/load')>();
  return {
    ...actual,
    loadTraceFromFile: async (file: File) => {
      const text = await file.text();
      return JSON.parse(text);
    },
  };
});

function renderApp(initialEntries: string[] = ['/']) {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/trace/:id" element={<div data-testid="viewer">v</div>} />
        <Route path="/diff/:id" element={<DiffViewerPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  clearTraceCache();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DiffViewerPage routing', () => {
  it('shows an error when no diff pair is staged for the id', () => {
    render(
      <MemoryRouter initialEntries={['/diff/missing']}>
        <Routes>
          <Route path="/diff/:id" element={<DiffViewerPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText(/no diff pair found/i)).toBeInTheDocument();
  });

  it('renders the DiffView when a pair is staged for the id', () => {
    const { traceA, traceB } = makeTwoActivationTraces();
    putDiffPair('abc', traceA, traceB);
    render(
      <MemoryRouter initialEntries={['/diff/abc']}>
        <Routes>
          <Route path="/diff/:id" element={<DiffViewerPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByTestId('diff-viewer-page')).toBeInTheDocument();
    expect(screen.getByTestId('diff-view-content')).toBeInTheDocument();
  });

  it('landing → drop two files → routed to /diff/<id> and renders the diff', async () => {
    const user = userEvent.setup();
    const { traceA, traceB } = makeTwoActivationTraces();
    const fileA = new File([JSON.stringify(traceA)], 'a.json', {
      type: 'application/json',
    });
    const fileB = new File([JSON.stringify(traceB)], 'b.json', {
      type: 'application/json',
    });

    renderApp();

    const input = screen.getByTestId(
      'empty-state-diff-input',
    ) as HTMLInputElement;
    await user.upload(input, [fileA, fileB]);

    await waitFor(() =>
      expect(screen.getByTestId('diff-viewer-page')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('diff-view-content')).toBeInTheDocument();
  });
});
