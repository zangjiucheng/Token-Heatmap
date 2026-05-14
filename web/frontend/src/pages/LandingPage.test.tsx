import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { UseTraceResult } from '@/hooks/useTrace';
import type { UseBackendHealthResult } from '@/hooks/useBackendHealth';
import { TraceLoadError } from '@/lib/trace/errors';
import { LandingPage } from '@/pages/LandingPage';

const useTraceMock = vi.fn<[], UseTraceResult>();
const useHealthMock = vi.fn<[], UseBackendHealthResult>();

vi.mock('@/hooks/useTrace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/useTrace')>();
  return {
    ...actual,
    useTrace: () => useTraceMock(),
  };
});

vi.mock('@/hooks/useBackendHealth', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@/hooks/useBackendHealth')>();
  return {
    ...actual,
    useBackendHealth: () => useHealthMock(),
  };
});

function renderLanding(startPath = '/') {
  return render(
    <MemoryRouter initialEntries={[startPath]}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route
          path="/trace/:id"
          element={<div data-testid="viewer">viewer</div>}
        />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  useTraceMock.mockReset();
  useHealthMock.mockReset();
  useHealthMock.mockReturnValue({ status: 'unknown', probe: vi.fn() });
});

describe('LandingPage', () => {
  it('renders the empty state with the sample button', () => {
    useTraceMock.mockReturnValue({
      status: 'idle',
      trace: null,
      error: null,
      load: vi.fn(),
    });
    renderLanding();
    expect(
      screen.getByRole('heading', { name: /no trace loaded/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /try sample data/i }),
    ).toBeInTheDocument();
  });

  it('clicking the sample button calls load({ type: "sample" }) and navigates to /trace/sample', async () => {
    const load = vi.fn().mockResolvedValue(undefined);
    let status: UseTraceResult['status'] = 'idle';
    let trace: UseTraceResult['trace'] = null;
    useTraceMock.mockImplementation(() => ({
      status,
      trace,
      error: null,
      load,
    }));

    const { rerender } = renderLanding();

    await userEvent.click(
      screen.getByRole('button', { name: /try sample data/i }),
    );

    expect(load).toHaveBeenCalledWith({ type: 'sample' });

    status = 'ready';
    trace = { schema_version: '2.0.0', metadata: {}, steps: [] } as unknown as UseTraceResult['trace'];
    rerender(
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route
            path="/trace/:id"
            element={<div data-testid="viewer">viewer</div>}
          />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(screen.getByTestId('viewer')).toBeInTheDocument(),
    );
  });

  it('shows an error state when status is error and retry calls load again', async () => {
    const load = vi.fn().mockResolvedValue(undefined);
    useTraceMock.mockReturnValue({
      status: 'error',
      trace: null,
      error: TraceLoadError.parse('boom'),
      load,
    });

    renderLanding();
    expect(await screen.findByText(/boom/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /retry/i }));
    await waitFor(() => expect(load).toHaveBeenCalledWith({ type: 'sample' }));
  });

  it('shows the backend status banner when health is unhealthy', () => {
    useTraceMock.mockReturnValue({
      status: 'idle',
      trace: null,
      error: null,
      load: vi.fn(),
    });
    useHealthMock.mockReturnValue({ status: 'unhealthy', probe: vi.fn() });
    renderLanding();
    expect(screen.getByText(/backend unreachable/i)).toBeInTheDocument();
    // File-drop path still works (the EmptyState is still rendered).
    expect(
      screen.getByRole('button', { name: /try sample data/i }),
    ).toBeInTheDocument();
  });
});
