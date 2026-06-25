import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { UseTraceResult } from '@/hooks/useTrace';
import type { UseBackendHealthResult } from '@/hooks/useBackendHealth';
import { BuildPage } from '@/pages/BuildPage';

const useTraceMock = vi.fn<() => UseTraceResult>();
const useHealthMock = vi.fn<() => UseBackendHealthResult>();

vi.mock('@/hooks/useTrace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/useTrace')>();
  return { ...actual, useTrace: () => useTraceMock() };
});

vi.mock('@/hooks/useBackendHealth', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@/hooks/useBackendHealth')>();
  return { ...actual, useBackendHealth: () => useHealthMock() };
});

function renderBuild() {
  return render(
    <MemoryRouter initialEntries={['/build']}>
      <Routes>
        <Route path="/build" element={<BuildPage />} />
        <Route
          path="/trace/:id"
          element={<div data-testid="viewer">viewer</div>}
        />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useTraceMock.mockReset();
  useHealthMock.mockReset();
  useHealthMock.mockReturnValue({ status: 'healthy', probe: vi.fn() });
});

describe('BuildPage', () => {
  it('renders the node graph and YAML preview', () => {
    useTraceMock.mockReturnValue({
      status: 'idle',
      trace: null,
      error: null,
      load: vi.fn(),
    });
    renderBuild();
    expect(screen.getByTestId('node-config-graph')).toBeInTheDocument();
    expect(screen.getByTestId('build-yaml-preview')).toHaveTextContent(
      'model:',
    );
  });

  it('Run calls load({type:"generate"}) and navigates when the trace resolves', async () => {
    const load = vi.fn().mockResolvedValue(undefined);
    let status: UseTraceResult['status'] = 'idle';
    let trace: UseTraceResult['trace'] = null;
    useTraceMock.mockImplementation(() => ({ status, trace, error: null, load }));

    const { rerender } = renderBuild();

    await userEvent.type(screen.getByTestId('node-input-prompt'), 'hello');
    await userEvent.click(screen.getByTestId('node-run'));

    expect(load).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'generate' }),
    );

    status = 'ready';
    trace = {
      schema_version: '2.0.0',
      metadata: {},
      steps: [],
    } as unknown as UseTraceResult['trace'];
    rerender(
      <MemoryRouter initialEntries={['/build']}>
        <Routes>
          <Route path="/build" element={<BuildPage />} />
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

  it('surfaces a backend error inside the Output node', () => {
    useTraceMock.mockReturnValue({
      status: 'error',
      trace: null,
      error: { message: 'model not found' } as UseTraceResult['error'],
      load: vi.fn(),
    });
    renderBuild();
    expect(screen.getByTestId('node-error')).toHaveTextContent(
      'model not found',
    );
  });
});
