import { render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { TraceViewerPage } from './TraceViewerPage';

function renderViewer(initialEntries: string[] = ['/trace/sample']) {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <TraceViewerPage />
    </MemoryRouter>,
  );
}

describe('TraceViewerPage layout', () => {
  it('renders the lens canvas inside the workspace', async () => {
    renderViewer();
    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    // jsdom doesn't apply stylesheet rules, so we only verify the wrapper
    // carries the expected class. The canvas body owns the scroll contract
    // (overflow: auto over a definite flex height) — exercised by the e2e suite.
    const canvas = screen.getByTestId('trace-viewer-center');
    expect(canvas.classList.contains('trace-workspace__canvas')).toBe(true);
  });

  it('gives the lens body its own scroll container', async () => {
    renderViewer();
    const body = await waitFor(() =>
      screen.getByTestId('trace-viewer-heatmap-container'),
    );
    expect(body.classList.contains('trace-workspace__canvas-body')).toBe(true);
  });

  it('renders both timelines as compact plots below the lens', async () => {
    renderViewer();
    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    const centerTimelines = screen.getByTestId('trace-viewer-center-timelines');
    expect(centerTimelines).toBeInTheDocument();
    expect(
      within(centerTimelines).getByTestId('entropy-timeline'),
    ).toBeInTheDocument();
    expect(
      within(centerTimelines).getByTestId('selected-probability-timeline'),
    ).toBeInTheDocument();
  });

  it('exposes the lens rail and inspector landmarks', async () => {
    renderViewer();
    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    expect(
      screen.getByRole('navigation', { name: /analysis lenses/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('complementary', { name: /inspector/i }),
    ).toBeInTheDocument();
  });

  it('marks the active lens in the rail', async () => {
    renderViewer();
    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    const heatmapLens = screen.getByTestId('heatmap-tab');
    expect(heatmapLens).toHaveAttribute('aria-current', 'page');
  });

  it('does not show the attention-pattern prompt before a head is selected', async () => {
    renderViewer(['/trace/sample?tab=attention']);
    await waitFor(() => screen.getByTestId('attention-tab-step-empty'));

    expect(
      screen.queryByText(/click a \(layer, head\) cell/i),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId('trace-viewer-right-attention-pattern'),
    ).not.toBeInTheDocument();
  });
});
