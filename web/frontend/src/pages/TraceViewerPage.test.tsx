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
  it('renders the center stack inside the scrollable parent pane', async () => {
    renderViewer();
    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    // jsdom doesn't apply stylesheet rules, so we only verify that the
    // wrapper carries the expected class. The actual scrolling contract
    // is owned by `.three-pane__center` (overflow: auto) and exercised
    // by the e2e suite; `.trace-viewer-center` uses min-height: 100%
    // (not a fixed height) so content past the viewport flows into the
    // pane's scrollbar instead of being clipped.
    const center = screen.getByTestId('trace-viewer-center');
    expect(center.classList.contains('trace-viewer-center')).toBe(true);
  });

  it('keeps a horizontal scroller on the heatmap for wide traces', async () => {
    renderViewer();
    const heatmap = await waitFor(() =>
      screen.getByTestId('trace-viewer-heatmap-container'),
    );
    // CSS class `.trace-viewer-center__heatmap` now sets only
    // `overflow-x: auto` — vertical overflow is delegated to the parent
    // pane so the whole center stack scrolls together.
    expect(heatmap.classList.contains('trace-viewer-center__heatmap')).toBe(
      true,
    );
  });

  it('renders both timelines as compact plots below the heatmap', async () => {
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
    expect(screen.queryByTestId('trace-viewer-right-timelines')).toBeNull();
  });

  it('uses the updated side-pane labels', async () => {
    renderViewer();
    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    expect(
      screen.getByRole('complementary', { name: /view settings/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('complementary', { name: /inspector/i }),
    ).toBeInTheDocument();
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
