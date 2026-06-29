import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vitest';
import { TraceViewerPage } from './TraceViewerPage';
import { putTrace, clearTraceCache } from '@/lib/trace/store';
import { makeActivationTrace } from '@/features/activations/testFixtures';
import type { Trace } from '@/types/trace';

// Must match TimelineChart's internal padding constants.
const PADDING_LEFT = 32;
const PADDING_RIGHT = 8;

function readChartWidth(container: HTMLElement): number {
  const svg = container.querySelector('svg');
  if (!svg) throw new Error('chart svg not found');
  return Number(svg.getAttribute('width'));
}

function totalStepsFromHeatmap(): number {
  const heatmap = screen.getByRole('application');
  const match = /(\d+) generation steps/.exec(
    heatmap.getAttribute('aria-label') ?? '',
  );
  if (!match) throw new Error('could not infer step count');
  return Number(match[1]);
}

function xForStep(
  chart: HTMLElement,
  step: number,
  totalSteps: number,
): number {
  const chartW = readChartWidth(chart);
  const plotW = chartW - PADDING_LEFT - PADDING_RIGHT;
  const stepWidth = plotW / totalSteps;
  return PADDING_LEFT + stepWidth * step + stepWidth / 2;
}

describe('Cross-component step interactions', () => {
  afterEach(() => {
    clearTraceCache();
  });

  it('hovering a timeline updates the heatmap external hovered step', async () => {
    render(
      <MemoryRouter initialEntries={['/trace/sample']}>
        <TraceViewerPage />
      </MemoryRouter>,
    );

    const heatmapPlot = await waitFor(() =>
      screen.getByTestId('token-heatmap-plot'),
    );

    const entropy = screen.getByTestId('entropy-timeline');
    const totalSteps = totalStepsFromHeatmap();
    expect(totalSteps).toBeGreaterThan(0);

    const target = Math.min(3, totalSteps - 1);
    const x = xForStep(entropy, target, totalSteps);

    fireEvent.mouseMove(entropy, { clientX: x, clientY: 40 });

    await waitFor(() => {
      expect(heatmapPlot.getAttribute('data-external-hovered-step')).toBe(
        String(target),
      );
    });
  });

  it('clicking a timeline updates the selected step in the detail panel', async () => {
    render(
      <MemoryRouter initialEntries={['/trace/sample']}>
        <TraceViewerPage />
      </MemoryRouter>,
    );

    await waitFor(() => screen.getByTestId('token-heatmap-plot'));

    const selProb = screen.getByTestId('selected-probability-timeline');
    const totalSteps = totalStepsFromHeatmap();
    const target = Math.min(1, totalSteps - 1);
    const x = xForStep(selProb, target, totalSteps);

    fireEvent.click(selProb, { clientX: x, clientY: 40 });

    await waitFor(() => {
      expect(screen.getByTestId('step-detail-panel-step')).toHaveTextContent(
        `Step ${target}`,
      );
    });
  });

  it('clicking the token strip updates the activations heatmap cursor', async () => {
    const id = 'activations-cross-test';
    putTrace(id, makeActivationTrace() as unknown as Trace);

    render(
      <MemoryRouter initialEntries={[`/trace/${id}?tab=activations`]}>
        <Routes>
          <Route path="/trace/:id" element={<TraceViewerPage />} />
        </Routes>
      </MemoryRouter>,
    );

    const heatmap = await waitFor(() =>
      screen.getByTestId('activation-heatmap'),
    );
    expect(heatmap.getAttribute('data-selected-step')).toBe('');

    fireEvent.click(screen.getByTestId('generated-token-1'));

    await waitFor(() => {
      expect(
        screen
          .getByTestId('activation-heatmap')
          .getAttribute('data-selected-step'),
      ).toBe('1');
    });
  });
});
