import { createRef } from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { syntheticTrace } from '@/features/heatmap/testFixtures';
import { ControlBar } from './ControlBar';

function renderControlBar() {
  const trace = syntheticTrace(6, 4);
  const ref = createRef<HTMLElement>();
  render(
    <MemoryRouter initialEntries={['/trace/sample']}>
      <ControlBar trace={trace} heatmapRef={ref} />
    </MemoryRouter>,
  );
}

describe('ControlBar', () => {
  it('renders the heatmap view controls in one toolbar', () => {
    renderControlBar();
    const bar = screen.getByTestId('trace-viewer-controls');
    expect(bar).toBeInTheDocument();
    expect(screen.getByTestId('comparison-toggle')).toBeInTheDocument();
    expect(screen.getByTestId('value-column-toggle')).toBeInTheDocument();
    expect(screen.getByTestId('step-range-filter')).toBeInTheDocument();
    expect(screen.getByTestId('color-range-controls')).toBeInTheDocument();
    expect(screen.getByTestId('export-controls')).toBeInTheDocument();
  });
});
