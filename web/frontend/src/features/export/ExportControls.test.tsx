import { createRef } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { ExportControls } from './ExportControls';

const trace = sampleTrace as unknown as Trace;

interface ClickSpy {
  mockRestore: () => void;
  mock: { calls: unknown[][]; instances: unknown[] };
}

describe('ExportControls', () => {
  let clickSpy: ClickSpy;

  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => 'blob:mock') as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn() as unknown as typeof URL.revokeObjectURL;
    clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {}) as unknown as ClickSpy;
  });

  afterEach(() => {
    clickSpy.mockRestore();
  });

  it('triggers a CSV download with the expected filename pattern', async () => {
    const heatmapRef = createRef<HTMLDivElement>();
    render(
      <div ref={heatmapRef}>
        <ExportControls trace={trace} heatmapRef={heatmapRef} />
      </div>,
    );
    await userEvent.click(screen.getByTestId('export-csv'));
    expect(clickSpy).toHaveBeenCalledTimes(1);
    // The synthetic <a> created by triggerDownload sets its `download` attr.
    const anchors = document.querySelectorAll('a[download]');
    // The anchor is removed immediately after click, so capture from clickSpy's
    // call instance via the `this` arg.
    const callContext = clickSpy.mock.instances[0] as HTMLAnchorElement;
    expect(callContext.download).toMatch(/\.csv$/);
    expect(anchors.length).toBe(0);
  });

  it('shows an error when no heatmap is rendered for PNG export', async () => {
    const heatmapRef = createRef<HTMLElement>();
    render(<ExportControls trace={trace} heatmapRef={heatmapRef} />);
    await userEvent.click(screen.getByTestId('export-png'));
    expect(await screen.findByTestId('export-error')).toBeInTheDocument();
  });

  it('exposes both buttons with accessible names', () => {
    const heatmapRef = createRef<HTMLElement>();
    render(<ExportControls trace={trace} heatmapRef={heatmapRef} />);
    expect(screen.getByRole('button', { name: /csv/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /png/i })).toBeInTheDocument();
  });
});
