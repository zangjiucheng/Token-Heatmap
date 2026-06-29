import { useCallback, useState, type RefObject } from 'react';
import type { Trace } from '@/types/trace';
import { traceToCsv } from './traceToCsv';
import { heatmapToPng, triggerDownload } from './heatmapToPng';
import './ExportControls.css';

export interface ExportControlsProps {
  trace: Trace;
  /** Container element wrapping the heatmap canvases for PNG composition. */
  heatmapRef: RefObject<HTMLElement>;
  /** Optional base name used for downloaded files. */
  baseName?: string;
}

function sanitize(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
}

function defaultBaseName(trace: Trace): string {
  const model = sanitize(trace.metadata.model.split('/').pop() ?? 'trace');
  const ts = trace.metadata.generated_at.replace(/[:.]/g, '-');
  return `${model}_${ts}`;
}

export function ExportControls({
  trace,
  heatmapRef,
  baseName,
}: ExportControlsProps) {
  const [busy, setBusy] = useState<null | 'csv' | 'png'>(null);
  const [error, setError] = useState<string | null>(null);

  const resolveBaseName = useCallback(() => {
    return sanitize(baseName ?? defaultBaseName(trace));
  }, [baseName, trace]);

  const handleCsv = useCallback(() => {
    setBusy('csv');
    setError(null);
    try {
      const csv = traceToCsv(trace);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
      triggerDownload(blob, `${resolveBaseName()}.csv`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'CSV export failed');
    } finally {
      setBusy(null);
    }
  }, [trace, resolveBaseName]);

  const handlePng = useCallback(async () => {
    setBusy('png');
    setError(null);
    try {
      const container = heatmapRef.current;
      if (!container) {
        setError('Heatmap is not rendered yet');
        return;
      }
      const blob = await heatmapToPng(container);
      if (!blob) {
        setError('PNG export not supported in this browser');
        return;
      }
      triggerDownload(blob, `${resolveBaseName()}.png`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'PNG export failed');
    } finally {
      setBusy(null);
    }
  }, [heatmapRef, resolveBaseName]);

  return (
    <div
      className="export-controls"
      data-testid="export-controls"
      role="group"
      aria-label="Export"
    >
      <span className="export-controls__label">Export</span>
      <div className="export-controls__buttons">
        <button
          type="button"
          className="export-controls__button"
          onClick={handleCsv}
          disabled={busy !== null}
          data-testid="export-csv"
        >
          {busy === 'csv' ? 'Exporting…' : 'CSV'}
        </button>
        <button
          type="button"
          className="export-controls__button"
          onClick={handlePng}
          disabled={busy !== null}
          data-testid="export-png"
        >
          {busy === 'png' ? 'Exporting…' : 'PNG'}
        </button>
      </div>
      {error && (
        <p
          className="export-controls__error"
          role="alert"
          data-testid="export-error"
        >
          {error}
        </p>
      )}
    </div>
  );
}
