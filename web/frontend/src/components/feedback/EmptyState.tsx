import { useRef, useState, type DragEvent, type ReactNode } from 'react';
import './EmptyState.css';

export interface EmptyStateProps {
  onLoadSample: () => void;
  onFileDropped?: (file: File) => void;
  /**
   * Called when the user drops or picks two JSON files for diff-mode.
   * The order in which the user provides files determines (A, B).
   */
  onTwoFilesDropped?: (fileA: File, fileB: File) => void;
  title?: string;
  description?: ReactNode;
}

export function EmptyState({
  onLoadSample,
  onFileDropped,
  onTwoFilesDropped,
  title = 'No trace loaded',
  description = (
    <>
      Drop a JSON or CSV trace file here, choose one from disk, or load the
      bundled sample to explore the interactive token heatmap.
    </>
  ),
}: EmptyStateProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isDiffDragging, setIsDiffDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const diffInputRef = useRef<HTMLInputElement | null>(null);

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const files = event.dataTransfer.files;
    // If the user drops two files on the primary zone and the diff callback
    // is available, route to diff mode; otherwise fall back to single-file.
    if (files.length >= 2 && onTwoFilesDropped) {
      onTwoFilesDropped(files[0], files[1]);
      return;
    }
    const file = files?.[0];
    if (file && onFileDropped) {
      onFileDropped(file);
    }
  };

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (
    event,
  ) => {
    const file = event.target.files?.[0];
    if (file && onFileDropped) {
      onFileDropped(file);
    }
  };

  const handleDiffDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDiffDragging(true);
  };

  const handleDiffDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDiffDragging(false);
  };

  const handleDiffDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDiffDragging(false);
    const files = event.dataTransfer.files;
    if (files.length >= 2 && onTwoFilesDropped) {
      onTwoFilesDropped(files[0], files[1]);
    }
  };

  const handleDiffFileChange: React.ChangeEventHandler<HTMLInputElement> = (
    event,
  ) => {
    const files = event.target.files;
    if (!files || files.length < 2 || !onTwoFilesDropped) return;
    onTwoFilesDropped(files[0], files[1]);
  };

  return (
    <section className="empty-state" aria-labelledby="empty-state-title">
      <div className="empty-state__content">
        <div
          className="empty-state__dropzone empty-state__dropzone--primary"
          data-dragging={isDragging ? 'true' : 'false'}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="empty-state__copy">
            <p className="empty-state__eyebrow">Open trace</p>
            <h1 id="empty-state-title" className="empty-state__title">
              {title}
            </h1>
            <p className="empty-state__description">{description}</p>
          </div>
          <div className="empty-state__actions">
            <button
              type="button"
              className="empty-state__primary"
              onClick={onLoadSample}
            >
              Try sample data
            </button>
            <button
              type="button"
              className="empty-state__secondary"
              onClick={() => inputRef.current?.click()}
            >
              Choose file…
            </button>
            <input
              ref={inputRef}
              type="file"
              accept=".json,application/json,.csv,text/csv"
              className="empty-state__file-input"
              onChange={handleFileChange}
              aria-label="Trace file"
            />
          </div>
        </div>

        {onTwoFilesDropped && (
          <div
            className="empty-state__dropzone empty-state__dropzone--diff"
            data-testid="empty-state-diff-dropzone"
            data-dragging={isDiffDragging ? 'true' : 'false'}
            onDragOver={handleDiffDragOver}
            onDragLeave={handleDiffDragLeave}
            onDrop={handleDiffDrop}
          >
            <div className="empty-state__copy">
              <p className="empty-state__eyebrow">Compare traces</p>
              <h2 className="empty-state__subtitle">Activation diff</h2>
              <p className="empty-state__description">
                Drop two activation-trace JSON files here, or pick two from
                disk.
              </p>
            </div>
            <div className="empty-state__actions">
              <button
                type="button"
                className="empty-state__secondary"
                data-testid="empty-state-diff-pick"
                onClick={() => diffInputRef.current?.click()}
              >
                Choose two files…
              </button>
              <input
                ref={diffInputRef}
                type="file"
                accept=".json,application/json"
                multiple
                className="empty-state__file-input"
                data-testid="empty-state-diff-input"
                onChange={handleDiffFileChange}
                aria-label="Two activation trace files"
              />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
