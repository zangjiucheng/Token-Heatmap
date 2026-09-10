import { useRef, useState, type DragEvent, type ReactNode } from 'react';
import './EmptyState.css';

export interface EmptyStateProps {
  onLoadSample: () => void;
  onFileDropped?: (file: File) => void;
  title?: string;
  description?: ReactNode;
}

export function EmptyState({
  onLoadSample,
  onFileDropped,
  title = 'No trace loaded',
  description = (
    <>
      Drop a JSON trace file here, choose one from disk, or load the bundled
      sample to explore the interactive token heatmap.
    </>
  ),
}: EmptyStateProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

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
              accept=".json,.gz,application/json,application/gzip"
              className="empty-state__file-input"
              onChange={handleFileChange}
              aria-label="Trace file"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
