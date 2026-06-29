import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type WheelEvent,
} from 'react';
import { createPortal } from 'react-dom';
import './VizModal.css';

export interface VizModalProps {
  open: boolean;
  onClose: () => void;
  /** Heading shown in the modal toolbar. */
  title: ReactNode;
  /** Natural width / height of the content, used to fit it on open. */
  aspect: number;
  /** The visualization to enlarge — any SVG; it fills the zoom stage. */
  children: ReactNode;
}

const MIN_SCALE = 1;
const MAX_SCALE = 8;
const STEP = 0.25;

const clamp = (s: number) =>
  Math.min(MAX_SCALE, Math.max(MIN_SCALE, Number(s.toFixed(2))));

/**
 * A lightbox for a single visualization: an overlay that shows the plot large
 * and lets you zoom (buttons, ⌘/Ctrl-wheel) and pan (scroll / drag). Closes on
 * Escape, backdrop click, or the ✕ button. The plot stays compact in its tab;
 * this is the "open it bigger to read the detail" affordance.
 */
export function VizModal({
  open,
  onClose,
  title,
  aspect,
  children,
}: VizModalProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [scale, setScale] = useState(1);

  // Reset the zoom level every time the modal is (re)opened.
  useEffect(() => {
    if (open) setScale(1);
  }, [open]);

  // Measure the scroll viewport so the plot can be fitted into it at 100%.
  useLayoutEffect(() => {
    if (!open) return undefined;
    const el = bodyRef.current;
    if (!el) return undefined;
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open]);

  // Escape closes.
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const onWheel = useCallback((e: WheelEvent) => {
    // Plain wheel pans (native scroll); modifier-wheel zooms.
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    setScale((s) => clamp(s - Math.sign(e.deltaY) * STEP));
  }, []);

  if (!open) return null;

  // Fit the content's aspect into the measured viewport, then apply the zoom.
  const pad = 24;
  const availW = Math.max(0, box.w - pad);
  const availH = Math.max(0, box.h - pad);
  let fitW = availW;
  let fitH = availW / aspect;
  if (fitH > availH) {
    fitH = availH;
    fitW = availH * aspect;
  }
  const stageW = Math.round(fitW * scale) || undefined;
  const stageH = Math.round(fitH * scale) || undefined;

  return createPortal(
    <div
      className="viz-modal"
      role="dialog"
      aria-modal="true"
      aria-label={typeof title === 'string' ? title : 'Visualization'}
      data-testid="viz-modal"
    >
      <button
        type="button"
        className="viz-modal__backdrop"
        aria-label="Close"
        onClick={onClose}
      />
      <div className="viz-modal__panel">
        <header className="viz-modal__header">
          <h2 className="viz-modal__title">{title}</h2>
          <div className="viz-modal__controls">
            <button
              type="button"
              className="viz-modal__btn"
              onClick={() => setScale((s) => clamp(s - STEP))}
              aria-label="Zoom out"
            >
              −
            </button>
            <span className="viz-modal__zoom" data-testid="viz-modal-zoom">
              {Math.round(scale * 100)}%
            </span>
            <button
              type="button"
              className="viz-modal__btn"
              onClick={() => setScale((s) => clamp(s + STEP))}
              aria-label="Zoom in"
            >
              +
            </button>
            <button
              type="button"
              className="viz-modal__btn viz-modal__btn--text"
              onClick={() => setScale(1)}
            >
              Reset
            </button>
            <button
              type="button"
              className="viz-modal__btn viz-modal__btn--close"
              onClick={onClose}
              aria-label="Close"
              data-testid="viz-modal-close"
            >
              ✕
            </button>
          </div>
        </header>
        <div className="viz-modal__body" ref={bodyRef} onWheel={onWheel}>
          <div
            className="viz-modal__stage"
            style={{ width: stageW, height: stageH }}
          >
            {children}
          </div>
        </div>
        <p className="viz-modal__hint">
          Scroll to pan · ⌘/Ctrl + scroll to zoom · Esc to close
        </p>
      </div>
    </div>,
    document.body,
  );
}
