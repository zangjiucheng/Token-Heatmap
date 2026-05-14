import { useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import { CATEGORY_LABELS, KEYMAP, type KeymapCategory } from '@/lib/keymap';
import './KeymapHelpDialog.css';

export interface KeymapHelpDialogProps {
  open: boolean;
  onClose: () => void;
  /**
   * Element that opened the dialog. Focus is restored to it on close. When
   * omitted the previously focused element at mount time is used.
   */
  triggerRef?: React.RefObject<HTMLElement | null>;
}

const CATEGORY_ORDER: ReadonlyArray<KeymapCategory> = [
  'selection',
  'view',
  'comparison',
  'layout',
  'navigation',
  'help',
];

export function KeymapHelpDialog({
  open,
  onClose,
  triggerRef,
}: KeymapHelpDialogProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    previousFocusRef.current =
      triggerRef?.current ??
      (document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null);
    // Focus the close button so Esc is immediately usable and screen readers
    // announce the dialog purpose.
    const id = window.setTimeout(() => {
      closeButtonRef.current?.focus();
    }, 0);
    return () => window.clearTimeout(id);
  }, [open, triggerRef]);

  useEffect(() => {
    if (open) return;
    // On close, return focus to whatever was focused before opening.
    const previous = previousFocusRef.current;
    if (previous && document.contains(previous)) {
      previous.focus();
    }
  }, [open]);

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      onClose();
      return;
    }
    // Simple focus trap: keep Tab inside the dialog.
    if (event.key === 'Tab') {
      const root = dialogRef.current;
      if (!root) return;
      const focusables = root.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  };

  if (!open) return null;

  /* eslint-disable jsx-a11y/click-events-have-key-events,
                    jsx-a11y/no-static-element-interactions,
                    jsx-a11y/no-noninteractive-element-interactions */
  return (
    <div
      className="keymap-help-backdrop"
      onClick={onClose}
      data-testid="keymap-help-backdrop"
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="keymap-help-title"
        className="keymap-help-dialog"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
        data-testid="keymap-help-dialog"
      >
        <header className="keymap-help-dialog__header">
          <h2 id="keymap-help-title" className="keymap-help-dialog__title">
            Keyboard shortcuts
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            className="keymap-help-dialog__close"
            onClick={onClose}
            aria-label="Close keyboard shortcut help"
          >
            ×
          </button>
        </header>
        <div className="keymap-help-dialog__body">
          {CATEGORY_ORDER.map((category) => {
            const bindings = KEYMAP.filter((b) => b.category === category);
            if (bindings.length === 0) return null;
            return (
              <section
                key={category}
                className="keymap-help-dialog__section"
                aria-label={CATEGORY_LABELS[category]}
              >
                <h3 className="keymap-help-dialog__section-title">
                  {CATEGORY_LABELS[category]}
                </h3>
                <dl className="keymap-help-dialog__list">
                  {bindings.map((b) => (
                    <div key={b.id} className="keymap-help-dialog__row">
                      <dt className="keymap-help-dialog__key">
                        <kbd>{b.display}</kbd>
                      </dt>
                      <dd className="keymap-help-dialog__description">
                        {b.description}
                      </dd>
                    </div>
                  ))}
                </dl>
              </section>
            );
          })}
        </div>
      </div>
    </div>
  );
  /* eslint-enable jsx-a11y/click-events-have-key-events,
                   jsx-a11y/no-static-element-interactions,
                   jsx-a11y/no-noninteractive-element-interactions */
}
