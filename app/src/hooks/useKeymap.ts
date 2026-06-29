import { useEffect, useRef } from 'react';
import {
  KEYMAP,
  isEditableTarget,
  matchesSingle,
  type KeyCombo,
  type KeymapBinding,
} from '@/lib/keymap';

export type KeymapHandler = (event: KeyboardEvent) => void;

export type KeymapHandlers = Partial<Record<string, KeymapHandler>>;

export interface UseKeymapOptions {
  /** When false the listener is detached. Defaults to true. */
  enabled?: boolean;
  /**
   * Maximum time between two keystrokes of a sequence (`g d`, `g h`) before
   * the partial chord resets. Defaults to 1200 ms.
   */
  sequenceTimeoutMs?: number;
  /**
   * Target to bind the listener to. Defaults to `window`. Useful for tests
   * that want to bind to a specific element.
   */
  target?: Window | HTMLElement | null;
}

interface PendingSequence {
  bindings: KeymapBinding[];
  expiresAt: number;
}

function isSequenceTrigger(
  binding: KeymapBinding,
): binding is KeymapBinding & { trigger: readonly [KeyCombo, KeyCombo] } {
  return Array.isArray(binding.trigger);
}

/**
 * Subscribe to keymap shortcuts. Re-renders are not triggered; handlers are
 * read from a ref so callers can pass inline closures without re-binding.
 */
export function useKeymap(
  handlers: KeymapHandlers,
  options: UseKeymapOptions = {},
): void {
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  const { enabled = true, sequenceTimeoutMs = 1200, target } = options;

  useEffect(() => {
    if (!enabled) return undefined;
    const resolvedTarget: Window | HTMLElement =
      target ?? (typeof window !== 'undefined' ? window : (null as never));
    if (!resolvedTarget) return undefined;

    let pending: PendingSequence | null = null;

    const onKeyDown = (event: Event) => {
      const kbEvent = event as KeyboardEvent;
      if (isEditableTarget(kbEvent.target)) return;
      // Ignore composing IME events.
      if (kbEvent.isComposing) return;

      const now = Date.now();
      if (pending && now > pending.expiresAt) {
        pending = null;
      }

      // First, see if a pending sequence can be completed.
      if (pending) {
        const completed = pending.bindings.find(
          (b) =>
            isSequenceTrigger(b) && matchesSingle(b.trigger[1], kbEvent),
        );
        pending = null;
        if (completed) {
          const handler = handlersRef.current[completed.id];
          if (handler) {
            kbEvent.preventDefault();
            handler(kbEvent);
          }
          return;
        }
        // Fall through — the new keystroke may start a fresh binding.
      }

      // Single-combo binding (with a registered handler).
      const single = KEYMAP.find(
        (b) =>
          !isSequenceTrigger(b) &&
          matchesSingle(b.trigger as KeyCombo, kbEvent) &&
          handlersRef.current[b.id] != null,
      );
      if (single) {
        kbEvent.preventDefault();
        handlersRef.current[single.id]!(kbEvent);
        return;
      }

      // Otherwise, see if this keystroke begins a sequence binding.
      const sequenceStarts = KEYMAP.filter(
        (b) =>
          isSequenceTrigger(b) &&
          matchesSingle(b.trigger[0], kbEvent) &&
          handlersRef.current[b.id] != null,
      );
      if (sequenceStarts.length > 0) {
        kbEvent.preventDefault();
        pending = {
          bindings: sequenceStarts,
          expiresAt: now + sequenceTimeoutMs,
        };
      }
    };

    resolvedTarget.addEventListener('keydown', onKeyDown);
    return () => {
      resolvedTarget.removeEventListener('keydown', onKeyDown);
    };
  }, [enabled, sequenceTimeoutMs, target]);
}
