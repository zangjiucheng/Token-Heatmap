import { render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useReducedMotion } from '@/hooks/useReducedMotion';

interface MockMql {
  matches: boolean;
  media: string;
  listeners: Set<(event: MediaQueryListEvent) => void>;
  addEventListener: (
    type: 'change',
    listener: (event: MediaQueryListEvent) => void,
  ) => void;
  removeEventListener: (
    type: 'change',
    listener: (event: MediaQueryListEvent) => void,
  ) => void;
  dispatchChange: (matches: boolean) => void;
}

function mockMatchMedia(initialMatches: boolean): MockMql {
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const mql: MockMql = {
    matches: initialMatches,
    media: '(prefers-reduced-motion: reduce)',
    listeners,
    addEventListener: (_type, listener) => listeners.add(listener),
    removeEventListener: (_type, listener) => listeners.delete(listener),
    dispatchChange(matches: boolean) {
      this.matches = matches;
      listeners.forEach((l) =>
        l({ matches } as MediaQueryListEvent),
      );
    },
  };
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    configurable: true,
    value: vi.fn().mockImplementation(() => mql),
  });
  return mql;
}

function ConsumerHarness() {
  useReducedMotion();
  return <div data-testid="consumer">consumer</div>;
}

describe('reduced motion', () => {
  let originalMatchMedia: typeof window.matchMedia;

  beforeEach(() => {
    originalMatchMedia = window.matchMedia;
    document.body.removeAttribute('data-reduced-motion');
  });

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    document.body.removeAttribute('data-reduced-motion');
  });

  it('sets data-reduced-motion on body when prefers-reduced-motion: reduce is active', () => {
    mockMatchMedia(true);
    render(<ConsumerHarness />);
    expect(document.body.getAttribute('data-reduced-motion')).toBe('true');
  });

  it('does not set data-reduced-motion when prefers-reduced-motion: no-preference', () => {
    mockMatchMedia(false);
    render(<ConsumerHarness />);
    expect(document.body.getAttribute('data-reduced-motion')).toBeNull();
  });

  it('toggles the attribute when the media-query value changes', () => {
    const mql = mockMatchMedia(false);
    render(<ConsumerHarness />);
    expect(document.body.getAttribute('data-reduced-motion')).toBeNull();
    mql.dispatchChange(true);
    expect(document.body.getAttribute('data-reduced-motion')).toBe('true');
    mql.dispatchChange(false);
    expect(document.body.getAttribute('data-reduced-motion')).toBeNull();
  });
});
