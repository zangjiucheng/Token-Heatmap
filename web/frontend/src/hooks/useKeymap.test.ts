import { renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useKeymap } from './useKeymap';

function dispatchKey(key: string, init: KeyboardEventInit = {}): void {
  window.dispatchEvent(new KeyboardEvent('keydown', { key, ...init }));
}

describe('useKeymap', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('invokes a registered handler when its key fires', () => {
    const onHelp = vi.fn();
    renderHook(() => useKeymap({ 'help.open': onHelp }));
    dispatchKey('?');
    expect(onHelp).toHaveBeenCalledTimes(1);
  });

  it('does not invoke handlers for keys that have no binding', () => {
    const onHelp = vi.fn();
    renderHook(() => useKeymap({ 'help.open': onHelp }));
    dispatchKey('a');
    expect(onHelp).not.toHaveBeenCalled();
  });

  it('unregisters the listener on unmount', () => {
    const onHelp = vi.fn();
    const { unmount } = renderHook(() => useKeymap({ 'help.open': onHelp }));
    unmount();
    dispatchKey('?');
    expect(onHelp).not.toHaveBeenCalled();
  });

  it('ignores keystrokes when an input is focused', () => {
    const onHelp = vi.fn();
    renderHook(() => useKeymap({ 'help.open': onHelp }));
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    input.dispatchEvent(
      new KeyboardEvent('keydown', { key: '?', bubbles: true }),
    );
    expect(onHelp).not.toHaveBeenCalled();
  });

  it('ignores keystrokes inside a textarea', () => {
    const onHelp = vi.fn();
    renderHook(() => useKeymap({ 'help.open': onHelp }));
    const ta = document.createElement('textarea');
    document.body.appendChild(ta);
    ta.focus();
    ta.dispatchEvent(
      new KeyboardEvent('keydown', { key: '?', bubbles: true }),
    );
    expect(onHelp).not.toHaveBeenCalled();
  });

  it('matches arrow-key bindings', () => {
    const onPrev = vi.fn();
    const onNext = vi.fn();
    renderHook(() =>
      useKeymap({ 'selection.prev': onPrev, 'selection.next': onNext }),
    );
    dispatchKey('ArrowLeft');
    dispatchKey('ArrowRight');
    expect(onPrev).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it('handles two-key sequences (g then d)', () => {
    const onGotoDetail = vi.fn();
    renderHook(() =>
      useKeymap({ 'navigation.gotoDetail': onGotoDetail }),
    );
    dispatchKey('g');
    dispatchKey('d');
    expect(onGotoDetail).toHaveBeenCalledTimes(1);
  });

  it('resets sequence state when a non-matching key follows the prefix', () => {
    const onGotoDetail = vi.fn();
    const onGotoHeatmap = vi.fn();
    renderHook(() =>
      useKeymap({
        'navigation.gotoDetail': onGotoDetail,
        'navigation.gotoHeatmap': onGotoHeatmap,
      }),
    );
    dispatchKey('g');
    dispatchKey('x'); // not part of any sequence
    dispatchKey('d'); // should not complete the chord
    expect(onGotoDetail).not.toHaveBeenCalled();
    expect(onGotoHeatmap).not.toHaveBeenCalled();
  });

  it('respects the enabled option', () => {
    const onHelp = vi.fn();
    renderHook(() => useKeymap({ 'help.open': onHelp }, { enabled: false }));
    dispatchKey('?');
    expect(onHelp).not.toHaveBeenCalled();
  });

  it('disables matching when alt modifier is present but not required', () => {
    const onHelp = vi.fn();
    renderHook(() => useKeymap({ 'help.open': onHelp }));
    dispatchKey('?', { altKey: true });
    expect(onHelp).not.toHaveBeenCalled();
  });
});
