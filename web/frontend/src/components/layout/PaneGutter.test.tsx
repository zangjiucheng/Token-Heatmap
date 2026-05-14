import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { PaneGutter, type PaneGutterProps } from '@/components/layout/PaneGutter';

function renderGutter(overrides: Partial<PaneGutterProps> = {}) {
  const onResize = vi.fn();
  const onCommit = vi.fn();
  const onReset = vi.fn();
  render(
    <PaneGutter
      side="left"
      width={280}
      minWidth={220}
      maxWidth={720}
      label="Resize controls panel"
      onResize={onResize}
      onCommit={onCommit}
      onReset={onReset}
      {...overrides}
    />,
  );
  return { onResize, onCommit, onReset };
}

describe('PaneGutter', () => {
  it('test_pointer_drag_invokes_onResize_with_delta', () => {
    const { onResize, onCommit } = renderGutter({ side: 'left' });
    const gutter = screen.getByRole('separator');
    fireEvent.pointerDown(gutter, {
      pointerId: 1,
      pointerType: 'mouse',
      button: 0,
      clientX: 100,
    });
    fireEvent.pointerMove(gutter, { pointerId: 1, clientX: 130 });
    fireEvent.pointerMove(gutter, { pointerId: 1, clientX: 145 });
    fireEvent.pointerUp(gutter, { pointerId: 1, clientX: 145 });

    // Two move events, deltas 30 and 15. For the left side the sign is
    // unchanged so onResize sees positive widening deltas.
    expect(onResize).toHaveBeenNthCalledWith(1, 30);
    expect(onResize).toHaveBeenNthCalledWith(2, 15);
    expect(onCommit).toHaveBeenCalledTimes(1);
  });

  it('inverts the sign for the right gutter so a leftward drag widens it', () => {
    const { onResize } = renderGutter({ side: 'right' });
    const gutter = screen.getByRole('separator');
    fireEvent.pointerDown(gutter, {
      pointerId: 1,
      pointerType: 'mouse',
      button: 0,
      clientX: 100,
    });
    fireEvent.pointerMove(gutter, { pointerId: 1, clientX: 80 });
    // Moved left by 20; the right pane should widen by 20.
    expect(onResize).toHaveBeenCalledWith(20);
  });

  it('test_keyboard_arrow_nudges_8px_and_shift_32px', () => {
    const { onResize, onCommit } = renderGutter();
    const gutter = screen.getByRole('separator');
    gutter.focus();
    fireEvent.keyDown(gutter, { key: 'ArrowRight' });
    fireEvent.keyDown(gutter, { key: 'ArrowLeft' });
    fireEvent.keyDown(gutter, { key: 'ArrowRight', shiftKey: true });
    fireEvent.keyDown(gutter, { key: 'ArrowLeft', shiftKey: true });
    expect(onResize).toHaveBeenNthCalledWith(1, 8);
    expect(onResize).toHaveBeenNthCalledWith(2, -8);
    expect(onResize).toHaveBeenNthCalledWith(3, 32);
    expect(onResize).toHaveBeenNthCalledWith(4, -32);
    // Each keyboard nudge commits so the parent persists.
    expect(onCommit).toHaveBeenCalledTimes(4);
  });

  it('test_double_click_invokes_onReset', () => {
    const { onReset } = renderGutter();
    fireEvent.doubleClick(screen.getByRole('separator'));
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it('test_aria_separator_attributes', () => {
    renderGutter({ width: 305, minWidth: 220, maxWidth: 720 });
    const gutter = screen.getByRole('separator');
    expect(gutter).toHaveAttribute('aria-orientation', 'vertical');
    expect(gutter).toHaveAttribute('aria-valuenow', '305');
    expect(gutter).toHaveAttribute('aria-valuemin', '220');
    expect(gutter).toHaveAttribute('aria-valuemax', '720');
    expect(gutter).toHaveAttribute('tabindex', '0');
  });

  it('does not invoke onResize when moving without a prior pointerdown', () => {
    const { onResize, onCommit } = renderGutter();
    const gutter = screen.getByRole('separator');
    fireEvent.pointerMove(gutter, { pointerId: 1, clientX: 50 });
    fireEvent.pointerUp(gutter, { pointerId: 1, clientX: 50 });
    expect(onResize).not.toHaveBeenCalled();
    expect(onCommit).not.toHaveBeenCalled();
  });
});
