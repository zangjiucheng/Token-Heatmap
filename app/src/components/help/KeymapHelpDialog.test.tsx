import { render, screen, fireEvent, act } from '@testing-library/react';
import { useRef, useState } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { KeymapHelpDialog } from './KeymapHelpDialog';
import { useKeymap } from '@/hooks/useKeymap';

function Harness() {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  useKeymap({ 'help.open': () => setOpen(true) });
  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen(true)}
        data-testid="open-help"
      >
        Open help
      </button>
      <KeymapHelpDialog
        open={open}
        onClose={() => setOpen(false)}
        triggerRef={triggerRef}
      />
    </>
  );
}

describe('KeymapHelpDialog', () => {
  it('does not render when closed', () => {
    render(<KeymapHelpDialog open={false} onClose={vi.fn()} />);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('opens when "?" is pressed', () => {
    render(<Harness />);
    expect(screen.queryByRole('dialog')).toBeNull();
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '?' }));
    });
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: /keyboard shortcuts/i }),
    ).toBeInTheDocument();
  });

  it('closes when Escape is pressed inside the dialog', async () => {
    render(<Harness />);
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '?' }));
    });
    const dialog = screen.getByRole('dialog');
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('returns focus to the trigger when closed', async () => {
    render(<Harness />);
    const trigger = screen.getByTestId('open-help');
    trigger.focus();
    fireEvent.click(trigger);
    const dialog = await screen.findByRole('dialog');
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(document.activeElement).toBe(trigger);
  });

  it('closes when the backdrop is clicked', () => {
    render(<Harness />);
    fireEvent.click(screen.getByTestId('open-help'));
    fireEvent.click(screen.getByTestId('keymap-help-backdrop'));
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
