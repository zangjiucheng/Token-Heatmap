import { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ThreePaneLayout } from '@/components/layout/ThreePaneLayout';
import { useKeymap } from '@/hooks/useKeymap';
import { PANE_WIDTHS_STORAGE_KEY } from '@/hooks/usePaneWidths';

beforeEach(() => {
  window.localStorage.removeItem(PANE_WIDTHS_STORAGE_KEY);
});

function setMatchMedia(matchesNarrow: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query === '(max-width: 1023px)' ? matchesNarrow : false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

beforeEach(() => {
  setMatchMedia(false);
});

function HarnessWithKeymap({
  initialLeftOpen = true,
  initialRightOpen = true,
}: {
  initialLeftOpen?: boolean;
  initialRightOpen?: boolean;
}) {
  const [leftOpen, setLeftOpen] = useState(initialLeftOpen);
  const [rightOpen, setRightOpen] = useState(initialRightOpen);
  useKeymap({
    'panel.toggleLeft': () => setLeftOpen((v) => !v),
    'panel.toggleRight': () => setRightOpen((v) => !v),
  });
  return (
    <ThreePaneLayout
      leftLabel="View settings"
      rightLabel="Inspector"
      leftOpen={leftOpen}
      rightOpen={rightOpen}
      onToggleLeft={setLeftOpen}
      onToggleRight={setRightOpen}
      left={
        <div>
          <button type="button">left first focusable</button>
          <button type="button">left second focusable</button>
        </div>
      }
      center={<div>center content</div>}
      right={
        <div>
          <button type="button">right first focusable</button>
        </div>
      }
    />
  );
}

describe('ThreePaneLayout', () => {
  it('renders three named slots', () => {
    render(
      <ThreePaneLayout
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    expect(screen.getByText(/left content/i)).toBeInTheDocument();
    expect(screen.getByText(/center content/i)).toBeInTheDocument();
    expect(screen.getByText(/right content/i)).toBeInTheDocument();
    expect(screen.getByTestId('three-pane-left')).toBeInTheDocument();
    expect(screen.getByTestId('three-pane-center')).toBeInTheDocument();
    expect(screen.getByTestId('three-pane-right')).toBeInTheDocument();
  });

  it('collapses to stacked layout when matchMedia reports a narrow viewport', () => {
    setMatchMedia(true);
    render(
      <ThreePaneLayout
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    const container = screen.getByTestId('three-pane-left').parentElement;
    expect(container).not.toBeNull();
    expect(container?.getAttribute('data-narrow')).toBe('true');
  });

  it('uses wide layout when matchMedia reports a wide viewport', () => {
    setMatchMedia(false);
    render(
      <ThreePaneLayout
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    const container = screen.getByTestId('three-pane-left').parentElement;
    expect(container?.getAttribute('data-narrow')).toBe('false');
  });

  it('rail buttons have accessible names when collapsed', () => {
    render(
      <ThreePaneLayout
        leftLabel="View settings"
        rightLabel="Inspector"
        leftOpen={false}
        rightOpen={false}
        onToggleLeft={() => {}}
        onToggleRight={() => {}}
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    expect(
      screen.getByRole('button', { name: /show view settings/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /show inspector/i }),
    ).toBeInTheDocument();
  });

  it('clicking the rail button invokes onToggleLeft(true)', async () => {
    const user = userEvent.setup();
    const onToggleLeft = vi.fn();
    render(
      <ThreePaneLayout
        leftLabel="View settings"
        rightLabel="Inspector"
        leftOpen={false}
        rightOpen
        onToggleLeft={onToggleLeft}
        onToggleRight={() => {}}
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    await user.click(
      screen.getByRole('button', { name: /show view settings/i }),
    );
    expect(onToggleLeft).toHaveBeenCalledWith(true);
  });

  it('keyboard left bracket toggles the left panel', async () => {
    render(<HarnessWithKeymap />);
    expect(
      screen.queryByTestId('three-pane-left-expand'),
    ).not.toBeInTheDocument();
    fireEvent.keyDown(window, { key: '[' });
    expect(screen.getByTestId('three-pane-left-expand')).toBeInTheDocument();
    fireEvent.keyDown(window, { key: '[' });
    expect(
      screen.queryByTestId('three-pane-left-expand'),
    ).not.toBeInTheDocument();
  });

  it('keyboard right bracket toggles the right panel', async () => {
    render(<HarnessWithKeymap />);
    expect(
      screen.queryByTestId('three-pane-right-expand'),
    ).not.toBeInTheDocument();
    fireEvent.keyDown(window, { key: ']' });
    expect(screen.getByTestId('three-pane-right-expand')).toBeInTheDocument();
    fireEvent.keyDown(window, { key: ']' });
    expect(
      screen.queryByTestId('three-pane-right-expand'),
    ).not.toBeInTheDocument();
  });

  it('focus moves into the body when a collapsed pane is expanded', async () => {
    const user = userEvent.setup();
    render(<HarnessWithKeymap initialLeftOpen={false} />);
    const railButton = screen.getByRole('button', {
      name: /show view settings/i,
    });
    await user.click(railButton);
    expect(document.activeElement).toBe(
      screen.getByRole('button', { name: /left first focusable/i }),
    );
  });

  it('renders pane gutters between each side panel and the center', () => {
    render(
      <ThreePaneLayout
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    expect(screen.getByTestId('three-pane-gutter-left')).toBeInTheDocument();
    expect(screen.getByTestId('three-pane-gutter-right')).toBeInTheDocument();
    // The grid wrapper exposes the current widths as CSS variables so the
    // layout responds to drag without re-rendering the pane bodies.
    const root = screen.getByTestId('three-pane-left').parentElement!;
    expect(root.style.getPropertyValue('--pane-left')).toBe('280px');
    expect(root.style.getPropertyValue('--pane-right')).toBe('320px');
  });

  it('test_gutter_hidden_when_panel_collapsed', () => {
    render(
      <ThreePaneLayout
        leftOpen={false}
        rightOpen
        onToggleLeft={() => {}}
        onToggleRight={() => {}}
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    // Left gutter is not rendered (and therefore not focusable) when the
    // left pane is collapsed; the right gutter stays.
    expect(
      screen.queryByTestId('three-pane-gutter-left'),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId('three-pane-gutter-right')).toBeInTheDocument();
  });

  it('double-clicking the gutter resets that side and clears its storage entry', async () => {
    window.localStorage.setItem(
      PANE_WIDTHS_STORAGE_KEY,
      JSON.stringify({ left: 305 }),
    );
    render(
      <ThreePaneLayout
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    const gutter = screen.getByTestId('three-pane-gutter-left');
    expect(gutter.getAttribute('aria-valuenow')).toBe('305');
    fireEvent.doubleClick(gutter);
    // Reset clears that side's stored entry; the remaining object becomes
    // empty so the key is removed entirely.
    expect(window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY)).toBeNull();
    expect(
      screen
        .getByTestId('three-pane-gutter-left')
        .getAttribute('aria-valuenow'),
    ).toBe('280');
  });

  it('keyboard nudging the left gutter persists the new width to localStorage', () => {
    render(
      <ThreePaneLayout
        left={<div>left content</div>}
        center={<div>center content</div>}
        right={<div>right content</div>}
      />,
    );
    const gutter = screen.getByTestId('three-pane-gutter-left');
    gutter.focus();
    fireEvent.keyDown(gutter, { key: 'ArrowRight' });
    const stored = JSON.parse(
      window.localStorage.getItem(PANE_WIDTHS_STORAGE_KEY) ?? '{}',
    ) as { left?: number };
    expect(stored.left).toBe(288);
  });
});
