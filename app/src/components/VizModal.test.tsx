import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { VizModal } from './VizModal';

function open() {
  const onClose = vi.fn();
  const utils = render(
    <VizModal open onClose={onClose} title="Test plot" aspect={2}>
      <svg data-testid="payload" viewBox="0 0 100 50" />
    </VizModal>,
  );
  return { onClose, ...utils };
}

describe('VizModal', () => {
  it('renders nothing when closed', () => {
    render(
      <VizModal open={false} onClose={() => {}} title="Hidden" aspect={1}>
        <svg data-testid="payload" />
      </VizModal>,
    );
    expect(screen.queryByTestId('viz-modal')).toBeNull();
  });

  it('renders the dialog, title and payload when open', () => {
    open();
    const modal = screen.getByTestId('viz-modal');
    expect(modal).toHaveAttribute('role', 'dialog');
    expect(screen.getByText('Test plot')).toBeInTheDocument();
    expect(screen.getByTestId('payload')).toBeInTheDocument();
  });

  it('zooms in, resets, and clamps at the 100% floor', async () => {
    const user = userEvent.setup();
    open();
    expect(screen.getByTestId('viz-modal-zoom')).toHaveTextContent('100%');
    await user.click(screen.getByLabelText('Zoom in'));
    expect(screen.getByTestId('viz-modal-zoom')).toHaveTextContent('125%');
    await user.click(screen.getByRole('button', { name: 'Reset' }));
    expect(screen.getByTestId('viz-modal-zoom')).toHaveTextContent('100%');
    await user.click(screen.getByLabelText('Zoom out'));
    expect(screen.getByTestId('viz-modal-zoom')).toHaveTextContent('100%');
  });

  it('closes via the ✕ button', async () => {
    const user = userEvent.setup();
    const { onClose } = open();
    await user.click(screen.getByTestId('viz-modal-close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes on Escape', async () => {
    const user = userEvent.setup();
    const { onClose } = open();
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalled();
  });
});
