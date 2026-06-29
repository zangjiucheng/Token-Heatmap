import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { PlaybackControls } from './PlaybackControls';

describe('PlaybackControls', () => {
  it('shows a play affordance when paused and pause when playing', () => {
    const { rerender } = render(
      <PlaybackControls
        playing={false}
        onToggle={() => {}}
        speed={1}
        onCycleSpeed={() => {}}
      />,
    );
    expect(screen.getByLabelText('Play token playback')).toBeInTheDocument();

    rerender(
      <PlaybackControls
        playing
        onToggle={() => {}}
        speed={1}
        onCycleSpeed={() => {}}
      />,
    );
    const toggle = screen.getByTestId('playback-toggle');
    expect(toggle).toHaveAttribute('aria-label', 'Pause token playback');
    expect(toggle).toHaveAttribute('aria-pressed', 'true');
  });

  it('toggles play/pause and cycles speed on click', () => {
    const onToggle = vi.fn();
    const onCycleSpeed = vi.fn();
    render(
      <PlaybackControls
        playing={false}
        onToggle={onToggle}
        speed={2}
        onCycleSpeed={onCycleSpeed}
      />,
    );
    fireEvent.click(screen.getByTestId('playback-toggle'));
    expect(onToggle).toHaveBeenCalledTimes(1);

    const speed = screen.getByTestId('playback-speed');
    expect(speed).toHaveTextContent('2×');
    fireEvent.click(speed);
    expect(onCycleSpeed).toHaveBeenCalledTimes(1);
  });

  it('disables the controls when there is nothing to play', () => {
    render(
      <PlaybackControls
        playing={false}
        onToggle={() => {}}
        speed={1}
        onCycleSpeed={() => {}}
        disabled
      />,
    );
    expect(screen.getByTestId('playback-toggle')).toBeDisabled();
    expect(screen.getByTestId('playback-speed')).toBeDisabled();
  });
});
