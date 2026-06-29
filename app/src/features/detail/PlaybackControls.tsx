import { PlayIcon, PauseIcon } from '@/features/workspace/icons';
import './PlaybackControls.css';

export interface PlaybackControlsProps {
  playing: boolean;
  onToggle: () => void;
  speed: number;
  onCycleSpeed: () => void;
  /** Disabled when the trace is too short to play (e.g. a single step). */
  disabled?: boolean;
}

/**
 * Transport for the token timeline: play/pause loops `selectedStep` through the
 * active window so the lenses animate, and the speed chip cycles the rate.
 */
export function PlaybackControls({
  playing,
  onToggle,
  speed,
  onCycleSpeed,
  disabled = false,
}: PlaybackControlsProps) {
  return (
    <div
      className="playback-controls"
      role="group"
      aria-label="Token playback"
      data-testid="playback-controls"
    >
      <button
        type="button"
        className="playback-controls__play"
        onClick={onToggle}
        disabled={disabled}
        aria-label={playing ? 'Pause token playback' : 'Play token playback'}
        aria-pressed={playing}
        title={
          disabled
            ? 'Need at least two steps to play'
            : `${playing ? 'Pause' : 'Play'} (Space)`
        }
        data-testid="playback-toggle"
      >
        {playing ? <PauseIcon size={15} /> : <PlayIcon size={15} />}
      </button>
      <button
        type="button"
        className="playback-controls__speed"
        onClick={onCycleSpeed}
        disabled={disabled}
        aria-label={`Playback speed ${speed}×, click to change`}
        title="Playback speed"
        data-testid="playback-speed"
      >
        {speed}×
      </button>
    </div>
  );
}
