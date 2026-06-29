import { useCallback, useEffect, useRef, useState } from 'react';

/** Selectable playback rates, cycled by the speed control. */
export const PLAYBACK_SPEEDS = [0.5, 1, 2, 4] as const;

/** Step interval at 1× — chosen so changes are readable but the loop moves. */
const BASE_INTERVAL_MS = 700;

export interface UsePlaybackOptions {
  selectedStep: number | null;
  setSelectedStep: (step: number) => void;
  /** Inclusive [start, end] step window to loop over. */
  range: [number, number];
  /** When false, playback is impossible (e.g. a single-step trace). */
  enabled?: boolean;
}

export interface Playback {
  playing: boolean;
  speed: number;
  toggle: () => void;
  play: () => void;
  pause: () => void;
  cycleSpeed: () => void;
}

/**
 * Drive `selectedStep` like a media player: when playing, advance one step
 * every tick and loop back to the window start at the end. The interval reads
 * the current step / range from refs so it never re-arms mid-playback (smooth,
 * and the user can scrub or change the window without interrupting the loop).
 */
export function usePlayback({
  selectedStep,
  setSelectedStep,
  range,
  enabled = true,
}: UsePlaybackOptions): Playback {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const stepRef = useRef(selectedStep);
  stepRef.current = selectedStep;
  const rangeRef = useRef(range);
  rangeRef.current = range;

  // Nothing to play — make sure we don't sit in a "playing" state.
  useEffect(() => {
    if (!enabled && playing) setPlaying(false);
  }, [enabled, playing]);

  useEffect(() => {
    if (!playing || !enabled) return undefined;
    const intervalMs = Math.max(80, Math.round(BASE_INTERVAL_MS / speed));
    const id = window.setInterval(() => {
      const [start, end] = rangeRef.current;
      const cur = stepRef.current;
      const next = cur == null || cur >= end ? start : cur + 1;
      setSelectedStep(next);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [playing, enabled, speed, setSelectedStep]);

  const play = useCallback(() => {
    if (enabled) setPlaying(true);
  }, [enabled]);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(
    () => setPlaying((p) => (enabled ? !p : false)),
    [enabled],
  );
  const cycleSpeed = useCallback(() => {
    setSpeed((sp) => {
      const i = PLAYBACK_SPEEDS.indexOf(sp as (typeof PLAYBACK_SPEEDS)[number]);
      return PLAYBACK_SPEEDS[(i + 1) % PLAYBACK_SPEEDS.length];
    });
  }, []);

  return { playing, speed, toggle, play, pause, cycleSpeed };
}
