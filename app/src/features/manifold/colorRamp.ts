/** Perceptually-uniform viridis-like ramp, eight stops — matched to the
 * activation/attention heatmaps so colour reads consistently across tabs. */
const COLOR_RAMP = [
  '#440154',
  '#482878',
  '#3e4989',
  '#31688e',
  '#26828e',
  '#1f9e89',
  '#35b779',
  '#fde725',
] as const;

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

/**
 * Continuous viridis colour for ``t`` in [0, 1], linearly interpolating between
 * the eight ramp stops. Used to colour scatter points by generation step.
 */
export function rampColor(t: number): string {
  const x = clamp01(t) * (COLOR_RAMP.length - 1);
  const i = Math.floor(x);
  if (i >= COLOR_RAMP.length - 1) return COLOR_RAMP[COLOR_RAMP.length - 1];
  const f = x - i;
  const [r1, g1, b1] = hexToRgb(COLOR_RAMP[i]);
  const [r2, g2, b2] = hexToRgb(COLOR_RAMP[i + 1]);
  const r = Math.round(r1 + (r2 - r1) * f);
  const g = Math.round(g1 + (g2 - g1) * f);
  const b = Math.round(b1 + (b2 - b1) * f);
  return `rgb(${r}, ${g}, ${b})`;
}
