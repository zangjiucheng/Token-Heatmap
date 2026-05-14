import { describe, expect, it } from 'vitest';
import { sampleColor } from './colormap';

describe('sampleColor (viridis)', () => {
  it('maps t=0 to the bottom of the scale', () => {
    expect(sampleColor(0)).toEqual([68, 1, 84]);
  });

  it('maps t=1 to the top of the scale', () => {
    expect(sampleColor(1)).toEqual([253, 254, 73]);
  });

  it('clamps t<0 to the bottom and t>1 to the top', () => {
    expect(sampleColor(-1)).toEqual([68, 1, 84]);
    expect(sampleColor(2)).toEqual([253, 254, 73]);
  });

  it('returns non-NaN values for NaN input', () => {
    const c = sampleColor(NaN);
    expect(c.every((v) => Number.isFinite(v))).toBe(true);
  });

  it('is monotonic in the perceptual lightness channel between endpoints', () => {
    // viridis maps low values to dark purple and high values to bright yellow,
    // so the sum R+G+B is monotonically non-decreasing across the scale.
    let prev = -Infinity;
    for (let i = 0; i <= 32; i += 1) {
      const t = i / 32;
      const [r, g, b] = sampleColor(t);
      const sum = r + g + b;
      expect(sum).toBeGreaterThanOrEqual(prev);
      prev = sum;
    }
  });

  it('interpolates linearly between adjacent stops', () => {
    const a = sampleColor(0);
    const b = sampleColor(1 / 255);
    const mid = sampleColor(0.5 / 255);
    for (let i = 0; i < 3; i += 1) {
      const expected = Math.round((a[i] + b[i]) / 2);
      expect(Math.abs(mid[i] - expected)).toBeLessThanOrEqual(1);
    }
  });
});
