import { describe, expect, it } from 'vitest';
import {
  projectManifold,
  rotate3d,
  VIEW_H,
  VIEW_W,
} from './manifold3d';

describe('rotate3d', () => {
  it('is the identity at zero rotation', () => {
    expect(rotate3d(1, 2, 3, 0, 0)).toEqual([1, 2, 3]);
  });

  it('rotates the x axis onto -z for a quarter yaw turn', () => {
    const [x, y, z] = rotate3d(1, 0, 0, Math.PI / 2, 0);
    expect(x).toBeCloseTo(0, 6);
    expect(y).toBeCloseTo(0, 6);
    expect(z).toBeCloseTo(-1, 6);
  });

  it('returns to the original point after a full turn', () => {
    const [x, y, z] = rotate3d(0.4, -0.7, 1.2, Math.PI * 2, Math.PI * 2);
    expect(x).toBeCloseTo(0.4, 6);
    expect(y).toBeCloseTo(-0.7, 6);
    expect(z).toBeCloseTo(1.2, 6);
  });
});

describe('projectManifold', () => {
  const cloud = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];

  it('projects the centroid to the view centre', () => {
    const { points } = projectManifold(cloud, 0, 0);
    // This symmetric cloud is centred on the origin; the +x/-x pair must
    // straddle the horizontal centre.
    const cxAvg = points.reduce((s, p) => s + p.px, 0) / points.length;
    const cyAvg = points.reduce((s, p) => s + p.py, 0) / points.length;
    expect(cxAvg).toBeCloseTo(VIEW_W / 2, 4);
    expect(cyAvg).toBeCloseTo(VIEW_H / 2, 4);
  });

  it('keeps every projected point inside the viewport', () => {
    const { points } = projectManifold(cloud, 0.7, 0.4);
    for (const p of points) {
      expect(p.px).toBeGreaterThanOrEqual(0);
      expect(p.px).toBeLessThanOrEqual(VIEW_W);
      expect(p.py).toBeGreaterThanOrEqual(0);
      expect(p.py).toBeLessThanOrEqual(VIEW_H);
      expect(p.depth).toBeGreaterThanOrEqual(-1.001);
      expect(p.depth).toBeLessThanOrEqual(1.001);
    }
  });

  it('changes the screen x of a point when yaw changes', () => {
    const a = projectManifold(cloud, 0, 0).points[0].px;
    const b = projectManifold(cloud, 0.9, 0).points[0].px;
    expect(a).not.toBeCloseTo(b, 2);
  });

  it('emits three labelled axes', () => {
    const { axes } = projectManifold(cloud, 0.2, 0.2);
    expect(axes.map((a) => a.label)).toEqual(['PC1', 'PC2', 'PC3']);
  });
});
