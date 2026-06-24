/**
 * Hand-rolled orthographic 3-D projection for the manifold scatter. Keeps the
 * project's no-chart-library approach: rotate the (PC1, PC2, PC3) cloud by a
 * yaw/pitch and project to 2-D SVG space, preserving true geometry (one uniform
 * scale across all three axes so a helix stays a helix). Pure + testable.
 */

export const VIEW_W = 480;
export const VIEW_H = 400;
export const VIEW_PAD = 46;

export interface ProjectedPoint {
  index: number;
  px: number;
  py: number;
  /** Camera-space depth normalised to ~[-1, 1]; larger = nearer the viewer. */
  depth: number;
}

export interface ProjectedAxis {
  label: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  depth: number;
}

export interface Projection {
  points: ProjectedPoint[];
  axes: ProjectedAxis[];
}

/** Rotate a point by yaw (around Y) then pitch (around X). */
export function rotate3d(
  x: number,
  y: number,
  z: number,
  yaw: number,
  pitch: number,
): [number, number, number] {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const x1 = x * cy + z * sy;
  const z1 = -x * sy + z * cy;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;
  return [x1, y2, z2];
}

export interface ProjectOptions {
  w?: number;
  h?: number;
  pad?: number;
}

/**
 * Project a cloud's first three components into 2-D screen space for the given
 * rotation. Centers on the centroid and scales uniformly so the shape is not
 * distorted.
 */
export function projectManifold(
  coords: number[][],
  yaw: number,
  pitch: number,
  options: ProjectOptions = {},
): Projection {
  const w = options.w ?? VIEW_W;
  const h = options.h ?? VIEW_H;
  const pad = options.pad ?? VIEW_PAD;

  const n = coords.length || 1;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const row of coords) {
    cx += row[0] ?? 0;
    cy += row[1] ?? 0;
    cz += row[2] ?? 0;
  }
  cx /= n;
  cy /= n;
  cz /= n;

  let maxR = 1e-9;
  for (const row of coords) {
    maxR = Math.max(
      maxR,
      Math.abs((row[0] ?? 0) - cx),
      Math.abs((row[1] ?? 0) - cy),
      Math.abs((row[2] ?? 0) - cz),
    );
  }

  const half = Math.min(w, h) / 2 - pad;
  const scale = half / maxR;
  const ox = w / 2;
  const oy = h / 2;

  const points: ProjectedPoint[] = coords.map((row, index) => {
    const [rx, ry, rz] = rotate3d(
      (row[0] ?? 0) - cx,
      (row[1] ?? 0) - cy,
      (row[2] ?? 0) - cz,
      yaw,
      pitch,
    );
    return {
      index,
      px: ox + rx * scale,
      py: oy - ry * scale,
      depth: rz / maxR,
    };
  });

  const axisDefs: Array<[string, number, number, number]> = [
    ['PC1', maxR, 0, 0],
    ['PC2', 0, maxR, 0],
    ['PC3', 0, 0, maxR],
  ];
  const axes: ProjectedAxis[] = axisDefs.map(([label, ax, ay, az]) => {
    const [rx, ry, rz] = rotate3d(ax, ay, az, yaw, pitch);
    return {
      label,
      x1: ox,
      y1: oy,
      x2: ox + rx * scale,
      y2: oy - ry * scale,
      depth: rz / maxR,
    };
  });

  return { points, axes };
}
