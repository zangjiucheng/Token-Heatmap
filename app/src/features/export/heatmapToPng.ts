/**
 * Composite all `<canvas>` elements inside `root` into a single PNG and return
 * it as a Blob. Canvases are drawn in DOM order so the overlay (selection
 * highlight, hover crosshair) sits on top of the data canvas, matching what
 * the user sees on screen.
 *
 * Returns `null` when no canvases are present or when the browser cannot
 * convert to a Blob.
 */
export async function heatmapToPng(root: HTMLElement): Promise<Blob | null> {
  const canvases = Array.from(
    root.querySelectorAll<HTMLCanvasElement>('canvas'),
  );
  if (canvases.length === 0) return null;

  // Use the first canvas as the size reference; downstream canvases are
  // layered on top at their own positions relative to it.
  const baseRect = canvases[0].getBoundingClientRect();
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const out = document.createElement('canvas');
  out.width = Math.max(1, Math.floor(baseRect.width * dpr));
  out.height = Math.max(1, Math.floor(baseRect.height * dpr));
  const ctx = out.getContext('2d');
  if (!ctx) return null;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, baseRect.width, baseRect.height);

  for (const canvas of canvases) {
    const rect = canvas.getBoundingClientRect();
    const dx = rect.left - baseRect.left;
    const dy = rect.top - baseRect.top;
    ctx.drawImage(canvas, dx, dy, rect.width, rect.height);
  }

  return await new Promise<Blob | null>((resolve) => {
    out.toBlob((blob) => resolve(blob), 'image/png');
  });
}

/** Trigger a browser download for `blob` with the given filename. */
export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Allow the browser to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
