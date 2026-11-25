/**
 * Off-main-thread gunzip for trace files.
 *
 * Why only gunzip, and not the whole load: the parsed trace cannot be handed
 * back cheaply. Measured on a real 4096-step trace (83 MB gzipped, 444 MB of
 * JSON):
 *
 *     gunzip                              1407 ms
 *     JSON.parse                          2366 ms
 *     structuredClone(parsed object)      8831 ms   <-- posting the result back
 *     transfer ArrayBuffer + TextDecoder   253 ms
 *
 * A worker that parses and posts the Trace back would spend more time cloning
 * the object across the boundary than the entire single-threaded path costs.
 * So the worker does the decompression and returns the raw UTF-8 bytes as a
 * *transferable* ArrayBuffer, which moves with no copy at all; the main thread
 * decodes and parses. That takes the 1.4 s decompression off the main thread
 * and keeps the loading UI painting instead of locking up.
 */

export interface GunzipRequest {
  buffer: ArrayBuffer;
}

export type GunzipResponse =
  | { ok: true; buffer: ArrayBuffer }
  | { ok: false; message: string };

self.onmessage = async (event: MessageEvent<GunzipRequest>) => {
  try {
    const stream = new Blob([event.data.buffer])
      .stream()
      .pipeThrough(new DecompressionStream('gzip'));
    const out = await new Response(stream).arrayBuffer();
    const reply: GunzipResponse = { ok: true, buffer: out };
    // Second argument transfers ownership — no copy, and `out` becomes
    // unusable here, which is exactly what we want.
    (self as unknown as Worker).postMessage(reply, [out]);
  } catch (err) {
    const reply: GunzipResponse = {
      ok: false,
      message: err instanceof Error ? err.message : 'gunzip failed',
    };
    (self as unknown as Worker).postMessage(reply);
  }
};
