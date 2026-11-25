import type { Trace } from '@/types/trace';
import { TraceLoadError } from './errors';
import type { GunzipResponse } from './gunzip.worker';
import { validateTrace } from './validate';

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch (cause) {
    const message =
      cause instanceof Error
        ? `Malformed JSON: ${cause.message}`
        : 'Malformed JSON';
    throw TraceLoadError.parse(message, cause);
  }
}

/**
 * True when the dropped file is a gzipped trace. The CLI writes
 * `adaptive_token_trace.json.gz` by default — a 2048-step run is 197 MB of JSON
 * and 34 MB gzipped — so the viewer has to accept both encodings. Sniffing the
 * two-byte DEFLATE magic (0x1f 0x8b) rather than trusting the filename means a
 * renamed or extension-less file still opens.
 */
async function isGzip(file: File): Promise<boolean> {
  const slice = file.slice(0, 2);
  // Not every Blob implementation exposes arrayBuffer() on a slice (jsdom does
  // not). Fall back to the extension there rather than throwing.
  if (typeof slice.arrayBuffer !== 'function') {
    return /\.gz$/i.test(file.name);
  }
  const head = new Uint8Array(await slice.arrayBuffer());
  return head.length === 2 && head[0] === 0x1f && head[1] === 0x8b;
}

/**
 * Whole-file bytes. Blob.arrayBuffer() is the fast path; FileReader is the
 * fallback for environments whose Blob lacks it (jsdom, so the tests below
 * exercise the real decode path rather than a stub).
 */
function fileBytes(file: File): Promise<Uint8Array<ArrayBuffer>> {
  if (typeof file.arrayBuffer === 'function') {
    return file.arrayBuffer().then((b) => new Uint8Array(b));
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Decompress in a worker so the main thread keeps painting. Only the raw bytes
 * cross the boundary, and they cross as a transferable (zero copy) — see
 * gunzip.worker.ts for the measurements that forced this shape. Resolves to
 * null when a worker cannot be used, so the caller falls back to inline.
 */
function gunzipInWorker(bytes: Uint8Array<ArrayBuffer>): Promise<string | null> {
  if (typeof Worker === 'undefined') return Promise.resolve(null);
  let worker: Worker;
  try {
    worker = new Worker(new URL('./gunzip.worker.ts', import.meta.url), {
      type: 'module',
    });
  } catch {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    worker.onmessage = (event: MessageEvent<GunzipResponse>) => {
      worker.terminate();
      const data = event.data;
      resolve(data.ok ? new TextDecoder().decode(data.buffer) : null);
    };
    // A worker that fails to boot (bundler quirk, CSP) must not wedge the load;
    // fall back rather than hang.
    const fallBack = () => {
      worker.terminate();
      resolve(null);
    };
    worker.onerror = fallBack;
    worker.onmessageerror = fallBack;
    const buffer = bytes.buffer;
    try {
      worker.postMessage({ buffer }, [buffer]);
    } catch {
      fallBack();
    }
  });
}

async function gunzipToText(file: File): Promise<string> {
  if (typeof DecompressionStream === 'undefined') {
    throw TraceLoadError.parse(
      'This trace is gzipped, but this browser has no DecompressionStream. ' +
        'Re-run the trace with --no-compress, or open it in the desktop app.',
    );
  }
  try {
    const bytes = await fileBytes(file);

    const offThread = await gunzipInWorker(bytes);
    if (offThread !== null) return offThread;

    // A successful transfer detaches the original buffer even if the worker
    // fails afterward. Read the file again before trying inline decompression.
    const fallbackBytes = bytes.byteLength === 0 ? await fileBytes(file) : bytes;
    // Fallback: same work, on this thread. Feed a stream we construct ourselves
    // rather than file.stream() — the latter is missing on some Blob
    // implementations, and we need the whole payload in memory to JSON.parse it
    // regardless.
    const source = new ReadableStream<BufferSource>({
      start(controller) {
        controller.enqueue(fallbackBytes);
        controller.close();
      },
    });
    return await new Response(
      source.pipeThrough(new DecompressionStream('gzip')),
    ).text();
  } catch (cause) {
    if (cause instanceof TraceLoadError) throw cause;
    throw TraceLoadError.parse(
      'Could not decompress this .gz trace — the file may be truncated.',
      cause,
    );
  }
}

/** Read, parse, and validate a user-dropped local trace (.json or .json.gz). */
export async function loadTraceFromFile(file: File): Promise<Trace> {
  const text = (await isGzip(file)) ? await gunzipToText(file) : await file.text();
  const data = parseJson(text);
  return validateTrace(data);
}

/**
 * Return the bundled sample trace used in dev and on the "Try sample data"
 * empty-state action. Dynamically imported so the ~400 kB showcase trace lands
 * in a lazily-fetched chunk instead of the initial bundle — it's only loaded
 * when someone actually asks for it. Validates on every call so a drift between
 * the sample and the schema fails fast rather than silently feeding malformed
 * data to the UI.
 */
export async function loadSampleTrace(): Promise<Trace> {
  const { default: sampleTrace } = await import('@/lib/sample/trace.json');
  return validateTrace(sampleTrace);
}
