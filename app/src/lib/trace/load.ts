import type { Trace } from '@/types/trace';
import { TraceLoadError } from './errors';
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

/** Read, parse, and validate a user-dropped local JSON file. */
export async function loadTraceFromFile(file: File): Promise<Trace> {
  const text = await file.text();
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
