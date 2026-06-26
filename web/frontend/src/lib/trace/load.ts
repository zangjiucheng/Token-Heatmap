import type { Trace } from '@/types/trace';
import sampleTrace from '@/lib/sample/trace.json';
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

/** Fetch, parse, and validate a trace from a URL. */
export async function loadTraceFromUrl(url: string): Promise<Trace> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (cause) {
    const message =
      cause instanceof Error
        ? `Network error: ${cause.message}`
        : 'Network error';
    throw TraceLoadError.network(message, undefined, cause);
  }

  if (!response.ok) {
    throw TraceLoadError.network(
      `Request failed with status ${response.status}`,
      response.status,
    );
  }

  let text: string;
  try {
    text = await response.text();
  } catch (cause) {
    const message =
      cause instanceof Error
        ? `Failed to read response body: ${cause.message}`
        : 'Failed to read response body';
    throw TraceLoadError.network(message, response.status, cause);
  }

  const data = parseJson(text);
  return validateTrace(data);
}

/**
 * Return the bundled minimal sample trace used in Storybook, dev, and tests.
 * Validates on every call so a drift between the sample and the schema fails
 * fast rather than silently feeding malformed data to the UI.
 */
export async function loadSampleTrace(): Promise<Trace> {
  return validateTrace(sampleTrace);
}
