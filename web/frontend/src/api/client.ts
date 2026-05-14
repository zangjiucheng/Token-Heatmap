/**
 * Hand-written, typed wrapper around the FastAPI backend.
 *
 * The generated client at `src/api/generated/` carries the request/response
 * types and a Cancelable promise abstraction. We don't use its
 * `Service.fetch()` directly because it predates `AbortSignal` integration in
 * a way that's awkward to test; a thin `fetch` wrapper here keeps abort
 * semantics standard and the error mapping pinned in one place.
 */

import type { Trace } from '@/types/trace';
import { TraceLoadError } from '@/lib/trace/errors';
import { validateTrace } from '@/lib/trace/validate';
import { mapBackendError, mapTransportError } from './errors';

export interface ClientOptions {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

const DEFAULT_BASE_URL = 'http://localhost:8000';

function resolveBaseUrl(explicit?: string): string {
  if (explicit) return stripTrailingSlash(explicit);
  // `import.meta.env` is only defined in a Vite/build context. Fall back to
  // the documented dev default so unit tests that import this module
  // outside Vite don't crash.
  const meta = import.meta as unknown as {
    env?: { VITE_API_BASE_URL?: string };
  };
  const fromEnv = meta?.env?.VITE_API_BASE_URL;
  return stripTrailingSlash(fromEnv ?? DEFAULT_BASE_URL);
}

function stripTrailingSlash(url: string): string {
  return url.endsWith('/') ? url.slice(0, -1) : url;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = resolveBaseUrl(options.baseUrl);
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  private url(path: string): string {
    return `${this.baseUrl}${path.startsWith('/') ? '' : '/'}${path}`;
  }

  /** `GET /health` — returns true iff the backend reports `{status: "ok"}`. */
  async health(options: RequestOptions = {}): Promise<boolean> {
    let response: Response;
    try {
      response = await this.fetchImpl(this.url('/health'), {
        method: 'GET',
        signal: options.signal,
      });
    } catch (cause) {
      throw mapTransportError(cause);
    }
    if (!response.ok) {
      const body = await readJson(response);
      throw mapBackendError(response.status, body);
    }
    const body = (await readJson(response)) as { status?: string } | null;
    return body?.status === 'ok';
  }

  /**
   * `GET /schema` — fetch the canonical trace JSON Schema.
   *
   * Returns the schema object as-is; the caller is responsible for swapping
   * the bundled copy with this one.
   */
  async getSchema(options: RequestOptions = {}): Promise<object> {
    let response: Response;
    try {
      response = await this.fetchImpl(this.url('/schema'), {
        method: 'GET',
        signal: options.signal,
      });
    } catch (cause) {
      throw mapTransportError(cause);
    }
    if (!response.ok) {
      const body = await readJson(response);
      throw mapBackendError(response.status, body);
    }
    const body = await readJson(response);
    if (typeof body !== 'object' || body === null) {
      throw TraceLoadError.network('Schema response was not a JSON object');
    }
    return body as object;
  }

  /**
   * `POST /trace/convert-csv` — multipart upload of a CSV produced by
   * `trace_to_dataframe`. Returns the validated JSON trace.
   */
  async convertCsv(file: File, options: RequestOptions = {}): Promise<Trace> {
    const form = new FormData();
    form.append('file', file, file.name);
    let response: Response;
    try {
      response = await this.fetchImpl(this.url('/trace/convert-csv'), {
        method: 'POST',
        body: form,
        signal: options.signal,
      });
    } catch (cause) {
      throw mapTransportError(cause);
    }
    if (!response.ok) {
      const body = await readJson(response);
      throw mapBackendError(response.status, body);
    }
    const body = await readJson(response);
    return validateTrace(body);
  }
}

let defaultClient: ApiClient | null = null;

/** Return a process-wide singleton client built from `VITE_API_BASE_URL`. */
export function getApiClient(): ApiClient {
  if (!defaultClient) {
    defaultClient = new ApiClient();
  }
  return defaultClient;
}

/** Test-only: replace the singleton. */
export function setApiClientForTests(client: ApiClient | null): void {
  defaultClient = client;
}
