/**
 * Maps the backend's structured error envelope into the UI's
 * `TraceLoadError` taxonomy.
 *
 * Backend envelope shape (see `web/backend/llm_token_heatmap_api/errors.py`):
 *
 *   { "error": { "kind": "<enum>", "message": "...", "details": ... } }
 *
 * The mapping is intentionally narrow: only `kind`s the API can emit are
 * handled by name; anything unrecognized becomes a generic network error so
 * the UI degrades gracefully rather than throwing an opaque `Error`.
 */

import { TraceLoadError } from '@/lib/trace/errors';

export type BackendErrorKind =
  | 'request_too_large'
  | 'timeout'
  | 'invalid_csv'
  | 'invalid_params'
  | 'model_load_failed'
  | 'generation_failed'
  | 'schema_unavailable'
  | 'http_error'
  | 'internal_error';

export interface BackendErrorBody {
  kind: BackendErrorKind | string;
  message: string;
  details?: unknown;
}

export interface BackendErrorEnvelope {
  error: BackendErrorBody;
}

export function isBackendErrorEnvelope(value: unknown): value is BackendErrorEnvelope {
  return (
    typeof value === 'object' &&
    value !== null &&
    'error' in value &&
    typeof (value as { error: unknown }).error === 'object' &&
    (value as { error: { kind?: unknown } }).error !== null &&
    typeof (value as { error: { kind?: unknown } }).error.kind === 'string' &&
    typeof (value as { error: { message?: unknown } }).error.message === 'string'
  );
}

/**
 * Pydantic invalid_params details are an array of `{loc, msg, ...}` records.
 * Pull the first field name out of `loc` (skipping the leading "body"/"query"
 * scope) so the UI can highlight it on the form.
 */
function fieldFromValidationDetails(details: unknown): string | undefined {
  if (!Array.isArray(details) || details.length === 0) return undefined;
  const first = details[0] as { loc?: unknown };
  if (!Array.isArray(first.loc)) return undefined;
  for (let i = first.loc.length - 1; i >= 0; i--) {
    const part = first.loc[i];
    if (typeof part === 'string' && part !== 'body' && part !== 'query') {
      return part;
    }
  }
  return undefined;
}

/**
 * Map an HTTP status + JSON body to a `TraceLoadError`.
 *
 *   request_too_large / invalid_csv / invalid_params → kind: "validate"
 *   timeout                                          → kind: "network" with status
 *   anything else / unparseable body                 → kind: "network"
 */
export function mapBackendError(status: number, body: unknown): TraceLoadError {
  if (isBackendErrorEnvelope(body)) {
    const kind = body.error.kind;
    const message = body.error.message;

    if (kind === 'request_too_large') {
      const details = body.error.details as { field?: string } | undefined;
      const field = details?.field ?? 'max_new_tokens';
      return TraceLoadError.validate(
        [{ pointer: `/${field}`, message }],
        message,
      );
    }
    if (kind === 'invalid_csv') {
      return TraceLoadError.validate([{ pointer: '/file', message }], message);
    }
    if (kind === 'invalid_params') {
      const field = fieldFromValidationDetails(body.error.details);
      return TraceLoadError.validate(
        [{ pointer: field ? `/${field}` : '/', message }],
        message,
      );
    }
    return TraceLoadError.network(message, status);
  }
  return TraceLoadError.network(
    `Request failed with status ${status}`,
    status,
  );
}

export function mapTransportError(cause: unknown): TraceLoadError {
  if (cause instanceof DOMException && cause.name === 'AbortError') {
    return TraceLoadError.network('Request aborted', undefined, cause);
  }
  const message =
    cause instanceof Error ? `Network error: ${cause.message}` : 'Network error';
  return TraceLoadError.network(message, undefined, cause);
}
