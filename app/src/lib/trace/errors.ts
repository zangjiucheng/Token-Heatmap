/**
 * Typed error envelope produced by every trace loader so the UI can render
 * a single error surface regardless of the input source.
 *
 * `kind` discriminates the failure mode:
 *   - "parse"    — the input was not valid JSON
 *   - "validate" — the input parsed but did not match the trace schema
 *   - "network"  — fetch failed or the server returned a non-2xx status
 */

export interface ValidationIssue {
  /** JSON Pointer (RFC 6901) to the offending value, e.g. `/steps/0/entropy`. */
  pointer: string;
  /** Human-readable Ajv message, e.g. "must have required property 'entropy'". */
  message: string;
}

export type TraceLoadErrorKind = 'parse' | 'validate' | 'network';

export class TraceLoadError extends Error {
  override readonly name = 'TraceLoadError';
  readonly kind: TraceLoadErrorKind;
  readonly issues?: ValidationIssue[];
  readonly status?: number;

  private constructor(
    kind: TraceLoadErrorKind,
    message: string,
    extras: { issues?: ValidationIssue[]; status?: number; cause?: unknown } = {},
  ) {
    super(message);
    this.kind = kind;
    if (extras.issues) this.issues = extras.issues;
    if (extras.status !== undefined) this.status = extras.status;
    if (extras.cause !== undefined) {
      (this as { cause?: unknown }).cause = extras.cause;
    }
  }

  static parse(message = 'Failed to parse trace JSON', cause?: unknown): TraceLoadError {
    return new TraceLoadError('parse', message, { cause });
  }

  static validate(
    issues: ValidationIssue[],
    message = 'Trace failed schema validation',
  ): TraceLoadError {
    return new TraceLoadError('validate', message, { issues });
  }

  static network(message: string, status?: number, cause?: unknown): TraceLoadError {
    return new TraceLoadError('network', message, { status, cause });
  }
}

export function isTraceLoadError(value: unknown): value is TraceLoadError {
  return value instanceof TraceLoadError;
}
