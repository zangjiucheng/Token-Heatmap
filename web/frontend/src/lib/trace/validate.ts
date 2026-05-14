import Ajv2020, {
  type ErrorObject,
  type ValidateFunction,
} from 'ajv/dist/2020';
import addFormats from 'ajv-formats';

import type { Trace } from '@/types/trace';
import bundledSchema from './trace.schema.json';
import { TraceLoadError, type ValidationIssue } from './errors';

/**
 * Chosen runtime validator: Ajv (Draft 2020-12) over Zod.
 *
 * Rationale: the trace schema is hand-authored as JSON Schema and lives in
 * docs/web/trace.schema.json so the Python producer and TypeScript consumer
 * share a single source of truth. Ajv
 * validates that schema directly with no second-source authoring step;
 * Zod would have required maintaining a parallel definition.
 */

let activeSchema: object = bundledSchema as object;
let cached: ValidateFunction<Trace> | null = null;

function build(schema: object): ValidateFunction<Trace> {
  const ajv = new Ajv2020({
    allErrors: true,
    strict: false,
    allowUnionTypes: true,
  });
  addFormats(ajv);
  return ajv.compile<Trace>(schema);
}

function getValidator(): ValidateFunction<Trace> {
  if (cached) return cached;
  cached = build(activeSchema);
  return cached;
}

/**
 * Replace the runtime schema used for validation.
 *
 * The frontend bundles a copy of the schema for offline use; on startup it
 * fetches `/schema` from the backend and swaps that copy in via this
 * function so the UI validates against whatever the server is currently
 * serving. If the fetch fails, the bundled copy is left in place.
 */
export function setActiveTraceSchema(schema: object): void {
  activeSchema = schema;
  cached = null;
}

/** Reset to the bundled schema. Primarily useful for tests. */
export function resetActiveTraceSchema(): void {
  activeSchema = bundledSchema as object;
  cached = null;
}

function toIssue(err: ErrorObject): ValidationIssue {
  // Ajv exposes the path as `instancePath` already in JSON Pointer form.
  // For `required` errors the pointer points at the parent, so append the
  // missing property name to make the pointer specific.
  let pointer = err.instancePath || '';
  if (err.keyword === 'required') {
    const missing = (err.params as { missingProperty?: string })
      .missingProperty;
    if (missing) pointer = `${pointer}/${missing}`;
  }
  return {
    pointer: pointer || '/',
    message: err.message ?? 'invalid value',
  };
}

/**
 * Validate an arbitrary value against the trace schema.
 *
 * Returns the value typed as `Trace` on success; throws a `TraceLoadError`
 * with `kind: "validate"` and JSON Pointer-tagged issues on failure.
 */
export function validateTrace(data: unknown): Trace {
  const validate = getValidator();
  if (validate(data)) {
    return data;
  }
  const issues = (validate.errors ?? []).map(toIssue);
  throw TraceLoadError.validate(issues);
}
