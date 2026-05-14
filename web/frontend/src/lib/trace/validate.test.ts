import { describe, expect, it } from 'vitest';

import sampleTrace from '@/lib/sample/trace.json';
import { TraceLoadError, isTraceLoadError } from './errors';
import { validateTrace } from './validate';

function cloneSample(): Record<string, unknown> {
  return JSON.parse(JSON.stringify(sampleTrace));
}

describe('validateTrace', () => {
  it('sample_trace_validates_against_schema', () => {
    const trace = validateTrace(cloneSample());
    expect(trace.schema_version).toBe('2.0.0');
    expect(trace.steps).toHaveLength(5);
  });

  it('validate_rejects_missing_required_fields', () => {
    const trace = cloneSample() as {
      steps: Array<{ raw: Record<string, unknown> }>;
    };
    delete trace.steps[0].raw.entropy;

    let caught: unknown;
    try {
      validateTrace(trace);
    } catch (err) {
      caught = err;
    }

    expect(isTraceLoadError(caught)).toBe(true);
    const err = caught as TraceLoadError;
    expect(err.kind).toBe('validate');
    expect(err.issues).toBeDefined();
    const pointers = err.issues!.map((i) => i.pointer);
    expect(pointers).toContain('/steps/0/raw/entropy');
  });

  it('validate_rejects_wrong_types', () => {
    const trace = cloneSample() as {
      steps: Array<{ raw: { entropy: unknown } }>;
    };
    trace.steps[0].raw.entropy = 'high';

    let caught: unknown;
    try {
      validateTrace(trace);
    } catch (err) {
      caught = err;
    }

    expect(isTraceLoadError(caught)).toBe(true);
    const err = caught as TraceLoadError;
    expect(err.kind).toBe('validate');
    const pointers = err.issues!.map((i) => i.pointer);
    expect(pointers).toContain('/steps/0/raw/entropy');
  });
});
