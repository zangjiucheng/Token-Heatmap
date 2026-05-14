import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import sampleTrace from '@/lib/sample/trace.json';
import {
  loadSampleTrace,
  loadTraceFromFile,
  loadTraceFromUrl,
} from './load';
import { TraceLoadError, isTraceLoadError } from './errors';

function sampleJson(): string {
  return JSON.stringify(sampleTrace);
}

describe('loadTraceFromFile', () => {
  it('load_from_file_happy_path', async () => {
    const file = new File([sampleJson()], 'trace.json', { type: 'application/json' });
    const trace = await loadTraceFromFile(file);
    expect(trace.steps).toHaveLength(5);
  });

  it('rejects malformed JSON with kind=parse', async () => {
    const file = new File(['{not json'], 'broken.json', { type: 'application/json' });
    let caught: unknown;
    try {
      await loadTraceFromFile(file);
    } catch (err) {
      caught = err;
    }
    expect(isTraceLoadError(caught)).toBe(true);
    expect((caught as TraceLoadError).kind).toBe('parse');
  });
});

describe('loadTraceFromUrl', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('load_from_url_happy_path', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(sampleJson(), { status: 200 }),
    );
    const trace = await loadTraceFromUrl('/traces/sample.json');
    expect(trace.steps).toHaveLength(5);
  });

  it('load_from_url_network_error', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    let caught: unknown;
    try {
      await loadTraceFromUrl('/traces/sample.json');
    } catch (err) {
      caught = err;
    }
    expect(isTraceLoadError(caught)).toBe(true);
    const err = caught as TraceLoadError;
    expect(err.kind).toBe('network');
    expect(err.status).toBeUndefined();
  });

  it('load_from_url_bad_status', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('boom', { status: 500 }),
    );
    let caught: unknown;
    try {
      await loadTraceFromUrl('/traces/sample.json');
    } catch (err) {
      caught = err;
    }
    expect(isTraceLoadError(caught)).toBe(true);
    const err = caught as TraceLoadError;
    expect(err.kind).toBe('network');
    expect(err.status).toBe(500);
  });

  it('rejects malformed JSON body with kind=parse', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('not json', { status: 200 }),
    );
    let caught: unknown;
    try {
      await loadTraceFromUrl('/traces/sample.json');
    } catch (err) {
      caught = err;
    }
    expect(isTraceLoadError(caught)).toBe(true);
    expect((caught as TraceLoadError).kind).toBe('parse');
  });
});

describe('loadSampleTrace', () => {
  it('returns a valid bundled trace', async () => {
    const trace = await loadSampleTrace();
    expect(trace.schema_version).toBe('2.0.0');
    expect(trace.steps.length).toBeGreaterThan(0);
  });
});
