import { describe, expect, it } from 'vitest';

import sampleTrace from '@/lib/sample/trace.json';
import { loadSampleTrace, loadTraceFromFile } from './load';
import { TraceLoadError, isTraceLoadError } from './errors';

function sampleJson(): string {
  return JSON.stringify(sampleTrace);
}

describe('loadTraceFromFile', () => {
  it('load_from_file_happy_path', async () => {
    const file = new File([sampleJson()], 'trace.json', {
      type: 'application/json',
    });
    const trace = await loadTraceFromFile(file);
    expect(trace.steps).toHaveLength(5);
  });

  it('loads a gzipped .json.gz trace', async () => {
    // The CLI gzips by default, so this is the common case, not an edge case.
    // Build a real gzip stream so the test exercises DecompressionStream rather
    // than a stub.
    const { gzipSync } = await import('node:zlib');
    const bytes = new Uint8Array(gzipSync(Buffer.from(sampleJson())));
    expect(bytes[0]).toBe(0x1f);
    expect(bytes[1]).toBe(0x8b);

    const file = new File([bytes], 'adaptive_token_trace.json.gz', {
      type: 'application/gzip',
    });
    const trace = await loadTraceFromFile(file);
    expect(trace.steps).toHaveLength(5);
  });

  it('rejects malformed JSON with kind=parse', async () => {
    const file = new File(['{not json'], 'broken.json', {
      type: 'application/json',
    });
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

describe('loadSampleTrace', () => {
  it('returns a valid bundled trace', async () => {
    const trace = await loadSampleTrace();
    expect(trace.schema_version).toBe('2.0.0');
    expect(trace.steps.length).toBeGreaterThan(0);
  });
});
