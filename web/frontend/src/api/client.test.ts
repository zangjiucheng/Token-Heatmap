import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import sampleTrace from '@/lib/sample/trace.json';
import { ApiClient } from './client';
import { TraceLoadError } from '@/lib/trace/errors';
import { resetActiveTraceSchema } from '@/lib/trace/validate';

function makeJsonResponse(body: unknown, status = 200, headers: HeadersInit = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

describe('ApiClient', () => {
  beforeEach(() => {
    resetActiveTraceSchema();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('strips a trailing slash from the base URL', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(makeJsonResponse({ status: 'ok' }));
    const client = new ApiClient({ baseUrl: 'http://api/', fetchImpl });
    await client.health();
    expect(fetchImpl.mock.calls[0]?.[0]).toBe('http://api/health');
  });

  it('maps network failures to TraceLoadError(network)', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    const client = new ApiClient({ baseUrl: 'http://api', fetchImpl });
    const file = new File(['x'], 'trace.csv', { type: 'text/csv' });
    await expect(client.convertCsv(file)).rejects.toMatchObject({ kind: 'network' });
  });

  it('forwards AbortSignal.aborted into the fetch call', async () => {
    const controller = new AbortController();
    const fetchImpl = vi.fn(async (_url: URL | RequestInfo, init?: RequestInit) => {
      if (init?.signal && (init.signal as AbortSignal).aborted) {
        throw new DOMException('aborted', 'AbortError');
      }
      controller.abort();
      throw new DOMException('aborted', 'AbortError');
    });
    const client = new ApiClient({
      baseUrl: 'http://api',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(client.health({ signal: controller.signal })).rejects.toMatchObject({
      kind: 'network',
    });
    const [, init] = fetchImpl.mock.calls[0] as [unknown, RequestInit];
    expect(init.signal).toBe(controller.signal);
  });

  it('health returns true when /health responds {status: "ok"}', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(makeJsonResponse({ status: 'ok' }));
    const client = new ApiClient({ baseUrl: 'http://api', fetchImpl });
    expect(await client.health()).toBe(true);
  });

  it('health surfaces non-2xx as a backend error', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      makeJsonResponse({ error: { kind: 'internal_error', message: 'down' } }, 500),
    );
    const client = new ApiClient({ baseUrl: 'http://api', fetchImpl });
    await expect(client.health()).rejects.toBeInstanceOf(TraceLoadError);
  });

  it('convertCsv posts a multipart form to /trace/convert-csv', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(makeJsonResponse(sampleTrace));
    const client = new ApiClient({ baseUrl: 'http://api', fetchImpl });
    const file = new File(['x'], 'trace.csv', { type: 'text/csv' });
    const result = await client.convertCsv(file);
    expect(result.steps).toHaveLength(5);
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api/trace/convert-csv');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
  });

  it('getSchema returns the JSON object from /schema and lets validation succeed against it', async () => {
    const schema = { type: 'object' };
    const fetchImpl = vi.fn().mockResolvedValue(makeJsonResponse(schema));
    const client = new ApiClient({ baseUrl: 'http://api', fetchImpl });
    expect(await client.getSchema()).toEqual(schema);
  });
});
