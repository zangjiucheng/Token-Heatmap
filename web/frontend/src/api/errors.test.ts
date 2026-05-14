import { describe, expect, it } from 'vitest';

import { mapBackendError, mapTransportError, isBackendErrorEnvelope } from './errors';

describe('mapBackendError', () => {
  it('maps request_too_large into a validate error with the field pointer', () => {
    const err = mapBackendError(422, {
      error: {
        kind: 'request_too_large',
        message: 'max_new_tokens too large',
        details: { field: 'max_new_tokens', limit: 256, value: 999 },
      },
    });
    expect(err.kind).toBe('validate');
    expect(err.issues).toEqual([
      { pointer: '/max_new_tokens', message: 'max_new_tokens too large' },
    ]);
  });

  it('maps invalid_params into a validate error using loc[last] as the field', () => {
    const err = mapBackendError(422, {
      error: {
        kind: 'invalid_params',
        message: 'Request validation failed.',
        details: [
          {
            loc: ['body', 'temperature'],
            msg: 'must be greater than 0',
            type: 'greater_than',
          },
        ],
      },
    });
    expect(err.kind).toBe('validate');
    expect(err.issues?.[0]?.pointer).toBe('/temperature');
  });

  it('maps timeout to a network error with status preserved', () => {
    const err = mapBackendError(504, {
      error: { kind: 'timeout', message: 'Generation timed out.' },
    });
    expect(err.kind).toBe('network');
    expect(err.status).toBe(504);
    expect(err.message).toBe('Generation timed out.');
  });

  it('falls back to a generic network error when the body has no envelope', () => {
    const err = mapBackendError(500, 'plain text body');
    expect(err.kind).toBe('network');
    expect(err.status).toBe(500);
  });
});

describe('mapTransportError', () => {
  it('treats AbortError as a network failure with aborted message', () => {
    const cause = new DOMException('aborted', 'AbortError');
    const err = mapTransportError(cause);
    expect(err.kind).toBe('network');
    expect(err.message).toMatch(/abort/i);
  });

  it('wraps generic errors with the prefix "Network error:"', () => {
    const err = mapTransportError(new Error('Failed to fetch'));
    expect(err.kind).toBe('network');
    expect(err.message).toMatch(/Network error/);
  });
});

describe('isBackendErrorEnvelope', () => {
  it('recognizes a valid envelope', () => {
    expect(
      isBackendErrorEnvelope({ error: { kind: 'timeout', message: 'x' } }),
    ).toBe(true);
  });
  it('rejects malformed payloads', () => {
    expect(isBackendErrorEnvelope({ error: { kind: 'x' } })).toBe(false);
    expect(isBackendErrorEnvelope(null)).toBe(false);
    expect(isBackendErrorEnvelope('plain')).toBe(false);
  });
});
