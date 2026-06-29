import type { AttentionSidecar } from './attention-types';

const cache = new Map<string, Promise<AttentionSidecar>>();

function cacheKey(step: number, ref: string): string {
  return `${step}::${ref}`;
}

/**
 * Fetch a sidecar JSON payload for one step, caching the in-flight promise
 * by `(step, ref)` so concurrent calls share a single network request and
 * repeat opens return synchronously from cache.
 *
 * The ref is treated as a URL resolved against the current document; the
 * existing static trace dir is expected to serve the file. Backend changes
 * are not part of this ticket.
 */
export function loadAttentionSidecar(
  step: number,
  ref: string,
  fetchImpl: typeof fetch = globalThis.fetch,
): Promise<AttentionSidecar> {
  const key = cacheKey(step, ref);
  const cached = cache.get(key);
  if (cached) return cached;
  const inflight = fetchImpl(ref).then(async (res) => {
    if (!res.ok) {
      cache.delete(key);
      throw new Error(
        `Failed to load attention sidecar for step ${step}: HTTP ${res.status}`,
      );
    }
    return (await res.json()) as AttentionSidecar;
  });
  // Cache the promise so concurrent calls coalesce, but evict on rejection
  // so callers can retry after a transient network failure.
  inflight.catch(() => {
    cache.delete(key);
  });
  cache.set(key, inflight);
  return inflight;
}

/** Clear the entire sidecar cache. Exposed for tests. */
export function clearAttentionSidecarCache(): void {
  cache.clear();
}
