/**
 * Module-level cache of the most recently loaded trace.
 *
 * The landing page and the viewer page each instantiate their own
 * `useTrace` hook; without a shared cache the viewer would re-fetch what
 * the landing page just loaded. For the generate path the same problem is
 * worse — the trace lives only in memory and has no URL to refetch from.
 *
 * This is intentionally minimal: a single slot, last-write-wins. The
 * `id` parameter lets the viewer assert it picks up the right trace.
 */

import type { Trace } from '@/types/trace';

interface Entry {
  id: string;
  trace: Trace;
}

interface DiffPairEntry {
  id: string;
  traceA: Trace;
  traceB: Trace;
}

let cached: Entry | null = null;
let cachedDiffPair: DiffPairEntry | null = null;

export function putTrace(id: string, trace: Trace): void {
  cached = { id, trace };
}

export function takeTrace(id: string): Trace | null {
  if (cached?.id === id) {
    return cached.trace;
  }
  return null;
}

export function clearTraceCache(): void {
  cached = null;
  cachedDiffPair = null;
}

export function putDiffPair(id: string, traceA: Trace, traceB: Trace): void {
  cachedDiffPair = { id, traceA, traceB };
}

export function takeDiffPair(
  id: string,
): { traceA: Trace; traceB: Trace } | null {
  if (cachedDiffPair?.id === id) {
    return { traceA: cachedDiffPair.traceA, traceB: cachedDiffPair.traceB };
  }
  return null;
}
