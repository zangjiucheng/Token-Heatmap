import { useCallback, useRef, useState } from 'react';

import type { Trace } from '@/types/trace';
import type { GenerateParams } from '@/api/client';
import {
  convertCsvToTrace,
  generateTrace,
  loadSampleTrace,
  loadTraceFromFile,
  loadTraceFromUrl,
} from '@/lib/trace/load';
import { TraceLoadError, isTraceLoadError } from '@/lib/trace/errors';
import { takeTrace } from '@/lib/trace/store';

export type TraceStatus = 'idle' | 'loading' | 'ready' | 'error';

export type TraceSource =
  | { type: 'file'; file: File }
  | { type: 'url'; url: string }
  | { type: 'csv'; file: File }
  | { type: 'sample' }
  /** Generate a fresh trace on the backend from model + prompt + params. */
  | { type: 'generate'; params: GenerateParams }
  /** Adopt a trace another flow has already put in the shared store. */
  | { type: 'cached'; id: string }
  /** Adopt a trace value directly (e.g. just returned by the API). */
  | { type: 'inline'; trace: Trace };

export interface UseTraceResult {
  trace: Trace | null;
  status: TraceStatus;
  error: TraceLoadError | null;
  load: (source: TraceSource) => Promise<void>;
}

async function loadSource(source: TraceSource): Promise<Trace> {
  switch (source.type) {
    case 'file':
      return loadTraceFromFile(source.file);
    case 'url':
      return loadTraceFromUrl(source.url);
    case 'csv':
      return convertCsvToTrace(source.file);
    case 'sample':
      return loadSampleTrace();
    case 'generate':
      return generateTrace(source.params);
    case 'cached': {
      const trace = takeTrace(source.id);
      if (!trace) {
        throw TraceLoadError.network(`No cached trace for id "${source.id}"`);
      }
      return trace;
    }
    case 'inline':
      return source.trace;
  }
}

/**
 * React hook wrapping the current trace, loader state, and a `load` action.
 *
 * Concurrent calls to `load` are resolved by last-write-wins: only the most
 * recent invocation updates state, so a stale slow URL fetch can never
 * clobber a fresh sample load.
 */
export function useTrace(): UseTraceResult {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [status, setStatus] = useState<TraceStatus>('idle');
  const [error, setError] = useState<TraceLoadError | null>(null);
  const requestId = useRef(0);

  const load = useCallback(async (source: TraceSource): Promise<void> => {
    const id = ++requestId.current;
    setStatus('loading');
    setError(null);
    try {
      const loaded = await loadSource(source);
      if (id !== requestId.current) return;
      setTrace(loaded);
      setStatus('ready');
    } catch (err) {
      if (id !== requestId.current) return;
      const wrapped = isTraceLoadError(err)
        ? err
        : TraceLoadError.parse(
            err instanceof Error ? err.message : 'Unknown loader failure',
            err,
          );
      setError(wrapped);
      setStatus('error');
    }
  }, []);

  return { trace, status, error, load };
}
