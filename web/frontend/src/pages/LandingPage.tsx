import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import type { GenerateParams } from '@/api/client';
import { BackendStatusBanner } from '@/components/feedback/BackendStatusBanner';
import { EmptyState } from '@/components/feedback/EmptyState';
import { ErrorState } from '@/components/feedback/ErrorState';
import { LoadingState } from '@/components/feedback/LoadingState';
import { useBackendHealth } from '@/hooks/useBackendHealth';
import { useTrace } from '@/hooks/useTrace';
import { loadTraceFromFile } from '@/lib/trace/load';
import { putDiffPair, putTrace } from '@/lib/trace/store';
import './LandingPage.css';

export function LandingPage() {
  const { trace, load, status, error } = useTrace();
  const health = useBackendHealth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [dismissed, setDismissed] = useState(false);
  const [pendingNavId, setPendingNavId] = useState<string | null>(null);
  const urlLoadFired = useRef(false);

  // Auto-load trace from ?trace=<url> query param (set by `token-heatmap trace --serve`).
  useEffect(() => {
    if (urlLoadFired.current) return;
    const traceUrl = searchParams.get('trace');
    if (!traceUrl) return;
    urlLoadFired.current = true;
    setPendingNavId('url-loaded');
    void load({ type: 'url', url: traceUrl });
  }, [searchParams, load]);

  useEffect(() => {
    if (trace && pendingNavId) {
      // The sample route always reloads the bundled sample on the viewer,
      // so we only seed the store for non-sample loads.
      if (pendingNavId !== 'sample') {
        putTrace(pendingNavId, trace);
      }
      navigate(`/trace/${pendingNavId}`);
      setPendingNavId(null);
    }
  }, [trace, pendingNavId, navigate]);

  const handleLoadSample = async () => {
    setDismissed(false);
    setPendingNavId('sample');
    await load({ type: 'sample' });
  };

  const handleUrlSubmit = async (url: string) => {
    setDismissed(false);
    setPendingNavId('url-loaded');
    await load({ type: 'url', url });
  };

  const handleGenerate = async (params: GenerateParams) => {
    setDismissed(false);
    setPendingNavId('generated');
    await load({ type: 'generate', params });
  };

  const handleFileDropped = async (file: File) => {
    setDismissed(false);
    const isCsv =
      file.name.toLowerCase().endsWith('.csv') || file.type === 'text/csv';
    setPendingNavId(isCsv ? 'uploaded-csv' : 'uploaded');
    if (isCsv) {
      await load({ type: 'csv', file });
    } else {
      await load({ type: 'file', file });
    }
  };

  const handleTwoFilesDropped = async (fileA: File, fileB: File) => {
    setDismissed(false);
    try {
      const [traceA, traceB] = await Promise.all([
        loadTraceFromFile(fileA),
        loadTraceFromFile(fileB),
      ]);
      const id = `diff-${Date.now().toString(36)}-${Math.random()
        .toString(36)
        .slice(2, 8)}`;
      putDiffPair(id, traceA, traceB);
      navigate(`/diff/${id}`);
    } catch (err) {
      // Route the error through the existing load() machinery so the same
      // ErrorState rendering applies. Re-throwing the parsed trace error
      // from a single-trace load() also produces a friendly message.
      await load({ type: 'file', file: fileA });
      void err;
    }
  };

  if (status === 'loading') {
    const label =
      pendingNavId === 'generated'
        ? 'Generating trace (this can take a while)…'
        : 'Loading trace';
    return <LoadingState label={label} />;
  }

  if (status === 'error' && error && !dismissed) {
    const isValidate = error.kind === 'validate';
    return (
      <ErrorState
        title={
          isValidate ? 'Trace failed schema validation' : 'Something went wrong'
        }
        message={
          isValidate
            ? 'The trace JSON did not match the expected shape. Expand the issues below for the exact field paths the validator rejected.'
            : error.message
        }
        issues={error.issues}
        onRetry={handleLoadSample}
        onReset={() => setDismissed(true)}
      />
    );
  }

  return (
    <div className="landing-page">
      <div className="landing-page__status">
        <BackendStatusBanner
          status={health.status}
          onRetry={() => void health.probe()}
        />
      </div>
      <EmptyState
        onLoadSample={handleLoadSample}
        onFileDropped={handleFileDropped}
        onTwoFilesDropped={handleTwoFilesDropped}
        onUrlSubmit={handleUrlSubmit}
        onGenerate={handleGenerate}
      />
    </div>
  );
}
