import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { EmptyState } from '@/components/feedback/EmptyState';
import { ErrorState } from '@/components/feedback/ErrorState';
import { LoadingState } from '@/components/feedback/LoadingState';
import { useTrace } from '@/hooks/useTrace';
import { loadTraceFromFile } from '@/lib/trace/load';
import { putDiffPair, putTrace } from '@/lib/trace/store';
import './LandingPage.css';

export function LandingPage() {
  const { trace, load, status, error } = useTrace();
  const navigate = useNavigate();
  const [dismissed, setDismissed] = useState(false);
  const [pendingNavId, setPendingNavId] = useState<string | null>(null);

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

  const handleFileDropped = async (file: File) => {
    setDismissed(false);
    setPendingNavId('uploaded');
    await load({ type: 'file', file });
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
    return <LoadingState label="Loading trace" />;
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
      <EmptyState
        onLoadSample={handleLoadSample}
        onFileDropped={handleFileDropped}
        onTwoFilesDropped={handleTwoFilesDropped}
      />
    </div>
  );
}
