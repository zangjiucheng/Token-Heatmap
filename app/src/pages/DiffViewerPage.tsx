import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ErrorState } from '@/components/feedback/ErrorState';
import { LoadingState } from '@/components/feedback/LoadingState';
import { DiffView } from '@/features/activations';
import type { TraceWithActivations } from '@/types/activation';
import { takeDiffPair } from '@/lib/trace/store';

export function DiffViewerPage() {
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();
  const [pair, setPair] = useState<{
    traceA: TraceWithActivations;
    traceB: TraceWithActivations;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) {
      setError('Missing diff id.');
      return;
    }
    const cached = takeDiffPair(id);
    if (!cached) {
      setError(
        'No diff pair found for this id. Return to the landing page to drop two trace files.',
      );
      return;
    }
    const traceA = cached.traceA as TraceWithActivations;
    const traceB = cached.traceB as TraceWithActivations;
    if (!traceA.activation_metadata || !traceB.activation_metadata) {
      setError(
        'One of the dropped traces has no activation_metadata. Re-run the producer with --capture-activations to enable diffing.',
      );
      return;
    }
    setPair({ traceA, traceB });
  }, [id]);

  if (error) {
    return (
      <ErrorState
        title="Cannot render diff"
        message={error}
        onRetry={() => navigate('/')}
      />
    );
  }

  if (!pair) {
    return <LoadingState label={`Loading diff ${id ?? ''}`} />;
  }

  return (
    <div className="diff-viewer-page" data-testid="diff-viewer-page">
      <h1 className="visually-hidden">Activation diff viewer</h1>
      <DiffView traceA={pair.traceA} traceB={pair.traceB} />
    </div>
  );
}
