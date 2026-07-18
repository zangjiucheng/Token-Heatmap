import type { TraceWithActivations } from '@/types/activation';

export type ActivationMetric = 'l2_norm' | 'mean_abs' | 'sparsity';

/** True when any step of the trace carries activation histograms (traces
 * captured before distribution stats were added have none). */
export function traceHasHistograms(trace: TraceWithActivations): boolean {
  return trace.steps.some((s) =>
    (s.activations ?? []).some((a) => a.histogram),
  );
}

export const ACTIVATION_METRICS: ActivationMetric[] = [
  'l2_norm',
  'mean_abs',
  'sparsity',
];

export const ACTIVATION_METRIC_LABELS: Record<ActivationMetric, string> = {
  l2_norm: 'L2 norm',
  mean_abs: 'Mean |x|',
  sparsity: 'Sparsity',
};
