export type ActivationMetric = 'l2_norm' | 'mean_abs' | 'sparsity';

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
