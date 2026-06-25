import type { ViewerTab } from '@/hooks/useViewState';

/**
 * The trace workspace presents one *lens* at a time — a way of looking at the
 * generation. Lenses are grouped by what they reveal so the rail reads as a
 * small, legible menu instead of a flat row of seven tabs:
 *
 *   Generation — what the model produced (always available)
 *   Internals  — what happened inside (needs a capture flag at trace time)
 *   Geometry   — the shape of the representation space
 */
export type LensGroup = 'generation' | 'internals' | 'geometry';

/** Availability keys map to the capture flags a trace may or may not carry. */
export type LensAvailabilityKey =
  | 'attention'
  | 'logitLens'
  | 'activations'
  | 'manifold';

export interface LensDef {
  id: ViewerTab;
  label: string;
  group: LensGroup;
  /**
   * Stable `data-testid` carried over from the old tab strip so existing unit
   * and e2e selectors keep targeting the same lens after the redesign.
   */
  testId: string;
  /** Undefined ⇒ always available. Otherwise gated on the matching capture flag. */
  availabilityKey?: LensAvailabilityKey;
  /** Shown as a tooltip when the lens is locked (capture flag missing). */
  lockedHint?: string;
}

export const LENS_GROUP_LABELS: Record<LensGroup, string> = {
  generation: 'Generation',
  internals: 'Internals',
  geometry: 'Geometry',
};

export const LENS_GROUP_ORDER: readonly LensGroup[] = [
  'generation',
  'internals',
  'geometry',
];

export const LENSES: readonly LensDef[] = [
  {
    id: 'heatmap',
    label: 'Token Heatmap',
    group: 'generation',
    testId: 'heatmap-tab',
  },
  {
    id: 'output',
    label: 'Output',
    group: 'generation',
    testId: 'output-tab-button',
  },
  {
    id: 'model',
    label: 'Model',
    group: 'generation',
    testId: 'model-tab',
  },
  {
    id: 'attention',
    label: 'Attention',
    group: 'internals',
    testId: 'attention-tab',
    availabilityKey: 'attention',
    lockedHint:
      'This trace was generated without --capture-attention. Re-run the CLI with that flag to inspect attention.',
  },
  {
    id: 'logit-lens',
    label: 'Logit Lens',
    group: 'internals',
    testId: 'logit-lens-tab',
    availabilityKey: 'logitLens',
    lockedHint:
      'This trace was generated without --capture-logit-lens. Re-run the CLI with that flag to inspect per-layer predictions.',
  },
  {
    id: 'activations',
    label: 'Activations',
    group: 'internals',
    testId: 'activations-tab',
    availabilityKey: 'activations',
    lockedHint:
      'This trace was generated without an ActivationProbe. Re-run the CLI with --capture-activations to inspect activations.',
  },
  {
    id: 'manifold',
    label: 'Manifold',
    group: 'geometry',
    testId: 'manifold-tab',
    availabilityKey: 'manifold',
    lockedHint:
      'This trace has no manifold analysis. Run `token-heatmap manifold --trace <file>` (needs --capture-full-activations) to add it.',
  },
];

export interface LensAvailability {
  attention: boolean;
  logitLens: boolean;
  activations: boolean;
  manifold: boolean;
}

export function isLensAvailable(
  lens: LensDef,
  availability: LensAvailability,
): boolean {
  if (!lens.availabilityKey) return true;
  return availability[lens.availabilityKey];
}

export function lensesInGroup(group: LensGroup): LensDef[] {
  return LENSES.filter((lens) => lens.group === group);
}

/**
 * Lenses that surface the heatmap ControlBar (distribution mode, value column,
 * colour range, export). Other lenses carry their own controls, so showing the
 * heatmap controls on them was the main source of the old left-pane clutter.
 */
export const HEATMAP_CONTROL_LENSES: ReadonlySet<ViewerTab> = new Set<ViewerTab>(
  ['heatmap'],
);
