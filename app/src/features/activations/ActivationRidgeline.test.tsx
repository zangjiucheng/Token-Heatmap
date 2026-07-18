import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ActivationRidgeline } from './ActivationRidgeline';
import { traceHasHistograms } from './activation-types';
import {
  FIXTURE_NUM_LAYERS,
  FIXTURE_SUBMODULES,
  makeActivationTrace,
} from './testFixtures';

describe('ActivationRidgeline', () => {
  it('renders one silhouette per captured layer for the selected step', () => {
    render(
      <ActivationRidgeline
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={1}
      />,
    );

    expect(screen.getByTestId('activation-ridgeline')).toBeInTheDocument();
    for (let layer = 0; layer < FIXTURE_NUM_LAYERS; layer += 1) {
      expect(screen.getByTestId(`ridgeline-row-${layer}`)).toBeInTheDocument();
    }
    expect(screen.getByText(/step 1/)).toBeInTheDocument();
  });

  it('falls back to the first step with activations when none is selected', () => {
    render(
      <ActivationRidgeline
        trace={makeActivationTrace()}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={null}
      />,
    );
    expect(screen.getByText(/step 0/)).toBeInTheDocument();
  });

  it('shows the empty state when the trace has no histograms', () => {
    const trace = makeActivationTrace();
    for (const step of trace.steps) {
      for (const entry of step.activations ?? []) {
        delete entry.histogram;
      }
    }
    expect(traceHasHistograms(trace)).toBe(false);

    render(
      <ActivationRidgeline
        trace={trace}
        submodule={FIXTURE_SUBMODULES[0]}
        selectedStep={0}
      />,
    );
    expect(screen.getByTestId('activation-ridgeline-empty')).toBeInTheDocument();
  });
});
