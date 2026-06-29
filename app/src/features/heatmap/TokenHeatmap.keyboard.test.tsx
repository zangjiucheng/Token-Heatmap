import { render, fireEvent, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import sampleTrace from '@/lib/sample/trace.json';
import type { Trace } from '@/types/trace';
import { TokenHeatmap } from './TokenHeatmap';

const trace = sampleTrace as unknown as Trace;
const lastStep = trace.steps.length - 1;

function renderWith(initial: number | null) {
  const onSelectStep = vi.fn();
  const utils = render(
    <TokenHeatmap
      trace={trace}
      valueCol="logprob"
      selectedStep={initial}
      onSelectStep={onSelectStep}
      width={800}
      height={400}
    />,
  );
  const plot = utils.getByTestId('token-heatmap-plot');
  return { plot, onSelectStep };
}

afterEach(() => {
  cleanup();
});

describe('TokenHeatmap keyboard navigation', () => {
  it('ArrowRight advances selectedStep', () => {
    const { plot, onSelectStep } = renderWith(0);
    fireEvent.keyDown(plot, { key: 'ArrowRight' });
    expect(onSelectStep).toHaveBeenCalledWith(1);
  });

  it('ArrowLeft retreats selectedStep', () => {
    const { plot, onSelectStep } = renderWith(2);
    fireEvent.keyDown(plot, { key: 'ArrowLeft' });
    expect(onSelectStep).toHaveBeenCalledWith(1);
  });

  it('ArrowLeft from step 0 stays at 0 (clamped)', () => {
    const { plot, onSelectStep } = renderWith(0);
    fireEvent.keyDown(plot, { key: 'ArrowLeft' });
    expect(onSelectStep).toHaveBeenCalledWith(0);
  });

  it('Home jumps to step 0', () => {
    const { plot, onSelectStep } = renderWith(2);
    fireEvent.keyDown(plot, { key: 'Home' });
    expect(onSelectStep).toHaveBeenCalledWith(0);
  });

  it('End jumps to the last step', () => {
    const { plot, onSelectStep } = renderWith(0);
    fireEvent.keyDown(plot, { key: 'End' });
    expect(onSelectStep).toHaveBeenCalledWith(lastStep);
  });

  it('ArrowRight from the last step is a no-op (clamped)', () => {
    const { plot, onSelectStep } = renderWith(lastStep);
    fireEvent.keyDown(plot, { key: 'ArrowRight' });
    expect(onSelectStep).toHaveBeenCalledWith(lastStep);
  });

  it('Enter re-confirms the current selection (no-op)', () => {
    const { plot, onSelectStep } = renderWith(1);
    fireEvent.keyDown(plot, { key: 'Enter' });
    expect(onSelectStep).toHaveBeenCalledWith(1);
  });
});
