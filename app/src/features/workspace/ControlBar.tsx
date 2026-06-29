import { useMemo, type RefObject } from 'react';
import type { Trace } from '@/types/trace';
import { buildGrid } from '@/features/heatmap';
import { ComparisonToggle } from '@/features/comparison';
import {
  ColorRangeControls,
  StepRangeFilter,
  ValueColumnToggle,
  type ColorRangeValue,
} from '@/features/controls';
import { ExportControls } from '@/features/export';
import { useViewState } from '@/hooks/useViewState';
import './ControlBar.css';

export interface ControlBarProps {
  trace: Trace;
  heatmapRef: RefObject<HTMLElement>;
}

/**
 * Compact, horizontal controls for the heatmap family of lenses. This is the
 * old left-pane `ControlsPanel`, reused verbatim but laid out as a toolbar and
 * shown *only* on the lenses the controls actually affect — so the Model,
 * Output and Manifold lenses are no longer crowded by heatmap knobs.
 */
export function ControlBar({ trace, heatmapRef }: ControlBarProps) {
  const { state, setMode, setValueCol, setStepRange, setColorRange } =
    useViewState();

  const totalSteps = trace.steps.length;
  const lastStep = Math.max(0, totalSteps - 1);

  const effectiveStepRange: [number, number] = state.stepRange
    ? [
        Math.max(0, Math.min(lastStep, state.stepRange[0])),
        Math.max(0, Math.min(lastStep, state.stepRange[1])),
      ]
    : [0, lastStep];

  // Compute the auto-mode colour bounds so the manual seed has a reference.
  const autoBounds = useMemo(() => {
    const source = state.mode === 'raw' ? 'raw' : 'processed';
    const grid = buildGrid(trace, state.valueCol, source);
    return { min: grid.valueMin, max: grid.valueMax };
  }, [trace, state.valueCol, state.mode]);

  const handleColorRangeChange = (next: ColorRangeValue) => {
    setColorRange(next);
  };

  return (
    <div
      className="control-bar"
      data-testid="trace-viewer-controls"
      role="group"
      aria-label="Heatmap view settings"
    >
      <ComparisonToggle value={state.mode} onChange={setMode} />
      <ValueColumnToggle value={state.valueCol} onChange={setValueCol} />
      <StepRangeFilter
        min={0}
        max={lastStep}
        value={effectiveStepRange}
        onChange={(next) => {
          if (next[0] === 0 && next[1] === lastStep) {
            setStepRange(null);
          } else {
            setStepRange(next);
          }
        }}
      />
      <ColorRangeControls
        value={state.colorRange}
        onChange={handleColorRangeChange}
        autoMin={autoBounds.min}
        autoMax={autoBounds.max}
      />
      <ExportControls trace={trace} heatmapRef={heatmapRef} />
    </div>
  );
}
