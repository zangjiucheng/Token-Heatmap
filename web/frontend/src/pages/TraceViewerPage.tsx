import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ThreePaneLayout } from '@/components/layout/ThreePaneLayout';
import { LoadingState } from '@/components/feedback/LoadingState';
import { ErrorState } from '@/components/feedback/ErrorState';
import { EmptyState } from '@/components/feedback/EmptyState';
import { useTrace } from '@/hooks/useTrace';
import { useViewState } from '@/hooks/useViewState';
import { useKeymap } from '@/hooks/useKeymap';
import { announceLiveRegion } from '@/lib/a11y/announceLiveRegion';
import type { ComparisonMode } from '@/features/comparison';
import { takeTrace } from '@/lib/trace/store';
import { TokenHeatmap, buildGrid } from '@/features/heatmap';
import { ComparisonToggle, SplitHeatmap } from '@/features/comparison';
import {
  ValueColumnToggle,
  StepRangeFilter,
  ColorRangeControls,
  type ColorRangeValue,
} from '@/features/controls';
import { ExportControls } from '@/features/export';
import { GeneratedTokenStrip, StepDetailPanel } from '@/features/detail';
import {
  EntropyTimeline,
  SelectedProbabilityTimeline,
} from '@/features/timelines';
import { AttentionTab, AttentionHeadPattern, LogitLensTab } from '@/features/attention';
import { ActivationsTab } from '@/features/activations';
import { ManifoldTab } from '@/features/manifold';
import { ModelTab } from '@/features/model';
import { OutputTab } from '@/features/output';
import type { Trace } from '@/types/trace';
import type { TraceWithActivations } from '@/types/activation';
import './TraceViewerPage.css';

function Placeholder({ name }: { name: string }) {
  return (
    <div data-testid={`placeholder-${name}`}>
      <p>{name} placeholder — wired in a later ticket.</p>
    </div>
  );
}

interface ControlsPanelProps {
  trace: Trace;
  heatmapRef: React.RefObject<HTMLElement>;
}

function ControlsPanel({ trace, heatmapRef }: ControlsPanelProps) {
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

  // Compute the auto-mode color bounds for display in the manual seed.
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
      className="trace-viewer-controls"
      data-testid="trace-viewer-controls"
      style={{ display: 'flex', flexDirection: 'column', gap: 16 }}
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

const COMPARISON_CYCLE: ComparisonMode[] = ['raw', 'processed', 'split'];

function TraceContextHeader({
  trace,
  selectedStep,
}: {
  trace: Trace;
  selectedStep: number | null;
}) {
  const model = trace.metadata?.model ?? 'Unknown model';
  const prompt = trace.metadata?.prompt ?? '';
  const stepCount = trace.steps.length;
  const selectedLabel =
    selectedStep == null ? 'No step selected' : `Step ${selectedStep}`;

  return (
    <div className="trace-viewer-center__context">
      <div className="trace-viewer-center__context-main">
        <div className="trace-viewer-center__eyebrow">Trace workspace</div>
        <h2 className="trace-viewer-center__title">{model}</h2>
        {prompt ? (
          <p className="trace-viewer-center__prompt">{prompt}</p>
        ) : null}
      </div>
      <dl className="trace-viewer-center__meta" aria-label="Trace summary">
        <div>
          <dt>Steps</dt>
          <dd>{stepCount}</dd>
        </div>
        <div>
          <dt>Selection</dt>
          <dd>{selectedLabel}</dd>
        </div>
      </dl>
    </div>
  );
}

function TimelineOverview({
  trace,
  selectedStep,
  hoveredStep,
  onSelectStep,
  onHoverStep,
  stepRange,
}: {
  trace: Trace;
  selectedStep: number | null;
  hoveredStep: number | null;
  onSelectStep: (step: number) => void;
  onHoverStep: (step: number | null) => void;
  stepRange?: [number, number];
}) {
  return (
    <section
      className="trace-viewer-center__overview"
      aria-label="Trace overview timelines"
      data-testid="trace-viewer-center-timelines"
    >
      <EntropyTimeline
        trace={trace}
        selectedStep={selectedStep}
        hoveredStep={hoveredStep}
        onSelectStep={onSelectStep}
        onHoverStep={onHoverStep}
        stepRange={stepRange}
        height={128}
      />
      <SelectedProbabilityTimeline
        trace={trace}
        selectedStep={selectedStep}
        hoveredStep={hoveredStep}
        onSelectStep={onSelectStep}
        onHoverStep={onHoverStep}
        stepRange={stepRange}
        height={128}
      />
    </section>
  );
}

export function TraceViewerPage() {
  const { id } = useParams<{ id?: string }>();
  const { trace, status, error, load } = useTrace();
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const { state, setMode, setLeftOpen, setRightOpen, setTab, setSelectedHead } =
    useViewState();
  const heatmapRef = useRef<HTMLDivElement | null>(null);

  const totalSteps = trace?.steps.length ?? 0;
  const lastStep = Math.max(0, totalSteps - 1);

  useKeymap({
    'selection.prev': () => {
      if (!trace) return;
      const next = selectedStep == null ? 0 : Math.max(0, selectedStep - 1);
      setSelectedStep(next);
      announceLiveRegion(`Step ${next} of ${lastStep}`);
    },
    'selection.next': () => {
      if (!trace) return;
      const next =
        selectedStep == null ? 0 : Math.min(lastStep, selectedStep + 1);
      setSelectedStep(next);
      announceLiveRegion(`Step ${next} of ${lastStep}`);
    },
    'selection.first': () => {
      if (!trace) return;
      setSelectedStep(0);
      announceLiveRegion(`Step 0 of ${lastStep}`);
    },
    'selection.last': () => {
      if (!trace) return;
      setSelectedStep(lastStep);
      announceLiveRegion(`Step ${lastStep} of ${lastStep}`);
    },
    'selection.clear': () => {
      setSelectedStep(null);
    },
    'comparison.cycle': () => {
      const idx = COMPARISON_CYCLE.indexOf(state.mode);
      const next = COMPARISON_CYCLE[(idx + 1) % COMPARISON_CYCLE.length];
      setMode(next);
      announceLiveRegion(`Comparison mode: ${next}`);
    },
    'view.reset': () => {
      // The heatmap exposes a Reset view button — trigger it programmatically.
      const reset = document.querySelector<HTMLButtonElement>(
        '[data-testid="token-heatmap-reset"]',
      );
      reset?.click();
    },
    'panel.toggleLeft': () => {
      setLeftOpen(!state.leftOpen);
    },
    'panel.toggleRight': () => {
      setRightOpen(!state.rightOpen);
    },
  });

  useEffect(() => {
    if (status === 'idle') {
      if (id && id !== 'sample' && takeTrace(id)) {
        void load({ type: 'cached', id });
      } else {
        void load({ type: 'sample' });
      }
    }
  }, [status, load, id]);

  // Clamp selectedStep into the active step window whenever the window shrinks.
  useEffect(() => {
    if (selectedStep == null) return;
    if (!state.stepRange) return;
    const [s, e] = state.stepRange;
    if (selectedStep < s) setSelectedStep(s);
    else if (selectedStep > e) setSelectedStep(e);
  }, [state.stepRange, selectedStep]);

  const valueRangeOverride =
    state.colorRange.mode === 'manual' &&
    state.colorRange.min != null &&
    state.colorRange.max != null
      ? { min: state.colorRange.min, max: state.colorRange.max }
      : undefined;

  const hasAttention = trace?.attention_metadata != null;
  const hasActivations =
    (trace as TraceWithActivations | null)?.activation_metadata != null;
  const hasLogitLens =
    trace?.steps.some(
      (s) => Array.isArray(s.logit_lens) && s.logit_lens.length > 0,
    ) ?? false;
  const hasManifold = (trace?.manifold?.layers?.length ?? 0) > 0;
  let activeTab = state.tab;
  if (!hasAttention && activeTab === 'attention') activeTab = 'heatmap';
  if (!hasLogitLens && activeTab === 'logit-lens') activeTab = 'heatmap';
  if (!hasActivations && activeTab === 'activations') activeTab = 'heatmap';
  if (!hasManifold && activeTab === 'manifold') activeTab = 'heatmap';

  let center: React.ReactNode;
  if (status === 'loading' || status === 'idle') {
    center = <LoadingState label={`Loading trace ${id ?? ''}`} />;
  } else if (status === 'error' && error) {
    center = (
      <ErrorState
        message={error.message}
        onRetry={() => load({ type: 'sample' })}
      />
    );
  } else if (trace) {
    const tokenStrip = (
      <GeneratedTokenStrip
        trace={trace}
        selectedStep={selectedStep}
        onSelectStep={setSelectedStep}
        hoveredStep={hoveredStep}
        onHoverStep={setHoveredStep}
      />
    );

    const traceWithActivations = trace as TraceWithActivations;

    const tabStrip = (
      <div
        className="trace-viewer-center__tabs"
        role="tablist"
        aria-label="Trace view"
        data-testid="trace-viewer-tabs"
      >
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'heatmap'}
          className={
            activeTab === 'heatmap'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('heatmap')}
          data-testid="heatmap-tab"
        >
          Token Heatmap
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'model'}
          className={
            activeTab === 'model'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('model')}
          data-testid="model-tab"
        >
          Model
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'output'}
          className={
            activeTab === 'output'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('output')}
          data-testid="output-tab-button"
        >
          Output
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'attention'}
          disabled={!hasAttention}
          title={
            hasAttention
              ? undefined
              : 'This trace was generated without --capture-attention. Re-run the CLI with that flag to inspect attention.'
          }
          className={
            activeTab === 'attention'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('attention')}
          data-testid="attention-tab"
        >
          Attention
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'logit-lens'}
          disabled={!hasLogitLens}
          title={
            hasLogitLens
              ? undefined
              : 'This trace was generated without --capture-logit-lens. Re-run the CLI with that flag to inspect per-layer predictions.'
          }
          className={
            activeTab === 'logit-lens'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('logit-lens')}
          data-testid="logit-lens-tab"
        >
          Logit Lens
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'activations'}
          disabled={!hasActivations}
          title={
            hasActivations
              ? undefined
              : 'This trace was generated without an ActivationProbe. Re-run the CLI with --capture-activations to inspect activations.'
          }
          className={
            activeTab === 'activations'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('activations')}
          data-testid="activations-tab"
        >
          Activations
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'manifold'}
          disabled={!hasManifold}
          title={
            hasManifold
              ? undefined
              : 'This trace has no manifold analysis. Run `token-heatmap manifold --trace <file>` (needs --capture-full-activations) to add it.'
          }
          className={
            activeTab === 'manifold'
              ? 'trace-viewer-center__tab trace-viewer-center__tab--active'
              : 'trace-viewer-center__tab'
          }
          onClick={() => setTab('manifold')}
          data-testid="manifold-tab"
        >
          Manifold
        </button>
      </div>
    );

    const heatmapBody =
      state.mode === 'split' ? (
        <SplitHeatmap
          trace={trace}
          valueCol={state.valueCol}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
          stepRange={state.stepRange ?? undefined}
          valueRange={valueRangeOverride}
        />
      ) : (
        <TokenHeatmap
          trace={trace}
          valueCol={state.valueCol}
          source={state.mode === 'raw' ? 'raw' : 'processed'}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
          externalHoveredStep={hoveredStep}
          stepRange={state.stepRange ?? undefined}
          valueRange={valueRangeOverride}
        />
      );

    let body: React.ReactNode;
    if (activeTab === 'model') {
      body = <ModelTab trace={trace} />;
    } else if (activeTab === 'output') {
      body = <OutputTab trace={trace} />;
    } else if (activeTab === 'attention') {
      body = (
        <AttentionTab
          trace={trace}
          selectedStep={selectedStep}
          selectedHead={state.selectedHead}
          onSelectHead={setSelectedHead}
        />
      );
    } else if (activeTab === 'logit-lens') {
      body = (
        <LogitLensTab trace={trace} selectedStep={selectedStep} />
      );
    } else if (activeTab === 'activations') {
      body = (
        <ActivationsTab
          trace={traceWithActivations}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
          hoveredStep={hoveredStep}
          onHoverStep={setHoveredStep}
        />
      );
    } else if (activeTab === 'manifold') {
      body = (
        <ManifoldTab
          trace={trace}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
          hoveredStep={hoveredStep}
          onHoverStep={setHoveredStep}
        />
      );
    } else {
      body = heatmapBody;
    }

    center = (
      <div className="trace-viewer-center" data-testid="trace-viewer-center">
        <TraceContextHeader trace={trace} selectedStep={selectedStep} />
        {tokenStrip}
        {tabStrip}
        <div
          ref={heatmapRef}
          className="trace-viewer-center__heatmap"
          data-testid="trace-viewer-heatmap-container"
        >
          {body}
        </div>
        <TimelineOverview
          trace={trace}
          selectedStep={selectedStep}
          hoveredStep={hoveredStep}
          onSelectStep={setSelectedStep}
          onHoverStep={setHoveredStep}
          stepRange={state.stepRange ?? undefined}
        />
      </div>
    );
  } else {
    center = <EmptyState onLoadSample={() => load({ type: 'sample' })} />;
  }

  const left =
    status === 'ready' && trace ? (
      <ControlsPanel trace={trace} heatmapRef={heatmapRef} />
    ) : (
      <Placeholder name="controls" />
    );

  return (
    <>
      <h1 className="visually-hidden">Trace viewer</h1>
      <ThreePaneLayout
        leftLabel="View settings"
        rightLabel="Inspector"
        leftOpen={state.leftOpen}
        rightOpen={state.rightOpen}
        onToggleLeft={setLeftOpen}
        onToggleRight={setRightOpen}
        left={left}
        center={center}
        right={
          <div className="trace-viewer-right" data-testid="trace-viewer-right">
            <StepDetailPanel trace={trace} selectedStep={selectedStep} />
            {trace &&
            hasAttention &&
            activeTab === 'attention' &&
            state.selectedHead ? (
              <div
                className="trace-viewer-right__attention-pattern"
                data-testid="trace-viewer-right-attention-pattern"
              >
                <AttentionHeadPattern
                  trace={trace}
                  selectedStep={selectedStep}
                  selectedHead={state.selectedHead}
                />
              </div>
            ) : null}
          </div>
        }
      />
    </>
  );
}
