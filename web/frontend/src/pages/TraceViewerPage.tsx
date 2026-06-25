import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { LoadingState } from '@/components/feedback/LoadingState';
import { ErrorState } from '@/components/feedback/ErrorState';
import { EmptyState } from '@/components/feedback/EmptyState';
import { useTrace } from '@/hooks/useTrace';
import { useViewState } from '@/hooks/useViewState';
import { useKeymap } from '@/hooks/useKeymap';
import { announceLiveRegion } from '@/lib/a11y/announceLiveRegion';
import type { ComparisonMode } from '@/features/comparison';
import { takeTrace } from '@/lib/trace/store';
import { TokenHeatmap } from '@/features/heatmap';
import { SplitHeatmap } from '@/features/comparison';
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
import { DirectLogitAttributionTab } from '@/features/dla';
import { AttributionGraphTab } from '@/features/graph';
import {
  ControlBar,
  HEATMAP_CONTROL_LENSES,
  LensRail,
  TraceWorkspace,
} from '@/features/workspace';
import type { Trace } from '@/types/trace';
import type { TraceWithActivations } from '@/types/activation';
import './TraceViewerPage.css';

const COMPARISON_CYCLE: ComparisonMode[] = ['raw', 'processed', 'split'];

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
    <>
      <EntropyTimeline
        trace={trace}
        selectedStep={selectedStep}
        hoveredStep={hoveredStep}
        onSelectStep={onSelectStep}
        onHoverStep={onHoverStep}
        stepRange={stepRange}
        height={190}
      />
      <SelectedProbabilityTimeline
        trace={trace}
        selectedStep={selectedStep}
        hoveredStep={hoveredStep}
        onSelectStep={onSelectStep}
        onHoverStep={onHoverStep}
        stepRange={stepRange}
        height={190}
      />
    </>
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
  // The spine timelines are compact but can be collapsed entirely to give the
  // lens full height. Persisted locally so the choice survives a reload.
  const [timelinesOpen, setTimelinesOpen] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem('llm-heatmap-overview') !== '0';
    } catch {
      return true;
    }
  });
  const toggleTimelines = () => {
    setTimelinesOpen((open) => {
      const next = !open;
      try {
        window.localStorage.setItem('llm-heatmap-overview', next ? '1' : '0');
      } catch {
        // ignore — storage unavailable
      }
      return next;
    });
  };

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
      // Left binding now collapses / expands the lens rail.
      setLeftOpen(!state.leftOpen);
    },
    'panel.toggleRight': () => {
      // Right binding now toggles the inspector.
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

  let content: React.ReactNode;

  if (status === 'loading' || status === 'idle') {
    content = <LoadingState label={`Loading trace ${id ?? ''}`} />;
  } else if (status === 'error' && error) {
    content = (
      <ErrorState
        message={error.message}
        onRetry={() => load({ type: 'sample' })}
      />
    );
  } else if (trace) {
    const valueRangeOverride =
      state.colorRange.mode === 'manual' &&
      state.colorRange.min != null &&
      state.colorRange.max != null
        ? { min: state.colorRange.min, max: state.colorRange.max }
        : undefined;

    const hasAttention = trace.attention_metadata != null;
    const hasActivations =
      (trace as TraceWithActivations).activation_metadata != null;
    const hasLogitLens = trace.steps.some(
      (s) => Array.isArray(s.logit_lens) && s.logit_lens.length > 0,
    );
    const hasManifold = (trace.manifold?.layers?.length ?? 0) > 0;
    const hasDirectLogitAttribution =
      (trace.direct_logit_attribution?.steps?.length ?? 0) > 0;
    const availability = {
      attention: hasAttention,
      logitLens: hasLogitLens,
      activations: hasActivations,
      directLogitAttribution: hasDirectLogitAttribution,
      manifold: hasManifold,
    };

    let activeTab = state.tab;
    if (!hasAttention && activeTab === 'attention') activeTab = 'heatmap';
    if (!hasLogitLens && activeTab === 'logit-lens') activeTab = 'heatmap';
    if (!hasActivations && activeTab === 'activations') activeTab = 'heatmap';
    if (!hasDirectLogitAttribution && activeTab === 'direct-logit-attribution')
      activeTab = 'heatmap';
    if (!hasDirectLogitAttribution && activeTab === 'attribution-graph')
      activeTab = 'heatmap';
    if (!hasManifold && activeTab === 'manifold') activeTab = 'heatmap';

    const traceWithActivations = trace as TraceWithActivations;

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

    let canvas: React.ReactNode;
    if (activeTab === 'model') {
      canvas = <ModelTab trace={trace} />;
    } else if (activeTab === 'output') {
      canvas = <OutputTab trace={trace} />;
    } else if (activeTab === 'attention') {
      canvas = (
        <AttentionTab
          trace={trace}
          selectedStep={selectedStep}
          selectedHead={state.selectedHead}
          onSelectHead={setSelectedHead}
        />
      );
    } else if (activeTab === 'logit-lens') {
      canvas = <LogitLensTab trace={trace} selectedStep={selectedStep} />;
    } else if (activeTab === 'direct-logit-attribution') {
      canvas = (
        <DirectLogitAttributionTab trace={trace} selectedStep={selectedStep} />
      );
    } else if (activeTab === 'attribution-graph') {
      canvas = (
        <AttributionGraphTab trace={trace} selectedStep={selectedStep} />
      );
    } else if (activeTab === 'activations') {
      canvas = (
        <ActivationsTab
          trace={traceWithActivations}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
          hoveredStep={hoveredStep}
          onHoverStep={setHoveredStep}
        />
      );
    } else if (activeTab === 'manifold') {
      canvas = (
        <ManifoldTab
          trace={trace}
          selectedStep={selectedStep}
          onSelectStep={setSelectedStep}
          hoveredStep={hoveredStep}
          onHoverStep={setHoveredStep}
        />
      );
    } else {
      canvas = heatmapBody;
    }

    const inspector = (
      <div className="trace-viewer-right" data-testid="trace-viewer-right">
        <StepDetailPanel trace={trace} selectedStep={selectedStep} />
        {hasAttention &&
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
    );

    content = (
      <TraceWorkspace
        model={trace.metadata?.model ?? 'Unknown model'}
        prompt={trace.metadata?.prompt ?? ''}
        stepCount={trace.steps.length}
        selectedStep={selectedStep}
        railCollapsed={!state.leftOpen}
        inspectorOpen={state.rightOpen}
        onToggleInspector={() => setRightOpen(!state.rightOpen)}
        timelinesOpen={timelinesOpen}
        onToggleTimelines={toggleTimelines}
        tokenStrip={
          <GeneratedTokenStrip
            trace={trace}
            selectedStep={selectedStep}
            onSelectStep={setSelectedStep}
            hoveredStep={hoveredStep}
            onHoverStep={setHoveredStep}
          />
        }
        timelines={
          <TimelineOverview
            trace={trace}
            selectedStep={selectedStep}
            hoveredStep={hoveredStep}
            onSelectStep={setSelectedStep}
            onHoverStep={setHoveredStep}
            stepRange={state.stepRange ?? undefined}
          />
        }
        rail={
          <LensRail
            activeLens={activeTab}
            availability={availability}
            onSelect={setTab}
            collapsed={!state.leftOpen}
            onToggleCollapsed={() => setLeftOpen(!state.leftOpen)}
          />
        }
        controlBar={
          HEATMAP_CONTROL_LENSES.has(activeTab) ? (
            <ControlBar trace={trace} heatmapRef={heatmapRef} />
          ) : undefined
        }
        canvas={canvas}
        canvasRef={heatmapRef}
        inspector={inspector}
      />
    );
  } else {
    content = <EmptyState onLoadSample={() => load({ type: 'sample' })} />;
  }

  return (
    <>
      <h1 className="visually-hidden">Trace viewer</h1>
      {content}
    </>
  );
}
