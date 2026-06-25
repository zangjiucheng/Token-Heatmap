import type { CSSProperties, ReactNode, RefObject } from 'react';
import { ChevronIcon, InspectorIcon } from './icons';
import './TraceWorkspace.css';

export interface TraceWorkspaceProps {
  /** Trace identity, shown in the sticky sub-header. */
  model: string;
  prompt: string;
  stepCount: number;
  selectedStep: number | null;
  /** Spine — generated-token strip, always visible above the lens. */
  tokenStrip: ReactNode;
  /** Spine — entropy / probability timelines, docked below the lens. */
  timelines: ReactNode;
  /** The grouped lens navigation (LensRail). */
  rail: ReactNode;
  /** Lens-contextual controls; omit on lenses that carry their own. */
  controlBar?: ReactNode;
  /** Active lens body. */
  canvas: ReactNode;
  /** Wraps the canvas body so the heatmap PNG export can compose it. */
  canvasRef?: RefObject<HTMLDivElement>;
  /** Contextual inspector content (step / head / neuron detail). */
  inspector: ReactNode;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
  railCollapsed: boolean;
}

/**
 * Trace workspace layout — three clearly-separated roles instead of three
 * competing panes:
 *   Spine     — what the model generated (token strip + timelines), always on.
 *   Lens      — how you're looking (the rail + the canvas body).
 *   Inspector — the selected detail, contextual and collapsible.
 */
export function TraceWorkspace({
  model,
  prompt,
  stepCount,
  selectedStep,
  tokenStrip,
  timelines,
  rail,
  controlBar,
  canvas,
  canvasRef,
  inspector,
  inspectorOpen,
  onToggleInspector,
  railCollapsed,
}: TraceWorkspaceProps) {
  const selectedLabel =
    selectedStep == null ? 'No step selected' : `Step ${selectedStep}`;

  const rootStyle = {
    '--rail-w': railCollapsed ? '56px' : '208px',
    '--inspector-w': inspectorOpen ? '340px' : '40px',
  } as CSSProperties;

  return (
    <div
      className="trace-workspace"
      data-testid="trace-workspace"
      data-inspector-open={inspectorOpen ? 'true' : 'false'}
      style={rootStyle}
    >
      <header className="trace-workspace__subheader">
        <div className="trace-workspace__context">
          <div className="trace-workspace__context-main">
            <div className="trace-workspace__eyebrow">Trace workspace</div>
            <h2 className="trace-workspace__title">{model}</h2>
            {prompt ? (
              <p className="trace-workspace__prompt">{prompt}</p>
            ) : null}
          </div>
          <dl className="trace-workspace__meta" aria-label="Trace summary">
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
        <div className="trace-workspace__spine">{tokenStrip}</div>
      </header>

      <div className="trace-workspace__body">
        <div className="trace-workspace__rail-slot">{rail}</div>

        <section
          className="trace-workspace__canvas"
          aria-label="Active lens"
          data-testid="trace-viewer-center"
        >
          {controlBar ? (
            <div className="trace-workspace__controls">{controlBar}</div>
          ) : null}
          <div
            ref={canvasRef}
            className="trace-workspace__canvas-body"
            data-testid="trace-viewer-heatmap-container"
          >
            {canvas}
          </div>
          <div
            className="trace-workspace__timelines"
            aria-label="Trace overview timelines"
            data-testid="trace-viewer-center-timelines"
          >
            {timelines}
          </div>
        </section>

        <aside
          className="trace-workspace__inspector"
          aria-label="Inspector"
          data-inspector-open={inspectorOpen ? 'true' : 'false'}
        >
          {inspectorOpen ? (
            <>
              <div className="trace-workspace__inspector-header">
                <span className="eyebrow">Inspector</span>
                <button
                  type="button"
                  className="trace-workspace__inspector-toggle"
                  onClick={onToggleInspector}
                  aria-expanded
                  aria-label="Hide inspector"
                  title="Hide inspector"
                  data-testid="inspector-collapse"
                >
                  <ChevronIcon direction="right" />
                </button>
              </div>
              <div className="trace-workspace__inspector-body">{inspector}</div>
            </>
          ) : (
            <button
              type="button"
              className="trace-workspace__inspector-rail"
              onClick={onToggleInspector}
              aria-expanded={false}
              aria-label="Show inspector"
              title="Show inspector"
              data-testid="inspector-expand"
            >
              <InspectorIcon />
            </button>
          )}
        </aside>
      </div>
    </div>
  );
}
