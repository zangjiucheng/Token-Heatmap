import {
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
  type RefObject,
} from 'react';
import { PaneGutter } from '@/components/layout/PaneGutter';
import { ChevronIcon, InspectorIcon } from './icons';
import './TraceWorkspace.css';

const INSPECTOR_DEFAULT = 420;
const INSPECTOR_MIN = 300;
const INSPECTOR_MAX = 760;
const INSPECTOR_WIDTH_KEY = 'llm-heatmap-inspector-w';

function clampInspector(w: number): number {
  return Math.max(INSPECTOR_MIN, Math.min(INSPECTOR_MAX, w));
}

function readStoredInspectorWidth(): number {
  try {
    const raw = window.localStorage.getItem(INSPECTOR_WIDTH_KEY);
    const v = raw == null ? NaN : Number(raw);
    return Number.isFinite(v) && v > 0 ? clampInspector(v) : INSPECTOR_DEFAULT;
  } catch {
    return INSPECTOR_DEFAULT;
  }
}

export interface TraceWorkspaceProps {
  /** Trace identity, shown in the sticky sub-header. */
  model: string;
  prompt: string;
  stepCount: number;
  selectedStep: number | null;
  /** Spine — generated-token strip, always visible above the lens. */
  tokenStrip: ReactNode;
  /** Spine — playback transport, pinned left of the token strip. */
  transport?: ReactNode;
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
  /** Whether the spine timelines are expanded. Defaults to true. */
  timelinesOpen?: boolean;
  onToggleTimelines?: () => void;
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
  transport,
  timelines,
  rail,
  controlBar,
  canvas,
  canvasRef,
  inspector,
  inspectorOpen,
  onToggleInspector,
  railCollapsed,
  timelinesOpen = true,
  onToggleTimelines,
}: TraceWorkspaceProps) {
  const selectedLabel =
    selectedStep == null ? 'No step selected' : `Step ${selectedStep}`;

  // Inspector is resizable by dragging the gutter on its left edge; the chosen
  // width persists across reloads. `pendingDelta` tracks an in-progress drag so
  // we only write to storage on commit.
  const [inspectorWidth, setInspectorWidth] = useState(
    readStoredInspectorWidth,
  );
  const [pendingDelta, setPendingDelta] = useState(0);
  const widthRef = useRef(inspectorWidth);
  widthRef.current = inspectorWidth;
  const effectiveWidth = clampInspector(inspectorWidth + pendingDelta);

  const commitInspectorWidth = () => {
    setPendingDelta((delta) => {
      if (delta === 0) return 0;
      const next = clampInspector(widthRef.current + delta);
      setInspectorWidth(next);
      try {
        window.localStorage.setItem(INSPECTOR_WIDTH_KEY, String(next));
      } catch {
        // storage unavailable — keep the in-memory width
      }
      return 0;
    });
  };

  const resetInspectorWidth = () => {
    setPendingDelta(0);
    setInspectorWidth(INSPECTOR_DEFAULT);
    try {
      window.localStorage.setItem(
        INSPECTOR_WIDTH_KEY,
        String(INSPECTOR_DEFAULT),
      );
    } catch {
      // ignore
    }
  };

  const rootStyle = {
    '--rail-w': railCollapsed ? '56px' : '208px',
    '--inspector-w': inspectorOpen ? `${Math.round(effectiveWidth)}px` : '40px',
    '--gutter-w': inspectorOpen ? '6px' : '0px',
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
        <div className="trace-workspace__spine">
          {transport ? (
            <div className="trace-workspace__transport">{transport}</div>
          ) : null}
          {tokenStrip}
        </div>
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
            className="trace-workspace__overview"
            data-open={timelinesOpen ? 'true' : 'false'}
          >
            <button
              type="button"
              className="trace-workspace__overview-toggle"
              onClick={onToggleTimelines}
              aria-expanded={timelinesOpen}
              aria-controls="trace-overview-timelines"
              data-testid="overview-toggle"
            >
              <span className="eyebrow">
                Overview · entropy &amp; probability
              </span>
              <ChevronIcon direction={timelinesOpen ? 'down' : 'up'} />
            </button>
            {timelinesOpen ? (
              <div
                id="trace-overview-timelines"
                className="trace-workspace__timelines"
                aria-label="Trace overview timelines"
                data-testid="trace-viewer-center-timelines"
              >
                {timelines}
              </div>
            ) : null}
          </div>
        </section>

        <div className="trace-workspace__gutter">
          {inspectorOpen ? (
            <PaneGutter
              side="right"
              width={effectiveWidth}
              minWidth={INSPECTOR_MIN}
              maxWidth={INSPECTOR_MAX}
              label="Resize inspector panel"
              onResize={(delta) => setPendingDelta((d) => d + delta)}
              onCommit={commitInspectorWidth}
              onReset={resetInspectorWidth}
              testId="inspector-gutter"
            />
          ) : null}
        </div>

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
