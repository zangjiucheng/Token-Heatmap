import { useEffect, useMemo, useState } from 'react';
import type { Trace } from '@/types/trace';
import { ManifoldScatter } from './ManifoldScatter';
import { ManifoldScatter3D } from './ManifoldScatter3D';
import { ManifoldScreePlot } from './ManifoldScreePlot';
import './ManifoldTab.css';

export interface ManifoldTabProps {
  trace: Trace;
  selectedStep: number | null;
  onSelectStep: (step: number) => void;
  hoveredStep: number | null;
  onHoverStep: (step: number | null) => void;
}

function layerKey(layer: number, submodule: string): string {
  return `${layer}:${submodule}`;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

/**
 * Manifold tab — renders the post-hoc manifold analysis (`trace.manifold`,
 * produced by `token-heatmap manifold`). For the chosen (layer, submodule) it
 * shows the 2-D PCA projection of the activation cloud, a scree plot of the
 * variance spectrum, and the scalar geometry metrics (participation ratio,
 * TwoNN intrinsic dimension, trajectory curvature, periodicity).
 */
export function ManifoldTab({
  trace,
  selectedStep,
  onSelectStep,
  hoveredStep,
  onHoverStep,
}: ManifoldTabProps) {
  const layers = useMemo(() => trace.manifold?.layers ?? [], [trace.manifold]);
  const [selectedKey, setSelectedKey] = useState<string>(() =>
    layers.length ? layerKey(layers[0].layer, layers[0].submodule) : '',
  );
  const [xComponent, setXComponent] = useState(0);
  const [yComponent, setYComponent] = useState(1);
  const [view, setView] = useState<'2d' | '3d'>('3d');

  // Keep the selection valid if the trace (and thus its layers) changes.
  useEffect(() => {
    if (!layers.length) return;
    const keys = layers.map((l) => layerKey(l.layer, l.submodule));
    if (!keys.includes(selectedKey)) setSelectedKey(keys[0]);
  }, [layers, selectedKey]);

  const active = useMemo(
    () =>
      layers.find((l) => layerKey(l.layer, l.submodule) === selectedKey) ??
      layers[0],
    [layers, selectedKey],
  );

  if (!trace.manifold || layers.length === 0 || !active) {
    return (
      <div
        className="manifold-tab manifold-tab--empty"
        data-testid="manifold-tab-empty"
        role="region"
        aria-label="Manifold tab empty state"
      >
        <p>
          This trace has no manifold analysis. Generate a trace with{' '}
          <code>--capture-activations --capture-full-activations</code>, then
          run <code>token-heatmap manifold --trace &lt;file&gt;</code> to add
          it.
        </p>
      </div>
    );
  }

  const components = active.projection.n_components;
  const componentOptions = Array.from(
    { length: Math.max(1, components) },
    (_, i) => i,
  );
  // Clamp axis selectors to the available components for this cloud.
  const safeX = Math.min(xComponent, components - 1);
  const safeY = Math.min(yComponent, Math.max(0, components - 1));
  // 3-D needs three components; otherwise fall back to the 2-D projection.
  const can3d = components >= 3;
  const activeView: '2d' | '3d' = can3d ? view : '2d';

  return (
    <div
      className="manifold-tab"
      data-testid="manifold-tab-content"
      role="region"
      aria-label="Manifold tab"
    >
      <div className="manifold-tab__controls">
        <div className="manifold-tab__control">
          <label htmlFor="manifold-layer-select">Layer · submodule</label>
          <select
            id="manifold-layer-select"
            value={selectedKey}
            onChange={(e) => setSelectedKey(e.target.value)}
            data-testid="manifold-layer-select"
          >
            {layers.map((l) => {
              const key = layerKey(l.layer, l.submodule);
              return (
                <option key={key} value={key}>
                  {`L${l.layer} · ${l.submodule}`}
                </option>
              );
            })}
          </select>
        </div>
        {can3d && (
          <div className="manifold-tab__control">
            <span className="manifold-tab__control-label">View</span>
            <div
              className="manifold-tab__view"
              role="group"
              aria-label="Projection view"
            >
              <button
                type="button"
                className="manifold-tab__view-option"
                data-selected={activeView === '3d' ? 'true' : 'false'}
                data-testid="manifold-view-3d"
                onClick={() => setView('3d')}
              >
                3D
              </button>
              <button
                type="button"
                className="manifold-tab__view-option"
                data-selected={activeView === '2d' ? 'true' : 'false'}
                data-testid="manifold-view-2d"
                onClick={() => setView('2d')}
              >
                2D
              </button>
            </div>
          </div>
        )}
        {activeView === '2d' && (
          <>
            <div className="manifold-tab__control">
              <label htmlFor="manifold-x-select">X axis</label>
              <select
                id="manifold-x-select"
                value={safeX}
                onChange={(e) => setXComponent(Number(e.target.value))}
                data-testid="manifold-x-select"
              >
                {componentOptions.map((i) => (
                  <option key={i} value={i}>{`PC${i + 1}`}</option>
                ))}
              </select>
            </div>
            <div className="manifold-tab__control">
              <label htmlFor="manifold-y-select">Y axis</label>
              <select
                id="manifold-y-select"
                value={safeY}
                onChange={(e) => setYComponent(Number(e.target.value))}
                data-testid="manifold-y-select"
              >
                {componentOptions.map((i) => (
                  <option key={i} value={i}>{`PC${i + 1}`}</option>
                ))}
              </select>
            </div>
          </>
        )}
      </div>

      <div className="manifold-tab__body">
        <div className="manifold-tab__scatter">
          {activeView === '3d' ? (
            <>
              <ManifoldScatter3D
                coords={active.projection.coords}
                positions={active.positions}
                selectedStep={selectedStep}
                hoveredStep={hoveredStep}
                onSelectStep={onSelectStep}
                onHoverStep={onHoverStep}
              />
              <p className="manifold-tab__hint">
                Drag to rotate · colour follows generation step
              </p>
            </>
          ) : (
            <ManifoldScatter
              coords={active.projection.coords}
              positions={active.positions}
              xComponent={safeX}
              yComponent={safeY}
              selectedStep={selectedStep}
              hoveredStep={hoveredStep}
              onSelectStep={onSelectStep}
              onHoverStep={onHoverStep}
            />
          )}
        </div>

        <div className="manifold-tab__side">
          <dl className="manifold-tab__metrics" data-testid="manifold-metrics">
            <div className="manifold-tab__metric">
              <dt>Participation ratio</dt>
              <dd data-testid="manifold-metric-pr">
                {formatNumber(active.participation_ratio)}
              </dd>
            </div>
            <div className="manifold-tab__metric">
              <dt>Intrinsic dim (TwoNN)</dt>
              <dd data-testid="manifold-metric-twonn">
                {formatNumber(active.intrinsic_dimension.twonn)}
              </dd>
            </div>
            <div className="manifold-tab__metric">
              <dt>Trajectory curvature</dt>
              <dd data-testid="manifold-metric-curvature">
                {formatNumber(active.trajectory_curvature.mean, 3)}
              </dd>
            </div>
            <div className="manifold-tab__metric">
              <dt>Periodicity (period · power)</dt>
              <dd data-testid="manifold-metric-periodicity">
                {`${formatNumber(active.periodicity.dominant_period, 1)} · ${formatNumber(
                  active.periodicity.power,
                )}`}
              </dd>
            </div>
          </dl>

          <div className="manifold-tab__scree">
            <h3 className="manifold-tab__scree-title">Variance spectrum</h3>
            <ManifoldScreePlot
              explainedVarianceRatio={active.pca.explained_variance_ratio}
              cumulativeVarianceRatio={active.pca.cumulative_variance_ratio}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
