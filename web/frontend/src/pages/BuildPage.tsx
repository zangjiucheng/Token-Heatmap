import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BackendStatusBanner } from '@/components/feedback/BackendStatusBanner';
import {
  DEFAULT_BUILD_CONFIG,
  NodeConfigGraph,
  buildConfigToParams,
  buildConfigToYaml,
  type BuildConfig,
} from '@/features/build';
import { useBackendHealth } from '@/hooks/useBackendHealth';
import { useTrace } from '@/hooks/useTrace';
import { putTrace } from '@/lib/trace/store';
import './BuildPage.css';

/**
 * Build page — a visual node editor for assembling a trace config and either
 * running it live on the backend (`/trace/generate`, then jump to the viewer)
 * or exporting the equivalent `token-heatmap trace --config` YAML for the CLI.
 * Replaces the old flat generate form with a wired Input→…→Output pipeline.
 */
export function BuildPage() {
  const { trace, load, status, error } = useTrace();
  const health = useBackendHealth();
  const navigate = useNavigate();
  const [config, setConfig] = useState<BuildConfig>(DEFAULT_BUILD_CONFIG);
  const [pendingNav, setPendingNav] = useState(false);

  // Once the live run resolves, seed the store and jump to the viewer.
  useEffect(() => {
    if (trace && pendingNav) {
      putTrace('generated', trace);
      navigate('/trace/generated');
      setPendingNav(false);
    }
  }, [trace, pendingNav, navigate]);

  const patch = (p: Partial<BuildConfig>) =>
    setConfig((prev) => ({ ...prev, ...p }));

  const handleRun = () => {
    setPendingNav(true);
    void load({ type: 'generate', params: buildConfigToParams(config) });
  };

  const handleExport = () => {
    const yaml = buildConfigToYaml(config);
    const blob = new Blob([yaml], { type: 'text/yaml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'trace-config.yaml';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  // This page only ever triggers `generate` loads, so loading == a run is in
  // flight and an error == that run failed.
  const running = status === 'loading';
  const errorMessage = status === 'error' && error ? error.message : null;

  const yamlPreview = useMemo(() => buildConfigToYaml(config), [config]);

  return (
    <div className="build-page" data-testid="build-page">
      <header className="build-page__intro">
        <p className="build-page__eyebrow">Build a trace</p>
        <h1 className="build-page__title">Config pipeline</h1>
        <p className="build-page__description">
          Wire the run from prompt to output. Each node is one concern — edit
          them, then <strong>Run</strong> to generate live on the backend, or{' '}
          <strong>Export YAML</strong> to drive the CLI on another machine.
        </p>
      </header>

      <div className="build-page__status">
        <BackendStatusBanner
          status={health.status}
          onRetry={() => void health.probe()}
        />
      </div>

      <NodeConfigGraph
        config={config}
        onChange={patch}
        onRun={handleRun}
        onExport={handleExport}
        running={running}
        error={errorMessage}
      />

      <details className="build-page__yaml">
        <summary className="build-page__yaml-summary">YAML preview</summary>
        <pre className="build-page__yaml-pre" data-testid="build-yaml-preview">
          {yamlPreview}
        </pre>
      </details>
    </div>
  );
}
