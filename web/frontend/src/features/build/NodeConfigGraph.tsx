import { BuildNode, PipelineConnector } from './BuildNode';
import { MODEL_PRESETS, isBuildConfigValid, type BuildConfig } from './config';
import './NodeConfigGraph.css';

export interface NodeConfigGraphProps {
  config: BuildConfig;
  onChange: (patch: Partial<BuildConfig>) => void;
  /** Run the config live via the backend `/trace/generate` endpoint. */
  onRun: () => void;
  /** Download the equivalent `token-heatmap trace --config` YAML. */
  onExport: () => void;
  /** True while a generation is in flight (disables inputs + Run). */
  running?: boolean;
  /** Optional error/status message rendered in the Output node. */
  error?: string | null;
}

/** Guard `input.valueAsNumber` (NaN when the field is cleared). */
function numOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

/**
 * The visual node editor: Input → Model → Sampling → Capture → Output, wired by
 * directional connectors. Each concern of the old flat generate form is its own
 * node. The Output node both runs the pipeline live (backend `/trace/generate`)
 * and exports the equivalent YAML config for the CLI. Fully controlled.
 */
export function NodeConfigGraph({
  config,
  onChange,
  onRun,
  onExport,
  running = false,
  error = null,
}: NodeConfigGraphProps) {
  const valid = isBuildConfigValid(config);

  return (
    <div className="node-graph" data-testid="node-config-graph">
      <div className="node-graph__rail">
        {/* 1 · Input */}
        <BuildNode
          step={1}
          title="Input"
          subtitle="Prompt"
          hasInput={false}
        >
          <label className="node-field">
            <span className="node-field__label">Prompt</span>
            <textarea
              className="node-field__textarea"
              rows={5}
              value={config.prompt}
              disabled={running}
              placeholder="Explain what a large language model is in one sentence."
              onChange={(e) => onChange({ prompt: e.target.value })}
              data-testid="node-input-prompt"
            />
          </label>
        </BuildNode>

        <PipelineConnector />

        {/* 2 · Model */}
        <BuildNode step={2} title="Model" subtitle="HF model id">
          <label className="node-field">
            <span className="node-field__label">Model</span>
            <input
              type="text"
              className="node-field__input"
              value={config.model}
              disabled={running}
              placeholder="Qwen/Qwen2.5-0.5B-Instruct"
              onChange={(e) => onChange({ model: e.target.value })}
              data-testid="node-input-model"
            />
          </label>
          <div className="node-presets" role="group" aria-label="Model presets">
            {MODEL_PRESETS.map((preset) => (
              <button
                key={preset}
                type="button"
                className={
                  preset === config.model
                    ? 'node-preset node-preset--active'
                    : 'node-preset'
                }
                disabled={running}
                onClick={() => onChange({ model: preset })}
              >
                {preset.replace(/^Qwen\//, '')}
              </button>
            ))}
          </div>
        </BuildNode>

        <PipelineConnector />

        {/* 3 · Sampling */}
        <BuildNode step={3} title="Sampling" subtitle="Decode + probe">
          <div className="node-grid">
            <label className="node-field">
              <span className="node-field__label">Max new tokens</span>
              <input
                type="number"
                className="node-field__input"
                min={1}
                max={512}
                step={1}
                value={config.max_new_tokens}
                disabled={running}
                onChange={(e) =>
                  onChange({ max_new_tokens: numOr(e.target.valueAsNumber, 64) })
                }
                data-testid="node-input-max-new-tokens"
              />
            </label>
            <label className="node-field">
              <span className="node-field__label">Temperature</span>
              <input
                type="number"
                className="node-field__input"
                min={0}
                max={5}
                step={0.05}
                value={config.temperature}
                disabled={running}
                onChange={(e) =>
                  onChange({ temperature: numOr(e.target.valueAsNumber, 0.8) })
                }
              />
            </label>
            <label className="node-field">
              <span className="node-field__label">Top-p</span>
              <input
                type="number"
                className="node-field__input"
                min={0}
                max={1}
                step={0.01}
                value={config.top_p}
                disabled={running}
                onChange={(e) =>
                  onChange({ top_p: numOr(e.target.valueAsNumber, 0.95) })
                }
              />
            </label>
            <label className="node-field">
              <span className="node-field__label">Min-k</span>
              <input
                type="number"
                className="node-field__input"
                min={1}
                step={1}
                value={config.min_k}
                disabled={running}
                onChange={(e) =>
                  onChange({ min_k: numOr(e.target.valueAsNumber, 8) })
                }
              />
            </label>
            <label className="node-field">
              <span className="node-field__label">Max-k</span>
              <input
                type="number"
                className="node-field__input"
                min={1}
                step={1}
                value={config.max_k}
                disabled={running}
                onChange={(e) =>
                  onChange({ max_k: numOr(e.target.valueAsNumber, 64) })
                }
              />
            </label>
            <label className="node-field">
              <span className="node-field__label">Mass threshold</span>
              <input
                type="number"
                className="node-field__input"
                min={0}
                max={1}
                step={0.01}
                value={config.mass_threshold}
                disabled={running}
                onChange={(e) =>
                  onChange({
                    mass_threshold: numOr(e.target.valueAsNumber, 0.95),
                  })
                }
              />
            </label>
          </div>
        </BuildNode>

        <PipelineConnector />

        {/* 4 · Capture */}
        <BuildNode step={4} title="Capture" subtitle="Inline probes">
          <fieldset className="node-captures">
            <legend className="node-field__label">What to record</legend>
            <label className="node-check">
              <input
                type="checkbox"
                checked={config.capture_attention}
                disabled={running}
                onChange={(e) =>
                  onChange({ capture_attention: e.target.checked })
                }
                data-testid="node-capture-attention"
              />
              <span>Attention</span>
            </label>
            <label className="node-check">
              <input
                type="checkbox"
                checked={config.capture_logit_lens}
                disabled={running}
                onChange={(e) =>
                  onChange({ capture_logit_lens: e.target.checked })
                }
              />
              <span>Logit lens</span>
            </label>
            <label className="node-check">
              <input
                type="checkbox"
                checked={config.capture_activations}
                disabled={running}
                onChange={(e) =>
                  onChange({ capture_activations: e.target.checked })
                }
              />
              <span>Activations</span>
            </label>
          </fieldset>
        </BuildNode>

        <PipelineConnector />

        {/* 5 · Output */}
        <BuildNode
          step={5}
          title="Output"
          subtitle="Run or export"
          hasOutput={false}
        >
          <label className="node-field">
            <span className="node-field__label">Output dir (YAML only)</span>
            <input
              type="text"
              className="node-field__input"
              value={config.out}
              disabled={running}
              placeholder="outputs/run"
              onChange={(e) => onChange({ out: e.target.value })}
              data-testid="node-input-out"
            />
          </label>
          <div className="node-actions">
            <button
              type="button"
              className="node-action node-action--run"
              disabled={!valid || running}
              onClick={onRun}
              data-testid="node-run"
            >
              {running ? 'Running…' : '▶ Run'}
            </button>
            <button
              type="button"
              className="node-action node-action--export"
              disabled={!valid}
              onClick={onExport}
              data-testid="node-export"
            >
              ⤓ Export YAML
            </button>
          </div>
          {!valid ? (
            <p className="node-hint" data-testid="node-invalid-hint">
              Set a model and a prompt to run or export.
            </p>
          ) : null}
          {error ? (
            <p className="node-error" role="alert" data-testid="node-error">
              {error}
            </p>
          ) : null}
        </BuildNode>
      </div>
    </div>
  );
}
