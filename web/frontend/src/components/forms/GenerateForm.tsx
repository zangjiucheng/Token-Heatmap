import { useState, type FormEventHandler } from 'react';
import type { GenerateParams } from '@/api/client';
import './GenerateForm.css';

export interface GenerateFormProps {
  /** Called with the assembled, numerically-parsed params on submit. */
  onGenerate: (params: GenerateParams) => void;
  /** Disable the whole form (e.g. while a generation is already running). */
  disabled?: boolean;
}

const DEFAULTS = {
  model: 'Qwen/Qwen2.5-0.5B-Instruct',
  prompt: '',
  max_new_tokens: 64,
  temperature: 0.8,
  top_p: 0.95,
  min_k: 8,
  max_k: 64,
  mass_threshold: 0.95,
};

export function GenerateForm({ onGenerate, disabled = false }: GenerateFormProps) {
  const [model, setModel] = useState(DEFAULTS.model);
  const [prompt, setPrompt] = useState(DEFAULTS.prompt);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULTS.max_new_tokens);
  const [temperature, setTemperature] = useState(DEFAULTS.temperature);
  const [topP, setTopP] = useState(DEFAULTS.top_p);
  const [minK, setMinK] = useState(DEFAULTS.min_k);
  const [maxK, setMaxK] = useState(DEFAULTS.max_k);
  const [massThreshold, setMassThreshold] = useState(DEFAULTS.mass_threshold);
  const [captureAttention, setCaptureAttention] = useState(false);
  const [captureLogitLens, setCaptureLogitLens] = useState(false);
  const [captureActivations, setCaptureActivations] = useState(false);

  const canSubmit = model.trim().length > 0 && prompt.trim().length > 0 && !disabled;

  const handleSubmit: FormEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault();
    if (!canSubmit) return;
    onGenerate({
      model: model.trim(),
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      min_k: minK,
      max_k: maxK,
      mass_threshold: massThreshold,
      capture_attention: captureAttention,
      capture_logit_lens: captureLogitLens,
      capture_activations: captureActivations,
    });
  };

  return (
    <form className="generate-form" onSubmit={handleSubmit} aria-labelledby="generate-form-title">
      <div className="generate-form__copy">
        <p className="generate-form__eyebrow">Generate</p>
        <h2 id="generate-form-title" className="generate-form__title">
          Run a model
        </h2>
        <p className="generate-form__description">
          Generate a fresh trace from a prompt. Inference runs on the backend
          (GPU), so this can take a while for larger models.
        </p>
      </div>

      <div className="generate-form__field">
        <label className="generate-form__label" htmlFor="generate-model">
          Model
        </label>
        <input
          id="generate-model"
          type="text"
          className="generate-form__input"
          value={model}
          disabled={disabled}
          onChange={(e) => setModel(e.target.value)}
          placeholder="Qwen/Qwen2.5-0.5B-Instruct"
        />
      </div>

      <div className="generate-form__field">
        <label className="generate-form__label" htmlFor="generate-prompt">
          Prompt
        </label>
        <textarea
          id="generate-prompt"
          className="generate-form__textarea"
          value={prompt}
          disabled={disabled}
          rows={3}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Explain what a large language model is in one sentence."
        />
      </div>

      <div className="generate-form__row">
        <div className="generate-form__field">
          <label className="generate-form__label" htmlFor="generate-max-new-tokens">
            Max new tokens
          </label>
          <input
            id="generate-max-new-tokens"
            type="number"
            className="generate-form__input"
            min={1}
            max={512}
            step={1}
            value={maxNewTokens}
            disabled={disabled}
            onChange={(e) => setMaxNewTokens(e.target.valueAsNumber)}
          />
        </div>
        <div className="generate-form__field">
          <label className="generate-form__label" htmlFor="generate-temperature">
            Temperature
          </label>
          <input
            id="generate-temperature"
            type="number"
            className="generate-form__input"
            min={0}
            max={5}
            step={0.05}
            value={temperature}
            disabled={disabled}
            onChange={(e) => setTemperature(e.target.valueAsNumber)}
          />
        </div>
        <div className="generate-form__field">
          <label className="generate-form__label" htmlFor="generate-top-p">
            Top-p
          </label>
          <input
            id="generate-top-p"
            type="number"
            className="generate-form__input"
            min={0}
            max={1}
            step={0.01}
            value={topP}
            disabled={disabled}
            onChange={(e) => setTopP(e.target.valueAsNumber)}
          />
        </div>
      </div>

      <details className="generate-form__advanced">
        <summary className="generate-form__summary">Advanced</summary>
        <div className="generate-form__row">
          <div className="generate-form__field">
            <label className="generate-form__label" htmlFor="generate-min-k">
              Min-k
            </label>
            <input
              id="generate-min-k"
              type="number"
              className="generate-form__input"
              min={1}
              step={1}
              value={minK}
              disabled={disabled}
              onChange={(e) => setMinK(e.target.valueAsNumber)}
            />
          </div>
          <div className="generate-form__field">
            <label className="generate-form__label" htmlFor="generate-max-k">
              Max-k
            </label>
            <input
              id="generate-max-k"
              type="number"
              className="generate-form__input"
              min={1}
              step={1}
              value={maxK}
              disabled={disabled}
              onChange={(e) => setMaxK(e.target.valueAsNumber)}
            />
          </div>
          <div className="generate-form__field">
            <label className="generate-form__label" htmlFor="generate-mass-threshold">
              Mass threshold
            </label>
            <input
              id="generate-mass-threshold"
              type="number"
              className="generate-form__input"
              min={0}
              max={1}
              step={0.01}
              value={massThreshold}
              disabled={disabled}
              onChange={(e) => setMassThreshold(e.target.valueAsNumber)}
            />
          </div>
        </div>

        <fieldset className="generate-form__captures">
          <legend className="generate-form__label">Captures (inline)</legend>
          <label className="generate-form__checkbox">
            <input
              type="checkbox"
              checked={captureAttention}
              disabled={disabled}
              onChange={(e) => setCaptureAttention(e.target.checked)}
            />
            Attention
          </label>
          <label className="generate-form__checkbox">
            <input
              type="checkbox"
              checked={captureLogitLens}
              disabled={disabled}
              onChange={(e) => setCaptureLogitLens(e.target.checked)}
            />
            Logit lens
          </label>
          <label className="generate-form__checkbox">
            <input
              type="checkbox"
              checked={captureActivations}
              disabled={disabled}
              onChange={(e) => setCaptureActivations(e.target.checked)}
            />
            Activations
          </label>
        </fieldset>
      </details>

      <div className="generate-form__actions">
        <button type="submit" className="generate-form__submit" disabled={!canSubmit}>
          Generate
        </button>
      </div>
    </form>
  );
}
