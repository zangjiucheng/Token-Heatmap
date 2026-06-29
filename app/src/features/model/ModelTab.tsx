import { useMemo } from 'react';
import type { Trace } from '@/types/trace';
import type { TraceWithActivations } from '@/types/activation';
import './ModelTab.css';

export interface ModelTabProps {
  trace: Trace;
}

/** Humanize a parameter count: 1234 → "1.2K", 7.6e9 → "7.6B". */
function humanizeCount(n: number | undefined): string | null {
  if (n == null || !Number.isFinite(n) || n <= 0) return null;
  const units: [number, string][] = [
    [1e12, 'T'],
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ];
  for (const [scale, suffix] of units) {
    if (n >= scale) {
      const v = n / scale;
      return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)}${suffix}`;
    }
  }
  return String(n);
}

function fmtInt(n: number | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toLocaleString('en-US');
}

/** Pull the best available structural fields, preferring the self-contained
 *  `model_architecture` block but falling back to the probe metadata blocks so
 *  older traces (pre-`model_architecture`) still render a partial summary. */
function useArchitecture(trace: Trace) {
  return useMemo(() => {
    const arch = trace.model_architecture ?? {};
    const attn = trace.attention_metadata;
    const act = (trace as TraceWithActivations).activation_metadata;
    const meta = trace.metadata;

    const numLayers =
      arch.num_layers ?? attn?.num_layers ?? act?.num_layers ?? undefined;
    const numHeads = arch.num_attention_heads ?? attn?.num_attention_heads;
    const numKvHeads =
      arch.num_key_value_heads ?? attn?.num_key_value_heads ?? numHeads;
    const headDim = arch.head_dim ?? attn?.head_dim;
    const hiddenSize =
      arch.hidden_size ??
      act?.hidden_dim ??
      (numHeads && headDim ? numHeads * headDim : undefined);
    const vocabSize = arch.vocab_size ?? meta?.vocab_size ?? undefined;
    const dtype = arch.dtype ?? meta?.dtype ?? undefined;

    const gqaGroup =
      numHeads && numKvHeads && numKvHeads > 0
        ? numHeads / numKvHeads
        : undefined;

    return {
      name: meta?.model ?? 'Unknown model',
      architecture: arch.architecture,
      modelType: arch.model_type,
      numLayers,
      hiddenSize,
      numHeads,
      numKvHeads,
      headDim,
      gqaGroup,
      intermediateSize: arch.intermediate_size,
      vocabSize,
      maxPos: arch.max_position_embeddings,
      ropeTheta: arch.rope_theta,
      tied: arch.tie_word_embeddings,
      numParams: arch.num_parameters,
      dtype,
      device: meta?.device,
      // Probe coverage (which layers/submodules were actually captured).
      capturedAttnLayers: attn?.captured_layers,
      capturedActLayers: act?.captured_layers,
      capturedSubmodules: act?.captured_submodules,
    };
  }, [trace]);
}

interface StatProps {
  label: string;
  value: string;
  hint?: string;
}

function Stat({ label, value, hint }: StatProps) {
  return (
    <div className="model-tab__stat">
      <dt className="model-tab__stat-label">{label}</dt>
      <dd className="model-tab__stat-value">{value}</dd>
      {hint ? <span className="model-tab__stat-hint">{hint}</span> : null}
    </div>
  );
}

/**
 * Model tab — a structural overview of the traced model read from
 * `trace.model_architecture` (with graceful fallback to attention/activation
 * metadata for older traces). Shows a headline spec card, a layer-stack
 * schematic (embedding → N× decoder block → norm → unembed), and a full
 * dimensions table. Probe coverage is overlaid on the schematic so you can see
 * which layers/submodules a given trace actually captured.
 */
export function ModelTab({ trace }: ModelTabProps) {
  const a = useArchitecture(trace);

  const params = humanizeCount(a.numParams);
  const headline: string[] = [];
  if (a.modelType) headline.push(a.modelType);
  if (params) headline.push(`${params} params`);
  if (a.dtype) headline.push(a.dtype);
  if (a.device) headline.push(a.device);

  const hasAnyDims =
    a.numLayers != null ||
    a.hiddenSize != null ||
    a.numHeads != null ||
    a.vocabSize != null;

  const attnCovered = a.capturedAttnLayers?.length ?? 0;
  const actCovered = a.capturedActLayers?.length ?? 0;
  const coverageNotes: string[] = [];
  if (attnCovered) coverageNotes.push(`attention · ${attnCovered} layers`);
  if (actCovered && a.capturedSubmodules?.length)
    coverageNotes.push(
      `activations · ${a.capturedSubmodules.join(', ')} @ ${actCovered} layers`,
    );

  return (
    <div className="model-tab" data-testid="model-tab">
      {/* Headline spec card */}
      <header className="model-tab__hero">
        <div className="model-tab__hero-main">
          <div className="model-tab__eyebrow">Model architecture</div>
          <h3 className="model-tab__title">{a.name}</h3>
          {a.architecture ? (
            <code className="model-tab__arch-chip">{a.architecture}</code>
          ) : null}
        </div>
        {headline.length ? (
          <ul className="model-tab__hero-tags" aria-label="Model summary">
            {headline.map((t) => (
              <li key={t} className="model-tab__hero-tag">
                {t}
              </li>
            ))}
          </ul>
        ) : null}
      </header>

      {!hasAnyDims ? (
        <p className="model-tab__empty" data-testid="model-tab-empty">
          This trace carries no architecture metadata. Re-run the CLI to capture
          a <code>model_architecture</code> block, or attach an attention /
          activation probe.
        </p>
      ) : (
        <div className="model-tab__body">
          {/* Layer-stack schematic */}
          <section
            className="model-tab__schematic"
            aria-label="Architecture schematic"
          >
            <div className="model-tab__block model-tab__block--io">
              <span className="model-tab__block-name">Token Embedding</span>
              <span className="model-tab__block-dims">
                {fmtInt(a.vocabSize)} × {fmtInt(a.hiddenSize)}
              </span>
            </div>

            <div className="model-tab__flow" aria-hidden="true" />

            <div className="model-tab__block model-tab__block--decoder">
              <div className="model-tab__decoder-head">
                <span className="model-tab__block-name">Decoder block</span>
                <span className="model-tab__repeat-badge">
                  × {a.numLayers ?? '—'}
                </span>
              </div>
              <div className="model-tab__subblocks">
                <div className="model-tab__subblock">
                  <span className="model-tab__subblock-name">
                    Self-Attention
                  </span>
                  <ul className="model-tab__subblock-facts">
                    <li>
                      <strong>{fmtInt(a.numHeads)}</strong> query heads
                    </li>
                    <li>
                      <strong>{fmtInt(a.numKvHeads)}</strong> KV heads
                      {a.gqaGroup && a.gqaGroup > 1 ? (
                        <span className="model-tab__gqa">
                          {' '}
                          GQA ×{a.gqaGroup}
                        </span>
                      ) : null}
                    </li>
                    <li>
                      head dim <strong>{fmtInt(a.headDim)}</strong>
                    </li>
                  </ul>
                </div>
                <div className="model-tab__subblock">
                  <span className="model-tab__subblock-name">MLP</span>
                  <ul className="model-tab__subblock-facts">
                    <li>
                      d_model <strong>{fmtInt(a.hiddenSize)}</strong>
                    </li>
                    <li>
                      d_ff <strong>{fmtInt(a.intermediateSize)}</strong>
                    </li>
                    {a.hiddenSize && a.intermediateSize ? (
                      <li>
                        expansion{' '}
                        <strong>
                          ×{(a.intermediateSize / a.hiddenSize).toFixed(1)}
                        </strong>
                      </li>
                    ) : null}
                  </ul>
                </div>
              </div>
            </div>

            <div className="model-tab__flow" aria-hidden="true" />

            <div className="model-tab__block model-tab__block--io">
              <span className="model-tab__block-name">
                Final Norm → LM Head
              </span>
              <span className="model-tab__block-dims">
                {fmtInt(a.hiddenSize)} × {fmtInt(a.vocabSize)}
                {a.tied ? ' · tied' : ''}
              </span>
            </div>

            {coverageNotes.length ? (
              <p className="model-tab__coverage">
                Captured in this trace: {coverageNotes.join('  ·  ')}
              </p>
            ) : null}
          </section>

          {/* Dimensions table */}
          <section
            className="model-tab__specs"
            aria-label="Model dimensions"
          >
            <h4 className="model-tab__specs-title">Dimensions</h4>
            <dl className="model-tab__stats">
              <Stat label="Layers" value={fmtInt(a.numLayers)} />
              <Stat label="Hidden size" value={fmtInt(a.hiddenSize)} />
              <Stat label="Attention heads" value={fmtInt(a.numHeads)} />
              <Stat
                label="KV heads"
                value={fmtInt(a.numKvHeads)}
                hint={
                  a.gqaGroup && a.gqaGroup > 1
                    ? `GQA group ×${a.gqaGroup}`
                    : 'MHA'
                }
              />
              <Stat label="Head dim" value={fmtInt(a.headDim)} />
              <Stat label="MLP intermediate" value={fmtInt(a.intermediateSize)} />
              <Stat label="Vocab size" value={fmtInt(a.vocabSize)} />
              <Stat
                label="Max context"
                value={fmtInt(a.maxPos)}
                hint={a.ropeTheta ? `RoPE θ ${humanizeCount(a.ropeTheta)}` : undefined}
              />
              <Stat
                label="Parameters"
                value={params ?? fmtInt(a.numParams)}
                hint={a.numParams ? fmtInt(a.numParams) : undefined}
              />
              <Stat
                label="Precision"
                value={a.dtype ?? '—'}
                hint={a.device}
              />
            </dl>
          </section>
        </div>
      )}
    </div>
  );
}
