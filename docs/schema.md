# Trace schema

The on-disk JSON format produced by `serialize_trace_to_json` (and therefore by
the CLI, the example scripts, and the `POST /trace/convert-csv` endpoint).

The canonical machine-readable definition is
[`web/trace.schema.json`](web/trace.schema.json) (JSON Schema Draft 2020-12) —
this page is the human-readable companion.

## Top-level shape

| Field                | Type   | Required                | Notes                                                                                                                                                                    |
| -------------------- | ------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `schema_version`     | string | ✓                       | Semver, e.g. `"2.0.0"`. Consumers must hard-fail on an unknown major.                                                                                                    |
| `metadata`           | object | ✓                       | Model identity, prompt, sampling/probe parameters.                                                                                                                       |
| `tokens`             | object | ✓                       | Prompt token ids and decoded text (lets the UI render the prompt with the same hover affordances as generated tokens).                                                   |
| `steps`              | array  | ✓                       | Per-generation-step records. `steps[i].step` MUST equal `i`. Iteration ends early on EOS, so length ≤ `metadata.generation_params.max_new_tokens`.                       |
| `attention_metadata` | object | when attention captured | Architecture metadata describing the attention captures. Present whenever any step carries an `attention` array; absent for traces produced without an `AttentionProbe`. |

## `metadata`

Required: `model`, `prompt`, `generated_text`, `generated_at`,
`generation_params`, `probe_config`.

Optional: `system_prompt`, `use_chat_template`, `device` (`cpu` | `cuda` | `mps`),
`dtype`, `vocab_size`.

`generation_params` carries `max_new_tokens`, `temperature`, `top_p`,
`sample_top_k` (0 disables top-k filtering).

`probe_config` carries `min_k`, `max_k`, `mass_threshold`, and an optional `eps`.

## `steps[i]`

| Field                   | Type           | Notes                                                                    |
| ----------------------- | -------------- | ------------------------------------------------------------------------ |
| `step`                  | integer        | Zero-indexed position; MUST equal `i`.                                   |
| `selected`              | object         | The token actually appended at this step (`token_id`, `token`).          |
| `raw`                   | object         | Adaptive probe on the raw temperature-scaled logits (pre top-p / top-k). |
| `processed`             | object         | Adaptive probe on the post-sampling-filter logits.                       |
| `logit_lens`            | array          | Optional; per-layer logit-lens projections.                              |
| `attention`             | array          | Optional Tier-1 inline attention summary, one entry per captured layer.  |
| `attention_sidecar_ref` | string or null | URI pointer to a Tier-2 sidecar file under `<out>/attention/`, or null.  |

Each `Distribution` block (`raw` / `processed`) carries:

- `k_used` — number of candidates kept by the adaptive rule (clamped to `[min_k, max_k]`)
- `entropy` — Shannon entropy of the full next-token distribution, in nats
- `top_mass_used` — cumulative probability mass covered by the kept candidates, in `[0, 1]`
- `selected_prob`, `selected_logprob` — likelihood of the actually-sampled token under this distribution
- `selected_rank` — 1-indexed rank of the selected token across the FULL vocabulary
- `candidates` — array of `{rank, token_id, token, prob, logprob}`, length == `k_used`, sorted by descending `prob`

## `attention_metadata`

Required when any step carries an `attention` array.

```json
{
  "num_layers": 24,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "head_dim": 64,
  "captured_layers": [0, 3, 7, 11]
}
```

`num_key_value_heads` equals `num_attention_heads` for plain MHA models and is
smaller under Grouped Query Attention.

## Attention sidecars

When the producer was asked for the full attention distribution, each step
points at a Tier-2 sidecar file via `attention_sidecar_ref` (relative URI under
`<out>/attention/`). Sidecar files are `numpy.savez_compressed` archives whose
shape is described by [`web/attention-sidecar.schema.json`](web/attention-sidecar.schema.json).

## Activation trace

Produced by an `ActivationProbe`. The CLI embeds the
activation payload **inline** into the parent `adaptive_token_trace.json`:
the trace gains a top-level `activation_metadata` block, and every step
gains `token_id`, `decoded_text_offset`, and `activations` fields. The
projected subset (`schema_version` + `activation_metadata` + step
projections) validates against
[`web/activation.schema.json`](web/activation.schema.json) (JSON Schema
Draft 2020-12), and the relaxed
[`web/trace.schema.json`](web/trace.schema.json) declares the extra fields
as optional so the same file passes both schemas.

### Top-level shape

| Field                 | Type   | Required | Notes                                                                   |
| --------------------- | ------ | -------- | ----------------------------------------------------------------------- |
| `schema_version`      | string | ✓        | Semver, e.g. `"1.0.0"`. Bumped independently of the main trace schema.  |
| `activation_metadata` | object | ✓        | Architecture + tokenizer metadata for the activation captures.          |
| `steps`               | array  | ✓        | Per-generation-step activation records. `steps[i].step` MUST equal `i`. |

### `activation_metadata`

| Field                   | Type             | Required | Notes                                                                                                                    |
| ----------------------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `captured_submodules`   | array of string  | ✓        | Submodule names in capture order (e.g. `["resid_pre", "resid_post", "mlp.down_proj", "o_proj"]`).                        |
| `num_layers`            | integer          | ✓        | Total decoder layers in the model.                                                                                       |
| `hidden_dim`            | integer          | ✓        | Per-layer residual / hidden dimension.                                                                                   |
| `tokenizer_fingerprint` | string           | ✓        | Stable id for the tokenizer that produced the parent trace's tokens.                                                     |
| `captured_layers`       | array of integer | optional | Zero-indexed layers actually captured (ascending; duplicate-free). When omitted, consumers MAY assume `[0, num_layers)`. |

### `steps[i]`

| Field                 | Type                            | Notes                                                                                                                                                                                          |
| --------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `step`                | integer                         | Zero-indexed; MUST equal `i`.                                                                                                                                                                  |
| `token_id`            | integer                         | Selected token id at this step (drives `token_id` alignment in diffs).                                                                                                                         |
| `decoded_text_offset` | integer                         | Character offset of this step's decoded token in the concatenated decoded text. Carried **on every step** so two traces produced by different tokenizers can be aligned position-for-position. |
| `activations`         | array of `ActivationLayerEntry` | One entry per (layer, submodule). Ordering is layer-major then submodule-major following `captured_submodules`.                                                                                |

### `ActivationLayerEntry`

| Field         | Type                      | Notes                                                                             |
| ------------- | ------------------------- | --------------------------------------------------------------------------------- | ----- | --------------------- |
| `layer`       | integer                   | Zero-indexed decoder layer. When `captured_layers` is present, must appear in it. |
| `submodule`   | string                    | Must appear in `activation_metadata.captured_submodules`.                         |
| `l2_norm`     | number                    | L2 norm of the captured activation vector.                                        |
| `mean_abs`    | number                    | Mean absolute value across the `hidden_dim` neurons.                              |
| `sparsity`    | number in `[0, 1]`        | Fraction of neurons below the probe's near-zero threshold.                        |
| `top_neurons` | array of `{index, value}` | Highest-magnitude neurons, sorted by descending `                                 | value | `. `value` is signed. |

## Activation diff

On-disk payload produced by `compare_activations(trace_a, trace_b, ...)`.
Defined by
[`web/activation-diff.schema.json`](web/activation-diff.schema.json) (JSON
Schema Draft 2020-12).

### Top-level shape

| Field            | Type   | Required | Notes                                                                                  |
| ---------------- | ------ | -------- | -------------------------------------------------------------------------------------- |
| `schema_version` | string | ✓        | Semver, e.g. `"1.0.0"`. Bumped independently of the main trace and activation schemas. |
| `alignment`      | object | ✓        | How the two source traces were aligned, plus any unalignable steps.                    |
| `steps`          | array  | ✓        | Per-step diff records for successfully aligned steps.                                  |

### `alignment`

| Field                     | Type                         | Required | Notes                                                                                                                                                                                                                          |
| ------------------------- | ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`                    | string enum                  | ✓        | One of `token_id`, `position`, `auto`. `token_id` zips on identical token ids; `position` zips on `decoded_text_offset` (supports two different tokenizers); `auto` picks `token_id` when fingerprints match, else `position`. |
| `tokenizer_a_fingerprint` | string                       | ✓        | `activation_metadata.tokenizer_fingerprint` of trace A.                                                                                                                                                                        |
| `tokenizer_b_fingerprint` | string                       | ✓        | `activation_metadata.tokenizer_fingerprint` of trace B.                                                                                                                                                                        |
| `mismatches`              | array of `AlignmentMismatch` | ✓        | Steps the comparator could not align. Empty when the two traces aligned cleanly.                                                                                                                                               |

Each `AlignmentMismatch` carries `{step_a, step_b, reason}` — `step_a` or
`step_b` may be `null` when the unmatched step exists only on one side.

### `steps[i]`

| Field                   | Type                  | Notes                                                                                                                                |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `step`                  | integer               | Zero-indexed step in the aligned output.                                                                                             |
| `token_id_a`            | integer               | Selected token id in trace A at this step.                                                                                           |
| `token_id_b`            | integer               | Selected token id in trace B at this step (equal to `token_id_a` under `token_id` alignment; may differ under `position` alignment). |
| `decoded_text_offset_a` | integer               | Char offset of the step's token in trace A's decoded text.                                                                           |
| `decoded_text_offset_b` | integer               | Char offset of the step's token in trace B's decoded text.                                                                           |
| `delta`                 | array of `LayerDelta` | Per (layer, submodule) deltas.                                                                                                       |

### `LayerDelta`

| Field                 | Type                      | Notes                                                                                            |
| --------------------- | ------------------------- | ------------------------------------------------------------------------------------------------ |
| `layer`               | integer                   | Zero-indexed decoder layer.                                                                      |
| `submodule`           | string                    | Must appear in both source traces' `captured_submodules`.                                        |
| `l2`                  | number                    | L2 norm of `(a - b)` at this (layer, submodule).                                                 |
| `cosine`              | number in `[-1, 1]`       | Cosine similarity between trace A's and trace B's activation vectors.                            |
| `top_changed_neurons` | array of `{index, delta}` | Neurons with the largest absolute delta, sorted by descending absolute value; `delta` is signed. |

## Compatibility

The current trace schema is `"2.0.0"`; the activation and activation-diff
schemas each start at `"1.0.0"` and version independently. Bump the major on
breaking changes; consumers must hard-fail on unknown major versions.
