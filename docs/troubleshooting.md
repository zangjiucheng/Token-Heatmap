# Troubleshooting

### `TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not BatchEncoding`

You're on a transformers version that leaves a `BatchEncoding` wrapper after
`.to(device)`. `generation.py` already extracts `input_ids` defensively — pull
`main` and re-run.

### Frontend says "trace failed schema validation"

Expand the **Show N validation issues** block on the error banner to see the
exact JSON-Pointer path the validator rejected. The most common causes:

- A producer emitting an older trace shape (e.g. parallel `top_k_token_ids` arrays inside `logit_lens` instead of a flat `top_k` array of candidates). Update the producer to use `serialize_trace_to_json`; see [`schema.md`](schema.md).
- A float overshoot like `top_mass_used: 1.0000001`. The current serializer clamps to `[0, 1]` at the JSON boundary; if you're producing JSON by hand, do the same.

### Attention tab shows "trace generated without `--capture-attention`"

The frontend gates that tab on the top-level `attention_metadata` block. If
you're using the CLI, pass `--capture-attention`. If you're writing the JSON
yourself, build an `AttentionMetadata` dict from the probe's
`num_attention_heads`, `num_key_value_heads`, `head_dim`, and
`target_layers` and pass it into `serialize_trace_to_json` instead of `None`.
[`examples/qwen_attention_inspect.py`](https://github.com/zangjiucheng/Token-Heatmap/blob/main/examples/qwen_attention_inspect.py)
does this end-to-end.

### Gated model (e.g. Llama) returns 401

```bash
export HUGGINGFACE_HUB_TOKEN=hf_...
```

…and retry.

### CLI generation takes minutes on first run for a new model

That's the HuggingFace download. Subsequent runs hit the on-disk cache
(`~/.cache/huggingface` by default; override with `HF_HOME`) and are fast.

### Heatmap is huge / unreadable for long generations

Use the step-range slider in the SPA, or pass `--max-new-tokens` lower on the
CLI, or only render `source="raw"` in the matplotlib plot.

### Frontend always shows the bundled sample after I drop a file

Fixed: the landing page now seeds the trace store with the loaded trace and
routes to `/trace/uploaded` (or `/trace/uploaded-csv`) instead of
`/trace/sample`. Pull `main` and re-run.
