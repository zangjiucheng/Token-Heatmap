# Troubleshooting

### `ModuleNotFoundError: Could not import module 'Qwen2ForCausalLM'`

Missing tokenizer dependencies for Qwen2. Install them:

```bash
pip install tiktoken einops
```

If the error persists, the installed `transformers` version may be a pre-release with broken Qwen2 support. Pin to a known-stable release:

```bash
pip install "transformers==4.46.3" tiktoken einops
```

### `--config` fails with `error: --config requires PyYAML`

```bash
pip install pyyaml
```

### Frontend says "trace failed schema validation"

Expand the **Show N validation issues** block on the error banner to see the
exact JSON-Pointer path the validator rejected. The most common causes:

- A producer emitting an older trace shape. Update the producer to use `serialize_trace_to_json`; see [`schema.md`](schema.md).
- A float overshoot like `top_mass_used: 1.0000001`. The current serializer clamps to `[0, 1]` at the JSON boundary.

### Attention tab shows "trace generated without `--capture-attention`"

Pass `--capture-attention` to the CLI. If you're writing the JSON yourself,
build an `AttentionMetadata` dict and pass it into `serialize_trace_to_json`.

### Logit Lens tab shows empty state

Pass `--capture-logit-lens` to the CLI. The tab always appears in the nav but
shows an empty-state message when the trace lacks `logit_lens` data.

### Gated model (e.g. Llama) returns 401

```bash
export HUGGINGFACE_HUB_TOKEN=hf_...
```

### CLI generation takes minutes on first run for a new model

That's the HuggingFace download. Subsequent runs hit the on-disk cache
(`~/.cache/huggingface` by default; override with `HF_HOME`) and are fast.

### Heatmap is huge / unreadable for long generations

Use the step-range slider in the SPA, pass `--max-new-tokens` lower, or only
render `source="raw"` in the matplotlib plot.

### `--serve` prints port 8000 but that port is already in use

Pass a different port:

```bash
token-heatmap trace --config configs/example.yaml --serve --port 9000
```

And update the SSH port-forward on your laptop accordingly:

```bash
ssh -L 9000:localhost:9000 user@hpc
```

### `?trace=<url>` won't load the trace (`ERR_CONNECTION_REFUSED` or CORS error)

The viewer fetches the trace JSON from the URL in `?trace=`. There is no
application backend — only the dependency-free CORS file server started by
`token-heatmap serve` / `trace --serve`.

1. **File server not running / wrong port** — the URL's host:port must point at a
   running `token-heatmap serve` (default `:8000`). Start it on the host that has
   the trace files, e.g. `token-heatmap serve outputs/<name> --port 9000`.
2. **SSH tunnel not running** — make sure you have `ssh -L PORT:localhost:PORT user@hpc` open in another terminal.
3. **CORS** — `token-heatmap serve` / `trace --serve` already sends permissive
   CORS headers, so any viewer origin can fetch the trace. If you front it with a
   different static server, ensure it sets `Access-Control-Allow-Origin`.
