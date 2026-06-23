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

### `TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not BatchEncoding`

You're on a transformers version that leaves a `BatchEncoding` wrapper after
`.to(device)`. `generation.py` already extracts `input_ids` defensively — pull
`main` and re-run.

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

### Frontend can't reach the backend (`ERR_CONNECTION_REFUSED` or CORS error)

1. **Wrong backend port** — the frontend defaults to `http://localhost:8000`. If your backend (or SSH tunnel) is on a different port, set it before starting `npm run dev`:
   ```bash
   # one-off
   VITE_API_BASE_URL=http://localhost:9000 npm run dev
   # persistent — create web/frontend/.env.local:
   echo "VITE_API_BASE_URL=http://localhost:9000" > web/frontend/.env.local
   ```
2. **SSH tunnel not running** — make sure you have `ssh -L PORT:localhost:PORT user@hpc` open in another terminal.
3. **CORS mismatch** — if you run the frontend on a port other than 5173, pass that origin to the file server:
   ```bash
   token-heatmap trace --config configs/example.yaml \
     --serve --port 8000 --frontend-url http://localhost:3000
   ```
   For the FastAPI backend, set `LLM_HEATMAP_ALLOWED_ORIGINS=http://localhost:3000`.

### Frontend always shows the bundled sample after I drop a file

Fixed: the landing page now seeds the trace store with the loaded trace and
routes to `/trace/uploaded`. Pull `main` and re-run.
