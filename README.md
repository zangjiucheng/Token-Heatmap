# LLM Token Heatmap

A PyTorch toolkit for analyzing and visualizing how a causal language model
picks each next token during generation. Captures per-step probability
distributions, exports them to CSV / JSON, renders static heatmaps, and ships
with an interactive React web app for drill-down exploration.

```
prompt + previous tokens → logits → probabilities → next token
                                       │
                                       └─▶ recorded per step:
                                             top tokens, prob, logprob, rank,
                                             entropy, k_used, selected token
```

## What you get

- **`llm_token_heatmap`** — Python library: `AdaptiveTokenProbe`, a manual generation loop, sampling helpers, CSV/JSON/DataFrame export, attention + logit-lens + activation probes, matplotlib heatmaps, and activation diff.
- **`token-heatmap`** — CLI that takes a model + prompt (or a YAML config file) and writes a full trace bundle to disk. Includes `--serve` to instantly view the result in the browser.
- **`web/backend`** — FastAPI service: `/health`, `/schema`, `/trace/convert-csv`, `/trace/diff`, `/outputs/{path}`. Also serves the pre-built frontend when `web/frontend/dist/` exists.
- **`web/frontend`** — React + Vite SPA: interactive heatmap, step detail, entropy / selected-probability timelines, Attention tab, Logit Lens tab, Activations tab, Manifold tab, CSV/PNG export, diff view.
- **`scripts/dev.sh`** — boots backend + frontend together for local development.
- **`scripts/build-frontend.sh`** — builds the frontend for deployment on servers without Node.js.

## Documentation

| Topic                              | Page                                                 |
| ---------------------------------- | ---------------------------------------------------- |
| Setting up the environment         | [`docs/installation.md`](docs/installation.md)       |
| The `token-heatmap` CLI            | [`docs/cli.md`](docs/cli.md)                         |
| Using the Python library           | [`docs/python-api.md`](docs/python-api.md)           |
| Running the web app                | [`docs/web-app.md`](docs/web-app.md)                 |
| Trace JSON schema                  | [`docs/schema.md`](docs/schema.md)                   |
| Interpreting the metrics and plots | [`docs/interpreting.md`](docs/interpreting.md)       |
| Common issues                      | [`docs/troubleshooting.md`](docs/troubleshooting.md) |

The docs index lives at [`docs/README.md`](docs/README.md).

## Quick start

```bash
git clone <repo-url> llm-token-heatmap
cd llm-token-heatmap

# Option A — pip + venv
./scripts/setup.sh
source .venv/bin/activate

# Option B — conda (no Node.js required on this machine)
conda env create -f environment.yml
conda activate token-heatmap
```

```bash
# CLI: generate a trace bundle
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain diffusion models." \
  --max-new-tokens 80 \
  --out outputs/
```

Or use a YAML config file (see `configs/example.yaml`):

```bash
token-heatmap trace --config configs/example.yaml
```

Full CLI flags: [`docs/cli.md`](docs/cli.md). Python equivalent: [`docs/python-api.md`](docs/python-api.md).

## Viewing the trace

**Local machine with Node.js:**

```bash
./scripts/dev.sh          # backend :8000, frontend :5173
```

Open <http://localhost:5173> and drop `outputs/adaptive_token_trace.json` onto the landing page.

**HPC / server without Node.js:**

```bash
# 1. Generate trace and start a file server (no extra deps needed)
token-heatmap trace --config configs/example.yaml --serve --port 8000

# 2. On your laptop — SSH port-forward
ssh -L 8000:localhost:8000 user@hpc

# 3. On your laptop — run the frontend
cd web/frontend && npm run dev   # http://localhost:5173

# 4. Open the URL printed by --serve
#    e.g. http://localhost:5173/?trace=http://localhost:8000/adaptive_token_trace.json
```

The frontend auto-loads the trace via the `?trace=` URL param — no manual file drop needed.
See [`docs/web-app.md`](docs/web-app.md) for the full HPC guide including the pre-built frontend option.

## License

See [MIT LICENSE](LICENSE)
