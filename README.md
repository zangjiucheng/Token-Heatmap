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

- **`llm_token_heatmap`** — Python library: `AdaptiveTokenProbe`, a manual generation loop, sampling helpers, CSV/JSON/DataFrame export, attention + logit-lens probes, and matplotlib heatmaps.
- **`token-heatmap`** — CLI that takes a model + prompt and writes a full trace bundle (CSV, JSON, 4 plots) to disk.
- **`web/backend`** — FastAPI service exposing `/health`, `/schema`, and `/trace/convert-csv`.
- **`web/frontend`** — React + Vite SPA: interactive heatmap, step detail panel, entropy and selected-probability timelines, raw-vs-processed comparison, attention/logit-lens views, CSV/PNG export.
- **`scripts/dev.sh`** — one-shot script that boots both backend and frontend for local development.

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

./scripts/setup.sh                  # creates .venv and installs everything
source .venv/bin/activate

# CLI: generate a trace bundle
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain diffusion models." \
  --max-new-tokens 80 \
  --out outputs/
```

Full CLI flags: [`docs/cli.md`](docs/cli.md). Python equivalent:
[`docs/python-api.md`](docs/python-api.md).

To explore the trace interactively (web app deps were already installed by
`scripts/setup.sh` if Node was available):

```bash
./scripts/dev.sh              # backend :8000, frontend :5173
```

Open <http://localhost:5173> and drop `outputs/adaptive_token_trace.json`
(or the CSV) onto the landing page. More in [`docs/web-app.md`](docs/web-app.md).

## License

See [MIT LICENSE](LICENSE)
