# LLM Token Heatmap

A PyTorch toolkit for analyzing and visualizing how a causal language model
picks each next token during generation. Captures per-step probability
distributions, attention, logit-lens, and activations; **decomposes each token's
logit into per-layer — and per attention head — contributions (direct logit
attribution)**; analyzes the activation geometry (PCA / intrinsic dimension /
a TWERA-style neuron ranking); exports to CSV / JSON; renders static heatmaps;
and ships with an interactive React web app — a **lens workspace** that is a
static, file-based **trace viewer** — for drill-down exploration. Works with any
HuggingFace causal LM (Qwen, Llama, Mistral, Gemma, Phi, …).

```
prompt + previous tokens → logits → probabilities → next token
                                       │
                                       └─▶ recorded per step:
                                             top tokens, prob, logprob, rank,
                                             entropy, k_used, selected token
```

## What you get

- **`llm_token_heatmap`** — Python library: `AdaptiveTokenProbe`, a manual generation loop, sampling helpers, CSV/JSON/DataFrame export, attention + logit-lens + activation probes, a self-contained model-architecture summary, TWERA-style neuron attribution, **direct logit attribution (per-layer + per-head)**, component / head ablation primitives, post-hoc manifold geometry, matplotlib heatmaps, and activation diff. Loads on GPU in **bf16** (with optional **`--load-in-4bit`** NF4 for big models).
- **`token-heatmap`** — CLI that takes a model + prompt (or a YAML config) and writes a full trace bundle to disk: `trace` (generate + capture) and `manifold` (analyze the activation clouds). Includes `--serve` to view the result in the browser.
- **`web/frontend`** — React + Vite SPA: a static, file-based **trace viewer** — a **lens workspace** with a grouped lens rail (**Generation / Internals / Geometry**), a persistent generation spine (token strip + entropy / selected-probability timelines), and a resizable inspector. Lenses: **Token Heatmap**, **Model** (architecture overview), **Output** (complete generated-text render), **Attention**, **Logit Lens**, **Activations** (per-step ↔ whole-trace **TWERA** ranking toggle), **Attribution** (**direct logit attribution** — each token's logit split by layer, expandable to **per head**), **Graph** (the same attribution as a pruned node-link **attribution graph**), **Manifold** (3-D rotatable cloud + probe/helix readouts); plus step detail, CSV/PNG export, and a diff view. It loads traces from a dropped file, a `?trace=<url>` URL, or the bundled sample — no backend server. (Interactive ablation will return later via the CLI precomputing ablations into the trace.)
- **`token-heatmap web build`** — builds the static viewer for deployment on servers without Node.js.
- **`token-heatmap hpc run <config>`** — one command from your laptop: do the GPU compute on an HPC (Slurm), then rsync the whole run back so viewing needs no GPU. Companions: `token-heatmap hpc setup` (build the GPU venv) and `token-heatmap hpc serve` (SSH tunnel + remote file server).

## Documentation

| Topic                              | Page                                                 |
| ---------------------------------- | ---------------------------------------------------- |
| Setting up the environment         | [`docs/installation.md`](docs/installation.md)       |
| The `token-heatmap` CLI            | [`docs/cli.md`](docs/cli.md)                         |
| Using the Python library           | [`docs/python-api.md`](docs/python-api.md)           |
| Running the web app                | [`docs/web-app.md`](docs/web-app.md)                 |
| Trace JSON schema                  | [`docs/schema.md`](docs/schema.md)                   |
| Interpreting the metrics and plots | [`docs/interpreting.md`](docs/interpreting.md)       |
| Manifold / counting reproduction (+ GPU & HPC guide) | [`docs/manifold-reproduction.md`](docs/manifold-reproduction.md) |
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

Any HuggingFace causal LM works — swap `--model` for a Llama / Mistral / Gemma /
Phi id (gated repos need an `HF_TOKEN`). For large models on a single GPU add
`--load-in-4bit`. To capture everything the web app can show, add
`--capture-attention --capture-logit-lens --capture-activations
--capture-full-activations`, then `token-heatmap manifold --trace <trace>.json`.

Prefer a GUI? The web app's **Build** page (`/build`) lets you wire the config
as a node graph and either run it live or export the YAML — no flags to memorize.

Full CLI flags: [`docs/cli.md`](docs/cli.md). Python equivalent: [`docs/python-api.md`](docs/python-api.md).

## Viewing the trace

**Local machine with Node.js:**

```bash
cd web/frontend && npm run dev          # http://localhost:5173
```

Open <http://localhost:5173> and drop `outputs/adaptive_token_trace.json` onto the
landing page (or let the CLI do it for you: `token-heatmap trace … --serve --frontend`).

**HPC round-trip (compute on the cluster, view locally):**

```bash
# One command from your laptop: scp the config up, run the GPU job on Slurm,
# then rsync the whole run back to ./outputs/<name>/ — no GPU/tunnel to view.
token-heatmap hpc setup                      # one-time: build the GPU venv
token-heatmap hpc run configs/example.yaml   # --gpu, --4bit, --serve, … (see --help)
```

It refuses runs that won't fit the GPU's VRAM before submitting, and pairs with
the Build page (export YAML → `token-heatmap hpc run that.yaml`). Details + Slurm/qos notes:
[`docs/manifold-reproduction.md`](docs/manifold-reproduction.md).

**HPC / server without Node.js (manual):**

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
