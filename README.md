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
- **`token-heatmap`** — CLI that takes a model + prompt (or a YAML config) and writes a full trace bundle to disk: `trace` (generate + capture), `manifold` (activation-cloud geometry), `causal` (per-layer × token ablation grid), and `tuned-lens` (a fitted lens that fixes the vanilla logit lens's early-layer basis mismatch).
- **`app`** — the desktop-app frontend (React + Vite UI in a [Tauri](https://tauri.app/) shell): a static, file-based **trace viewer** — a **lens workspace** with a grouped lens rail (**Generation / Internals / Geometry**), a persistent generation spine (token strip + entropy / selected-probability timelines), and a resizable inspector. Lenses: **Token Heatmap**, **Model** (architecture overview), **Output** (complete generated-text render), **Attention** (per-head grid + a **head-roles** scatter of attention magnitude vs contribution), **Logit Lens** (a layer × token **decision map**, a **tuned-lens** toggle, and a prediction-convergence curve), **Activations** (per-step ↔ whole-trace **TWERA** ranking toggle), **Attribution** (**direct logit attribution** — each token's logit split by layer, expandable to **per head** — plus a layer × token **causal-effect** ablation heatmap), **Graph** (the same attribution as a pruned node-link **attribution graph**), **Manifold** (3-D rotatable cloud + probe/helix readouts + an **intrinsic-dimension × depth** curve); plus step detail and CSV/PNG export. It loads traces from a dropped file or the bundled sample — no backend server. (Interactive ablation will return later via the CLI precomputing ablations into the trace.)
- **Desktop app** — the viewer packaged as a native [Tauri](https://tauri.app/) app (no browser/Node.js needed). Grab an installer from the [Releases](https://github.com/zangjiucheng/Token-Heatmap/releases), or build it with `cd app && npm run build:app`. See [`docs/web-app.md`](docs/web-app.md#desktop-app-tauri).
- **`token-heatmap hpc run <config>`** — one command from your laptop: do the GPU compute on an HPC (Slurm), then rsync the whole run back so viewing needs no GPU. Companion: `token-heatmap hpc setup` (build the GPU venv).

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
`--load-in-4bit`. To capture everything the web app can show, run the
full-capture example and add the post-hoc analyses:

```bash
token-heatmap trace     --config configs/example.yaml               # full capture
T=outputs/example-run/adaptive_token_trace.json
token-heatmap manifold  --trace "$T" --components 6                 # Manifold + ID-depth curve
token-heatmap causal    --trace "$T"                                # causal-effect heatmap
token-heatmap tuned-lens --trace "$T"                               # tuned-lens toggle
```

Traces are **gzipped by default** (`adaptive_token_trace.json.gz`) — a 2048-step
chain-of-thought run is 197.5 MB of JSON and 34.6 MB gzipped. Every reader and the
viewer accept either encoding, and the plain `.json` name resolves to the `.gz`
file, so the `$T` above works unchanged; `--no-compress` opts out. `trace` also
writes just one plot now (`entropy.png`); `--plots all` restores the full set.

Full CLI flags: [`docs/cli.md`](docs/cli.md). Python equivalent: [`docs/python-api.md`](docs/python-api.md).

## Viewing the trace

The viewer is a static, file-based React app — no backend. The CLI just generates
the trace bundle to disk; you then open the JSON in the viewer:

```bash
token-heatmap trace --config configs/example.yaml   # writes outputs/example-run/
cd app && npm run dev                       # http://localhost:5173
# then drag outputs/example-run/adaptive_token_trace.json.gz onto the page
```

Compute on a GPU cluster and view locally with no GPU:

```bash
token-heatmap hpc setup                      # one-time: build the GPU venv
token-heatmap hpc run configs/example.yaml   # runs on Slurm, rsyncs results back
# then drag outputs/example-run/adaptive_token_trace.json.gz onto the viewer
```

Prefer a native window? Grab a desktop installer from the
[Releases](https://github.com/zangjiucheng/Token-Heatmap/releases) or build it with
`cd app && npm run build:app`, then open the JSON in the app.

Full guide — manual file drop, the bundled sample, prebuilt `dist/`, desktop app:
[`docs/web-app.md`](docs/web-app.md). Slurm/qos notes:
[`docs/manifold-reproduction.md`](docs/manifold-reproduction.md).

## License

See [MIT LICENSE](LICENSE)
