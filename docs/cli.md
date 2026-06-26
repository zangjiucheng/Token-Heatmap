# Command-line interface

After `pip install -e .` (or `conda env create -f environment.yml`) the
`token-heatmap` command is on your `PATH`.

## YAML config files

All `trace` flags can be set in a YAML file so you don't have to repeat long
command lines. CLI flags always override YAML values.

```bash
pip install pyyaml          # one-time; only needed for --config
token-heatmap trace --config configs/example.yaml
token-heatmap trace --config configs/example.yaml --max-new-tokens 128   # override one field
```

`configs/example.yaml` (included in the repo):

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
prompt: "Explain what a large language model is in one sentence."
max_new_tokens: 64
temperature: 0.8
top_p: 0.95
out: outputs/example-run
capture_logit_lens: true
```

All keys are optional — any missing key falls back to the CLI default.

## Basic trace

```bash
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain diffusion models." \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-p 0.95 \
  --min-k 8 --max-k 64 --mass-threshold 0.95 \
  --out outputs/
```

Output directory:

```
outputs/
├── generated.txt
├── adaptive_token_trace.csv
├── adaptive_token_trace.json
├── adaptive_heatmap.png
├── entropy.png
└── selected_probability.png
```

Run `token-heatmap trace --help` for the full flag list.

## Serving the result instantly (`--serve`)

Add `--serve` to start a file server immediately after generation and print a
ready-made URL to open in the browser. Uses Python's built-in `http.server` —
no uvicorn, no npm, no extra installs required.

```bash
token-heatmap trace --config configs/example.yaml --serve
```

```
[token-heatmap] Serving output directory …
[token-heatmap] Files: http://localhost:8000/
[token-heatmap] Open the viewer at:
[token-heatmap]   http://localhost:5173/?trace=http://localhost:8000/adaptive_token_trace.json
[token-heatmap] (Press Ctrl+C to stop)
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--serve` | off | Start file server after generation |
| `--port` | `8000` | Port for the file server |
| `--frontend-url` | `http://localhost:5173` | Frontend origin used to build the printed URL |
| `--frontend` | off | Also start the Vite frontend (`npm run dev`) and open the viewer. Implies `--serve`. Needs Node.js + a repo checkout. |
| `--no-open` | off | With `--frontend`, don't auto-open the browser |

Example — file server on port 9000, viewer on port 3000:

```bash
token-heatmap trace --config configs/example.yaml \
  --serve --port 9000 --frontend-url http://localhost:3000
# prints: http://localhost:3000/?trace=http://localhost:9000/adaptive_token_trace.json
```

On HPC, SSH port-forward the file-server port to your laptop before opening the URL:

```bash
ssh -L 9000:localhost:9000 user@hpc
```

### One command, frontend included (`--frontend`)

On a **local machine with Node.js and a repo checkout**, `--frontend` starts the
file server *and* the Vite dev server, then opens the ready-made viewer URL once
the frontend is up. The dev server binds to the port in `--frontend-url`
(default `5173`). One `Ctrl+C` stops both.

```bash
token-heatmap trace --config configs/example.yaml --serve --frontend
```

```
[token-heatmap] Starting frontend (npm run dev) on port 5173 …
[token-heatmap] Serving output directory …
[token-heatmap] Files: http://localhost:8000/
[token-heatmap] Frontend (npm run dev): http://localhost:5173
[token-heatmap] Open the viewer at:
[token-heatmap]   http://localhost:5173/?trace=http://localhost:8000/adaptive_token_trace.json
[token-heatmap] (Press Ctrl+C to stop)
```

If `npm` isn't on `PATH` or `web/frontend` is missing (e.g. a pip-only install),
it prints a warning and degrades to serving files only. This flag is for local
use — the HPC node typically has no Node.js, which is exactly why the
SSH-forward-the-port workflow above exists.

## Inspecting attention and the logit lens

`AttentionProbe` and `LogitLens` are wired through `token-heatmap trace` as
opt-in flags. They are off by default because forcing eager attention is
significantly slower than the SDPA / FlashAttention kernel.

```bash
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain attention in one sentence." \
  --max-new-tokens 12 \
  --capture-attention --attention-layers all --attention-top-k 8 \
  --capture-full-attention \
  --capture-logit-lens --lens-layers 0,3,7,11 --lens-top-k 5 \
  --out outputs/inspect/
```

| Flag                       | Meaning                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--capture-attention`      | Attach an `AttentionProbe`. Forces eager attention (slow).                                                   |
| `--attention-layers`       | `all` (default) or comma-separated layer indices (e.g. `0,3,7,11`).                                          |
| `--attention-top-k`        | Top-k attended positions kept inline per head (default 8).                                                   |
| `--capture-full-attention` | Also write a Tier-2 `attention.<step>.npz` per step under `<out>/attention/`. Implies `--capture-attention`. |
| `--capture-logit-lens`     | Attach a `LogitLens`.                                                                                        |
| `--lens-layers`            | Same syntax as `--attention-layers`.                                                                         |
| `--lens-top-k`             | Top-k tokens retained per layer (default 8).                                                                 |

Extra output files when these flags are active:

- `adaptive_token_trace.json` — includes inline attention aggregates, per-layer logit-lens projections, and `attention_sidecar_ref` pointers.
- `attention_layer_head_grid.png` — per-step layer × head entropy grid.
- `logit_lens.png` — per-layer top-k table (first step).
- `selected_rank_heatmap.png` — selected-token rank by layer × step.

The **Logit Lens** tab in the web app shows this data interactively, synced to the heatmap cursor.

## Capturing activations

`ActivationProbe` captures per-layer / per-submodule summary statistics and,
optionally, full activation tensors as `.npz` sidecars.

```bash
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain attention in one sentence." \
  --max-new-tokens 12 \
  --capture-activations \
  --activation-layers all \
  --activation-submodules residual_post,mlp_out,o_proj \
  --activation-top-k 8 \
  --out outputs/activations/
```

Add `--capture-full-activations` to also write `activations/activation.<step>.npz`
(full hidden-state tensors) and embed `activation_sidecar_ref` pointers in the
JSON trace. It additionally embeds, **inline** in the trace, the **TWERA neuron
ranking** and **Direct Logit Attribution** — each generated token's logit
decomposed into per-layer attention (`o_proj`) and MLP (`mlp_out`) contributions,
expandable to **per attention head** — which the web app's Activations and
Attribution lenses read. Implies `--capture-activations`.

| Flag                         | Meaning                                                                                                                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--capture-activations`      | Attach an `ActivationProbe`. Off by default.                                                                                                                                        |
| `--activation-layers`        | `all` (default) or comma-separated decoder layer indices.                                                                                                                           |
| `--activation-submodules`    | Comma-separated submodule keys (default `residual_post,mlp_out,o_proj`). Supported: `resid_pre`/`residual_pre`, `resid_post`/`residual_post`, `mlp_out`/`mlp.down_proj`, `o_proj`. |
| `--activation-top-k`         | Top-k highest-magnitude neurons retained per (layer, submodule) (default 8).                                                                                                        |
| `--capture-full-activations` | Write full tensors as `.npz` sidecars under `<out>/activations/`. Implies `--capture-activations`.                                                                                  |

## Comparing two activation traces

```bash
token-heatmap diff outA/adaptive_token_trace.json outB/adaptive_token_trace.json \
  --out diff/ \
  --metric l2
```

The subcommand projects each input's activation subset, calls
`compare_activations` with `align="auto"`, and writes:

- `activation_diff.json` — schema-shaped diff payload.
- `activation_delta.png` — stacked layer × step heatmap, one subplot per captured submodule.

The CLI refuses to diff (non-zero exit) when the two parent traces have
different `metadata.prompt` values or when zero steps align between them.

## Manifold analysis

Analyze the _geometry_ of the captured activation clouds — inspired by
[“When Models Manipulate Manifolds”](https://transformer-circuits.pub/2025/linebreaks/index.html),
which finds a model encoding a scalar on a low-dimensional, curved (helical)
manifold. For each `(layer, submodule)` the analysis stacks the full per-token
activation vectors into a matrix and computes PCA spectrum + participation
ratio, TwoNN intrinsic dimension, a 2-D/3-D projection, trajectory curvature,
and FFT periodicity.

It reads the full activation vectors from the sidecars, so generate the trace
with `--capture-full-activations` first:

```bash
# 1. generate with full activation sidecars
token-heatmap trace --config configs/example.yaml \
  --capture-activations --capture-full-activations

# 2. add the manifold analysis to the trace
token-heatmap manifold --trace outputs/adaptive_token_trace.json
```

This writes a top-level `manifold` field back into the trace (in place by
default), which the web app's **Manifold tab** then renders. The analysis is
pure-numpy — no torch needed to run it.

| Flag           | Default                  | Meaning                                                       |
| -------------- | ------------------------ | ------------------------------------------------------------- |
| `--trace`      | _required_               | Path to a trace JSON that has `activation_sidecar_ref`s.      |
| `--out`        | _overwrite `--trace`_    | Write the augmented trace elsewhere instead of in place.      |
| `--layers`     | all captured             | Subset of layer indices to analyze.                           |
| `--submodules` | all captured             | Subset of submodule names to analyze.                         |
| `--components` | `3`                      | Number of PCA projection components to keep.                  |

Exits non-zero when the trace has no `activation_metadata`, carries no
`activation_sidecar_ref` (i.e. was generated without `--capture-full-activations`),
or when no `(layer, submodule)` cloud has at least two positions to analyze.

See [`interpreting.md`](interpreting.md#manifold-metrics) for what the metrics mean.

## Serving an existing trace (`serve`)

`trace --serve` regenerates before serving, so it can't serve a trace you've
just augmented with `manifold`. The `serve` subcommand serves an existing
directory over HTTP with CORS — **no regeneration** — so the full flow fits in
one terminal:

```bash
token-heatmap trace --config configs/example.yaml \
  --capture-activations --capture-full-activations          # writes outputs/example-run/
token-heatmap manifold --trace outputs/example-run/adaptive_token_trace.json
token-heatmap serve outputs/example-run                     # serves it, CORS, no regen
```

| Flag             | Default                   | Meaning                                                      |
| ---------------- | ------------------------- | ----------------------------------------------------------- |
| `dir`            | `outputs/`                | Directory to serve.                                         |
| `--port`         | `8000`                    | File-server port.                                          |
| `--frontend-url` | `http://localhost:5173`   | Frontend origin used to build the printed viewer URL.      |
| `--frontend`     | off                       | Also start the Vite frontend (`npm run dev`) and open it.  |
| `--no-open`      | off                       | With `--frontend`, don't auto-open the browser.            |

On HPC, SSH port-forward the file-server port to your laptop (use a free local
port — see [`web-app.md`](web-app.md)) before opening the URL.

## Running the viewer locally

The web app is a static, file-based viewer — there is no backend to run. Start
the Vite dev server and open a trace by dropping a file or pasting a
`?trace=<url>`:

```bash
cd web/frontend && npm run dev          # http://localhost:5173
```

The easiest path is to let the CLI start the viewer for you with `--frontend`
(see [`--serve` / `--frontend`](#serving-the-result-instantly---serve) above),
which boots `npm run dev`, serves the trace files, and opens the viewer pointed
at the new trace.

## Building the frontend (`web build`)

Run `npm install` + `npm run build` in `web/frontend` to produce a static
`dist/`. The viewer is backend-free, so you can serve `dist/` from any static
file server on a host with no Node.js.

```bash
token-heatmap web build                 # output: web/frontend/dist/
# then serve it anywhere, e.g.:
python -m http.server -d web/frontend/dist 8080
# open http://localhost:8080/?trace=<trace-url>
```

## HPC: build the GPU venv (`hpc setup`)

Idempotently build the dedicated cu124 torch venv on the HPC so `token-heatmap`
runs on the GPU instead of silently falling back to CPU.

```bash
token-heatmap hpc setup            # build/update the GPU venv
token-heatmap hpc setup --verify   # also run a real GPU matmul check (queues a short srun)
```

| Flag                | Default                    | Meaning                                         |
| ------------------- | -------------------------- | ----------------------------------------------- |
| `--verify`          | off                        | Also run a real GPU matmul check.               |
| `--ssh-host`        | `j7zang-gpu`               | SSH host alias.                                  |
| `--remote-repo`     | `/work/j7zang/Token-Heatmap` | Repo checkout on the HPC.                      |
| `--remote-venv`     | `/work/j7zang/th-gpu`      | GPU venv path.                                  |
| `--anaconda-python` | _base interpreter_         | Base interpreter used to create the venv.       |

## HPC round-trip (`hpc run`)

One command from your laptop. It uploads the config to the HPC, submits a Slurm
GPU job (the *only* remote step) that runs `trace` + `manifold`, polls it to
completion, then rsyncs the whole `outputs/<name>/` folder back — so you view it
locally with **no GPU and no tunnel** (drag the JSON onto the viewer, or
`token-heatmap serve outputs/<name>`). A pre-flight check refuses runs that
won't fit the GPU's VRAM before submitting.

```bash
token-heatmap hpc run configs/wrap-text.yaml --model Qwen/Qwen2.5-14B-Instruct \
  --capture activations --probe line_position --extra "--max-new-tokens 320"
# 32B on one GPU: add  --4bit
# auto-open it locally afterwards: add  --serve
```

| Flag             | Default                          | Meaning                                                                |
| ---------------- | -------------------------------- | ---------------------------------------------------------------------- |
| `config`         | _required_                       | Trace config YAML (its basename is the default run name).              |
| `--name`         | config basename                  | Run name → `outputs/NAME` locally + on the HPC.                        |
| `--model`        | _from config_                    | Override the model id.                                                  |
| `--gpu`          | `rtx6000`                        | GPU type (`rtx6000` or `l40s`); both 48 GB.                            |
| `--qos`          | `qos_rtx6000_max` / `normal`     | Slurm qos (defaults by GPU type).                                      |
| `--mem`          | `64G` / `28G`                    | Host memory (default depends on qos).                                  |
| `--time`         | `01:00:00`                       | Walltime `HH:MM:SS`.                                                    |
| `--capture`      | `full`                           | `full` = +attention (slower); `activations` = manifold-only.          |
| `--probe`        | _none_                           | Add a supervised manifold probe scalar (e.g. `line_position`).        |
| `--extra`        | _none_                           | Extra `trace` flags (e.g. `--max-new-tokens 320`).                    |
| `--4bit`         | off                              | Load in 4-bit NF4 (for 32B+).                                          |
| `--serve`        | off                              | After pulling, start a local file server + print the viewer URL.      |
| `--no-manifold`  | off                              | Skip the manifold pass.                                                |
| `--no-sync`      | off                              | Don't `git pull` the HPC repo first.                                   |
| `--no-pull`      | off                              | Leave outputs on the HPC (no rsync back).                              |
| `--setup`        | off                              | Build/verify the GPU venv on the HPC first (one-time).                 |
| `--force`        | off                              | Skip the pre-flight "won't fit in VRAM" size check.                   |

Connection / path overrides also exist (`--ssh-host`, `--remote-repo`,
`--remote-venv`, `--anaconda-python`, `--remote-bin-gpu`, `--local-view-port`,
`--frontend-port`, `--poll-seconds`); run `token-heatmap hpc run --help` for the
full list.

## HPC: serve a remote run (`hpc serve`)

Start the `token-heatmap` file server on the HPC and forward it to a local port
so the local frontend can fetch the trace. One SSH session does both, so a
single `Ctrl+C` tears down both.

```bash
token-heatmap hpc serve outputs/wrap-text                    # serve an existing remote run
token-heatmap hpc serve --gen --config configs/wrap-text.yaml   # regenerate first, then serve
```

| Flag              | Default                      | Meaning                                            |
| ----------------- | ---------------------------- | -------------------------------------------------- |
| `dir`             | _remote run dir_             | Remote run dir to serve.                           |
| `--gen`           | off                          | Regenerate trace + manifold first, then serve.     |
| `--config`        | _none_                       | Config used with `--gen`.                          |
| `--ssh-host`      | `j7zang-gpu`                 | SSH host alias.                                     |
| `--remote-repo`   | `/work/j7zang/Token-Heatmap` | Repo checkout on the HPC.                           |
| `--remote-bin`    | `~/.local/bin`               | `token-heatmap` path on the HPC.                   |
| `--remote-port`   | `8000`                       | File-server port on the HPC.                       |
| `--local-port`    | `8001`                       | Local forwarded port.                              |
| `--frontend-port` | `5173`                       | Frontend port for the printed viewer URL.          |
