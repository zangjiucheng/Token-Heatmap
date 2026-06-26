# Web app

A React + Vite SPA that is a **static, file-based trace viewer** — there is no
backend server. Use the CLI or the Python library to *generate* traces from a
model; the web app loads and explores traces that already exist.

The viewer loads a trace from exactly three sources:

- a JSON file you drop or pick from disk (including two files for a diff),
- a `?trace=<url>` query param (auto-fetched on page open),
- the bundled sample (**Try sample data**).

## The lens workspace

The trace viewer is organized into three roles: a **generation spine** (the
token strip + entropy / selected-probability timelines, always visible), a
**lens rail** on the left grouping the views into **Generation / Internals /
Geometry**, and a **resizable inspector** on the right for the selected step's
detail. Each lens is one way of looking at the same generation; lenses that need
a capture flag are shown but locked until the trace carries that data.

## What you can do in the UI

- Drop a JSON trace file → view the interactive heatmap
- Click **Try sample data** → loads a small bundled trace
- Pass `?trace=<url>` in the URL → auto-loads the trace on page open
- Toggle **raw / processed / split** comparison
- Switch the color scale between `prob` and `logprob`
- Filter the step range, adjust the color range
- Hover the heatmap → step detail panel and timeline cursors follow
- Click a generated token in the strip above the heatmap to jump to that step
- **Attention lens** — layer × head attention grids and Q/K/V stats (requires `--capture-attention`)
- **Logit Lens lens** — per-layer top-k next-token predictions (requires `--capture-logit-lens`)
- **Activations lens** — per-layer activation summary stats, with a per-step ↔ whole-trace TWERA neuron ranking (requires `--capture-activations`)
- **Attribution lens** — **direct logit attribution**: the selected token's logit decomposed into per-layer attention (`o_proj`) and MLP (`mlp_out`) contributions (orange promotes, blue suppresses) with an explicit *unexplained* bar; expand an attention bar to see **per-head** contributions (requires `--capture-full-activations`)
- **Graph lens** — the same direct logit attribution rendered as a pruned, layer-ordered **node-link graph**: the target token (right) built from its top contributors (attention heads / MLP blocks / embedding), sized + coloured by signed contribution (requires `--capture-full-activations`)
- **Manifold lens** — 2-D PCA projection of the activation cloud (coloured by step) plus participation ratio, intrinsic dimension, curvature, periodicity, and a variance-spectrum scree plot (requires `token-heatmap manifold`; see [`cli.md`](cli.md#manifold-analysis))
- Export the current trace as CSV or the current heatmap as PNG
- Persist view state in the URL — share a link to a specific view

> Interactive ablation/intervention is not available in the viewer. It will
> return later via the CLI **precomputing ablations into the trace**, which the
> viewer then renders statically like every other lens.

## Producing traces and opening them

Generate a trace with the CLI, then open it in the viewer. The simplest path
runs the trace and boots the viewer in one command — `--frontend` starts a
stdlib CORS file server *and* `npm run dev`, then opens the viewer pointed at the
trace via `?trace=`:

```bash
token-heatmap trace --config configs/example.yaml --serve --frontend
```

To view a run you already produced (no regeneration), serve its output directory
and open the printed viewer URL:

```bash
token-heatmap serve outputs/ioi --frontend
```

Or run the dev server by hand and drop a file in:

```bash
cd web/frontend && npm run dev          # http://localhost:5173
# then drag a *.json trace onto the page, or paste a ?trace=<url>
```

## HPC / no Node.js

After generation the CLI starts a stdlib `http.server` that serves the output
directory with CORS headers. You run the viewer locally on your laptop.

```bash
# HPC: generate and serve the trace files
token-heatmap trace --config configs/example.yaml --serve

# Optional flags
#   --port 9000              file server port (default 8000)
#   --frontend-url http://localhost:3000   adjust the printed viewer URL if your
#                                          local viewer is on a different port
```

Output:

```
[token-heatmap] Serving output directory …
[token-heatmap] Files: http://localhost:8000/
[token-heatmap] Open the viewer at:
[token-heatmap]   http://localhost:5173/?trace=http://localhost:8000/adaptive_token_trace.json
[token-heatmap] (Press Ctrl+C to stop)
```

Then on your laptop:

```bash
# Terminal 1 — port-forward (adjust port to match --port)
ssh -L 8000:localhost:8000 user@hpc

# Terminal 2 — viewer (any port; pass --frontend-url to match)
cd web/frontend && npm run dev

# Open the URL printed by --serve
```

The `?trace=<url>` query param makes the viewer auto-fetch and display the trace
immediately — no manual file drag needed.

### Hosting the viewer without Node.js

The viewer is a static SPA, so you can build it once (on any machine with
Node.js) and serve the resulting `dist/` from any static file server — no Node.js
ever needed on the host:

```bash
# On a machine with Node.js — build the static viewer
token-heatmap web build

# Serve dist/ with any static file server
python -m http.server -d web/frontend/dist 8080
# open http://localhost:8080/?trace=<trace-url>
```

## Architecture

| Layer | Tech | Serves |
|---|---|---|
| Python library | `llm_token_heatmap` | generation, probes, serialization |
| CLI | `token-heatmap` | produce traces; `serve` a CORS static file server |
| Frontend | React + Vite (`web/frontend/`) | static, file-based trace viewer |

There is no application backend. `token-heatmap serve` (and `trace --serve`) is a
dependency-free stdlib `http.server` with CORS headers — it only serves the
trace JSON and sidecar files; it has no API.

The trace JSON contract is the same for the CLI and the example scripts — they
both go through `llm_token_heatmap.trace_payload.serialize_trace_to_json`, which
conforms to the schema described in [`schema.md`](schema.md).
