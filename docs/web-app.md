# Web app

A React + Vite SPA backed by a FastAPI service. Use the CLI or the Python
library to *generate* traces from a model; the web app is for loading and
exploring traces that already exist.

## The lens workspace

The trace viewer is organized into three roles: a **generation spine** (the
token strip + entropy / selected-probability timelines, always visible), a
**lens rail** on the left grouping the views into **Generation / Internals /
Geometry**, and a **resizable inspector** on the right for the selected step's
detail. Each lens is one way of looking at the same generation; lenses that need
a capture flag are shown but locked until the trace carries that data.

## What you can do in the UI

- Drop a CSV or JSON trace file → view the interactive heatmap
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
- **Attribution lens** — **direct logit attribution**: the selected token's logit decomposed into per-layer attention (`o_proj`) and MLP (`mlp_out`) contributions (orange promotes, blue suppresses) with an explicit *unexplained* bar; expand an attention bar to see **per-head** contributions. Each component/head has an **ablate** button that re-runs the model with that part zeroed and shows the next-token distribution change — KL, target-probability delta, top-token flips — turning the attribution into a causal experiment (decomposition requires `--capture-full-activations`; ablation requires the live backend)
- **Graph lens** — the same direct logit attribution rendered as a pruned, layer-ordered **node-link graph**: the target token (right) built from its top contributors (attention heads / MLP blocks / embedding), sized + coloured by signed contribution. Click a node to ablate that component and validate the edge (requires `--capture-full-activations`; ablation requires the live backend)
- **Manifold lens** — 2-D PCA projection of the activation cloud (coloured by step) plus participation ratio, intrinsic dimension, curvature, periodicity, and a variance-spectrum scree plot (requires `token-heatmap manifold`; see [`cli.md`](cli.md#manifold-analysis))
- Export the current trace as CSV or the current heatmap as PNG
- Persist view state in the URL — share a link to a specific view

## Running locally (with Node.js)

```bash
./scripts/dev.sh                        # backend :8000, frontend :5173
BACKEND_PORT=8765 FRONTEND_PORT=5180 ./scripts/dev.sh   # custom ports
```

Or generate a trace and boot the frontend in one command — `--frontend` starts
the file server *and* `npm run dev`, then opens the viewer (see
[`docs/cli.md`](cli.md#one-command-frontend-included---frontend)):

```bash
token-heatmap trace --config configs/example.yaml --serve --frontend
```

Or manually in two terminals:

```bash
# Terminal 1
cd web/backend && uvicorn llm_token_heatmap_api.main:app --reload --port 8000

# Terminal 2
cd web/frontend && npm run dev          # http://localhost:5173
```

### Changing the backend port the frontend connects to

The frontend reads `VITE_API_BASE_URL` at startup (default `http://localhost:8000`).

**Persistent** — create `web/frontend/.env.local` (gitignored):

```bash
cp web/frontend/.env.local.example web/frontend/.env.local
# edit: VITE_API_BASE_URL=http://localhost:9000
```

**One-off** — pass it inline:

```bash
VITE_API_BASE_URL=http://localhost:9000 npm run dev
```

## HPC / no Node.js

Two patterns depending on how much you want to install on the server.

### Pattern 1 — `--serve` (simplest, zero extra deps)

After generation the CLI starts a stdlib `http.server` that serves the output
directory with CORS headers. You run the frontend locally on your laptop.

```bash
# HPC: generate and serve
token-heatmap trace --config configs/example.yaml --serve

# Optional flags
#   --port 9000              file server port (default 8000)
#   --frontend-url http://localhost:3000   adjust the printed URL if your
#                                          local frontend is on a different port
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

# Terminal 2 — frontend (any port; pass --frontend-url to match)
cd web/frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
# or if you already set it in .env.local:
npm run dev

# Open the URL printed by --serve
```

The `?trace=<url>` query param makes the frontend auto-fetch and display the
trace immediately — no manual file drag needed.

### Pattern 2 — pre-built frontend served by FastAPI

Build the frontend once (on any machine with Node.js), copy `dist/` to the
server, and the FastAPI backend serves both the API and the UI from the same
port. No Node.js ever needed on the server.

```bash
# On your laptop — build (VITE_API_BASE_URL='' → relative/same-origin API calls)
./scripts/build-frontend.sh

# Copy to HPC
rsync -av web/frontend/dist/ user@hpc:/path/to/Token-Heatmap/web/frontend/dist/
```

```bash
# On HPC — start the backend (also serves the frontend)
cd web/backend
uvicorn llm_token_heatmap_api.main:app --host :: --port 8000
# or with the CLI after a run:
token-heatmap trace --config configs/example.yaml
# then start uvicorn manually as above
```

```bash
# On your laptop — port-forward and open
ssh -L 8000:localhost:8000 user@hpc
open http://localhost:8000
```

When `web/frontend/dist/` exists, `create_app()` mounts the Vite assets and
adds an SPA fallback route so every non-API path returns `index.html`. If
`dist/` is absent the backend still works as an API-only service.

## Architecture

| Layer | Tech | Serves |
|---|---|---|
| Python library | `llm_token_heatmap` | generation, probes, serialization |
| Backend | FastAPI (`web/backend/`) | `/health`, `/schema`, `/trace/generate`, `/trace/intervene`, `/trace/convert-csv`, `/trace/diff`, `/outputs/{path}`, SPA (when `dist/` present) |
| Frontend | React + Vite (`web/frontend/`) | interactive trace viewer |

Key backend endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness |
| `GET` | `/schema` | canonical `trace.schema.json` |
| `POST` | `/trace/generate` | run a model server-side and return a trace |
| `POST` | `/trace/intervene` | ablate / scale a layer block or attention head and diff the next-token distribution |
| `POST` | `/trace/convert-csv` | CSV → JSON trace |
| `POST` | `/trace/diff` | compare two activation traces |
| `GET` | `/outputs/{path}` | serve files from `LLM_HEATMAP_OUTPUT_DIR` |

> Both `/trace/generate` and `/trace/intervene` load arbitrary models with
> `trust_remote_code=True`. Expose them only over a trusted channel (e.g. an SSH
> tunnel), never the public internet.

The trace JSON contract is the same for the CLI, example scripts, and backend —
they all go through `llm_token_heatmap.trace_payload.serialize_trace_to_json`,
which conforms to the schema described in [`schema.md`](schema.md).
