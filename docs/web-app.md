# Web app

A React + Vite SPA that is a **static, file-based trace viewer** — there is no
backend server. Use the CLI or the Python library to *generate* traces from a
model; the web app loads and explores traces that already exist.

The viewer loads a trace from exactly two sources:

- a JSON file you drop or pick from disk (including two files for a diff),
- the bundled sample (**Try sample data**).

The desktop app additionally lets you open a trace file directly from disk.

## Desktop app (Tauri)

The same viewer is also packaged as a native desktop app via
[Tauri](https://tauri.app/) — no Node.js or browser tab needed, just open a
trace file. The Rust shell lives in `web/frontend/src-tauri/`.

**Download a prebuilt installer.** Every published
[GitHub Release](https://github.com/zangjiucheng/Token-Heatmap/releases) ships
installers for macOS (Apple Silicon + Intel), Windows (`.msi`), and Linux
(`.AppImage` / `.deb`), built by the `app-release.yml` workflow.

**Build it yourself.** You need [Node.js](https://nodejs.org) and the
[Rust toolchain](https://rustup.rs) (plus the platform's webview deps — see the
[Tauri prerequisites](https://tauri.app/start/prerequisites/); on Linux that is
`libwebkit2gtk-4.1-dev`, `librsvg2-dev`, `patchelf`, …).

```bash
cd web/frontend
npm install              # first time only

npm run app:dev          # hot-reloading desktop dev window (tauri dev)
npm run app:build        # produce a release installer for the current OS
```

`app:build` runs the frontend build (`npm run build`) and then `tauri build`.
The installers land in:

```
web/frontend/src-tauri/target/release/bundle/
  macos/   Token Heatmap.app  +  dmg/Token Heatmap_<ver>_<arch>.dmg
  msi/     Token Heatmap_<ver>_<arch>_en-US.msi      (Windows)
  appimage/ token-heatmap_<ver>_<arch>.AppImage      (Linux)
  deb/     token-heatmap_<ver>_<arch>.deb            (Linux)
```

`tauri build` only bundles for the OS it runs on; cross-compiling needs the
matching Rust target (e.g. `rustup target add x86_64-apple-darwin`) and toolchain.

**Continuous build.** The `app` job in `ci.yml` compiles the desktop app
(frontend + Rust shell) on **every push and pull request** as a fast Linux-only
compile check, so a change that breaks the native build is caught immediately.
Full multi-OS installers are produced only on a published Release.

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

Generate a trace with the CLI, then open it in the viewer — the CLI only writes
the bundle to disk; there is nothing to serve. Run the dev server and drop the
file in:

```bash
token-heatmap trace --config configs/example.yaml   # writes outputs/example-run/
cd web/frontend && npm run dev                       # http://localhost:5173
# then drag outputs/example-run/adaptive_token_trace.json onto the page
```

Or open the same `adaptive_token_trace.json` in the desktop app. Either way the
viewer reads the file directly — no backend, no network fetch.

## HPC / no GPU locally

The HPC does the GPU compute; you view the trace locally with no GPU. Use
`token-heatmap hpc run <config>` (see [`cli.md`](cli.md#hpc-round-trip-hpc-run)),
which rsyncs the whole output dir back to `./outputs/<name>/`. Then open the
local JSON in the viewer:

```bash
cd web/frontend && npm run dev          # http://localhost:5173
# then drag outputs/<name>/adaptive_token_trace.json onto the page
```

### Hosting the viewer without Node.js

The viewer is a static SPA, so you can build it once (on any machine with
Node.js) and serve the resulting `dist/` from any static file server — no Node.js
ever needed on the host:

```bash
# On a machine with Node.js — build the static viewer
token-heatmap web build

# Serve dist/ with any static file server
python -m http.server -d web/frontend/dist 8080
# open http://localhost:8080/ and drag a trace JSON onto the page
```

## Architecture

| Layer | Tech | Serves |
|---|---|---|
| Python library | `llm_token_heatmap` | generation, probes, serialization |
| CLI | `token-heatmap` | produce traces (writes the bundle to disk) |
| Frontend | React + Vite (`web/frontend/`) | static, file-based trace viewer |

There is no application backend and no network loading. The viewer reads the
trace JSON directly from a dropped file (or the desktop app's file open); it
never fetches over HTTP.

The trace JSON contract is the same for the CLI and the example scripts — they
both go through `llm_token_heatmap.trace_payload.serialize_trace_to_json`, which
conforms to the schema described in [`schema.md`](schema.md).
