# Web app

A React + Vite SPA backed by a FastAPI service. Use the CLI or the Python
library to *generate* traces from a model; the web app is for loading and
exploring traces that already exist.

## What you can do in the UI

- Drop a CSV or JSON trace file → view the interactive heatmap
- Click **Try sample data** → loads a small bundled trace
- Toggle **raw / processed / split** comparison
- Switch the color scale between `prob` and `logprob`
- Filter the step range, adjust the color range
- Hover the heatmap → the step detail panel and timeline cursors follow
- Click a generated token in the strip above the heatmap to jump to that step
- Inspect attention layer/head grids and the logit lens (only for traces produced with `--capture-attention` / `--capture-logit-lens`)
- Export the current trace as CSV or the current heatmap as PNG
- Persist view state in the URL — share a link to a specific view

## Running locally

See [`installation.md`](installation.md#running-the-web-app) for the install steps and
the `scripts/dev.sh` helper.

## Architecture

- **Backend** (`web/backend/`) — FastAPI. Endpoints: `GET /health`, `GET /schema` (returns [`docs/web/trace.schema.json`](web/trace.schema.json) byte-for-byte), `POST /trace/convert-csv`.
- **Frontend** (`web/frontend/`) — React + Vite SPA. Bundles a copy of `trace.schema.json` for offline validation; on startup fetches `/schema` and swaps in the live copy so the UI validates against whatever the server is serving.

The trace JSON contract is the same for the CLI, the example scripts, and the
backend — they all go through `llm_token_heatmap.trace_payload.serialize_trace_to_json`,
which conforms to the schema described in [`schema.md`](schema.md).
