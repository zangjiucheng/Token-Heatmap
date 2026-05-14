# LLM Token Heatmap API

FastAPI service that exposes the canonical trace JSON Schema and converts the
`llm_token_heatmap` library's CSV export into the JSON payload the frontend
consumes.

Trace **generation** is intentionally not a backend responsibility — use the
`token-heatmap` CLI or the Python library to produce a trace, then upload it
through the SPA (drag-and-drop a JSON file, or upload the CSV to be converted
server-side).

## Endpoints

| Method | Path                  | Description                                            |
|--------|-----------------------|--------------------------------------------------------|
| GET    | `/health`             | Liveness probe (`{"status": "ok"}`).                   |
| GET    | `/schema`             | Returns `docs/web/trace.schema.json` byte-for-byte.    |
| POST   | `/trace/convert-csv`  | Convert a `trace_to_dataframe` CSV into JSON.          |

OpenAPI docs are served at `/docs`.

`/trace/convert-csv` responses include a `Schema-Version` header matching the
schema major/minor/patch baked into the payload.

## Running locally

From the repository root:

```bash
# 1. Install the core library (editable) and the backend in one venv.
./scripts/setup.sh
source .venv/bin/activate
pip install -e ./web/backend

# 2. Launch the service.
cd web/backend
uvicorn llm_token_heatmap_api.main:app --reload --port 8000
```

Visit <http://localhost:8000/docs> for interactive API docs.

## Environment variables

| Variable                            | Default                  | Description                                                           |
|-------------------------------------|--------------------------|-----------------------------------------------------------------------|
| `LLM_HEATMAP_API_PORT`              | `8000`                   | Port for local convenience (uvicorn still takes its own `--port`).    |
| `LLM_HEATMAP_ALLOWED_ORIGINS`       | `http://localhost:5173`  | Comma-separated origins allowed by CORS.                              |
| `LLM_HEATMAP_API_WORKERS`           | `1`                      | Uvicorn worker count.                                                 |
| `LLM_HEATMAP_SCHEMA_PATH`           | _(repo-relative)_        | Override path to `trace.schema.json`.                                 |

## Error envelope

Every non-2xx response shares the structured shape:

```json
{
  "error": {
    "kind": "invalid_csv",
    "message": "Uploaded CSV is empty.",
    "details": null
  }
}
```

`kind` matches the frontend's `TraceLoadError` taxonomy:

| `kind`               | Status | Meaning                                                       |
|----------------------|--------|---------------------------------------------------------------|
| `invalid_params`     | 422    | Pydantic validation failed.                                   |
| `invalid_csv`        | 422    | Uploaded CSV is unparseable or missing required columns.      |
| `http_error`         | 4xx    | Generic Starlette HTTP error.                                 |
| `internal_error`     | 500    | Any other server-side failure.                                |

## Running tests

```bash
pip install -e ./web/backend[dev]
pytest web/backend/tests
```
