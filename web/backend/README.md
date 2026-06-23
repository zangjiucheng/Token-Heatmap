# LLM Token Heatmap API

FastAPI service that exposes the canonical trace JSON Schema and converts the
`llm_token_heatmap` library's CSV export into the JSON payload the frontend
consumes.

Traces can be produced two ways: upload an existing one through the SPA
(drag-and-drop a JSON file, or upload a CSV to be converted server-side), or
**generate** one on the backend from a prompt via `POST /trace/generate` — the
"Generate" panel in the SPA drives this. Generation requires the full
`llm_token_heatmap` stack (torch/transformers) on the server and is best run on
a GPU box.

> ⚠️ **Security:** `/trace/generate` loads the requested model with
> `trust_remote_code=True`, which can execute model-author code, and accepts an
> arbitrary model id. Only expose this service over a trusted channel (e.g. an
> SSH tunnel: `ssh -L 8000:localhost:8000 user@host`), never the public
> internet. Generation is blocking and serialized server-side (one at a time).

## Endpoints

| Method | Path                  | Description                                            |
|--------|-----------------------|--------------------------------------------------------|
| GET    | `/health`             | Liveness probe (`{"status": "ok"}`).                   |
| GET    | `/schema`             | Returns `docs/web/trace.schema.json` byte-for-byte.    |
| POST   | `/trace/convert-csv`  | Convert a `trace_to_dataframe` CSV into JSON.          |
| POST   | `/trace/generate`     | Generate a trace from `model` + `prompt` + params.     |
| POST   | `/trace/diff`         | Compare two activation traces.                         |
| GET    | `/trace/outputs/{p}`  | Serve a file from `LLM_HEATMAP_OUTPUT_DIR`.            |

OpenAPI docs are served at `/docs`.

`/trace/convert-csv` and `/trace/generate` responses include a `Schema-Version`
header matching the schema major/minor/patch baked into the payload.

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
| `generation_failed`  | 422/500| Model load/run failed (422 for a bad model id, 500 otherwise).|
| `http_error`         | 4xx    | Generic Starlette HTTP error.                                 |
| `internal_error`     | 500    | Any other server-side failure.                                |

## Running tests

```bash
pip install -e ./web/backend[dev]
pytest web/backend/tests
```
