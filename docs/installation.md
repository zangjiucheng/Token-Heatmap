# Installation

## Requirements

- Python 3.10+
- PyTorch (CUDA optional; CPU works for small models)
- Node 20+ — only needed if you want to run the web app

## One-shot setup

```bash
git clone <repo-url> llm-token-heatmap
cd llm-token-heatmap

./scripts/setup.sh                  # everything
```

`scripts/setup.sh` is idempotent and does all of the following:

1. Creates `.venv` if it doesn't already exist.
2. Installs the core package in editable mode (`pip install -r requirements.txt`, `pip install -e ".[dev]"`).
3. Installs the FastAPI backend in editable mode (`pip install -e ./web/backend`).
4. Runs `npm install` in `web/frontend` — but only if `npm` is on `PATH`. If you don't have Node, that step is skipped with a friendly message; install Node 20+ and re-run later if you want the web app.

After it finishes, the `token-heatmap` CLI is on your `PATH` (see [`cli.md`](cli.md)).

## Running the web app

The dev script starts the FastAPI backend and the Vite frontend together and
tears them both down on Ctrl+C:

```bash
source .venv/bin/activate
./scripts/dev.sh                       # backend :8000, frontend :5173
```

Override ports with environment variables:

```bash
BACKEND_PORT=8765 FRONTEND_PORT=5180 ./scripts/dev.sh
```

You can also run the services manually in two terminals:

```bash
# terminal 1 — backend
cd web/backend
uvicorn llm_token_heatmap_api.main:app --reload --port 8000
```

```bash
# terminal 2 — frontend
cd web/frontend
npm install
npm run dev                                  # http://localhost:5173
```

The frontend calls the backend URL from `VITE_API_BASE_URL`, defaulting to
`http://localhost:8000`. Set `VITE_API_BASE_URL=http://localhost:8000` in
`web/frontend/.env.local` if you run the backend on a non-default host or port.

| Service             | URL                        |
| ------------------- | -------------------------- |
| Frontend (Vite dev) | http://localhost:5173      |
| Backend             | http://localhost:8000      |
| API docs (Swagger)  | http://localhost:8000/docs |
