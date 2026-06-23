"""FastAPI application factory for the llm-token-heatmap backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llm_token_heatmap_api import __version__
from llm_token_heatmap_api.config import Settings, get_settings
from llm_token_heatmap_api.errors import register_exception_handlers
from llm_token_heatmap_api.routes import health, schema, trace

# Canonical location of the Vite production build relative to this file:
# web/backend/llm_token_heatmap_api/main.py → repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_DIST = _REPO_ROOT / "web" / "frontend" / "dist"


def _mount_frontend(app: FastAPI) -> None:
    """Mount the pre-built Vite SPA if ``web/frontend/dist/`` exists.

    API routes are registered before this function is called, so they always
    take precedence over the static-file catch-all.

    Build the frontend once (on any machine with Node.js):
        cd web/frontend && VITE_API_BASE_URL='' npm run build
    Then copy ``dist/`` to this server and restart the backend.
    """
    if not _FRONTEND_DIST.is_dir():
        return

    # Vite emits hashed JS/CSS under dist/assets/ — serve them efficiently.
    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="spa-assets")

    index_html = _FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        # Serve exact files that exist in dist/ (favicon, manifest, …).
        candidate = (_FRONTEND_DIST / full_path).resolve()
        # Guard against path traversal before checking existence.
        if str(candidate).startswith(str(_FRONTEND_DIST)) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app.

    Accepts an optional ``Settings`` instance so tests can inject overrides
    without touching the process environment.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="LLM Token Heatmap API",
        description=(
            "Converts the library's long-format CSV into the canonical JSON "
            "trace payload and exposes the trace JSON Schema."
        ),
        version=__version__,
        openapi_tags=[
            {"name": "health", "description": "Liveness checks."},
            {"name": "schema", "description": "Trace JSON Schema."},
            {"name": "trace", "description": "Convert trace payloads."},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # API routers must be registered before the SPA catch-all so they win.
    app.include_router(health.router)
    app.include_router(schema.router)
    app.include_router(trace.router)

    _mount_frontend(app)

    return app


app = create_app()
