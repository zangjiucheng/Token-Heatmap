"""FastAPI application factory for the llm-token-heatmap backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_token_heatmap_api import __version__
from llm_token_heatmap_api.config import Settings, get_settings
from llm_token_heatmap_api.errors import register_exception_handlers
from llm_token_heatmap_api.routes import health, schema, trace


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

    app.include_router(health.router)
    app.include_router(schema.router)
    app.include_router(trace.router)

    return app


app = create_app()
