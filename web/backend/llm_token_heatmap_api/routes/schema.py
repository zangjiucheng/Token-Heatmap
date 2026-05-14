"""Serve the canonical trace and activation JSON Schemas."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from llm_token_heatmap_api.config import Settings, get_settings
from llm_token_heatmap_api.errors import APIError

router = APIRouter(tags=["schema"])


class _SchemaFileMissingError(APIError):
    status_code = 500
    kind = "schema_unavailable"


def _serve_schema_file(path: Path) -> Response:
    if not path.is_file():
        raise _SchemaFileMissingError(
            f"Schema file not found at {path}.",
            details={"path": str(path)},
        )
    return Response(content=path.read_bytes(), media_type="application/schema+json")


@router.get("/schema")
def get_schema(settings: Settings = Depends(get_settings)) -> Response:
    """Return ``docs/web/trace.schema.json`` byte-for-byte.

    Uses ``application/schema+json`` as the content type so caches and the
    frontend can distinguish a schema document from a plain JSON payload.
    """
    return _serve_schema_file(settings.schema_path)


@router.get("/schema/activation")
def get_activation_schema(settings: Settings = Depends(get_settings)) -> Response:
    """Return ``docs/web/activation.schema.json`` byte-for-byte."""
    return _serve_schema_file(settings.activation_schema_path)


@router.get("/schema/activation-diff")
def get_activation_diff_schema(settings: Settings = Depends(get_settings)) -> Response:
    """Return ``docs/web/activation-diff.schema.json`` byte-for-byte."""
    return _serve_schema_file(settings.activation_diff_schema_path)


@router.get("/schema/activation-sidecar")
def get_activation_sidecar_schema(settings: Settings = Depends(get_settings)) -> Response:
    """Return ``docs/web/activation-sidecar.schema.json`` byte-for-byte."""
    return _serve_schema_file(settings.activation_sidecar_schema_path)
