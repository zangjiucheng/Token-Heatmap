"""Trace CSV-to-JSON conversion, activation-diff, and output-file serving endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from llm_token_heatmap_api import SCHEMA_VERSION
from llm_token_heatmap_api.config import Settings, get_settings
from llm_token_heatmap_api.errors import InvalidActivationTraceError, InvalidCsvError
from llm_token_heatmap_api.services.trace_serializer import csv_to_trace_json

router = APIRouter(prefix="/trace", tags=["trace"])

SCHEMA_VERSION_HEADER = "Schema-Version"


def _with_schema_header(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        content=payload,
        headers={SCHEMA_VERSION_HEADER: SCHEMA_VERSION},
    )


@router.post("/convert-csv")
async def convert_csv(file: UploadFile = File(...)) -> JSONResponse:
    """Accept the CSV emitted by ``trace_to_dataframe`` and return JSON."""
    raw = await file.read()
    if not raw:
        raise InvalidCsvError("Uploaded CSV is empty.")

    payload = csv_to_trace_json(raw)
    return _with_schema_header(payload)


class DiffRequest(BaseModel):
    trace_a: dict[str, Any]
    trace_b: dict[str, Any]
    metric: Literal["l2", "cosine"] = "l2"
    align: Literal["token_id", "position", "auto"] = "auto"


def _project_activation_subset(trace: dict[str, Any], label: str) -> dict[str, Any]:
    """Extract the fields ``compare_activations`` needs from a full trace payload."""
    if "activation_metadata" not in trace:
        raise InvalidActivationTraceError(
            f"{label} has no `activation_metadata` block. "
            "Re-run the producer with --capture-activations to enable diffing.",
            details={"label": label},
        )
    steps = []
    for step in trace.get("steps", []):
        if "activations" not in step:
            continue
        steps.append(
            {
                "step": int(step["step"]),
                "token_id": int(step["token_id"]),
                "decoded_text_offset": int(step["decoded_text_offset"]),
                "activations": step["activations"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "activation_metadata": trace["activation_metadata"],
        "steps": steps,
    }


@router.get("/outputs/{path:path}")
async def serve_output_file(
    path: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve a file from the configured output directory (LLM_HEATMAP_OUTPUT_DIR).

    Rejects path-traversal attempts and returns 404 when the output dir is
    not configured or the file does not exist.
    """
    if settings.output_dir is None:
        raise HTTPException(
            status_code=404,
            detail="Output directory not configured (set LLM_HEATMAP_OUTPUT_DIR).",
        )
    base = settings.output_dir.resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Path traversal rejected.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(target)


@router.post("/diff")
async def diff_traces(body: DiffRequest) -> JSONResponse:
    """Compare two activation traces and return an activation-diff payload.

    Both ``trace_a`` and ``trace_b`` must include an ``activation_metadata``
    block (i.e. be produced with ``--capture-activations``). The response
    conforms to ``docs/web/activation-diff.schema.json``.
    """
    from llm_token_heatmap import compare_activations

    proj_a = _project_activation_subset(body.trace_a, "trace_a")
    proj_b = _project_activation_subset(body.trace_b, "trace_b")

    diff = compare_activations(proj_a, proj_b, metric=body.metric, align=body.align)
    return JSONResponse(content=diff)
