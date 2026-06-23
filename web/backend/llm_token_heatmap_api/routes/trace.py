"""Trace CSV-to-JSON conversion, activation-diff, and output-file serving endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from llm_token_heatmap.runner import GenerateTraceConfig, generate_trace_payload
from llm_token_heatmap_api import SCHEMA_VERSION
from llm_token_heatmap_api.config import Settings, get_settings
from llm_token_heatmap_api.errors import (
    GenerationError,
    InvalidActivationTraceError,
    InvalidCsvError,
)
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


class GenerateRequest(BaseModel):
    """Parameters for server-side trace generation. Bounds mirror the CLI."""

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0, le=5)
    top_p: float = Field(default=0.95, gt=0, le=1)
    min_k: int = Field(default=8, ge=1, le=10000)
    max_k: int = Field(default=64, ge=1, le=10000)
    mass_threshold: float = Field(default=0.95, gt=0, le=1)
    capture_attention: bool = False
    capture_logit_lens: bool = False
    capture_activations: bool = False

    @model_validator(mode="after")
    def _max_k_ge_min_k(self) -> GenerateRequest:
        if self.max_k < self.min_k:
            raise ValueError("max_k must be >= min_k")
        return self


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


@router.post("/generate")
async def generate_trace(body: GenerateRequest) -> JSONResponse:
    """Generate a trace on the server and return the canonical JSON payload.

    Loads ``body.model`` with ``trust_remote_code=True`` (which can execute
    model-author code) and runs the adaptive token probe plus any requested
    *inline* captures (attention / logit-lens / activations). Generation is
    blocking and serialized on the server, so it runs in a worker thread.

    Security: only expose this endpoint over a trusted channel (e.g. an SSH
    tunnel), never the public internet — an attacker who can reach it can load
    an arbitrary model and run its code.
    """
    config = GenerateTraceConfig(
        model=body.model,
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        min_k=body.min_k,
        max_k=body.max_k,
        mass_threshold=body.mass_threshold,
        capture_attention=body.capture_attention,
        capture_logit_lens=body.capture_logit_lens,
        capture_activations=body.capture_activations,
    )
    try:
        payload = await run_in_threadpool(generate_trace_payload, config)
    except (OSError, ValueError) as exc:
        # Unknown model id / repo-not-found / config transformers rejects —
        # client-fixable, so 422 rather than 500.
        raise GenerationError(
            f"Could not load or run model {body.model!r}: {exc}",
            status_code=422,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Generation failed: {exc}") from exc
    return _with_schema_header(payload)
