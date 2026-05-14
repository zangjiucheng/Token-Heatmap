"""Trace CSV-to-JSON conversion endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from llm_token_heatmap_api import SCHEMA_VERSION
from llm_token_heatmap_api.errors import InvalidCsvError
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
