"""Structured error envelope and FastAPI exception handlers.

Every non-2xx response from this service shares the shape::

    {"error": {"kind": "<stable_enum>", "message": "...", "details": {...}?}}

The ``kind`` taxonomy matches the frontend's ``TraceLoadError`` enum.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorBody(BaseModel):
    kind: str
    message: str
    details: Any | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class APIError(Exception):
    """Base class for service errors. Subclasses set status code and kind."""

    status_code: int = 500
    kind: str = "internal_error"

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidCsvError(APIError):
    status_code = 422
    kind = "invalid_csv"


def _envelope(kind: str, message: str, details: Any | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"kind": kind, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.kind, exc.message, exc.details),
    )


async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_envelope(
            "invalid_params",
            "Request validation failed.",
            details=exc.errors(),
        ),
    )


async def http_exception_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(
            "http_error",
            str(exc.detail) if exc.detail is not None else "HTTP error.",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire all structured-envelope handlers onto a FastAPI app."""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
