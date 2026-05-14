"""Liveness probe."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Return a static OK payload so orchestrators can verify the process is alive."""
    return {"status": "ok"}
