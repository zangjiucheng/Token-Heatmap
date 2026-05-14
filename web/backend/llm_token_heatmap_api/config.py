"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _docs_web_dir() -> Path:
    """Return the ``docs/web`` directory relative to the repository root."""
    here = Path(__file__).resolve()
    # web/backend/llm_token_heatmap_api/config.py -> repo root is three parents up.
    repo_root = here.parents[3]
    return repo_root / "docs" / "web"


def _default_schema_path() -> Path:
    """Locate the canonical trace schema relative to the repository root."""
    return _docs_web_dir() / "trace.schema.json"


def _default_activation_schema_path() -> Path:
    return _docs_web_dir() / "activation.schema.json"


def _default_activation_diff_schema_path() -> Path:
    return _docs_web_dir() / "activation-diff.schema.json"


def _default_activation_sidecar_schema_path() -> Path:
    return _docs_web_dir() / "activation-sidecar.schema.json"


class Settings(BaseSettings):
    """Backend service settings.

    All values are overridable via environment variables (prefix
    ``LLM_HEATMAP_``) or a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_HEATMAP_",
        env_file=".env",
        extra="ignore",
    )

    api_port: int = Field(default=8000, ge=1, le=65535)
    allowed_origins: str = Field(default="http://localhost:5173")
    schema_path: Path = Field(default_factory=_default_schema_path)
    activation_schema_path: Path = Field(default_factory=_default_activation_schema_path)
    activation_diff_schema_path: Path = Field(
        default_factory=_default_activation_diff_schema_path
    )
    activation_sidecar_schema_path: Path = Field(
        default_factory=_default_activation_sidecar_schema_path
    )

    @field_validator("allowed_origins")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached Settings (used by tests that mutate the environment)."""
    get_settings.cache_clear()
