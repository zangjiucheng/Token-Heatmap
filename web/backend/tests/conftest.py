"""Shared fixtures for the backend test suite."""

from __future__ import annotations

import pytest

from llm_token_heatmap_api.config import Settings, reset_settings_cache
from llm_token_heatmap_api.main import create_app


@pytest.fixture
def test_settings(monkeypatch) -> Settings:
    """Settings overrides applied via environment to drive the real loader."""
    monkeypatch.setenv(
        "LLM_HEATMAP_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    )
    reset_settings_cache()
    yield Settings()
    reset_settings_cache()


@pytest.fixture
def app(test_settings):
    return create_app(settings=test_settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
