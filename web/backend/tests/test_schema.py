"""Schema-serving endpoint tests."""

from __future__ import annotations

from pathlib import Path

from llm_token_heatmap_api.config import get_settings


def test_schema_matches_docs_file(client) -> None:
    response = client.get("/schema")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/schema+json")

    schema_path: Path = get_settings().schema_path
    assert response.content == schema_path.read_bytes()


def test_cors_allows_configured_origin(client) -> None:
    response = client.options(
        "/schema",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_get_activation_schema_returns_file_bytes(client) -> None:
    response = client.get("/schema/activation")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/schema+json")

    schema_path: Path = get_settings().activation_schema_path
    assert response.content == schema_path.read_bytes()


def test_get_activation_diff_schema_returns_file_bytes(client) -> None:
    response = client.get("/schema/activation-diff")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/schema+json")

    schema_path: Path = get_settings().activation_diff_schema_path
    assert response.content == schema_path.read_bytes()


def test_get_activation_sidecar_schema_returns_file_bytes(client) -> None:
    response = client.get("/schema/activation-sidecar")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/schema+json")

    schema_path: Path = get_settings().activation_sidecar_schema_path
    assert response.content == schema_path.read_bytes()


def test_openapi_includes_new_schema_endpoints(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/schema/activation" in paths
    assert "/schema/activation-diff" in paths
    assert "/schema/activation-sidecar" in paths
    for route in ("/schema/activation", "/schema/activation-diff", "/schema/activation-sidecar"):
        assert "get" in paths[route]
        assert "schema" in paths[route]["get"]["tags"]
