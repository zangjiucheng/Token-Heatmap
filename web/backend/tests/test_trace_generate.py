"""Tests for ``POST /trace/generate``.

The real generator loads a HuggingFace model, so every test here monkeypatches
``generate_trace_payload`` — we exercise request validation, the success
envelope, and error mapping without touching torch/transformers.
"""

from __future__ import annotations

import llm_token_heatmap_api.routes.trace as trace_routes


def _valid_body(**overrides) -> dict:
    body = {"model": "tiny-model", "prompt": "hello"}
    body.update(overrides)
    return body


def test_generate_returns_payload_with_schema_header(client, monkeypatch) -> None:
    canned = {"schema_version": "2.0.0", "metadata": {}, "steps": []}

    def fake_generate(config):
        # The route should hand us a fully-populated config object.
        assert config.model == "tiny-model"
        assert config.prompt == "hello"
        assert config.max_new_tokens == 64
        return canned

    monkeypatch.setattr(trace_routes, "generate_trace_payload", fake_generate)

    response = client.post("/trace/generate", json=_valid_body())
    assert response.status_code == 200, response.text
    assert response.json() == canned
    assert response.headers["Schema-Version"] == "2.0.0"


def test_generate_forwards_capture_flags(client, monkeypatch) -> None:
    seen: dict = {}

    def fake_generate(config):
        seen["capture_attention"] = config.capture_attention
        seen["capture_logit_lens"] = config.capture_logit_lens
        seen["capture_activations"] = config.capture_activations
        return {"schema_version": "2.0.0", "metadata": {}, "steps": []}

    monkeypatch.setattr(trace_routes, "generate_trace_payload", fake_generate)

    response = client.post(
        "/trace/generate",
        json=_valid_body(capture_attention=True, capture_activations=True),
    )
    assert response.status_code == 200, response.text
    assert seen == {
        "capture_attention": True,
        "capture_logit_lens": False,
        "capture_activations": True,
    }


def test_generate_rejects_out_of_range_max_new_tokens(client) -> None:
    response = client.post("/trace/generate", json=_valid_body(max_new_tokens=99999))
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_params"


def test_generate_rejects_max_k_less_than_min_k(client) -> None:
    response = client.post("/trace/generate", json=_valid_body(min_k=10, max_k=5))
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_params"


def test_generate_requires_non_empty_prompt(client) -> None:
    response = client.post("/trace/generate", json={"model": "m", "prompt": ""})
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_params"


def test_generate_maps_model_load_failure_to_422(client, monkeypatch) -> None:
    def boom(config):
        raise OSError("repo not found")

    monkeypatch.setattr(trace_routes, "generate_trace_payload", boom)

    response = client.post("/trace/generate", json=_valid_body())
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["kind"] == "generation_failed"
    assert "tiny-model" in body["error"]["message"]


def test_generate_maps_unexpected_failure_to_500(client, monkeypatch) -> None:
    def boom(config):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(trace_routes, "generate_trace_payload", boom)

    response = client.post("/trace/generate", json=_valid_body())
    assert response.status_code == 500
    assert response.json()["error"]["kind"] == "generation_failed"
