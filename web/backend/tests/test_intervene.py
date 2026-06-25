"""Tests for ``POST /trace/intervene``.

The real path loads a model, so we monkeypatch ``intervene_payload`` and exercise
request validation + the success/error envelopes without torch.
"""

from __future__ import annotations

import llm_token_heatmap_api.routes.trace as trace_routes


def _valid_body(**overrides) -> dict:
    body = {
        "model": "tiny-model",
        "prompt": "hello",
        "interventions": [{"layer": 1, "component": "attn", "op": "zero"}],
    }
    body.update(overrides)
    return body


_CANNED = {
    "target_token_id": 5,
    "baseline": {"top": [], "target_prob": 0.6, "target_logit": 3.1},
    "patched": {"top": [], "target_prob": 0.1, "target_logit": 0.4},
    "diff": {"kl": 2.3, "target_prob_delta": -0.5, "target_logit_delta": -2.7, "top_flips": []},
    "interventions": [],
}


def test_intervene_returns_diff_payload(client, monkeypatch) -> None:
    def fake(config):
        assert config.model == "tiny-model"
        assert config.interventions == [
            {"layer": 1, "component": "attn", "head": None, "op": "zero", "factor": 0.0}
        ]
        return _CANNED

    monkeypatch.setattr(trace_routes, "intervene_payload", fake)
    response = client.post("/trace/intervene", json=_valid_body())
    assert response.status_code == 200, response.text
    assert response.json() == _CANNED


def test_intervene_forwards_continuation_and_target(client, monkeypatch) -> None:
    seen: dict = {}

    def fake(config):
        seen["continuation"] = config.continuation_token_ids
        seen["target"] = config.target_token_id
        return _CANNED

    monkeypatch.setattr(trace_routes, "intervene_payload", fake)
    response = client.post(
        "/trace/intervene",
        json=_valid_body(continuation_token_ids=[10, 11], target_token_id=42),
    )
    assert response.status_code == 200, response.text
    assert seen == {"continuation": [10, 11], "target": 42}


def test_intervene_forwards_head_component(client, monkeypatch) -> None:
    seen: dict = {}

    def fake(config):
        seen["iv"] = config.interventions[0]
        return _CANNED

    monkeypatch.setattr(trace_routes, "intervene_payload", fake)
    response = client.post(
        "/trace/intervene",
        json=_valid_body(
            interventions=[{"layer": 5, "component": "head", "head": 7, "op": "zero"}]
        ),
    )
    assert response.status_code == 200, response.text
    assert seen["iv"] == {
        "layer": 5,
        "component": "head",
        "head": 7,
        "op": "zero",
        "factor": 0.0,
    }


def test_intervene_rejects_empty_interventions(client) -> None:
    response = client.post("/trace/intervene", json=_valid_body(interventions=[]))
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_params"


def test_intervene_rejects_bad_component(client) -> None:
    response = client.post(
        "/trace/intervene",
        json=_valid_body(interventions=[{"layer": 0, "component": "bogus"}]),
    )
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_params"


def test_intervene_maps_model_load_failure_to_422(client, monkeypatch) -> None:
    def boom(config):
        raise OSError("repo not found")

    monkeypatch.setattr(trace_routes, "intervene_payload", boom)
    response = client.post("/trace/intervene", json=_valid_body())
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "generation_failed"
