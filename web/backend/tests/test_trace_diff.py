"""Tests for ``POST /trace/diff``."""

from __future__ import annotations

from typing import Any


def _make_activation_step(
    step: int,
    token_id: int,
    offset: int,
    l2: float,
    value: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "token_id": token_id,
        "decoded_text_offset": offset,
        "activations": [
            {
                "layer": 0,
                "submodule": "resid_post",
                "l2_norm": l2,
                "mean_abs": abs(value) / 4,
                "sparsity": 0.0,
                "top_neurons": [{"index": 0, "value": value}],
            }
        ],
    }


def _make_trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap activation steps into a minimal full-trace payload."""
    return {
        "schema_version": "2.0.0",
        "metadata": {
            "model": "fake/model",
            "prompt": "hello",
            "generated_text": "hello world",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "generation_params": {
                "max_new_tokens": 2,
                "temperature": 1.0,
                "top_p": 1.0,
                "sample_top_k": 0,
            },
            "probe_config": {"min_k": 1, "max_k": 4, "mass_threshold": 0.95},
        },
        "tokens": {"prompt_token_ids": [1, 2], "prompt_tokens": ["he", "llo"]},
        "activation_metadata": {
            "captured_submodules": ["resid_post"],
            "num_layers": 1,
            "hidden_dim": 4,
            "tokenizer_fingerprint": "sha256:fake",
            "captured_layers": [0],
        },
        "steps": steps,
    }


def test_diff_identical_traces_returns_zero_l2(client) -> None:
    steps = [
        _make_activation_step(0, 10, 0, 1.0, 1.0),
        _make_activation_step(1, 11, 5, 0.5, 0.5),
    ]
    trace = _make_trace(steps)

    resp = client.post(
        "/trace/diff",
        json={"trace_a": trace, "trace_b": trace, "metric": "l2", "align": "auto"},
    )
    assert resp.status_code == 200, resp.text
    diff = resp.json()
    assert diff["alignment"]["mode"] == "token_id"
    assert diff["alignment"]["mismatches"] == []
    assert len(diff["steps"]) == 2
    for step in diff["steps"]:
        for delta in step["delta"]:
            assert delta["l2"] == 0.0


def test_diff_different_traces_returns_nonzero_l2(client) -> None:
    steps_a = [_make_activation_step(0, 10, 0, 1.0, 1.0)]
    steps_b = [_make_activation_step(0, 10, 0, 2.0, -1.0)]
    trace_a = _make_trace(steps_a)
    trace_b = _make_trace(steps_b)

    resp = client.post(
        "/trace/diff",
        json={"trace_a": trace_a, "trace_b": trace_b, "metric": "l2"},
    )
    assert resp.status_code == 200, resp.text
    diff = resp.json()
    assert len(diff["steps"]) == 1
    l2 = diff["steps"][0]["delta"][0]["l2"]
    assert l2 > 0.0


def test_diff_rejects_trace_without_activation_metadata(client) -> None:
    no_activation_trace = {
        "schema_version": "2.0.0",
        "metadata": {"model": "fake/model"},
        "tokens": {"prompt_token_ids": [], "prompt_tokens": []},
        "steps": [],
    }
    steps = [_make_activation_step(0, 10, 0, 1.0, 1.0)]
    valid_trace = _make_trace(steps)

    resp = client.post(
        "/trace/diff",
        json={"trace_a": no_activation_trace, "trace_b": valid_trace},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["kind"] == "invalid_activation_trace"
    assert "trace_a" in body["error"]["message"]


def test_diff_cosine_metric(client) -> None:
    steps_a = [_make_activation_step(0, 10, 0, 1.0, 1.0)]
    steps_b = [_make_activation_step(0, 10, 0, 1.0, 0.5)]
    resp = client.post(
        "/trace/diff",
        json={
            "trace_a": _make_trace(steps_a),
            "trace_b": _make_trace(steps_b),
            "metric": "cosine",
        },
    )
    assert resp.status_code == 200, resp.text
    diff = resp.json()
    cosine = diff["steps"][0]["delta"][0]["cosine"]
    assert -1.0 <= cosine <= 1.0


def test_diff_rejects_invalid_metric(client) -> None:
    steps = [_make_activation_step(0, 10, 0, 1.0, 1.0)]
    trace = _make_trace(steps)
    resp = client.post(
        "/trace/diff",
        json={"trace_a": trace, "trace_b": trace, "metric": "manhattan"},
    )
    assert resp.status_code == 422
