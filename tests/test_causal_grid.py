"""Tests for the causal-effect grid (``token-heatmap causal``)."""

from __future__ import annotations

import pytest

from llm_token_heatmap.analysis.causal_grid import compute_causal_grid
from tests.fixtures.tiny_attention_model import build_tiny_model


def _trace(n_layers: int) -> dict:
    return {
        "tokens": {"prompt_token_ids": [1, 2, 3]},
        "attention_metadata": {"num_layers": n_layers},
        "steps": [
            {"selected": {"token": " a", "token_id": 4}},
            {"selected": {"token": " b", "token_id": 5}},
        ],
    }


def test_causal_grid_shape_and_cells() -> None:
    model = build_tiny_model(num_hidden_layers=3)
    grid = compute_causal_grid(model, None, _trace(3), components=("attn", "mlp"))

    assert grid["num_layers"] == 3
    assert grid["components"] == ["attn", "mlp"]
    assert len(grid["steps"]) == 2

    step0 = grid["steps"][0]
    assert step0["step"] == 0 and step0["token"] == " a"
    assert 0.0 <= step0["base_prob"] <= 1.0
    assert len(step0["layers"]) == 3

    cell = step0["layers"][0]
    assert cell["layer"] == 0
    for comp in ("attn", "mlp"):
        assert "kl" in cell[comp] and "dp" in cell[comp]
        # KL(base || patched) is non-negative; dp is a probability delta.
        assert cell[comp]["kl"] >= -1e-6
        assert -1.0 - 1e-6 <= cell[comp]["dp"] <= 1.0 + 1e-6


def test_causal_grid_single_component() -> None:
    model = build_tiny_model(num_hidden_layers=2)
    grid = compute_causal_grid(model, None, _trace(2), components=("attn",))
    assert grid["components"] == ["attn"]
    assert "attn" in grid["steps"][0]["layers"][0]
    assert "mlp" not in grid["steps"][0]["layers"][0]


def test_causal_grid_rejects_unknown_component() -> None:
    model = build_tiny_model(num_hidden_layers=2)
    with pytest.raises(ValueError, match="unknown component"):
        compute_causal_grid(model, None, _trace(2), components=("bogus",))


def test_causal_grid_requires_tokens_and_steps() -> None:
    model = build_tiny_model(num_hidden_layers=2)
    with pytest.raises(ValueError, match="tokens.prompt_token_ids"):
        compute_causal_grid(model, None, {"steps": []})
