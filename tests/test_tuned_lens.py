"""Tests for the tuned lens (``token-heatmap tuned-lens``)."""

from __future__ import annotations

import pytest
import torch

from llm_token_heatmap.probes.tuned_lens import (
    apply_tuned_lens,
    compute_tuned_lens,
    fit_tuned_lens,
)
from tests.fixtures.tiny_attention_model import build_tiny_model

_CORPUS = [
    "one two three four five",
    "the cat sat on the mat",
    "hello world foo bar baz qux",
    "a quick brown fox jumps high",
]


class _Tok:
    """Deterministic word→id tokenizer for the tiny model's small vocab."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [(sum(ord(c) for c in w) % 28) + 1 for w in text.split()] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _trace() -> dict:
    return {
        "tokens": {"prompt_token_ids": [1, 2, 3]},
        "steps": [
            {"selected": {"token": " a", "token_id": 4}},
            {"selected": {"token": " b", "token_id": 5}},
        ],
    }


def test_fit_produces_affine_probe_per_layer() -> None:
    model = build_tiny_model(num_hidden_layers=3)
    probes, num_layers = fit_tuned_lens(model, _Tok(), corpus=_CORPUS, ridge=1.0)
    assert num_layers == 3
    assert set(probes) == {0, 1, 2}
    d = next(iter(probes.values())).shape[1]
    for w in probes.values():
        assert w.shape == (d + 1, d)  # affine: weight rows + a bias row


def test_apply_produces_per_layer_rank_and_prob() -> None:
    model = build_tiny_model(num_hidden_layers=3)
    probes, _ = fit_tuned_lens(model, _Tok(), corpus=_CORPUS)
    tuned = apply_tuned_lens(model, _Tok(), probes, _trace())

    assert len(tuned["steps"]) == 2
    step0 = tuned["steps"][0]
    assert step0["step"] == 0 and len(step0["layers"]) == 3
    cell = step0["layers"][0]
    assert {"layer_idx", "selected_token_rank", "selected_token_prob"} <= set(cell)
    assert cell["selected_token_rank"] >= 1
    assert 0.0 <= cell["selected_token_prob"] <= 1.0


def test_compute_end_to_end() -> None:
    model = build_tiny_model(num_hidden_layers=2)
    tuned = compute_tuned_lens(model, _Tok(), _trace(), corpus=_CORPUS)
    assert tuned["num_layers"] == 2
    assert len(tuned["steps"]) == 2


def test_apply_requires_tokens_and_steps() -> None:
    model = build_tiny_model(num_hidden_layers=2)
    probes, _ = fit_tuned_lens(model, _Tok(), corpus=_CORPUS)
    with pytest.raises(ValueError, match="tokens.prompt_token_ids"):
        apply_tuned_lens(model, _Tok(), probes, {"steps": []})
