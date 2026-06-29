"""Tests for the prompt-position logit lens (decode every prompt position)."""

from __future__ import annotations

from llm_token_heatmap.probes.prompt_logit_lens import compute_prompt_logit_lens
from tests.fixtures.tiny_attention_model import build_tiny_model


def test_decodes_every_position_at_every_layer() -> None:
    model = build_tiny_model(num_hidden_layers=3, vocab_size=32)
    ids = [1, 5, 9, 3]
    out = compute_prompt_logit_lens(model, None, ids, layers="all", top_k=4)
    assert out is not None
    assert out["num_layers"] == 3
    assert len(out["positions"]) == len(ids)  # one entry per prompt position

    pos0 = out["positions"][0]
    assert pos0["position"] == 0
    assert pos0["token_id"] == 1
    assert len(pos0["layers"]) == 3  # all layers decoded

    layer0 = pos0["layers"][0]
    assert layer0["layer_idx"] == 0
    assert len(layer0["top_k"]) == 4
    assert layer0["top_k"][0]["rank"] == 1
    probs = [c["prob"] for c in layer0["top_k"]]
    assert probs == sorted(probs, reverse=True)  # top-k is descending
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_respects_layer_subset_and_topk() -> None:
    model = build_tiny_model(num_hidden_layers=4, vocab_size=32)
    out = compute_prompt_logit_lens(model, None, [2, 7, 1], layers=[0, 2], top_k=3)
    assert out is not None
    layers_seen = {e["layer_idx"] for e in out["positions"][0]["layers"]}
    assert layers_seen == {0, 2}
    assert all(len(e["top_k"]) == 3 for e in out["positions"][0]["layers"])


def test_returns_none_on_empty_input() -> None:
    model = build_tiny_model(num_hidden_layers=2, vocab_size=32)
    assert compute_prompt_logit_lens(model, None, [], layers="all") is None
