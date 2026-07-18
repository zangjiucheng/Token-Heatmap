"""Tests for the prompt-position logit lens (decode every prompt position)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_token_heatmap.probes.prompt_logit_lens import compute_prompt_logit_lens
from llm_token_heatmap.viz.plotting import plot_prompt_logit_lens_heatmap
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


def test_kl_metrics_per_position() -> None:
    """Every (position, layer) entry carries kl_to_final (0 at the final
    captured layer) and kl_from_prev on all but the earliest layer."""

    model = build_tiny_model(num_hidden_layers=3, vocab_size=32)
    out = compute_prompt_logit_lens(model, None, [1, 5, 9], layers="all", top_k=3)
    assert out is not None

    for pos_entry in out["positions"]:
        layers = pos_entry["layers"]
        assert layers[0].get("kl_from_prev") is None
        for i, layer in enumerate(layers):
            assert layer["kl_to_final"] >= 0.0
            if i > 0:
                assert layer["kl_from_prev"] >= 0.0
        assert layers[-1]["kl_to_final"] == 0.0


def test_plot_prompt_logit_lens_heatmap_smoke(tmp_path: Path) -> None:
    model = build_tiny_model(num_hidden_layers=3, vocab_size=32)
    ids = [1, 5, 9, 3]
    payload = compute_prompt_logit_lens(model, None, ids, layers="all", top_k=4)
    assert payload is not None

    out_path = tmp_path / "prompt_logit_lens_heatmap.png"
    fig = plot_prompt_logit_lens_heatmap(payload, save_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Grid shape: one row per layer, one column per prompt position.
    image = fig.axes[0].get_images()[0]
    assert image.get_array().shape == (3, len(ids))


def test_plot_prompt_logit_lens_heatmap_rejects_empty_payload() -> None:
    with pytest.raises(ValueError):
        plot_prompt_logit_lens_heatmap({"positions": []})
