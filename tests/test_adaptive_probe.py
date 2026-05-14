"""Unit tests for `AdaptiveTokenProbe`."""

from __future__ import annotations

import pytest
import torch

from llm_token_heatmap.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe

EXPECTED_KEYS_NO_SELECTED = {
    "top_ids",
    "top_probs",
    "top_logprobs",
    "valid_mask",
    "k_used",
    "entropy",
    "top_mass_used",
}

EXPECTED_KEYS_WITH_SELECTED = EXPECTED_KEYS_NO_SELECTED | {
    "selected_ids",
    "selected_prob",
    "selected_logprob",
    "selected_rank",
}


def _build_probe(
    min_k: int = 4, max_k: int = 16, mass_threshold: float = 0.95
) -> AdaptiveTokenProbe:
    return AdaptiveTokenProbe(
        AdaptiveProbeConfig(min_k=min_k, max_k=max_k, mass_threshold=mass_threshold)
    )


def test_output_keys_without_selected(uniform_logits_factory):
    probe = _build_probe()
    out = probe(uniform_logits_factory(32))

    assert EXPECTED_KEYS_NO_SELECTED.issubset(out.keys())
    assert "selected_rank" not in out


def test_output_keys_with_selected(uniform_logits_factory):
    probe = _build_probe()
    selected = torch.tensor([0])
    out = probe(uniform_logits_factory(32), selected_ids=selected)

    assert EXPECTED_KEYS_WITH_SELECTED.issubset(out.keys())


def test_output_shapes(threshold_logits_factory):
    vocab = 64
    probe = _build_probe(min_k=4, max_k=16)
    out = probe(threshold_logits_factory(vocab))

    assert out["top_ids"].shape == (1, 16)
    assert out["top_probs"].shape == (1, 16)
    assert out["top_logprobs"].shape == (1, 16)
    assert out["valid_mask"].shape == (1, 16)
    assert out["k_used"].shape == (1,)
    assert out["entropy"].shape == (1,)


def test_entropy_non_negative(uniform_logits_factory, sharp_logits_factory):
    probe = _build_probe()
    out_uniform = probe(uniform_logits_factory(32))
    out_sharp = probe(sharp_logits_factory(32))

    assert float(out_uniform["entropy"][0]) >= 0.0
    assert float(out_sharp["entropy"][0]) >= 0.0


def test_k_used_within_bounds(threshold_logits_factory):
    vocab = 64
    probe = _build_probe(min_k=4, max_k=16)
    out = probe(threshold_logits_factory(vocab))

    effective_max_k = min(probe.config.max_k, vocab)
    k_used = int(out["k_used"][0])
    assert probe.config.min_k <= k_used <= effective_max_k


def test_k_used_respects_threshold(threshold_logits_factory):
    """A distribution where top-3 covers >95% mass should yield k_used == max(min_k, 3)."""

    probe = _build_probe(min_k=2, max_k=16, mass_threshold=0.95)
    out = probe(threshold_logits_factory(64, top=3))

    assert int(out["k_used"][0]) == max(probe.config.min_k, 3)


def test_k_used_clamped_to_min_k(sharp_logits_factory):
    """A sharp distribution where top-1 covers ~99% should still yield k_used == min_k."""

    probe = _build_probe(min_k=5, max_k=16, mass_threshold=0.95)
    out = probe(sharp_logits_factory(64, peak_value=50.0))

    assert int(out["k_used"][0]) == probe.config.min_k
    assert float(out["top_probs"][0, 0]) > 0.99


def test_k_used_falls_back_to_max_k(uniform_logits_factory):
    """A uniform distribution should fall back to effective_max_k."""

    probe = _build_probe(min_k=4, max_k=16, mass_threshold=0.95)
    out = probe(uniform_logits_factory(64))

    effective_max_k = min(probe.config.max_k, 64)
    assert int(out["k_used"][0]) == effective_max_k


def test_selected_rank_for_argmax(threshold_logits_factory):
    """When `selected_ids` is the argmax of the logits, selected_rank == 1."""

    logits = threshold_logits_factory(64, top=3)
    selected = torch.argmax(logits, dim=-1)

    probe = _build_probe()
    out = probe(logits, selected_ids=selected)

    assert int(out["selected_rank"][0]) == 1


def test_selected_rank_greater_than_one_for_non_argmax():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    selected = torch.tensor([2])

    probe = _build_probe(min_k=2, max_k=4)
    out = probe(logits, selected_ids=selected)

    assert int(out["selected_rank"][0]) == 3


def test_valid_mask_count_equals_k_used(threshold_logits_factory):
    probe = _build_probe(min_k=4, max_k=16)
    out = probe(threshold_logits_factory(64, top=3))

    assert int(out["valid_mask"][0].sum()) == int(out["k_used"][0])


def test_invalid_logits_shape_raises():
    probe = _build_probe()
    with pytest.raises(ValueError):
        probe(torch.zeros(3))


def test_invalid_temperature_raises(uniform_logits_factory):
    probe = _build_probe()
    with pytest.raises(ValueError):
        probe(uniform_logits_factory(32), temperature=0.0)
