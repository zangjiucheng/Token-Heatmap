"""Tests for `AttentionProbe`.

All tests use the synthetic `tiny_attention_model` fixture so the suite stays
fast and network-free. The probe's contract is exercised against MHA, GQA,
and RoPE-enabled variants of the same minimal architecture.
"""

from __future__ import annotations

import pytest
import torch

from llm_token_heatmap.attention_probe import (
    AttentionProbe,
    AttentionProbeConfig,
    AttentionProbeError,
    AttentionStats,
)
from tests.fixtures.tiny_attention_model import build_tiny_model

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run_forward(model, seq_len: int = 5, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
    with torch.no_grad():
        return model(input_ids)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_attach_detach_idempotent() -> None:
    model = build_tiny_model()
    probe = AttentionProbe(AttentionProbeConfig())

    assert probe.is_attached is False
    probe.attach(model)
    assert probe.is_attached is True
    handles_after_first = list(probe._handles)
    probe.attach(model)  # second call is a no-op
    assert probe._handles == handles_after_first

    probe.detach()
    assert probe.is_attached is False
    assert probe._handles == []
    probe.detach()  # second detach is a no-op
    assert probe.is_attached is False


def test_eager_attention_forced_and_restored() -> None:
    model = build_tiny_model(attn_implementation="sdpa")
    assert model.config.attn_implementation == "sdpa"
    assert model.config._attn_implementation == "sdpa"

    probe = AttentionProbe(AttentionProbeConfig())
    probe.attach(model)
    assert model.config.attn_implementation == "eager"
    assert model.config._attn_implementation == "eager"

    probe.detach()
    assert model.config.attn_implementation == "sdpa"
    assert model.config._attn_implementation == "sdpa"


def test_capture_shapes_match_config() -> None:
    seq_len = 6
    model = build_tiny_model(num_hidden_layers=2, num_attention_heads=4, head_dim=4)
    probe = AttentionProbe(AttentionProbeConfig(capture_full_distribution=True))
    probe.attach(model)
    try:
        _run_forward(model, seq_len=seq_len)
        stats = probe.capture_step()
    finally:
        probe.detach()

    assert isinstance(stats, AttentionStats)
    assert stats.num_attention_heads == 4
    assert stats.num_key_value_heads == 4
    assert stats.head_dim == 4
    assert set(stats.layers.keys()) == {0, 1}

    for layer_idx, layer in stats.layers.items():
        assert layer.layer_idx == layer_idx
        assert isinstance(layer.attention_weights, torch.Tensor)
        assert layer.attention_weights.shape == (4, seq_len)
        # Softmax row sums to 1 along the key dimension.
        assert torch.allclose(layer.attention_weights.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert layer.q_last is not None and layer.q_last.shape == (4, 4)
        assert layer.k_last is not None and layer.k_last.shape == (4, 4)
        assert layer.v_last is not None and layer.v_last.shape == (4, 4)


def test_gqa_shapes() -> None:
    model = build_tiny_model(
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=4
    )
    probe = AttentionProbe(AttentionProbeConfig(capture_full_distribution=True))
    probe.attach(model)
    try:
        _run_forward(model, seq_len=5)
        stats = probe.capture_step()
    finally:
        probe.detach()

    assert stats.num_attention_heads == 4
    assert stats.num_key_value_heads == 2
    assert stats.head_to_kv_group == [0, 0, 1, 1]

    layer = stats.layers[0]
    assert layer.q_last is not None and layer.q_last.shape == (4, 4)
    assert layer.k_last is not None and layer.k_last.shape == (2, 4)
    assert layer.v_last is not None and layer.v_last.shape == (2, 4)


def test_pre_post_rope_differ() -> None:
    model = build_tiny_model(num_hidden_layers=1, use_rope=True)
    probe = AttentionProbe(
        AttentionProbeConfig(capture_full_distribution=True, capture_pre_rope_qk=True)
    )
    probe.attach(model)
    try:
        _run_forward(model, seq_len=6)
        stats = probe.capture_step()
    finally:
        probe.detach()

    layer = stats.layers[0]
    assert layer.q_last_pre_rope is not None
    assert layer.k_last_pre_rope is not None
    assert layer.q_last is not None
    assert layer.k_last is not None
    # Rotation should have moved Q and K — they must not be exactly equal.
    assert not torch.allclose(layer.q_last, layer.q_last_pre_rope)
    assert not torch.allclose(layer.k_last, layer.k_last_pre_rope)
    # Norms are preserved by rotation; cheap sanity that we captured the rotated form.
    assert torch.allclose(layer.q_last.norm(dim=-1), layer.q_last_pre_rope.norm(dim=-1), atol=1e-4)


def test_sparse_top_k_capture() -> None:
    seq_len = 8
    top_k = 3
    model = build_tiny_model(num_hidden_layers=1, num_attention_heads=4, head_dim=4)
    probe = AttentionProbe(
        AttentionProbeConfig(capture_full_distribution=False, top_k_positions=top_k)
    )
    probe.attach(model)
    try:
        _run_forward(model, seq_len=seq_len)
        stats = probe.capture_step()
    finally:
        probe.detach()

    layer = stats.layers[0]
    assert isinstance(layer.attention_weights, dict)
    positions = layer.attention_weights["positions"]
    weights = layer.attention_weights["weights"]
    assert positions.shape == (4, top_k)
    assert weights.shape == (4, top_k)
    assert positions.dtype == torch.long
    # Top-k weights per head should be sorted descending.
    diffs = weights[:, 1:] - weights[:, :-1]
    assert (diffs <= 1e-6).all()
    # Positions are valid indices into the key sequence.
    assert int(positions.min()) >= 0
    assert int(positions.max()) < seq_len


def test_layer_subset_only_records_requested_layers() -> None:
    model = build_tiny_model(num_hidden_layers=4)
    probe = AttentionProbe(AttentionProbeConfig(layers=[0, 2], capture_full_distribution=True))
    probe.attach(model)
    try:
        assert probe.target_layers == [0, 2]
        _run_forward(model, seq_len=5)
        stats = probe.capture_step()
    finally:
        probe.detach()

    assert set(stats.layers.keys()) == {0, 2}


def test_probe_detached_does_not_alter_generation() -> None:
    # Run baseline forward with no probe ever touching the model.
    model_a = build_tiny_model(seed=42)
    baseline = _run_forward(model_a, seq_len=6, seed=7)

    # Run on a freshly built identical model with attach -> detach cycle.
    model_b = build_tiny_model(seed=42)
    probe = AttentionProbe(AttentionProbeConfig())
    probe.attach(model_b)
    _run_forward(model_b, seq_len=6, seed=7)
    probe.detach()

    after_detach = _run_forward(model_b, seq_len=6, seed=7)
    assert torch.allclose(baseline, after_detach, atol=1e-6)


def test_two_probes_coexist() -> None:
    model_a = build_tiny_model(seed=1, num_hidden_layers=2)
    model_b = build_tiny_model(seed=2, num_hidden_layers=3)

    probe_a = AttentionProbe(AttentionProbeConfig(capture_full_distribution=True))
    probe_b = AttentionProbe(AttentionProbeConfig(layers=[0, 2], capture_full_distribution=True))

    probe_a.attach(model_a)
    probe_b.attach(model_b)
    try:
        _run_forward(model_a, seq_len=4, seed=11)
        _run_forward(model_b, seq_len=5, seed=13)
        stats_a = probe_a.capture_step()
        stats_b = probe_b.capture_step()
    finally:
        probe_a.detach()
        probe_b.detach()

    assert set(stats_a.layers.keys()) == {0, 1}
    assert set(stats_b.layers.keys()) == {0, 2}
    # The two captures are independent: shapes match each model's seq_len.
    assert stats_a.layers[0].attention_weights.shape[-1] == 4
    assert stats_b.layers[0].attention_weights.shape[-1] == 5


def test_unsupported_model_raises_clear_error() -> None:
    class BareModel:
        def __init__(self) -> None:
            self.config = type("Cfg", (), {"num_attention_heads": 4, "hidden_size": 16})()

    probe = AttentionProbe(AttentionProbeConfig())
    with pytest.raises(AttentionProbeError, match="decoder layer list"):
        probe.attach(BareModel())
