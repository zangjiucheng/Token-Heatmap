"""Tests for :mod:`llm_token_heatmap.attention_stats`.

All tests are pure-Python / pure-PyTorch over synthetic ``AttentionStats``
payloads; no model download or network access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from llm_token_heatmap.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe
from llm_token_heatmap.attention_probe import (
    AttentionLayerStats,
    AttentionProbe,
    AttentionProbeConfig,
    AttentionStats,
)
from llm_token_heatmap.attention_stats import (
    AttentionDerivedStats,
    compute_attention_stats,
)
from llm_token_heatmap.generation import generate_with_adaptive_probe
from tests.fixtures.tiny_attention_model import TinyCausalLM, build_tiny_model


def _stats_from_weights(
    weights_per_layer: dict[int, torch.Tensor],
    *,
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    head_dim: int = 4,
) -> AttentionStats:
    layers: dict[int, AttentionLayerStats] = {}
    num_heads = next(iter(weights_per_layer.values())).shape[0]
    for layer_idx, w in weights_per_layer.items():
        layers[layer_idx] = AttentionLayerStats(
            layer_idx=layer_idx,
            attention_weights=w,
            q_last=q,
            k_last=k,
            v_last=v,
        )
    return AttentionStats(
        layers=layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
        head_to_kv_group=list(range(num_heads)),
    )


# --------------------------------------------------------------------------- #
# Acceptance criteria tests
# --------------------------------------------------------------------------- #


def test_compute_attention_stats_is_pure() -> None:
    """Same input -> same output; AttentionStats is not mutated."""

    weights = torch.softmax(torch.randn(3, 5), dim=-1)
    stats = _stats_from_weights({0: weights.clone()})

    snapshot_weights = stats.layers[0].attention_weights.clone()
    a = compute_attention_stats(stats)
    b = compute_attention_stats(stats)

    assert torch.allclose(stats.layers[0].attention_weights, snapshot_weights)
    assert len(a.layers) == len(b.layers)
    for layer_idx, layer_a in a.layers.items():
        layer_b = b.layers[layer_idx]
        for head_a, head_b in zip(layer_a.heads, layer_b.heads, strict=True):
            assert head_a.entropy == head_b.entropy
            assert head_a.self_weight == head_b.self_weight
            assert head_a.specialization_fingerprint == head_b.specialization_fingerprint


def test_entropy_uniform_equals_log_n() -> None:
    n = 8
    uniform = torch.full((1, n), 1.0 / n)
    stats = _stats_from_weights({0: uniform})
    derived = compute_attention_stats(stats)
    head = derived.layers[0].heads[0]
    assert abs(head.entropy - math.log(n)) < 1e-6


def test_entropy_onehot_equals_zero() -> None:
    one_hot = torch.zeros(1, 6)
    one_hot[0, 2] = 1.0
    stats = _stats_from_weights({0: one_hot})
    derived = compute_attention_stats(stats)
    head = derived.layers[0].heads[0]
    assert abs(head.entropy) < 1e-6


def test_self_weight_matches_distribution() -> None:
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    stats = _stats_from_weights({0: weights})
    head = compute_attention_stats(stats).layers[0].heads[0]
    assert head.self_weight == float(weights[0, -1].item())


def test_bos_weight_matches_distribution() -> None:
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    stats = _stats_from_weights({0: weights})
    head = compute_attention_stats(stats).layers[0].heads[0]
    assert head.bos_weight == float(weights[0, 0].item())


def test_qk_alignment_parallel_and_orthogonal() -> None:
    # Two heads: one with parallel Q,K (angle 0), one with orthogonal (angle 90).
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    k = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    # Make the single-position attention so the "top attended" position is the last.
    weights = torch.tensor([[1.0], [1.0]])
    stats = _stats_from_weights({0: weights}, q=q, k=k, head_dim=4)
    derived = compute_attention_stats(stats)
    heads = derived.layers[0].heads
    assert 0.0 <= heads[0].qk_alignment_angle_deg <= 180.0
    assert 0.0 <= heads[1].qk_alignment_angle_deg <= 180.0
    assert abs(heads[0].qk_alignment_angle_deg - 0.0) < 1e-4
    assert abs(heads[1].qk_alignment_angle_deg - 90.0) < 1e-4


def test_effective_span_uniform_eight_positions() -> None:
    n = 8
    uniform = torch.full((1, n), 1.0 / n)
    stats = _stats_from_weights({0: uniform})
    head = compute_attention_stats(stats).layers[0].heads[0]
    assert abs(head.effective_attention_span - 8.0) <= 0.1


def test_layer_aggregates_copy_and_sink_fractions() -> None:
    # Four heads: head 0 is a copy head (self_weight > 0.5),
    # head 1 is a sink head (bos_weight > 0.5),
    # heads 2 and 3 are neither.
    seq_len = 4
    weights = torch.zeros(4, seq_len)
    weights[0, -1] = 0.9
    weights[0, 0] = 0.1
    weights[1, 0] = 0.95
    weights[1, -1] = 0.05
    weights[2] = torch.tensor([0.25, 0.25, 0.25, 0.25])
    weights[3] = torch.tensor([0.4, 0.2, 0.2, 0.2])

    stats = _stats_from_weights({0: weights})
    aggregates = compute_attention_stats(stats).layers[0].aggregates

    assert aggregates.copy_head_fraction == 0.25  # only head 0
    assert aggregates.sink_head_fraction == 0.25  # only head 1
    entropies = []
    for head in compute_attention_stats(stats).layers[0].heads:
        entropies.append(head.entropy)
    assert abs(aggregates.mean_entropy - sum(entropies) / len(entropies)) < 1e-6
    assert abs(aggregates.max_entropy - max(entropies)) < 1e-6


def test_specialization_fingerprint_normalized() -> None:
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(3, 12), dim=-1)
    stats = _stats_from_weights({0: weights})
    derived = compute_attention_stats(stats)
    for head in derived.layers[0].heads:
        assert len(head.specialization_fingerprint) == 16
        total = sum(head.specialization_fingerprint)
        assert abs(total - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Generation integration
# --------------------------------------------------------------------------- #


@dataclass
class _ModelOutput:
    logits: torch.Tensor
    past_key_values: Any


class _AttentionGenerationModel:
    """Wrap a :class:`TinyCausalLM` so it satisfies the generation loop's API.

    The generation loop expects ``model(input_ids=..., use_cache=True)`` to
    return an object with ``logits`` and ``past_key_values`` attributes; on
    subsequent steps it passes only the last token. We do not implement an
    actual KV cache (the tiny model is fast and the per-step probe just needs
    a fresh forward to fill its hook buffers), so we re-run the cumulative
    prefix every step.
    """

    def __init__(self, model: TinyCausalLM) -> None:
        self._model = model
        self.config = model.config
        self.model = model.model  # exposes `.layers` for the probe.
        self.device = torch.device("cpu")
        self._prefix: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> _ModelOutput:
        if past_key_values is None:
            self._prefix = input_ids
        else:
            assert self._prefix is not None
            self._prefix = torch.cat([self._prefix, input_ids], dim=-1)
        logits = self._model(self._prefix)
        return _ModelOutput(logits=logits, past_key_values=("cache", self._prefix.shape[-1]))


class _Tokenizer:
    vocab_size = 32
    eos_token_id = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
        ids = [ord(c) % self.vocab_size for c in prompt[:4]] or [0]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        return "ok"


def test_generation_with_probe_attaches_attention_block() -> None:
    tiny = build_tiny_model(num_hidden_layers=2, num_attention_heads=4, head_dim=4)
    wrapped = _AttentionGenerationModel(tiny)
    probe = AttentionProbe(AttentionProbeConfig(capture_full_distribution=True))
    probe.attach(tiny)

    try:
        _text, trace = generate_with_adaptive_probe(
            wrapped,
            _Tokenizer(),
            prompt="hello",
            probe=AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=2, max_k=4)),
            max_new_tokens=3,
            temperature=1.0,
            top_p=1.0,
            sample_top_k=1,
            attention_probe=probe,
        )
    finally:
        probe.detach()

    assert len(trace) == 3
    for entry in trace:
        assert "attention" in entry
        assert isinstance(entry["attention"], list)
        assert entry["attention"], "attention block must list captured layers"
        for layer_entry in entry["attention"]:
            assert {
                "layer",
                "entropy",
                "self_weight",
                "bos_weight",
                "top_positions",
                "q_norm",
                "k_norm",
                "v_norm",
                "qk_alignment_angle",
            }.issubset(layer_entry.keys())
            assert 0.0 <= layer_entry["self_weight"] <= 1.0
            assert 0.0 <= layer_entry["bos_weight"] <= 1.0
            assert 0.0 <= layer_entry["qk_alignment_angle"] <= 180.0


def test_compute_attention_stats_top_k_per_head() -> None:
    """`top_k` controls per-head top-position lists."""

    seq_len = 6
    weights = torch.zeros(2, seq_len)
    weights[0] = torch.tensor([0.05, 0.05, 0.6, 0.1, 0.1, 0.1])
    weights[1] = torch.tensor([0.7, 0.05, 0.05, 0.1, 0.05, 0.05])
    stats = _stats_from_weights({0: weights})
    derived: AttentionDerivedStats = compute_attention_stats(stats, top_k=3)
    head0 = derived.layers[0].heads[0]
    head1 = derived.layers[0].heads[1]
    assert len(head0.top_k_positions) == 3
    assert head0.top_k_positions[0][0] == 2
    assert head1.top_k_positions[0][0] == 0
