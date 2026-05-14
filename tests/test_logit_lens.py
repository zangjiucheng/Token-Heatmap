"""Tests for `LogitLens`.

All tests use the synthetic `tiny_attention_model` fixture so the suite stays
fast and network-free. The probe's contract is exercised against the same
minimal architecture used by `AttentionProbe`, with the addition of a final
`LayerNorm` on the inner decoder so the architecture mirrors Llama-style HF
models that LogitLens targets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_token_heatmap.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe
from llm_token_heatmap.attention_probe import AttentionProbe, AttentionProbeConfig
from llm_token_heatmap.generation import generate_with_adaptive_probe
from llm_token_heatmap.logit_lens import (
    LogitLens,
    LogitLensConfig,
    LogitLensError,
    LogitLensStats,
)
from llm_token_heatmap.plotting import (
    plot_logit_lens,
    plot_logit_lens_selected_rank,
)
from tests.conftest import FakeTokenizer
from tests.fixtures.tiny_attention_model import build_tiny_model


def _final_position_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids)
    return logits[0, -1, :]


def _capture(model, lens: LogitLens, input_ids: torch.Tensor, selected: int) -> LogitLensStats:
    with torch.no_grad():
        model(input_ids)
    return lens.capture_step(selected_token_id=selected)


def test_final_layer_matches_model_output() -> None:
    model = build_tiny_model(num_hidden_layers=3, vocab_size=32)
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    final_logits = _final_position_logits(model, input_ids)
    final_probs = torch.softmax(final_logits.float(), dim=-1)
    selected = int(final_probs.argmax().item())

    lens = LogitLens(LogitLensConfig(layers="all", top_k=32, apply_final_layernorm=True))
    lens.attach(model)
    try:
        stats = _capture(model, lens, input_ids, selected)
    finally:
        lens.detach()

    final_idx = max(stats.layers.keys())
    layer = stats.layers[final_idx]
    sorted_probs, _ = torch.sort(final_probs, descending=True)
    assert torch.allclose(layer.top_k_probs, sorted_probs[: layer.top_k_probs.numel()], atol=1e-5)


def test_earlier_layers_differ_from_final() -> None:
    model = build_tiny_model(num_hidden_layers=4, vocab_size=32)
    torch.manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))

    lens = LogitLens(LogitLensConfig(layers="all", top_k=16))
    lens.attach(model)
    try:
        stats = _capture(model, lens, input_ids, selected=0)
    finally:
        lens.detach()

    final_idx = max(stats.layers.keys())
    final_top_ids = stats.layers[final_idx].top_k_token_ids
    for layer_idx, layer in stats.layers.items():
        if layer_idx == final_idx:
            continue
        # Different layers should produce different rank-1 predictions OR
        # different probability profiles than the final layer.
        same_top1 = int(layer.top_k_token_ids[0]) == int(final_top_ids[0])
        same_dist = torch.allclose(
            layer.top_k_probs,
            stats.layers[final_idx].top_k_probs,
            atol=1e-4,
        )
        assert not (same_top1 and same_dist), (
            f"layer {layer_idx} distribution is degenerate vs final layer"
        )


def test_raw_lens_differs_from_normalized_lens() -> None:
    model = build_tiny_model(num_hidden_layers=2, vocab_size=32)
    torch.manual_seed(2)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))

    normed = LogitLens(LogitLensConfig(apply_final_layernorm=True, top_k=16))
    raw = LogitLens(LogitLensConfig(apply_final_layernorm=False, top_k=16))

    normed.attach(model)
    try:
        normed_stats = _capture(model, normed, input_ids, selected=0)
    finally:
        normed.detach()

    raw.attach(model)
    try:
        raw_stats = _capture(model, raw, input_ids, selected=0)
    finally:
        raw.detach()

    # At least one layer's top-k distribution should differ.
    any_diff = False
    for layer_idx in normed_stats.layers:
        normed_probs = normed_stats.layers[layer_idx].top_k_probs
        raw_probs = raw_stats.layers[layer_idx].top_k_probs
        if not torch.allclose(normed_probs, raw_probs, atol=1e-4):
            any_diff = True
            break
    assert any_diff, "raw lens produced identical distributions to normalized lens"


def test_selected_rank_one_at_final_layer() -> None:
    model = build_tiny_model(num_hidden_layers=3, vocab_size=32)
    torch.manual_seed(3)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    final_logits = _final_position_logits(model, input_ids)
    selected = int(final_logits.argmax().item())

    lens = LogitLens(LogitLensConfig(layers="all", top_k=8))
    lens.attach(model)
    try:
        stats = _capture(model, lens, input_ids, selected)
    finally:
        lens.detach()

    final_idx = max(stats.layers.keys())
    assert stats.layers[final_idx].selected_token_rank == 1


def test_attached_does_not_change_generation_output() -> None:
    torch.manual_seed(42)
    base_model = build_tiny_model(seed=42, num_hidden_layers=2, vocab_size=32)
    torch.manual_seed(7)
    input_ids = torch.randint(0, base_model.config.vocab_size, (1, 6))
    baseline = base_model(input_ids).detach()

    lens = LogitLens(LogitLensConfig(layers="all", top_k=4))
    lens.attach(base_model)
    try:
        with_probe = base_model(input_ids).detach()
    finally:
        lens.detach()

    assert torch.allclose(baseline, with_probe, atol=1e-6)


def test_coexists_with_attention_probe() -> None:
    model = build_tiny_model(num_hidden_layers=2, vocab_size=32)
    torch.manual_seed(11)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))

    attn = AttentionProbe(AttentionProbeConfig(capture_full_distribution=True))
    lens = LogitLens(LogitLensConfig(layers="all", top_k=4))

    attn.attach(model)
    lens.attach(model)
    try:
        with torch.no_grad():
            model(input_ids)
        attn_stats = attn.capture_step()
        lens_stats = lens.capture_step(selected_token_id=0)
    finally:
        lens.detach()
        attn.detach()

    assert set(attn_stats.layers.keys()) == {0, 1}
    assert set(lens_stats.layers.keys()) == {0, 1}
    for layer in attn_stats.layers.values():
        assert layer.attention_weights.shape == (4, 5)
    for layer in lens_stats.layers.values():
        assert layer.top_k_probs.numel() == 4


def test_generation_records_logit_lens_per_step() -> None:
    """End-to-end check: an attached LogitLens records per-step lens entries."""

    # FakeModel used by tests/conftest doesn't carry a real hidden stack;
    # build a TinyCausalLM and a FakeTokenizer matched to its vocab.
    model = build_tiny_model(num_hidden_layers=2, vocab_size=32)
    tokenizer = FakeTokenizer(vocab_size=model.config.vocab_size)

    class _ModelAdapter(torch.nn.Module):
        """Adapter so `model(input_ids=..., past_key_values=..., use_cache=True)` works."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.config = inner.config
            self.device = torch.device("cpu")
            self.model = inner.model
            self.lm_head = inner.lm_head

        def __call__(self, input_ids, past_key_values=None, use_cache=False):
            class _Out:
                pass

            out = _Out()
            out.logits = self.inner(input_ids)
            out.past_key_values = past_key_values or "cache"
            return out

    adapter = _ModelAdapter(model)
    probe = AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=1, max_k=8, mass_threshold=0.9))
    lens = LogitLens(LogitLensConfig(layers="all", top_k=4))
    lens.attach(adapter)
    try:
        _text, trace = generate_with_adaptive_probe(
            model=adapter,
            tokenizer=tokenizer,
            prompt="hi",
            probe=probe,
            max_new_tokens=2,
            temperature=1.0,
            top_p=1.0,
            sample_top_k=0,
            logit_lens=lens,
        )
    finally:
        lens.detach()

    assert len(trace) == 2
    for entry in trace:
        assert "logit_lens" in entry
        assert isinstance(entry["logit_lens"], list)
        assert len(entry["logit_lens"]) == 2  # two layers
        for layer in entry["logit_lens"]:
            assert "layer_idx" in layer
            assert "top_k_token_ids" in layer
            assert "top_k_probs" in layer
            assert "entropy" in layer
            assert "selected_token_rank" in layer
            assert "selected_token_prob" in layer


def test_plot_logit_lens_smoke(tmp_path: Path) -> None:
    model = build_tiny_model(num_hidden_layers=2, vocab_size=32)
    torch.manual_seed(5)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))

    lens = LogitLens(LogitLensConfig(layers="all", top_k=3))
    lens.attach(model)
    try:
        stats = _capture(model, lens, input_ids, selected=0)
    finally:
        lens.detach()

    trace_step = {
        "step": 0,
        "logit_lens": [
            {
                "layer_idx": layer.layer_idx,
                "top_k_token_ids": layer.top_k_token_ids,
                "top_k_probs": layer.top_k_probs,
                "top_k_logprobs": layer.top_k_logprobs,
                "entropy": layer.entropy,
                "selected_token_rank": layer.selected_token_rank,
                "selected_token_prob": layer.selected_token_prob,
            }
            for layer in stats.layers.values()
        ],
    }

    out_path = tmp_path / "logit_lens.png"
    tokenizer = FakeTokenizer(vocab_size=model.config.vocab_size)
    plot_logit_lens(trace_step, tokenizer, save_path=out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_logit_lens_selected_rank_smoke(tmp_path: Path) -> None:
    model = build_tiny_model(num_hidden_layers=3, vocab_size=32)
    lens = LogitLens(LogitLensConfig(layers="all", top_k=4))
    lens.attach(model)
    trace: list[dict] = []
    try:
        torch.manual_seed(6)
        for step in range(3):
            input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
            final_logits = _final_position_logits(model, input_ids)
            selected = int(final_logits.argmax().item())
            stats = lens.capture_step(selected_token_id=selected)
            trace.append(
                {
                    "step": step,
                    "logit_lens": [
                        {
                            "layer_idx": layer.layer_idx,
                            "top_k_token_ids": layer.top_k_token_ids,
                            "top_k_probs": layer.top_k_probs,
                            "top_k_logprobs": layer.top_k_logprobs,
                            "entropy": layer.entropy,
                            "selected_token_rank": layer.selected_token_rank,
                            "selected_token_prob": layer.selected_token_prob,
                        }
                        for layer in stats.layers.values()
                    ],
                }
            )
    finally:
        lens.detach()

    out_path = tmp_path / "logit_lens_rank.png"
    plot_logit_lens_selected_rank(trace, tokenizer=None, save_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Final row (highest layer index) should be all 1s by construction.
    final_layer_idx = max(int(layer["layer_idx"]) for layer in trace[0]["logit_lens"])
    for entry in trace:
        for layer in entry["logit_lens"]:
            if int(layer["layer_idx"]) == final_layer_idx:
                assert int(layer["selected_token_rank"]) == 1


def test_unsupported_model_raises_clear_error() -> None:
    class BareModel:
        def __init__(self) -> None:
            self.config = type("Cfg", (), {})()

    lens = LogitLens(LogitLensConfig())
    with pytest.raises(LogitLensError, match="decoder layer list"):
        lens.attach(BareModel())

    # Model that has layers but no lm_head.
    class NoHead:
        def __init__(self) -> None:
            self.layers = [torch.nn.Linear(2, 2)]

    lens2 = LogitLens(LogitLensConfig())
    with pytest.raises(LogitLensError, match="lm_head"):
        lens2.attach(NoHead())
