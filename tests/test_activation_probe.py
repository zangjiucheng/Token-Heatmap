"""Tests for `ActivationProbe`.

All tests use the synthetic `tiny_attention_model` fixture so the suite stays
fast and network-free. The probe's contract is exercised against attach /
detach lifecycle, generation-loop integration, the force-prefix capture path,
hand-computable summary stats on a zeroed model, and dict-shape conformance
to `docs/web/activation.schema.json`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from llm_token_heatmap.activation_probe import (
    ActivationProbe,
    ActivationProbeConfig,
)
from llm_token_heatmap.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe
from llm_token_heatmap.generation import generate_with_adaptive_probe
from tests.fixtures.tiny_attention_model import TinyCausalLM, build_tiny_model

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation.schema.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _GenOutput:
    """Minimal stand-in for HF `CausalLMOutput` with past_key_values support."""

    def __init__(self, logits: torch.Tensor, past_key_values: Any) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _HybridCausalLM(nn.Module):
    """Wraps a `TinyCausalLM` with the past-key-values interface
    `generate_with_adaptive_probe` expects.

    TinyCausalLM has no KV cache; this wrapper maintains the running prefix
    manually and re-runs the full sequence each step. That is fine for the
    integration test: the probe's hooks fire on the full prefix, and the
    last-position reduction still yields the new token's activation.
    """

    def __init__(self, tiny: TinyCausalLM) -> None:
        super().__init__()
        self._tiny = tiny
        self.config = tiny.config
        # Expose `.model` so `_resolve_decoder_layers` can find the layer list.
        self.model = tiny.model
        self.device = torch.device("cpu")
        self._prefix: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> _GenOutput:
        if past_key_values is None:
            self._prefix = input_ids
        else:
            assert self._prefix is not None
            self._prefix = torch.cat([self._prefix, input_ids], dim=-1)
        logits = self._tiny(self._prefix)
        return _GenOutput(logits=logits, past_key_values=("step",))


def _zero_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_activation_probe_attach_detach_idempotent() -> None:
    model = build_tiny_model(num_hidden_layers=3)
    probe = ActivationProbe(
        ActivationProbeConfig(layers="all", submodules=["resid_post", "mlp_out"])
    )

    assert probe.is_attached is False
    probe.attach(model)
    assert probe.is_attached is True

    handles_after_first = list(probe._handles)
    # Exactly one hook per (layer, submodule): 3 layers * 2 submodules == 6.
    assert len(handles_after_first) == 3 * 2

    probe.attach(model)  # second attach is a no-op
    assert probe._handles == handles_after_first
    assert probe.is_attached is True

    probe.detach()
    assert probe.is_attached is False
    assert probe._handles == []

    probe.detach()  # second detach is a no-op
    assert probe.is_attached is False
    assert probe._handles == []


def test_activation_probe_captures_each_step_during_generation(fake_tokenizer) -> None:
    tiny = build_tiny_model(num_hidden_layers=2, vocab_size=fake_tokenizer.vocab_size)
    model = _HybridCausalLM(tiny)

    activation_probe = ActivationProbe(
        ActivationProbeConfig(layers="all", submodules=["resid_post", "mlp_out"], top_k=4)
    )
    adaptive = AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=4, max_k=8))

    activation_probe.attach(model)
    try:
        records_per_step = len(activation_probe.target_layers) * len(
            activation_probe.submodule_keys
        )
        _, trace = generate_with_adaptive_probe(
            model,
            fake_tokenizer,
            prompt="hi",
            probe=adaptive,
            max_new_tokens=4,
            temperature=1.0,
            top_p=1.0,
            sample_top_k=1,
            activation_probe=activation_probe,
        )
    finally:
        activation_probe.detach()

    assert len(trace) == 4
    assert records_per_step == 2 * 2
    for entry in trace:
        assert "activations" in entry
        assert len(entry["activations"]) == records_per_step


def test_activation_probe_force_prefix_matches_sequence_length() -> None:
    model = build_tiny_model(num_hidden_layers=2)
    probe = ActivationProbe(
        ActivationProbeConfig(layers="all", submodules=["resid_post", "mlp_out"], top_k=4)
    )
    probe.attach(model)
    try:
        seq_len = 5
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
        per_position = probe.capture_along_sequence(model, input_ids)
        records_per_position = len(probe.target_layers) * len(probe.submodule_keys)
    finally:
        probe.detach()

    assert len(per_position) == seq_len
    for position_entries in per_position:
        assert len(position_entries) == records_per_position


def test_activation_probe_summary_stats_on_known_input() -> None:
    """Zeroed model + zeroed input → activations are exactly zero everywhere,
    so the hand-computed summary stats are L2=0, mean_abs=0, sparsity=1, and
    every top-neuron value is 0."""

    model = build_tiny_model(num_hidden_layers=2, num_attention_heads=4, head_dim=4)
    _zero_parameters(model)

    config = ActivationProbeConfig(
        layers="all",
        submodules=["resid_post", "mlp_out", "o_proj"],
        top_k=4,
    )
    probe = ActivationProbe(config)
    probe.attach(model)
    try:
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        with torch.no_grad():
            model(input_ids)
        entries = probe.capture_step()
    finally:
        probe.detach()

    # 2 layers * 3 submodules
    assert len(entries) == 6
    for entry in entries:
        assert entry.l2_norm == pytest.approx(0.0, abs=1e-5)
        assert entry.mean_abs == pytest.approx(0.0, abs=1e-5)
        assert entry.sparsity == pytest.approx(1.0, abs=1e-5)
        assert len(entry.top_neurons) == config.top_k
        for neuron in entry.top_neurons:
            assert neuron.value == pytest.approx(0.0, abs=1e-5)


def test_activation_probe_summary_stats_field_names_match_schema() -> None:
    schema = json.loads(ACTIVATION_SCHEMA_PATH.read_text())
    layer_entry_def = schema["$defs"]["ActivationLayerEntry"]
    top_neuron_def = schema["$defs"]["TopNeuron"]
    # `additionalProperties: false` + no optional fields → required IS the
    # exact key set every captured dict must produce.
    expected_entry_keys = set(layer_entry_def["required"])
    expected_neuron_keys = set(top_neuron_def["required"])

    model = build_tiny_model(num_hidden_layers=2)
    probe = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out", "o_proj"],
            top_k=3,
        )
    )
    probe.attach(model)
    try:
        input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
        with torch.no_grad():
            model(input_ids)
        entries = probe.capture_step()
    finally:
        probe.detach()

    assert entries, "probe produced no entries"
    for entry in entries:
        as_dict = asdict(entry)
        assert set(as_dict.keys()) == expected_entry_keys
        assert as_dict["top_neurons"], "top_neurons should be populated"
        for neuron in as_dict["top_neurons"]:
            assert set(neuron.keys()) == expected_neuron_keys
