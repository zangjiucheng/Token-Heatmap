"""Tests for `compare_activations`.

Synthetic activation-trace dicts are built directly here (no model, no
network) so the comparator can be pinned to closed-form numerics. Each test
uses ``top_k == hidden_dim`` so the sparse ``top_neurons`` reconstruction is
exact and L2 / cosine values match to 1e-6.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from llm_token_heatmap.diff import DIFF_SCHEMA_VERSION, compare_activations

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_DIFF_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation-diff.schema.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _entry(layer: int, submodule: str, values: list[float]) -> dict[str, Any]:
    """Build a schema-shaped `ActivationLayerEntry` from a dense value list.

    `top_neurons` carries every neuron (one per index) so the comparator
    reconstructs the full vector — required for closed-form L2 / cosine.
    """

    l2 = math.sqrt(sum(v * v for v in values))
    mean_abs = sum(abs(v) for v in values) / len(values) if values else 0.0
    sparsity = (
        sum(1 for v in values if abs(v) < 1e-12) / len(values) if values else 1.0
    )
    return {
        "layer": layer,
        "submodule": submodule,
        "l2_norm": l2,
        "mean_abs": mean_abs,
        "sparsity": sparsity,
        "top_neurons": [
            {"index": idx, "value": float(v)} for idx, v in enumerate(values)
        ],
    }


def _trace(
    *,
    fingerprint: str,
    steps: list[tuple[int, int, list[tuple[int, str, list[float]]]]],
    hidden_dim: int = 4,
    num_layers: int = 2,
    captured_submodules: tuple[str, ...] = ("resid_post",),
) -> dict[str, Any]:
    """Build a schema-shaped activation trace.

    `steps` is a list of `(token_id, decoded_text_offset, layer_entries)`
    where `layer_entries` is `(layer, submodule, dense_values)`.
    """

    return {
        "schema_version": "1.0.0",
        "activation_metadata": {
            "captured_submodules": list(captured_submodules),
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "tokenizer_fingerprint": fingerprint,
        },
        "steps": [
            {
                "step": i,
                "token_id": token_id,
                "decoded_text_offset": offset,
                "activations": [
                    _entry(layer, submodule, values)
                    for layer, submodule, values in entries
                ],
            }
            for i, (token_id, offset, entries) in enumerate(steps)
        ],
    }


def _make_baseline_trace(fingerprint: str = "sha256:qwen-0.5b") -> dict[str, Any]:
    """Two-step, two-layer baseline trace with known activation values."""

    return _trace(
        fingerprint=fingerprint,
        hidden_dim=4,
        num_layers=2,
        captured_submodules=("resid_post",),
        steps=[
            (
                42,
                0,
                [
                    (0, "resid_post", [1.0, -2.0, 0.5, 0.25]),
                    (1, "resid_post", [0.0, 3.0, -1.0, 0.0]),
                ],
            ),
            (
                51,
                3,
                [
                    (0, "resid_post", [0.5, 0.5, 0.5, 0.5]),
                    (1, "resid_post", [-1.0, 1.0, -2.0, 2.0]),
                ],
            ),
        ],
    )


def _scaled_trace(trace: dict[str, Any], factor: float) -> dict[str, Any]:
    """Return a deep copy of `trace` with every activation scaled by `factor`."""

    scaled = copy.deepcopy(trace)
    for step in scaled["steps"]:
        for entry in step["activations"]:
            entry["l2_norm"] = entry["l2_norm"] * abs(factor)
            entry["mean_abs"] = entry["mean_abs"] * abs(factor)
            for neuron in entry["top_neurons"]:
                neuron["value"] = neuron["value"] * factor
    return scaled


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_compare_activations_self_diff_is_zero() -> None:
    """Comparing a trace to itself yields l2 == 0 and cosine == 1.0 everywhere."""

    trace = _make_baseline_trace()
    diff = compare_activations(trace, trace)

    assert diff["schema_version"] == DIFF_SCHEMA_VERSION
    assert diff["alignment"]["mode"] == "token_id"
    assert diff["alignment"]["mismatches"] == []
    assert len(diff["steps"]) == len(trace["steps"])

    for step in diff["steps"]:
        assert step["delta"], "every aligned step must carry at least one layer-delta"
        for delta in step["delta"]:
            assert delta["l2"] == pytest.approx(0.0, abs=1e-6)
            assert delta["cosine"] == pytest.approx(1.0, abs=1e-6)
            for changed in delta["top_changed_neurons"]:
                assert changed["delta"] == pytest.approx(0.0, abs=1e-6)


def test_compare_activations_scaled_trace_l2_matches_closed_form() -> None:
    """trace_b = 2 * trace_a → L2(a - b) = ||a|| and cosine(a, b) = 1.0."""

    trace_a = _make_baseline_trace()
    trace_b = _scaled_trace(trace_a, factor=2.0)

    diff = compare_activations(trace_a, trace_b)

    # Build a lookup of expected ||a|| values per (step, layer, submodule).
    expected_l2: dict[tuple[int, int, str], float] = {}
    for step_idx, step in enumerate(trace_a["steps"]):
        for entry in step["activations"]:
            expected_l2[(step_idx, int(entry["layer"]), str(entry["submodule"]))] = float(
                entry["l2_norm"]
            )

    assert diff["alignment"]["mode"] == "token_id"
    assert diff["alignment"]["mismatches"] == []
    for diff_step in diff["steps"]:
        step_idx = int(diff_step["step"])
        for delta in diff_step["delta"]:
            key = (step_idx, int(delta["layer"]), str(delta["submodule"]))
            assert delta["l2"] == pytest.approx(expected_l2[key], abs=1e-6)
            assert delta["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_compare_activations_align_token_id_vs_position_switch() -> None:
    """`auto` resolves to `token_id` for matching fingerprints, else `position`."""

    trace_a = _make_baseline_trace(fingerprint="sha256:qwen-0.5b")
    trace_b_same = _make_baseline_trace(fingerprint="sha256:qwen-0.5b")
    trace_b_diff = _make_baseline_trace(fingerprint="sha256:phi-2")

    diff_same = compare_activations(trace_a, trace_b_same, align="auto")
    diff_diff = compare_activations(trace_a, trace_b_diff, align="auto")

    assert diff_same["alignment"]["mode"] == "token_id"
    assert diff_diff["alignment"]["mode"] == "position"

    # Explicit modes are honored regardless of fingerprints.
    forced_position = compare_activations(trace_a, trace_b_same, align="position")
    forced_token_id = compare_activations(trace_a, trace_b_diff, align="token_id")
    assert forced_position["alignment"]["mode"] == "position"
    assert forced_token_id["alignment"]["mode"] == "token_id"


def test_compare_activations_cross_tokenizer_aligns_by_offset() -> None:
    """Same decoded text under different tokenizers aligns cleanly by offset."""

    trace_a = _trace(
        fingerprint="sha256:qwen-0.5b",
        hidden_dim=4,
        num_layers=1,
        steps=[
            (10, 0, [(0, "resid_post", [1.0, 0.0, 0.0, 0.0])]),
            (11, 3, [(0, "resid_post", [0.0, 1.0, 0.0, 0.0])]),
            (12, 6, [(0, "resid_post", [0.0, 0.0, 1.0, 0.0])]),
        ],
    )
    # Different tokenizer, different token_ids, same decoded_text_offsets.
    trace_b = _trace(
        fingerprint="sha256:phi-2",
        hidden_dim=4,
        num_layers=1,
        steps=[
            (777, 0, [(0, "resid_post", [1.0, 0.0, 0.0, 0.0])]),
            (888, 3, [(0, "resid_post", [0.0, 1.0, 0.0, 0.0])]),
            (999, 6, [(0, "resid_post", [0.0, 0.0, 1.0, 0.0])]),
        ],
    )

    diff = compare_activations(trace_a, trace_b)

    assert diff["alignment"]["mode"] == "position"
    assert diff["alignment"]["mismatches"] == []
    assert len(diff["steps"]) == 3
    for i, step in enumerate(diff["steps"]):
        assert step["step"] == i
        assert step["decoded_text_offset_a"] == step["decoded_text_offset_b"]
        # token_ids legitimately differ across tokenizers.
        assert step["token_id_a"] != step["token_id_b"]
        for delta in step["delta"]:
            assert delta["l2"] == pytest.approx(0.0, abs=1e-6)
            assert delta["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_compare_activations_output_validates_against_diff_schema() -> None:
    """The emitted payload validates against `docs/web/activation-diff.schema.json`."""

    schema = json.loads(ACTIVATION_DIFF_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    trace_a = _make_baseline_trace(fingerprint="sha256:qwen-0.5b")
    trace_b = _scaled_trace(
        _make_baseline_trace(fingerprint="sha256:phi-2"), factor=0.5
    )

    diff = compare_activations(trace_a, trace_b, align="auto")
    validator.validate(diff)
