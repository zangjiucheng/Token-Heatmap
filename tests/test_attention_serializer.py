"""Tests for the attention serializer plumbing layer.

Covers:

* Tier 1 inline payload shape and schema validation, with and without an
  ``attention`` block present on the trace.
* Tier 2 sidecar round-trip (write → read) preserves layer arrays.
* ``SCHEMA_VERSION`` is consistent between library and backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator

from llm_token_heatmap import SCHEMA_VERSION
from llm_token_heatmap.attention_probe import (
    AttentionLayerStats,
    AttentionProbe,
    AttentionProbeConfig,
    AttentionStats,
)
from llm_token_heatmap.attention_serializer import (
    SIDECAR_SCHEMA_VERSION,
    attention_stats_to_payload,
    read_sidecar,
    write_sidecar,
)
from tests.fixtures.tiny_attention_model import build_tiny_model

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "trace.schema.json"
SIDECAR_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "attention-sidecar.schema.json"
SAMPLE_TRACE_PATH = REPO_ROOT / "web" / "frontend" / "src" / "lib" / "sample" / "trace.json"


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


def _build_synthetic_stats(*, num_heads: int = 4, head_dim: int = 4, seq_len: int = 5) -> AttentionStats:
    """Produce a small AttentionStats payload without actually running a model."""

    torch.manual_seed(0)
    layers = {}
    for layer_idx in (0, 1):
        # Full dense attention weights per head, last query row over key seq.
        raw = torch.softmax(torch.randn(num_heads, seq_len), dim=-1)
        layers[layer_idx] = AttentionLayerStats(
            layer_idx=layer_idx,
            attention_weights=raw,
            q_last=torch.randn(num_heads, head_dim),
            k_last=torch.randn(num_heads, head_dim),
            v_last=torch.randn(num_heads, head_dim),
        )
    return AttentionStats(
        layers=layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
        head_to_kv_group=list(range(num_heads)),
    )


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def test_schema_validates_with_attention_present() -> None:
    schema = _load_schema(SCHEMA_PATH)
    sample = json.loads(SAMPLE_TRACE_PATH.read_text())

    # Sanity: the bundled sample must actually carry the new fields.
    assert "attention_metadata" in sample
    assert any("attention" in step for step in sample["steps"])

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(sample)


def test_schema_validates_without_attention_present() -> None:
    """Backward compatibility: a 2.0.0 trace with no attention fields validates."""

    schema = _load_schema(SCHEMA_PATH)
    sample = json.loads(SAMPLE_TRACE_PATH.read_text())

    sample.pop("attention_metadata", None)
    for step in sample["steps"]:
        step.pop("attention", None)
        step.pop("attention_sidecar_ref", None)

    Draft202012Validator(schema).validate(sample)


def test_attention_stats_to_payload_matches_schema() -> None:
    schema = _load_schema(SCHEMA_PATH)
    stats = _build_synthetic_stats()
    payload = attention_stats_to_payload(stats, capture_full=True)

    assert payload["attention_metadata"]["captured_layers"] == [0, 1]
    assert len(payload["attention"]) == 2

    entry = payload["attention"][0]
    assert set(entry) == {
        "layer",
        "entropy",
        "self_weight",
        "bos_weight",
        "top_positions",
        "q_norm",
        "k_norm",
        "v_norm",
        "qk_alignment_angle",
        "per_head",
    }
    assert 0.0 <= entry["self_weight"] <= 1.0
    assert 0.0 <= entry["bos_weight"] <= 1.0
    assert entry["entropy"] >= 0.0

    # per_head carries one entry per head with the 7 grid scalars, and the
    # heads must be genuinely distinct (regression: the inline serializer used
    # to emit only the layer mean, so the grid broadcast it across all heads).
    per_head = entry["per_head"]
    assert len(per_head) == stats.num_attention_heads
    assert all(
        set(h) == {"entropy", "self_weight", "bos_weight", "top1_weight", "q_norm", "k_norm", "v_norm"}
        for h in per_head
    )
    assert len({h["self_weight"] for h in per_head}) > 1
    mean_self = sum(h["self_weight"] for h in per_head) / len(per_head)
    assert entry["self_weight"] == pytest.approx(mean_self)

    # Embed inside a step and validate against the trace schema.
    sample = json.loads(SAMPLE_TRACE_PATH.read_text())
    sample["attention_metadata"] = payload["attention_metadata"]
    sample["steps"][0]["attention"] = payload["attention"]
    sample["steps"][0]["attention_sidecar_ref"] = None
    Draft202012Validator(schema).validate(sample)


# --------------------------------------------------------------------------- #
# Sidecar round trip
# --------------------------------------------------------------------------- #


def test_sidecar_round_trip(tmp_path: Path) -> None:
    stats = _build_synthetic_stats()

    out = write_sidecar(stats, tmp_path / "attention.0", step=0)
    assert out.exists()
    assert out.suffix == ".npz"

    payload = read_sidecar(out)

    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    Draft202012Validator.check_schema(sidecar_schema)
    Draft202012Validator(sidecar_schema).validate(payload)

    assert payload["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert payload["step"] == 0
    assert payload["num_attention_heads"] == stats.num_attention_heads
    assert payload["num_key_value_heads"] == stats.num_key_value_heads
    assert payload["head_dim"] == stats.head_dim
    assert [layer["layer"] for layer in payload["layers"]] == sorted(stats.layers.keys())

    original = stats.layers[0].attention_weights
    roundtripped = torch.tensor(payload["layers"][0]["attention_weights"])
    assert roundtripped.shape == original.shape
    assert torch.allclose(roundtripped, original, atol=1e-5)


def test_sidecar_round_trip_without_qkv(tmp_path: Path) -> None:
    """capture_qkv=False traces still round-trip; q/k/v_last are null."""

    stats = _build_synthetic_stats()
    for layer in stats.layers.values():
        layer.q_last = None
        layer.k_last = None
        layer.v_last = None

    out = write_sidecar(stats, tmp_path / "attention.3.npz", step=3)
    payload = read_sidecar(out)

    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    Draft202012Validator(sidecar_schema).validate(payload)

    for layer_payload in payload["layers"]:
        assert layer_payload["q_last"] is None
        assert layer_payload["k_last"] is None
        assert layer_payload["v_last"] is None


# --------------------------------------------------------------------------- #
# Cross-cutting: SCHEMA_VERSION consistency
# --------------------------------------------------------------------------- #


def test_schema_version_bumped_consistently() -> None:
    assert SCHEMA_VERSION == "2.0.0"

    schema = _load_schema(SCHEMA_PATH)
    sample = json.loads(SAMPLE_TRACE_PATH.read_text())
    assert sample["schema_version"] == SCHEMA_VERSION
    assert "2.0.0" in schema["properties"]["schema_version"].get("examples", [])


def test_schema_version_matches_backend() -> None:
    """Backend SCHEMA_VERSION must track the library's. The backend package is
    not always on ``sys.path`` for the main suite, so this loads it directly
    from its source file to avoid coupling pytest config to backend layout."""

    backend_init = REPO_ROOT / "web" / "backend" / "llm_token_heatmap_api" / "__init__.py"
    if not backend_init.exists():
        pytest.skip("backend package not present in this checkout")

    namespace: dict[str, object] = {}
    exec(compile(backend_init.read_text(), str(backend_init), "exec"), namespace)
    assert namespace["SCHEMA_VERSION"] == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Probe end-to-end smoke
# --------------------------------------------------------------------------- #


def test_payload_from_real_probe_capture() -> None:
    """A capture from the synthetic tiny model serializes and validates."""

    model = build_tiny_model()
    probe = AttentionProbe(AttentionProbeConfig(capture_full_distribution=True))
    probe.attach(model)
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    with torch.no_grad():
        model(input_ids)
    stats = probe.capture_step()
    probe.detach()

    payload = attention_stats_to_payload(stats, capture_full=True)
    assert payload["attention_metadata"]["num_attention_heads"] == model.config.num_attention_heads
    assert payload["attention_metadata"]["captured_layers"]
    for entry in payload["attention"]:
        assert entry["entropy"] >= 0.0
