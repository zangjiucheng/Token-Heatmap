"""Tests for the Tier 2 activation sidecar serializer.

Covers the four acceptance criteria for the activation sidecar serializer:

* ``test_activation_sidecar_round_trip`` — write then read; every captured
  tensor round-trips within ``torch.allclose(atol=1e-5)``.
* ``test_activation_sidecar_round_trip_without_full_tensors`` —
  ``capture_full=False`` produces no ``ActivationFullStats``; calling
  :func:`write_sidecar` with that result is a no-op (no file written).
* ``test_activation_sidecar_schema_validates_against_draft_2020`` — the
  sidecar JSON schema is itself a valid Draft 2020-12 schema.
* ``test_activation_sidecar_payload_validates_against_schema`` — the dict
  produced by :func:`read_sidecar` validates against that schema.

A real-probe smoke test ties the pieces together so a regression in the
probe→stats→serializer chain doesn't slip past the unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from jsonschema import Draft202012Validator

from llm_token_heatmap.activation_probe import (
    ActivationFullStats,
    ActivationProbe,
    ActivationProbeConfig,
)
from llm_token_heatmap.activation_serializer import (
    SIDECAR_SCHEMA_VERSION,
    read_sidecar,
    write_sidecar,
)
from tests.fixtures.tiny_attention_model import build_tiny_model

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation-sidecar.schema.json"
ACTIVATION_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation.schema.json"


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


def _build_synthetic_stats(
    *,
    num_layers: int = 3,
    hidden_dim: int = 8,
    submodules: tuple[str, ...] = ("resid_post", "mlp_out"),
) -> ActivationFullStats:
    """Synthetic ActivationFullStats avoiding any model forward pass."""

    torch.manual_seed(0)
    layer_tensors: dict[tuple[int, str], torch.Tensor] = {}
    for layer_idx in range(num_layers):
        for sub in submodules:
            layer_tensors[(layer_idx, sub)] = torch.randn(hidden_dim)
    return ActivationFullStats(
        layer_tensors=layer_tensors,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        captured_layers=list(range(num_layers)),
        captured_submodules=list(submodules),
    )


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def test_activation_sidecar_schema_validates_against_draft_2020() -> None:
    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    Draft202012Validator.check_schema(sidecar_schema)


def test_activation_sidecar_payload_validates_against_schema(tmp_path: Path) -> None:
    stats = _build_synthetic_stats()
    out = write_sidecar(stats, tmp_path / "activation.5", step=5)
    assert out is not None
    payload = read_sidecar(out)

    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    Draft202012Validator(sidecar_schema).validate(payload)


def test_activation_schema_allows_sidecar_ref_field() -> None:
    """The 1.1.0 schema bump adds the optional `activation_sidecar_ref`
    field to `ActivationStep`; producers should be able to set it without
    tripping `additionalProperties: false`."""

    schema = _load_schema(ACTIVATION_SCHEMA_PATH)
    payload = {
        "schema_version": "1.1.0",
        "activation_metadata": {
            "captured_submodules": ["resid_post"],
            "num_layers": 1,
            "hidden_dim": 4,
            "tokenizer_fingerprint": "sha256:test",
            "captured_layers": [0],
        },
        "steps": [
            {
                "step": 0,
                "token_id": 1,
                "decoded_text_offset": 0,
                "activations": [
                    {
                        "layer": 0,
                        "submodule": "resid_post",
                        "l2_norm": 1.0,
                        "mean_abs": 0.5,
                        "sparsity": 0.0,
                        "top_neurons": [{"index": 0, "value": 1.0}],
                    }
                ],
                "activation_sidecar_ref": "activations/step_0.npz",
            },
            {
                "step": 1,
                "token_id": 2,
                "decoded_text_offset": 1,
                "activations": [
                    {
                        "layer": 0,
                        "submodule": "resid_post",
                        "l2_norm": 1.0,
                        "mean_abs": 0.5,
                        "sparsity": 0.0,
                        "top_neurons": [{"index": 0, "value": 1.0}],
                    }
                ],
                "activation_sidecar_ref": None,
            },
        ],
    }
    Draft202012Validator(schema).validate(payload)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_activation_sidecar_round_trip(tmp_path: Path) -> None:
    stats = _build_synthetic_stats()

    out = write_sidecar(stats, tmp_path / "activation.0", step=0)
    assert out is not None
    assert out.exists()
    assert out.suffix == ".npz"

    payload = read_sidecar(out)

    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    Draft202012Validator(sidecar_schema).validate(payload)

    assert payload["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert payload["step"] == 0
    assert payload["num_layers"] == stats.num_layers
    assert payload["hidden_dim"] == stats.hidden_dim
    assert payload["captured_submodules"] == list(stats.captured_submodules)
    assert payload["captured_layers"] == stats.captured_layers

    for entry in payload["layers"]:
        layer_idx = entry["layer"]
        for sub, values in entry["submodule_tensors"].items():
            original = stats.layer_tensors[(layer_idx, sub)]
            roundtripped = torch.tensor(values, dtype=original.dtype)
            assert roundtripped.shape == original.shape
            assert torch.allclose(roundtripped, original, atol=1e-5)


def test_activation_sidecar_round_trip_without_full_tensors(tmp_path: Path) -> None:
    """When `capture_full=False` the probe produces no ActivationFullStats;
    `write_sidecar` called with that result is a no-op (returns None, no file)."""

    model = build_tiny_model(num_hidden_layers=2)
    probe = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out"],
            capture_full=False,
        )
    )
    probe.attach(model)
    try:
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        with torch.no_grad():
            model(input_ids)
        _ = probe.capture_step()
        full_stats = probe.last_full_stats
    finally:
        probe.detach()

    assert full_stats is None

    out = write_sidecar(full_stats, tmp_path / "activation.0", step=0)
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_activation_sidecar_appends_npz_suffix(tmp_path: Path) -> None:
    """Callers passing a stem without `.npz` get one appended."""

    stats = _build_synthetic_stats()
    out = write_sidecar(stats, tmp_path / "activation.7", step=7)
    assert out is not None
    assert out.suffix == ".npz"
    assert out.name == "activation.7.npz"


# --------------------------------------------------------------------------- #
# End-to-end smoke
# --------------------------------------------------------------------------- #


def test_activation_sidecar_from_real_probe_capture(tmp_path: Path) -> None:
    """End-to-end: ActivationProbe with `capture_full=True` produces stats a
    serializer round-trips against the schema."""

    model = build_tiny_model(num_hidden_layers=2)
    probe = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out"],
            capture_full=True,
        )
    )
    probe.attach(model)
    try:
        input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
        with torch.no_grad():
            model(input_ids)
        _ = probe.capture_step()
        full_stats = probe.last_full_stats
    finally:
        probe.detach()

    assert full_stats is not None
    assert full_stats.hidden_dim == model.config.hidden_size
    assert full_stats.num_layers == model.config.num_hidden_layers
    assert set(full_stats.captured_submodules) == {"resid_post", "mlp_out"}
    for tensor in full_stats.layer_tensors.values():
        assert tensor.shape == (full_stats.hidden_dim,)

    out = write_sidecar(full_stats, tmp_path / "activation.0", step=0)
    assert out is not None
    payload = read_sidecar(out)

    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    Draft202012Validator(sidecar_schema).validate(payload)
    assert payload["hidden_dim"] == model.config.hidden_size
    assert payload["num_layers"] == model.config.num_hidden_layers


def test_activation_sidecar_per_position_capture(tmp_path: Path) -> None:
    """`capture_along_sequence` populates `full_stats_per_position`; each
    position's stats serialize to a distinct sidecar."""

    model = build_tiny_model(num_hidden_layers=2)
    probe = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post"],
            capture_full=True,
        )
    )
    probe.attach(model)
    try:
        seq_len = 3
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
        probe.capture_along_sequence(model, input_ids)
        per_position = probe.full_stats_per_position
    finally:
        probe.detach()

    assert len(per_position) == seq_len
    sidecar_schema = _load_schema(SIDECAR_SCHEMA_PATH)
    for pos, stats in enumerate(per_position):
        out = write_sidecar(stats, tmp_path / f"activation.{pos}.npz", step=pos)
        assert out is not None
        payload = read_sidecar(out)
        Draft202012Validator(sidecar_schema).validate(payload)
        assert payload["step"] == pos
