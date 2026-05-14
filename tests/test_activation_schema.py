"""Schema-conformance tests for the activation trace and diff schemas.

The activation toolbox writes two new on-disk shapes alongside the canonical
trace.json: an activation companion (``docs/web/activation.schema.json``) and
a diff payload (``docs/web/activation-diff.schema.json``). Both must:

1. Be themselves valid Draft 2020-12 JSON Schemas.
2. Accept the minimal example a producer is expected to emit.

These tests guard the on-disk contract that every downstream consumer
(``compare_activations``, the FastAPI schema endpoints, and the Activations tab
UI) is built against.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation.schema.json"
ACTIVATION_DIFF_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation-diff.schema.json"


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


def _minimal_activation_payload() -> dict:
    """Single-step activation companion payload covering every required field."""

    return {
        "schema_version": "1.0.0",
        "activation_metadata": {
            "captured_submodules": ["resid_pre", "resid_post", "mlp.down_proj"],
            "num_layers": 4,
            "hidden_dim": 8,
            "tokenizer_fingerprint": "sha256:qwen-0.5b",
            "captured_layers": [0, 1, 2, 3],
        },
        "steps": [
            {
                "step": 0,
                "token_id": 42,
                "decoded_text_offset": 0,
                "activations": [
                    {
                        "layer": 0,
                        "submodule": "resid_pre",
                        "l2_norm": 1.25,
                        "mean_abs": 0.4,
                        "sparsity": 0.125,
                        "top_neurons": [
                            {"index": 3, "value": 0.95},
                            {"index": 7, "value": -0.72},
                        ],
                    }
                ],
            }
        ],
    }


def _minimal_diff_payload() -> dict:
    """Single-step diff payload covering every required field, including a mismatch."""

    return {
        "schema_version": "1.0.0",
        "alignment": {
            "mode": "auto",
            "tokenizer_a_fingerprint": "sha256:qwen-0.5b",
            "tokenizer_b_fingerprint": "sha256:phi-2",
            "mismatches": [
                {
                    "step_a": 3,
                    "step_b": None,
                    "reason": "trailing_steps_in_a",
                }
            ],
        },
        "steps": [
            {
                "step": 0,
                "token_id_a": 42,
                "token_id_b": 51,
                "decoded_text_offset_a": 0,
                "decoded_text_offset_b": 0,
                "delta": [
                    {
                        "layer": 0,
                        "submodule": "resid_post",
                        "l2": 0.31,
                        "cosine": 0.97,
                        "top_changed_neurons": [
                            {"index": 4, "delta": 0.42},
                            {"index": 2, "delta": -0.18},
                        ],
                    }
                ],
            }
        ],
    }


def test_activation_schema_is_valid_draft_2020() -> None:
    schema = _load_schema(ACTIVATION_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)


def test_activation_diff_schema_is_valid_draft_2020() -> None:
    schema = _load_schema(ACTIVATION_DIFF_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)


def test_activation_schema_examples_validate() -> None:
    activation_schema = _load_schema(ACTIVATION_SCHEMA_PATH)
    Draft202012Validator(activation_schema).validate(_minimal_activation_payload())

    diff_schema = _load_schema(ACTIVATION_DIFF_SCHEMA_PATH)
    Draft202012Validator(diff_schema).validate(_minimal_diff_payload())
