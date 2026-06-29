"""Schema-conformance tests for the trace JSON serializer.

Regression coverage for the bug where ``adaptive_token_trace.json`` produced
with ``--capture-logit-lens`` failed frontend schema validation because the
generation loop emitted the raw ``LogitLensLayerStats`` dataclass shape
(parallel ``top_k_token_ids`` / ``top_k_probs`` / ``top_k_logprobs`` arrays)
while the schema requires a flat ``top_k`` array of decoded candidate objects.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from llm_token_heatmap import SCHEMA_VERSION
from llm_token_heatmap.serialize.trace_payload import (
    distribution_payload,
    logit_lens_payload,
    serialize_trace_to_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "trace.schema.json"


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer.

    Implements only the surface the serializer touches: ``decode`` for token
    rendering and the ``__call__`` interface for ``prompt_tokens_payload``.
    """

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:  # noqa: ARG002
        return "".join(f"<{i}>" for i in ids)

    def __call__(self, text: str, return_tensors: object = None) -> dict:  # noqa: ARG002
        # Trivial whitespace-split tokenizer: assign each word a stable id.
        ids = [hash(w) % 50_000 for w in text.split()]
        return {"input_ids": ids}


def _stats_dict(*, selected_id: int, top_ids: list[int]) -> dict:
    """Build a probe stats dict the way the generation loop would produce it."""

    n = len(top_ids)
    probs = [1.0 / n] * n
    logprobs = [-1.0] * n
    return {
        "top_ids": top_ids,
        "top_probs": probs,
        "top_logprobs": logprobs,
        "valid_mask": [1] * n,
        "k_used": n,
        "entropy": 0.5,
        "top_mass_used": 1.0,
        "selected_prob": 1.0 / n,
        "selected_logprob": -1.0,
        "selected_rank": 1,
        "selected_ids": selected_id,
    }


def _logit_lens_layer(layer_idx: int) -> dict:
    """Build a logit-lens layer entry in the dataclass shape the loop emits."""

    return {
        "layer_idx": layer_idx,
        "top_k_token_ids": [10, 20, 30],
        "top_k_probs": [0.6, 0.3, 0.1],
        "top_k_logprobs": [-0.51, -1.2, -2.3],
        "entropy": 0.42,
        "selected_token_rank": 1,
        "selected_token_prob": 0.6,
    }


def test_logit_lens_payload_reshapes_to_schema_shape() -> None:
    tokenizer = _FakeTokenizer()
    layers = [_logit_lens_layer(0), _logit_lens_layer(3)]

    out = logit_lens_payload(layers, tokenizer)

    assert [layer["layer_idx"] for layer in out] == [0, 3]
    for layer in out:
        assert set(layer) == {
            "layer_idx",
            "top_k",
            "entropy",
            "selected_token_rank",
            "selected_token_prob",
        }
        assert len(layer["top_k"]) == 3
        for rank, candidate in enumerate(layer["top_k"], start=1):
            assert set(candidate) == {"rank", "token_id", "token", "prob", "logprob"}
            assert candidate["rank"] == rank
        assert layer["top_k"][0]["token_id"] == 10
        assert layer["top_k"][0]["token"] == "<10>"


def test_distribution_payload_clamps_unit_overshoots() -> None:
    """Float roundoff after softmax/cumsum can push probs marginally above 1.0.

    The schema requires top_mass_used / selected_prob / candidate prob in
    [0, 1]; clamping at the serializer keeps the JSON schema-conformant
    without rewriting the probe's tensor math.
    """

    tokenizer = _FakeTokenizer()
    overshoot = 1.0 + 1e-7
    stats = {
        "top_ids": [10, 20],
        "top_probs": [overshoot, 0.5],  # candidate prob just above 1.0
        "top_logprobs": [0.0, -0.69],
        "valid_mask": [1, 1],
        "k_used": 2,
        "entropy": 0.5,
        "top_mass_used": overshoot,
        "selected_prob": overshoot,
        "selected_logprob": 0.0,
        "selected_rank": 1,
        "selected_ids": 10,
    }
    out = distribution_payload(stats, tokenizer)
    assert out["top_mass_used"] == 1.0
    assert out["selected_prob"] == 1.0
    assert out["candidates"][0]["prob"] == 1.0
    # The non-overshoot candidate must pass through unchanged.
    assert out["candidates"][1]["prob"] == 0.5


def test_serialize_trace_with_logit_lens_validates_against_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    tokenizer = _FakeTokenizer()

    trace = [
        {
            "step": 0,
            "raw": _stats_dict(selected_id=10, top_ids=[10, 20, 30]),
            "processed": _stats_dict(selected_id=10, top_ids=[10, 20]),
            "logit_lens": [_logit_lens_layer(0), _logit_lens_layer(3)],
        }
    ]
    metadata = {
        "model": "fake/model",
        "prompt": "hello world",
        "generated_text": "hello world foo",
        "generated_at": "2026-05-13T00:00:00Z",
        "generation_params": {
            "max_new_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "sample_top_k": 0,
        },
        "probe_config": {"min_k": 1, "max_k": 8, "mass_threshold": 0.9},
    }

    payload = serialize_trace_to_json(
        trace=trace,
        metadata=metadata,
        attention_metadata=None,
        sidecar_refs={},
        tokenizer=tokenizer,
        prompt="hello world",
    )

    Draft202012Validator(schema).validate(payload)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "logit_lens" in payload["steps"][0]
    assert "top_k" in payload["steps"][0]["logit_lens"][0]
    # The legacy parallel-array fields must not leak through.
    leaked = {"top_k_token_ids", "top_k_probs", "top_k_logprobs"}
    assert not leaked.intersection(payload["steps"][0]["logit_lens"][0])
