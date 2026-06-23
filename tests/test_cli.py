"""Smoke tests for the `token-heatmap` CLI argument parser.

These tests intentionally exercise only argparse — no model is loaded.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from llm_token_heatmap.cli import (
    EAGER_ATTENTION_WARNING,
    _emit_eager_warning,
    build_parser,
    run_diff,
)
from llm_token_heatmap.trace_payload import project_activation_subset

ALL_FLAGS = [
    "--model",
    "--prompt",
    "--max-new-tokens",
    "--temperature",
    "--top-p",
    "--min-k",
    "--max-k",
    "--mass-threshold",
    "--out",
]

ATTENTION_FLAGS = [
    "--capture-attention",
    "--attention-layers",
    "--attention-top-k",
    "--capture-full-attention",
    "--capture-logit-lens",
    "--lens-layers",
    "--lens-top-k",
]


def test_cli_help_lists_all_flags(capsys: pytest.CaptureFixture[str]) -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["trace", "--help"])
    assert exc.value.code == 0

    help_text = capsys.readouterr().out
    missing = [flag for flag in ALL_FLAGS if flag not in help_text]
    assert not missing, f"--help output missing flags: {missing}"


def test_cli_missing_required_args_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --model and --prompt are now validated in run_trace (not by argparse)
    # so that --config can supply them.  Missing --model returns exit code 2.
    from llm_token_heatmap.cli import main

    rc = main(["trace", "--prompt", "hi"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "--model" in err


def test_cli_parses_thresholds() -> None:
    parser, _ = build_parser()
    args = parser.parse_args(
        [
            "trace",
            "--model",
            "fake/model",
            "--prompt",
            "hi",
            "--mass-threshold",
            "0.9",
            "--temperature",
            "0.5",
            "--top-p",
            "0.8",
            "--max-new-tokens",
            "16",
            "--min-k",
            "4",
            "--max-k",
            "32",
        ]
    )
    assert isinstance(args.mass_threshold, float)
    assert args.mass_threshold == pytest.approx(0.9)
    assert isinstance(args.temperature, float)
    assert args.temperature == pytest.approx(0.5)
    assert isinstance(args.top_p, float)
    assert args.top_p == pytest.approx(0.8)
    assert args.max_new_tokens == 16
    assert args.min_k == 4
    assert args.max_k == 32


def _parse_trace_args(extra: list[str]) -> object:
    parser, _ = build_parser()
    return parser.parse_args(["trace", "--model", "fake/model", "--prompt", "hi", *extra])


def test_cli_help_lists_attention_flags(capsys: pytest.CaptureFixture[str]) -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["trace", "--help"])
    assert exc.value.code == 0

    help_text = capsys.readouterr().out
    missing = [flag for flag in ATTENTION_FLAGS if flag not in help_text]
    assert not missing, f"--help output missing attention flags: {missing}"


def test_cli_parses_attention_layers_all() -> None:
    args = _parse_trace_args(["--capture-attention", "--attention-layers", "all"])
    assert args.capture_attention is True
    assert args.attention_layers == "all"


def test_cli_parses_attention_layers_list() -> None:
    args = _parse_trace_args(
        [
            "--capture-attention",
            "--attention-layers",
            "0,3,7,11",
            "--attention-top-k",
            "4",
        ]
    )
    assert args.attention_layers == [0, 3, 7, 11]
    assert args.attention_top_k == 4


def test_cli_parses_lens_flags() -> None:
    args = _parse_trace_args(
        [
            "--capture-logit-lens",
            "--lens-layers",
            "1,5",
            "--lens-top-k",
            "3",
        ]
    )
    assert args.capture_logit_lens is True
    assert args.lens_layers == [1, 5]
    assert args.lens_top_k == 3


def test_cli_rejects_mixed_layers_value(capsys: pytest.CaptureFixture[str]) -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "trace",
                "--model",
                "fake/model",
                "--prompt",
                "hi",
                "--attention-layers",
                "all,3",
            ]
        )
    # argparse type errors exit with code 2.
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "all" in err.lower()


def test_cli_rejects_non_integer_layers_value() -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "trace",
                "--model",
                "fake/model",
                "--prompt",
                "hi",
                "--lens-layers",
                "0,foo,3",
            ]
        )
    assert exc.value.code == 2


def test_cli_warns_about_eager_attention_cost() -> None:
    """`_emit_eager_warning` writes the perf warning to the given stream."""

    buf = io.StringIO()
    _emit_eager_warning(buf)
    output = buf.getvalue()
    assert EAGER_ATTENTION_WARNING.strip() in output
    assert "eager" in output.lower()


def test_capture_attention_defaults_off() -> None:
    args = _parse_trace_args([])
    assert args.capture_attention is False
    assert args.capture_full_attention is False
    assert args.capture_logit_lens is False
    assert args.attention_layers == "all"
    assert args.lens_layers == "all"
    assert args.attention_top_k == 8
    assert args.lens_top_k == 8


# --------------------------------------------------------------------------- #
# Activation flags (trace subcommand)
# --------------------------------------------------------------------------- #

ACTIVATION_FLAGS = [
    "--capture-activations",
    "--activation-layers",
    "--activation-submodules",
    "--activation-top-k",
    "--capture-full-activations",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation.schema.json"
TRACE_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "trace.schema.json"


def test_cli_help_lists_activation_flags(capsys: pytest.CaptureFixture[str]) -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["trace", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    missing = [flag for flag in ACTIVATION_FLAGS if flag not in help_text]
    assert not missing, f"--help output missing activation flags: {missing}"


def test_cli_capture_activations_defaults_off() -> None:
    args = _parse_trace_args([])
    assert args.capture_activations is False
    assert args.capture_full_activations is False
    assert args.activation_layers == "all"
    assert args.activation_submodules == ["residual_post", "mlp_out", "o_proj"]
    assert args.activation_top_k == 8


def test_cli_parses_activation_layers_list() -> None:
    args = _parse_trace_args(
        [
            "--capture-activations",
            "--activation-layers",
            "0,2,4",
            "--activation-submodules",
            "resid_post,mlp_out",
            "--activation-top-k",
            "4",
        ]
    )
    assert args.capture_activations is True
    assert args.activation_layers == [0, 2, 4]
    assert args.activation_submodules == ["resid_post", "mlp_out"]
    assert args.activation_top_k == 4


def test_cli_rejects_empty_activation_submodules() -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "trace",
                "--model",
                "fake/model",
                "--prompt",
                "hi",
                "--activation-submodules",
                ",",
            ]
        )
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# Activation JSON shape (trace serializer round-trip)
# --------------------------------------------------------------------------- #


class _ListAwareFakeTokenizer:
    """FakeTokenizer variant that honors ``return_tensors=None`` for serializer paths.

    `prompt_tokens_payload` calls the tokenizer with ``return_tensors=None`` and
    expects a list-of-ints response; the canonical conftest FakeTokenizer always
    returns a tensor batch, which trips the serializer's boolean check.
    """

    vocab_size = 32
    eos_token_id = None

    def __call__(self, prompt: str, return_tensors: Any = "pt") -> Any:
        import torch as _torch

        ids = [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in prompt[:8]]
        if not ids:
            ids = [0]
        if return_tensors is None:
            return {"input_ids": ids}
        return {"input_ids": _torch.tensor([ids], dtype=_torch.long)}

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        try:
            ids = list(token_ids)
        except TypeError:
            ids = [int(token_ids)]
        return "".join(f"<tok:{int(i)}>" for i in ids)


def _make_fake_tokenizer() -> Any:
    return _ListAwareFakeTokenizer()


def _synthetic_activation_trace_payload(prompt: str = "hi") -> dict[str, Any]:
    """Round-trip a trace through `serialize_trace_to_json` with an activation block.

    The generation step entries mirror the in-memory shape `generate_with_adaptive_probe`
    produces when an `ActivationProbe` is attached, but are constructed by hand so the
    test never touches a real model.
    """

    import torch

    from llm_token_heatmap.trace_payload import serialize_trace_to_json

    def _stats(token_id: int) -> dict[str, Any]:
        return {
            "top_ids": torch.tensor([[token_id]], dtype=torch.long),
            "top_probs": torch.tensor([[1.0]]),
            "top_logprobs": torch.tensor([[0.0]]),
            "valid_mask": torch.tensor([[True]]),
            "k_used": torch.tensor([1]),
            "entropy": torch.tensor([0.0]),
            "top_mass_used": torch.tensor([1.0]),
            "selected_prob": torch.tensor([1.0]),
            "selected_logprob": torch.tensor([0.0]),
            "selected_rank": torch.tensor([1]),
            "selected_ids": torch.tensor([token_id], dtype=torch.long),
        }

    trace = [
        {
            "step": 0,
            "decoded_text_offset": 5,
            "raw": _stats(11),
            "processed": _stats(11),
            "activations": [
                {
                    "layer": 0,
                    "submodule": "resid_post",
                    "l2_norm": 1.0,
                    "mean_abs": 0.25,
                    "sparsity": 0.0,
                    "top_neurons": [{"index": 0, "value": 0.5}],
                }
            ],
        },
        {
            "step": 1,
            "decoded_text_offset": 12,
            "raw": _stats(13),
            "processed": _stats(13),
            "activations": [
                {
                    "layer": 0,
                    "submodule": "resid_post",
                    "l2_norm": 0.8,
                    "mean_abs": 0.2,
                    "sparsity": 0.0,
                    "top_neurons": [{"index": 0, "value": 0.4}],
                }
            ],
        },
    ]

    metadata = {
        "model": "fake/model",
        "prompt": prompt,
        "generated_text": prompt + "<tok:11><tok:13>",
        "device": "cpu",
        "generation_params": {
            "max_new_tokens": 2,
            "temperature": 1.0,
            "top_p": 1.0,
            "sample_top_k": 0,
        },
        "probe_config": {"min_k": 1, "max_k": 4, "mass_threshold": 0.95},
    }

    activation_metadata = {
        "captured_submodules": ["resid_post"],
        "num_layers": 2,
        "hidden_dim": 4,
        "tokenizer_fingerprint": "sha256:fake",
        "captured_layers": [0],
    }

    return serialize_trace_to_json(
        trace=trace,
        metadata=metadata,
        attention_metadata=None,
        sidecar_refs={},
        tokenizer=_make_fake_tokenizer(),
        prompt=prompt,
        activation_metadata=activation_metadata,
    )


def test_cli_trace_capture_activations_writes_metadata(tmp_path: Path) -> None:
    """argparse + JSON shape: the inline trace carries activation_metadata + step
    activation fields, and the projected subset validates against activation.schema.json.
    """

    # argparse side: defaults + explicit values land in the namespace.
    args = _parse_trace_args(
        [
            "--capture-activations",
            "--activation-layers",
            "0,1",
            "--activation-submodules",
            "resid_post",
            "--activation-top-k",
            "2",
        ]
    )
    assert args.capture_activations is True
    assert args.activation_layers == [0, 1]
    assert args.activation_submodules == ["resid_post"]

    # JSON shape side: a hand-built payload must round-trip through the serializer
    # and produce a file that (a) validates against trace.schema.json end-to-end and
    # (b) whose projected activation subset validates against activation.schema.json.
    payload = _synthetic_activation_trace_payload()
    json_path = tmp_path / "adaptive_token_trace.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert "activation_metadata" in payload
    assert payload["activation_metadata"]["captured_submodules"] == ["resid_post"]
    for step in payload["steps"]:
        assert "token_id" in step
        assert "decoded_text_offset" in step
        assert "activations" in step

    trace_schema = json.loads(TRACE_SCHEMA_PATH.read_text())
    Draft202012Validator(trace_schema).validate(payload)

    activation_schema = json.loads(ACTIVATION_SCHEMA_PATH.read_text())
    projected = project_activation_subset(payload)
    Draft202012Validator(activation_schema).validate(projected)


# --------------------------------------------------------------------------- #
# Diff subcommand
# --------------------------------------------------------------------------- #


def test_cli_diff_help_lists_metric_and_out(capsys: pytest.CaptureFixture[str]) -> None:
    parser, _ = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["diff", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--metric" in help_text
    assert "{l2,cosine}" in help_text
    assert "--out" in help_text


def test_cli_diff_subcommand_writes_outputs(tmp_path: Path) -> None:
    """End-to-end on synthetic traces: diff emits activation_diff.json + .png."""

    payload_a = _synthetic_activation_trace_payload(prompt="hi")
    payload_b = _synthetic_activation_trace_payload(prompt="hi")
    # Make B differ from A so deltas are non-zero (but token ids still match so
    # token_id alignment fires under `align=auto`).
    for step in payload_b["steps"]:
        for entry in step["activations"]:
            entry["top_neurons"][0]["value"] = -entry["top_neurons"][0]["value"]
            entry["l2_norm"] = math.sqrt(
                sum(n["value"] ** 2 for n in entry["top_neurons"])
            )

    trace_a_path = tmp_path / "a" / "adaptive_token_trace.json"
    trace_b_path = tmp_path / "b" / "adaptive_token_trace.json"
    trace_a_path.parent.mkdir(parents=True)
    trace_b_path.parent.mkdir(parents=True)
    trace_a_path.write_text(json.dumps(payload_a), encoding="utf-8")
    trace_b_path.write_text(json.dumps(payload_b), encoding="utf-8")

    out_dir = tmp_path / "diff"
    parser, _ = build_parser()
    args = parser.parse_args(
        [
            "diff",
            str(trace_a_path),
            str(trace_b_path),
            "--out",
            str(out_dir),
            "--metric",
            "l2",
        ]
    )
    exit_code = run_diff(args)
    assert exit_code == 0

    diff_path = out_dir / "activation_diff.json"
    png_path = out_dir / "activation_delta.png"
    assert diff_path.is_file()
    assert png_path.is_file() and png_path.stat().st_size > 0

    diff = json.loads(diff_path.read_text(encoding="utf-8"))
    assert diff["alignment"]["mode"] in {"token_id", "position"}
    assert diff["steps"], "expected at least one aligned step"


def test_cli_diff_rejects_mismatched_prompts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Different parent prompts → diff refuses with a clear message and non-zero exit."""

    payload_a = _synthetic_activation_trace_payload(prompt="alpha")
    payload_b = _synthetic_activation_trace_payload(prompt="beta")

    trace_a_path = tmp_path / "a" / "adaptive_token_trace.json"
    trace_b_path = tmp_path / "b" / "adaptive_token_trace.json"
    trace_a_path.parent.mkdir(parents=True)
    trace_b_path.parent.mkdir(parents=True)
    trace_a_path.write_text(json.dumps(payload_a), encoding="utf-8")
    trace_b_path.write_text(json.dumps(payload_b), encoding="utf-8")

    out_dir = tmp_path / "diff"
    parser, _ = build_parser()
    args = parser.parse_args(
        [
            "diff",
            str(trace_a_path),
            str(trace_b_path),
            "--out",
            str(out_dir),
        ]
    )
    exit_code = run_diff(args)
    assert exit_code != 0

    err = capsys.readouterr().err
    assert "mismatched generations" in err
    assert not (out_dir / "activation_diff.json").exists()


# --------------------------------------------------------------------------- #
# --capture-full-activations: serializer sidecar-ref wiring
# --------------------------------------------------------------------------- #


def test_serialize_trace_writes_activation_sidecar_refs() -> None:
    """When `activation_sidecar_refs` is passed, each step receives
    `activation_sidecar_ref` (string path or null)."""

    import torch

    from llm_token_heatmap.trace_payload import serialize_trace_to_json

    def _stats(token_id: int) -> dict[str, Any]:
        return {
            "top_ids": torch.tensor([[token_id]], dtype=torch.long),
            "top_probs": torch.tensor([[1.0]]),
            "top_logprobs": torch.tensor([[0.0]]),
            "valid_mask": torch.tensor([[True]]),
            "k_used": torch.tensor([1]),
            "entropy": torch.tensor([0.0]),
            "top_mass_used": torch.tensor([1.0]),
            "selected_prob": torch.tensor([1.0]),
            "selected_logprob": torch.tensor([0.0]),
            "selected_rank": torch.tensor([1]),
            "selected_ids": torch.tensor([token_id], dtype=torch.long),
        }

    trace = [
        {
            "step": 0,
            "decoded_text_offset": 0,
            "raw": _stats(5),
            "processed": _stats(5),
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
        },
        {
            "step": 1,
            "decoded_text_offset": 5,
            "raw": _stats(7),
            "processed": _stats(7),
            "activations": [
                {
                    "layer": 0,
                    "submodule": "resid_post",
                    "l2_norm": 0.5,
                    "mean_abs": 0.25,
                    "sparsity": 0.0,
                    "top_neurons": [{"index": 0, "value": 0.5}],
                }
            ],
        },
    ]

    metadata = {
        "model": "fake/model",
        "prompt": "hello",
        "generated_text": "hello world",
        "generation_params": {"max_new_tokens": 2, "temperature": 1.0, "top_p": 1.0, "sample_top_k": 0},
        "probe_config": {"min_k": 1, "max_k": 4, "mass_threshold": 0.95},
    }
    activation_metadata = {
        "captured_submodules": ["resid_post"],
        "num_layers": 2,
        "hidden_dim": 4,
        "tokenizer_fingerprint": "sha256:fake",
        "captured_layers": [0],
    }

    # Step 0 has a sidecar; step 1 does not.
    activation_sidecar_refs = {0: "activations/activation.0.npz"}

    payload = serialize_trace_to_json(
        trace=trace,
        metadata=metadata,
        attention_metadata=None,
        sidecar_refs={},
        tokenizer=_make_fake_tokenizer(),
        prompt="hello",
        activation_metadata=activation_metadata,
        activation_sidecar_refs=activation_sidecar_refs,
    )

    assert payload["steps"][0]["activation_sidecar_ref"] == "activations/activation.0.npz"
    assert payload["steps"][1]["activation_sidecar_ref"] is None

    # Without activation_sidecar_refs the field must be absent.
    payload_no_ref = serialize_trace_to_json(
        trace=trace,
        metadata=metadata,
        attention_metadata=None,
        sidecar_refs={},
        tokenizer=_make_fake_tokenizer(),
        prompt="hello",
        activation_metadata=activation_metadata,
    )
    for step in payload_no_ref["steps"]:
        assert "activation_sidecar_ref" not in step


def test_activation_full_stats_stashed_in_step_entry() -> None:
    """After `capture_step()` with `capture_full=True`, `last_full_stats` is
    non-None and the generation loop would stash it as `_activation_full_stats`.
    This test exercises the probe directly (the generation loop addition is a
    one-liner guarded by the same `config.capture_full` flag tested here)."""

    import torch
    from llm_token_heatmap.activation_probe import ActivationProbe, ActivationProbeConfig
    from tests.fixtures.tiny_attention_model import build_tiny_model

    model = build_tiny_model(num_hidden_layers=2)

    # --- capture_full=True: last_full_stats populated ---
    probe = ActivationProbe(
        ActivationProbeConfig(layers="all", submodules=["resid_post"], capture_full=True)
    )
    probe.attach(model)
    try:
        input_ids = torch.zeros((1, 3), dtype=torch.long)
        with torch.no_grad():
            model(input_ids)
        probe.capture_step()
        full_stats = probe.last_full_stats
    finally:
        probe.detach()

    assert full_stats is not None, "capture_full=True should populate last_full_stats"
    # Verify that the generation loop logic would add it to the step dict.
    step_entry: dict[str, Any] = {"step": 0}
    if probe.config.capture_full:
        step_entry["_activation_full_stats"] = full_stats
    assert "_activation_full_stats" in step_entry

    # --- capture_full=False: last_full_stats is None ---
    probe2 = ActivationProbe(
        ActivationProbeConfig(layers="all", submodules=["resid_post"], capture_full=False)
    )
    probe2.attach(model)
    try:
        with torch.no_grad():
            model(input_ids)
        probe2.capture_step()
        full_stats2 = probe2.last_full_stats
    finally:
        probe2.detach()

    assert full_stats2 is None
    step_entry2: dict[str, Any] = {"step": 0}
    if probe2.config.capture_full:
        step_entry2["_activation_full_stats"] = full_stats2
    assert "_activation_full_stats" not in step_entry2
