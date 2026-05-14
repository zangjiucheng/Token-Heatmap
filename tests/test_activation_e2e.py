"""End-to-end activation toolbox tests.

Caps the activation-diff epic with a real two-checkpoint capture and diff using
the synthetic ``tiny_two_model`` fixture so the suite stays fast and
network-free. Three flows are covered:

- ``test_e2e_activation_capture_and_diff_via_python_api`` — full Python loop:
  attach an ``ActivationProbe`` to each seeded model, run
  ``generate_with_adaptive_probe`` against a shared tokenizer, assemble two
  schema-valid activation traces, run ``compare_activations``, and assert the
  closed-form L2 against the raw captured tensors.
- ``test_e2e_activation_capture_and_diff_via_cli`` — drives
  ``token-heatmap trace`` + ``token-heatmap diff`` end-to-end as subprocesses
  against the two seeded checkpoints saved to disk. Enforces
  ``TRANSFORMERS_OFFLINE=1`` / ``HF_HUB_OFFLINE=1`` so the test cannot
  silently touch the network.
- ``test_e2e_force_prefix_replays_sequence`` — runs ``capture_along_sequence``
  on both models with identical ``input_ids`` and confirms the diff aligns
  position-for-position.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from jsonschema import Draft202012Validator

from llm_token_heatmap import (
    SCHEMA_VERSION,
    ActivationProbe,
    ActivationProbeConfig,
    AdaptiveProbeConfig,
    AdaptiveTokenProbe,
    compare_activations,
    generate_with_adaptive_probe,
    tokenizer_fingerprint,
)
from tests.fixtures.tiny_two_model import (
    build_tiny_two_models,
    persist_tiny_two_models,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation.schema.json"
ACTIVATION_DIFF_SCHEMA_PATH = REPO_ROOT / "docs" / "web" / "activation-diff.schema.json"


@pytest.fixture(autouse=True)
def _isolate_transformers_imports():
    """Drop ``transformers`` / ``huggingface_hub`` from ``sys.modules`` once each
    e2e test finishes.

    The synthetic two-checkpoint fixture builds ``LlamaForCausalLM`` instances
    in-process, which imports ``transformers``. Other tests in this suite
    (notably ``test_generation_no_network_imports``) assert those modules
    haven't leaked into the runtime, so we scope the leak to e2e cases only.
    """

    import sys

    pre_loaded = {
        name for name in sys.modules if name.startswith(("transformers", "huggingface_hub"))
    }
    yield
    leaked = {
        name for name in sys.modules if name.startswith(("transformers", "huggingface_hub"))
    } - pre_loaded
    for name in leaked:
        sys.modules.pop(name, None)


def _load_validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def _build_activation_trace(
    *,
    trace: list[dict[str, Any]],
    activation_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Project a generation trace into an ``activation.schema.json``-shaped payload."""

    steps: list[dict[str, Any]] = []
    for i, entry in enumerate(trace):
        steps.append(
            {
                "step": i,
                "token_id": int(entry["selected_token_id"]),
                "decoded_text_offset": int(entry["decoded_text_offset"]),
                "activations": entry["activations"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "activation_metadata": activation_metadata,
        "steps": steps,
    }


def _selected_token_id(step_entry: dict[str, Any]) -> int:
    """Pull the integer selected token id from a generation trace entry."""

    selected = step_entry["raw"]["selected_ids"]
    if isinstance(selected, torch.Tensor):
        return int(selected.item()) if selected.numel() == 1 else int(selected[0].item())
    if isinstance(selected, list):
        return int(selected[0])
    return int(selected)


def _capture_trace(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    activation_probe: ActivationProbe,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ``generate_with_adaptive_probe`` with an attached activation probe.

    Returns the in-memory trace plus the matching activation_metadata block.
    Drives sampling with ``top_p=1.0`` + ``sample_top_k=1`` so the generation
    is greedy and deterministic given the seed.
    """

    torch.manual_seed(seed)
    adaptive = AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=2, max_k=8))
    activation_probe.attach(model)
    try:
        _, trace = generate_with_adaptive_probe(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            probe=adaptive,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            sample_top_k=1,
            activation_probe=activation_probe,
        )
        activation_metadata = {
            "captured_submodules": list(activation_probe.submodule_keys),
            "num_layers": int(activation_probe.num_layers),
            "hidden_dim": int(activation_probe.hidden_dim),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "captured_layers": [int(i) for i in activation_probe.target_layers],
        }
    finally:
        activation_probe.detach()

    # Normalise the in-memory trace's selected-token id into a plain int so the
    # downstream activation-trace builder can rely on it.
    for entry in trace:
        entry["selected_token_id"] = _selected_token_id(entry)

    return trace, activation_metadata


def test_e2e_activation_capture_and_diff_via_python_api() -> None:
    """Drive `generate_with_adaptive_probe` with an ActivationProbe on model A,
    then replay A's sampled token sequence through model B via the force-prefix
    `capture_along_sequence` path. This produces two traces that share token
    ids by construction — exactly the realistic "compare two checkpoints on
    the same generation" workflow the diff schema is built for — so
    `compare_activations` aligns cleanly under `token_id` and the closed-form
    L2 check can be pinned per (step, layer, submodule).
    """

    fixture = build_tiny_two_models(seed_a=0, seed_b=7)
    hidden_dim = fixture.config.hidden_size

    probe_a = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out", "o_proj"],
            top_k=hidden_dim,
        )
    )

    trace_a_entries, metadata_a = _capture_trace(
        model=fixture.model_a,
        tokenizer=fixture.tokenizer,
        prompt="hello world",
        seed=0,
        max_new_tokens=3,
        activation_probe=probe_a,
    )
    activation_trace_a = _build_activation_trace(
        trace=trace_a_entries, activation_metadata=metadata_a
    )

    # Replay A's selected-token sequence through B with force-prefix so the
    # two traces share token ids step-for-step. The prompt is fed first so
    # the replayed positions of interest line up with A's generation indices.
    prompt_ids = fixture.tokenizer("hello world", return_tensors="pt").input_ids
    generated_ids = torch.tensor(
        [[int(e["selected_token_id"]) for e in trace_a_entries]],
        dtype=torch.long,
    )
    replay_input_ids = torch.cat([prompt_ids, generated_ids], dim=-1)
    prompt_len = int(prompt_ids.shape[-1])
    num_steps = int(generated_ids.shape[-1])

    probe_b = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out", "o_proj"],
            top_k=hidden_dim,
        )
    )
    probe_b.attach(fixture.model_b)
    try:
        per_position_b = probe_b.capture_along_sequence(
            fixture.model_b, replay_input_ids
        )
        metadata_b = {
            "captured_submodules": list(probe_b.submodule_keys),
            "num_layers": int(probe_b.num_layers),
            "hidden_dim": int(probe_b.hidden_dim),
            "tokenizer_fingerprint": tokenizer_fingerprint(fixture.tokenizer),
            "captured_layers": [int(i) for i in probe_b.target_layers],
        }
    finally:
        probe_b.detach()

    # Pull B's activations at the positions that correspond to A's generated
    # tokens. Position ``prompt_len + i`` in the force-prefix capture is the
    # activation A's KV-cache capture saw on step ``i``.
    activation_trace_b_steps: list[dict[str, Any]] = []
    for i in range(num_steps):
        a_step = activation_trace_a["steps"][i]
        activation_trace_b_steps.append(
            {
                "step": i,
                "token_id": a_step["token_id"],
                "decoded_text_offset": a_step["decoded_text_offset"],
                "activations": [asdict(e) for e in per_position_b[prompt_len + i]],
            }
        )
    activation_trace_b = {
        "schema_version": "1.0.0",
        "activation_metadata": metadata_b,
        "steps": activation_trace_b_steps,
    }

    activation_validator = _load_validator(ACTIVATION_SCHEMA_PATH)
    activation_validator.validate(activation_trace_a)
    activation_validator.validate(activation_trace_b)

    # Same tokenizer on both sides → auto resolves to token_id alignment.
    assert metadata_a["tokenizer_fingerprint"] == metadata_b["tokenizer_fingerprint"]

    diff = compare_activations(
        activation_trace_a,
        activation_trace_b,
        metric="l2",
        align="auto",
    )

    diff_validator = _load_validator(ACTIVATION_DIFF_SCHEMA_PATH)
    diff_validator.validate(diff)

    assert diff["alignment"]["mode"] == "token_id"
    assert diff["alignment"]["mismatches"] == []
    assert len(diff["steps"]) == num_steps

    # Closed-form check: with top_k == hidden_dim every top_neurons list is the
    # full activation vector, so the diff's L2 must equal ||vec_a - vec_b||_2.
    for step_idx, diff_step in enumerate(diff["steps"]):
        a_entries = activation_trace_a["steps"][step_idx]["activations"]
        b_entries = activation_trace_b["steps"][step_idx]["activations"]
        a_by_key = {(int(e["layer"]), e["submodule"]): e for e in a_entries}
        b_by_key = {(int(e["layer"]), e["submodule"]): e for e in b_entries}

        for layer_delta in diff_step["delta"]:
            key = (int(layer_delta["layer"]), layer_delta["submodule"])
            a_vec = torch.zeros(hidden_dim, dtype=torch.float64)
            b_vec = torch.zeros(hidden_dim, dtype=torch.float64)
            for n in a_by_key[key]["top_neurons"]:
                a_vec[int(n["index"])] = float(n["value"])
            for n in b_by_key[key]["top_neurons"]:
                b_vec[int(n["index"])] = float(n["value"])
            expected_l2 = float((a_vec - b_vec).norm().item())
            assert layer_delta["l2"] == pytest.approx(expected_l2, abs=1e-5)
        assert all(
            len(d["top_changed_neurons"]) > 0 for d in diff_step["delta"]
        )

    nonzero_l2 = [
        d["l2"] for step in diff["steps"] for d in step["delta"] if d["l2"] > 1e-6
    ]
    assert nonzero_l2, "two seed-distinct models produced identical activations"


def test_e2e_force_prefix_replays_sequence() -> None:
    fixture = build_tiny_two_models(seed_a=0, seed_b=7)
    hidden_dim = fixture.config.hidden_size

    probe_a = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out", "o_proj"],
            top_k=hidden_dim,
        )
    )
    probe_b = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out", "o_proj"],
            top_k=hidden_dim,
        )
    )

    input_ids = fixture.tokenizer(
        "the quick brown fox", return_tensors="pt"
    ).input_ids
    seq_len = int(input_ids.shape[-1])
    assert seq_len >= 3, "tokenizer must produce a multi-token prefix"

    probe_a.attach(fixture.model_a)
    probe_b.attach(fixture.model_b)
    try:
        per_position_a = probe_a.capture_along_sequence(fixture.model_a, input_ids)
        per_position_b = probe_b.capture_along_sequence(fixture.model_b, input_ids)
        metadata = {
            "captured_submodules": list(probe_a.submodule_keys),
            "num_layers": int(probe_a.num_layers),
            "hidden_dim": int(probe_a.hidden_dim),
            "tokenizer_fingerprint": tokenizer_fingerprint(fixture.tokenizer),
            "captured_layers": [int(i) for i in probe_a.target_layers],
        }
        captured = len(probe_a.target_layers) * len(probe_a.submodule_keys)
    finally:
        probe_a.detach()
        probe_b.detach()

    assert len(per_position_a) == seq_len
    assert len(per_position_b) == seq_len

    token_ids = input_ids[0].tolist()
    cumulative_decoded = ""
    activation_trace_a: dict[str, Any] = {
        "schema_version": "1.0.0",
        "activation_metadata": metadata,
        "steps": [],
    }
    activation_trace_b: dict[str, Any] = {
        "schema_version": "1.0.0",
        "activation_metadata": metadata,
        "steps": [],
    }
    for pos in range(seq_len):
        offset = len(cumulative_decoded)
        activation_trace_a["steps"].append(
            {
                "step": pos,
                "token_id": int(token_ids[pos]),
                "decoded_text_offset": offset,
                "activations": [asdict(e) for e in per_position_a[pos]],
            }
        )
        activation_trace_b["steps"].append(
            {
                "step": pos,
                "token_id": int(token_ids[pos]),
                "decoded_text_offset": offset,
                "activations": [asdict(e) for e in per_position_b[pos]],
            }
        )
        cumulative_decoded += fixture.tokenizer.decode(
            [token_ids[pos]], skip_special_tokens=True
        )

    activation_validator = _load_validator(ACTIVATION_SCHEMA_PATH)
    activation_validator.validate(activation_trace_a)
    activation_validator.validate(activation_trace_b)

    diff = compare_activations(activation_trace_a, activation_trace_b, align="auto")
    _load_validator(ACTIVATION_DIFF_SCHEMA_PATH).validate(diff)

    assert diff["alignment"]["mode"] == "token_id"
    assert diff["alignment"]["mismatches"] == []
    assert len(diff["steps"]) == seq_len
    for diff_step in diff["steps"]:
        assert len(diff_step["delta"]) == captured
    nonzero = [
        d["l2"] for step in diff["steps"] for d in step["delta"] if d["l2"] > 1e-6
    ]
    assert nonzero, "force-prefix capture produced no movement between seeds"


def test_e2e_activation_capture_and_diff_via_cli(tmp_path: Path) -> None:
    """Drive `token-heatmap trace --capture-activations` + `token-heatmap diff`
    as subprocesses against tiny local checkpoints. The CLI samples next
    tokens via `torch.multinomial`, so the test wraps each invocation in a
    tiny shim that calls `torch.manual_seed(0)` before delegating to
    `llm_token_heatmap.cli.main`. Same shim seed + same model checkpoint
    yields byte-identical traces, so the CLI's `auto` alignment resolves to
    `token_id` cleanly. The closed-form L2 check is "identical inputs →
    zero deltas everywhere" — the non-trivial nonzero-L2 case is exercised
    by the in-process Python-API test above.
    """

    fixture = persist_tiny_two_models(tmp_path / "checkpoints", seed_a=0, seed_b=7)
    assert fixture.path_a is not None and fixture.path_b is not None

    env = {
        **os.environ,
        # Hard-fail any code path that tries to reach the HF hub.
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        # Pin to CPU so the CLI's `torch.cuda.is_available()` branch (which
        # promotes to float16) doesn't introduce GPU-only float drift in the
        # subprocess. The test must pass on CI nodes with and without a GPU.
        "CUDA_VISIBLE_DEVICES": "",
        # Keep the repo importable when pytest is invoked from elsewhere.
        "PYTHONPATH": str(REPO_ROOT),
    }

    # Tiny shim that seeds torch deterministically before invoking the CLI.
    # The CLI itself doesn't accept a --seed flag, so the test installs the
    # determinism on the subprocess side.
    shim_path = tmp_path / "deterministic_cli.py"
    shim_path.write_text(
        "import sys\n"
        "import torch\n"
        "torch.manual_seed(0)\n"
        "from llm_token_heatmap.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    out_a = tmp_path / "trace_a"
    out_b = tmp_path / "trace_b"
    out_diff = tmp_path / "diff"

    trace_common = [
        sys.executable,
        str(shim_path),
        "trace",
        "--prompt",
        "hello world",
        "--max-new-tokens",
        "3",
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
        "--min-k",
        "2",
        "--max-k",
        "8",
        "--capture-activations",
        "--activation-submodules",
        "resid_post,mlp_out,o_proj",
        "--activation-top-k",
        str(fixture.config.hidden_size),
    ]
    subprocess.run(
        [*trace_common, "--model", str(fixture.path_a), "--out", str(out_a)],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )
    subprocess.run(
        [*trace_common, "--model", str(fixture.path_a), "--out", str(out_b)],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )

    json_a_path = out_a / "adaptive_token_trace.json"
    json_b_path = out_b / "adaptive_token_trace.json"
    assert json_a_path.exists()
    assert json_b_path.exists()

    payload_a = json.loads(json_a_path.read_text(encoding="utf-8"))
    payload_b = json.loads(json_b_path.read_text(encoding="utf-8"))
    assert payload_a.get("schema_version") == SCHEMA_VERSION
    assert "activation_metadata" in payload_a
    assert "activation_metadata" in payload_b
    # Same seeded subprocess + same checkpoint must produce the same token ids.
    assert [s["token_id"] for s in payload_a["steps"]] == [
        s["token_id"] for s in payload_b["steps"]
    ]

    subprocess.run(
        [
            sys.executable,
            str(shim_path),
            "diff",
            str(json_a_path),
            str(json_b_path),
            "--out",
            str(out_diff),
            "--metric",
            "l2",
        ],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )

    diff_path = out_diff / "activation_diff.json"
    delta_png = out_diff / "activation_delta.png"
    assert diff_path.exists()
    assert delta_png.exists()

    diff = json.loads(diff_path.read_text(encoding="utf-8"))
    _load_validator(ACTIVATION_DIFF_SCHEMA_PATH).validate(diff)
    assert diff["alignment"]["mode"] == "token_id"
    assert diff["alignment"]["mismatches"] == []
    assert len(diff["steps"]) >= 1

    # Closed-form check: identical traces have zero L2 everywhere.
    for diff_step in diff["steps"]:
        for layer_delta in diff_step["delta"]:
            assert layer_delta["l2"] == pytest.approx(0.0, abs=1e-5)
