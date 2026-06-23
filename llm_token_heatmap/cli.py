"""Command-line interface for llm-token-heatmap.

Provides a `token-heatmap` executable with a `trace` sub-command that wraps the
library to run an adaptive token probe on a HuggingFace causal LM and write a
trace CSV plus three plots to an output directory. Optionally attaches an
``AttentionProbe`` and/or a ``LogitLens`` for richer captures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_token_heatmap.trace_payload import (
    serialize_trace_to_json as _serialize_trace_to_json,
)

EAGER_ATTENTION_WARNING = (
    "[token-heatmap] WARNING: --capture-attention forces eager attention; "
    "generation will be significantly slower than the SDPA / FlashAttention default."
)


DEFAULT_ACTIVATION_SUBMODULES = ("residual_post", "mlp_out", "o_proj")


def _parse_submodules_spec(value: str) -> list[str]:
    """Parse the ``--activation-submodules`` value into a list of submodule keys."""

    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError(
            "submodules spec must be a comma-separated list of submodule keys; got empty value."
        )
    pieces = [piece.strip() for piece in text.split(",")]
    if any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError(
            f"submodules spec contains an empty entry: {value!r}."
        )
    return pieces


def _parse_layers_spec(value: str) -> str | list[int]:
    """Parse the ``--attention-layers`` / ``--lens-layers`` value.

    Accepts ``"all"`` or a comma-separated list of non-negative integers. Mixing
    the keyword with integers (e.g. ``"all,3"``) is rejected so the user sees a
    clean error rather than a silent partial capture.
    """

    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError(
            "layers spec must be 'all' or a comma-separated list of integers; got empty value."
        )
    if text == "all":
        return "all"

    pieces = [piece.strip() for piece in text.split(",")]
    if any(piece == "all" for piece in pieces):
        raise argparse.ArgumentTypeError(
            f"layers spec mixes 'all' with integers: {value!r}. "
            "Use either 'all' or a comma-separated list of integers, not both."
        )
    indices: list[int] = []
    for piece in pieces:
        if not piece:
            raise argparse.ArgumentTypeError(f"layers spec contains an empty entry: {value!r}.")
        try:
            idx = int(piece)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"layers spec entry {piece!r} is not 'all' or an integer."
            ) from exc
        if idx < 0:
            raise argparse.ArgumentTypeError(f"layers spec entry {piece!r} must be non-negative.")
        indices.append(idx)
    return indices


def _load_yaml_config(path: Path) -> dict:
    """Load a YAML config file and return a dict of argparse-dest → value.

    Requires PyYAML (``pip install pyyaml``).  Converts typed values so that
    argparse ``set_defaults`` receives the same Python types it would from the
    command line (e.g. Path for ``out``, list[int] for layer specs).
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "error: --config requires PyYAML. Install it with: pip install pyyaml"
        ) from exc

    try:
        raw: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"error: failed to read config file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SystemExit(f"error: config file {path} must be a YAML mapping, got {type(raw).__name__}")

    out: dict = {}
    for key, val in raw.items():
        if key == "out":
            out[key] = Path(val)
        elif key in ("attention_layers", "lens_layers", "activation_layers"):
            out[key] = _parse_layers_spec(str(val))
        elif key == "activation_submodules":
            out[key] = _parse_submodules_spec(str(val)) if isinstance(val, str) else list(val)
        else:
            out[key] = val
    return out


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the top-level argument parser for the `token-heatmap` CLI.

    Returns ``(parser, trace_subparser)`` so callers can apply YAML defaults
    directly onto ``trace_subparser`` before the final ``parse_args`` call.
    """
    parser = argparse.ArgumentParser(
        prog="token-heatmap",
        description=(
            "Analyze LLM inference-time token probability distributions "
            "with adaptive top-k tracing and heatmap visualization."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace_parser = subparsers.add_parser(
        "trace",
        help="Run the adaptive token probe on a model and write trace + plots.",
        description=(
            "Generate text with the given model and write generated.txt, "
            "adaptive_token_trace.csv, adaptive_token_trace.json, "
            "adaptive_heatmap.png, entropy.png, and selected_probability.png "
            "into the output directory."
        ),
    )
    trace_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "YAML config file. All CLI flags override config file values. "
            "Requires PyYAML (pip install pyyaml)."
        ),
    )
    trace_parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model id or local path (e.g. Qwen/Qwen2.5-0.5B-Instruct).",
    )
    trace_parser.add_argument(
        "--prompt",
        default=None,
        help="Input prompt string for generation.",
    )
    trace_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate (default: 32).",
    )
    trace_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    trace_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling cutoff (default: 0.95).",
    )
    trace_parser.add_argument(
        "--min-k",
        type=int,
        default=8,
        help="Minimum adaptive top-k (default: 8).",
    )
    trace_parser.add_argument(
        "--max-k",
        type=int,
        default=64,
        help="Maximum adaptive top-k (default: 64).",
    )
    trace_parser.add_argument(
        "--mass-threshold",
        type=float,
        default=0.95,
        help="Cumulative probability mass target for adaptive top-k (default: 0.95).",
    )
    trace_parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory (created if missing; default: outputs/).",
    )
    trace_parser.add_argument(
        "--capture-attention",
        action="store_true",
        help=(
            "Attach an AttentionProbe to capture per-layer attention weights "
            "and Q/K/V vectors. Forces eager attention (slow). Off by default."
        ),
    )
    trace_parser.add_argument(
        "--attention-layers",
        type=_parse_layers_spec,
        default="all",
        help=(
            "Which decoder layers the AttentionProbe captures: 'all' (default) "
            "or a comma-separated list of indices, e.g. '0,3,7,11'."
        ),
    )
    trace_parser.add_argument(
        "--attention-top-k",
        type=int,
        default=8,
        help="Top-k attended positions kept inline per head (default: 8).",
    )
    trace_parser.add_argument(
        "--capture-full-attention",
        action="store_true",
        help=(
            "When set, write a per-step sidecar attention.<step>.npz under "
            "<out>/attention/ containing the full attention distribution and "
            "Q/K/V tensors. Requires --capture-attention."
        ),
    )
    trace_parser.add_argument(
        "--capture-logit-lens",
        action="store_true",
        help=(
            "Attach a LogitLens probe to capture per-layer next-token predictions. Off by default."
        ),
    )
    trace_parser.add_argument(
        "--lens-layers",
        type=_parse_layers_spec,
        default="all",
        help=(
            "Which decoder layers the LogitLens captures: 'all' (default) or a "
            "comma-separated list of indices, e.g. '0,3,7,11'."
        ),
    )
    trace_parser.add_argument(
        "--lens-top-k",
        type=int,
        default=8,
        help="Top-k tokens retained per layer in the logit-lens output (default: 8).",
    )
    trace_parser.add_argument(
        "--capture-activations",
        action="store_true",
        help=(
            "Attach an ActivationProbe to capture per-layer / per-submodule summary "
            "stats. Off by default."
        ),
    )
    trace_parser.add_argument(
        "--activation-layers",
        type=_parse_layers_spec,
        default="all",
        help=(
            "Which decoder layers the ActivationProbe captures: 'all' (default) or "
            "a comma-separated list of indices, e.g. '0,3,7,11'."
        ),
    )
    trace_parser.add_argument(
        "--activation-submodules",
        type=_parse_submodules_spec,
        default=list(DEFAULT_ACTIVATION_SUBMODULES),
        help=(
            "Comma-separated submodule keys captured per layer "
            f"(default: {','.join(DEFAULT_ACTIVATION_SUBMODULES)}). "
            "Supported: resid_pre, resid_post (alias: residual_pre, residual_post), "
            "mlp_out (alias: mlp.down_proj), o_proj."
        ),
    )
    trace_parser.add_argument(
        "--activation-top-k",
        type=int,
        default=8,
        help="Top-k highest-magnitude neurons retained per (layer, submodule) (default: 8).",
    )
    trace_parser.add_argument(
        "--capture-full-activations",
        action="store_true",
        help=(
            "Reserved for the Tier-2 activation sidecar path. Tracked here "
            "so the flag surface is stable."
        ),
    )
    trace_parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "After generation, start the FastAPI backend so the frontend can "
            "load the trace via a URL. Requires uvicorn and llm_token_heatmap_api "
            "to be installed. Press Ctrl+C to stop."
        ),
    )
    trace_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend port when --serve is set (default: 8000).",
    )
    trace_parser.add_argument(
        "--frontend-url",
        default="http://localhost:5173",
        help=(
            "Frontend origin printed with --serve so you can copy the ready-made URL "
            "(default: http://localhost:5173)."
        ),
    )
    trace_parser.set_defaults(func=run_trace)

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two adaptive_token_trace.json files and write a delta trace + plot.",
        description=(
            "Run `compare_activations` on two activation-capturing trace JSON files "
            "and write activation_diff.json + activation_delta.png to the output "
            "directory."
        ),
    )
    diff_parser.add_argument(
        "trace_a",
        type=Path,
        help="Path to the first adaptive_token_trace.json (contains activation_metadata).",
    )
    diff_parser.add_argument(
        "trace_b",
        type=Path,
        help="Path to the second adaptive_token_trace.json (contains activation_metadata).",
    )
    diff_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (created if missing). Receives activation_diff.json + activation_delta.png.",
    )
    diff_parser.add_argument(
        "--metric",
        choices=("l2", "cosine"),
        default="l2",
        help="Per-layer metric to record / colour the heatmap by (default: l2).",
    )
    diff_parser.set_defaults(func=run_diff)

    return parser, trace_parser


def _emit_eager_warning(stream: Any = None) -> None:
    """Print the eager-attention performance warning to ``stream`` (defaults to stderr)."""

    target = stream if stream is not None else sys.stderr
    print(EAGER_ATTENTION_WARNING, file=target)


def _build_attention_metadata(
    probe: Any, captured_layers: list[int], total_layers: int | None
) -> dict[str, Any]:
    return {
        "num_layers": int(total_layers)
        if total_layers
        else (max(captured_layers) + 1 if captured_layers else 0),
        "num_attention_heads": int(getattr(probe, "_num_attention_heads", 0)),
        "num_key_value_heads": int(getattr(probe, "_num_key_value_heads", 0)),
        "head_dim": int(getattr(probe, "_head_dim", 0)),
        "captured_layers": [int(i) for i in captured_layers],
    }



def run_trace(args: argparse.Namespace) -> int:
    """Execute the `trace` sub-command.

    Imports torch/transformers lazily so `--help` and argument parsing remain
    fast and cheap to test without model dependencies installed at parse time.
    """
    if args.model is None:
        print(
            "error: --model is required (pass it on the command line or set 'model:' in --config).",
            file=sys.stderr,
        )
        return 2
    if args.prompt is None:
        print(
            "error: --prompt is required (pass it on the command line or set 'prompt:' in --config).",
            file=sys.stderr,
        )
        return 2

    if args.capture_attention:
        _emit_eager_warning()
    if args.capture_full_attention and not args.capture_attention:
        print(
            "[token-heatmap] --capture-full-attention requires --capture-attention; "
            "enabling --capture-attention implicitly.",
            file=sys.stderr,
        )
        args.capture_attention = True
        _emit_eager_warning()
    if args.capture_full_activations and not args.capture_activations:
        print(
            "[token-heatmap] --capture-full-activations requires --capture-activations; "
            "enabling --capture-activations implicitly.",
            file=sys.stderr,
        )
        args.capture_activations = True

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_token_heatmap import (
        ActivationProbe,
        ActivationProbeConfig,
        AdaptiveProbeConfig,
        AdaptiveTokenProbe,
        AttentionProbe,
        AttentionProbeConfig,
        LogitLens,
        LogitLensConfig,
        compute_attention_stats,
        generate_with_adaptive_probe,
        plot_adaptive_heatmap,
        plot_attention_layer_head_grid,
        plot_entropy,
        plot_logit_lens,
        plot_logit_lens_selected_rank,
        plot_selected_probability,
        tokenizer_fingerprint,
        trace_to_dataframe,
        write_sidecar,
    )

    output_dir: Path = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading tokenizer and model: {args.model} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()

    probe = AdaptiveTokenProbe(
        AdaptiveProbeConfig(
            min_k=args.min_k,
            max_k=args.max_k,
            mass_threshold=args.mass_threshold,
        )
    )

    attention_probe: AttentionProbe | None = None
    if args.capture_attention:
        attention_probe = AttentionProbe(
            AttentionProbeConfig(
                layers=args.attention_layers,
                capture_full_distribution=args.capture_full_attention,
                top_k_positions=args.attention_top_k,
            )
        )
        attention_probe.attach(model)

    logit_lens: LogitLens | None = None
    if args.capture_logit_lens:
        logit_lens = LogitLens(
            LogitLensConfig(
                layers=args.lens_layers,
                top_k=args.lens_top_k,
            )
        )
        logit_lens.attach(model)

    activation_probe: ActivationProbe | None = None
    if args.capture_activations:
        activation_probe = ActivationProbe(
            ActivationProbeConfig(
                layers=args.activation_layers,
                submodules=list(args.activation_submodules),
                top_k=args.activation_top_k,
                capture_full=args.capture_full_activations,
            )
        )
        activation_probe.attach(model)

    print(f"Generating up to {args.max_new_tokens} tokens...")
    try:
        text, trace = generate_with_adaptive_probe(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            probe=probe,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            attention_probe=attention_probe,
            logit_lens=logit_lens,
            activation_probe=activation_probe,
        )
    finally:
        if attention_probe is not None:
            attention_probe.detach()
        if logit_lens is not None:
            logit_lens.detach()
        if activation_probe is not None:
            activation_probe.detach()

    generated_path = output_dir / "generated.txt"
    generated_path.write_text(text, encoding="utf-8")

    df = trace_to_dataframe(trace, tokenizer)
    csv_path = output_dir / "adaptive_token_trace.csv"
    df.to_csv(csv_path, index=False)

    sidecar_refs: dict[int, str] = {}
    attention_metadata: dict[str, Any] | None = None
    if attention_probe is not None:
        total_layers = None
        decoder_root = getattr(model, "model", None) or model
        decoder_layers = getattr(decoder_root, "layers", None)
        if decoder_layers is not None:
            try:
                total_layers = len(decoder_layers)
            except TypeError:
                total_layers = None
        attention_metadata = _build_attention_metadata(
            attention_probe, attention_probe.target_layers, total_layers
        )

    activation_metadata: dict[str, Any] | None = None
    if activation_probe is not None:
        activation_metadata = {
            "captured_submodules": list(activation_probe.submodule_keys),
            "num_layers": int(activation_probe.num_layers),
            "hidden_dim": int(activation_probe.hidden_dim),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "captured_layers": [int(i) for i in activation_probe.target_layers],
        }

    if args.capture_full_attention:
        sidecar_dir = output_dir / "attention"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        for entry in trace:
            stats = entry.get("_attention_stats")
            if stats is None:
                continue
            step_idx = int(entry["step"])
            sidecar_path = write_sidecar(
                stats, sidecar_dir / f"attention.{step_idx}", step=step_idx
            )
            sidecar_refs[step_idx] = str(sidecar_path.relative_to(output_dir))

    activation_sidecar_refs: dict[int, str] | None = None
    if args.capture_full_activations:
        from llm_token_heatmap.activation_serializer import (
            write_sidecar as write_activation_sidecar,
        )

        act_sidecar_dir = output_dir / "activations"
        act_sidecar_dir.mkdir(parents=True, exist_ok=True)
        activation_sidecar_refs = {}
        for entry in trace:
            full_stats = entry.get("_activation_full_stats")
            if full_stats is None:
                continue
            step_idx = int(entry["step"])
            act_path = write_activation_sidecar(
                full_stats, act_sidecar_dir / f"activation.{step_idx}", step=step_idx
            )
            if act_path is not None:
                activation_sidecar_refs[step_idx] = str(act_path.relative_to(output_dir))

    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "generated_text": text,
        "device": device,
        "generation_params": {
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "sample_top_k": 0,
        },
        "probe_config": {
            "min_k": int(args.min_k),
            "max_k": int(args.max_k),
            "mass_threshold": float(args.mass_threshold),
        },
        "capture_attention": bool(args.capture_attention),
        "capture_full_attention": bool(args.capture_full_attention),
        "capture_logit_lens": bool(args.capture_logit_lens),
    }

    # Strip all private (underscore-prefixed) keys before JSON dump.
    trace_for_json: list[dict[str, Any]] = []
    for entry in trace:
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        trace_for_json.append(clean)

    json_payload = _serialize_trace_to_json(
        trace=trace_for_json,
        metadata=metadata,
        attention_metadata=attention_metadata,
        sidecar_refs=sidecar_refs,
        tokenizer=tokenizer,
        prompt=args.prompt,
        activation_metadata=activation_metadata,
        activation_sidecar_refs=activation_sidecar_refs,
    )
    json_path = output_dir / "adaptive_token_trace.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    plot_adaptive_heatmap(
        df,
        value_col="logprob",
        save_path=output_dir / "adaptive_heatmap.png",
    )
    plot_selected_probability(df, save_path=output_dir / "selected_probability.png")
    plot_entropy(df, save_path=output_dir / "entropy.png")

    if attention_probe is not None and trace:
        first_with_stats = next((e for e in trace if e.get("_attention_stats") is not None), None)
        if first_with_stats is not None:
            derived = compute_attention_stats(
                first_with_stats["_attention_stats"], top_k=args.attention_top_k
            )
            plot_attention_layer_head_grid(
                derived,
                value="entropy",
                save_path=output_dir / "attention_layer_head_grid.png",
            )

    if logit_lens is not None and trace:
        first_with_lens = next((e for e in trace if "logit_lens" in e), None)
        if first_with_lens is not None:
            plot_logit_lens(first_with_lens, tokenizer, save_path=output_dir / "logit_lens.png")
            plot_logit_lens_selected_rank(
                trace, tokenizer, save_path=output_dir / "selected_rank_heatmap.png"
            )

    print(f"Wrote outputs to {output_dir}/")

    if getattr(args, "serve", False):
        _serve_outputs(output_dir, port=args.port, frontend_url=args.frontend_url)

    return 0


def _serve_outputs(output_dir: Path, port: int = 8000, frontend_url: str = "http://localhost:5173") -> None:
    """Start the FastAPI backend with LLM_HEATMAP_OUTPUT_DIR set to output_dir.

    Blocks until Ctrl+C, then terminates the server.
    """
    import os
    import signal
    import subprocess

    env = os.environ.copy()
    env["LLM_HEATMAP_OUTPUT_DIR"] = str(output_dir.resolve())
    env["LLM_HEATMAP_ALLOWED_ORIGINS"] = frontend_url

    cmd = [
        sys.executable, "-m", "uvicorn",
        "llm_token_heatmap_api.main:app",
        "--host", "::",
        "--port", str(port),
    ]

    backend_url = f"http://localhost:{port}"
    trace_file_url = f"{backend_url}/outputs/adaptive_token_trace.json"
    viewer_url = f"{frontend_url}/?trace={trace_file_url}"

    print("\n[token-heatmap] Starting backend …")
    print(f"[token-heatmap] Backend:  {backend_url}")
    print(f"[token-heatmap] Open the viewer at:")
    print(f"[token-heatmap]   {viewer_url}")
    print("[token-heatmap] (Press Ctrl+C to stop)\n")

    try:
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[token-heatmap] Shutting down …")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _load_activation_trace(path: Path) -> dict[str, Any]:
    """Read a trace JSON file and project its activation subset for `compare_activations`.

    Raises a CLI-friendly error string in `RuntimeError` when the file is
    missing the activation fields. The projection mirrors
    ``trace_payload.project_activation_subset`` so the consumer side stays
    isomorphic to the serializer side.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"trace file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"trace file is not valid JSON: {path}: {exc}") from exc

    if "activation_metadata" not in payload:
        raise RuntimeError(
            f"trace file {path} has no `activation_metadata` block; "
            "rerun the producer with --capture-activations."
        )
    projected_steps: list[dict[str, Any]] = []
    for step in payload.get("steps", []):
        if "activations" not in step:
            continue
        projected_steps.append(
            {
                "step": int(step["step"]),
                "token_id": int(step["token_id"]),
                "decoded_text_offset": int(step["decoded_text_offset"]),
                "activations": step["activations"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "activation_metadata": payload["activation_metadata"],
        "steps": projected_steps,
        "_source_metadata": payload.get("metadata", {}),
    }


def run_diff(args: argparse.Namespace) -> int:
    """Execute the `diff` sub-command.

    Loads two adaptive_token_trace.json files, projects out the activation
    subset that ``compare_activations`` consumes, refuses to diff traces whose
    parent prompts differ or whose activation captures never aligned, and
    writes ``activation_diff.json`` plus ``activation_delta.png`` into the
    output directory.
    """

    from llm_token_heatmap import compare_activations, plot_activation_delta

    try:
        trace_a = _load_activation_trace(args.trace_a)
        trace_b = _load_activation_trace(args.trace_b)
    except RuntimeError as exc:
        print(f"[token-heatmap diff] {exc}", file=sys.stderr)
        return 2

    prompt_a = trace_a["_source_metadata"].get("prompt")
    prompt_b = trace_b["_source_metadata"].get("prompt")
    if prompt_a is not None and prompt_b is not None and prompt_a != prompt_b:
        print(
            "[token-heatmap diff] refusing to diff: parent traces have mismatched "
            f"generations — prompt_a={prompt_a!r} prompt_b={prompt_b!r}.",
            file=sys.stderr,
        )
        return 2

    diff = compare_activations(
        {k: v for k, v in trace_a.items() if not k.startswith("_")},
        {k: v for k, v in trace_b.items() if not k.startswith("_")},
        metric=args.metric,
        align="auto",
    )

    if not diff["steps"]:
        print(
            "[token-heatmap diff] refusing to diff: zero steps aligned between the "
            "two traces — mismatched generations or non-overlapping decoded offsets.",
            file=sys.stderr,
        )
        return 2

    output_dir: Path = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    diff_path = output_dir / "activation_diff.json"
    diff_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")

    plot_activation_delta(
        diff,
        save_path=output_dir / "activation_delta.png",
        metric=args.metric,
    )

    print(f"Wrote outputs to {output_dir}/")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `token-heatmap` console script."""
    parser, trace_parser = build_parser()

    # Two-pass: do a lenient first parse to discover --config, apply its
    # values as defaults on the trace sub-parser, then do the real parse so
    # that explicit CLI flags always win over config-file values.
    ns, _ = parser.parse_known_args(argv)
    if getattr(ns, "config", None) is not None:
        yaml_defaults = _load_yaml_config(ns.config)
        trace_parser.set_defaults(**yaml_defaults)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
