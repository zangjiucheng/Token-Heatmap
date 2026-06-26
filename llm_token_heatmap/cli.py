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
    build_model_architecture as _build_model_architecture_summary,
)
from llm_token_heatmap.trace_payload import (
    selected_token_payload as _selected_token_payload,
)
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


# Built-in probe scalar names, mirrored from `probe.SCALARS`. Kept as a literal
# here so building the parser never imports numpy (the real registry lives in
# `llm_token_heatmap.probe` and is loaded lazily inside `run_manifold`).
_PROBE_SCALARS = ("line_position",)


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the top-level argument parser for the `token-heatmap` CLI.

    Returns ``(parser, trace_subparser)`` so callers can apply YAML defaults
    directly onto ``trace_subparser`` before the final ``parse_args`` call.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        _version = _pkg_version("llm-token-heatmap")
    except PackageNotFoundError:  # source tree without install metadata
        _version = "unknown"

    parser = argparse.ArgumentParser(
        prog="token-heatmap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Analyze LLM inference-time token probability distributions with adaptive\n"
            "top-k tracing, then explore them in the web app's lenses — Heatmap, Logit\n"
            "Lens, Attention, Direct Logit Attribution (+ per-head), Attribution Graph,\n"
            "and Manifold — plus interventions/ablation against a live backend."
        ),
        epilog=(
            "command groups:\n"
            "  generate & analyze   trace, diff, manifold\n"
            "  view                 serve\n"
            "  develop & deploy     dev, web build, hpc {setup,run,serve}\n"
            "\n"
            "examples:\n"
            "  token-heatmap trace --config configs/example.yaml --serve --frontend\n"
            "  token-heatmap trace --config configs/ioi.yaml      # per-head DLA / circuit demo\n"
            "  token-heatmap dev                                  # backend + frontend for local dev\n"
            "  token-heatmap serve outputs/ioi                    # view a run you already produced\n"
            "  token-heatmap hpc run configs/wrap-text.yaml --gpu l40s\n"
            "\n"
            "Run `token-heatmap <command> --help` for a command's options.\n"
            "Configs: configs/README.md   ·   Docs: docs/cli.md"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version}",
        help="Show the installed token-heatmap version and exit.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="<command>",
    )

    trace_parser = subparsers.add_parser(
        "trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Run the adaptive token probe on a model and write trace + plots.",
        description=(
            "Generate text with the given model and write generated.txt, "
            "adaptive_token_trace.csv, adaptive_token_trace.json, "
            "adaptive_heatmap.png, entropy.png, and selected_probability.png "
            "into the output directory."
        ),
        epilog=(
            "examples:\n"
            "  # smallest run — heatmap + logit lens, then open the viewer\n"
            "  token-heatmap trace --config configs/example.yaml --serve --frontend\n"
            "\n"
            "  # everything the Attribution / Graph / Attention lenses + ablation need\n"
            "  token-heatmap trace --model Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "      --prompt 'The capital of France is' \\\n"
            "      --capture-logit-lens --capture-attention --capture-full-attention \\\n"
            "      --capture-activations --capture-full-activations --serve --frontend\n"
            "\n"
            "  # quick CLI-only trace (no web app)\n"
            "  token-heatmap trace --model Qwen/Qwen2.5-0.5B-Instruct --prompt 'Hello' --max-new-tokens 16\n"
            "\n"
            "Most runs are easier to launch from a config (configs/README.md).\n"
            "Per-head DLA / the Attribution Graph need --capture-full-activations."
        ),
    )
    model_grp = trace_parser.add_argument_group("model")
    gen_grp = trace_parser.add_argument_group("generation & sampling")
    attn_grp = trace_parser.add_argument_group("capture: attention (Attention lens)")
    lens_grp = trace_parser.add_argument_group("capture: logit lens (Logit Lens)")
    act_grp = trace_parser.add_argument_group(
        "capture: activations (Attribution / Graph / Manifold)"
    )
    serve_grp = trace_parser.add_argument_group("serve & view")
    model_grp.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "YAML config file. All CLI flags override config file values. "
            "Requires PyYAML (pip install pyyaml)."
        ),
    )
    model_grp.add_argument(
        "--model",
        default=None,
        help="HuggingFace model id or local path (e.g. Qwen/Qwen2.5-0.5B-Instruct).",
    )
    model_grp.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model in 4-bit NF4 (bitsandbytes) on GPU — fits a 32B on a "
        "single 48 GB GPU. Ignored on CPU. Activations are still captured in fp16.",
    )
    gen_grp.add_argument(
        "--prompt",
        default=None,
        help="Input prompt string for generation.",
    )
    gen_grp.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate (default: 32).",
    )
    gen_grp.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    gen_grp.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling cutoff (default: 0.95).",
    )
    gen_grp.add_argument(
        "--min-k",
        type=int,
        default=8,
        help="Minimum adaptive top-k (default: 8).",
    )
    gen_grp.add_argument(
        "--max-k",
        type=int,
        default=64,
        help="Maximum adaptive top-k (default: 64).",
    )
    gen_grp.add_argument(
        "--mass-threshold",
        type=float,
        default=0.95,
        help="Cumulative probability mass target for adaptive top-k (default: 0.95).",
    )
    gen_grp.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory (created if missing; default: outputs/).",
    )
    attn_grp.add_argument(
        "--capture-attention",
        action="store_true",
        help=(
            "Attach an AttentionProbe to capture per-layer attention weights "
            "and Q/K/V vectors. Forces eager attention (slow). Off by default."
        ),
    )
    attn_grp.add_argument(
        "--attention-layers",
        type=_parse_layers_spec,
        default="all",
        help=(
            "Which decoder layers the AttentionProbe captures: 'all' (default) "
            "or a comma-separated list of indices, e.g. '0,3,7,11'."
        ),
    )
    attn_grp.add_argument(
        "--attention-top-k",
        type=int,
        default=8,
        help="Top-k attended positions kept inline per head (default: 8).",
    )
    attn_grp.add_argument(
        "--capture-full-attention",
        action="store_true",
        help=(
            "When set, write a per-step sidecar attention.<step>.npz under "
            "<out>/attention/ containing the full attention distribution and "
            "Q/K/V tensors. Requires --capture-attention."
        ),
    )
    lens_grp.add_argument(
        "--capture-logit-lens",
        action="store_true",
        help=(
            "Attach a LogitLens probe to capture per-layer next-token predictions. Off by default."
        ),
    )
    lens_grp.add_argument(
        "--lens-layers",
        type=_parse_layers_spec,
        default="all",
        help=(
            "Which decoder layers the LogitLens captures: 'all' (default) or a "
            "comma-separated list of indices, e.g. '0,3,7,11'."
        ),
    )
    lens_grp.add_argument(
        "--lens-top-k",
        type=int,
        default=8,
        help="Top-k tokens retained per layer in the logit-lens output (default: 8).",
    )
    act_grp.add_argument(
        "--capture-activations",
        action="store_true",
        help=(
            "Attach an ActivationProbe to capture per-layer / per-submodule summary "
            "stats. Off by default."
        ),
    )
    act_grp.add_argument(
        "--activation-layers",
        type=_parse_layers_spec,
        default="all",
        help=(
            "Which decoder layers the ActivationProbe captures: 'all' (default) or "
            "a comma-separated list of indices, e.g. '0,3,7,11'."
        ),
    )
    act_grp.add_argument(
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
    act_grp.add_argument(
        "--activation-top-k",
        type=int,
        default=8,
        help="Top-k highest-magnitude neurons retained per (layer, submodule) (default: 8).",
    )
    act_grp.add_argument(
        "--capture-full-activations",
        action="store_true",
        help=(
            "Write the full per-(layer, submodule) activation vectors to Tier-2 "
            ".npz sidecars (requires --capture-activations). Needed for `token-heatmap "
            "manifold` analysis and per-head DLA / the Attribution Graph."
        ),
    )
    serve_grp.add_argument(
        "--serve",
        action="store_true",
        help=(
            "After generation, serve the output directory over HTTP (Python's "
            "stdlib http.server with CORS — no extra dependencies) so the frontend "
            "can load the trace via a URL. Press Ctrl+C to stop."
        ),
    )
    serve_grp.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend port when --serve is set (default: 8000).",
    )
    serve_grp.add_argument(
        "--frontend-url",
        default="http://localhost:5173",
        help=(
            "Frontend origin printed with --serve so you can copy the ready-made URL "
            "(default: http://localhost:5173). With --frontend, the dev server binds "
            "to this URL's port."
        ),
    )
    serve_grp.add_argument(
        "--frontend",
        action="store_true",
        help=(
            "Also start the Vite frontend (npm run dev) from web/frontend and open "
            "the viewer in your browser. Requires Node.js and a repo checkout. "
            "Implies --serve. Press Ctrl+C to stop both."
        ),
    )
    serve_grp.add_argument(
        "--no-open",
        action="store_true",
        help="With --frontend, do not auto-open the viewer in a browser.",
    )
    trace_parser.set_defaults(func=run_trace)

    diff_parser = subparsers.add_parser(
        "diff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Compare two adaptive_token_trace.json files and write a delta trace + plot.",
        description=(
            "Run `compare_activations` on two activation-capturing trace JSON files "
            "and write activation_diff.json + activation_delta.png to the output "
            "directory."
        ),
        epilog=(
            "example:\n"
            "  token-heatmap diff outputs/a/adaptive_token_trace.json \\\n"
            "      outputs/b/adaptive_token_trace.json --out outputs/diff --metric cosine"
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

    manifold_parser = subparsers.add_parser(
        "manifold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Add manifold analysis of captured activations to a trace JSON.",
        description=(
            "Read an adaptive_token_trace.json plus its activation sidecars (written "
            "with --capture-full-activations), run PCA / intrinsic-dimension / curvature "
            "/ periodicity analysis per (layer, submodule), and write a `manifold` field "
            "back into the trace so the web app's Manifold tab can render it."
        ),
        epilog=(
            "example:\n"
            "  token-heatmap manifold --trace outputs/wrap-text/adaptive_token_trace.json \\\n"
            "      --components 6 --probe line_position"
        ),
    )
    manifold_parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to an adaptive_token_trace.json with activation sidecars.",
    )
    manifold_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the augmented trace (default: overwrite --trace in place).",
    )
    manifold_parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Subset of layer indices to analyze (default: all captured layers).",
    )
    manifold_parser.add_argument(
        "--submodules",
        nargs="*",
        default=None,
        help="Subset of submodule names to analyze (default: all captured submodules).",
    )
    manifold_parser.add_argument(
        "--components",
        type=int,
        default=3,
        help="Number of PCA projection components to keep (default: 3).",
    )
    manifold_parser.add_argument(
        "--probe",
        choices=sorted(_PROBE_SCALARS),
        default=None,
        help=(
            "Fit a supervised linear probe of a per-position scalar against each "
            "cloud (reports r2_cv per layer; colours the manifold by the scalar in "
            "the web app). 'line_position' = characters since the last newline."
        ),
    )
    manifold_parser.add_argument(
        "--scalar-max",
        type=float,
        default=None,
        help=(
            "Drop positions whose probe scalar exceeds this before fitting (keeps "
            "the manifold geometry untouched). Use it to exclude run-on outliers — "
            "e.g. a single line the model failed to wrap — that otherwise skew the "
            "period sweep and inflate the helix R²."
        ),
    )
    manifold_parser.set_defaults(func=run_manifold)

    serve_parser = subparsers.add_parser(
        "serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Serve an existing output directory over HTTP with CORS (no generation).",
        description=(
            "Start a stdlib http.server with CORS headers serving DIR so the frontend "
            "(possibly on another host, via an SSH port-forward) can fetch the trace "
            "JSON. Unlike `trace --serve`, this does NOT regenerate — use it to serve a "
            "trace you have already produced (and, optionally, augmented with "
            "`token-heatmap manifold`). Press Ctrl+C to stop."
        ),
        epilog=(
            "examples:\n"
            "  token-heatmap serve outputs/ioi                 # files only (view via ?trace=… URL)\n"
            "  token-heatmap serve outputs/ioi --frontend      # also start the viewer + open it"
        ),
    )
    serve_parser.add_argument(
        "dir",
        type=Path,
        nargs="?",
        default=Path("outputs"),
        help="Directory to serve (default: outputs/).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the file server (default: 8000).",
    )
    serve_parser.add_argument(
        "--frontend-url",
        default="http://localhost:5173",
        help=(
            "Frontend origin used to build the printed viewer URL (default: "
            "http://localhost:5173). With --frontend, the dev server binds to this "
            "URL's port."
        ),
    )
    serve_parser.add_argument(
        "--frontend",
        action="store_true",
        help=(
            "Also start the Vite frontend (npm run dev) from web/frontend and open the "
            "viewer in your browser. Requires Node.js and a repo checkout."
        ),
    )
    serve_parser.add_argument(
        "--no-open",
        action="store_true",
        help="With --frontend, do not auto-open the viewer in a browser.",
    )
    serve_parser.set_defaults(func=run_serve)

    # Operational sub-commands (dev / web build / hpc …) live in
    # `llm_token_heatmap.commands` so they replace scripts/*.sh without bloating
    # this module. They import only stdlib, so registering them keeps `--help`
    # cheap (no torch/numpy pulled in).
    from llm_token_heatmap.commands import dev as _dev_cmd
    from llm_token_heatmap.commands import hpc as _hpc_cmd
    from llm_token_heatmap.commands import web as _web_cmd

    _dev_cmd.register(subparsers)
    _web_cmd.register(subparsers)
    _hpc_cmd.register(subparsers)

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
    # Prefer bfloat16 on CUDA. Qwen2.5 (and most modern LLMs) ship in bf16; fp16's
    # narrow exponent range overflows to inf on their large activations —
    # especially under the eager attention kernel forced by --capture-attention —
    # producing NaN logits that make multinomial sampling device-side-assert and
    # crash the run. bf16 has fp32's exponent range, so it doesn't overflow.
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    load_in_4bit = bool(getattr(args, "load_in_4bit", False)) and use_cuda
    print(
        f"Loading tokenizer and model: {args.model} "
        f"(device={device}{', 4-bit NF4' if load_in_4bit else ''})"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_cuda:
        # Stream the weight shards straight onto the GPU rather than building the
        # full model in host RAM and then `.to(cuda)` (a 14B fp16 peaks ~28 GB of
        # CPU memory; a 32B would blow a tight Slurm --mem cap). Needs accelerate.
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["low_cpu_mem_usage"] = True
        if load_in_4bit:
            # 4-bit NF4 — fits a 32B on a single 48 GB GPU. Needs bitsandbytes.
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not use_cuda:
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

    attention_metadata: dict[str, Any] | None = None
    activation_metadata: dict[str, Any] | None = None
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
        # Snapshot probe metadata BEFORE detaching: detach() clears each probe's
        # captured-layer / submodule lists, so building the metadata afterwards
        # would record empty lists and silently break the Activations and
        # Manifold views (which key off captured_submodules / captured_layers).
        if attention_probe is not None:
            decoder_root = getattr(model, "model", None) or model
            decoder_layers = getattr(decoder_root, "layers", None)
            try:
                total_layers = len(decoder_layers) if decoder_layers is not None else None
            except TypeError:
                total_layers = None
            attention_metadata = _build_attention_metadata(
                attention_probe, attention_probe.target_layers, total_layers
            )
            attention_probe.detach()
        if logit_lens is not None:
            logit_lens.detach()
        if activation_probe is not None:
            activation_metadata = {
                "captured_submodules": list(activation_probe.submodule_keys),
                "num_layers": int(activation_probe.num_layers),
                "hidden_dim": int(activation_probe.hidden_dim),
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
                "captured_layers": [int(i) for i in activation_probe.target_layers],
            }
            activation_probe.detach()

    generated_path = output_dir / "generated.txt"
    generated_path.write_text(text, encoding="utf-8")

    df = trace_to_dataframe(trace, tokenizer)
    csv_path = output_dir / "adaptive_token_trace.csv"
    df.to_csv(csv_path, index=False)

    sidecar_refs: dict[int, str] = {}

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

    model_architecture = _build_model_architecture_summary(model, dtype=dtype)

    # TWERA-style neuron attribution (a single-trace approximation). Needs the
    # full per-step activation vectors (--capture-full-activations) and the
    # unembedding; skipped otherwise. See llm_token_heatmap.neuron_attribution.
    neuron_attribution = None
    direct_logit_attribution = None
    if args.capture_full_activations:
        from llm_token_heatmap.activation_probe import (
            _resolve_decoder_layers,
            _resolve_submodule_target,
        )
        from llm_token_heatmap.direct_logit_attribution import (
            compute_direct_logit_attribution,
        )
        from llm_token_heatmap.logit_lens import _resolve_final_norm
        from llm_token_heatmap.neuron_attribution import compute_neuron_attribution

        out_emb = model.get_output_embeddings()
        if out_emb is not None and getattr(out_emb, "weight", None) is not None:
            target_ids = [
                int(_selected_token_payload(entry["raw"], tokenizer)["token_id"])
                for entry in trace
            ]
            neuron_attribution = compute_neuron_attribution(
                trace=trace,
                target_token_ids=target_ids,
                unembedding=out_emb.weight,
                top_n=max(8, int(args.activation_top_k)),
            )
            # Per-head DLA needs each layer's o_proj weight (W_O) + head geometry.
            _layers = _resolve_decoder_layers(model)
            o_proj_weights = {}
            for _i, _layer in enumerate(_layers):
                _op = _resolve_submodule_target(_layer, "o_proj")
                _w = getattr(_op, "weight", None) if _op is not None else None
                if _w is not None:
                    o_proj_weights[_i] = _w
            _cfg = getattr(model, "config", None)
            _nh = int(getattr(_cfg, "num_attention_heads", 0) or 0)
            _hd = getattr(_cfg, "head_dim", None)
            if not _hd and _nh:
                _hs = getattr(_cfg, "hidden_size", 0) or 0
                _hd = _hs // _nh if _hs else 0
            # Direct logit attribution reuses the same captured tensors +
            # unembedding, folding the model's real final norm. See
            # docs/epics/01-direct-logit-attribution.md.
            direct_logit_attribution = compute_direct_logit_attribution(
                trace=trace,
                target_token_ids=target_ids,
                unembedding=out_emb.weight,
                final_norm=_resolve_final_norm(model),
                o_proj_weights=o_proj_weights or None,
                num_heads=_nh or None,
                head_dim=int(_hd) if _hd else None,
            )

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
        model_architecture=model_architecture,
        neuron_attribution=neuron_attribution,
        direct_logit_attribution=direct_logit_attribution,
    )
    json_path = output_dir / "adaptive_token_trace.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    # Plots are a secondary convenience — the JSON/CSV above are the product.
    # A single matplotlib hiccup (a '$' or an exotic glyph in a token label)
    # must never abort the run after the data is written, nor block the
    # downstream `manifold` step. Degrade to a warning instead.
    try:
        plot_adaptive_heatmap(
            df,
            value_col="logprob",
            save_path=output_dir / "adaptive_heatmap.png",
        )
        plot_selected_probability(df, save_path=output_dir / "selected_probability.png")
        plot_entropy(df, save_path=output_dir / "entropy.png")

        if attention_probe is not None and trace:
            first_with_stats = next(
                (e for e in trace if e.get("_attention_stats") is not None), None
            )
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
                plot_logit_lens(
                    first_with_lens, tokenizer, save_path=output_dir / "logit_lens.png"
                )
                plot_logit_lens_selected_rank(
                    trace, tokenizer, save_path=output_dir / "selected_rank_heatmap.png"
                )
    except Exception as exc:  # noqa: BLE001 — plots must not fail the run after JSON is written.
        print(
            f"[token-heatmap] WARNING: plot generation failed "
            f"({type(exc).__name__}: {exc}); the trace JSON/CSV were still written.",
            file=sys.stderr,
        )

    print(f"Wrote outputs to {output_dir}/")

    start_frontend = getattr(args, "frontend", False)
    if getattr(args, "serve", False) or start_frontend:
        _serve_outputs(
            output_dir,
            port=args.port,
            frontend_url=args.frontend_url,
            start_frontend=start_frontend,
            open_browser=not getattr(args, "no_open", False),
        )

    return 0


def _serve_outputs(
    output_dir: Path,
    port: int = 8000,
    frontend_url: str = "http://localhost:5173",
    start_frontend: bool = False,
    open_browser: bool = True,
) -> None:
    """Serve the output directory over HTTP using Python's stdlib http.server.

    No extra dependencies required — works with any Python 3.10+ installation.
    CORS headers are added so the frontend (running on a different port or host)
    can fetch the trace JSON.

    When ``start_frontend`` is set, also launch the bundled Vite frontend
    (``npm run dev`` in ``web/frontend``) and, unless ``open_browser`` is False,
    open the ready-made viewer URL once the dev server is accepting connections.
    The frontend subprocess is terminated on shutdown.

    Blocks until Ctrl+C.
    """
    import http.server
    import os
    import socketserver

    class _CORSHandler(http.server.SimpleHTTPRequestHandler):
        """Static-file handler with permissive CORS headers."""

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self) -> None:  # pre-flight
            self.send_response(200)
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # suppress per-request noise; the URL is already printed below

    backend_url = f"http://localhost:{port}"
    trace_file_url = f"{backend_url}/adaptive_token_trace.json"
    viewer_url = f"{frontend_url}/?trace={trace_file_url}"

    # Start the frontend before the file server so its npm/Vite startup logs
    # print first and the "open the viewer" line lands last.
    frontend_proc = None
    if start_frontend:
        frontend_proc = _start_frontend_dev_server(frontend_url, backend_url)
        if frontend_proc is None:
            # npm or the frontend dir was unavailable; degrade to files-only.
            start_frontend = False

    # Bind BEFORE announcing, and set allow_reuse_address before the bind (it has
    # no effect once TCPServer.__init__ has already bound). A port conflict then
    # gives a clear one-line message instead of a raw OSError traceback printed
    # after a misleading "Serving …".
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("", port), _CORSHandler)
    except OSError as exc:
        if frontend_proc is not None:
            _terminate_process(frontend_proc)
        raise SystemExit(
            f"[token-heatmap] ERROR: could not bind port {port} "
            f"({getattr(exc, 'strerror', None) or exc}). It's likely already in "
            f"use — rerun with --port <N> (e.g. --port {port + 1})."
        ) from exc

    print("\n[token-heatmap] Serving output directory …")
    print(f"[token-heatmap] Files: {backend_url}/")
    if start_frontend:
        print(f"[token-heatmap] Frontend (npm run dev): {frontend_url}")
    print("[token-heatmap] Open the viewer at:")
    print(f"[token-heatmap]   {viewer_url}")
    print("[token-heatmap] (Press Ctrl+C to stop)\n")

    orig_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        with httpd:
            if start_frontend and open_browser:
                _open_viewer_when_ready(viewer_url, frontend_url)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[token-heatmap] Shutting down …")
    finally:
        os.chdir(orig_dir)
        if frontend_proc is not None:
            _terminate_process(frontend_proc)


def _start_frontend_dev_server(frontend_url: str, backend_url: str) -> Any:
    """Launch ``npm run dev`` for the bundled ``web/frontend``.

    Returns the ``subprocess.Popen`` handle, or ``None`` (after printing a
    warning) when npm or the frontend directory is unavailable so the caller
    can fall back to serving files only.
    """
    import os
    import shutil
    import subprocess
    from urllib.parse import urlparse

    # llm_token_heatmap/cli.py -> repo root is one parent up.
    repo_root = Path(__file__).resolve().parents[1]
    frontend_dir = repo_root / "web" / "frontend"
    if not frontend_dir.is_dir():
        print(
            f"[token-heatmap] WARNING: --frontend set but {frontend_dir} was not found. "
            "Run from a repo checkout to use it. Serving files only."
        )
        return None

    npm = shutil.which("npm")
    if npm is None:
        print(
            "[token-heatmap] WARNING: --frontend set but 'npm' is not on PATH. "
            "Install Node.js 20+ to use it. Serving files only."
        )
        return None

    frontend_port = urlparse(frontend_url).port or 5173

    env = dict(os.environ)
    # Point the SPA's API base at our file server so its same-origin assumptions
    # hold; the trace itself loads via the ?trace= URL regardless. Respect a
    # caller-provided value.
    env.setdefault("VITE_API_BASE_URL", backend_url)

    print(f"[token-heatmap] Starting frontend (npm run dev) on port {frontend_port} …")
    try:
        return subprocess.Popen(
            [npm, "run", "dev", "--", "--port", str(frontend_port), "--strictPort"],
            cwd=str(frontend_dir),
            env=env,
        )
    except OSError as exc:
        print(f"[token-heatmap] WARNING: failed to start npm ({exc}). Serving files only.")
        return None


def _open_viewer_when_ready(viewer_url: str, frontend_url: str) -> None:
    """Open ``viewer_url`` once the frontend port accepts connections.

    Polls in a daemon thread so it never blocks the file server. Gives up
    quietly after a timeout — the URL is already printed for manual use.
    """
    import socket
    import threading
    import time
    import webbrowser
    from urllib.parse import urlparse

    parsed = urlparse(frontend_url)
    host = parsed.hostname or "localhost"
    fport = parsed.port or 5173

    def _wait_and_open() -> None:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, fport), timeout=1.0):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            return  # never came up; nothing to open
        try:
            webbrowser.open(viewer_url)
        except Exception:  # pragma: no cover - platform-dependent
            pass

    threading.Thread(target=_wait_and_open, daemon=True).start()


def _terminate_process(proc: Any) -> None:
    """Terminate a subprocess, escalating to kill if it ignores SIGTERM."""
    import subprocess

    if proc.poll() is not None:
        return
    proc.terminate()
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


def run_manifold(args: argparse.Namespace) -> int:
    """Execute the `manifold` sub-command.

    Re-hydrates the full per-(layer, submodule) activation vectors from the
    sidecars referenced by each step, stacks them into a ``(positions, hidden)``
    matrix, runs :func:`llm_token_heatmap.manifold.analyze_manifold` on each, and
    writes the results back into the trace under a top-level ``manifold`` key.
    """
    import numpy as np

    from llm_token_heatmap.activation_serializer import read_sidecar
    from llm_token_heatmap.manifold import MANIFOLD_SCHEMA_VERSION, analyze_manifold

    trace_path: Path = args.trace
    if not trace_path.is_file():
        print(f"error: trace file not found: {trace_path}", file=sys.stderr)
        return 2

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    meta = payload.get("activation_metadata")
    if meta is None:
        print(
            "error: trace has no activation_metadata. Re-run `token-heatmap trace` "
            "with --capture-activations --capture-full-activations.",
            file=sys.stderr,
        )
        return 2

    steps = payload.get("steps", [])
    refs = [
        (int(s["step"]), s["activation_sidecar_ref"])
        for s in steps
        if s.get("activation_sidecar_ref")
    ]
    if not refs:
        print(
            "error: no step carries an activation_sidecar_ref. Re-run `token-heatmap "
            "trace` with --capture-full-activations.",
            file=sys.stderr,
        )
        return 2

    # Resolve which (layer, submodule) clouds to analyze. Priority: explicit CLI
    # flags, then the trace's captured_* metadata, then — if that metadata is
    # empty (older traces recorded empty lists) — fall back to whatever the
    # sidecars actually contain. ``None`` means "accept everything".
    target_layer_set: set[int] | None = (
        {int(layer) for layer in args.layers}
        if args.layers
        else ({int(layer) for layer in meta.get("captured_layers") or []} or None)
    )
    target_submodule_set: set[str] | None = (
        set(args.submodules)
        if args.submodules
        else (set(meta.get("captured_submodules") or []) or None)
    )

    base = trace_path.parent
    # (layer, submodule) -> (positions, list-of-vectors), accumulated in step order.
    collected: dict[tuple[int, str], tuple[list[int], list[list[float]]]] = {}
    for step_idx, ref in refs:
        sidecar = read_sidecar(base / ref)
        for layer_entry in sidecar.get("layers", []):
            layer = int(layer_entry["layer"])
            if target_layer_set is not None and layer not in target_layer_set:
                continue
            for submodule, vector in layer_entry.get("submodule_tensors", {}).items():
                if target_submodule_set is not None and submodule not in target_submodule_set:
                    continue
                pos_list, vec_list = collected.setdefault((layer, submodule), ([], []))
                pos_list.append(step_idx)
                vec_list.append(vector)

    # Optional supervised probe: compute the per-position scalar once (in
    # generation order) and a step -> value lookup to align with each cloud.
    scalar_by_step: dict[int, float] | None = None
    scalar_block: dict[str, Any] | None = None
    if args.probe:
        from llm_token_heatmap.probe import SCALARS, helix_probe, linear_probe

        token_texts = [str(s.get("selected", {}).get("token", "")) for s in steps]
        step_numbers = [int(s["step"]) for s in steps]
        values = SCALARS[args.probe](token_texts)
        scalar_by_step = {step_numbers[i]: values[i] for i in range(len(steps))}
        scalar_block = {
            "name": args.probe,
            "positions": step_numbers,
            "values": [float(v) for v in values],
        }

    layers_out: list[dict[str, Any]] = []
    for (layer, submodule), (positions, vectors) in sorted(collected.items()):
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            continue  # need at least two positions to have any geometry
        entry = analyze_manifold(matrix, positions=positions, n_components=args.components)
        if scalar_by_step is not None:
            cloud_scalar = [scalar_by_step.get(p, 0.0) for p in positions]
            probe_matrix = matrix
            probe_scalar = cloud_scalar
            if args.scalar_max is not None:
                keep = [i for i, v in enumerate(cloud_scalar) if v <= args.scalar_max]
                probe_matrix = matrix[keep]
                probe_scalar = [cloud_scalar[i] for i in keep]
            probe = linear_probe(probe_matrix, probe_scalar)
            helix = helix_probe(probe_matrix, probe_scalar)
            entry["probe"] = {
                "scalar": args.probe,
                "r2_cv": probe["r2_cv"],
                "r2_full": probe["r2_full"],
                "n_components": probe["n_components"],
                "cv_folds": probe["cv_folds"],
                "circular": {
                    "best_period": helix["best_period"],
                    "r2_cv": helix["r2_cv"],
                    "r2_full": helix["r2_full"],
                },
            }
        layers_out.append({"layer": layer, "submodule": submodule, **entry})

    if not layers_out:
        print(
            "error: no (layer, submodule) cloud had >= 2 positions to analyze.",
            file=sys.stderr,
        )
        return 2

    payload["manifold"] = {
        "schema_version": MANIFOLD_SCHEMA_VERSION,
        "method": "pca",
        "n_components": int(args.components),
        "layers": layers_out,
    }
    if scalar_block is not None:
        payload["manifold"]["scalar"] = scalar_block

    out_path: Path = args.out if args.out is not None else trace_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Added manifold analysis ({len(layers_out)} layer/submodule "
        f"cloud{'s' if len(layers_out) != 1 else ''}) to {out_path}"
    )
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Execute the `serve` sub-command: serve an existing directory, no generation.

    Thin wrapper over :func:`_serve_outputs` (the same CORS file server the
    ``trace --serve`` flag uses) so a trace produced earlier — and augmented with
    ``token-heatmap manifold`` — can be served without re-running generation.
    """
    serve_dir: Path = args.dir
    if not serve_dir.is_dir():
        print(f"error: directory not found: {serve_dir}", file=sys.stderr)
        return 2

    _serve_outputs(
        serve_dir,
        port=args.port,
        frontend_url=args.frontend_url,
        start_frontend=getattr(args, "frontend", False),
        open_browser=not getattr(args, "no_open", False),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `token-heatmap` console script."""
    parser, trace_parser = build_parser()

    # Two-pass: do a lenient first parse to discover --config, apply its
    # values as defaults on the trace sub-parser, then do the real parse so
    # that explicit CLI flags always win over config-file values.
    ns, _ = parser.parse_known_args(argv)
    # Only the `trace` sub-command takes a YAML config to preload as defaults.
    # Other sub-commands (e.g. `hpc run <config>`, `hpc serve --config`) also
    # have a `config` dest, but theirs is a plain path string handled by their
    # own run function — don't try to load it here.
    if getattr(ns, "command", None) == "trace" and getattr(ns, "config", None) is not None:
        yaml_defaults = _load_yaml_config(ns.config)
        trace_parser.set_defaults(**yaml_defaults)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
