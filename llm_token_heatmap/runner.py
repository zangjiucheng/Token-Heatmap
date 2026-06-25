"""Programmatic trace generation for the web backend.

Loads a HuggingFace causal LM, runs the adaptive token probe (plus optional
*inline* attention / logit-lens / activation captures), and returns the same
JSON payload that ``token-heatmap trace`` writes to
``adaptive_token_trace.json`` — without touching disk or producing plots.

This mirrors the inline-JSON core of :func:`llm_token_heatmap.cli.run_trace`
(model load → generate → metadata → serialize) but trims the CSV/PNG/sidecar
side effects. The serializer
(:func:`llm_token_heatmap.trace_payload.serialize_trace_to_json`) stays the
single source of truth for the payload shape; only the orchestration is
duplicated here.

Security note: like the CLI, this loads arbitrary models with
``trust_remote_code=True``, which can execute model-author code. Only expose a
service that calls this over a trusted channel (e.g. an SSH tunnel), never the
public internet.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from llm_token_heatmap.trace_payload import (
    build_model_architecture,
    serialize_trace_to_json,
)

DEFAULT_ACTIVATION_SUBMODULES: tuple[str, ...] = ("residual_post", "mlp_out", "o_proj")

# A layer selector is either the string "all" or an explicit list of indices,
# matching the CLI's ``_parse_layers_spec`` output.
LayerSpec = Any


@dataclass
class GenerateTraceConfig:
    """Inputs for a single generation. Mirrors the CLI ``trace`` flags."""

    model: str
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    min_k: int = 8
    max_k: int = 64
    mass_threshold: float = 0.95
    capture_attention: bool = False
    capture_logit_lens: bool = False
    capture_activations: bool = False
    attention_layers: LayerSpec = "all"
    attention_top_k: int = 8
    lens_layers: LayerSpec = "all"
    lens_top_k: int = 8
    activation_layers: LayerSpec = "all"
    activation_submodules: tuple[str, ...] = DEFAULT_ACTIVATION_SUBMODULES
    activation_top_k: int = 8


# A single global lock serializes the whole generate path. The backend runs
# this in a threadpool, but on a single GPU box concurrent generations would
# only contend for memory and race on shared probe/model state — so we run them
# one at a time. This also makes the model cache below single-threaded.
_GEN_LOCK = threading.Lock()

# Small LRU of loaded (model, tokenizer, device) keyed by model id so repeated
# generations with the same model skip the multi-second load. Touched only
# while holding ``_GEN_LOCK``.
_MODEL_CACHE: OrderedDict[str, tuple[Any, Any, str]] = OrderedDict()
_CACHE_MAX = 1


def _load_model_and_tokenizer(model_id: str) -> tuple[Any, Any, str]:
    """Return ``(model, tokenizer, device)``, reusing the cache when possible.

    Caller must hold ``_GEN_LOCK``.
    """
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None:
        _MODEL_CACHE.move_to_end(model_id)
        return cached

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if use_cuda:
        # Stream the weight shards straight onto the GPU rather than
        # materialising the whole model in host RAM first and then `.to(cuda)`.
        # A 14B fp16 model would otherwise peak ~28 GB of CPU memory and blow a
        # tight Slurm --mem cap. Needs `accelerate` (a declared dependency).
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if not use_cuda:
        model = model.to(device)  # type: ignore[arg-type]
    model.eval()

    entry = (model, tokenizer, device)
    _MODEL_CACHE[model_id] = entry
    _MODEL_CACHE.move_to_end(model_id)
    while len(_MODEL_CACHE) > _CACHE_MAX:
        _evicted_id, evicted = _MODEL_CACHE.popitem(last=False)
        del evicted
        if use_cuda:
            torch.cuda.empty_cache()
    return entry


def _count_decoder_layers(model: Any) -> int | None:
    """Best-effort decoder-layer count for attention metadata (see cli.run_trace)."""
    decoder_root = getattr(model, "model", None) or model
    decoder_layers = getattr(decoder_root, "layers", None)
    if decoder_layers is None:
        return None
    try:
        return len(decoder_layers)
    except TypeError:
        return None


def _build_attention_metadata(
    probe: Any, captured_layers: list[int], total_layers: int | None
) -> dict[str, Any]:
    """Mirror of ``cli._build_attention_metadata`` (kept local to decouple modules)."""
    return {
        "num_layers": int(total_layers)
        if total_layers
        else (max(captured_layers) + 1 if captured_layers else 0),
        "num_attention_heads": int(getattr(probe, "_num_attention_heads", 0)),
        "num_key_value_heads": int(getattr(probe, "_num_key_value_heads", 0)),
        "head_dim": int(getattr(probe, "_head_dim", 0)),
        "captured_layers": [int(i) for i in captured_layers],
    }


def generate_trace_payload(config: GenerateTraceConfig) -> dict[str, Any]:
    """Run generation and return the canonical trace JSON payload (no disk I/O).

    The returned dict has the same shape as ``adaptive_token_trace.json`` and
    validates against ``docs/web/trace.schema.json``. Raises whatever
    transformers / torch raise on a bad model id, OOM, etc. — callers are
    responsible for mapping those to a response.
    """
    from llm_token_heatmap import (
        ActivationProbe,
        ActivationProbeConfig,
        AdaptiveProbeConfig,
        AdaptiveTokenProbe,
        AttentionProbe,
        AttentionProbeConfig,
        LogitLens,
        LogitLensConfig,
        generate_with_adaptive_probe,
        tokenizer_fingerprint,
    )

    with _GEN_LOCK:
        model, tokenizer, device = _load_model_and_tokenizer(config.model)

        probe = AdaptiveTokenProbe(
            AdaptiveProbeConfig(
                min_k=config.min_k,
                max_k=config.max_k,
                mass_threshold=config.mass_threshold,
            )
        )

        attention_probe: Any | None = None
        logit_lens: Any | None = None
        activation_probe: Any | None = None
        try:
            if config.capture_attention:
                attention_probe = AttentionProbe(
                    AttentionProbeConfig(
                        layers=config.attention_layers,
                        capture_full_distribution=False,  # inline only
                        top_k_positions=config.attention_top_k,
                    )
                )
                attention_probe.attach(model)
            if config.capture_logit_lens:
                logit_lens = LogitLens(
                    LogitLensConfig(layers=config.lens_layers, top_k=config.lens_top_k)
                )
                logit_lens.attach(model)
            if config.capture_activations:
                activation_probe = ActivationProbe(
                    ActivationProbeConfig(
                        layers=config.activation_layers,
                        submodules=list(config.activation_submodules),
                        top_k=config.activation_top_k,
                        capture_full=False,  # inline only
                    )
                )
                activation_probe.attach(model)

            text, trace = generate_with_adaptive_probe(
                model=model,
                tokenizer=tokenizer,
                prompt=config.prompt,
                probe=probe,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
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

        attention_metadata: dict[str, Any] | None = None
        if attention_probe is not None:
            attention_metadata = _build_attention_metadata(
                attention_probe,
                attention_probe.target_layers,
                _count_decoder_layers(model),
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

        metadata = {
            "model": config.model,
            "prompt": config.prompt,
            "generated_text": text,
            "device": device,
            "generation_params": {
                "max_new_tokens": int(config.max_new_tokens),
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "sample_top_k": 0,
            },
            "probe_config": {
                "min_k": int(config.min_k),
                "max_k": int(config.max_k),
                "mass_threshold": float(config.mass_threshold),
            },
            "capture_attention": bool(config.capture_attention),
            "capture_full_attention": False,
            "capture_logit_lens": bool(config.capture_logit_lens),
        }

        param0 = next(model.parameters(), None)
        model_architecture = build_model_architecture(
            model, dtype=getattr(param0, "dtype", None)
        )

        # Strip all private (underscore-prefixed) keys before serialization,
        # exactly as cli.run_trace does.
        trace_for_json = [
            {k: v for k, v in entry.items() if not k.startswith("_")} for entry in trace
        ]

        return serialize_trace_to_json(
            trace=trace_for_json,
            metadata=metadata,
            attention_metadata=attention_metadata,
            sidecar_refs={},
            tokenizer=tokenizer,
            prompt=config.prompt,
            activation_metadata=activation_metadata,
            activation_sidecar_refs=None,
            model_architecture=model_architecture,
        )
