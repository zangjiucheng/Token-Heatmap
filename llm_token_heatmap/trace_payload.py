"""Build schema-conformant JSON trace payloads from in-memory probe output.

The library's probe / generation functions return their natural in-memory
shape: parallel `top_*` arrays, batch-dimensioned tensors for scalars, and a
flat dict per (step, source). The on-disk JSON format documented at
``docs/web/trace.schema.json`` is different: per-step ``selected`` block,
per-source ``candidates: [{rank, token_id, token, prob, logprob}]`` list,
top-level ``tokens`` block, ISO timestamp in ``metadata.generated_at``, etc.

This module is the single source of truth for that transformation so the
CLI, the runnable examples, and any future producer all emit identical
schema-valid JSON.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from llm_token_heatmap import SCHEMA_VERSION

# Keys the trace schema allows under `metadata`. Anything else
# (e.g. our internal capture_* flags) is dropped before serialization
# because the schema sets `additionalProperties: false`.
_ALLOWED_METADATA_KEYS = frozenset(
    {
        "model",
        "prompt",
        "system_prompt",
        "use_chat_template",
        "generated_text",
        "generated_at",
        "generation_params",
        "probe_config",
        "device",
        "dtype",
        "vocab_size",
    }
)


def tensor_to_jsonable(value: Any) -> Any:
    """Recursively convert tensors / dataclasses inside ``value`` into JSON-safe primitives."""
    import torch  # lazy: keep argparse-only paths free of torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return tensor_to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): tensor_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    return value


def _scalar(value: Any) -> Any:
    """Pull a Python scalar out of a 1-element tensor or list."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def build_model_architecture(model: Any, *, dtype: Any = None) -> dict[str, Any] | None:
    """Self-contained architecture summary read straight from ``model.config``.

    Unlike ``attention_metadata`` / ``activation_metadata`` (which only exist
    when those probes ran), this is captured on every trace so the web "Model"
    tab can render a structural overview without any capture flags. Every field
    is best-effort: anything ``model.config`` doesn't expose is simply omitted,
    so exotic architectures still produce a (possibly partial) block rather than
    raising. Returns ``None`` when nothing could be read.
    """
    cfg = getattr(model, "config", None)
    arch: dict[str, Any] = {}

    def _put_int(key: str, value: Any) -> None:
        if value is not None:
            try:
                arch[key] = int(value)
            except (TypeError, ValueError):
                pass

    if cfg is not None:
        architectures = getattr(cfg, "architectures", None)
        if architectures:
            arch["architecture"] = str(architectures[0])
        model_type = getattr(cfg, "model_type", None)
        if model_type:
            arch["model_type"] = str(model_type)
        _put_int("num_layers", getattr(cfg, "num_hidden_layers", None))
        _put_int("hidden_size", getattr(cfg, "hidden_size", None))
        _put_int("num_attention_heads", getattr(cfg, "num_attention_heads", None))
        _put_int(
            "num_key_value_heads",
            getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", None)),
        )
        _put_int("head_dim", getattr(cfg, "head_dim", None))
        _put_int("intermediate_size", getattr(cfg, "intermediate_size", None))
        _put_int("vocab_size", getattr(cfg, "vocab_size", None))
        _put_int("max_position_embeddings", getattr(cfg, "max_position_embeddings", None))
        rope = getattr(cfg, "rope_theta", None)
        if rope is not None:
            try:
                arch["rope_theta"] = float(rope)
            except (TypeError, ValueError):
                pass
        tie = getattr(cfg, "tie_word_embeddings", None)
        if tie is not None:
            arch["tie_word_embeddings"] = bool(tie)
        # Derive head_dim when the config omits it (hidden_size / heads).
        if "head_dim" not in arch and arch.get("hidden_size") and arch.get("num_attention_heads"):
            arch["head_dim"] = arch["hidden_size"] // arch["num_attention_heads"]

    # Total parameter count — cheap and the single most useful "how big" number.
    try:
        total = sum(int(p.numel()) for p in model.parameters())
        if total > 0:
            arch["num_parameters"] = total
    except Exception:  # noqa: BLE001 — never let a param sweep break serialization
        pass

    if dtype is not None:
        arch["dtype"] = str(dtype).replace("torch.", "")

    return arch or None


def _seq(value: Any) -> list[Any]:
    """Pull a 1-D list out of either a [batch, K] tensor or a nested list."""
    import torch

    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.ndim == 2:
            return t[0].tolist()
        return t.tolist()
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return value[0]
        return value
    return list(value)


def _clamp_unit(value: float) -> float:
    """Clamp a probability into [0.0, 1.0].

    Why: ``torch.cumsum`` over the kept top-k probabilities accumulates float32
    roundoff, and after ``torch.softmax`` renormalizes the post-top-p survivors
    the cumulative mass can land at ``1.0 + ~1e-7``. The schema's
    ``top_mass_used`` / ``selected_prob`` / candidate ``prob`` are all bounded
    ``[0, 1]``, so a single overshoot from float math would fail validation.
    Clamping at the JSON boundary keeps in-memory numerics honest for plotting
    and dataframe export while making the serialized trace schema-conformant.
    """

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def distribution_payload(stats: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Convert a probe stats dict into the schema's ``Distribution`` shape."""
    top_ids = _seq(stats["top_ids"])
    top_probs = _seq(stats["top_probs"])
    top_logprobs = _seq(stats["top_logprobs"])
    valid_mask = _seq(stats["valid_mask"])
    candidates: list[dict[str, Any]] = []
    rank = 0
    for token_id, prob, logprob, ok in zip(
        top_ids, top_probs, top_logprobs, valid_mask, strict=False
    ):
        if not bool(ok):
            continue
        rank += 1
        candidates.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "prob": _clamp_unit(float(prob)),
                "logprob": float(logprob),
            }
        )
    return {
        "k_used": int(_scalar(stats["k_used"])),
        "entropy": float(_scalar(stats["entropy"])),
        "top_mass_used": _clamp_unit(float(_scalar(stats["top_mass_used"]))),
        "selected_prob": _clamp_unit(float(_scalar(stats["selected_prob"]))),
        "selected_logprob": float(_scalar(stats["selected_logprob"])),
        "selected_rank": int(_scalar(stats["selected_rank"])),
        "candidates": candidates,
    }


def selected_token_payload(stats: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    token_id = int(_scalar(stats["selected_ids"]))
    return {
        "token_id": token_id,
        "token": tokenizer.decode([token_id], skip_special_tokens=False),
    }


def logit_lens_payload(layers: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Convert per-step ``LogitLensLayerStats`` dicts into the schema's ``LogitLensLayer`` shape.

    The generation loop emits ``layers[i]`` with parallel ``top_k_token_ids`` /
    ``top_k_probs`` / ``top_k_logprobs`` arrays (the dataclass shape, kept that
    way so plotting helpers can index by tensor). The trace schema instead
    wants a flat ``top_k`` array of decoded ``LogitLensCandidate`` objects,
    with ``additionalProperties: false`` — so the parallel fields would fail
    validation even if we kept them alongside ``top_k``.
    """

    out: list[dict[str, Any]] = []
    for layer in layers:
        token_ids = _seq(layer["top_k_token_ids"])
        probs = _seq(layer["top_k_probs"])
        logprobs = _seq(layer["top_k_logprobs"])
        candidates: list[dict[str, Any]] = []
        for rank, (tid, prob, logprob) in enumerate(
            zip(token_ids, probs, logprobs, strict=False), start=1
        ):
            candidates.append(
                {
                    "rank": rank,
                    "token_id": int(tid),
                    "token": tokenizer.decode([int(tid)], skip_special_tokens=False),
                    "prob": _clamp_unit(float(prob)),
                    "logprob": float(logprob),
                }
            )
        out.append(
            {
                "layer_idx": int(_scalar(layer["layer_idx"])),
                "top_k": candidates,
                "entropy": float(_scalar(layer["entropy"])),
                "selected_token_rank": int(_scalar(layer["selected_token_rank"])),
                "selected_token_prob": _clamp_unit(float(_scalar(layer["selected_token_prob"]))),
            }
        )
    return out


def prompt_tokens_payload(prompt: str, tokenizer: Any) -> dict[str, Any]:
    """Tokenize the prompt the same way the generation loop does (no chat template)
    and decode each ID for surface display."""
    encoded = tokenizer(prompt, return_tensors=None)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    # The non-tensor call returns a list of ints (no batch dim).
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    prompt_token_ids = [int(i) for i in ids]
    prompt_tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in prompt_token_ids]
    return {"prompt_token_ids": prompt_token_ids, "prompt_tokens": prompt_tokens}


def serialize_trace_to_json(
    *,
    trace: list[dict[str, Any]],
    metadata: dict[str, Any],
    attention_metadata: dict[str, Any] | None,
    sidecar_refs: dict[int, str],
    tokenizer: Any,
    prompt: str,
    activation_metadata: dict[str, Any] | None = None,
    activation_sidecar_refs: dict[int, str] | None = None,
    model_architecture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe payload conforming to ``docs/web/trace.schema.json``.

    When ``activation_metadata`` is provided, the per-step ``token_id`` /
    ``decoded_text_offset`` / ``activations`` fields are populated from the
    trace and the top-level ``activation_metadata`` block is included. The
    projected activation subset of the resulting payload (top-level
    ``schema_version`` + ``activation_metadata`` + ``steps[i].step`` +
    ``token_id`` + ``decoded_text_offset`` + ``activations``) validates
    against ``docs/web/activation.schema.json``.
    """
    steps: list[dict[str, Any]] = []
    for entry in trace:
        step_idx = int(entry["step"])
        raw_stats = entry["raw"]
        processed_stats = entry["processed"]
        selected_payload = selected_token_payload(raw_stats, tokenizer)
        step_payload: dict[str, Any] = {
            "step": step_idx,
            "selected": selected_payload,
            "raw": distribution_payload(raw_stats, tokenizer),
            "processed": distribution_payload(processed_stats, tokenizer),
        }
        if "attention" in entry:
            step_payload["attention"] = tensor_to_jsonable(entry["attention"])
        if "logit_lens" in entry:
            step_payload["logit_lens"] = logit_lens_payload(entry["logit_lens"], tokenizer)
        ref = sidecar_refs.get(step_idx)
        step_payload["attention_sidecar_ref"] = ref
        if activation_metadata is not None:
            step_payload["token_id"] = int(selected_payload["token_id"])
            step_payload["decoded_text_offset"] = int(entry.get("decoded_text_offset", 0))
            step_payload["activations"] = tensor_to_jsonable(entry.get("activations", []))
            if activation_sidecar_refs is not None:
                step_payload["activation_sidecar_ref"] = activation_sidecar_refs.get(step_idx)
        steps.append(step_payload)

    md = {k: v for k, v in metadata.items() if k in _ALLOWED_METADATA_KEYS}
    md.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": md,
        "tokens": prompt_tokens_payload(prompt, tokenizer),
        "steps": steps,
    }
    if attention_metadata is not None:
        payload["attention_metadata"] = attention_metadata
    if activation_metadata is not None:
        payload["activation_metadata"] = activation_metadata
    if model_architecture is not None:
        payload["model_architecture"] = model_architecture
    return payload


def project_activation_subset(
    payload: dict[str, Any],
    *,
    activation_schema_version: str = "1.0.0",
) -> dict[str, Any]:
    """Extract the activation subset from a trace payload.

    Returns a dict shaped like ``docs/web/activation.schema.json``: top-level
    ``schema_version`` + ``activation_metadata`` + ``steps`` where each step
    carries ``step`` / ``token_id`` / ``decoded_text_offset`` / ``activations``.
    Used by the CLI activation tests to validate the inline trace against the
    activation schema without duplicating the file.
    """

    activation_metadata = payload.get("activation_metadata")
    if activation_metadata is None:
        raise ValueError(
            "payload has no `activation_metadata`; ActivationProbe was likely not attached."
        )
    projected_steps = []
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
        "schema_version": activation_schema_version,
        "activation_metadata": activation_metadata,
        "steps": projected_steps,
    }
