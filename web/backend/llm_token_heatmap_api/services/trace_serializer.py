"""Convert in-memory generation traces (or CSVs of them) into schema-conformant JSON.

The library's ``generate_with_adaptive_probe`` returns a list of per-step dicts
whose tensors live on whatever device the model ran on. The web payload, by
contrast, must be plain JSON whose shape matches ``docs/web/trace.schema.json``.

This module also handles the inverse problem for ``/trace/convert-csv``:
re-hydrating the long-format CSV produced by
``llm_token_heatmap.export.trace_to_dataframe`` into the same JSON shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import torch

from llm_token_heatmap_api import SCHEMA_VERSION
from llm_token_heatmap_api.errors import InvalidCsvError

REQUIRED_CSV_COLUMNS: list[str] = [
    "step",
    "source",
    "rank",
    "token_id",
    "token",
    "prob",
    "logprob",
    "selected_token_id",
    "selected_token",
    "selected_prob",
    "selected_logprob",
    "selected_rank",
    "entropy",
    "k_used",
]


def _decode(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _distribution_from_stats(
    stats: dict[str, Any], tokenizer: Any, batch_index: int = 0
) -> dict[str, Any]:
    """Slice the probe stats for one (step, source) into a JSON distribution dict."""
    top_ids = stats["top_ids"][batch_index]
    top_probs = stats["top_probs"][batch_index]
    top_logprobs = stats["top_logprobs"][batch_index]
    valid_mask = stats["valid_mask"][batch_index]
    k_used = int(stats["k_used"][batch_index])

    candidates: list[dict[str, Any]] = []
    kept = 0
    for rank_idx in range(top_ids.shape[0]):
        if not bool(valid_mask[rank_idx]):
            continue
        token_id = int(top_ids[rank_idx])
        kept += 1
        candidates.append(
            {
                "rank": kept,
                "token_id": token_id,
                "token": _decode(tokenizer, token_id),
                "prob": float(top_probs[rank_idx]),
                "logprob": float(top_logprobs[rank_idx]),
            }
        )

    return {
        "k_used": k_used,
        "entropy": float(stats["entropy"][batch_index]),
        "top_mass_used": float(stats["top_mass_used"][batch_index]),
        "selected_prob": float(stats["selected_prob"][batch_index]),
        "selected_logprob": float(stats["selected_logprob"][batch_index]),
        "selected_rank": int(stats["selected_rank"][batch_index]),
        "candidates": candidates,
    }


def trace_to_json(
    *,
    trace: list[dict[str, Any]],
    tokenizer: Any,
    model_name: str,
    prompt: str,
    generated_text: str,
    prompt_token_ids: list[int],
    generation_params: dict[str, Any],
    probe_config: dict[str, Any],
    use_chat_template: bool,
    system_prompt: str | None,
    device: str,
    dtype: str,
    vocab_size: int | None,
    batch_index: int = 0,
) -> dict[str, Any]:
    """Build a schema-conformant JSON payload from a generation trace."""
    steps: list[dict[str, Any]] = []
    for entry in trace:
        step_index = int(entry["step"])
        selected_id = int(entry["raw"]["selected_ids"][batch_index])
        steps.append(
            {
                "step": step_index,
                "selected": {
                    "token_id": selected_id,
                    "token": _decode(tokenizer, selected_id),
                },
                "raw": _distribution_from_stats(entry["raw"], tokenizer, batch_index),
                "processed": _distribution_from_stats(
                    entry["processed"], tokenizer, batch_index
                ),
            }
        )

    metadata: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "use_chat_template": use_chat_template,
        "generated_text": generated_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "dtype": dtype,
        "generation_params": generation_params,
        "probe_config": probe_config,
    }
    if vocab_size is not None:
        metadata["vocab_size"] = int(vocab_size)

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "tokens": {
            "prompt_token_ids": [int(tid) for tid in prompt_token_ids],
            "prompt_tokens": [_decode(tokenizer, tid) for tid in prompt_token_ids],
        },
        "steps": steps,
    }


def csv_to_trace_json(csv_bytes: bytes) -> dict[str, Any]:
    """Reconstruct a JSON trace payload from a CSV produced by ``trace_to_dataframe``.

    Metadata that can't be recovered from the CSV is filled with neutral defaults
    so the result still validates against the schema.
    """
    import io

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except (pd.errors.ParserError, ValueError, UnicodeDecodeError) as exc:
        raise InvalidCsvError(f"Could not parse CSV: {exc}") from None

    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise InvalidCsvError(
            "CSV is missing required columns.",
            details=missing,
        )

    if df.empty:
        raise InvalidCsvError("CSV contains no rows.")

    steps: list[dict[str, Any]] = []
    max_k_seen = 0
    min_k_seen: int | None = None
    mass_threshold = 0.0

    grouped = df.groupby("step", sort=True)
    for step_index, step_df in grouped:
        sources: dict[str, dict[str, Any]] = {}
        selected: dict[str, Any] | None = None

        for source in ("raw", "processed"):
            source_df = step_df[step_df["source"] == source].sort_values("rank")
            if source_df.empty:
                raise InvalidCsvError(
                    f"Step {int(step_index)} is missing rows for source '{source}'."
                )

            first_row = source_df.iloc[0]
            k_used = int(first_row["k_used"])
            max_k_seen = max(max_k_seen, k_used)
            min_k_seen = k_used if min_k_seen is None else min(min_k_seen, k_used)
            mass = float(source_df["prob"].sum())
            mass_threshold = max(mass_threshold, mass)

            candidates = [
                {
                    "rank": int(row["rank"]),
                    "token_id": int(row["token_id"]),
                    "token": str(row["token"]),
                    "prob": float(row["prob"]),
                    "logprob": float(row["logprob"]),
                }
                for _, row in source_df.iterrows()
            ]

            sources[source] = {
                "k_used": k_used,
                "entropy": float(first_row["entropy"]),
                "top_mass_used": min(1.0, max(0.0, mass)),
                "selected_prob": float(first_row["selected_prob"]),
                "selected_logprob": float(first_row["selected_logprob"]),
                "selected_rank": int(first_row["selected_rank"]),
                "candidates": candidates,
            }

            if selected is None:
                selected = {
                    "token_id": int(first_row["selected_token_id"]),
                    "token": str(first_row["selected_token"]),
                }

        steps.append(
            {
                "step": int(step_index),
                "selected": selected,
                "raw": sources["raw"],
                "processed": sources["processed"],
            }
        )

    # The CSV does not preserve probe min_k/max_k/threshold or generation
    # params, so we provide schema-valid placeholders derived from the data.
    effective_min_k = min_k_seen if min_k_seen is not None else 1
    effective_max_k = max(max_k_seen, effective_min_k)
    effective_mass = min(1.0, max(0.01, mass_threshold))

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "model": "unknown",
            "prompt": "",
            "system_prompt": None,
            "use_chat_template": False,
            "generated_text": "",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_params": {
                "max_new_tokens": max(1, len(steps)),
                "temperature": 1.0,
                "top_p": 1.0,
                "sample_top_k": 0,
            },
            "probe_config": {
                "min_k": effective_min_k,
                "max_k": effective_max_k,
                "mass_threshold": effective_mass,
            },
        },
        "tokens": {
            "prompt_token_ids": [],
            "prompt_tokens": [],
        },
        "steps": steps,
    }


def extract_prompt_token_ids(tokenizer: Any, prompt: str, *, use_chat_template: bool,
                              system_prompt: str | None) -> list[int]:
    """Return the token ids the model actually saw before step 0.

    Mirrors the chat-template / plain-tokenize branching in
    ``generate_with_adaptive_probe`` so the JSON's ``tokens.prompt_*`` arrays
    match what was fed to the model.
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict):
            encoded = encoded["input_ids"]
    else:
        out = tokenizer(prompt, return_tensors="pt")
        encoded = out.input_ids if hasattr(out, "input_ids") else out["input_ids"]

    if torch.is_tensor(encoded):
        ids = encoded.flatten().tolist()
    else:
        ids = list(encoded)
    return [int(i) for i in ids]
