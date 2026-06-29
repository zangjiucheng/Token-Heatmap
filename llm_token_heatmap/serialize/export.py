"""Convert generation traces into pandas DataFrames for analysis and CSV export."""

from typing import Any

import pandas as pd

ATTENTION_AGGREGATE_COLUMNS = (
    "attention_entropy_mean",
    "attention_self_weight_mean",
    "attention_bos_weight_mean",
)


def _attention_aggregates(entry: dict[str, Any]) -> dict[str, float] | None:
    """Return per-step attention aggregates if the step carries an `attention` block.

    The full per-layer attention payload is intentionally NOT flattened into
    the long-format CSV (it would explode the row count); only the mean over
    captured layers of each scalar is surfaced. The JSON path carries the full
    per-layer detail.
    """

    layers = entry.get("attention")
    if not layers:
        return None
    n = len(layers)
    if n == 0:
        return None
    entropy_sum = 0.0
    self_sum = 0.0
    bos_sum = 0.0
    for layer in layers:
        entropy_sum += float(layer.get("entropy", 0.0))
        self_sum += float(layer.get("self_weight", 0.0))
        bos_sum += float(layer.get("bos_weight", 0.0))
    return {
        "attention_entropy_mean": entropy_sum / n,
        "attention_self_weight_mean": self_sum / n,
        "attention_bos_weight_mean": bos_sum / n,
    }


def _rows_for_source(
    step: int,
    source: str,
    stats: dict[str, Any],
    tokenizer: Any,
    batch_index: int,
    attention_aggregates: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    top_ids = stats["top_ids"][batch_index]
    top_probs = stats["top_probs"][batch_index]
    top_logprobs = stats["top_logprobs"][batch_index]
    valid_mask = stats["valid_mask"][batch_index]
    k_used = int(stats["k_used"][batch_index])
    entropy = float(stats["entropy"][batch_index])

    selected_id = int(stats["selected_ids"][batch_index])
    selected_prob = float(stats["selected_prob"][batch_index])
    selected_logprob = float(stats["selected_logprob"][batch_index])
    selected_rank = int(stats["selected_rank"][batch_index])
    selected_token = tokenizer.decode([selected_id], skip_special_tokens=False)

    rows: list[dict[str, Any]] = []
    for rank_idx in range(top_ids.shape[0]):
        if not bool(valid_mask[rank_idx]):
            continue
        token_id = int(top_ids[rank_idx])
        row: dict[str, Any] = {
            "step": step,
            "source": source,
            "rank": rank_idx + 1,
            "token_id": token_id,
            "token": tokenizer.decode([token_id], skip_special_tokens=False),
            "prob": float(top_probs[rank_idx]),
            "logprob": float(top_logprobs[rank_idx]),
            "selected_token_id": selected_id,
            "selected_token": selected_token,
            "selected_prob": selected_prob,
            "selected_logprob": selected_logprob,
            "selected_rank": selected_rank,
            "entropy": entropy,
            "k_used": k_used,
        }
        if attention_aggregates is not None:
            row.update(attention_aggregates)
        rows.append(row)
    return rows


def trace_to_dataframe(
    trace: list[dict[str, Any]],
    tokenizer: Any,
    batch_index: int = 0,
) -> pd.DataFrame:
    """Flatten a per-step trace into a long-format DataFrame.

    Each row represents one candidate token at one generation step, for one
    distribution ``source`` (``"raw"`` or ``"processed"``). Only candidates
    within the adaptive ``k_used`` (i.e. ``valid_mask == True``) are included.

    Args:
        trace: List of per-step dicts produced by ``generate_with_adaptive_probe``;
            each dict has ``step``, ``raw`` and ``processed`` keys.
        tokenizer: HuggingFace tokenizer used for decoding token IDs.
        batch_index: Which batch row to flatten. Defaults to 0.

    Returns:
        DataFrame with one row per (step, source, candidate rank).
    """
    rows: list[dict[str, Any]] = []

    for entry in trace:
        step = int(entry["step"])
        attention_aggregates = _attention_aggregates(entry)
        rows.extend(
            _rows_for_source(
                step, "raw", entry["raw"], tokenizer, batch_index, attention_aggregates
            )
        )
        rows.extend(
            _rows_for_source(
                step,
                "processed",
                entry["processed"],
                tokenizer,
                batch_index,
                attention_aggregates,
            )
        )

    return pd.DataFrame(rows)
