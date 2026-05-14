"""Unit tests for `trace_to_dataframe`."""

from __future__ import annotations

import torch

from llm_token_heatmap.export import ATTENTION_AGGREGATE_COLUMNS, trace_to_dataframe

EXPECTED_COLUMNS = [
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


def _make_stats(k_used: int, max_k: int = 6) -> dict:
    top_ids = torch.arange(max_k).unsqueeze(0)
    top_probs = torch.linspace(0.5, 0.05, max_k).unsqueeze(0)
    top_logprobs = torch.log(top_probs + 1e-12)
    valid_mask = torch.zeros(1, max_k, dtype=torch.bool)
    valid_mask[0, :k_used] = True

    return {
        "top_ids": top_ids,
        "top_probs": top_probs,
        "top_logprobs": top_logprobs,
        "valid_mask": valid_mask,
        "k_used": torch.tensor([k_used]),
        "entropy": torch.tensor([1.234]),
        "selected_ids": torch.tensor([0]),
        "selected_prob": torch.tensor([0.5]),
        "selected_logprob": torch.tensor([torch.log(torch.tensor(0.5)).item()]),
        "selected_rank": torch.tensor([1]),
    }


def _make_step(
    step_idx: int,
    k_used: int,
    max_k: int = 6,
    *,
    processed_k_used: int | None = None,
) -> dict:
    """Build a synthetic trace entry with the new nested raw/processed schema."""

    raw = _make_stats(k_used=k_used, max_k=max_k)
    processed = _make_stats(
        k_used=processed_k_used if processed_k_used is not None else k_used,
        max_k=max_k,
    )
    return {"step": step_idx, "raw": raw, "processed": processed}


def test_columns_exactly_match_spec(fake_tokenizer):
    trace = [_make_step(0, k_used=3), _make_step(1, k_used=5)]

    df = trace_to_dataframe(trace, fake_tokenizer)

    assert list(df.columns) == EXPECTED_COLUMNS


def test_row_count_per_step_equals_raw_plus_processed_k_used(fake_tokenizer):
    pairs = [(2, 1), (4, 2), (6, 3)]
    trace = [
        _make_step(i, k_used=raw, processed_k_used=proc) for i, (raw, proc) in enumerate(pairs)
    ]

    df = trace_to_dataframe(trace, fake_tokenizer)

    for step, (raw_k, proc_k) in enumerate(pairs):
        rows_for_step = df[df["step"] == step]
        assert len(rows_for_step) == raw_k + proc_k


def test_trace_to_dataframe_row_count(fake_tokenizer):
    k_values = [3, 5, 6, 2]
    trace = [_make_step(i, k_used=k) for i, k in enumerate(k_values)]

    df = trace_to_dataframe(trace, fake_tokenizer)
    assert len(df) == 2 * sum(k_values)


def test_rank_is_one_indexed_and_sequential(fake_tokenizer):
    trace = [_make_step(0, k_used=4)]

    df = trace_to_dataframe(trace, fake_tokenizer)

    raw_rows = df[df["source"] == "raw"]
    processed_rows = df[df["source"] == "processed"]
    assert list(raw_rows["rank"]) == [1, 2, 3, 4]
    assert list(processed_rows["rank"]) == [1, 2, 3, 4]


def test_token_decoded_with_tokenizer(fake_tokenizer):
    trace = [_make_step(0, k_used=3)]

    df = trace_to_dataframe(trace, fake_tokenizer)

    for _, row in df.iterrows():
        assert row["token"] == fake_tokenizer.decode([int(row["token_id"])])
    assert df["selected_token"].iloc[0] == fake_tokenizer.decode([0])


def test_empty_trace_returns_empty_dataframe(fake_tokenizer):
    df = trace_to_dataframe([], fake_tokenizer)

    assert len(df) == 0


def _attention_block_for(layer_count: int) -> list[dict]:
    """Build a synthetic per-step attention payload with `layer_count` layers."""

    return [
        {
            "layer": i,
            "entropy": 1.0 + 0.1 * i,
            "self_weight": 0.2 + 0.05 * i,
            "bos_weight": 0.1 + 0.05 * i,
            "top_positions": [{"position": 0, "weight": 0.5}],
            "q_norm": 1.0,
            "k_norm": 1.0,
            "v_norm": 1.0,
            "qk_alignment_angle": 30.0,
        }
        for i in range(layer_count)
    ]


def test_export_omits_attention_columns_when_attention_not_captured(fake_tokenizer):
    trace = [_make_step(0, k_used=3), _make_step(1, k_used=2)]

    df = trace_to_dataframe(trace, fake_tokenizer)

    for col in ATTENTION_AGGREGATE_COLUMNS:
        assert col not in df.columns
    assert list(df.columns) == EXPECTED_COLUMNS


def test_export_adds_aggregate_columns_only_when_attention_captured(fake_tokenizer):
    trace = [
        _make_step(0, k_used=3),
        _make_step(1, k_used=2),
    ]
    # Attach attention only to step 1.
    trace[1]["attention"] = _attention_block_for(2)

    df = trace_to_dataframe(trace, fake_tokenizer)

    for col in ATTENTION_AGGREGATE_COLUMNS:
        assert col in df.columns

    step1_rows = df[df["step"] == 1]
    expected_entropy = sum(layer["entropy"] for layer in trace[1]["attention"]) / 2
    assert step1_rows["attention_entropy_mean"].iloc[0] == expected_entropy

    # Rows for the step without attention are missing the value (NaN).
    step0_rows = df[df["step"] == 0]
    assert step0_rows["attention_entropy_mean"].isna().all()


def test_export_emits_source_column(make_run) -> None:
    trace, tokenizer = make_run(n_steps=3, top_p=0.5)
    df = trace_to_dataframe(trace, tokenizer)
    assert "source" in df.columns


def test_source_values_only_raw_or_processed(make_run) -> None:
    trace, tokenizer = make_run(n_steps=3, top_p=0.5)
    df = trace_to_dataframe(trace, tokenizer)
    assert set(df["source"].unique()) == {"raw", "processed"}


def test_row_count_equals_raw_plus_processed_k_used(make_run) -> None:
    trace, tokenizer = make_run(n_steps=4, top_p=0.5)
    df = trace_to_dataframe(trace, tokenizer)

    for entry in trace:
        step = int(entry["step"])
        raw_k = int(entry["raw"]["k_used"][0])
        proc_k = int(entry["processed"]["k_used"][0])
        rows_for_step = df[df["step"] == step]
        assert len(rows_for_step) == raw_k + proc_k


def test_raw_equals_processed_when_no_filter(make_run) -> None:
    trace, _tokenizer = make_run(
        n_steps=3,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
    )
    for entry in trace:
        raw_probs = entry["raw"]["top_probs"]
        processed_probs = entry["processed"]["top_probs"]
        assert torch.allclose(raw_probs, processed_probs, atol=1e-6)


def test_processed_k_le_raw_k_with_top_p(make_run) -> None:
    trace, _tokenizer = make_run(
        n_steps=4,
        temperature=1.0,
        top_p=0.5,
        top_k=0,
        min_k=1,
        max_k=32,
    )
    for entry in trace:
        raw_k = int(entry["raw"]["k_used"][0])
        proc_k = int(entry["processed"]["k_used"][0])
        assert proc_k <= raw_k
