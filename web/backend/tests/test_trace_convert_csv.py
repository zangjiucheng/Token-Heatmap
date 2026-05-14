"""Tests for ``POST /trace/convert-csv``."""

from __future__ import annotations

import io

import pandas as pd


def _build_minimal_csv() -> bytes:
    """Build a minimal CSV with two steps × two sources × two candidates each."""
    rows = []
    for step in (0, 1):
        for source in ("raw", "processed"):
            for rank, prob, logprob, token_id in (
                (1, 0.7, -0.36, 10 + step),
                (2, 0.3, -1.20, 20 + step),
            ):
                rows.append(
                    {
                        "step": step,
                        "source": source,
                        "rank": rank,
                        "token_id": token_id,
                        "token": f"<tok:{token_id}>",
                        "prob": prob,
                        "logprob": logprob,
                        "selected_token_id": 10 + step,
                        "selected_token": f"<tok:{10 + step}>",
                        "selected_prob": 0.7,
                        "selected_logprob": -0.36,
                        "selected_rank": 1,
                        "entropy": 0.5,
                        "k_used": 2,
                    }
                )
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_convert_csv_round_trip(client) -> None:
    csv_bytes = _build_minimal_csv()
    response = client.post(
        "/trace/convert-csv",
        files={"file": ("trace.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["schema_version"] == "2.0.0"
    assert len(payload["steps"]) == 2

    step0 = payload["steps"][0]
    assert step0["step"] == 0
    assert step0["selected"]["token_id"] == 10
    assert len(step0["raw"]["candidates"]) == 2
    assert len(step0["processed"]["candidates"]) == 2
    assert step0["raw"]["candidates"][0]["rank"] == 1
    assert step0["raw"]["candidates"][0]["token_id"] == 10
    assert step0["raw"]["k_used"] == 2


def test_convert_csv_missing_columns(client) -> None:
    df = pd.DataFrame(
        [
            {
                "step": 0,
                "source": "raw",
                "rank": 1,
                "token_id": 0,
                "token": "<tok:0>",
                "prob": 1.0,
                "logprob": 0.0,
                "selected_token_id": 0,
                "selected_token": "<tok:0>",
                "selected_prob": 1.0,
                "selected_logprob": 0.0,
                "selected_rank": 1,
                "k_used": 1,
                # Note: 'entropy' is intentionally absent.
            }
        ]
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    response = client.post(
        "/trace/convert-csv",
        files={"file": ("bad.csv", buf.getvalue(), "text/csv")},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["kind"] == "invalid_csv"
    assert body["error"]["details"] == ["entropy"]


def test_convert_csv_validates_against_schema(client) -> None:
    import json

    import jsonschema

    csv_bytes = _build_minimal_csv()
    response = client.post(
        "/trace/convert-csv",
        files={"file": ("trace.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text

    schema_response = client.get("/schema")
    schema = json.loads(schema_response.content)
    jsonschema.validate(response.json(), schema)
