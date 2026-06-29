"""Tests for the contribution-first head role taxonomy."""

from __future__ import annotations

from llm_token_heatmap.analysis.head_roles import compute_head_roles


def _trace() -> dict:
    """A minimal trace with one captured layer and four hand-tuned heads:
    a sink, a worker, an induction head, and a local head."""
    return {
        "steps": [
            {
                "attention": [
                    {
                        "layer": 0,
                        "per_head": [
                            {"bos_weight": 0.99, "self_weight": 0.0, "induction": 0.0},
                            {"bos_weight": 0.0, "self_weight": 0.1, "induction": 0.0},
                            {"bos_weight": 0.1, "self_weight": 0.0, "induction": 0.6},
                            {"bos_weight": 0.0, "self_weight": 0.9, "induction": 0.0},
                        ],
                    }
                ]
            }
        ],
        "direct_logit_attribution": {
            "steps": [
                {
                    "layers": [
                        {
                            "layer": 0,
                            "heads": [
                                {"head": 0, "attn": 0.01},  # sink: ~no contribution
                                {"head": 1, "attn": 1.50},  # worker: big contribution
                                {"head": 2, "attn": 0.20},  # induction head
                                {"head": 3, "attn": 0.00},  # local
                            ],
                        }
                    ]
                }
            ]
        },
    }


def test_classifies_each_role() -> None:
    hr = compute_head_roles(_trace())
    assert hr is not None
    roles = {(h["layer"], h["head"]): h["role"] for h in hr["heads"]}
    assert roles[(0, 0)] == "sink"
    assert roles[(0, 1)] == "worker"
    assert roles[(0, 2)] == "induction"
    assert roles[(0, 3)] == "local"


def test_worker_beats_sink_when_it_also_contributes() -> None:
    """A head that attends to BOS but still writes a big DLA is a worker, not a
    sink — contribution wins (the -0.31 corr is a tendency, not a rule)."""
    tr = _trace()
    tr["steps"][0]["attention"][0]["per_head"][1]["bos_weight"] = 0.95
    hr = compute_head_roles(tr)
    roles = {(h["layer"], h["head"]): h["role"] for h in hr["heads"]}
    assert roles[(0, 1)] == "worker"


def test_summary_counts_and_top_lists() -> None:
    hr = compute_head_roles(_trace())
    assert hr["summary"]["counts"]["sink"] == 1
    assert hr["summary"]["counts"]["worker"] == 1
    top_workers = hr["summary"]["top_workers"]
    assert top_workers and top_workers[0]["head"] == 1  # biggest |DLA| first


def test_returns_none_without_dla() -> None:
    tr = _trace()
    del tr["direct_logit_attribution"]
    assert compute_head_roles(tr) is None


def test_returns_none_without_per_head() -> None:
    tr = _trace()
    for entry in tr["steps"][0]["attention"]:
        entry.pop("per_head")
    assert compute_head_roles(tr) is None
