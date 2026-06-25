"""Unit tests for component-level interventions.

The causal-validation contract: ablating a block that the model uses changes the
next-token distribution (KL > 0); a no-op scale (factor 1.0) leaves it unchanged
(KL ~ 0). Runs on the tiny test model — no downloads.
"""

from __future__ import annotations

import torch

from llm_token_heatmap.intervention import run_intervention
from tests.fixtures.tiny_attention_model import build_tiny_model


def _ids(model, seq_len: int = 6, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, model.config.vocab_size, (seq_len,), generator=g
    ).tolist()


def test_ablation_changes_the_distribution():
    model = build_tiny_model(num_hidden_layers=3)
    model.eval()
    input_ids = _ids(model)

    out = run_intervention(
        model,
        input_ids=input_ids,
        interventions=[{"layer": 1, "component": "attn", "op": "zero", "factor": 0.0}],
        top_k=5,
    )
    assert set(out) >= {"baseline", "patched", "diff", "target_token_id"}
    assert len(out["baseline"]["top"]) == 5
    # Zeroing a used block must move the output distribution.
    assert out["diff"]["kl"] > 1e-6
    assert "top_flips" in out["diff"]
    # Top entries are sorted by descending probability.
    probs = [t["prob"] for t in out["baseline"]["top"]]
    assert probs == sorted(probs, reverse=True)


def test_identity_scale_is_a_noop():
    model = build_tiny_model(num_hidden_layers=3)
    model.eval()
    input_ids = _ids(model, seed=1)

    out = run_intervention(
        model,
        input_ids=input_ids,
        interventions=[
            {"layer": 0, "component": "mlp", "op": "scale", "factor": 1.0}
        ],
        top_k=5,
    )
    # Scaling by 1.0 changes nothing → distributions identical.
    assert out["diff"]["kl"] < 1e-6
    assert out["diff"]["target_prob_delta"] == 0.0 or abs(out["diff"]["target_prob_delta"]) < 1e-6
    assert out["diff"]["top_flips"] == []


def test_mlp_ablation_also_moves_output():
    model = build_tiny_model(num_hidden_layers=2)
    model.eval()
    input_ids = _ids(model, seed=2)
    out = run_intervention(
        model,
        input_ids=input_ids,
        interventions=[{"layer": 0, "component": "mlp", "op": "zero"}],
        top_k=4,
    )
    assert out["diff"]["kl"] > 1e-6


def test_target_token_id_is_tracked_when_given():
    model = build_tiny_model(num_hidden_layers=2)
    model.eval()
    input_ids = _ids(model, seed=3)
    out = run_intervention(
        model,
        input_ids=input_ids,
        interventions=[{"layer": 1, "component": "attn", "op": "zero"}],
        target_token_id=7,
        top_k=4,
    )
    assert out["target_token_id"] == 7


def test_out_of_range_layer_is_ignored_as_noop():
    model = build_tiny_model(num_hidden_layers=2)
    model.eval()
    input_ids = _ids(model, seed=4)
    out = run_intervention(
        model,
        input_ids=input_ids,
        interventions=[{"layer": 99, "component": "attn", "op": "zero"}],
        top_k=4,
    )
    # No valid hook attached → distribution unchanged.
    assert out["diff"]["kl"] < 1e-6


def test_per_head_ablation_changes_distribution():
    model = build_tiny_model(num_hidden_layers=2, num_attention_heads=4, head_dim=4)
    model.eval()
    input_ids = _ids(model, seed=5)
    out = run_intervention(
        model,
        input_ids=input_ids,
        interventions=[{"layer": 1, "component": "head", "head": 2, "op": "zero"}],
        top_k=4,
    )
    assert out["diff"]["kl"] > 1e-6


def test_single_head_differs_from_whole_attn_block():
    model = build_tiny_model(num_hidden_layers=2, num_attention_heads=4, head_dim=4)
    model.eval()
    ids = _ids(model, seed=6)
    head = run_intervention(
        model,
        input_ids=ids,
        interventions=[{"layer": 1, "component": "head", "head": 0, "op": "zero"}],
    )
    block = run_intervention(
        model,
        input_ids=ids,
        interventions=[{"layer": 1, "component": "attn", "op": "zero"}],
    )
    # Both move the output, but ablating one head is not the same as the block.
    assert head["diff"]["kl"] > 1e-6
    assert block["diff"]["kl"] > 1e-6
    assert head["diff"]["kl"] != block["diff"]["kl"]


def test_out_of_range_head_is_noop():
    model = build_tiny_model(num_hidden_layers=2, num_attention_heads=4, head_dim=4)
    model.eval()
    ids = _ids(model, seed=7)
    out = run_intervention(
        model,
        input_ids=ids,
        interventions=[{"layer": 1, "component": "head", "head": 99, "op": "zero"}],
    )
    assert out["diff"]["kl"] < 1e-6
