"""Unit tests for the single-trace TWERA-style neuron attribution."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from llm_token_heatmap.neuron_attribution import compute_neuron_attribution


def _entry(layer_tensors: dict[tuple[int, str], torch.Tensor]) -> dict:
    """A trace entry exposing full activation stats, as generation.py builds."""
    return {"_activation_full_stats": SimpleNamespace(layer_tensors=layer_tensors)}


def test_twera_is_mean_of_activation_times_unembedding_row():
    # hidden=3, vocab=4 unembedding.
    w_u = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # token 0
            [1.0, 0.0, 2.0],  # token 1  <- target at step 0
            [0.0, 3.0, 0.0],  # token 2  <- target at step 1
            [0.0, 0.0, 0.0],  # token 3
        ]
    )
    key = (0, "resid_post")
    step0 = _entry({key: torch.tensor([2.0, 5.0, 1.0])})
    step1 = _entry({key: torch.tensor([4.0, 1.0, 0.0])})

    out = compute_neuron_attribution(
        trace=[step0, step1],
        target_token_ids=[1, 2],
        unembedding=w_u,
        top_n=3,
    )

    assert out is not None
    assert out["method"] == "twera_approx"
    assert out["n_steps"] == 2
    layer = out["layers"][0]
    assert layer["layer"] == 0 and layer["submodule"] == "resid_post"

    # twera_i = mean_t a_i(t) * W_U[target_t, i]
    #   i=0: (2*1 + 4*0)/2 = 1.0
    #   i=1: (5*0 + 1*3)/2 = 1.5
    #   i=2: (1*2 + 0*0)/2 = 1.0
    by_index = {n["index"]: n for n in layer["neurons"]}
    assert by_index[1]["twera"] == 1.5
    assert by_index[0]["twera"] == 1.0
    assert by_index[2]["twera"] == 1.0
    # Ranked descending → neuron 1 first.
    assert layer["neurons"][0]["index"] == 1
    # mean_activation reported alongside: neuron 1 = (5+1)/2 = 3.0
    assert by_index[1]["mean_activation"] == 3.0


def test_skips_non_residual_basis_submodules():
    """A submodule whose vector isn't hidden-dim (e.g. an intermediate width) is
    skipped rather than mis-scored."""
    w_u = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # vocab=2, hidden=2
    key_ok = (0, "resid_post")
    key_bad = (0, "mlp.up_proj")  # width 4 != hidden 2
    entry = _entry(
        {
            key_ok: torch.tensor([1.0, 2.0]),
            key_bad: torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
    )
    out = compute_neuron_attribution(
        trace=[entry], target_token_ids=[1], unembedding=w_u, top_n=4
    )
    assert out is not None
    submodules = {layer["submodule"] for layer in out["layers"]}
    assert submodules == {"resid_post"}


def test_returns_none_without_full_activations():
    w_u = torch.zeros(3, 4)
    # An entry with no `_activation_full_stats` (inline-only capture).
    out = compute_neuron_attribution(
        trace=[{"raw": {}}], target_token_ids=[0], unembedding=w_u
    )
    assert out is None


def test_returns_none_without_unembedding():
    out = compute_neuron_attribution(
        trace=[_entry({(0, "resid_post"): torch.ones(4)})],
        target_token_ids=[0],
        unembedding=None,
    )
    assert out is None
