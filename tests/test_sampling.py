"""Unit tests for `sample_next_token`."""

from __future__ import annotations

import torch

from llm_token_heatmap.sampling import sample_next_token


def _logits_with_argmax_at(index: int, vocab: int = 16) -> torch.Tensor:
    logits = torch.zeros(1, vocab)
    logits[0, index] = 10.0
    return logits


def test_temperature_near_zero_picks_argmax():
    """A near-zero temperature concentrates all mass on the argmax."""

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    argmax = int(torch.argmax(logits, dim=-1))

    token, _ = sample_next_token(logits, temperature=1e-4, top_p=1.0, top_k=0)
    assert int(token[0]) == argmax


def test_top_k_one_always_picks_argmax():
    logits = torch.tensor([[0.5, 1.2, 3.7, 0.1, 2.0]])
    argmax = int(torch.argmax(logits, dim=-1))

    for _ in range(5):
        token, _ = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=1)
        assert int(token[0]) == argmax


def test_seeded_determinism_yields_identical_output():
    logits = torch.tensor([[0.1, 0.4, 0.3, 0.9, 0.2, 0.7]])

    torch.manual_seed(0)
    first, _ = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0)

    torch.manual_seed(0)
    second, _ = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert torch.equal(first, second)


def test_seeded_determinism_with_top_p():
    logits = torch.randn(1, 50)

    torch.manual_seed(0)
    first, _ = sample_next_token(logits, temperature=1.0, top_p=0.8, top_k=0)

    torch.manual_seed(0)
    second, _ = sample_next_token(logits, temperature=1.0, top_p=0.8, top_k=0)

    assert torch.equal(first, second)


def test_top_p_includes_first_token():
    """Even when top-1 alone exceeds top_p, it must remain a valid sample."""

    logits = torch.tensor([[10.0, 1.0, 0.5, 0.0]])
    argmax = int(torch.argmax(logits, dim=-1))

    torch.manual_seed(0)
    for _ in range(20):
        token, _ = sample_next_token(logits, temperature=1.0, top_p=0.5, top_k=0)
        assert int(token[0]) == argmax


def test_output_shape():
    logits = torch.randn(3, 16)
    token, processed = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert token.shape == (3,)
    assert token.dtype in (torch.int32, torch.int64)
    assert processed.shape == (3, 16)


def test_top_k_constrains_choice():
    logits = _logits_with_argmax_at(7, vocab=16)
    logits[0, 5] = 9.0  # second largest
    keep = {7, 5}

    torch.manual_seed(0)
    for _ in range(20):
        token, _ = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=2)
        assert int(token[0]) in keep


def test_nan_inf_logits_do_not_crash():
    """A NaN/inf logit (e.g. an fp16 overflow upstream) must not reach
    multinomial — it would device-side-assert. The guard keeps a finite entry
    so a valid token is still drawn."""
    logits = torch.tensor([[float("nan"), float("inf"), 3.0, float("-inf"), 1.0]])
    token, _ = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0)
    assert token.shape == (1,)
    assert int(token[0]) >= 0


def test_all_nan_row_falls_back_to_greedy():
    """If every logit in a row is NaN/inf the distribution has no finite mass;
    the guard falls back to the argmax of the sanitized logits instead of
    producing an invalid (all-zero) probability vector."""
    logits = torch.full((1, 8), float("nan"))
    logits[0, 3] = float("inf")  # the lone 'best' after nan_to_num
    token, _ = sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0)
    assert int(token[0]) == 3


def test_batched_partial_nan_row_recovers():
    """One good row + one all-bad row in the same batch: both yield valid
    tokens, the bad row via the greedy fallback."""
    logits = torch.zeros(2, 6)
    logits[0, 2] = 10.0  # clean row -> argmax 2
    logits[1] = float("nan")
    logits[1, 4] = float("inf")  # degenerate row -> fallback 4
    token, _ = sample_next_token(logits, temperature=1e-4, top_p=1.0, top_k=0)
    assert int(token[0]) == 2
    assert int(token[1]) == 4
