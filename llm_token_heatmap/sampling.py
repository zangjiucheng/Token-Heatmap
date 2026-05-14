"""Sampling utilities for next-token selection (temperature, top-p, top-k)."""

import torch


@torch.no_grad()
def apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
) -> torch.Tensor:
    """Apply temperature scaling and top-k / top-p masking to next-token logits.

    Returns the post-filter logits that `sample_next_token` would softmax. Out-of-mask
    entries are set to ``-inf``.

    Args:
        logits: Tensor of shape [batch, vocab] with raw next-token logits.
        temperature: Sampling temperature. Must be positive.
        top_p: Nucleus sampling cumulative probability cutoff. Disabled when >= 1.0.
        top_k: Keep only the top_k highest-probability tokens. Disabled when <= 0.

    Returns:
        Tensor of shape [batch, vocab] with temperature-scaled, optionally masked logits.
    """
    logits = logits.float() / temperature

    if top_k and top_k > 0:
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        kth_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < kth_value,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[:, 1:] = remove_mask[:, :-1].clone()
        remove_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        logits = filtered_logits

    return logits


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the next token from logits using temperature, top-p, and top-k.

    Args:
        logits: Tensor of shape [batch, vocab] with raw next-token logits.
        temperature: Sampling temperature. Must be positive.
        top_p: Nucleus sampling cumulative probability cutoff. Disabled when >= 1.0.
        top_k: Keep only the top_k highest-probability tokens. Disabled when <= 0.

    Returns:
        (next_token, processed_logits): ``next_token`` is shape [batch] with sampled
        token IDs; ``processed_logits`` is the [batch, vocab] post-filter logits that
        were softmaxed to draw the sample.
    """
    processed_logits = apply_sampling_filters(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    probs = torch.softmax(processed_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token, processed_logits
