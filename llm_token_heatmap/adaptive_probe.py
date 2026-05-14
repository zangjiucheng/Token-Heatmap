"""Adaptive token probe for analyzing LLM next-token logits."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaptiveProbeConfig:
    """Configuration for adaptive token tracing.

    Attributes:
        min_k: Minimum number of tokens to keep.
        max_k: Maximum number of tokens to keep.
        mass_threshold: Target cumulative probability mass.
        eps: Numerical stability constant.
    """

    min_k: int = 8
    max_k: int = 64
    mass_threshold: float = 0.95
    eps: float = 1e-12


class AdaptiveTokenProbe(nn.Module):
    """Analyze next-token logits during LLM inference.

    Adaptive behavior:
    - Computes top max_k tokens.
    - Chooses the smallest k >= min_k whose cumulative probability mass
      reaches mass_threshold.
    - Falls back to max_k if the probability mass is too diffuse.
    """

    def __init__(self, config: AdaptiveProbeConfig):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        selected_ids: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        cfg = self.config

        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape [batch, vocab], got {tuple(logits.shape)}")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        scaled_logits = logits.float() / temperature

        log_probs = F.log_softmax(scaled_logits, dim=-1)
        probs = log_probs.exp()

        _batch_size, vocab_size = probs.shape
        effective_max_k = min(cfg.max_k, vocab_size)

        top_probs, top_ids = torch.topk(
            probs,
            k=effective_max_k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        top_logprobs = torch.log(top_probs + cfg.eps)
        cumulative_mass = torch.cumsum(top_probs, dim=-1)

        reached = cumulative_mass >= cfg.mass_threshold
        first_reached = torch.argmax(reached.int(), dim=-1) + 1
        has_reached = reached.any(dim=-1)

        k_used = torch.where(
            has_reached,
            first_reached,
            torch.full_like(first_reached, effective_max_k),
        )

        k_used = torch.clamp(k_used, min=cfg.min_k, max=effective_max_k)

        rank_positions = torch.arange(effective_max_k, device=logits.device).unsqueeze(0)
        valid_mask = rank_positions < k_used.unsqueeze(1)

        plogp = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
        entropy = -plogp.sum(dim=-1)

        result: dict[str, Any] = {
            "top_ids": top_ids,
            "top_probs": top_probs,
            "top_logprobs": top_logprobs,
            "valid_mask": valid_mask,
            "k_used": k_used,
            "entropy": entropy,
            "top_mass_used": torch.gather(
                cumulative_mass,
                dim=-1,
                index=(k_used - 1).unsqueeze(-1),
            ).squeeze(-1),
        }

        if selected_ids is not None:
            if selected_ids.ndim != 1:
                raise ValueError("selected_ids should have shape [batch]")

            selected_ids = selected_ids.to(logits.device)

            selected_prob = probs.gather(
                dim=-1,
                index=selected_ids.unsqueeze(-1),
            ).squeeze(-1)

            selected_logprob = torch.log(selected_prob + cfg.eps)

            selected_rank = (probs > selected_prob.unsqueeze(-1)).sum(dim=-1) + 1

            result.update(
                {
                    "selected_ids": selected_ids,
                    "selected_prob": selected_prob,
                    "selected_logprob": selected_logprob,
                    "selected_rank": selected_rank,
                }
            )

        return result
