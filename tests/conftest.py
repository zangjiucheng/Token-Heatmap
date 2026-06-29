"""Shared pytest fixtures for the llm-token-heatmap test suite.

These fixtures intentionally avoid any HuggingFace / network access. The fake
tokenizer is a stand-in for a HF tokenizer that supports the small surface
area exercised by `generate_with_adaptive_probe` and `trace_to_dataframe`.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import pytest
import torch

# Force a non-interactive backend so tests never need a display.
matplotlib.use("Agg")

from llm_token_heatmap.tracing.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe
from llm_token_heatmap.tracing.generation import generate_with_adaptive_probe


class _FakeEncoding(dict):
    """Dict-like encoding that mimics HuggingFace's `BatchEncoding.to(device)`."""

    def to(self, device: Any) -> _FakeEncoding:
        moved = _FakeEncoding(
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.items()}
        )
        return moved


class FakeTokenizer:
    """Minimal tokenizer stub compatible with the library's generation and export paths.

    - Encoding (`__call__`): maps characters to byte values clipped to `vocab_size`,
      and returns an object that supports `.to(device)` like HF's `BatchEncoding`.
    - Decoding: returns a deterministic "<tok:ID>" string for each token id.
    - Exposes `eos_token_id`, which can be set to None or an int.
    """

    def __init__(self, vocab_size: int = 64, eos_token_id: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def __call__(self, prompt: str, return_tensors: str = "pt") -> _FakeEncoding:
        ids = [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in prompt[:8]]
        if not ids:
            ids = [0]
        return _FakeEncoding({"input_ids": torch.tensor([ids], dtype=torch.long)})

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        try:
            ids = list(token_ids)
        except TypeError:
            ids = [int(token_ids)]
        return "".join(f"<tok:{int(i)}>" for i in ids)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """A small deterministic tokenizer with no network or model dependency."""

    return FakeTokenizer(vocab_size=64, eos_token_id=None)


@pytest.fixture
def small_vocab_size() -> int:
    return 32


@pytest.fixture
def torch_seed() -> int:
    """Default seed used by tests that exercise sampling determinism."""

    return 0


def make_sharp_logits(
    vocab_size: int, peak_index: int = 0, peak_value: float = 50.0
) -> torch.Tensor:
    """Build a [1, vocab_size] logits tensor with a single dominant entry."""

    logits = torch.zeros(1, vocab_size)
    logits[0, peak_index] = peak_value
    return logits


def make_uniform_logits(vocab_size: int) -> torch.Tensor:
    """Build a [1, vocab_size] logits tensor that yields a uniform distribution."""

    return torch.zeros(1, vocab_size)


def make_threshold_logits(vocab_size: int, top: int = 3) -> torch.Tensor:
    """Build a [1, vocab_size] logits tensor where the top-`top` entries hold >95% mass."""

    logits = torch.full((1, vocab_size), -20.0)
    logits[0, :top] = 5.0
    return logits


@pytest.fixture
def sharp_logits_factory():
    return make_sharp_logits


@pytest.fixture
def uniform_logits_factory():
    return make_uniform_logits


@pytest.fixture
def threshold_logits_factory():
    return make_threshold_logits


class _FakeOutputs:
    def __init__(self, logits: torch.Tensor, past_key_values: Any) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class FakeModel:
    """Returns canned next-token logits per call. ``logits_per_step[i]`` is shape [vocab]."""

    def __init__(self, logits_per_step: list[torch.Tensor]) -> None:
        self.logits_per_step = logits_per_step
        self.call_count = 0
        self.device = torch.device("cpu")

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> _FakeOutputs:
        step_logits = self.logits_per_step[self.call_count]
        self.call_count += 1
        logits = step_logits.view(1, 1, -1)
        return _FakeOutputs(logits=logits, past_key_values="cache")


@pytest.fixture
def make_run():
    """Build a generation trace with controllable sampling params and canned logits.

    Used by the raw-vs-processed tests to drive ``generate_with_adaptive_probe`` end
    to end without any model download.
    """

    def _run(
        *,
        n_steps: int = 3,
        vocab_size: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_k: int = 1,
        max_k: int = 16,
        mass_threshold: float = 0.95,
        seed: int = 0,
    ) -> tuple[list[dict[str, Any]], FakeTokenizer]:
        torch.manual_seed(seed)
        logits_per_step = [torch.randn(vocab_size) for _ in range(n_steps)]

        tokenizer = FakeTokenizer(vocab_size=vocab_size)
        model = FakeModel(logits_per_step)
        probe = AdaptiveTokenProbe(
            AdaptiveProbeConfig(
                min_k=min_k,
                max_k=max_k,
                mass_threshold=mass_threshold,
            )
        )
        _text, trace = generate_with_adaptive_probe(
            model=model,
            tokenizer=tokenizer,
            prompt="hello",
            probe=probe,
            max_new_tokens=n_steps,
            temperature=temperature,
            top_p=top_p,
            sample_top_k=top_k,
        )
        return trace, tokenizer

    return _run
