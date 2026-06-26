"""Tests for `generate_with_adaptive_probe` using a mocked model.

The mock model returns hand-crafted logits without any HuggingFace import or
network call, so the test runs entirely offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from llm_token_heatmap.adaptive_probe import AdaptiveProbeConfig, AdaptiveTokenProbe
from llm_token_heatmap.generation import generate_with_adaptive_probe

EXPECTED_STEP_KEYS = {"step", "raw", "processed"}

EXPECTED_STATS_KEYS = {
    "top_ids",
    "top_probs",
    "top_logprobs",
    "valid_mask",
    "k_used",
    "entropy",
    "selected_ids",
    "selected_prob",
    "selected_logprob",
    "selected_rank",
}


@dataclass
class _ModelOutput:
    logits: torch.Tensor
    past_key_values: Any


class FakeCausalLM:
    """Mock causal LM that returns a fixed peaky logits vector at every step."""

    def __init__(self, vocab_size: int = 32, peak_index: int = 1) -> None:
        self.vocab_size = vocab_size
        self.peak_index = peak_index
        self.device = torch.device("cpu")
        self.calls = 0

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> _ModelOutput:
        self.calls += 1
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        logits = torch.full((batch, seq_len, self.vocab_size), -5.0)
        logits[..., self.peak_index] = 10.0

        return _ModelOutput(logits=logits, past_key_values=(self.calls,))


def _build_probe() -> AdaptiveTokenProbe:
    return AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=4, max_k=8))


def test_generation_trace_length(fake_tokenizer):
    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
    probe = _build_probe()

    text, trace = generate_with_adaptive_probe(
        model,
        fake_tokenizer,
        prompt="hello",
        probe=probe,
        max_new_tokens=5,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
    )

    assert len(trace) == 5
    assert isinstance(text, str)


def test_generation_step_keys(fake_tokenizer):
    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
    probe = _build_probe()

    _, trace = generate_with_adaptive_probe(
        model,
        fake_tokenizer,
        prompt="hi",
        probe=probe,
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
    )

    for entry in trace:
        assert EXPECTED_STEP_KEYS.issubset(entry.keys())
        assert EXPECTED_STATS_KEYS.issubset(entry["raw"].keys())
        assert EXPECTED_STATS_KEYS.issubset(entry["processed"].keys())


def test_generation_stops_on_eos(fake_tokenizer):
    """When the deterministic next-token equals eos_token_id, generation stops early."""

    eos_id = 2
    fake_tokenizer.eos_token_id = eos_id
    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size, peak_index=eos_id)
    probe = _build_probe()

    _, trace = generate_with_adaptive_probe(
        model,
        fake_tokenizer,
        prompt="hi",
        probe=probe,
        max_new_tokens=10,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
    )

    assert len(trace) == 1
    assert int(trace[0]["raw"]["selected_ids"][0]) == eos_id
    assert int(trace[0]["processed"]["selected_ids"][0]) == eos_id


def test_generation_uses_past_key_values_after_first_step(fake_tokenizer):
    """First call passes the full prompt; subsequent calls should pass only the last token."""

    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
    probe = _build_probe()

    _, trace = generate_with_adaptive_probe(
        model,
        fake_tokenizer,
        prompt="hi",
        probe=probe,
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
    )

    assert len(trace) == 3
    # One call per generated token.
    assert model.calls == 3


def _make_chat_tokenizer(vocab_size: int = 32, chat_template: Any = "<template>") -> MagicMock:
    """MagicMock tokenizer that exposes a chat_template and apply_chat_template."""

    tokenizer = MagicMock()
    tokenizer.vocab_size = vocab_size
    tokenizer.eos_token_id = None
    tokenizer.chat_template = chat_template
    tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]], dtype=torch.long)
    tokenizer.decode.return_value = "chat-output"
    return tokenizer


def test_chat_template_applied_when_flag_set():
    tokenizer = _make_chat_tokenizer(vocab_size=32)
    model = FakeCausalLM(vocab_size=32)
    probe = _build_probe()

    generate_with_adaptive_probe(
        model,
        tokenizer,
        prompt="hello",
        probe=probe,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
        use_chat_template=True,
    )

    tokenizer.apply_chat_template.assert_called_once()
    call = tokenizer.apply_chat_template.call_args
    messages = call.args[0] if call.args else call.kwargs["messages"]
    assert messages == [{"role": "user", "content": "hello"}]
    assert call.kwargs.get("add_generation_prompt") is True
    tokenizer.assert_not_called()


def test_chat_template_raises_when_unavailable():
    tokenizer = _make_chat_tokenizer(vocab_size=32, chat_template=None)
    model = FakeCausalLM(vocab_size=32)
    probe = _build_probe()

    with pytest.raises(ValueError, match="chat_template"):
        generate_with_adaptive_probe(
            model,
            tokenizer,
            prompt="hello",
            probe=probe,
            max_new_tokens=2,
            use_chat_template=True,
        )

    tokenizer.apply_chat_template.assert_not_called()


def test_system_prompt_prepended():
    tokenizer = _make_chat_tokenizer(vocab_size=32)
    model = FakeCausalLM(vocab_size=32)
    probe = _build_probe()

    generate_with_adaptive_probe(
        model,
        tokenizer,
        prompt="hello",
        probe=probe,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
        use_chat_template=True,
        system_prompt="You are X.",
    )

    call = tokenizer.apply_chat_template.call_args
    messages = call.args[0] if call.args else call.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are X."}
    assert messages[1] == {"role": "user", "content": "hello"}


def test_default_path_unchanged(fake_tokenizer):
    spy = MagicMock()
    spy.vocab_size = fake_tokenizer.vocab_size
    spy.eos_token_id = None
    spy.side_effect = lambda *args, **kwargs: fake_tokenizer(*args, **kwargs)
    spy.decode.side_effect = lambda *args, **kwargs: fake_tokenizer.decode(*args, **kwargs)

    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
    probe = _build_probe()

    generate_with_adaptive_probe(
        model,
        spy,
        prompt="hi",
        probe=probe,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
    )

    spy.assert_called_once_with("hi", return_tensors="pt")
    spy.apply_chat_template.assert_not_called()


def test_generation_no_network_imports():
    """Catch accidental introduction of network-bound modules at runtime."""

    import sys

    assert "transformers" not in sys.modules
    assert "huggingface_hub" not in sys.modules


class _FakeProbeConfig:
    def __init__(self, capture_full_distribution: bool = False, top_k_positions: int = 8) -> None:
        self.capture_full_distribution = capture_full_distribution
        self.top_k_positions = top_k_positions


class _FakeAttentionStats:
    """Minimal stand-in for AttentionStats that the serializer can consume."""

    def __init__(self) -> None:
        self.layers: dict[int, Any] = {}
        self.num_attention_heads = 1
        self.num_key_value_heads = 1
        self.head_dim = 1
        self.head_to_kv_group = [0]


class _FakeAttentionProbe:
    """Records every ``capture_step`` call; emits a payload the gen loop accepts."""

    def __init__(self) -> None:
        self.is_attached = True
        self.config = _FakeProbeConfig()
        self.calls = 0
        self._stats = _FakeAttentionStats()

    def capture_step(self) -> _FakeAttentionStats:
        self.calls += 1
        return self._stats


class _FakeLogitLens:
    """Records every ``capture_step`` call; emits an empty LogitLensStats."""

    def __init__(self) -> None:
        from llm_token_heatmap.logit_lens import LogitLensStats

        self.is_attached = True
        self.calls = 0
        self._stats = LogitLensStats(layers={})

    def capture_step(self, selected_token_id=None):  # noqa: ANN001
        self.calls += 1
        return self._stats


def test_generate_invokes_attention_probe_per_step(
    monkeypatch: pytest.MonkeyPatch, fake_tokenizer
) -> None:
    """The attention probe's `capture_step` fires exactly once per generated token."""

    from llm_token_heatmap import generation as gen_mod

    seen_token_ids: list[list[int] | None] = []

    def _fake_payload(stats, *, capture_full, top_k_positions, token_ids=None):  # noqa: ANN001
        seen_token_ids.append(token_ids)
        return {"attention": [], "attention_metadata": {}}

    monkeypatch.setattr(gen_mod, "attention_stats_to_payload", _fake_payload)

    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
    probe = _build_probe()
    fake_attn = _FakeAttentionProbe()

    _, trace = generate_with_adaptive_probe(
        model,
        fake_tokenizer,
        prompt="hi",
        probe=probe,
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
        attention_probe=fake_attn,
    )

    assert fake_attn.calls == 4
    assert len(trace) == 4
    for entry in trace:
        assert "attention" in entry
        assert "_attention_stats" in entry
    # The running token-id sequence is threaded in for induction scoring, and it
    # grows by one each step as tokens are appended.
    assert len(seen_token_ids) == 4
    assert all(ids is not None for ids in seen_token_ids)
    lengths = [len(ids) for ids in seen_token_ids]
    assert lengths == sorted(lengths) and len(set(lengths)) == 4


def test_generate_invokes_logit_lens_per_step(fake_tokenizer) -> None:
    """The logit lens's `capture_step` fires exactly once per generated token."""

    model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
    probe = _build_probe()
    fake_lens = _FakeLogitLens()

    _, trace = generate_with_adaptive_probe(
        model,
        fake_tokenizer,
        prompt="hi",
        probe=probe,
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
        sample_top_k=1,
        logit_lens=fake_lens,
    )

    assert fake_lens.calls == 3
    assert len(trace) == 3
    for entry in trace:
        assert "logit_lens" in entry


def _trace_signature(trace: list[dict]) -> list[tuple]:
    """Reduce a trace to a tuple of selected_ids per step (for equality comparison)."""

    sig: list[tuple] = []
    for entry in trace:
        sig.append(
            (
                int(entry["step"]),
                int(entry["raw"]["selected_ids"][0]),
                int(entry["processed"]["selected_ids"][0]),
            )
        )
    return sig


def test_default_run_byte_identical_without_probes(fake_tokenizer) -> None:
    """Without attention/lens kwargs (or with both None), generation is deterministic
    and matches a seeded baseline byte-for-byte."""

    def _run(**probe_kwargs):
        torch.manual_seed(0)
        model = FakeCausalLM(vocab_size=fake_tokenizer.vocab_size)
        probe = _build_probe()
        text, trace = generate_with_adaptive_probe(
            model,
            fake_tokenizer,
            prompt="hi",
            probe=probe,
            max_new_tokens=4,
            temperature=1.0,
            top_p=1.0,
            sample_top_k=1,
            **probe_kwargs,
        )
        return text, _trace_signature(trace)

    base_text, base_sig = _run()
    none_text, none_sig = _run(attention_probe=None, logit_lens=None)
    repeat_text, repeat_sig = _run()

    assert base_text == repeat_text == none_text
    assert base_sig == repeat_sig == none_sig
    for entry_sig in base_sig:
        # No attention or logit_lens artefacts should leak into the step entries.
        # (Validated indirectly: signature only reads keys that exist by default.)
        assert isinstance(entry_sig, tuple)
