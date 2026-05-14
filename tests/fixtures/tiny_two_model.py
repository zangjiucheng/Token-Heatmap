"""Synthetic two-checkpoint HuggingFace causal LM fixture for the activation
end-to-end suite.

Two seed-distinct ``LlamaForCausalLM`` instances share a single tiny
``LlamaConfig``. They expose the Llama-style decoder layer tree the
``ActivationProbe`` resolves (``model.layers[i].self_attn.o_proj`` and
``model.layers[i].mlp.down_proj``), so the probe attaches and captures without
any monkey-patching.

The accompanying tokenizer is a small byte-pair-encoding ``PreTrainedTokenizerFast``
trained inline on a fixed corpus. It has no network dependency and is shared
between the two models so traces produced by either model can be aligned on
``token_id`` rather than falling back to character offsets.

The pair-builder also writes both models + tokenizer to disk via
``save_pretrained`` so the subprocess CLI test can pass local paths to
``AutoModelForCausalLM.from_pretrained``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_TINY_TRAINING_CORPUS = (
    "the quick brown fox jumps over the lazy dog",
    "hello world hello there hello again",
    "lorem ipsum dolor sit amet consectetur adipiscing elit",
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
    "one two three four five six seven eight nine ten",
)


def _build_tiny_tokenizer(vocab_size: int = 64) -> Any:
    """Train a deterministic byte-pair tokenizer on a tiny fixed corpus.

    Imports the ``tokenizers`` / ``transformers`` machinery lazily so importing
    this fixture module does not drag those heavyweight modules into
    ``sys.modules`` for unrelated tests.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    inner = Tokenizer(BPE(unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
        show_progress=False,
    )
    inner.train_from_iterator(_TINY_TRAINING_CORPUS, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )


def build_tiny_config(
    *,
    vocab_size: int = 128,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 128,
) -> Any:
    """Build a tiny ``LlamaConfig`` shared between the two seed checkpoints."""

    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def build_tiny_model(*, seed: int, config: Any | None = None) -> Any:
    """Construct a deterministic tiny ``LlamaForCausalLM`` for the given seed."""

    from transformers import LlamaForCausalLM

    cfg = config or build_tiny_config()
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


@dataclass
class TinyTwoModelFixture:
    """The two seeded models, the shared tokenizer, and (if persisted) their paths."""

    config: Any
    model_a: Any
    model_b: Any
    tokenizer: Any
    path_a: Path | None = None
    path_b: Path | None = None


def build_tiny_two_models(
    *,
    seed_a: int = 0,
    seed_b: int = 7,
    config: Any | None = None,
    tokenizer: Any | None = None,
) -> TinyTwoModelFixture:
    """Build the two seeded models plus a shared tiny tokenizer."""

    cfg = config or build_tiny_config()
    tok = tokenizer or _build_tiny_tokenizer()
    model_a = build_tiny_model(seed=seed_a, config=cfg)
    model_b = build_tiny_model(seed=seed_b, config=cfg)
    return TinyTwoModelFixture(
        config=cfg,
        model_a=model_a,
        model_b=model_b,
        tokenizer=tok,
    )


def persist_tiny_two_models(
    base_dir: Path,
    *,
    seed_a: int = 0,
    seed_b: int = 7,
    config: LlamaConfig | None = None,
) -> TinyTwoModelFixture:
    """Build the two seeded models and write them to disk via ``save_pretrained``.

    Each model lands in ``base_dir/model_a`` and ``base_dir/model_b``; the shared
    tokenizer is saved alongside each so ``AutoTokenizer.from_pretrained`` works
    on either path. Returns the in-memory fixture with the on-disk paths set.
    """

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_tiny_two_models(seed_a=seed_a, seed_b=seed_b, config=config)

    path_a = base_dir / "model_a"
    path_b = base_dir / "model_b"
    path_a.mkdir(parents=True, exist_ok=True)
    path_b.mkdir(parents=True, exist_ok=True)

    fixture.model_a.save_pretrained(path_a)
    fixture.model_b.save_pretrained(path_b)
    fixture.tokenizer.save_pretrained(path_a)
    fixture.tokenizer.save_pretrained(path_b)

    fixture.path_a = path_a
    fixture.path_b = path_b
    return fixture
