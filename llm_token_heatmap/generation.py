"""Manual generation loop with adaptive token probe tracing."""

from dataclasses import asdict, is_dataclass
from typing import Any

import torch

from llm_token_heatmap.activation_probe import ActivationProbe
from llm_token_heatmap.adaptive_probe import AdaptiveTokenProbe
from llm_token_heatmap.attention_probe import AttentionProbe
from llm_token_heatmap.attention_serializer import attention_stats_to_payload
from llm_token_heatmap.logit_lens import LogitLens, LogitLensStats
from llm_token_heatmap.sampling import sample_next_token


def _stats_to_cpu(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in stats.items()}


def _logit_lens_to_dict(stats: LogitLensStats) -> list[dict[str, Any]]:
    """Flatten a LogitLensStats payload into a JSON-friendly list of layer dicts."""

    out: list[dict[str, Any]] = []
    for _, layer in sorted(stats.layers.items()):
        payload = asdict(layer) if is_dataclass(layer) else dict(layer.__dict__)
        out.append(payload)
    return out


def _extract_input_ids(tokenizer_output: Any) -> torch.Tensor:
    """Pull the `input_ids` tensor out of whatever the tokenizer returned.

    Handles plain tensors, `BatchEncoding`, plain dicts, and objects that
    expose `input_ids` as an attribute. Newer transformers releases sometimes
    leave a `BatchEncoding` wrapper in place after `.to(device)`, so we
    refuse to forward anything that isn't a tensor.
    """
    candidate = tokenizer_output
    if torch.is_tensor(candidate):
        return candidate
    if hasattr(candidate, "input_ids") and not isinstance(candidate, dict):
        candidate = candidate.input_ids
    elif isinstance(candidate, dict) or hasattr(candidate, "__getitem__"):
        try:
            candidate = candidate["input_ids"]
        except (KeyError, TypeError) as exc:
            raise TypeError(
                f"Tokenizer output {type(tokenizer_output).__name__} did not expose 'input_ids'."
            ) from exc

    if not torch.is_tensor(candidate):
        raise TypeError(
            f"Expected input_ids to be a torch.Tensor, got "
            f"{type(candidate).__name__}. If you are passing the result of "
            "tokenizer.apply_chat_template with return_dict=True, extract "
            "input_ids explicitly first."
        )

    return candidate


@torch.no_grad()
def generate_with_adaptive_probe(
    model: Any,
    tokenizer: Any,
    prompt: str,
    probe: AdaptiveTokenProbe,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    sample_top_k: int = 0,
    use_chat_template: bool = False,
    system_prompt: str | None = None,
    logit_lens: LogitLens | None = None,
    attention_probe: AttentionProbe | None = None,
    activation_probe: ActivationProbe | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate text and capture adaptive token-probability traces step by step.

    At every step the probe runs twice: once on the raw temperature-scaled logits
    and once on the post-sampling-filter logits (after top-p / top-k). Each trace
    entry therefore carries both a ``raw`` and a ``processed`` stats sub-dict.

    Args:
        model: HuggingFace causal LM with `past_key_values` support.
        tokenizer: HuggingFace tokenizer matched to the model.
        prompt: Input prompt string.
        probe: AdaptiveTokenProbe instance for analyzing per-step logits.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling cutoff.
        sample_top_k: Optional top-k filter for sampling.
        use_chat_template: If True, wrap the prompt via
            ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
            so instruct/chat models receive properly tagged input. Raises
            ``ValueError`` when the tokenizer has no ``chat_template``.
        system_prompt: Optional system message prepended when
            ``use_chat_template=True``.

    Returns:
        A tuple of (decoded_text, trace) where trace is a list of per-step dicts
        each containing ``step``, ``raw`` and ``processed`` keys.
    """
    if use_chat_template:
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                "Tokenizer has no `chat_template`; pass an instruct tokenizer "
                "or call with `use_chat_template=False`."
            )
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_output = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = _extract_input_ids(chat_output).to(model.device)
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = _extract_input_ids(encoded).to(model.device)

    generated = input_ids
    past_key_values = None
    trace: list[dict[str, Any]] = []

    # Track the cumulative decoded text so each step can record the character
    # offset where its new token begins. The activation diff schema uses this
    # as the cross-tokenizer alignment key (two traces produced by different
    # tokenizers can still be zipped by decoded offset when token ids diverge).
    prev_decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    for step in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(
                input_ids=generated,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        next_token, processed_logits = sample_next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=sample_top_k,
        )

        raw_stats = probe(
            logits,
            selected_ids=next_token,
            temperature=temperature,
        )
        processed_stats = probe(
            processed_logits,
            selected_ids=next_token,
            temperature=1.0,
        )

        step_entry: dict[str, Any] = {
            "step": step,
            "decoded_text_offset": len(prev_decoded_text),
            "raw": _stats_to_cpu(raw_stats),
            "processed": _stats_to_cpu(processed_stats),
        }
        if logit_lens is not None and logit_lens.is_attached:
            step_entry["logit_lens"] = _logit_lens_to_dict(logit_lens.capture_step(next_token))
        if attention_probe is not None and attention_probe.is_attached:
            attention_stats = attention_probe.capture_step()
            payload = attention_stats_to_payload(
                attention_stats,
                capture_full=attention_probe.config.capture_full_distribution,
                top_k_positions=attention_probe.config.top_k_positions,
                # The current query is the last position of `generated`; its
                # token-id sequence lets the serializer score induction heads.
                token_ids=[int(t) for t in generated[0].tolist()],
            )
            step_entry["attention"] = payload["attention"]
            # Stash the raw stats so callers (e.g. the CLI) can write Tier 2
            # sidecars or compute downstream derived stats without re-running
            # the model. The key is underscore-prefixed so JSON serializers
            # treat it as private.
            step_entry["_attention_stats"] = attention_stats
        if activation_probe is not None and activation_probe.is_attached:
            activation_entries = activation_probe.capture_step()
            step_entry["activations"] = [asdict(entry) for entry in activation_entries]
            if activation_probe.config.capture_full:
                step_entry["_activation_full_stats"] = activation_probe.last_full_stats
        trace.append(step_entry)

        generated = torch.cat([generated, next_token[:, None]], dim=-1)
        prev_decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        if tokenizer.eos_token_id is not None and int(next_token[0]) == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return text, trace
