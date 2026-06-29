"""Component-level interventions / ablations for causal validation.

The attribution lenses (DLA, TWERA, …) rank *what contributed* to a token. The
attribution-graphs methods paper
(https://transformer-circuits.pub/2025/attribution-graphs/methods.html) stresses
that those are hypotheses until validated by **intervention**: overwrite a
component's output and measure the effect on the next-token distribution.

This module ablates / scales a decoder layer's **attention block** (``o_proj``
output) or **MLP block** (``mlp_out`` output) at the final sequence position — the
exact residual delta that Direct Logit Attribution measures — and reports the
baseline-vs-patched next-token distribution (top-k, KL, target-prob change,
top-token flips). Pure torch + forward hooks; no server dependency, so it is
unit-testable on a tiny model.

Caveats (mirroring the paper): attention *patterns* are held fixed (we ablate the
block's write to the final residual, after attention has already read), and this
is the direct/OV path — it does not model how attention patterns themselves form.
"""

from __future__ import annotations

from typing import Any

from llm_token_heatmap.probes.activation_probe import (
    _resolve_decoder_layers,
    _resolve_submodule_target,
)

_COMPONENT_TO_CANONICAL = {"attn": "o_proj", "mlp": "mlp_out"}


def _last_logits(model: Any, input_ids: Any) -> Any:
    """Final-position logits of a single forward pass, as float32 [vocab]."""
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids)
    # HF models return a ModelOutput with `.logits`; the tiny test model (and
    # some bare modules) return the logits tensor [batch, seq, vocab] directly.
    logits = out.logits if hasattr(out, "logits") else out
    return logits[0, -1, :].to(dtype=torch.float32)


def _make_hook(op: str, factor: float) -> Any:
    """Forward hook that zeros / scales the last sequence position of a block's
    output (the residual delta it writes), leaving every other position intact."""

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        patched = tensor.clone()
        if op == "scale":
            patched[..., -1, :] = patched[..., -1, :] * factor
        else:  # "zero"
            patched[..., -1, :] = 0
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


def _make_pre_hook(op: str, factor: float, head_slice: slice) -> Any:
    """Forward *pre*-hook on o_proj that zeros / scales one head's slice of the
    last-position input (the concatenated per-head output `z`), ablating just
    that head's write to the residual."""

    def hook(_module: Any, args: Any) -> Any:
        if not args or not hasattr(args[0], "clone"):
            return None
        patched = args[0].clone()
        if op == "scale":
            patched[..., -1, head_slice] = patched[..., -1, head_slice] * factor
        else:  # "zero"
            patched[..., -1, head_slice] = 0
        return (patched, *args[1:])

    return hook


def _head_geometry(model: Any) -> tuple[int, int]:
    """(num_attention_heads, head_dim) from the model config; (0, 0) if unknown."""
    cfg = getattr(model, "config", None)
    num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    head_dim = getattr(cfg, "head_dim", None)
    if not head_dim and num_heads:
        hidden = getattr(cfg, "hidden_size", 0) or 0
        head_dim = hidden // num_heads if hidden else 0
    return num_heads, int(head_dim or 0)


def _attach(model: Any, interventions: list[dict[str, Any]]) -> list[Any]:
    """Register the intervention hooks; returns handles to remove afterwards."""
    layers = _resolve_decoder_layers(model)
    num_heads, head_dim = _head_geometry(model)
    handles: list[Any] = []
    for spec in interventions:
        layer_idx = int(spec["layer"])
        component = str(spec["component"])
        if not (0 <= layer_idx < len(layers)):
            continue
        op = str(spec.get("op", "zero"))
        factor = float(spec.get("factor", 0.0))
        if component == "head":
            head = int(spec.get("head", -1))
            if not num_heads or not head_dim or not (0 <= head < num_heads):
                continue
            o_proj = _resolve_submodule_target(layers[layer_idx], "o_proj")
            if o_proj is None:
                continue
            sl = slice(head * head_dim, (head + 1) * head_dim)
            handles.append(
                o_proj.register_forward_pre_hook(_make_pre_hook(op, factor, sl))
            )
            continue
        canonical = _COMPONENT_TO_CANONICAL.get(component)
        if canonical is None:
            continue
        target = _resolve_submodule_target(layers[layer_idx], canonical)
        if target is None:
            continue
        handles.append(target.register_forward_hook(_make_hook(op, factor)))
    return handles


def _decode(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return f"<{token_id}>"
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:  # pragma: no cover - tokenizer quirks
        return f"<{token_id}>"


def _top(probs: Any, logits: Any, tokenizer: Any, top_k: int) -> list[dict[str, Any]]:
    import torch

    k = min(int(top_k), int(probs.shape[0]))
    vals, idx = torch.topk(probs, k)
    out = []
    for p, i in zip(vals.tolist(), idx.tolist(), strict=False):
        out.append(
            {
                "token": _decode(tokenizer, i),
                "token_id": int(i),
                "prob": float(p),
                "logit": float(logits[int(i)].item()),
            }
        )
    return out


def run_intervention(
    model: Any,
    *,
    input_ids: list[int],
    interventions: list[dict[str, Any]],
    top_k: int = 10,
    target_token_id: int | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Run a baseline and a patched forward over ``input_ids`` and diff the
    final-position next-token distribution.

    Args:
        model: a loaded HF causal LM (``model(input_ids=…).logits``).
        input_ids: the full context whose last position predicts the next token
            (prompt ids + any already-generated continuation ids).
        interventions: list of ``{layer:int, component:'attn'|'mlp',
            op:'zero'|'scale', factor:float}``. Ablates / scales the named block's
            write to the final-position residual.
        top_k: how many top tokens to report per distribution.
        target_token_id: the realized token to track (defaults to baseline argmax).
        tokenizer: optional, used only to decode token strings.

    Returns:
        ``{baseline, patched, diff, interventions, target_token_id}``.
    """
    import torch

    if not input_ids:
        raise ValueError("input_ids must be non-empty")
    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)

    base_logits = _last_logits(model, ids)
    handles = _attach(model, interventions)
    try:
        patched_logits = _last_logits(model, ids)
    finally:
        for h in handles:
            h.remove()

    base_p = torch.softmax(base_logits, dim=-1)
    patched_p = torch.softmax(patched_logits, dim=-1)

    if target_token_id is None or not (0 <= int(target_token_id) < base_p.shape[0]):
        target_token_id = int(torch.argmax(base_logits).item())
    tid = int(target_token_id)

    # KL(baseline || patched) in nats, with a small floor for numerical safety.
    eps = 1e-12
    kl = float(
        (base_p * (torch.log(base_p + eps) - torch.log(patched_p + eps))).sum().item()
    )

    base_top = _top(base_p, base_logits, tokenizer, top_k)
    patched_top = _top(patched_p, patched_logits, tokenizer, top_k)

    flips: list[dict[str, Any]] = []
    for rank in range(min(len(base_top), len(patched_top))):
        if base_top[rank]["token_id"] != patched_top[rank]["token_id"]:
            flips.append(
                {
                    "rank": rank + 1,
                    "from_token": base_top[rank]["token"],
                    "to_token": patched_top[rank]["token"],
                }
            )

    base_target_prob = float(base_p[tid].item())
    patched_target_prob = float(patched_p[tid].item())
    base_target_logit = float(base_logits[tid].item())
    patched_target_logit = float(patched_logits[tid].item())

    return {
        "target_token_id": tid,
        "target_token": _decode(tokenizer, tid),
        "baseline": {
            "top": base_top,
            "target_prob": base_target_prob,
            "target_logit": base_target_logit,
        },
        "patched": {
            "top": patched_top,
            "target_prob": patched_target_prob,
            "target_logit": patched_target_logit,
        },
        "diff": {
            "kl": kl,
            "target_prob_delta": patched_target_prob - base_target_prob,
            "target_logit_delta": patched_target_logit - base_target_logit,
            "top_flips": flips,
        },
        "interventions": list(interventions),
    }
