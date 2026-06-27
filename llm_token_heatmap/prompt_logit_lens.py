"""Prompt-position logit lens — decode *every* prompt position at every layer.

The per-step logit lens (``LogitLens.capture_step``) only decodes the **answer**
position. But an intermediate computation can live over an earlier token: in the
multihop case ("the capital of the state containing Dallas is Austin") the bridge
feature **Texas** is written over the *Dallas* token in the middle layers, not at
the answer. To see it, we project the residual at every prompt position through
the final norm + unembedding and keep the top-k decoded tokens.

One forward pass over the prompt; hooks each decoder layer to grab all positions
(reusing the resolver helpers in :mod:`llm_token_heatmap.logit_lens`), so it works
on any HF-style causal LM without relying on ``output_hidden_states``.
"""

from __future__ import annotations

from typing import Any

from llm_token_heatmap.logit_lens import (
    _resolve_decoder_layers,
    _resolve_final_norm,
    _resolve_lm_head,
)


def compute_prompt_logit_lens(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    *,
    layers: str | list[int] = "all",
    top_k: int = 8,
) -> dict[str, Any] | None:
    """Decode each prompt position at each layer via the logit lens.

    Returns a ``prompt_logit_lens`` payload, or ``None`` if the model exposes no
    usable unembedding head.
    """
    import torch

    lm_head = _resolve_lm_head(model)
    if lm_head is None:
        return None
    final_norm = _resolve_final_norm(model)
    decoder_layers = _resolve_decoder_layers(model)
    n_layers = len(decoder_layers)
    if n_layers == 0 or not input_ids:
        return None

    if isinstance(layers, str):
        target = list(range(n_layers))
    else:
        target = sorted({int(i) for i in layers if 0 <= int(i) < n_layers})
    if not target:
        return None

    buf: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(idx: int):
        def hook(_m: Any, _i: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden, torch.Tensor):
                buf[idx] = hidden[0].detach()  # [seq, hidden] (batch 0)

        return hook

    for idx in target:
        handles.append(decoder_layers[idx].register_forward_hook(_make_hook(idx)))

    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    seq = ids.shape[1]
    k = max(1, int(top_k))
    positions: list[dict[str, Any]] = []
    for pos in range(seq):
        layer_entries: list[dict[str, Any]] = []
        for idx in target:
            hidden = buf.get(idx)
            if hidden is None:
                continue
            h = hidden[pos]
            if final_norm is not None:
                h = final_norm(h)
            with torch.no_grad():
                logits = lm_head(h).float()
                probs = torch.softmax(logits, dim=-1)
                kk = min(k, probs.shape[-1])
                vals, idxs = torch.topk(probs, kk)
            layer_entries.append(
                {
                    "layer_idx": int(idx),
                    "top_k": [
                        {
                            "rank": r + 1,
                            "token_id": int(t),
                            "token": _decode(tokenizer, int(t)),
                            "prob": float(p),
                        }
                        for r, (p, t) in enumerate(zip(vals.tolist(), idxs.tolist(), strict=False))
                    ],
                }
            )
        positions.append(
            {
                "position": pos,
                "token_id": int(ids[0, pos]),
                "token": _decode(tokenizer, int(ids[0, pos])),
                "layers": layer_entries,
            }
        )

    return {
        "schema_version": "1.0",
        "top_k": k,
        "num_layers": n_layers,
        "positions": positions,
    }


def _decode(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return f"<{token_id}>"
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:  # pragma: no cover - tokenizer quirks
        return f"<{token_id}>"
