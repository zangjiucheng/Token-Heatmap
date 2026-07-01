"""Causal-effect grid — the causal counterpart to Direct Logit Attribution.

For every generated token (step) and every decoder layer, ablate that layer's
attention block (``o_proj`` output) and/or MLP block (``mlp_out`` output) at the
final position, then measure how much the *realized* token's probability moves.
The result is a ``layer × token`` grid of causal effects — the same coordinate
system as the token heatmap, and the ROME-style "which layer at which token
carries the answer" picture. Where DLA is correlational, this is causal:
ablating a load-bearing block drops the token's probability.

Reuses the ablation hooks from :mod:`llm_token_heatmap.analysis.intervention`.
Cost is ``#steps × #layers × #components`` forward passes (the baseline is
computed once per step), so it is opt-in via ``token-heatmap causal``.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"

_COMPONENTS = ("attn", "mlp")


def _num_layers(model: Any, trace: dict[str, Any]) -> int:
    meta = trace.get("attention_metadata") or trace.get("activation_metadata") or {}
    n = meta.get("num_layers")
    if isinstance(n, int) and n > 0:
        return n
    cfg = getattr(model, "config", None)
    return int(getattr(cfg, "num_hidden_layers", 0) or 0)


def compute_causal_grid(
    model: Any,
    tokenizer: Any,
    trace: dict[str, Any],
    *,
    components: tuple[str, ...] = _COMPONENTS,
    op: str = "zero",
    factor: float = 0.0,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Build the layer × token causal-effect grid for a trace.

    Returns a ``causal_grid`` dict; each cell records the ablation's KL and the
    signed change in the realized token's probability (``dp``): a strongly
    negative ``dp`` means the block was load-bearing for that token.
    """
    import torch

    from llm_token_heatmap.analysis.intervention import _attach, _last_logits

    bad = [c for c in components if c not in _COMPONENTS]
    if bad:
        raise ValueError(f"unknown component(s): {bad}; expected {_COMPONENTS}")

    tokens = trace.get("tokens") or {}
    prompt_ids = [int(x) for x in tokens.get("prompt_token_ids", [])]
    steps = trace.get("steps", [])
    if not prompt_ids or not steps:
        raise ValueError(
            "trace lacks tokens.prompt_token_ids or steps; regenerate with `token-heatmap trace`."
        )
    gen_ids = [int(s["selected"]["token_id"]) for s in steps]

    total_layers = _num_layers(model, trace)
    layer_list = list(layers) if layers is not None else list(range(total_layers))

    device = next(model.parameters()).device
    eps = 1e-12
    out_steps: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        target = gen_ids[i]
        input_ids = prompt_ids + gen_ids[:i]
        if not input_ids:
            continue
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        base_logits = _last_logits(model, ids)
        base_p = torch.softmax(base_logits, dim=-1)
        base_target_prob = float(base_p[target].item())

        layers_out: list[dict[str, Any]] = []
        for layer in layer_list:
            cell: dict[str, Any] = {"layer": int(layer)}
            for comp in components:
                handles = _attach(
                    model,
                    [{"layer": int(layer), "component": comp, "op": op, "factor": factor}],
                )
                try:
                    patched_logits = _last_logits(model, ids)
                finally:
                    for h in handles:
                        h.remove()
                patched_p = torch.softmax(patched_logits, dim=-1)
                kl = float(
                    (base_p * (torch.log(base_p + eps) - torch.log(patched_p + eps)))
                    .sum()
                    .item()
                )
                dp = float(patched_p[target].item()) - base_target_prob
                cell[comp] = {"kl": kl, "dp": dp}
            layers_out.append(cell)

        out_steps.append(
            {
                "step": i,
                "token": step["selected"].get("token", ""),
                "token_id": target,
                "base_prob": base_target_prob,
                "layers": layers_out,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "op": op,
        "factor": factor,
        "components": list(components),
        "num_layers": total_layers,
        "steps": out_steps,
    }
