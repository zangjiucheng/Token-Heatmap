"""Direct Logit Attribution (DLA) for a single generation trace.

Decomposes the realized next-token logit at each step into additive
contributions from the residual input (embedding) and each layer's
attention-block (``o_proj``) and MLP-block (``mlp_out``) outputs, projected
through the unembedding with the final norm *folded* as a fixed per-token scale.
This is the standard "direct logit attribution" used throughout Anthropic's
*Circuit Tracing* methods
(https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

It reuses the same per-``(layer, submodule)`` tensors captured for the TWERA
neuron attribution (``--capture-full-activations``) — no extra hooks. For
standard pre-norm transformers the residual stream is exactly

    h = embed + Σ_layers (o_proj_L + mlp_out_L)

so for RMSNorm models (Qwen/Llama/Mistral/…) the decomposition is *exact*: the
sum of contributions equals the model's logit. We still report a per-step
``error`` so any residual — the final-norm linearization, a norm variant we
didn't special-case, or missing layers — is visible rather than hidden (the
paper's "error node" discipline; see docs/epics/04-faithfulness-error-reporting).

Honest caveats (mirroring the paper):
- Direct path only: this explains the OV/residual-direct contribution to the
  logit, not how attention *patterns* form (QK circuits).
- Per-block, not yet per-head: attention is one ``attn`` term per layer (v2 will
  split per head from the attention sidecar).
- The final norm is folded as a fixed scale (its only nonlinearity); ``error``
  surfaces the linearization gap.
"""

from __future__ import annotations

from typing import Any

_ATTN_KEYS = {"o_proj"}
_MLP_KEYS = {"mlp_out", "mlp.down_proj"}
_RESID_KEYS = {"residual_post", "resid_post"}


def _norm_params(final_norm: Any, hidden: int, torch: Any, device: Any) -> tuple[Any, Any, float, bool]:
    """Extract (weight, bias, eps, gemma_plus_one) from a final-norm module.

    ``bias is None`` ⇒ RMSNorm-family (no mean subtraction). Gemma's RMSNorm
    scales by ``(1 + weight)``; we detect it by class name so the decomposition
    stays exact for that family.
    """
    if final_norm is None:
        return torch.ones(hidden, dtype=torch.float32, device=device), None, 1e-6, False
    weight = getattr(final_norm, "weight", None)
    bias = getattr(final_norm, "bias", None)
    w = (
        weight.detach().reshape(-1).to(device=device, dtype=torch.float32)
        if weight is not None
        else torch.ones(hidden, dtype=torch.float32, device=device)
    )
    b = (
        bias.detach().reshape(-1).to(device=device, dtype=torch.float32)
        if bias is not None
        else None
    )
    eps = getattr(final_norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(final_norm, "eps", None)
    eps = float(eps) if eps is not None else 1e-6
    plus_one = "gemma" in type(final_norm).__name__.lower()
    return w, b, eps, plus_one


def _contrib(c: Any, weight_eff: Any, d: Any, u: Any, is_layernorm: bool) -> float:
    """Contribution of residual component ``c`` to the target logit, folding the
    final norm as the fixed scale ``d``: ``(weight_eff ⊙ ĉ / d) · W_U[target]``,
    where ``ĉ`` is mean-centred for LayerNorm and ``c`` itself for RMSNorm."""
    cc = (c - c.mean()) if is_layernorm else c
    return float(((weight_eff * cc) / d * u).sum().item())


def _true_target_logit(final_norm: Any, h: Any, u: Any, torch: Any, device: Any) -> float | None:
    """The model's actual logit for the target token, ``norm(h) · W_U[target]``,
    applied through the real norm module. Returns None if it can't be applied
    cleanly (then the caller falls back to the analytic total)."""
    if final_norm is None:
        return None
    try:
        p = getattr(final_norm, "weight", None)
        nd = p.device if p is not None else device
        ndt = p.dtype if p is not None else torch.float32
        with torch.no_grad():
            g = final_norm(h.to(device=nd, dtype=ndt))
        g = g.reshape(-1).to(device=device, dtype=torch.float32)
        return float((g * u).sum().item())
    except Exception:  # pragma: no cover - norm-module quirks
        return None


def _per_head_contribs(
    z: Any,
    w_o: Any,
    num_heads: int,
    head_dim: int,
    weight_eff: Any,
    d: Any,
    u: Any,
    is_layernorm: bool,
) -> list[dict[str, Any]]:
    """Split a layer's attention contribution per head.

    ``z`` is the ``o_proj`` input (concatenated per-head outputs,
    ``[num_heads*head_dim]``); ``w_o`` is ``o_proj.weight`` (``[hidden, num_heads*head_dim]``).
    Head ``h``'s residual write is ``w_o[:, slice_h] @ z[slice_h]``; these sum to
    the layer's full attention contribution.
    """
    heads: list[dict[str, Any]] = []
    for hh in range(num_heads):
        sl = slice(hh * head_dim, (hh + 1) * head_dim)
        contrib_vec = w_o[:, sl] @ z[sl]  # [hidden]
        heads.append(
            {"head": hh, "attn": _contrib(contrib_vec, weight_eff, d, u, is_layernorm)}
        )
    return heads


def compute_direct_logit_attribution(
    *,
    trace: list[dict[str, Any]],
    target_token_ids: list[int],
    unembedding: Any,
    final_norm: Any = None,
    o_proj_weights: dict[int, Any] | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> dict[str, Any] | None:
    """Per-step direct logit attribution of the realized next token.

    Args:
        trace: per-step entries from ``generate_with_adaptive_probe``; entries
            carrying full activations expose them at
            ``entry["_activation_full_stats"].layer_tensors[(layer, submodule)]``.
            Needs ``--capture-full-activations``.
        target_token_ids: realized next-token id per step, parallel to ``trace``.
        unembedding: ``W_U`` of shape ``[vocab, hidden]``
            (``model.get_output_embeddings().weight``).
        final_norm: the model's final norm module (e.g. from
            ``llm_token_heatmap.probes.logit_lens._resolve_final_norm``). ``None`` falls
            back to an identity-weight RMSNorm (logit-lens style).

    Returns:
        A ``direct_logit_attribution`` dict (see ``docs/web/trace.schema.json``),
        or ``None`` when no full activations / unembedding are available.
    """
    import torch

    if unembedding is None or not trace:
        return None
    w_u = unembedding
    if getattr(w_u, "ndim", 0) != 2:
        return None
    vocab, hidden = int(w_u.shape[0]), int(w_u.shape[1])
    device = w_u.device
    weight, bias, eps, plus_one = _norm_params(final_norm, hidden, torch, device)
    weight_eff = (weight + 1.0) if plus_one else weight
    is_layernorm = bias is not None

    per_head_on = bool(o_proj_weights) and bool(num_heads) and bool(head_dim)
    nh = int(num_heads) if num_heads else 0
    hd = int(head_dim) if head_dim else 0

    steps_out: list[dict[str, Any]] = []
    num_layers = 0

    with torch.no_grad():
        for entry, tok in zip(trace, target_token_ids, strict=False):
            full = entry.get("_activation_full_stats")
            tensors = getattr(full, "layer_tensors", None) if full is not None else None
            if not tensors or not (0 <= int(tok) < vocab):
                continue
            attn_z_map = (getattr(full, "attn_z", None) or {}) if per_head_on else {}

            attn: dict[int, Any] = {}
            mlp: dict[int, Any] = {}
            resid: dict[int, Any] = {}
            for (layer, sub), vec in tensors.items():
                if vec is None:
                    continue
                v = vec.reshape(-1).to(device=device, dtype=torch.float32)
                if v.shape[0] != hidden:
                    continue  # not residual-basis — skip
                s = str(sub)
                L = int(layer)
                if s in _ATTN_KEYS:
                    attn[L] = v
                elif s in _MLP_KEYS:
                    mlp[L] = v
                elif s in _RESID_KEYS:
                    resid[L] = v

            if not resid or (not attn and not mlp):
                continue

            h = resid[max(resid.keys())]  # final residual at this position
            u = w_u[int(tok)].to(device=device, dtype=torch.float32)
            if is_layernorm:
                mu = h.mean()
                d = torch.sqrt(((h - mu) ** 2).mean() + eps)
            else:
                d = torch.sqrt((h * h).mean() + eps)

            layer_ids = sorted(set(attn.keys()) | set(mlp.keys()))
            num_layers = max(num_layers, len(layer_ids))
            layers_out: list[dict[str, Any]] = []
            summed = torch.zeros(hidden, dtype=torch.float32, device=device)
            explained = 0.0
            for L in layer_ids:
                a = attn.get(L)
                m = mlp.get(L)
                a_c = _contrib(a, weight_eff, d, u, is_layernorm) if a is not None else 0.0
                m_c = _contrib(m, weight_eff, d, u, is_layernorm) if m is not None else 0.0
                layer_out: dict[str, Any] = {"layer": L, "attn": a_c, "mlp": m_c}
                # Per-head split of this layer's attention contribution, when the
                # o_proj input (z) and weight (W_O) are available.
                z = attn_z_map.get(L)
                w_o = o_proj_weights.get(L) if o_proj_weights is not None else None
                if z is not None and w_o is not None:
                    zt = z.reshape(-1).to(device=device, dtype=torch.float32)
                    wt = w_o.to(device=device, dtype=torch.float32)
                    if zt.shape[0] == nh * hd and wt.shape[1] == nh * hd:
                        layer_out["heads"] = _per_head_contribs(
                            zt, wt, nh, hd, weight_eff, d, u, is_layernorm
                        )
                layers_out.append(layer_out)
                explained += a_c + m_c
                if a is not None:
                    summed = summed + a
                if m is not None:
                    summed = summed + m

            embed_c = _contrib(h - summed, weight_eff, d, u, is_layernorm)
            explained += embed_c
            bias_c = float((bias * u).sum().item()) if bias is not None else 0.0
            true_logit = _true_target_logit(final_norm, h, u, torch, device)
            total = true_logit if true_logit is not None else (explained + bias_c)

            steps_out.append(
                {
                    "step": int(entry.get("step", len(steps_out))),
                    "target_token_id": int(tok),
                    "total_logit": float(total),
                    "embed": float(embed_c),
                    "bias": float(bias_c),
                    "error": float(total - explained - bias_c),
                    "layers": layers_out,
                }
            )

    if not steps_out:
        return None

    return {
        "method": "dla_fold_norm",
        "n_steps": len(steps_out),
        "num_layers": int(num_layers),
        "note": (
            "Direct logit attribution: the realized next-token logit decomposed "
            "into additive contributions from the residual input (embed) and each "
            "layer's attention-block (o_proj) and MLP-block (mlp_out) outputs, "
            "projected through the unembedding with the final norm folded as a "
            "fixed scale. 'error' is the unexplained residual (final-norm "
            "linearization); for RMSNorm models it is ~0. Direct/OV path only — "
            "it does not explain how attention patterns form."
        ),
        "steps": steps_out,
    }
