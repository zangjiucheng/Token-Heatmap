"""Single-trace TWERA-style neuron attribution.

A scoped, faithful approximation of **Target-Weighted Expected Residual
Attribution (TWERA)** from Anthropic's *Circuit Tracing* methods
(https://transformer-circuits.pub/2025/attribution-graphs/methods.html),
adapted to a single generation trace.

In the paper, TWERA ranks the connections of an attribution graph by their
virtual weight reweighted by empirical coactivation statistics, so the ordering
reflects on-distribution effects rather than raw (interference-dominated)
weights. We apply the same idea to rank the residual-stream neurons of each
captured ``(layer, submodule)``: score neuron ``i`` by its *expected
residual-direct attribution to the realized next token*,

    twera_i = (1 / T) * sum_t  a_i(t) * W_U[target_t, i]

where ``a_i(t)`` is the neuron's activation at step ``t``, ``target_t`` is the
token the model actually produced at that step, and ``W_U`` is the unembedding.
This is the **expected** (mean over the trace's steps) **residual** (residual-
stream direct path) **attribution** to the **target**-weighted (realized-token)
logit — the four pieces of the TWERA name.

Caveats, stated honestly (mirroring the paper's own):
- Single-trace: the "coactivation"/expectation is over this trace's own steps,
  not a large prompt dataset, so it is a small-sample estimate.
- Residual-direct only: the attention-direct path is not included.
- The final LayerNorm before the unembedding is omitted (a logit-lens-style
  linearization), so the effect is approximate.
- Only submodules whose output lives in the residual basis (vector length ==
  hidden_dim, e.g. ``resid_post``/``mlp_out``/``o_proj``) are scored; others are
  skipped.
"""

from __future__ import annotations

from typing import Any


def compute_neuron_attribution(
    *,
    trace: list[dict[str, Any]],
    target_token_ids: list[int],
    unembedding: Any,
    top_n: int = 16,
) -> dict[str, Any] | None:
    """Rank each captured (layer, submodule)'s neurons by the TWERA-style score.

    Args:
        trace: the per-step entries from ``generate_with_adaptive_probe``. Each
            entry that carries full activations exposes them at
            ``entry["_activation_full_stats"].layer_tensors[(layer, submodule)]``
            as a ``[hidden_dim]`` tensor. Entries without full stats are skipped,
            so this needs ``--capture-full-activations``.
        target_token_ids: realized next-token id per step, parallel to ``trace``.
        unembedding: the model's output-embedding weight ``W_U`` of shape
            ``[vocab, hidden]`` (``model.get_output_embeddings().weight``).
        top_n: how many top neurons to keep per (layer, submodule).

    Returns:
        A ``neuron_attribution`` dict (see ``docs/web/trace.schema.json``), or
        ``None`` when no full activations / unembedding are available.
    """
    import torch

    if unembedding is None or not trace:
        return None
    w_u = unembedding
    if getattr(w_u, "ndim", 0) != 2:
        return None
    vocab, hidden = int(w_u.shape[0]), int(w_u.shape[1])
    device = w_u.device

    sums: dict[tuple[int, str], Any] = {}
    act_sums: dict[tuple[int, str], Any] = {}
    counts: dict[tuple[int, str], int] = {}
    n_steps = 0

    with torch.no_grad():
        for entry, tok in zip(trace, target_token_ids):
            full = entry.get("_activation_full_stats")
            tensors = getattr(full, "layer_tensors", None) if full is not None else None
            if not tensors or not (0 <= int(tok) < vocab):
                continue
            u_row = w_u[int(tok)].to(device=device, dtype=torch.float32)  # [hidden]
            used = False
            for (layer, submodule), vec in tensors.items():
                if vec is None:
                    continue
                v = vec.reshape(-1).to(device=device, dtype=torch.float32)
                if v.shape[0] != hidden:
                    continue  # not residual-basis (e.g. an intermediate dim) — skip
                key = (int(layer), str(submodule))
                if key not in sums:
                    sums[key] = torch.zeros(hidden, dtype=torch.float32, device=device)
                    act_sums[key] = torch.zeros(hidden, dtype=torch.float32, device=device)
                    counts[key] = 0
                sums[key] += v * u_row
                act_sums[key] += v
                counts[key] += 1
                used = True
            if used:
                n_steps += 1

    if not sums:
        return None

    layers: list[dict[str, Any]] = []
    for key in sorted(sums.keys()):
        layer, submodule = key
        n = max(1, counts[key])
        twera = sums[key] / n
        mean_act = act_sums[key] / n
        k = min(int(top_n), hidden)
        top_vals, top_idx = torch.topk(twera, k)
        neurons = [
            {
                "index": int(idx),
                "twera": float(val),
                "mean_activation": float(mean_act[int(idx)].item()),
            }
            for val, idx in zip(top_vals.tolist(), top_idx.tolist())
        ]
        layers.append({"layer": layer, "submodule": submodule, "neurons": neurons})

    return {
        "method": "twera_approx",
        "n_steps": int(n_steps),
        "note": (
            "Single-trace approximation of Target-Weighted Expected Residual "
            "Attribution: mean over steps of activation_i * W_U[target_token, i] "
            "(residual-direct path; final LayerNorm omitted). Ranks each "
            "(layer, submodule)'s neurons by their average contribution to the "
            "realized next-token logit."
        ),
        "layers": layers,
    }
