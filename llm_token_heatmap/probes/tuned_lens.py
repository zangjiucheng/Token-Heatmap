"""Tuned lens — a learned affine probe per layer that maps the residual stream
into the *final* layer's representation before decoding.

The vanilla logit lens (unembed each layer directly) has a basis mismatch in the
early layers, so their "predictions" look like noise even when the layer already
carries the information (Belrose et al., 2023, "Eliciting Latent Predictions
with the Tuned Lens"). The tuned lens fixes this with a per-layer affine
transform applied before the shared unembedding.

This is a **pragmatic, closed-form variant**: each layer's affine map is fit by
ridge least-squares to predict the final hidden state, rather than trained by KL
on the logits as in the paper. It is fast and GPU-free (a 897×897 solve per
layer, no vocab-sized softmax in the loop) and removes the early-layer basis
mismatch, at the cost of optimising representation match instead of logit KL.
Fit on a small bundled corpus so the probes generalise beyond one trace.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"

# A small, domain-diverse corpus — enough positions to fit a ridge-regularised
# affine probe over the hidden dim without overfitting a single trace.
_CORPUS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "She opened the door and found the room completely empty.",
    "In 1969, astronauts first walked on the surface of the Moon.",
    "The quick brown fox jumps over the lazy sleeping dog.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose.",
    "He whispered the answer so no one else could possibly hear it.",
    "The stock market fell sharply after the surprise interest rate hike.",
    "To sort a list in Python you can call the built-in sorted function.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "After the storm passed, a bright rainbow arched over the valley.",
    "Electrons carry a negative charge and orbit the atomic nucleus.",
    "The ancient library held thousands of fragile handwritten scrolls.",
    "When Mary and John went to the store, John gave a drink to Mary.",
    "A prime number is only divisible by one and by itself.",
    "The orchestra tuned their instruments before the concert began.",
    "Rising temperatures are melting glaciers around the polar regions.",
    "The detective examined every clue left behind at the crime scene.",
    "Bees communicate the location of flowers through an intricate dance.",
    "The function returns a new array without mutating its argument.",
    "During the meeting, they debated the merits of the new proposal.",
    "Light from the sun takes about eight minutes to reach the Earth.",
    "The children built an enormous sandcastle near the crashing waves.",
    "Shakespeare wrote both tragedies and comedies for the London stage.",
    "Compilers translate high-level source code into machine instructions.",
    "The nurse carefully checked the patient's temperature and pulse.",
    "A gentle breeze carried the scent of pine through the open window.",
    "The treaty was signed by representatives of all twelve nations.",
    "Gravity causes objects to accelerate toward the center of the planet.",
]


def _hidden_states(model: Any, input_ids: Any) -> Any:
    """Run the model with hidden-state output and return the tuple of residuals
    ``(embeddings, layer_0_out, …, final_out)``."""
    out = model(input_ids, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("model did not return hidden_states")
    return hs


def fit_tuned_lens(
    model: Any,
    tokenizer: Any,
    *,
    corpus: list[str] | None = None,
    ridge: float = 1.0,
    layers: list[int] | None = None,
) -> tuple[dict[int, Any], int]:
    """Fit a per-layer affine probe ``A_L h_L + b_L ≈ h_final`` by ridge
    least-squares over the corpus. Returns ``({layer: W [d+1, d]}, num_layers)``
    where the last row of ``W`` is the bias."""
    import numpy as np
    import torch

    texts = corpus if corpus is not None else _CORPUS
    device = next(model.parameters()).device

    per_layer: dict[int, list[Any]] = {}
    finals: list[Any] = []
    num_layers = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            hs = _hidden_states(model, ids)
            num_layers = len(hs) - 1
            finals.append(hs[-1][0].float().cpu().numpy())
            for layer in range(num_layers):
                per_layer.setdefault(layer, []).append(hs[layer + 1][0].float().cpu().numpy())

    y = np.concatenate(finals, axis=0)  # [N, d]
    layer_list = layers if layers is not None else sorted(per_layer)
    probes: dict[int, Any] = {}
    for layer in layer_list:
        x = np.concatenate(per_layer[layer], axis=0)  # [N, d]
        xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)  # [N, d+1]
        gram = xb.T @ xb
        gram += ridge * np.eye(gram.shape[0], dtype=gram.dtype)
        w = np.linalg.solve(gram, xb.T @ y)  # [d+1, d]
        probes[layer] = w
    return probes, num_layers


def apply_tuned_lens(
    model: Any,
    tokenizer: Any,
    probes: dict[int, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Decode each trace step's residual through the fitted probes + the model's
    real final norm and unembedding, recording the realized token's tuned-lens
    rank and probability per layer (same shape the decision map reads)."""
    import torch

    from llm_token_heatmap.probes.logit_lens import (
        _resolve_final_norm,
        _resolve_lm_head,
    )

    tokens = trace.get("tokens") or {}
    prompt_ids = [int(x) for x in tokens.get("prompt_token_ids", [])]
    steps = trace.get("steps", [])
    if not prompt_ids or not steps:
        raise ValueError(
            "trace lacks tokens.prompt_token_ids or steps; regenerate with `token-heatmap trace`."
        )
    gen_ids = [int(s["selected"]["token_id"]) for s in steps]

    final_norm = _resolve_final_norm(model)
    lm_head = _resolve_lm_head(model)
    if lm_head is None:
        raise RuntimeError("could not locate the model's lm_head / unembedding")

    device = next(model.parameters()).device
    param = next(model.parameters())
    layer_list = sorted(probes)
    weights = {
        layer: torch.tensor(w, dtype=param.dtype, device=device) for layer, w in probes.items()
    }

    out_steps: list[dict[str, Any]] = []
    with torch.no_grad():
        for i, step in enumerate(steps):
            target = gen_ids[i]
            input_ids = prompt_ids + gen_ids[:i]
            if not input_ids:
                continue
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            hs = _hidden_states(model, ids)
            layers_out: list[dict[str, Any]] = []
            for layer in layer_list:
                h = hs[layer + 1][0, -1, :].to(dtype=param.dtype)  # [d]
                w = weights[layer]  # [d+1, d]
                tuned = h @ w[:-1] + w[-1]  # affine map into the final basis
                x = final_norm(tuned) if final_norm is not None else tuned
                logits = lm_head(x).float()
                probs = torch.softmax(logits, dim=-1)
                rank = int((logits > logits[target]).sum().item()) + 1
                layers_out.append(
                    {
                        "layer_idx": int(layer),
                        "selected_token_rank": rank,
                        "selected_token_prob": float(probs[target].item()),
                    }
                )
            out_steps.append({"step": i, "token": step["selected"].get("token", ""), "layers": layers_out})

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "affine-to-final-hidden ridge least-squares (pragmatic tuned lens)",
        "num_layers": max((len(s["layers"]) for s in out_steps), default=0),
        "steps": out_steps,
    }


def compute_tuned_lens(
    model: Any,
    tokenizer: Any,
    trace: dict[str, Any],
    *,
    corpus: list[str] | None = None,
    ridge: float = 1.0,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Fit the probes on the corpus, then apply them to the trace's steps."""
    probes, _ = fit_tuned_lens(model, tokenizer, corpus=corpus, ridge=ridge, layers=layers)
    return apply_tuned_lens(model, tokenizer, probes, trace)
