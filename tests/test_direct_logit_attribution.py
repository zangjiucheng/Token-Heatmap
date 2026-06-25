"""Unit tests for Direct Logit Attribution.

The core property: for standard pre-norm transformers the residual stream is
``h = embed + Σ(o_proj_L + mlp_out_L)``, so folding the final norm as a fixed
scale makes the per-component contributions sum *exactly* to the model's logit
for the target token. We verify that against real RMSNorm / LayerNorm / Gemma
norm modules (``error`` ~ 0), plus the None-returning guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from llm_token_heatmap.direct_logit_attribution import (
    compute_direct_logit_attribution,
)


class _RMSNorm(torch.nn.Module):
    """Llama/Qwen-style RMSNorm: weight * (x / rms(x))."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.variance_epsilon))


class _GemmaRMSNorm(torch.nn.Module):
    """Gemma-style RMSNorm: (1 + weight) * (x / rms(x)). Class name carries
    'gemma' so the (1+weight) branch is exercised."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        var = x.pow(2).mean(-1, keepdim=True)
        return (1.0 + self.weight) * (x * torch.rsqrt(var + self.variance_epsilon))


def _entry(step: int, layer_tensors: dict[tuple[int, str], torch.Tensor]) -> dict:
    return {"step": step, "_activation_full_stats": SimpleNamespace(layer_tensors=layer_tensors)}


def _build_trace(hidden: int, n_layers: int, seed: int = 0):
    """One step whose residual_post is the exact sum of its block deltas."""
    g = torch.Generator().manual_seed(seed)
    attn = [torch.randn(hidden, generator=g) for _ in range(n_layers)]
    mlp = [torch.randn(hidden, generator=g) for _ in range(n_layers)]
    embed = torch.randn(hidden, generator=g)
    h = embed.clone()
    tensors: dict[tuple[int, str], torch.Tensor] = {}
    for layer in range(n_layers):
        tensors[(layer, "o_proj")] = attn[layer]
        tensors[(layer, "mlp_out")] = mlp[layer]
        h = h + attn[layer] + mlp[layer]
    tensors[(n_layers - 1, "residual_post")] = h
    return _entry(0, tensors), h


def test_contributions_sum_to_model_logit_rmsnorm():
    hidden, n_layers, vocab, target = 6, 3, 4, 2
    entry, _h = _build_trace(hidden, n_layers, seed=1)
    w_u = torch.randn(vocab, hidden)
    norm = _RMSNorm(hidden)

    out = compute_direct_logit_attribution(
        trace=[entry],
        target_token_ids=[target],
        unembedding=w_u,
        final_norm=norm,
    )
    assert out is not None
    assert out["method"] == "dla_fold_norm"
    assert out["n_steps"] == 1
    assert out["num_layers"] == n_layers

    step = out["steps"][0]
    assert step["step"] == 0
    assert step["target_token_id"] == target
    assert len(step["layers"]) == n_layers

    explained = step["embed"] + step["bias"] + sum(
        layer["attn"] + layer["mlp"] for layer in step["layers"]
    )
    # Decomposition reconstructs the model logit (RMSNorm is exact).
    assert abs(step["error"]) < 1e-3
    assert abs(explained - step["total_logit"]) < 1e-3
    assert step["bias"] == 0.0


def test_layernorm_bias_is_handled():
    hidden, n_layers, vocab, target = 5, 2, 3, 1
    entry, _h = _build_trace(hidden, n_layers, seed=2)
    w_u = torch.randn(vocab, hidden)
    norm = torch.nn.LayerNorm(hidden)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden))
        norm.bias.copy_(torch.randn(hidden))

    out = compute_direct_logit_attribution(
        trace=[entry], target_token_ids=[target], unembedding=w_u, final_norm=norm
    )
    assert out is not None
    step = out["steps"][0]
    assert step["bias"] != 0.0  # LayerNorm contributes a bias term
    explained = step["embed"] + step["bias"] + sum(
        layer["attn"] + layer["mlp"] for layer in step["layers"]
    )
    assert abs(step["error"]) < 1e-3
    assert abs(explained - step["total_logit"]) < 1e-3


def test_gemma_plus_one_branch_is_exact():
    hidden, n_layers, vocab, target = 5, 2, 3, 0
    entry, _h = _build_trace(hidden, n_layers, seed=3)
    w_u = torch.randn(vocab, hidden)
    norm = _GemmaRMSNorm(hidden)

    out = compute_direct_logit_attribution(
        trace=[entry], target_token_ids=[target], unembedding=w_u, final_norm=norm
    )
    assert out is not None
    step = out["steps"][0]
    # (1 + weight) scaling must be picked up, keeping error ~ 0.
    assert abs(step["error"]) < 1e-3


def test_returns_none_without_full_activations():
    out = compute_direct_logit_attribution(
        trace=[{"step": 0, "raw": {}}],
        target_token_ids=[0],
        unembedding=torch.zeros(3, 4),
    )
    assert out is None


def test_returns_none_without_unembedding():
    entry, _h = _build_trace(4, 1, seed=4)
    out = compute_direct_logit_attribution(
        trace=[entry], target_token_ids=[0], unembedding=None
    )
    assert out is None


def _per_head_entry(hidden, nh, hd, seed):
    """A one-layer entry whose o_proj output == W_O @ z, so per-head splits sum
    to the layer attention contribution."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(nh * hd, generator=g)
    w_o = torch.randn(hidden, nh * hd, generator=g)
    attn_out = w_o @ z
    mlp_out = torch.randn(hidden, generator=g)
    embed = torch.randn(hidden, generator=g)
    h = embed + attn_out + mlp_out
    tensors = {
        (0, "o_proj"): attn_out,
        (0, "mlp_out"): mlp_out,
        (0, "residual_post"): h,
    }
    full = SimpleNamespace(layer_tensors=tensors, attn_z={0: z})
    return {"step": 0, "_activation_full_stats": full}, w_o


def test_per_head_contributions_sum_to_layer_attn():
    hidden, nh, hd, vocab, target = 8, 2, 4, 4, 1
    entry, w_o = _per_head_entry(hidden, nh, hd, seed=7)
    w_u = torch.randn(vocab, hidden)

    out = compute_direct_logit_attribution(
        trace=[entry],
        target_token_ids=[target],
        unembedding=w_u,
        o_proj_weights={0: w_o},
        num_heads=nh,
        head_dim=hd,
    )
    assert out is not None
    layer = out["steps"][0]["layers"][0]
    assert "heads" in layer
    assert [h["head"] for h in layer["heads"]] == [0, 1]
    head_sum = sum(h["attn"] for h in layer["heads"])
    # Per-head contributions reconstruct the layer's attention bar.
    assert abs(head_sum - layer["attn"]) < 1e-3


def test_per_head_absent_without_weights():
    hidden, nh, hd, vocab = 8, 2, 4, 4
    entry, _w_o = _per_head_entry(hidden, nh, hd, seed=8)
    w_u = torch.randn(vocab, hidden)
    out = compute_direct_logit_attribution(
        trace=[entry], target_token_ids=[1], unembedding=w_u
    )
    assert out is not None
    assert "heads" not in out["steps"][0]["layers"][0]
