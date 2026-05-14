"""Smoke tests for the plotting module."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from llm_token_heatmap.attention_probe import AttentionLayerStats, AttentionStats
from llm_token_heatmap.attention_stats import compute_attention_stats
from llm_token_heatmap.plotting import (
    plot_activation_delta,
    plot_adaptive_heatmap,
    plot_attention_layer_head_grid,
    plot_attention_pattern,
    plot_entropy,
    plot_raw_vs_processed_heatmap,
    plot_raw_vs_processed_selected_prob,
    plot_selected_probability,
)


def _synthetic_dataframe(
    num_steps: int = 5,
    k_used: int = 4,
    sources: tuple[str, ...] = ("raw", "processed"),
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for step in range(num_steps):
        entropy = float(rng.uniform(0.5, 3.0))
        selected_prob = float(rng.uniform(0.1, 0.9))
        for source in sources:
            for rank in range(1, k_used + 1):
                prob = float(rng.uniform(0.01, 0.5))
                rows.append(
                    {
                        "step": step,
                        "source": source,
                        "rank": rank,
                        "token_id": rank * 10 + step,
                        "token": f"t{step}_{rank}",
                        "prob": prob,
                        "logprob": float(np.log(prob + 1e-12)),
                        "selected_token_id": 7,
                        "selected_token": "<sel>",
                        "selected_prob": selected_prob,
                        "selected_logprob": float(np.log(selected_prob + 1e-12)),
                        "selected_rank": 1,
                        "entropy": entropy,
                        "k_used": k_used,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _synthetic_dataframe()


def test_plot_heatmap_smoke(synthetic_df, tmp_path):
    save_path = tmp_path / "heatmap.png"
    fig = plot_adaptive_heatmap(
        synthetic_df, value_col="logprob", save_path=save_path, annotate=False
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_plot_heatmap_with_annotations(synthetic_df, tmp_path):
    save_path = tmp_path / "heatmap_annot.png"
    fig = plot_adaptive_heatmap(synthetic_df, value_col="prob", save_path=save_path, annotate=True)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_plot_heatmap_rejects_unknown_value_col(synthetic_df):
    with pytest.raises(ValueError):
        plot_adaptive_heatmap(synthetic_df, value_col="not_a_column")


def test_plot_entropy_smoke(synthetic_df, tmp_path):
    save_path = tmp_path / "entropy.png"
    fig = plot_entropy(synthetic_df, save_path=save_path)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def _synthetic_activation_diff() -> dict:
    """Build a `compare_activations`-shaped dict with two steps, two layers, two submodules."""

    layer_delta = lambda layer, sub, l2, cos: {  # noqa: E731
        "layer": layer,
        "submodule": sub,
        "l2": l2,
        "cosine": cos,
        "top_changed_neurons": [{"index": 0, "delta": l2}],
    }

    return {
        "schema_version": "1.0.0",
        "alignment": {
            "mode": "token_id",
            "tokenizer_a_fingerprint": "sha256:a",
            "tokenizer_b_fingerprint": "sha256:a",
            "mismatches": [],
        },
        "steps": [
            {
                "step": 0,
                "token_id_a": 1,
                "token_id_b": 1,
                "decoded_text_offset_a": 0,
                "decoded_text_offset_b": 0,
                "delta": [
                    layer_delta(0, "resid_post", 0.5, 0.9),
                    layer_delta(1, "resid_post", 0.7, 0.8),
                    layer_delta(0, "mlp_out", 0.1, 0.99),
                    layer_delta(1, "mlp_out", 0.2, 0.95),
                ],
            },
            {
                "step": 1,
                "token_id_a": 2,
                "token_id_b": 2,
                "decoded_text_offset_a": 4,
                "decoded_text_offset_b": 4,
                "delta": [
                    layer_delta(0, "resid_post", 0.3, 0.92),
                    layer_delta(1, "resid_post", 0.4, 0.85),
                    layer_delta(0, "mlp_out", 0.08, 0.99),
                    layer_delta(1, "mlp_out", 0.15, 0.96),
                ],
            },
        ],
    }


def test_plot_activation_delta_smoke(tmp_path):
    save_path = tmp_path / "activation_delta.png"
    diff = _synthetic_activation_diff()
    fig = plot_activation_delta(diff, save_path=save_path, metric="l2")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_plot_activation_delta_rejects_unknown_metric():
    diff = _synthetic_activation_diff()
    with pytest.raises(ValueError):
        plot_activation_delta(diff, metric="manhattan")


def test_plot_selected_probability_smoke(synthetic_df, tmp_path):
    save_path = tmp_path / "selected_prob.png"
    fig = plot_selected_probability(synthetic_df, save_path=save_path)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_plot_functions_return_figure_without_save(synthetic_df):
    fig_h = plot_adaptive_heatmap(synthetic_df, save_path=None, annotate=False)
    fig_e = plot_entropy(synthetic_df, save_path=None)
    fig_s = plot_selected_probability(synthetic_df, save_path=None)

    assert isinstance(fig_h, matplotlib.figure.Figure)
    assert isinstance(fig_e, matplotlib.figure.Figure)
    assert isinstance(fig_s, matplotlib.figure.Figure)

    plt.close(fig_h)
    plt.close(fig_e)
    plt.close(fig_s)


def _synthetic_attention_stats(num_layers: int = 2, num_heads: int = 4, seq_len: int = 5):
    torch.manual_seed(0)
    layers = {}
    head_dim = 4
    for layer_idx in range(num_layers):
        weights = torch.softmax(torch.randn(num_heads, seq_len), dim=-1)
        layers[layer_idx] = AttentionLayerStats(
            layer_idx=layer_idx,
            attention_weights=weights,
            q_last=torch.randn(num_heads, head_dim),
            k_last=torch.randn(num_heads, head_dim),
            v_last=torch.randn(num_heads, head_dim),
        )
    stats = AttentionStats(
        layers=layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
        head_to_kv_group=list(range(num_heads)),
    )
    return compute_attention_stats(stats)


def test_plot_attention_layer_head_grid_smoke(tmp_path):
    derived = _synthetic_attention_stats()
    save_path = tmp_path / "attention_grid.png"
    fig = plot_attention_layer_head_grid(derived, value="entropy", save_path=save_path)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0

    cbar_axes = [ax for ax in fig.axes if ax.get_ylabel() == "entropy"]
    assert cbar_axes, "colorbar must be labelled 'entropy'"
    plt.close(fig)


def test_plot_attention_pattern_smoke(tmp_path):
    derived = _synthetic_attention_stats()
    save_path = tmp_path / "attention_pattern.png"
    fig = plot_attention_pattern(derived, layer=0, head=0, save_path=save_path)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_raw_vs_processed_heatmap_smoke(synthetic_df, tmp_path):
    save_path = tmp_path / "rvp_heatmap.png"
    fig = plot_raw_vs_processed_heatmap(synthetic_df, save_path=save_path, annotate=False)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_raw_vs_processed_selected_prob_smoke(synthetic_df, tmp_path):
    save_path = tmp_path / "rvp_selprob.png"
    fig = plot_raw_vs_processed_selected_prob(synthetic_df, save_path=save_path)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)
