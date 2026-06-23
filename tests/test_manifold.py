"""Tests for the numpy-only manifold-analysis toolbox.

Each test builds synthetic activations whose geometry has a known answer — a
1-D line, a flat plane, a coiled helix — and checks the estimators recover it.
The helix is the key fixture: it is the line-break "counting manifold" shape
from the transformer-circuits paper (a 1-D curve coiled in higher dimensions),
so it exercises intrinsic dimension, curvature, and periodicity at once.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_token_heatmap.manifold import (
    analyze_manifold,
    detect_periodicity,
    intrinsic_dimension_twonn,
    pca_projection,
    pca_spectrum,
    participation_ratio,
    trajectory_curvature,
)


def _orthonormal(d: int, k: int, seed: int) -> np.ndarray:
    """Return ``k`` orthonormal columns in ``R^d``."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q[:, :k]


def _helix(n: int, *, turns: float = 4.0, pitch: float = 0.5, ambient: int = 12, seed: int = 0):
    """An ordered 1-D helix linearly embedded in ``ambient`` dims (regular grid).

    Sampled on a uniform ``linspace`` so the rows form a proper *trajectory* —
    the shape curvature and periodicity estimators rely on this ordering.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi * turns, n)
    base = np.stack([np.cos(t), np.sin(t), pitch * t], axis=1)  # (n, 3)
    embed = rng.standard_normal((3, ambient))
    return base @ embed, t


def _helix_iid(n: int, *, turns: float = 3.0, pitch: float = 0.3, ambient: int = 12, seed: int = 0):
    """A 1-D helix sampled at random ``t`` (i.i.d.), for the TwoNN estimator.

    TwoNN assumes near-Poisson sampling of the manifold; a regular grid makes
    every point's two nearest neighbours equidistant (degenerate), so the
    dimension test must use random samples.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2.0 * np.pi * turns, size=n)
    base = np.stack([np.cos(t), np.sin(t), pitch * t], axis=1)
    embed = rng.standard_normal((3, ambient))
    return base @ embed


# --- PCA spectrum & participation ratio ---------------------------------------


def test_pca_line_is_one_dimensional() -> None:
    direction = _orthonormal(12, 1, seed=0)[:, 0]
    t = np.linspace(-1.0, 1.0, 50)
    x = np.outer(t, direction)
    eigenvalues, ratio = pca_spectrum(x)
    assert ratio[0] > 0.999
    assert participation_ratio(eigenvalues) == pytest.approx(1.0, abs=1e-3)


def test_pca_two_equal_dims_gives_pr_two() -> None:
    basis = _orthonormal(12, 2, seed=1)
    rng = np.random.default_rng(2)
    coeffs = rng.standard_normal((400, 2))
    x = coeffs @ basis.T
    eigenvalues, ratio = pca_spectrum(x)
    assert ratio[0] + ratio[1] > 0.999
    assert participation_ratio(eigenvalues) == pytest.approx(2.0, abs=0.2)


def test_explained_variance_ratio_sums_to_one() -> None:
    x, _ = _helix(80, seed=3)
    _, ratio = pca_spectrum(x)
    assert float(ratio.sum()) == pytest.approx(1.0, abs=1e-9)


def test_participation_ratio_even_spread() -> None:
    k = 5
    basis = _orthonormal(20, k, seed=4)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((600, k)) @ basis.T
    eigenvalues, _ = pca_spectrum(x)
    assert participation_ratio(eigenvalues) == pytest.approx(k, abs=0.7)


# --- Intrinsic dimension (TwoNN) ----------------------------------------------


def test_twonn_recovers_plane_dimension() -> None:
    rng = np.random.default_rng(6)
    coords = rng.standard_normal((500, 2))
    embed = _orthonormal(10, 2, seed=7).T  # (2, 10)
    est = intrinsic_dimension_twonn(coords @ embed)
    assert 1.6 <= est <= 2.5


def test_twonn_sees_helix_as_one_dimensional() -> None:
    # A helix lives in 12-D but is intrinsically a 1-D curve — PCA would report
    # ~3 dims, TwoNN should see ~1 because it is curvature-aware. Sampled i.i.d.
    # so the estimator's assumptions hold.
    est = intrinsic_dimension_twonn(_helix_iid(600, seed=8))
    assert est < 1.6


def test_twonn_too_few_points_is_nan() -> None:
    assert np.isnan(intrinsic_dimension_twonn(np.zeros((3, 5))))


# --- Projection ---------------------------------------------------------------


def test_projection_shape_is_capped_by_rank() -> None:
    x, _ = _helix(40, ambient=12, seed=9)
    coords = pca_projection(x, n_components=3)
    assert coords.shape == (40, 3)
    # Asking for more components than the ambient dim caps at the dim.
    narrow = pca_projection(np.ones((5, 2)) * np.arange(5)[:, None], n_components=3)
    assert narrow.shape[1] <= 2


# --- Trajectory curvature -----------------------------------------------------


def test_curvature_of_straight_line_is_zero() -> None:
    coords = np.outer(np.arange(10), np.array([1.0, 0.0]))
    mean, per = trajectory_curvature(coords)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(per[0]) and np.isnan(per[-1])


def test_curvature_of_unit_circle_is_about_one() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    coords = np.stack([np.cos(t), np.sin(t)], axis=1)
    mean, _ = trajectory_curvature(coords)
    # True curvature of a unit circle is exactly 1; the discrete estimate is close.
    assert 0.7 <= mean <= 1.3


# --- Periodicity --------------------------------------------------------------


def test_periodicity_recovers_sine_period() -> None:
    period = 20
    sig = np.sin(2.0 * np.pi * np.arange(120) / period)
    out = detect_periodicity(sig)
    assert out["dominant_period"] == pytest.approx(period, rel=0.1)
    assert out["power"] > 0.5


def test_periodicity_of_flat_or_short_signal_is_none() -> None:
    assert detect_periodicity(np.zeros(50))["dominant_period"] is None
    assert detect_periodicity([1.0, 2.0])["dominant_period"] is None


# --- End-to-end analyze_manifold ----------------------------------------------


def test_analyze_manifold_shape_and_json_safe() -> None:
    x, _ = _helix(80, seed=10)
    res = analyze_manifold(x)
    for key in (
        "n_positions",
        "hidden_dim",
        "positions",
        "pca",
        "participation_ratio",
        "intrinsic_dimension",
        "projection",
        "trajectory_curvature",
        "periodicity",
    ):
        assert key in res
    assert res["n_positions"] == 80
    assert res["hidden_dim"] == 12
    assert len(res["projection"]["coords"]) == 80
    assert len(res["positions"]) == 80
    # Must serialize cleanly — no NaN/inf leaking past the None mapping.
    json.dumps(res)


def test_analyze_manifold_helix_signature() -> None:
    # An ordered (trajectory) helix: assert the trajectory-appropriate signals —
    # low PCA dimensionality, a strong periodic leading component, and non-zero
    # curvature. (TwoNN is unreliable on regular-grid trajectories, so it isn't
    # asserted here; it's validated separately on i.i.d. samples.)
    x, _ = _helix(300, turns=5.0, seed=11)
    res = analyze_manifold(x)
    assert res["participation_ratio"] < 4.0
    assert res["periodicity"]["power"] > 0.3
    assert res["trajectory_curvature"]["mean"] > 0.0


def test_analyze_manifold_positions_echoed() -> None:
    x, _ = _helix(5, seed=12)
    res = analyze_manifold(x, positions=[10, 20, 30, 40, 50])
    assert res["positions"] == [10, 20, 30, 40, 50]


def test_analyze_manifold_single_position_does_not_crash() -> None:
    res = analyze_manifold(np.zeros((1, 8)))
    assert res["n_positions"] == 1
    assert res["participation_ratio"] == 0.0
    json.dumps(res)
