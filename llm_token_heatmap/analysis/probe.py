"""Supervised linear probes — is a per-position scalar encoded in the activations?

:mod:`manifold` looks for structure *unsupervised* (PCA, curvature, periodicity).
But a task scalar like "characters since the last line break" — the quantity the
line-break paper (transformer-circuits.pub/2025/linebreaks) shows a model tracking
on a helix — is usually **not** the dominant direction of variance, so unsupervised
PCA misses it even when a linear probe decodes it at R² ≈ 0.8. This module fits
that probe: regress the scalar onto a low-rank PCA basis of one
``(layer, submodule)`` activation cloud with k-fold cross-validation, reporting how
linearly decodable the scalar is and the per-position decoded value.

The PCA basis is unsupervised (it never sees the scalar), so reducing to it before
the ridge fit only regularises the ``T << d`` regime; the supervised step (the
ridge regression of the scalar) is the part that is cross-validated.

numpy-only, like :mod:`manifold`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .manifold import _as_matrix, pca_projection

PROBE_SCHEMA_VERSION = "1.0.0"


def line_position_scalar(token_texts: list[str]) -> list[float]:
    """Characters since the last newline at the *start* of each token.

    This is the visual column / current line length a model must track to decide
    when to insert a hard line break — the scalar the line-break paper finds on a
    helical manifold. Computed purely from the generated token strings.
    """
    col = 0
    out: list[float] = []
    for token in token_texts:
        text = token if isinstance(token, str) else ""
        out.append(float(col))
        newline = text.rfind("\n")
        if newline >= 0:
            col = len(text) - newline - 1
        else:
            col += len(text)
    return out


# Built-in scalars derivable from the generated token texts.
SCALARS: dict[str, Callable[[list[str]], list[float]]] = {
    "line_position": line_position_scalar,
}


def _standardize(z: np.ndarray) -> np.ndarray:
    mu = z.mean(axis=0, keepdims=True)
    sd = z.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (z - mu) / sd


def _ridge_weights(z: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    k = z.shape[1]
    a = z.T @ z + ridge * np.eye(k)
    return np.linalg.solve(a, z.T @ y)


def _full_fit(
    z: np.ndarray, y_centered: np.ndarray, ss_tot: float, ridge: float
) -> tuple[float, np.ndarray]:
    """In-sample R² and predictions of ridge-regressing centered y on z."""
    pred = z @ _ridge_weights(z, y_centered, ridge)
    r2 = float(1.0 - float(((y_centered - pred) ** 2).sum()) / ss_tot)
    return r2, pred


def _cv_r2(
    z: np.ndarray, y: np.ndarray, ridge: float, folds: int
) -> tuple[float | None, int]:
    """k-fold CV R² of ridge-regressing y on standardized scores z.

    Returns ``(r2, n_folds)``; ``r2`` is ``None`` when y has no variance or there
    are too few points to cross-validate.
    """
    n = y.shape[0]
    yc = y - float(y.mean())
    ss_tot = float((yc**2).sum())
    if ss_tot <= 0:
        return None, 0
    nf = max(2, min(folds, n // 2))
    if n < 2 * nf:
        return None, 0
    k = z.shape[1]
    order = np.arange(n)
    preds = np.zeros(n)
    for f in range(nf):
        test = order[f::nf]
        train = np.setdiff1d(order, test)
        if train.size < k + 1:
            return None, 0
        preds[test] = z[test] @ _ridge_weights(z[train], yc[train], ridge)
    return float(1.0 - float(((yc - preds) ** 2).sum()) / ss_tot), int(nf)


def _standardized_scores(x: Any, n_components: int) -> np.ndarray | None:
    """Top-``n_components`` PCA scores of ``x``, column-standardized (or None)."""
    scores = pca_projection(_as_matrix(x), n_components=n_components)
    if scores.shape[1] == 0:
        return None
    return _standardize(scores)


def linear_probe(
    x: Any,
    scalar: Any,
    *,
    n_components: int = 8,
    ridge: float = 1.0,
    folds: int = 5,
) -> dict[str, Any]:
    """Cross-validated linear probe of ``scalar`` against activations ``x``.

    Args:
        x: Activation cloud ``(T, d)`` — one row per position.
        scalar: Per-position target values, length ``T``.
        n_components: PCA rank to reduce ``x`` to before regressing.
        ridge: L2 penalty for the ridge fit.
        folds: k for k-fold CV (auto-shrunk for small ``T``).

    Returns:
        ``{r2_cv, r2_full, decoded, n_components, cv_folds}``. ``r2_cv`` /
        ``r2_full`` are ``None`` when the probe can't be fit (too few points, a
        constant scalar, or no variance). ``decoded`` is the per-position
        decoded scalar from the full-data fit (or ``None``).
    """
    s = np.asarray(scalar, dtype=np.float64).ravel()
    n = _as_matrix(x).shape[0]
    out: dict[str, Any] = {
        "r2_cv": None,
        "r2_full": None,
        "decoded": None,
        "n_components": 0,
        "cv_folds": 0,
    }
    if n < 3 or s.size != n:
        return out
    s_mean = float(s.mean())
    s_centered = s - s_mean
    ss_tot = float((s_centered**2).sum())
    if ss_tot <= 0:  # constant scalar — nothing to decode
        return out
    z = _standardized_scores(x, n_components)
    if z is None:
        return out
    out["n_components"] = int(z.shape[1])

    r2_full, pred_full = _full_fit(z, s_centered, ss_tot, ridge)
    out["r2_full"] = r2_full
    out["decoded"] = [float(v + s_mean) for v in pred_full]

    r2_cv, nf = _cv_r2(z, s, ridge, folds)
    out["r2_cv"] = r2_cv
    out["cv_folds"] = nf
    return out


def circular_features(scalar: Any, period: float) -> tuple[np.ndarray, np.ndarray]:
    """``(cos, sin)`` of ``2π · scalar / period`` — the circular coordinate of a
    helix with the given pitch."""
    theta = 2.0 * np.pi * np.asarray(scalar, dtype=np.float64).ravel() / period
    return np.cos(theta), np.sin(theta)


def helix_probe(
    x: Any,
    scalar: Any,
    *,
    n_components: int = 8,
    ridge: float = 1.0,
    folds: int = 5,
    max_periods: int = 48,
) -> dict[str, Any]:
    """Test for a *helical* (periodic) encoding of ``scalar``.

    A helix encodes a scalar so that, beyond a linear ramp, the activation also
    carries a circular coordinate ``cos/sin(2π · s / p)``. The catch: a plain
    linear ramp aliases onto a period≈range cosine and masquerades as a helix.
    So this first **removes the linear-scalar direction** from the activation
    scores and only then measures, per candidate period, how decodable the
    circular coordinate is *in the residual* (CV R², averaged over cos and sin).
    A real helix keeps a high residual circular R² at some period; a plain ramp
    (or a single bend) collapses to near zero once its ramp direction is gone.
    High residual circular R² together with a high linear probe is the helix
    signature.

    Returns ``{best_period, r2_cv, r2_full, n_periods}`` (Nones when the scalar's
    range or the cloud is too small to resolve a period).
    """
    s = np.asarray(scalar, dtype=np.float64).ravel()
    n = _as_matrix(x).shape[0]
    out: dict[str, Any] = {
        "best_period": None,
        "r2_cv": None,
        "r2_full": None,
        "n_periods": 0,
    }
    if n < 6 or s.size != n:
        return out
    s_range = float(s.max() - s.min())
    if s_range < 3.0:  # can't resolve a period from too little spread
        return out
    z = _standardized_scores(x, n_components)
    if z is None:
        return out

    # Project out the linear-scalar (ramp) direction so the circular test sees
    # only periodic structure orthogonal to the ramp.
    s_centered = s - float(s.mean())
    if float((s_centered**2).sum()) > 0:
        w = _ridge_weights(z, s_centered, ridge)
        norm = float(np.linalg.norm(w))
        if norm > 0:
            u = w / norm
            z = z - np.outer(z @ u, u)

    # Periods from 3 (period 2 is the Nyquist-trivial alias) up to the range.
    p_max = max(3, int(round(s_range)))
    candidates = list(range(3, p_max + 1))
    if len(candidates) > max_periods:
        candidates = sorted(
            {int(round(p)) for p in np.linspace(2, p_max, max_periods)}
        )
    out["n_periods"] = len(candidates)

    best_period: int | None = None
    best_r2: float | None = None
    for p in candidates:
        cos_f, sin_f = circular_features(s, p)
        r2c, _ = _cv_r2(z, cos_f, ridge, folds)
        r2s, _ = _cv_r2(z, sin_f, ridge, folds)
        if r2c is None or r2s is None:
            continue
        r2 = 0.5 * (r2c + r2s)
        if best_r2 is None or r2 > best_r2:
            best_r2 = r2
            best_period = p
    if best_period is None or best_r2 is None:
        return out

    out["best_period"] = float(best_period)
    out["r2_cv"] = float(best_r2)
    # In-sample R² at the winning period (averaged over cos and sin).
    fulls: list[float] = []
    for feat in circular_features(s, best_period):
        fc = feat - float(feat.mean())
        sst = float((fc**2).sum())
        if sst > 0:
            fulls.append(_full_fit(z, fc, sst, ridge)[0])
    out["r2_full"] = float(np.mean(fulls)) if fulls else None
    return out
