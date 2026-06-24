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
    mat = _as_matrix(x)
    s = np.asarray(scalar, dtype=np.float64).ravel()
    n = mat.shape[0]
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

    scores = pca_projection(mat, n_components=n_components)
    if scores.shape[1] == 0:
        return out
    z = _standardize(scores)
    k = z.shape[1]
    out["n_components"] = int(k)

    # Full-data fit → in-sample R² and the decoded scalar per position.
    w = _ridge_weights(z, s_centered, ridge)
    pred_full = z @ w
    out["r2_full"] = float(1.0 - float(((s_centered - pred_full) ** 2).sum()) / ss_tot)
    out["decoded"] = [float(v + s_mean) for v in pred_full]

    # k-fold CV: the honest measure of decodability.
    nf = max(2, min(folds, n // 2))
    if n >= 2 * nf:
        order = np.arange(n)
        preds = np.zeros(n)
        ok = True
        for f in range(nf):
            test = order[f::nf]
            train = np.setdiff1d(order, test)
            if train.size < k + 1:
                ok = False
                break
            wf = _ridge_weights(z[train], s_centered[train], ridge)
            preds[test] = z[test] @ wf
        if ok:
            out["r2_cv"] = float(1.0 - float(((s_centered - preds) ** 2).sum()) / ss_tot)
            out["cv_folds"] = int(nf)
    return out
