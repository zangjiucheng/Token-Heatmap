"""Manifold analysis of captured activations — a numpy-only toolbox.

Inspired by Anthropic's *"When Models Manipulate Manifolds: The Geometry of a
Counting Task"* (transformer-circuits.pub/2025/linebreaks), which shows a model
encoding a scalar (characters since the last line break) on a low-dimensional,
curved — often helical — manifold in activation space. This module gives the
toolbox the geometric primitives to look for the same structure in any captured
activation cloud:

Given a matrix ``X`` of shape ``(T, d)`` — ``T`` token positions / generation
steps stacked along the hidden dimension ``d`` for one ``(layer, submodule)`` —
it computes:

* **PCA spectrum** and **participation ratio** — how few dimensions the cloud
  actually lives in (the "low-dimensional" claim, quantified).
* **TwoNN intrinsic dimension** (Facco et al. 2017) — a geometry-based estimate
  that, unlike PCA, sees curvature: a 1-D curve coiled in 3-D reads as ~1.
* **PCA projection** — 2-D/3-D coordinates per position for plotting the
  manifold itself.
* **Trajectory curvature** — how sharply the position-ordered path bends, the
  signature of a curved manifold.
* **Periodicity** — an FFT on the leading component, the signature of a helix /
  circular coordinate.

Everything here depends only on :mod:`numpy` (no torch, no other package
modules), so the analysis runs anywhere numpy is installed — including a laptop
that never loaded the model. :func:`analyze_manifold` ties the primitives
together into a JSON-shaped dict matching the ``Manifold`` ``$defs`` in
``docs/web/trace.schema.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Semver of the ``manifold`` payload shape emitted into the trace JSON. Mirrors
# the pattern used by the trace/activation schema versions.
MANIFOLD_SCHEMA_VERSION = "1.0.0"

# Cap how many eigenvalues / projection components land in the JSON payload so a
# wide hidden dim (or long generation) can't bloat the trace file.
_MAX_SPECTRUM = 64


def _as_matrix(x: Any) -> np.ndarray:
    """Coerce array-like activations to a 2-D ``float64`` ``(T, d)`` matrix."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (T, d) activation matrix, got shape {arr.shape}")
    return arr


def _finite_or_none(value: float) -> float | None:
    """Map NaN/inf to ``None`` so the value survives a strict JSON round-trip."""
    return float(value) if np.isfinite(value) else None


def pca_spectrum(x: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, explained_variance_ratio)`` of the centered cloud.

    Computed via the SVD of the mean-centered matrix, which is stable and cheap
    even when ``d >> T`` (the usual case: a few tokens in a wide hidden dim).
    Eigenvalues are the covariance eigenvalues ``s**2 / (T - 1)`` in descending
    order; the ratio sums to 1 (or is empty when there is no variance).
    """
    mat = _as_matrix(x)
    n = mat.shape[0]
    centered = mat - mat.mean(axis=0, keepdims=True)
    if n < 2:
        return np.zeros(0), np.zeros(0)
    # Singular values of the centered data; covariance eigenvalues = s**2/(n-1).
    sv = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = (sv**2) / (n - 1)
    total = float(eigenvalues.sum())
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return eigenvalues, ratio


def participation_ratio(eigenvalues: Any) -> float:
    """Effective dimensionality ``(Σλ)² / Σλ²``.

    Equals 1 when all variance is in one direction and ``k`` when variance is
    spread evenly over ``k`` directions — a smooth, basis-free count of "how
    many dimensions matter".
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    denom = float((ev**2).sum())
    if denom <= 0:
        return 0.0
    return float((ev.sum() ** 2) / denom)


def intrinsic_dimension_twonn(x: Any, *, discard_fraction: float = 0.1) -> float:
    """Estimate intrinsic dimension via the TwoNN estimator (Facco et al. 2017).

    For each point, ``mu = r2 / r1`` is the ratio of its second- to first-
    nearest-neighbour distance. Under a locally uniform density ``mu`` is
    Pareto-distributed with the intrinsic dimension as its shape parameter, so
    fitting ``-ln(1 - F(mu))`` against ``ln(mu)`` through the origin recovers
    that dimension. The most extreme ``discard_fraction`` of ratios is dropped
    to blunt heavy-tail noise. Returns ``nan`` when there are too few distinct
    points to estimate.

    Caveat: TwoNN assumes near-Poisson sampling of the manifold. A short,
    regularly-spaced trajectory (e.g. one activation per generation step) makes
    a point's two nearest neighbours nearly equidistant, which is degenerate and
    yields an unstable estimate — prefer :func:`participation_ratio` there. TwoNN
    shines on larger clouds pooled across many contexts.
    """
    mat = _as_matrix(x)
    n = mat.shape[0]
    if n < 4:
        return float("nan")

    # Pairwise Euclidean distances; the diagonal (self) is set to +inf.
    diff = mat[:, None, :] - mat[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, np.inf)

    dist.sort(axis=1)
    r1 = dist[:, 0]
    r2 = dist[:, 1]
    # Keep points with a strictly positive nearest-neighbour distance so mu is
    # finite and well-defined (degenerate duplicates would give r1 == 0).
    valid = (r1 > 0) & np.isfinite(r2)
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]
    if mu.size < 4:
        return float("nan")

    mu = np.sort(mu)
    n_mu = mu.size
    # Empirical CDF F(mu_(i)) = i / N over the *full* sorted set, then drop the
    # heaviest `discard_fraction` (which also removes the F == 1 point whose
    # -log(1 - F) would diverge). The remaining pairs are fit through the origin:
    # slope = Σ(x·y)/Σ(x²) is the dimension estimate.
    f = np.arange(1, n_mu + 1) / n_mu
    keep = max(4, int(round(n_mu * (1.0 - discard_fraction))))
    keep = min(keep, n_mu - 1)  # never include the diverging F == 1 tail point
    x_fit = np.log(mu[:keep])
    y_fit = -np.log1p(-f[:keep])
    denom = float((x_fit**2).sum())
    if denom <= 0:
        return float("nan")
    return float((x_fit * y_fit).sum() / denom)


def pca_projection(x: Any, n_components: int = 3) -> np.ndarray:
    """Project the cloud onto its top ``n_components`` principal axes.

    Returns the PC *scores* ``(T, k)`` with ``k = min(n_components, T, d)`` —
    i.e. the coordinates to scatter-plot the manifold. Columns are ordered by
    descending explained variance.
    """
    mat = _as_matrix(x)
    n = mat.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    centered = mat - mat.mean(axis=0, keepdims=True)
    k = int(max(1, min(n_components, mat.shape[0], mat.shape[1])))
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :k] * s[:k]
    return scores


def trajectory_curvature(coords: Any) -> tuple[float, np.ndarray]:
    """Discrete curvature of the position-ordered trajectory.

    For each interior point the turning angle between the incoming and outgoing
    segments is normalised by the local segment length, approximating the curve
    curvature ``|dθ/ds|``. Endpoints are ``nan``. Returns ``(mean_interior,
    per_position)``. A straight path → ~0; a tightly coiled helix → large.
    """
    pts = _as_matrix(coords)
    n = pts.shape[0]
    per = np.full(n, np.nan)
    if n < 3:
        return float("nan"), per

    for i in range(1, n - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        la = float(np.linalg.norm(a))
        lb = float(np.linalg.norm(b))
        if la <= 0 or lb <= 0:
            continue
        cos_theta = float(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
        per[i] = theta / (0.5 * (la + lb))

    interior = per[1 : n - 1]
    finite = interior[np.isfinite(interior)]
    mean = float(finite.mean()) if finite.size else float("nan")
    return mean, per


def detect_periodicity(signal: Any) -> dict[str, Any]:
    """FFT-based periodicity of a 1-D signal (e.g. PC1 over token position).

    Removes the mean, takes the real FFT, and reports the dominant non-DC
    period together with its normalised spectral power (peak / total). A strong,
    isolated peak is the signature of a circular / helical coordinate — exactly
    the line-break "counting" manifold the paper describes. ``dominant_period``
    is ``None`` when the signal is too short or flat.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    n = sig.size
    if n < 4:
        return {"dominant_period": None, "power": 0.0, "peak_frequency": None}
    sig = sig - sig.mean()
    if not np.any(sig):
        return {"dominant_period": None, "power": 0.0, "peak_frequency": None}

    spectrum = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Ignore the DC bin; find the strongest remaining frequency.
    ac = spectrum[1:]
    if ac.size == 0 or float(ac.sum()) <= 0:
        return {"dominant_period": None, "power": 0.0, "peak_frequency": None}
    idx = int(np.argmax(ac))
    peak_freq = float(freqs[1 + idx])
    power = float(ac[idx] / ac.sum())
    period = float(1.0 / peak_freq) if peak_freq > 0 else None
    return {"dominant_period": period, "power": power, "peak_frequency": peak_freq}


def analyze_manifold(
    x: Any,
    *,
    positions: Any = None,
    n_components: int = 3,
) -> dict[str, Any]:
    """Run the full manifold analysis on one ``(layer, submodule)`` cloud.

    Args:
        x: Array-like activations of shape ``(T, d)`` — one row per token
            position / generation step.
        positions: Optional integer labels for each row (defaults to
            ``range(T)``); echoed back so a caller can align the projection
            coordinates with steps.
        n_components: How many PCA components to keep for the projection.

    Returns:
        A JSON-serializable dict matching the ``Manifold`` layer-entry shape in
        ``docs/web/trace.schema.json``. NaN/inf scalars are emitted as ``None``.
    """
    mat = _as_matrix(x)
    t, d = mat.shape
    pos = list(range(t)) if positions is None else [int(p) for p in positions]

    eigenvalues, ratio = pca_spectrum(mat)
    spectrum_len = min(_MAX_SPECTRUM, eigenvalues.size)
    cumulative = np.cumsum(ratio[:spectrum_len]) if ratio.size else np.zeros(0)

    coords = pca_projection(mat, n_components=n_components)
    pc1 = coords[:, 0] if coords.shape[1] else np.zeros(t)
    curv_mean, curv_per = trajectory_curvature(coords)

    return {
        "n_positions": int(t),
        "hidden_dim": int(d),
        "positions": pos,
        "pca": {
            "eigenvalues": [float(v) for v in eigenvalues[:spectrum_len]],
            "explained_variance_ratio": [float(v) for v in ratio[:spectrum_len]],
            "cumulative_variance_ratio": [float(v) for v in cumulative],
        },
        "participation_ratio": _finite_or_none(participation_ratio(eigenvalues)),
        "intrinsic_dimension": {
            "twonn": _finite_or_none(intrinsic_dimension_twonn(mat)),
        },
        "projection": {
            "n_components": int(coords.shape[1]),
            "coords": [[float(v) for v in row] for row in coords],
        },
        "trajectory_curvature": {
            "mean": _finite_or_none(curv_mean),
            "per_position": [_finite_or_none(v) for v in curv_per],
        },
        "periodicity": detect_periodicity(pc1),
    }
