"""Tests for the supervised linear-probe toolbox (``llm_token_heatmap.probe``)."""

from __future__ import annotations

import numpy as np

from llm_token_heatmap.probe import (
    circular_features,
    helix_probe,
    line_position_scalar,
    linear_probe,
)


def test_line_position_basic() -> None:
    # Column at the start of each token; advances by token length, resets at \n.
    assert line_position_scalar(["ab", "c\nde", "f"]) == [0.0, 2.0, 2.0]


def test_line_position_resets_after_newline() -> None:
    # After "abc\n" the column is 0; the next token starts a fresh line.
    assert line_position_scalar(["abc\n", "x"]) == [0.0, 0.0]
    # No newlines → monotonically increasing column.
    assert line_position_scalar(["a", "bb", "c"]) == [0.0, 1.0, 3.0]


def test_line_position_trailing_chars_after_newline() -> None:
    # The reset column counts chars *after* the last newline in the token.
    assert line_position_scalar(["xx\nyy"]) == [0.0]
    assert line_position_scalar(["xx\nyy", "z"]) == [0.0, 2.0]


def _cloud_encoding(scalar: np.ndarray, d: int = 32, noise: float = 0.05, seed: int = 0):
    """Build an activation cloud whose dominant variation is `scalar` along a
    random direction (plus a little noise)."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    base = rng.standard_normal((scalar.size, d)) * noise
    return scalar[:, None] * direction[None, :] + base


def test_probe_recovers_an_encoded_scalar() -> None:
    s = np.linspace(0, 10, 48)
    x = _cloud_encoding(s)
    result = linear_probe(x, s, n_components=6)
    assert result["r2_cv"] is not None
    assert result["r2_cv"] > 0.8
    assert result["r2_full"] > result["r2_cv"] - 1e-9
    assert result["decoded"] is not None
    assert len(result["decoded"]) == s.size


def test_probe_rejects_an_unrelated_scalar() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((48, 32))
    s = rng.standard_normal(48)  # independent of x
    result = linear_probe(x, s, n_components=6)
    # An unrelated scalar must not decode well out of sample.
    assert result["r2_cv"] is not None
    assert result["r2_cv"] < 0.5


def test_probe_handles_constant_and_tiny_inputs() -> None:
    x = np.random.default_rng(2).standard_normal((10, 8))
    constant = linear_probe(x, np.full(10, 3.0))
    assert constant["r2_cv"] is None and constant["r2_full"] is None

    tiny = linear_probe(np.zeros((2, 8)), np.array([0.0, 1.0]))
    assert tiny["r2_cv"] is None


def test_circular_features_shape_and_values() -> None:
    cos, sin = circular_features([0.0, 2.0, 4.0], period=8.0)
    assert cos[0] == 1.0 and abs(sin[0]) < 1e-12
    # quarter turn at s = period/4 = 2.
    assert abs(cos[1]) < 1e-12 and abs(sin[1] - 1.0) < 1e-12


def _helix_cloud(s: np.ndarray, period: float, d: int = 24, noise: float = 0.03, seed: int = 0):
    """A cloud carrying a linear ramp of s plus a circular cos/sin(2π s/p) on
    orthogonal directions — i.e. a helix."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    ramp_dir, cos_dir, sin_dir = q[:, 0], q[:, 1], q[:, 2]
    cos, sin = circular_features(s, period)
    base = rng.standard_normal((s.size, d)) * noise
    return (
        (0.3 * s)[:, None] * ramp_dir[None, :]
        + cos[:, None] * cos_dir[None, :]
        + sin[:, None] * sin_dir[None, :]
        + base
    )


def test_helix_probe_recovers_planted_period() -> None:
    s = np.linspace(0, 24, 60)  # three turns of a period-8 helix
    x = _helix_cloud(s, period=8.0)
    result = helix_probe(x, s, n_components=6)
    assert result["best_period"] is not None
    assert abs(result["best_period"] - 8.0) <= 1.0  # discretised period
    assert result["r2_cv"] > 0.7


def test_helix_probe_rejects_a_plain_linear_ramp() -> None:
    s = np.linspace(0, 24, 60)
    rng = np.random.default_rng(3)
    direction = rng.standard_normal(24)
    x = s[:, None] * direction[None, :] + rng.standard_normal((60, 24)) * 0.05
    result = helix_probe(x, s, n_components=6)
    # No circular component → the best period's circular coordinate is not
    # linearly decodable.
    assert result["r2_cv"] is None or result["r2_cv"] < 0.5


def test_helix_probe_needs_enough_range() -> None:
    x = np.random.default_rng(4).standard_normal((8, 12))
    small = helix_probe(x, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 1.0, 0.5]))
    assert small["best_period"] is None
