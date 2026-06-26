#!/usr/bin/env python3
"""Summarise the supervised probe + helix test for a manifold-augmented trace.

Run the analysis first:
    token-heatmap manifold --trace <trace>.json --components 6 --probe line_position

then:
    python3 examples/helix-report.py <trace>.json [submodule]

Prints a per-layer table of the linear probe R² and the residualised circular
("helix") R², plus a one-line verdict. Pure stdlib (no numpy/torch) — runs
anywhere, including the submit node.
"""
from __future__ import annotations

import json
import sys


def _fmt(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: helix-report.py <trace.json> [submodule=resid_post]", file=sys.stderr)
        return 2
    submodule = sys.argv[2] if len(sys.argv) > 2 else "resid_post"

    trace = json.loads(open(sys.argv[1], encoding="utf-8").read())
    manifold = trace.get("manifold")
    if not manifold:
        print("error: trace has no `manifold` (run `token-heatmap manifold` first).", file=sys.stderr)
        return 2
    scalar = manifold.get("scalar")
    if not scalar:
        print("error: no probe scalar (re-run manifold with `--probe <scalar>`).", file=sys.stderr)
        return 2

    values = scalar["values"]
    s_min, s_max = min(values), max(values)
    s_range = s_max - s_min
    # Robust span (2.5–97.5 pct) for the "interior period" test, so a single
    # run-on line (e.g. the model failing to wrap) can't inflate the range and
    # fool the verdict into calling a boundary period "interior".
    sv = sorted(values)
    pct = lambda p: sv[min(len(sv) - 1, int(p / 100 * len(sv)))]
    robust_range = (pct(97.5) - pct(2.5)) or s_range
    outlier = s_max > 1.5 * pct(97.5) + 1e-9
    print(f"scalar: {scalar['name']}  |  positions: {len(values)}  |  range: {s_min:.0f}–{s_max:.0f}", end="")
    print(f"  (robust span ~{robust_range:.0f})" if outlier else "")
    if outlier:
        print(
            f"  ⚠ run-on outlier: max {s_max:.0f} >> typical ~{pct(97.5):.0f}. "
            "Re-run `manifold --probe` with --scalar-max to exclude it."
        )
    print()

    rows = []
    for layer in manifold["layers"]:
        if layer["submodule"] != submodule:
            continue
        probe = layer.get("probe") or {}
        circ = probe.get("circular") or {}
        rows.append((layer["layer"], probe.get("r2_cv"), circ.get("best_period"), circ.get("r2_cv")))
    if not rows:
        print(f"no '{submodule}' layers carry a probe (did you pass --probe?).", file=sys.stderr)
        return 2

    print(f"{'layer':>5} | {'linear R²':>9} | {'period':>6} | {'circular R²':>11}")
    print("-" * 42)
    for layer, lin, period, circ in rows:
        print(f"{layer:>5} | {_fmt(lin):>9} | {_fmt(period, 0):>6} | {_fmt(circ):>11}")

    # Verdict. A helix needs a high residual circular R² at an *interior* period
    # (well inside the range — a period≈range fit is a single bend / ramp alias,
    # not a repeating coil).
    linear = [r[1] for r in rows if r[1] is not None]
    best_lin = max(linear) if linear else None
    # Circular structure at an *interior* period (well inside the range — a
    # period≈range fit is a single bend / ramp alias, not a repeating coil).
    interior = (
        [(r[0], r[3], r[2]) for r in rows
         if r[3] is not None and r[2] is not None and r[2] < 0.9 * robust_range]
        if robust_range > 0
        else []
    )
    strong = [t for t in interior if t[1] > 0.5]
    partial = [t for t in interior if 0.33 < t[1] <= 0.5]

    print()
    if best_lin is not None and best_lin > 0.5:
        print(f"• line scalar is LINEARLY encoded (max CV R² = {best_lin:.2f}).")
    else:
        print("• line scalar is NOT clearly encoded linearly.")
    if strong:
        layer, circ, period = max(strong, key=lambda x: x[1])
        print(f"• HELIX: residual circular R² = {circ:.2f} at interior period {period:.0f} (layer {layer}).")
        print("  ⚠ confound: is the scalar decorrelated from token content? A repeating")
        print("    pattern (e.g. '0123456789') fakes a helix via the token manifold.")
    elif partial:
        layer, circ, period = max(partial, key=lambda x: x[1])
        print(f"• PARTIAL / emerging helix: residual circular R² = {circ:.2f} at interior period {period:.0f} (layer {layer}).")
        print("  above the single-bend baseline but short of a clean helix — consistent")
        print("  with the structure sharpening with model scale.")
    else:
        print("• no clean helix: residual circular structure is weak or sits at the range")
        print("  boundary (a single bend / ramp alias, not a repeating coil).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
