"""Two-trace activation diff helper.

`compare_activations` consumes two activation traces conforming to
``docs/web/activation.schema.json`` and returns a payload conforming to
``docs/web/activation-diff.schema.json``. It is the math heart of the
activation toolbox: the CLI ``diff`` subcommand and the diff-mode UI both
consume its output shape.

The Tier-1 activation trace records only summary statistics per
``(step, layer, submodule)`` -- ``l2_norm``, ``mean_abs``, ``sparsity``, plus a
sparse ``top_neurons`` list of the highest-magnitude positions. Full activation
tensors live in optional sidecars (Tier 2). The comparator therefore
reconstructs a sparse vector from each side's ``top_neurons`` (treating
unmentioned indices as 0) and computes the L2 of the difference and the cosine
similarity over those reconstructions. With ``top_k == hidden_dim`` the
reconstruction is exact, which is how the synthetic closed-form tests pin the
numerics.

Alignment modes:

- ``token_id`` zips both traces on step index, flagging any step where the
  selected ``token_id`` diverges and any trailing steps that are unique to one
  side.
- ``position`` zips on ``decoded_text_offset`` (carried on every step),
  enabling cross-tokenizer comparison.
- ``auto`` picks ``token_id`` when both traces carry the same
  ``tokenizer_fingerprint``, else ``position``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

DIFF_SCHEMA_VERSION = "1.0.0"

AlignMode = Literal["token_id", "position", "auto"]
Metric = Literal["l2", "cosine"]


def _resolve_align_mode(align: AlignMode, fp_a: str, fp_b: str) -> Literal["token_id", "position"]:
    """Collapse ``auto`` into a concrete alignment strategy."""

    if align == "auto":
        return "token_id" if fp_a == fp_b else "position"
    if align in ("token_id", "position"):
        return align
    raise ValueError(
        f"align must be one of 'token_id', 'position', 'auto'; got {align!r}."
    )


def _group_entries(activations: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    """Index a step's ``activations`` list by ``(layer, submodule)``."""

    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in activations:
        key = (int(entry["layer"]), str(entry["submodule"]))
        indexed[key] = entry
    return indexed


def _sparse_vector(entry: dict[str, Any]) -> dict[int, float]:
    """Return ``{neuron_index: value}`` from an entry's ``top_neurons``."""

    return {int(n["index"]): float(n["value"]) for n in entry.get("top_neurons", [])}


def _compare_layer_entry(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Build one ``LayerDelta`` record for a shared ``(layer, submodule)``."""

    vec_a = _sparse_vector(entry_a)
    vec_b = _sparse_vector(entry_b)

    indices = set(vec_a) | set(vec_b)

    # Per-index signed delta over the union; absent on either side means 0.
    deltas: list[tuple[int, float]] = []
    sum_sq = 0.0
    dot = 0.0
    for idx in indices:
        a_val = vec_a.get(idx, 0.0)
        b_val = vec_b.get(idx, 0.0)
        diff = a_val - b_val
        deltas.append((idx, diff))
        sum_sq += diff * diff
        dot += a_val * b_val

    l2 = math.sqrt(sum_sq)

    norm_a = float(entry_a["l2_norm"])
    norm_b = float(entry_b["l2_norm"])
    if norm_a == 0.0 and norm_b == 0.0:
        # Two zero vectors point in the same (undefined) direction; the schema
        # bounds cosine in [-1, 1] and downstream consumers treat 1.0 as
        # identical, so reporting 1.0 keeps the self-diff invariant.
        cosine = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (norm_a * norm_b)
        # Guard against float drift past the schema's [-1, 1] bound.
        if cosine > 1.0:
            cosine = 1.0
        elif cosine < -1.0:
            cosine = -1.0

    deltas.sort(key=lambda item: abs(item[1]), reverse=True)
    top_changed = [
        {"index": int(idx), "delta": float(diff)} for idx, diff in deltas[: max(0, top_k)]
    ]

    return {
        "layer": int(entry_a["layer"]),
        "submodule": str(entry_a["submodule"]),
        "l2": float(l2),
        "cosine": float(cosine),
        "top_changed_neurons": top_changed,
    }


def _aligned_step_pair(
    step_a: dict[str, Any],
    step_b: dict[str, Any],
    aligned_index: int,
    top_k_changed_neurons: int,
) -> dict[str, Any]:
    """Build one ``DiffStep`` from two aligned source steps."""

    by_key_a = _group_entries(step_a["activations"])
    by_key_b = _group_entries(step_b["activations"])

    # The diff schema requires every emitted (layer, submodule) to be present
    # on both sides, so we intersect the two key sets and iterate in a stable
    # order that follows trace A's iteration order.
    shared_keys = [k for k in by_key_a if k in by_key_b]

    deltas = [
        _compare_layer_entry(by_key_a[k], by_key_b[k], top_k_changed_neurons)
        for k in shared_keys
    ]

    return {
        "step": int(aligned_index),
        "token_id_a": int(step_a["token_id"]),
        "token_id_b": int(step_b["token_id"]),
        "decoded_text_offset_a": int(step_a["decoded_text_offset"]),
        "decoded_text_offset_b": int(step_b["decoded_text_offset"]),
        "delta": deltas,
    }


def _align_by_token_id(
    steps_a: list[dict[str, Any]], steps_b: list[dict[str, Any]]
) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    """Pair steps on shared array index when ``token_id`` matches.

    Returns ``(aligned_pairs, mismatches)`` where ``aligned_pairs`` is a list of
    ``(aligned_index, step_a, step_b)`` and ``mismatches`` is the schema-shaped
    mismatch list (token_id divergence + trailing unmatched indices).
    """

    aligned: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    mismatches: list[dict[str, Any]] = []
    overlap = min(len(steps_a), len(steps_b))
    aligned_idx = 0
    for i in range(overlap):
        sa = steps_a[i]
        sb = steps_b[i]
        if int(sa["token_id"]) != int(sb["token_id"]):
            mismatches.append(
                {"step_a": i, "step_b": i, "reason": "token_id_divergence"}
            )
            continue
        aligned.append((aligned_idx, sa, sb))
        aligned_idx += 1
    for i in range(overlap, len(steps_a)):
        mismatches.append({"step_a": i, "step_b": None, "reason": "trailing_steps_in_a"})
    for i in range(overlap, len(steps_b)):
        mismatches.append({"step_a": None, "step_b": i, "reason": "trailing_steps_in_b"})
    return aligned, mismatches


def _align_by_position(
    steps_a: list[dict[str, Any]], steps_b: list[dict[str, Any]]
) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    """Pair steps by equal ``decoded_text_offset``.

    Steps that have no offset twin on the other side are reported as
    ``offset_gap`` mismatches. The aligned-index sequence is dense (0..N-1)
    independent of the source step indices.
    """

    by_offset_b: dict[int, list[int]] = {}
    for j, sb in enumerate(steps_b):
        by_offset_b.setdefault(int(sb["decoded_text_offset"]), []).append(j)

    aligned: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    mismatches: list[dict[str, Any]] = []
    used_b: set[int] = set()
    aligned_idx = 0

    for i, sa in enumerate(steps_a):
        offset_a = int(sa["decoded_text_offset"])
        candidates = by_offset_b.get(offset_a, [])
        match: int | None = next((j for j in candidates if j not in used_b), None)
        if match is None:
            mismatches.append({"step_a": i, "step_b": None, "reason": "offset_gap"})
            continue
        used_b.add(match)
        aligned.append((aligned_idx, sa, steps_b[match]))
        aligned_idx += 1

    for j in range(len(steps_b)):
        if j not in used_b:
            mismatches.append({"step_a": None, "step_b": j, "reason": "offset_gap"})

    return aligned, mismatches


def compare_activations(
    trace_a: dict[str, Any],
    trace_b: dict[str, Any],
    *,
    metric: Metric = "l2",
    align: AlignMode = "auto",
    top_k_changed_neurons: int | None = None,
) -> dict[str, Any]:
    """Compare two activation traces and return a schema-shaped diff payload.

    Args:
        trace_a: First activation trace (matches ``activation.schema.json``).
        trace_b: Second activation trace.
        metric: Primary diff metric (``"l2"`` or ``"cosine"``). Both fields
            are always populated on each layer-delta; this argument is
            recorded for downstream consumers (CLI ``--metric`` flag, UI
            colormap) and does not gate the output shape.
        align: Alignment strategy. ``"token_id"`` zips by step index and
            flags token-id divergences; ``"position"`` zips by
            ``decoded_text_offset``; ``"auto"`` picks ``token_id`` when both
            tokenizer fingerprints match, else ``"position"``.
        top_k_changed_neurons: Number of neurons retained in each
            ``top_changed_neurons`` list, sorted by descending ``|delta|``.
            Defaults to the maximum ``top_neurons`` length observed across
            the two traces' entries (i.e. all neurons present in either
            side's sparse representation).

    Returns:
        A dict conforming to ``docs/web/activation-diff.schema.json``.

    Raises:
        ValueError: If ``align`` is not one of the supported modes.
    """

    if metric not in ("l2", "cosine"):
        raise ValueError(f"metric must be 'l2' or 'cosine'; got {metric!r}.")

    metadata_a = trace_a["activation_metadata"]
    metadata_b = trace_b["activation_metadata"]
    fp_a = str(metadata_a["tokenizer_fingerprint"])
    fp_b = str(metadata_b["tokenizer_fingerprint"])

    resolved_mode = _resolve_align_mode(align, fp_a, fp_b)

    steps_a = list(trace_a.get("steps", []))
    steps_b = list(trace_b.get("steps", []))

    if resolved_mode == "token_id":
        aligned, mismatches = _align_by_token_id(steps_a, steps_b)
    else:
        aligned, mismatches = _align_by_position(steps_a, steps_b)

    if top_k_changed_neurons is None:
        observed = 0
        for step in (*steps_a, *steps_b):
            for entry in step.get("activations", []):
                observed = max(observed, len(entry.get("top_neurons", [])))
        top_k_changed_neurons = observed

    diff_steps = [
        _aligned_step_pair(sa, sb, aligned_idx, top_k_changed_neurons)
        for aligned_idx, sa, sb in aligned
    ]

    return {
        "schema_version": DIFF_SCHEMA_VERSION,
        "alignment": {
            "mode": resolved_mode,
            "tokenizer_a_fingerprint": fp_a,
            "tokenizer_b_fingerprint": fp_b,
            "mismatches": mismatches,
        },
        "steps": diff_steps,
    }
