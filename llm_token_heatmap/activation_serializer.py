"""Serialize raw ``ActivationFullStats`` into Tier 2 ``.npz`` sidecars.

This module is the activation-side parallel of
:mod:`llm_token_heatmap.attention_serializer`: it persists the bulky per-
``(layer, submodule)`` activation tensors that ``ActivationProbe`` retains
when ``ActivationProbeConfig.capture_full=True``, so the inline summary stats
on each step remain cheap to load while the full tensors can be re-hydrated
on demand for downstream analysis.

* :func:`write_sidecar` writes one ``numpy.savez_compressed`` archive per
  ``(trace, step)`` pair. Tensors are stored as float32 so the
  ``torch.allclose(atol=1e-5)`` round-trip contract holds. When called with
  ``None`` or empty stats (the default ``capture_full=False`` path), it is a
  no-op and returns ``None`` — that's the mechanism that satisfies
  "``capture_full=False`` → no sidecar written".
* :func:`read_sidecar` materializes the archive into a Python dict shaped
  exactly like ``docs/web/activation-sidecar.schema.json``, so callers can
  validate it with ``jsonschema`` if needed and consumers can JSON-serialize
  it without further conversion.

The CLI invokes ``write_sidecar`` per step when ``--capture-full-activations``
is set and embeds the resulting path as ``activation_sidecar_ref`` on each
``ActivationStep`` in the JSON trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_token_heatmap.activation_probe import ActivationFullStats

SIDECAR_SCHEMA_VERSION = "1.0.0"


def _ensure_npz_suffix(path: Path) -> Path:
    if path.suffix == ".npz":
        return path
    if path.suffix:
        return path.with_suffix(path.suffix + ".npz")
    return path.with_suffix(".npz")


def write_sidecar(
    stats: ActivationFullStats | None,
    path: str | Path,
    *,
    step: int,
) -> Path | None:
    """Write the Tier 2 activation sidecar archive for one ``(trace, step)`` pair.

    Args:
        stats: Raw tensors retained by ``ActivationProbe`` when
            ``capture_full=True``. Passing ``None`` (or stats with no
            captured tensors) makes this a no-op — the contract for the
            ``capture_full=False`` path.
        path: Destination path. A ``.npz`` suffix is appended when missing.
            Parent directories are created.
        step: Zero-indexed generation step the sidecar belongs to. Stored in
            the archive so a stray file can be matched back to its trace
            without filename parsing.

    Returns:
        The :class:`Path` actually written, or ``None`` when ``stats`` was
        empty/``None`` and no file was produced.
    """

    if stats is None or not stats.layer_tensors:
        return None

    out_path = _ensure_npz_suffix(Path(path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.array(SIDECAR_SCHEMA_VERSION),
        "step": np.array(int(step), dtype=np.int64),
        "num_layers": np.array(int(stats.num_layers), dtype=np.int64),
        "hidden_dim": np.array(int(stats.hidden_dim), dtype=np.int64),
        "captured_submodules": np.array(list(stats.captured_submodules)),
        "captured_layers": np.array(list(stats.captured_layers), dtype=np.int64),
    }

    for (layer_idx, submodule), tensor in stats.layer_tensors.items():
        arrays[f"layer_{int(layer_idx)}_{submodule}"] = (
            tensor.detach().to(torch.float32).cpu().numpy()
        )

    np.savez_compressed(out_path, **arrays)
    return out_path


def read_sidecar(path: str | Path) -> dict[str, Any]:
    """Inverse of :func:`write_sidecar`: load the archive into a JSON-shaped dict.

    The returned shape matches ``docs/web/activation-sidecar.schema.json``,
    so callers can validate it with ``jsonschema`` and serialize it to JSON
    without further conversion. Arrays are returned as nested Python lists.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        captured_layers = [int(i) for i in data["captured_layers"].tolist()]
        captured_submodules = [str(s) for s in data["captured_submodules"].tolist()]

        layers: list[dict[str, Any]] = []
        for layer_idx in captured_layers:
            submodule_tensors: dict[str, list[float]] = {}
            for submodule in captured_submodules:
                key = f"layer_{layer_idx}_{submodule}"
                if key in data.files:
                    submodule_tensors[submodule] = data[key].tolist()
            layers.append({"layer": int(layer_idx), "submodule_tensors": submodule_tensors})

        return {
            "schema_version": str(data["schema_version"]),
            "step": int(data["step"]),
            "num_layers": int(data["num_layers"]),
            "hidden_dim": int(data["hidden_dim"]),
            "captured_submodules": captured_submodules,
            "captured_layers": captured_layers,
            "layers": layers,
        }
