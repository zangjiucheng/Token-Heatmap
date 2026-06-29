"""Head role taxonomy — classify each attention head by *function*, not activity.

The guiding principle (validated on Qwen2.5-7B: ``corr(BOS-attention, logit
contribution) = -0.31``): a head's importance is its **contribution** to the
output, which is *anti-correlated* with how much it "fires". Attention sinks
attend ~100 % to the first token yet write ~0 to the logits; the heads that do
the work attend to content and have a large direct-logit-attribution (DLA).

This pass fuses the two signals already in a trace — per-head attention stats
(``steps[i].attention[L].per_head``) and per-head DLA
(``direct_logit_attribution``) — and labels each ``(layer, head)`` with a
functional role plus its mean logit contribution. It is pure-dict / no-torch,
like the manifold pass, so it runs post-hoc on any trace.

Roles
-----
* ``sink``      — attends mostly to BOS, contributes ~nothing (a no-op / bias head).
* ``induction`` — high induction score (attends to the token after the current
  token's last occurrence); the copy/induction signature.
* ``worker``    — large |DLA|: actually writes the answer. The load-bearing heads.
* ``local``     — attends mostly to the current/last token (positional / smoothing).
* ``minor``     — a small but non-negligible writer.
* ``other``     — none of the above stand out.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

SCHEMA_VERSION = "1.0"

# Thresholds (logit units for DLA; attention fractions for the rest). Tunable —
# contribution-first, so the writer test wins over the attention-pattern tests.
WORKER_DLA = 0.25  # |mean DLA| at/above which a head is a load-bearing writer
MINOR_DLA = 0.10  # small-but-real writer floor
SINK_BOS = 0.50  # mean attention on position 0 that marks a sink
SINK_DLA = 0.10  # a sink must also contribute below this
INDUCTION_MIN = 0.15  # mean induction score that flags an induction head
LOCAL_SELF = 0.50  # mean attention on the current/last token (local/positional)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _classify(bos: float, self_w: float, induction: float, dla_abs: float) -> str:
    # Contribution and the induction signature come first; attention-pattern
    # roles (sink / local) only apply to heads that are NOT strong writers.
    if induction >= INDUCTION_MIN:
        return "induction"
    if dla_abs >= WORKER_DLA:
        return "worker"
    if bos >= SINK_BOS and dla_abs < SINK_DLA:
        return "sink"
    if self_w >= LOCAL_SELF:
        return "local"
    if dla_abs >= MINOR_DLA:
        return "minor"
    return "other"


def compute_head_roles(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Classify every attention head from a trace's attention + DLA data.

    Returns a ``head_roles`` summary dict, or ``None`` if the trace lacks the
    per-head attention stats or the direct logit attribution needed.
    """
    steps = trace.get("steps") or []
    dla = trace.get("direct_logit_attribution")
    if not steps or not isinstance(dla, dict) or not dla.get("steps"):
        return None

    bos: dict[tuple[int, int], list[float]] = {}
    self_w: dict[tuple[int, int], list[float]] = {}
    induction: dict[tuple[int, int], list[float]] = {}
    have_per_head = False
    for step in steps:
        for entry in step.get("attention", []) or []:
            layer = entry.get("layer")
            per_head = entry.get("per_head")
            if not per_head:
                continue
            have_per_head = True
            if isinstance(per_head, dict):
                # Columnar form: parallel arrays keyed by metric.
                b = per_head.get("bos_weight") or []
                s = per_head.get("self_weight") or []
                ind = per_head.get("induction") or []
                for head in range(max(len(b), len(s), len(ind))):
                    bos.setdefault((layer, head), []).append(
                        float(b[head]) if head < len(b) else 0.0
                    )
                    self_w.setdefault((layer, head), []).append(
                        float(s[head]) if head < len(s) else 0.0
                    )
                    induction.setdefault((layer, head), []).append(
                        float(ind[head]) if head < len(ind) else 0.0
                    )
            else:
                # Legacy list-of-dicts form (pre-columnar traces).
                for head, hd in enumerate(per_head):
                    bos.setdefault((layer, head), []).append(float(hd.get("bos_weight", 0.0)))
                    self_w.setdefault((layer, head), []).append(float(hd.get("self_weight", 0.0)))
                    induction.setdefault((layer, head), []).append(float(hd.get("induction", 0.0)))
    if not have_per_head:
        return None

    dla_vals: dict[tuple[int, int], list[float]] = {}
    for ds in dla.get("steps", []):
        for layer_entry in ds.get("layers", []) or []:
            layer = layer_entry.get("layer")
            for hd in layer_entry.get("heads", []) or []:
                dla_vals.setdefault((layer, int(hd.get("head"))), []).append(
                    float(hd.get("attn", 0.0))
                )

    heads: list[dict[str, Any]] = []
    for layer, head in sorted(set(bos) | set(dla_vals)):
        key = (layer, head)
        b = _mean(bos.get(key, []))
        s = _mean(self_w.get(key, []))
        ind = _mean(induction.get(key, []))
        dv = dla_vals.get(key, [])
        dla_mean = _mean(dv)
        dla_absmean = _mean([abs(x) for x in dv])
        heads.append(
            {
                "layer": int(layer),
                "head": int(head),
                "role": _classify(b, s, ind, dla_absmean),
                "dla_mean": dla_mean,
                "dla_absmean": dla_absmean,
                "bos_weight": b,
                "self_weight": s,
                "induction": ind,
            }
        )

    counts = Counter(h["role"] for h in heads)
    workers = sorted(
        (h for h in heads if h["role"] in ("worker", "induction")),
        key=lambda h: -h["dla_absmean"],
    )
    sinks = sorted((h for h in heads if h["role"] == "sink"), key=lambda h: -h["bos_weight"])
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "attention+DLA fusion; contribution-first classification",
        "thresholds": {
            "worker_dla": WORKER_DLA,
            "minor_dla": MINOR_DLA,
            "sink_bos": SINK_BOS,
            "sink_dla": SINK_DLA,
            "induction_min": INDUCTION_MIN,
            "local_self": LOCAL_SELF,
        },
        "heads": heads,
        "summary": {
            "counts": dict(counts),
            "top_workers": [
                {"layer": h["layer"], "head": h["head"], "dla_absmean": h["dla_absmean"]}
                for h in workers[:10]
            ],
            "top_sinks": [
                {"layer": h["layer"], "head": h["head"], "bos_weight": h["bos_weight"]}
                for h in sinks[:10]
            ],
        },
    }
