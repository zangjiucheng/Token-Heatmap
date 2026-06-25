# Epic 02 — Interventions / ablation (causal validation)

**Status:** v1 + per-head (v2) landed on `main` (per-neuron remains) · **Effort:** M–L · **Depends on:** backend model held in memory

## Motivation

The paper stresses that attribution edges are *hypotheses* until validated by
**intervention**: "constrained patching" (overwrite a feature's activation across
a layer range and forward from there) and **multiplicative steering** (scale an
activation by a factor M and measure the downstream effect on other components
and on the logits). This is the single biggest differentiator vs. a plain
attention/heatmap viewer: it turns Token-Heatmap from *visualize* into
*experiment* — and it directly validates Epics 01/03.

## Scope

- Backend `/intervene` endpoint (FastAPI) + CLI: given a loaded model + prompt,
  apply a patch to a chosen **neuron / attention head / layer block** at a chosen
  position — `zero` (ablate), `scale ×M`, or `set` — via a forward hook, then
  recompute the forward pass **without re-running upstream** (constrained, to
  avoid second-order knock-on, matching the paper).
- Return the **delta in the next-token distribution**: KL divergence vs. the
  unpatched run, top-token flips, and the change in the target token's logit.
- Frontend: an "intervene" affordance on the Activations / Attribution / Attention
  lenses (e.g. right-click a neuron/head → "ablate" / "amplify ×2") with a
  before/after distribution diff.

## Notes / risks

- Interventions hold attention patterns fixed (paper's own caveat) — document it.
- Needs the model resident in the backend; the static `--serve` file server can't
  do this. Gate the UI on backend availability (we already track health).
- Per-arch hook points for heads (GQA grouping) need care — share helpers with
  the per-head work in [Epic 01](01-direct-logit-attribution.md) v2.

## Acceptance

- Ablating a head/neuron measurably changes the output distribution and the UI
  shows the KL + top-token flips; amplifying a known "say-X" direction increases
  X's probability. Covered by a backend test on a tiny model.
