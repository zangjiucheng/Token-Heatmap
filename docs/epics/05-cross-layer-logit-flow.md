# Epic 05 — Cross-layer logit flow

**Status:** Planned · **Effort:** S · **Depends on:** logit-lens capture (have it)

## Motivation

The paper observes that cross-layer features collapse repeated "amplification"
computations spread across layers (per-layer transcoder path length ~3.7 →
CLT ~2.3). Even without transcoders, a cheap and illuminating view is **where in
depth the prediction forms**: project the residual stream at each layer through
the (folded) final norm + unembedding and track the **selected token's logit /
rank as it accumulates across layers**. This is essentially a logit-lens focused
on the *realized* token, and we already capture per-layer logit-lens data.

## Scope

- From the captured per-layer logit-lens (or the residual tensors), compute the
  selected token's logit and rank at each layer for each step.
- A lens / overlay: a small per-step **"logit accumulation" curve** (layer on x,
  target-token logit or rank on y), so you can see the answer "snap in" at a
  particular depth — and pair it with the per-layer DLA bars from
  [Epic 01](01-direct-logit-attribution.md).

## Acceptance

- For a sample trace, the curve is monotone-ish and the layer where the target
  token reaches rank-1 matches the logit-lens table; renders for the selected step.
