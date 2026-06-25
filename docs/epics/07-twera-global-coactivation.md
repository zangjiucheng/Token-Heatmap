# Epic 07 — Global TWERA (cross-prompt co-activation)

**Status:** Future · **Effort:** M · **Depends on:** multi-trace capture

## Motivation

Our current TWERA (`llm_token_heatmap/neuron_attribution.py`)
is a **single-trace, local** approximation — honest, but the "expectation" is
over one trace's own steps. The paper's TWERA is a **global virtual-weights**
statistic that removes interference by reweighting with empirical co-activation:

```
V_ij^TWERA = ( E[a_j a_i] / E[a_j] ) · V_ij
```

where the expectations are over a large dataset. The paper notes this "heavily
relies on co-activation statistics and strongly changes which connections are
important beyond simply removing large interference weights." Bringing this in
makes our neuron rankings reflect on-distribution behavior, not just one prompt.

## Scope

- Accumulate per-`(layer, submodule)` co-activation statistics `E[a_j a_i]`,
  `E[a_j]` across **many prompts** (a corpus run), not just one trace.
- Compute global virtual weights `V_ij` (encoder/decoder inner products) and the
  TWERA reweighting; expose a "global vs. this-trace" toggle in the neuron view.
- Add the interference caveats the paper flags (inhibition, low-activation
  features remain imperfect) via [Epic 04](04-faithfulness-error-reporting.md).

## Acceptance

- Running over a small prompt corpus produces a global TWERA ranking that differs
  from (and is more stable than) the single-trace ranking; both are selectable.
