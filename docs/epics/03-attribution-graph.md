# Epic 03 — Attribution graph lens

**Status:** Planned · **Effort:** L · **Depends on:** [Epic 01](01-direct-logit-attribution.md)

## Motivation

The namesake method: a per-prompt **attribution graph** whose nodes are
output tokens, intermediate components, input embeddings, and **error nodes**,
with edges weighted by direct attribution computed under the local-linear
replacement. The paper then **prunes** to the influential subgraph (~10× fewer
nodes for ~80% of behavior) and groups nodes into **supernodes** by shared
facets. We can build a neuron/head-level version of this directly on the DLA +
TWERA data we already produce (features come later via [Epic 06](06-sae-transcoder-features.md)).

## Scope

- Build a per-step graph: `target token ← {top attention heads, top neurons} ←
  embedding`, edge weights from DLA (per-block/per-head) and TWERA (per-neuron),
  with an explicit **error node** for the unexplained residual.
- **Pruning**: keep only nodes/edges above a logit-influence threshold (port the
  paper's idea: rank by influence on the logit node).
- **Supernodes**: cluster neurons by similar input/output edges (start with a
  simple correlation/cosine grouping; allow manual expand/collapse).
- Frontend: a compact node-link lens (reuse our SVG/graph styling from the Build
  node editor and manifold scatter); hover a node → its top contributions; click
  → (later) trigger an [intervention](02-interventions-ablation.md) to validate.

## Acceptance

- For a sample trace, the lens renders a pruned graph for the selected step whose
  top paths reproduce the DLA ranking; an error node shows how much is unexplained.
- Node count stays legible after pruning; supernode grouping is toggleable.
