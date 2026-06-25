# Interpretability epics

A roadmap for taking Token-Heatmap from a **correlational** viewer (heatmaps,
attention patterns, logit-lens) toward the **causal + faithful** circuit-analysis
direction set out in Anthropic's *Circuit Tracing: Revealing Computational Graphs
in Language Models* / attribution-graphs methods paper
([transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).

## The strategic read

The paper's full pipeline hinges on **cross-layer transcoders (CLT)** — trained
sparse dictionaries that replace MLP neurons with monosemantic *features*. That
piece is a research project per model and is out of scope as a built-in
(tracked as [Epic 06](06-sae-transcoder-features.md), via *loading* external
SAEs rather than training our own).

The key insight: **~70% of the value does not need transcoders.** Direct logit
attribution, per-head contributions, interventions, and faithfulness/error
reporting are all computable on the **raw model we already load** plus the
activations/attention we already capture. Those are the near-term epics.

## Epics

| # | Epic | Paper basis | Effort | Status |
|---|------|-------------|--------|--------|
| [01](01-direct-logit-attribution.md) | Direct Logit Attribution lens | Direct logit attribution, "fold the final norm", per-head OV | M | **v1 landed** |
| [02](02-interventions-ablation.md) | Interventions / ablation (causal validation) | Constrained patching, multiplicative steering | M–L | Planned |
| [03](03-attribution-graph.md) | Attribution graph lens | Attribution graphs, pruning, supernodes | L | Planned |
| [04](04-faithfulness-error-reporting.md) | Faithfulness & error reporting | Error nodes, completeness/replacement scores | S | Planned |
| [05](05-cross-layer-logit-flow.md) | Cross-layer logit flow | Logit-lens of the selected token; path length | S | Planned |
| [06](06-sae-transcoder-features.md) | SAE / transcoder features | Cross-layer transcoders, feature dashboards | XL | Future |
| [07](07-twera-global-coactivation.md) | Global TWERA (cross-prompt) | Virtual weights, co-activation reweighting | M | Future |

## Sequencing

Epic 01 (in flight) produces the per-component logit decomposition that 03
(graph) and 04 (faithfulness) build directly on top of. 02 (interventions) is
the independent, highest-differentiation track — it turns the viewer from
"visualize" into "experiment". 05 is a quick win from data we already have.
06/07 are the heavy, research-adjacent future work.

> These are living documents, not a contract. Each is sized to be convertible to
> a GitHub issue/epic; statuses are updated as work lands on `main`.
