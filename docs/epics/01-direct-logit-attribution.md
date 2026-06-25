# Epic 01 — Direct Logit Attribution lens

**Status:** v1 + per-head (v2) landed on `main` (runner parity remains) · **Effort:** M · **Depends on:** `--capture-full-activations`

## Motivation

The paper decomposes a token's logit into additive contributions from each
component, projected through the unembedding ("direct logit attribution"), and
emphasizes *folding the final norm* as a fixed scale so the decomposition is
linear and exact. We already capture, per step, the per-layer **attention-block
output** (`o_proj`) and **MLP-block output** (`mlp_out`) plus the final residual
(`residual_post`) — the exact additive residual deltas DLA needs. This turns our
scattered attention/activations/logit-lens views into one **"why this token"**
answer, and generalizes the existing single-trace TWERA (residual-direct neuron
attribution) to a complete layer/block decomposition.

## Scope

**v1 (this epic, landing now):**
- New `llm_token_heatmap/direct_logit_attribution.py`: per step, decompose the
  realized next-token logit into `embed` + per-layer (`attn`, `mlp`)
  contributions, folding the final norm (`_resolve_final_norm`) as a fixed scale
  (RMSNorm exact; Gemma `(1+weight)` handled; LayerNorm mean-subtraction handled).
- Report `total_logit`, `explained`, and **`error`** (faithfulness residual) per
  step — honesty first (ties to [Epic 04](04-faithfulness-error-reporting.md)).
- Reuse the TWERA capture path: gate on `--capture-full-activations`, read the
  same `_activation_full_stats.layer_tensors`. No new hooks.
- Inline in the trace payload (`direct_logit_attribution`), schema + codegen.
- New **Attribution** lens (Internals group): per-step diverging contribution
  bars by layer (attn vs mlp) + embed + unexplained error.

**v2 (follow-ups in this epic):**
- **Per-attention-head** contributions (OV·W_U), reconstructed from the attention
  sidecar (`pattern @ V` per head → split `W_O`). The trickiest part across
  eager/sdpa attention impls — deferred from v1.
- `runner.py` parity (currently v1 wires the `trace` CLI path).
- Overview mode: mean |contribution| per layer across all steps.

## Files

- `llm_token_heatmap/direct_logit_attribution.py` (new), `cli.py` (wire after the
  neuron-attribution block), `trace_payload.py` (`serialize_trace_to_json` param),
  `docs/web/trace.schema.json` (+ `DirectLogitAttribution` `$defs`).
- `web/frontend`: `src/features/dla/*`, `lenses.ts`, `useViewState.ts`,
  `TraceViewerPage.tsx`, `icons.tsx`, regenerated `types/trace.ts` + bundled schema.

## Acceptance

- For an RMSNorm model, `Σ(embed + per-layer attn + mlp) ≈ total_logit` with
  `|error|` small; verified by a unit test (synthetic + LayerNorm bias case).
- Lens shows the selected step's decomposition with the target token + total
  logit; locked with a clear hint when full activations weren't captured.
- All suites + lint + build green.
