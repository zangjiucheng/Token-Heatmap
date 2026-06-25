# Interpreting the output

At every generation step the model produces a `[batch, vocab]` logit tensor.
The probe converts those logits into probabilities and records, **for both
the raw and the sampling-processed distributions**:

| Metric | Meaning |
| --- | --- |
| `top_ids` / `top_probs` / `top_logprobs` | Top-k candidate tokens and their probabilities |
| `k_used` | Adaptive top-k: smallest k whose cumulative mass ≥ `mass_threshold`, clamped to `[min_k, max_k]` |
| `top_mass_used` | Cumulative probability mass covered by `k_used` |
| `entropy` | Shannon entropy of the full next-token distribution (nats) |
| `selected_id` / `selected_prob` / `selected_logprob` | The token actually sampled and its likelihood under the analyzed distribution |
| `selected_rank` | Rank of the selected token across the full vocabulary |

The *adaptive* top-k matters because a fixed top-20 wastes detail on confident
steps and clips it on uncertain ones:

```
Step 1: top-8 tokens cover 95 % of mass
Step 2: top-45 tokens needed for 95 %
Step 3: top-12 tokens cover 95 %
```

The probe runs twice per step — once on the raw temperature-scaled logits,
once on the post-sampling logits (after `top_p` / `top_k` filtering) — so you
can see how sampling reshapes the distribution.

## Reading the plots

| Signal | Meaning |
| --- | --- |
| Low `k_used` (≈ `min_k`) | Confident step — distribution is concentrated |
| High `k_used` (≈ `max_k`) | Uncertain / diffuse — many plausible continuations |
| Low `selected_prob` | Sampling drew a low-probability token (high temp / wide top_p / open-ended step) |
| High `selected_rank` | The chosen token wasn't among the top candidates — useful for debugging sampling settings |
| Big gap between raw and processed `k_used` | Sampling filters are aggressively pruning the natural distribution |

## Direct Logit Attribution (the Attribution lens)

With `--capture-full-activations`, each generated token's logit is decomposed
into **additive contributions** through the unembedding, with the final norm
folded as a fixed scale (standard "direct logit attribution"):

```
logit(target) ≈ embed + Σ_layers ( attn_L + mlp_L ) + error
```

where `attn_L` is layer `L`'s attention block (`o_proj` output) and `mlp_L` its
MLP block (`mlp_out`). The **Attribution lens** shows these as diverging bars —
**orange promotes** the token, **blue suppresses** it — sorted by impact.

- **`error` / unexplained bar** — for RMSNorm models (Qwen / Llama / Mistral / …)
  the decomposition is exact, so `error ≈ 0`; a large error means the fold-norm
  linearization or a norm variant isn't fully captured. *Always read it* — it is
  how much of the logit the bars do **not** explain.
- **Per-head** — expand an attention bar to split it into per-head
  contributions, `W_O[:, head] · z_head` folded through the norm. These sum
  exactly to the layer's `attn` bar, so you can see *which head* wrote the token
  (induction / name-mover heads, etc.).

This is the **direct (OV) path** only: it explains how information already at the
final position maps to the logit, not how attention *patterns* formed (QK
circuits). It is correlational — to confirm a contribution is causal, ablate it.

## Interventions / ablation

The Attribution lens turns each bar into a hypothesis you can test. Click
**ablate** on a component or head (or pick one in the panel and **Run**) and the
backend re-runs the forward pass with that block's last-position output zeroed
(or scaled), then reports how the next-token distribution moved:

| Readout | Meaning |
| --- | --- |
| **KL (nats)** | Divergence between the baseline and patched next-token distributions — how much the ablation moved the output overall |
| **P(target) before → after** | The realized token's probability change; a top contributor's ablation should drop it noticeably |
| **Top-token flips** | Where the ranked candidates reordered (e.g. `#1 " Paris" → " London"`) |

A faithful attribution predicts the intervention: ablating a high-`attn` head
should drop the target probability more than a random head. Ablation holds the
attention *patterns* fixed (it removes a block's write to the residual at the
analyzed position), and needs the **live backend** — it loads/uses the trace's
model server-side, so it's available when running via `./scripts/dev.sh` or a
backend you've port-forwarded, not for purely static file views.

### Worked example — validating one head

`configs/recall-probe.yaml` is designed to make this concrete: a factual prompt
("The capital of France is" → ` Paris`) at low temperature with full capture.
`examples/dla_causal_validation.py` runs the argument on the small model in a few
forward passes (no GPU). On `Qwen/Qwen2.5-0.5B-Instruct`:

- The decomposition is **exact** — `error ≈ 0`, and the per-head contributions
  sum to the layer's attention bar to floating point (Δ ≈ 1e-7).
- DLA isolates one dominant promoter of ` Paris`: **layer 21, head 6**
  (`attn` ≈ +3.58), far above the rest.
- **Ablating it** drops `P(" Paris")` 0.302 → 0.232 (Δ −0.070, KL 0.051); a
  near-zero-DLA **control** head (L1 h3) moves it only −0.002 — **~36× less**.
  Ablating the whole L21 attention block drops it to 0.097, so head 6 is the
  largest piece of a layer-21 fact-writing computation.

The attribution *predicted* the intervention — the difference between a
suggestive chart and a causal claim.

## Manifold metrics

`token-heatmap manifold` ([CLI](cli.md#manifold-analysis)) treats the captured
activations for one `(layer, submodule)` as a cloud of points — one per token
position — and measures its geometry. The motivation comes from
[“When Models Manipulate Manifolds”](https://transformer-circuits.pub/2025/linebreaks/index.html):
models often encode a scalar (there, characters-until-line-break) on a smooth,
low-dimensional, _curved_ manifold, so the interesting structure is geometric,
not per-neuron.

| Metric | Meaning | What to look for |
| --- | --- | --- |
| **Participation ratio** | `(Σλ)² / Σλ²` over the PCA eigenvalues — a smooth "effective number of dimensions". | A small value (e.g. 2–3) despite a wide hidden dim ⇒ the cloud really lives on a low-dimensional manifold. |
| **Intrinsic dim (TwoNN)** | Geometry-based dimension estimate that, unlike PCA, sees curvature: a 1-D curve coiled in 3-D reads as ≈1. | Lower than the participation ratio ⇒ the cloud is a curved manifold, not a flat subspace. Unreliable for very short / regularly-spaced traces — prefer the participation ratio there. |
| **Trajectory curvature** | Mean turning of the position-ordered path through PCA space. | ≈0 ⇒ a straight sweep; large ⇒ the representation bends sharply as generation proceeds. |
| **Periodicity (period · power)** | Dominant period of the leading projection component (FFT) and its normalized power. | High power at a clean period is the signature of a circular / **helical** coordinate — the line-break "counting" manifold. |
| **Variance spectrum (scree)** | Explained-variance fraction per principal component, with the cumulative curve. | Variance collapsing into the first few bars ⇒ low-dimensional structure. |

The **Manifold tab** in the web app shows the 2-D PCA projection (coloured by
step, with the trajectory drawn through it) alongside these metrics, so you can
_see_ the manifold and read its summary numbers together.

## Recommended models

For local CPU / small GPU:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2`

For better output quality (GPU recommended):

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct` (gated — set `HUGGINGFACE_HUB_TOKEN`)

Tip: pass `use_chat_template=True` (or `--use-chat-template` on the CLI) when
using instruct models, otherwise the model sees a malformed prompt and tends
to ramble.
