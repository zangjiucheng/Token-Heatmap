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
