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
