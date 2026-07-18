# Python API

Use the library directly when you want programmatic access to the per-step
distributions, e.g. for downstream analysis or a custom notebook.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_token_heatmap import (
    AdaptiveProbeConfig, AdaptiveTokenProbe,
    generate_with_adaptive_probe,
    trace_to_dataframe,
    plot_adaptive_heatmap, plot_entropy, plot_selected_probability,
    plot_raw_vs_processed_heatmap,
)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto"
)

probe = AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=8, max_k=64, mass_threshold=0.95))

text, trace = generate_with_adaptive_probe(
    model=model, tokenizer=tokenizer,
    prompt="Explain why diffusion models work.",
    probe=probe,
    max_new_tokens=80, temperature=0.8, top_p=0.95,
    use_chat_template=True,           # for instruct models
)

df = trace_to_dataframe(trace, tokenizer)
df.to_csv("outputs/trace.csv", index=False)

plot_adaptive_heatmap(df, value_col="logprob", save_path="outputs/heatmap.png")
plot_entropy(df, save_path="outputs/entropy.png")
plot_selected_probability(df, save_path="outputs/selected.png")
plot_raw_vs_processed_heatmap(df, value_col="logprob", save_path="outputs/raw_vs_processed.png")
```

A runnable end-to-end example that also captures attention and the logit
lens lives at [`examples/qwen_attention_inspect.py`](https://github.com/zangjiucheng/Token-Heatmap/blob/main/examples/qwen_attention_inspect.py).

## API reference

### `AdaptiveProbeConfig`

| Field | Default | Description |
| --- | --- | --- |
| `min_k` | 8 | Lower clamp on the adaptive k |
| `max_k` | 64 | Upper clamp on the adaptive k |
| `mass_threshold` | 0.95 | Target cumulative probability mass |
| `eps` | 1e-12 | Numerical stability constant |

### `AdaptiveTokenProbe(config).forward(logits, selected_ids=None, temperature=1.0)`

Returns a dict of tensors keyed by `top_ids`, `top_probs`, `top_logprobs`,
`valid_mask`, `k_used`, `entropy`, `top_mass_used`. When `selected_ids` is
supplied it also returns `selected_ids`, `selected_prob`, `selected_logprob`,
`selected_rank`.

### `generate_with_adaptive_probe(...)`

```python
text, trace = generate_with_adaptive_probe(
    model, tokenizer, prompt, probe,
    max_new_tokens=64,
    temperature=0.8, top_p=0.95, sample_top_k=0,
    use_chat_template=False,         # wraps with tokenizer.apply_chat_template
    system_prompt=None,              # only used when use_chat_template=True
)
```

Each entry in `trace` is `{"step": int, "raw": stats_dict, "processed": stats_dict}`
with CPU tensors.

### `sample_next_token(logits, temperature=0.8, top_p=0.95, top_k=0)`

Returns `(next_token, processed_logits)`. `apply_sampling_filters(...)` returns
the masked logits without sampling.

### `trace_to_dataframe(trace, tokenizer, batch_index=0)`

Flattens the trace into a long-format DataFrame with one row per
`(step, source, rank)` where `source ∈ {"raw", "processed"}`.

### Plotting

- `plot_adaptive_heatmap(df, value_col, source="raw", save_path=...)`
- `plot_selected_probability(df, source="raw", save_path=...)`
- `plot_entropy(df, source="raw", save_path=...)`
- `plot_raw_vs_processed_heatmap(df, value_col, save_path=...)` — side by side with shared color scale
- `plot_raw_vs_processed_selected_prob(df, save_path=...)`
- `plot_logit_lens(trace_step, tokenizer, save_path=...)` — per-layer top-k table for one step
- `plot_logit_lens_selected_rank(trace, tokenizer, save_path=...)` — selected-token rank by layer × step
- `plot_logit_lens_heatmap(trace, tokenizer, save_path=...)` — each layer's top-1 predicted token by
  layer × generation step, treating every layer as if it were the final one; cell color is the
  token's probability, so the plot shows the prediction forming and sharpening with depth.
  Cells outlined in red mark top-1 *flips* — the depths where the predicted token changed
  identity (disable with `mark_flips=False`)
- `plot_prompt_logit_lens_heatmap(prompt_logit_lens, save_path=...)` — same heatmap over every
  *prompt* position, consuming the payload from `compute_prompt_logit_lens` (no tokenizer needed)
- `plot_logit_lens_kl(trace, save_path=...)` — two KL-vs-depth panels from the per-layer
  `kl_to_final` / `kl_from_prev` metrics: where each step's prediction converges to the final
  answer, and which block moved the prediction most
- `plot_activation_distributions(trace_step, save_path=...)` — ridgeline of per-layer activation
  value distributions (one silhouette per layer, one panel per submodule), built from the
  histograms the `ActivationProbe` records inline

### Serializing to the on-disk JSON

`generate_with_adaptive_probe` returns CPU tensors. To produce a
schema-conformant `adaptive_token_trace.json` (matching
[`docs/web/trace.schema.json`](web/trace.schema.json)) use:

```python
from llm_token_heatmap.trace_payload import serialize_trace_to_json

payload = serialize_trace_to_json(
    trace=trace,
    metadata={...},
    attention_metadata=None,        # or a dict when an AttentionProbe is attached
    sidecar_refs={},
    tokenizer=tokenizer,
    prompt=prompt,
)
```

The CLI does this for you; see [`examples/qwen_attention_inspect.py`](https://github.com/zangjiucheng/Token-Heatmap/blob/main/examples/qwen_attention_inspect.py)
for a hand-rolled version that also captures attention and the logit lens.
