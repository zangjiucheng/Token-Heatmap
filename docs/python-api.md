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
