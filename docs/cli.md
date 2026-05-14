# Command-line interface

After `pip install -e .` the `token-heatmap` command is on your `PATH`.

## Basic trace

```bash
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain diffusion models." \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-p 0.95 \
  --min-k 8 --max-k 64 --mass-threshold 0.95 \
  --out outputs/
```

Output:

```
outputs/
├── generated.txt
├── adaptive_token_trace.csv
├── adaptive_heatmap.png
├── entropy.png
├── selected_probability.png
└── raw_vs_processed.png
```

Run `token-heatmap trace --help` for the full flag list.

## Inspecting attention and the logit lens

`AttentionProbe` and `LogitLens` are wired through `token-heatmap trace` as
opt-in flags. They are off by default because forcing eager attention is
significantly slower than the SDPA / FlashAttention kernel.

```bash
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain attention in one sentence." \
  --max-new-tokens 12 \
  --capture-attention --attention-layers all --attention-top-k 8 \
  --capture-full-attention \
  --capture-logit-lens --lens-layers 0,3,7,11 --lens-top-k 5 \
  --out outputs/inspect/
```

| Flag                       | Meaning                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--capture-attention`      | Attach an `AttentionProbe`. Forces eager attention (slow).                                                   |
| `--attention-layers`       | `all` (default) or comma-separated layer indices (e.g. `0,3,7,11`).                                          |
| `--attention-top-k`        | Top-k attended positions kept inline per head (default 8).                                                   |
| `--capture-full-attention` | Also write a Tier-2 `attention.<step>.npz` per step under `<out>/attention/`. Implies `--capture-attention`. |
| `--capture-logit-lens`     | Attach a `LogitLens`.                                                                                        |
| `--lens-layers`            | Same syntax as `--attention-layers`.                                                                         |
| `--lens-top-k`             | Top-k tokens retained per layer (default 8).                                                                 |

When attention or lens capture is on, the CLI also writes:

- `adaptive_token_trace.json` — schema-shaped trace including inline attention aggregates, per-layer logit-lens projections, and `attention_sidecar_ref` pointers. See [`schema.md`](schema.md) for the layout.
- `attention_layer_head_grid.png` — per-step layer × head entropy grid (first step).
- `logit_lens.png` — per-layer top-k table (first step).
- `selected_rank_heatmap.png` — selected-token rank by layer × step.

A runnable end-to-end script lives at
[`examples/qwen_attention_inspect.py`](https://github.com/zangjiucheng/Token-Heatmap/blob/main/examples/qwen_attention_inspect.py);
it runs on `Qwen/Qwen2.5-0.5B-Instruct` in under 90 s on CPU.

## Capturing activations

`ActivationProbe` is the third opt-in capture path. When the flags below
are set, the CLI embeds `activation_metadata` and per-step `activations`
blocks directly into `adaptive_token_trace.json` (no sibling file).

```bash
token-heatmap trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain attention in one sentence." \
  --max-new-tokens 12 \
  --capture-activations \
  --activation-layers all \
  --activation-submodules residual_post,mlp_out,o_proj \
  --activation-top-k 8 \
  --out outputs/activations/
```

| Flag                         | Meaning                                                                                                                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--capture-activations`      | Attach an `ActivationProbe`. Off by default.                                                                                                                                       |
| `--activation-layers`        | `all` (default) or comma-separated decoder layer indices.                                                                                                                          |
| `--activation-submodules`    | Comma-separated submodule keys (default `residual_post,mlp_out,o_proj`). Supported: `resid_pre`/`residual_pre`, `resid_post`/`residual_post`, `mlp_out`/`mlp.down_proj`, `o_proj`. |
| `--activation-top-k`         | Top-k highest-magnitude neurons retained per (layer, submodule) (default 8).                                                                                                       |
| `--capture-full-activations` | Reserved for the Tier-2 sidecar path; implies `--capture-activations`.                                                                                                             |

## Comparing two activation traces

```bash
token-heatmap diff outA/adaptive_token_trace.json outB/adaptive_token_trace.json \
  --out diff/ \
  --metric l2
```

The subcommand projects each input's activation subset, calls
`compare_activations` with `align="auto"`, and writes:

- `activation_diff.json` — schema-shaped diff payload (matches
  [`web/activation-diff.schema.json`](web/activation-diff.schema.json)).
- `activation_delta.png` — stacked layer × step heatmap, one subplot per
  captured submodule, coloured by the chosen metric.

The CLI refuses to diff (non-zero exit) when the two parent traces have
different `metadata.prompt` values or when zero steps align between them
— these are the two ways "mismatched generations" present after the
comparator runs.
