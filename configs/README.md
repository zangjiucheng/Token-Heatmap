# Experiment configs

Each YAML here is a self-contained `token-heatmap trace` experiment. Run one with:

```bash
token-heatmap trace --config configs/<name>.yaml --serve --frontend   # local, opens the viewer
token-heatmap hpc run configs/<name>.yaml                             # on the HPC GPU, pulled back
```

Any key can be overridden on the CLI (`--model`, `--prompt`, `--max-new-tokens`, …).

| Config | Experiment | Model | What it shows |
| --- | --- | --- | --- |
| `example.yaml` | Quickstart | 0.5B | The core adaptive-top-k probability heatmap + Logit Lens. The canonical first run. |
| `recall-probe.yaml` | DLA causal validation | 0.5B | Per-head Direct Logit Attribution + ablation argument ("The capital of France is" → Paris). Pairs with `examples/dla_causal_validation.py`. |
| `induction.yaml` | Induction-head circuit | 0.5B | Repeated list → copy ("…apple, banana," → cherry); per-head DLA + Attribution Graph + ablation. |
| `ioi.yaml` | Name-mover circuit (IOI) | 0.5B | "…John gave a drink to" → Mary; the canonical per-head circuit. |
| `attention-demo.yaml` | Attention patterns | 0.5B | Few-shot table completion (capitals); readable per-head attention in the Attention lens. |
| `reasoning.yaml` | Multi-step reasoning | 14B (GPU) | Algebra word problem; Logit Lens evolution + DLA on the answer tokens. |
| `wrap-text.yaml` | Manifold / helix | 7B–32B (GPU) | Fixed-width line-wrapping — the "When Models Manipulate Manifolds" counting-helix test. See `docs/manifold-reproduction.md`. |

## Notes

- **Small (0.5B) configs run on a laptop CPU in seconds** and set both
  `capture_full_attention` and `capture_full_activations`, so a single run
  populates every lens — Heatmap, Logit Lens, Attention, Attribution, Graph
  (per-head), and interventions/ablation.
- **GPU configs** (`reasoning`, `wrap-text`) are meant for `token-heatmap hpc run`
  (or a local big GPU). `wrap-text` deliberately captures activations only.
- **Per-head circuits stay crisp on small models** — see the scaling caveat in
  [`docs/interpreting.md`](../docs/interpreting.md). That's why the circuit demos
  (`induction`, `ioi`, `recall-probe`) use Qwen2.5-0.5B rather than a larger model.
