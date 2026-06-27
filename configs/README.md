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
| `successor.yaml` | Successor heads (+1) | 7B (GPU) | "1 2 3 4 …" → the next number; the increment OV circuit (Gould et al.). Per-head DLA on a number token. |
| `greater-than.yaml` | Greater-than | 7B (GPU) | "…from 1732 to 17__" → a year > 32 (Hanna et al.); mass over the valid range in the Heatmap + DLA promoters/suppressors. |
| `multihop.yaml` | Two-hop factual recall | 7B (GPU) | "capital of the state containing Dallas" → Dallas → Texas → Austin; Logit Lens shows "Texas" peak mid-depth before "Austin". Attribution-graph case study. |
| `attention-demo.yaml` | Attention patterns | 0.5B | Few-shot table completion (capitals); readable per-head attention in the Attention lens. |
| `reasoning.yaml` | Multi-step reasoning | 14B (GPU) | Algebra word problem; Logit Lens evolution + DLA on the answer tokens. |
| `wrap-text.yaml` | Manifold / helix (linear) | 7B–32B (GPU) | Fixed-width line-wrapping — the "When Models Manipulate Manifolds" counting-helix test. See `docs/manifold-reproduction.md`. |
| `cyclic-days.yaml` | Manifold / ring (cyclic) | 7B (GPU) | Weekday cycle — does the day feature live on a *ring*? Manifold periodicity + a looping (not drifting) scatter. |

## Notes

- **Small (0.5B) configs run on a laptop CPU in seconds** and set both
  `capture_full_attention` and `capture_full_activations`, so a single run
  populates every lens — Heatmap, Logit Lens, Attention, Attribution, Graph
  (per-head), and interventions/ablation.
- **GPU configs** (`successor`, `greater-than`, `multihop`, `reasoning`,
  `wrap-text`, `cyclic-days`) are meant for `token-heatmap hpc run` (or a local big
  GPU). The geometry runs (`wrap-text`, `cyclic-days`) deliberately capture
  activations only — no logit lens.
- **Per-head circuits stay crisp on small models** — see the scaling caveat in
  [`docs/interpreting.md`](../docs/interpreting.md). That's why the crispest circuit
  demos (`induction`, `ioi`, `recall-probe`) use Qwen2.5-0.5B. `successor`,
  `greater-than`, and `multihop` are 7B instead because the *behaviour itself* needs
  a capable model (a 0.5B breaks the two-hop / inequality / increment); per-head DLA
  is fuzzier there, but block-level attribution stays legible.
