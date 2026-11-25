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
| `nqueens-verify.yaml` | Constraint check (valid) | 7B (GPU) | 4-queens board that *is* valid. ~250 tokens of step-by-step checking, ending on an attributable "Answer: Yes". Positive half of a minimal pair. |
| `nqueens-conflict.yaml` | Constraint check (invalid) | 7B (GPU) | Same board, rows 3/4 swapped: columns still distinct, fails on **one** diagonal. Does the reasoning actually reach that pair? |
| `nqueens-solve.yaml` | Reasoning-model CoT | 4B thinking (GPU) | Qwen3-4B-Thinking solves 6-queens; entropy spikes at backtracks + manifold geometry of a search. Attention capture off by design. |
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
- **Reasoning models need `use_chat_template: true`.** A Qwen3-Thinking or
  DeepSeek-R1-Distill only enters its `<think>` mode when it sees the chat
  template's assistant header — `Qwen3-4B-Thinking-2507` has no `enable_thinking`
  switch at all, its template just appends `<|im_start|>assistant\n<think>\n`.
  Without the flag you get a raw completion and no reasoning. Applies to the
  `nqueens-*` configs; the older configs above predate the flag and run raw.
- **Long chains of thought: turn `capture_attention` off.** Measured on
  `outputs/reasoning` (14B, 150 steps), attention was 428 KB of the 542 KB
  per-step JSON — 79% — and the per-step cost grows with sequence position, so
  it is quadratic over a run. A 1024-token CoT with attention on will not open in
  the viewer. Capture attention on short prompts (`nqueens-verify`), not on CoT.
- **Minimal pairs beat single runs.** `nqueens-verify` / `nqueens-conflict`
  differ by two digit tokens and are identical in everything else, so
  `token-heatmap diff` cancels the shared computation and leaves the constraint
  being tested. Same trick as `recall-probe`, applied to a decision instead of a
  recall.
- **Per-head circuits stay crisp on small models** — see the scaling caveat in
  [`docs/interpreting.md`](../docs/interpreting.md). That's why the crispest circuit
  demos (`induction`, `ioi`, `recall-probe`) use Qwen2.5-0.5B. `successor`,
  `greater-than`, and `multihop` are 7B instead because the *behaviour itself* needs
  a capable model (a 0.5B breaks the two-hop / inequality / increment); per-head DLA
  is fuzzier there, but block-level attribution stays legible.
