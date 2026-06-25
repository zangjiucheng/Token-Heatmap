# Epic 04 — Faithfulness & error reporting

**Status:** Planned (seeded by Epic 01) · **Effort:** S

## Motivation

The paper's most transferable *methodology* is intellectual honesty about what an
attribution **does not** explain: every graph carries **error nodes**, and they
report a **completeness score** (fraction of logit-weighted input edges coming
from real features vs. error) and a **replacement score** (fraction of end-to-end
paths that avoid error nodes). Their replacement model only explains ~50–80% of
the computation, and they say so prominently. Our attributions (TWERA, DLA)
should adopt the same discipline so users are never misled by a tidy-looking bar
chart that quietly omits most of the effect.

## Scope

- Add an **"unexplained / error"** quantity to every attribution surface:
  - DLA already reports per-step `error` ([Epic 01](01-direct-logit-attribution.md)) — surface it as a bar + a headline "explained X%".
  - TWERA: add an error/residual term (currently it ranks neurons with no
    indication of how much of the target logit the top-N capture).
- A small, reused **"completeness" badge** component (explained % + a tooltip
  describing the omitted paths: attention-direct, final-norm linearization,
  non-residual submodules).
- Document the honest caveats inline in each lens (we already do this in code
  comments / `neuron_attribution` docstring — surface it in the UI).

## Acceptance

- Each attribution lens shows an explained-vs-unexplained figure; the number is
  derived, not hardcoded, and matches the backend's reported `error`.
