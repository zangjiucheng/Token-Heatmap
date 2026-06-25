# Epic 06 — SAE / transcoder features

**Status:** Future · **Effort:** XL · **Depends on:** external dictionaries

## Motivation

The heart of the paper: replace MLP neurons with **cross-layer transcoder (CLT)**
features — sparse, mostly-monosemantic directions trained to reconstruct MLP
outputs across layers — which makes attribution interpretable (a feature "says X"
rather than "neuron 4823"). Training CLTs is a research project per model and is
**out of scope to build**. But we can deliver most of the user value by
**loading pretrained dictionaries**.

## Scope

- Support loading external **SAEs / transcoders** for supported models from
  `sae_lens` / Neuronpedia (resolve by model id + hook point).
- A capture path that records **feature activations** (encode the captured
  residual/MLP tensors through the loaded dictionary) instead of / alongside raw
  neurons; feed them into TWERA, DLA, and the attribution graph in place of
  neurons.
- **Feature dashboards**: top-activating dataset examples per feature, plus its
  direct logit effect (decoder · `W_U`) and embedding effect — mirroring the
  paper's feature-visualization panel.
- Honest limits (paper's own): reconstruction "dark matter" (~20–50%
  unexplained), feature splitting/absorption, polysemantic leftovers — surface
  via [Epic 04](04-faithfulness-error-reporting.md).

## Notes

- Dictionaries are model-and-hookpoint specific; gate the feature lens on
  "a dictionary is available for this model". Start with one well-supported model
  (e.g. a GPT-2 / Gemma SAE set) as a proof of concept.

## Acceptance

- For one supported model, a "Features" lens shows feature activations per step
  with a dashboard (top examples + logit/embedding effects), and TWERA/DLA can
  rank features instead of neurons.
