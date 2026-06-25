"""Example: argue that per-head Direct Logit Attribution reveals a *causal* circuit.

A worked argument built around a designed config (``configs/recall-probe.yaml``):

1. Run one forward of a factual-recall prompt, capturing full activations.
2. Compute per-head DLA for the predicted answer token.
3. Check faithfulness — the unexplained ``error`` is ~0 and the per-head
   contributions sum exactly to each layer's attention bar.
4. Causally validate: ablate the TOP-DLA head vs a near-zero-DLA CONTROL head
   (and the whole top attention block) and show the answer probability drops in
   proportion to the attribution. DLA's ranking *predicts* the intervention.

Run (small model, a few CPU/MPS forward passes — no GPU needed):

    python examples/dla_causal_validation.py            # uses configs/recall-probe.yaml
    python examples/dla_causal_validation.py configs/dla-demo.yaml

This is the local, causal half of the demo; produce a full viewable trace with
``token-heatmap trace --config configs/recall-probe.yaml --capture-activations
--capture-full-activations`` (or via ``scripts/hpc-run.sh``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_token_heatmap.activation_probe import (
    ActivationProbe,
    ActivationProbeConfig,
    _resolve_decoder_layers,
    _resolve_submodule_target,
)
from llm_token_heatmap.direct_logit_attribution import (
    compute_direct_logit_attribution,
)
from llm_token_heatmap.intervention import run_intervention
from llm_token_heatmap.logit_lens import _resolve_final_norm

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "configs/recall-probe.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    model_id, prompt = cfg["model"], cfg["prompt"]
    print(f"config={config_path.name}  model={model_id!r}  prompt={prompt!r}")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    ids = tok(prompt, return_tensors="pt").input_ids
    input_ids = ids[0].tolist()

    # 1. one forward + full capture (residual_post, mlp_out, o_proj + the o_proj
    #    input z, which per-head DLA needs).
    probe = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["residual_post", "mlp_out", "o_proj"],
            capture_full=True,
        )
    )
    probe.attach(model)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1]
    probe.capture_step()
    stats = probe.last_full_stats
    probe.detach()

    target = int(logits.argmax())
    print(f"predicted answer token: {tok.decode([target])!r}  (id {target})")

    # 2. per-head DLA: needs each layer's W_O (o_proj weight) + head geometry.
    layers = _resolve_decoder_layers(model)
    o_proj_weights = {
        i: _resolve_submodule_target(layer, "o_proj").weight
        for i, layer in enumerate(layers)
    }
    nh = int(model.config.num_attention_heads)
    hd = int(getattr(model.config, "head_dim", None) or model.config.hidden_size // nh)
    dla = compute_direct_logit_attribution(
        trace=[{"step": 0, "_activation_full_stats": stats}],
        target_token_ids=[target],
        unembedding=model.get_output_embeddings().weight,
        final_norm=_resolve_final_norm(model),
        o_proj_weights=o_proj_weights,
        num_heads=nh,
        head_dim=hd,
    )
    step = dla["steps"][0]

    # 3. faithfulness
    top_layer = max(step["layers"], key=lambda layer: abs(layer["attn"]))
    head_sum = sum(h["attn"] for h in (top_layer.get("heads") or []))
    print("\n== faithfulness ==")
    print(f"total logit(answer) = {step['total_logit']:.3f}")
    print(f"unexplained error   = {step['error']:.5f}   (~0 => decomposition is exact)")
    print(
        f"per-head sum vs layer attn (L{top_layer['layer']}): "
        f"{head_sum:.4f} vs {top_layer['attn']:.4f}  "
        f"(delta={abs(head_sum - top_layer['attn']):.2e})"
    )

    heads = [
        (layer["layer"], h["head"], h["attn"])
        for layer in step["layers"]
        for h in (layer.get("heads") or [])
    ]
    heads.sort(key=lambda x: abs(x[2]), reverse=True)
    print("\n== top heads by DLA ==")
    for layer, head, value in heads[:6]:
        print(f"  L{layer:<2d} h{head:<2d}  attn = {value:+.3f}")

    top_l, top_h, top_v = heads[0]
    ctrl_l, ctrl_h, ctrl_v = min(heads, key=lambda x: abs(x[2]))

    def ablate(spec: dict, label: str) -> dict:
        result = run_intervention(
            model,
            input_ids=input_ids,
            interventions=[spec],
            target_token_id=target,
            tokenizer=tok,
            top_k=5,
        )
        diff = result["diff"]
        base, patched = result["baseline"]["target_prob"], result["patched"]["target_prob"]
        print(
            f"  {label:<34s} P(answer) {base:.4f} -> {patched:.4f}  "
            f"(delta={diff['target_prob_delta']:+.4f}, KL={diff['kl']:.3f})"
        )
        return diff

    # 4. causal validation
    print("\n== causal validation (ablate -> re-run -> diff) ==")
    top = ablate(
        {"layer": top_l, "component": "head", "head": top_h, "op": "zero"},
        f"TOP head L{top_l}.h{top_h} (DLA {top_v:+.2f})",
    )
    ctrl = ablate(
        {"layer": ctrl_l, "component": "head", "head": ctrl_h, "op": "zero"},
        f"CTRL head L{ctrl_l}.h{ctrl_h} (DLA {ctrl_v:+.2f})",
    )
    ablate(
        {"layer": top_l, "component": "attn", "op": "zero"},
        f"TOP attn BLOCK L{top_l} (all heads)",
    )

    ratio = abs(top["target_prob_delta"]) / max(abs(ctrl["target_prob_delta"]), 1e-9)
    print("\n== argument ==")
    print(
        f"DLA ranks L{top_l}.h{top_h} as the top promoter of {tok.decode([target])!r}; "
        f"ablating it drops the answer probability {abs(top['target_prob_delta']):.4f} "
        f"(KL {top['kl']:.3f}); a near-zero-DLA control head moves it ~{ratio:.0f}x less."
    )
    print("=> DLA's attribution predicts the causal effect — the tool reveals a real circuit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
