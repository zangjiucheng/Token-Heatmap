"""Example: run Qwen with AttentionProbe + LogitLens and dump trace + plots.

Generates a short completion with both probes attached, then writes:

* ``outputs/qwen_attention_inspect/adaptive_token_trace.json`` -- full trace
  with inline attention aggregates and per-layer logit-lens projections.
* ``outputs/qwen_attention_inspect/attention_layer_head_grid.png`` --
  layer x head entropy heatmap from the first step.
* ``outputs/qwen_attention_inspect/logit_lens.png`` -- per-layer top-k table
  for the first step.
* ``outputs/qwen_attention_inspect/selected_rank_heatmap.png`` --
  selected-token rank by layer x step.

Defaults are tuned for a CPU run on ``Qwen/Qwen2.5-0.5B-Instruct`` finishing in
roughly a minute. Override ``MODEL_NAME`` / ``MAX_NEW_TOKENS`` if you have a
GPU and want a longer generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_token_heatmap import (
    ActivationProbe,
    ActivationProbeConfig,
    AdaptiveProbeConfig,
    AdaptiveTokenProbe,
    AttentionProbe,
    AttentionProbeConfig,
    LogitLens,
    LogitLensConfig,
    compute_attention_stats,
    generate_with_adaptive_probe,
    plot_attention_layer_head_grid,
    plot_logit_lens,
    plot_logit_lens_selected_rank,
)
from llm_token_heatmap.trace_payload import serialize_trace_to_json

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in one sentence why attention works."
MAX_NEW_TOKENS = 60


def main() -> int:
    output_dir = Path("outputs") / "qwen_attention_inspect"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading {MODEL_NAME} on {device}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    probe = AdaptiveTokenProbe(AdaptiveProbeConfig(min_k=8, max_k=32, mass_threshold=0.95))
    attention_probe = AttentionProbe(AttentionProbeConfig(layers="all", top_k_positions=8))
    logit_lens = LogitLens(LogitLensConfig(layers="all", top_k=5))
    activation_probe = ActivationProbe(
        ActivationProbeConfig(
            layers="all",
            submodules=["resid_post", "mlp_out"],
            top_k=8,
        )
    )

    attention_probe.attach(model)
    logit_lens.attach(model)
    activation_probe.attach(model)
    try:
        text, trace = generate_with_adaptive_probe(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT,
            probe=probe,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.95,
            use_chat_template=True,
            attention_probe=attention_probe,
            logit_lens=logit_lens,
            activation_probe=activation_probe,
        )
    finally:
        # Snapshot architecture metadata before detach() clears target_layers.
        # The frontend's Attention tab gates on the top-level attention_metadata
        # block; emitting it (rather than None) is what makes the captured
        # per-step `attention` arrays visible in the UI.
        captured_layers = list(attention_probe.target_layers)
        decoder_root = getattr(model, "model", None) or model
        decoder_layers = getattr(decoder_root, "layers", None)
        try:
            total_layers = len(decoder_layers) if decoder_layers is not None else None
        except TypeError:
            total_layers = None
        attention_metadata: dict | None = (
            {
                "num_layers": int(total_layers)
                if total_layers
                else max(captured_layers) + 1,
                "num_attention_heads": int(attention_probe._num_attention_heads),
                "num_key_value_heads": int(attention_probe._num_key_value_heads),
                "head_dim": int(attention_probe._head_dim),
                "captured_layers": [int(i) for i in captured_layers],
            }
            if captured_layers and attention_probe._num_attention_heads
            else None
        )
        activation_captured_layers = list(activation_probe.target_layers)
        activation_submodules = list(activation_probe.submodule_keys)
        activation_num_layers = int(activation_probe.num_layers)
        activation_hidden_dim = int(activation_probe.hidden_dim)
        activation_metadata: dict | None = (
            {
                "num_layers": activation_num_layers,
                "hidden_dim": activation_hidden_dim,
                "captured_layers": [int(i) for i in activation_captured_layers],
                "captured_submodules": activation_submodules,
                "tokenizer_fingerprint": MODEL_NAME,
            }
            if activation_captured_layers and activation_submodules
            else None
        )
        attention_probe.detach()
        logit_lens.detach()
        activation_probe.detach()

    print(text)

    # Strip private keys (raw AttentionStats) before passing to the serializer.
    trace_for_json = [
        {k: v for k, v in entry.items() if not k.startswith("_")} for entry in trace
    ]
    payload = serialize_trace_to_json(
        trace=trace_for_json,
        metadata={
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "generated_text": text,
            "device": device,
            "use_chat_template": True,
            "generation_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.7,
                "top_p": 0.95,
                "sample_top_k": 0,
            },
            "probe_config": {"min_k": 8, "max_k": 32, "mass_threshold": 0.95},
        },
        attention_metadata=attention_metadata,
        sidecar_refs={},
        tokenizer=tokenizer,
        prompt=PROMPT,
        activation_metadata=activation_metadata,
    )
    (output_dir / "adaptive_token_trace.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    first_with_attn = next((e for e in trace if "_attention_stats" in e), None)
    if first_with_attn is not None:
        derived = compute_attention_stats(first_with_attn["_attention_stats"], top_k=8)
        plot_attention_layer_head_grid(
            derived, value="entropy", save_path=output_dir / "attention_layer_head_grid.png"
        )

    first_with_lens = next((e for e in trace if "logit_lens" in e), None)
    if first_with_lens is not None:
        plot_logit_lens(first_with_lens, tokenizer, save_path=output_dir / "logit_lens.png")
        plot_logit_lens_selected_rank(
            trace, tokenizer, save_path=output_dir / "selected_rank_heatmap.png"
        )

    print(f"Wrote outputs to {output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
