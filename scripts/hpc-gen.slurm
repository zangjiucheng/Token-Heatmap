#!/usr/bin/env bash
# Slurm batch job: generate a token-heatmap trace for a larger model on a GPU
# node, then add manifold analysis. This template is normally submitted *for*
# you by `token-heatmap hpc run` (from the laptop); results are then viewed with
# `token-heatmap hpc serve` (no GPU needed to serve).
#
# You can also submit it by hand from the HPC repo checkout:
#   cd /work/j7zang/Token-Heatmap
#   sbatch scripts/hpc-gen.slurm
#
# Override the model / prompt / output dir without editing the file:
#   MODEL=Qwen/Qwen2.5-14B-Instruct OUT=outputs/qwen14b sbatch --gres=gpu:l40s:1 \
#     --export=ALL,MODEL,OUT scripts/hpc-gen.slurm
#
# fp16 single-GPU (the loader picks fp16 on CUDA automatically). 7B fits an
# rtx6000 (24G); 14B needs an l40s (48G). No quantization.
#
#SBATCH --job-name=th-gen
#SBATCH --partition=gpu-gen
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
# qos=normal caps mem at 30000M and cpu at 12 per user; 28G/8cpu stays under it.
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/j7zang/Token-Heatmap
#SBATCH --output=outputs/hpc-gen-%j.log

set -euo pipefail

# Reuse the 150G shared HF cache on /work (HOME is already /work/j7zang, so this
# is the default location -- set explicitly so it's robust to job env changes).
export HF_HOME="${HF_HOME:-/work/j7zang/.cache/huggingface}"

BIN="${BIN:-/work/j7zang/.local/bin/token-heatmap}"
# MODEL is OPTIONAL: empty -> use the model declared in the --config YAML.
# Set it only to override the config (e.g. MODEL=Qwen/Qwen2.5-14B-Instruct).
MODEL="${MODEL:-}"
MODEL_FLAG=""
[ -n "$MODEL" ] && MODEL_FLAG="--model $MODEL"
OUT="${OUT:-outputs/qwen-run}"
CONFIG="${CONFIG:-configs/example.yaml}"      # supplies prompt + sampling + logit-lens
# extra trace flags (e.g. override prompt/tokens): EXTRA="--max-new-tokens 96"
EXTRA="${EXTRA:-}"
# CAPTURE=full (default) captures attention too; CAPTURE=activations skips the
# (slow, eager) attention pass — use it for manifold/probe runs that only need
# activations, especially longer generations.
CAPTURE="${CAPTURE:-full}"
ATTN_FLAGS="--capture-attention --attention-layers all --capture-full-attention"
[ "$CAPTURE" = "activations" ] && ATTN_FLAGS=""

echo "[hpc-gen] job=$SLURM_JOB_ID node=$(hostname) model=${MODEL:-<from config>} out=$OUT capture=$CAPTURE"
echo "[hpc-gen] GPU:"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

echo "[hpc-gen] [1/2] trace..."
# --config supplies model/prompt/sampling/logit-lens; --out always overrides,
# and --model overrides only when MODEL is set (otherwise the config's wins).
"$BIN" trace --config "$CONFIG" $MODEL_FLAG --out "$OUT" \
  $ATTN_FLAGS \
  --capture-activations --capture-full-activations \
  $EXTRA

echo "[hpc-gen] [2/2] manifold analysis..."
# MANIFOLD_EXTRA e.g. "--components 6 --probe line_position"
"$BIN" manifold --trace "$OUT/adaptive_token_trace.json" ${MANIFOLD_EXTRA:-}

echo "[hpc-gen] DONE -> $OUT"
echo "[hpc-gen] view it with:  token-heatmap hpc serve $OUT"
