#!/usr/bin/env bash
# Full-pipeline Slurm job: trace -> manifold -> (causal) -> (tuned-lens) on a GPU
# node, for re-running the circuit suite through the finalized toolkit. Unlike
# the stock hpc-gen.slurm (trace + manifold only), this adds the new post-hoc
# analyses. Capture flags come from the --config YAML (no forcing here), so a
# geometry config (activations-only, no logit lens) is honoured as-is.
#
# Env (all overridable via --export):
#   BIN         token-heatmap binary (default: GPU venv)
#   CONFIG      trace config YAML (relative to repo)
#   OUT         output dir (relative to repo), e.g. outputs/multihop
#   RUN_CAUSAL  1 to add the causal-effect grid (needs logit-lens-y traces)
#   RUN_TUNED   1 to add the tuned lens
#   EXTRA / MANIFOLD_EXTRA  extra flags for trace / manifold
#
#SBATCH --job-name=th-full
#SBATCH --account=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/j7zang/Token-Heatmap
#SBATCH --output=outputs/hpc-full-%j.log

set -uo pipefail

export HF_HOME="${HF_HOME:-/work/j7zang/.cache/huggingface}"
BIN="${BIN:-/work/j7zang/th-gpu/bin/token-heatmap}"
CONFIG="${CONFIG:-configs/example.yaml}"
OUT="${OUT:-outputs/qwen-run}"
EXTRA="${EXTRA:-}"
MANIFOLD_EXTRA="${MANIFOLD_EXTRA:---components 6}"
RUN_CAUSAL="${RUN_CAUSAL:-0}"
RUN_TUNED="${RUN_TUNED:-0}"
# Traces gzip by default; readers take either name, shell tests do not.
TRACE="$OUT/adaptive_token_trace.json.gz"
[ -f "$TRACE" ] || TRACE="$OUT/adaptive_token_trace.json"

echo "[full] job=${SLURM_JOB_ID:-?} node=$(hostname) config=$CONFIG out=$OUT causal=$RUN_CAUSAL tuned=$RUN_TUNED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[full] [1/4] trace ..."
# Capture flags come from the config. Plotting can exit non-zero on exotic
# glyphs AFTER the JSON is written, so tolerate that and gate on the JSON.
"$BIN" trace --config "$CONFIG" --out "$OUT" $EXTRA || echo "[full] trace returned nonzero (plotting?) — checking JSON"
# Re-resolve: the trace step decides the encoding.
TRACE="$OUT/adaptive_token_trace.json.gz"; [ -f "$TRACE" ] || TRACE="$OUT/adaptive_token_trace.json"
if [ ! -f "$TRACE" ]; then echo "[full] FATAL: no trace at $OUT/adaptive_token_trace.json[.gz]"; exit 1; fi
echo "[full] trace JSON: $(wc -c < "$TRACE") bytes"

echo "[full] [2/4] manifold ..."
"$BIN" manifold --trace "$TRACE" $MANIFOLD_EXTRA || echo "[full] manifold skipped/failed"

if [ "$RUN_CAUSAL" = "1" ]; then
  echo "[full] [3/4] causal ..."
  "$BIN" causal --trace "$TRACE" || echo "[full] causal failed"
else
  echo "[full] [3/4] causal SKIPPED"
fi

if [ "$RUN_TUNED" = "1" ]; then
  echo "[full] [4/4] tuned-lens ..."
  "$BIN" tuned-lens --trace "$TRACE" || echo "[full] tuned-lens failed"
else
  echo "[full] [4/4] tuned-lens SKIPPED"
fi

echo "[full] DONE -> $OUT  ($(wc -c < "$TRACE") bytes)"
