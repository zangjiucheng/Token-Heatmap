#!/usr/bin/env bash
# Slurm batch job: the N-Queens reasoning suite on a GPU node.
#
# Two suites, selected with SUITE=:
#
#   SUITE=pair   (default) — the minimal-pair verification experiment.
#       trace configs/nqueens-verify.yaml    (valid board   -> expect "Yes")
#       trace configs/nqueens-conflict.yaml  (one / diagonal conflict -> expect "No")
#       diff  the two                        (isolates the diagonal computation)
#       causal + tuned-lens on the valid run
#     Short generations (8 tokens), full capture. ~20-40 min on an l40s.
#
#   SUITE=solve — the chain-of-thought run.
#       trace configs/nqueens-solve.yaml     (Qwen3-4B-Thinking, 2048 tokens)
#       manifold                             (does the search trajectory loop?)
#     Attention capture is off in that config on purpose — see its header for the
#     measured per-step byte breakdown. ~1-2 h.
#
# Submit from the HPC checkout (login shell — Slurm lives in /opt/uw/slurm):
#   cd /work/j7zang/Token-Heatmap
#   sbatch scripts/hpc-nqueens.slurm                                  # pair
#   export SUITE=solve && sbatch --export=ALL scripts/hpc-nqueens.slurm
#
# NOTE on --export: use the two-step `export VAR=... && sbatch --export=ALL`
# form above. The comma-list form `--export=ALL,SUITE` is accepted by sbatch on
# this cluster but the job is then killed by the scheduler ~1s in ("CANCELLED by
# 0", no log file written) — verified on jobs 6024 and 6025, while the same
# script submitted with --export=ALL ran fine as 6026.
#
# Or from the laptop:
#   ssh j7zang-gpu 'bash -lc "cd /work/j7zang/Token-Heatmap && sbatch scripts/hpc-nqueens.slurm"'
#
# Pull results back (no GPU needed to view):
#   rsync -avz j7zang-gpu:/work/j7zang/Token-Heatmap/outputs/nqueens-\* outputs/
#
#SBATCH --job-name=th-nqueens
#SBATCH --partition=gpu-gen
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
# qos=normal caps mem at 30000M and 12 cpu per user; 28G/8cpu stays under it.
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/j7zang/Token-Heatmap
#SBATCH --output=outputs/hpc-nqueens-%j.log

# NOTE: deliberately no `-e`. A trace can exit non-zero during PLOTTING, after
# the JSON is already on disk (matplotlib chokes on some glyphs). Gating on the
# JSON existing is the real success test, the same way hpc-gen-full.slurm does it.
set -uo pipefail

export HF_HOME="${HF_HOME:-/work/j7zang/.cache/huggingface}"
BIN="${BIN:-/work/j7zang/th-gpu/bin/token-heatmap}"
SUITE="${SUITE:-pair}"
EXTRA="${EXTRA:-}"
# Extra flags for the solve suite's manifold pass, e.g.
#   MANIFOLD_EXTRA="--submodules resid_post" to cut its host-RAM use ~3x.
MANIFOLD_EXTRA="${MANIFOLD_EXTRA:-}"

echo "[nq] job=${SLURM_JOB_ID:-?} node=$(hostname) suite=$SUITE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- preflight -------------------------------------------------------------
# The N-Queens configs set `use_chat_template: true`. That key only reaches the
# generator if this checkout has the CLI flag wired up — an older build silently
# IGNORES unknown config keys, so a reasoning model would run as a raw completion
# and never enter <think>, and you would not find out until you read the trace.
# Fail loudly here instead.
if [ ! -x "$BIN" ]; then
  echo "[nq] FATAL: token-heatmap not found at $BIN (run: token-heatmap hpc setup)"; exit 1
fi
if ! "$BIN" trace --help 2>&1 | grep -q -- "--use-chat-template"; then
  echo "[nq] FATAL: this build has no --use-chat-template."
  echo "[nq]        The nqueens configs need it. Sync the repo and reinstall:"
  echo "[nq]          git -C /work/j7zang/Token-Heatmap pull"
  echo "[nq]          /work/j7zang/th-gpu/bin/pip install -e /work/j7zang/Token-Heatmap"
  exit 1
fi
echo "[nq] preflight OK — chat-template support present"

# Traces are gzipped by default, so the artifact is adaptive_token_trace.json.gz.
# Every token-heatmap reader accepts either name, but shell tests do not — resolve
# it once here rather than hard-coding an extension in each check below.
trace_file() {
  local out="$1"
  for f in "$out/adaptive_token_trace.json.gz" "$out/adaptive_token_trace.json"; do
    [ -f "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}

# Run one trace config; return non-zero only if the trace file is missing.
run_trace() {
  local cfg="$1" out="$2" label="$3"
  echo "[nq] --- trace: $label ($cfg -> $out)"
  "$BIN" trace --config "$cfg" --out "$out" $EXTRA \
    || echo "[nq] trace returned nonzero (plotting?) — checking output"
  local tf
  if ! tf=$(trace_file "$out"); then
    echo "[nq] FATAL: no trace at $out/adaptive_token_trace.json[.gz]"; return 1
  fi
  echo "[nq] $label trace: $tf ($(wc -c < "$tf") bytes)"
  # The answer token is the whole point of the pair suite — surface it in the log
  # so a failed run is obvious from the log alone, without pulling the trace back.
  # generated.txt holds prompt+completion, so take the TAIL: the head is just the
  # chat-template-wrapped prompt.
  [ -f "$out/generated.txt" ] && echo "[nq] $label answer tail: $(tail -c 120 "$out/generated.txt" | tr '\n' ' ')"
  return 0
}

case "$SUITE" in
  pair)
    A=outputs/nqueens-verify
    B=outputs/nqueens-conflict

    run_trace configs/nqueens-verify.yaml   "$A" "verify(valid)"    || exit 1
    run_trace configs/nqueens-conflict.yaml "$B" "conflict(invalid)" || exit 1

    echo "[nq] --- diff (valid vs conflict)"
    # --allow-prompt-mismatch is required and correct here: the pair deliberately
    # differs in prompt. It is safe only because the two prompts SUBSTITUTE two
    # digit tokens at the same positions — diff re-checks that the token counts
    # match and prints which positions differ, so a broken pair fails loudly.
    "$BIN" diff "$A/adaptive_token_trace.json" "$B/adaptive_token_trace.json" \
      --out outputs/nqueens-diff --metric cosine --allow-prompt-mismatch \
      || echo "[nq] diff failed"

    echo "[nq] --- causal grid (valid run)"
    "$BIN" causal --trace "$A/adaptive_token_trace.json" || echo "[nq] causal failed"

    echo "[nq] --- tuned lens (valid run)"
    "$BIN" tuned-lens --trace "$A/adaptive_token_trace.json" || echo "[nq] tuned-lens failed"

    echo "[nq] DONE -> $A  $B  outputs/nqueens-diff"
    ;;

  solve)
    C=outputs/nqueens-solve
    run_trace configs/nqueens-solve.yaml "$C" "solve(CoT)" || exit 1

    # Did the chain of thought actually finish? If </think> never appears the run
    # was truncated at max_new_tokens and the "final answer" tokens are absent.
    # Read the configured budget rather than hard-coding it, so the suggestion
    # stays right after the config is bumped.
    if [ -f "$C/generated.txt" ] && ! grep -q "</think>" "$C/generated.txt"; then
      CUR=$(sed -n 's/^max_new_tokens:[[:space:]]*\([0-9]\+\).*/\1/p' configs/nqueens-solve.yaml | head -1)
      echo "[nq] WARNING: no </think> — CoT truncated at max_new_tokens=${CUR:-?}."
      echo "[nq]          Raise it (→ $(( ${CUR:-1024} * 2 ))) and resubmit, or accept a"
      echo "[nq]          truncated search: the entropy/manifold read-outs do not need"
      echo "[nq]          the final answer, only the trajectory."
    fi

    # Manifold is numpy-only (no GPU); its host-RAM need scales with step count,
    # not model size. At 2048 steps this OOM-killed job 6027 — the cause was a
    # (T, T, d) pairwise-distance allocation in the TwoNN estimator (86 GB at
    # T=2048, d=2560), since fixed in analysis/manifold.py. MANIFOLD_EXTRA remains
    # the escape hatch for even longer runs: --layers / --submodules analyse a
    # subset.
    echo "[nq] --- manifold (search trajectory geometry)"
    "$BIN" manifold --trace "$C/adaptive_token_trace.json" --components 6 \
      ${MANIFOLD_EXTRA:-} \
      || echo "[nq] manifold failed (OOM at long step counts? see MANIFOLD_EXTRA above)"

    echo "[nq] DONE -> $C"
    ;;

  *)
    echo "[nq] FATAL: unknown SUITE='$SUITE' (expected 'pair' or 'solve')"; exit 1
    ;;
esac

du -sh outputs/nqueens-* 2>/dev/null || true
