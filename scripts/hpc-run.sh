#!/usr/bin/env bash
# One command, run from the LAPTOP: do the GPU compute on the HPC, pull
# EVERYTHING back locally, so nothing but the compute step needs the cluster.
#
#   laptop                         HPC (j7zang-gpu)
#   ------                         ----------------
#   config.yaml  --- scp --->      outputs/<name>/config.yaml
#                                  sbatch -> GPU node: trace + manifold   (the
#                                                                          only
#                                                                          remote
#                                                                          step)
#   outputs/<name>/  <-- rsync --  outputs/<name>/   (config, slurm log, trace,
#                                                     csv, png, sidecars)
#
# Everything for one run lives in a single self-contained folder,
# outputs/<name>/ — config, the Slurm log, and all artifacts — both on the HPC
# and (after the pull) locally. Nothing is scattered across the home dir.
#
# After it returns, the whole run is in ./outputs/<name>/ and you view it
# locally with no GPU and no tunnel (drag the JSON onto the frontend, or
# `token-heatmap serve outputs/<name>`).
#
# Quick start (defaults: l40s, qos=normal, full capture + manifold):
#   ./scripts/hpc-run.sh configs/wrap-text.yaml
#   ./scripts/hpc-run.sh ~/Downloads/trace-config.yaml   # a Build-page export
#
# Common options:
#   --name NAME          run name -> outputs/NAME locally + on HPC (default: config basename)
#   --model ID           override the model (e.g. Qwen/Qwen2.5-14B-Instruct)
#   --gpu rtx6000|l40s   GPU type. Both 48 GB. Default rtx6000 (RTX 6000 Ada),
#                        which auto-selects qos_rtx6000_max (200 G mem, 1-day
#                        walltime, 1 GPU/user). --gpu l40s -> qos=normal (30 G,
#                        12 h), handy when an rtx6000 job is already queued.
#   --qos QOS            Slurm qos (default: qos_rtx6000_max for rtx6000, normal for l40s)
#   --mem MEM            host memory (default 28G under qos=normal / 64G under qos_rtx6000_max)
#   --time HH:MM:SS      walltime (default 01:00:00)
#   --capture full|activations   full = +attention (slower); activations = manifold-only (default full)
#   --probe SCALAR       add a supervised manifold probe (e.g. line_position)
#   --4bit               load in 4-bit NF4 (for 32B+ on a single GPU)
#   --extra "FLAGS"      extra `trace` flags (e.g. "--max-new-tokens 320")
#   --serve              after pulling, start a LOCAL file server + print the viewer URL
#   --no-manifold        skip the manifold analysis pass
#   --no-sync            don't `git pull` the HPC repo first
#   --no-pull            leave outputs on the HPC (don't rsync back)
#   --setup              build/verify the GPU venv on the HPC first (one-time)
#   --force              skip the pre-flight "won't fit in VRAM" size check
#
# Everything is also env-overridable: SSH_HOST, REMOTE_REPO, REMOTE_BIN_GPU,
# LOCAL_VIEW_PORT, FRONTEND_PORT, POLL_SECONDS.

set -euo pipefail

# --- config (override via env) ---------------------------------------------
SSH_HOST="${SSH_HOST:-j7zang-gpu}"
REMOTE_REPO="${REMOTE_REPO:-/work/j7zang/Token-Heatmap}"
REMOTE_BIN_GPU="${REMOTE_BIN_GPU:-/work/j7zang/th-gpu/bin/token-heatmap}"
LOCAL_VIEW_PORT="${LOCAL_VIEW_PORT:-8001}"   # 8000 is the WC2026 service
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
POLL_SECONDS="${POLL_SECONDS:-15}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(dirname "$SCRIPT_DIR")"

# --- defaults --------------------------------------------------------------
CONFIG_LOCAL=""
NAME=""
MODEL=""
GPU="rtx6000"    # RTX 6000 Ada (48 GB, 1 TB-RAM node, roomier qos); --gpu l40s to switch
QOS=""           # resolved from --gpu after parsing (see below)
MEM=""           # resolved from the qos after parsing
TIME="01:00:00"
CAPTURE="full"
PROBE=""
EXTRA=""
DO_MANIFOLD=1
DO_SYNC=1
DO_PULL=1
DO_SERVE=0
DO_SETUP=0
FOUR_BIT=0
FORCE=0

die() { echo "error: $*" >&2; exit 2; }

# --- args ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)        NAME="${2:?}"; shift 2 ;;
    --model)       MODEL="${2:?}"; shift 2 ;;
    --gpu)         GPU="${2:?}"; shift 2 ;;
    --qos)         QOS="${2:?}"; shift 2 ;;
    --mem)         MEM="${2:?}"; shift 2 ;;
    --time)        TIME="${2:?}"; shift 2 ;;
    --capture)     CAPTURE="${2:?}"; shift 2 ;;
    --probe)       PROBE="${2:?}"; shift 2 ;;
    --extra)       EXTRA="${2:?}"; shift 2 ;;
    --4bit)        FOUR_BIT=1; shift ;;
    --serve)       DO_SERVE=1; shift ;;
    --no-manifold) DO_MANIFOLD=0; shift ;;
    --no-sync)     DO_SYNC=0; shift ;;
    --no-pull)     DO_PULL=0; shift ;;
    --setup)       DO_SETUP=1; shift ;;
    --force)       FORCE=1; shift ;;
    -h|--help)     sed -n '2,55p' "$0"; exit 0 ;;
    -*)            die "unknown flag '$1'" ;;
    *)             [[ -z "$CONFIG_LOCAL" ]] && CONFIG_LOCAL="$1" || die "unexpected arg '$1'"; shift ;;
  esac
done

[[ -n "$CONFIG_LOCAL" ]] || die "usage: hpc-run.sh <config.yaml> [options]   (see --help)"
[[ -f "$CONFIG_LOCAL" ]] || die "config file not found: $CONFIG_LOCAL"

# Resolve the qos/mem from the GPU when not given explicitly. The rtx6000 cards
# are RTX 6000 Ada (48 GB, same as l40s) on a 1 TB-RAM node, and their dedicated
# qos (qos_rtx6000_max) grants 200 G host mem + a 1-day walltime — far roomier
# than qos=normal's 30 G / 12 h — so default rtx6000 runs onto it.
if [[ -z "$QOS" ]]; then
  if [[ "$GPU" == "rtx6000" ]]; then QOS="qos_rtx6000_max"; else QOS="normal"; fi
fi
if [[ -z "$MEM" ]]; then
  if [[ "$QOS" == "qos_rtx6000_max" ]]; then MEM="64G"; else MEM="28G"; fi
fi

# Run name: --name, else the config's basename (sans extension).
if [[ -z "$NAME" ]]; then
  NAME="$(basename "$CONFIG_LOCAL")"; NAME="${NAME%.*}"
fi
# Sanitize to a safe path segment.
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name '$NAME' has unsafe characters; pass --name with [A-Za-z0-9._-]."

OUT_REL="outputs/${NAME}"                       # the one folder a run lives in
REMOTE_CONFIG="${OUT_REL}/config.yaml"          # config travels inside the run folder
LOCAL_OUT="${LOCAL_REPO}/${OUT_REL}"

# 4-bit -> append the loader flag to EXTRA.
[[ "$FOUR_BIT" == 1 ]] && EXTRA="${EXTRA} --load-in-4bit"
# Manifold extra flags (probe).
MANIFOLD_EXTRA=""
[[ -n "$PROBE" ]] && MANIFOLD_EXTRA="--components 6 --probe ${PROBE}"

# --- pre-flight VRAM heuristic ---------------------------------------------
# A 32B model in bf16 (~64 GB) won't fit one l40s (~45 GB); without this guard
# you'd queue, load for minutes, then OOM. Estimate from the model id's size
# tag and refuse a clearly-too-big bf16 run before submitting. Heuristic only —
# override with --force.
EFFECTIVE_MODEL="$MODEL"
if [[ -z "$EFFECTIVE_MODEL" ]]; then  # model comes from the config YAML
  EFFECTIVE_MODEL="$(grep -iE '^[[:space:]]*model:' "$CONFIG_LOCAL" | head -1 \
    | sed -E 's/^[^:]*:[[:space:]]*//; s/^["'"'"']//; s/["'"'"']?[[:space:]]*$//')"
fi
SIZE_B="$(printf '%s' "$EFFECTIVE_MODEL" | grep -oiE '[0-9]+(\.[0-9]+)?[bB]' | head -1 | sed -E 's/[bB]$//')"
if [[ -n "$SIZE_B" && "$FORCE" != 1 ]]; then
  case "$GPU" in
    rtx6000) VRAM=47 ;;   # RTX 6000 Ada — 48 GB (NOT the old 24 GB Quadro)
    *)       VRAM=45 ;;   # l40s — 46 GB (and a safe default)
  esac
  # bytes/param ×100: bf16=200, NF4 4-bit≈65 (weights + overhead).
  BPP=200; [[ "$FOUR_BIT" == 1 ]] && BPP=65
  EST=$(( ${SIZE_B%.*} * BPP / 100 ))            # integer GB; sub-1B floors to 0
  if [[ "$EST" -gt $(( VRAM * 92 / 100 )) ]]; then
    if [[ "$FOUR_BIT" == 1 ]]; then
      die "model '${EFFECTIVE_MODEL}' (~${SIZE_B}B) is ~${EST} GB even in 4-bit — too big for one ${GPU} (~${VRAM} GB). Pick a smaller --model or shard across GPUs. (override: --force)"
    fi
    die "model '${EFFECTIVE_MODEL}' (~${SIZE_B}B) is ~${EST} GB in bf16 — won't fit one ${GPU} (~${VRAM} GB). Add --4bit (fits ~32B), pick a smaller --model, or override with --force."
  elif [[ "$EST" -gt $(( VRAM * 80 / 100 )) ]]; then
    echo "[hpc-run] ⚠ ${EFFECTIVE_MODEL} (~${EST} GB) is close to the ${GPU} limit (~${VRAM} GB) — may OOM under load."
  fi
fi

echo "[hpc-run] host=${SSH_HOST}  name=${NAME}  gpu=${GPU}  qos=${QOS}  capture=${CAPTURE}"
echo "[hpc-run] config(local)=${CONFIG_LOCAL}  ->  ${OUT_REL}  (manifold=$([[ $DO_MANIFOLD == 1 ]] && echo on || echo off))"

# --- 0. one-time GPU env setup (optional) ----------------------------------
if [[ "$DO_SETUP" == 1 ]]; then
  echo "[hpc-run] --setup: ensuring the GPU venv exists on the HPC..."
  "${SCRIPT_DIR}/hpc-setup.sh"
fi

# --- pre-flight: HPC reachable + GPU venv present --------------------------
echo "[hpc-run] checking the HPC..."
ssh -o ConnectTimeout=10 "$SSH_HOST" "test -x '${REMOTE_BIN_GPU}'" || die \
  "GPU venv not found at ${REMOTE_BIN_GPU} on ${SSH_HOST}. Build it once: ./scripts/hpc-run.sh ${CONFIG_LOCAL} --setup  (or run ./scripts/hpc-setup.sh)."

# --- 1. ship the config into the run folder --------------------------------
echo "[hpc-run] [1/4] uploading config -> ${SSH_HOST}:${REMOTE_REPO}/${REMOTE_CONFIG}"
# Create the run folder up front so the config and the Slurm log both land
# inside it (the Slurm --output path needs the dir to exist before the job).
ssh "$SSH_HOST" "mkdir -p '${REMOTE_REPO}/${OUT_REL}'"
scp -q "$CONFIG_LOCAL" "${SSH_HOST}:${REMOTE_REPO}/${REMOTE_CONFIG}"

# --- 2. submit the GPU job (the only remote compute) -----------------------
# The unquoted heredoc expands LOCAL variables before sending; bash -l gives a
# login shell so the Slurm binaries are on PATH. `git pull` goes to stderr so
# stdout carries only the parsable job id.
echo "[hpc-run] [2/4] submitting Slurm job (sbatch)..."
MODEL_ARG=""; [[ -n "$MODEL" ]] && MODEL_ARG="MODEL='${MODEL}'"
SYNC_CMD=":"
[[ "$DO_SYNC" == 1 ]] && SYNC_CMD="git -C '${REMOTE_REPO}' pull --ff-only >&2 || echo '[hpc-run] (remote git pull skipped/failed — using existing checkout)' >&2"
MANIFOLD_ENV=""
[[ "$DO_MANIFOLD" == 1 ]] && MANIFOLD_ENV="MANIFOLD_EXTRA='${MANIFOLD_EXTRA}'"

JOB_ID="$(ssh "$SSH_HOST" bash -l -s <<EOF
set -e
${SYNC_CMD}
cd '${REMOTE_REPO}'
BIN='${REMOTE_BIN_GPU}' \
CONFIG='${REMOTE_CONFIG}' \
OUT='${OUT_REL}' \
CAPTURE='${CAPTURE}' \
EXTRA='${EXTRA}' \
${MODEL_ARG} \
${MANIFOLD_ENV} \
sbatch --parsable \
  --job-name='th-${NAME}' \
  --qos='${QOS}' --gres='gpu:${GPU}:1' --mem='${MEM}' --time='${TIME}' \
  --output='${OUT_REL}/slurm-%j.log' \
  --export=ALL,BIN,CONFIG,OUT,CAPTURE,EXTRA,MODEL,MANIFOLD_EXTRA \
  scripts/hpc-gen.slurm
EOF
)"
JOB_ID="$(echo "$JOB_ID" | tr -d '[:space:]')"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || die "did not get a numeric Slurm job id back (got: '${JOB_ID}'). Check the HPC repo / venv."
REMOTE_LOG="${OUT_REL}/slurm-${JOB_ID}.log"   # inside the run folder
echo "[hpc-run] submitted job ${JOB_ID}  (log: ${REMOTE_REPO}/${REMOTE_LOG})"

# --- 3. wait for it (poll squeue; show the latest log line) -----------------
echo "[hpc-run] [3/4] waiting for the GPU job to finish (poll every ${POLL_SECONDS}s; Ctrl+C is safe — the job keeps running)..."
last_state=""
while true; do
  # The remote `bash -lc '…'` must reach the login shell as ONE arg, or ssh
  # token-joins it and `bash -lc squeue …` runs squeue with no args (dumping a
  # headered job list that never goes empty -> infinite loop). Quote the whole
  # `bash -lc '…'` as a single ssh argument.
  state="$(ssh "$SSH_HOST" "bash -lc 'squeue -h -j ${JOB_ID} -o %T 2>/dev/null'" | head -1 || true)"
  [[ -z "$state" ]] && break   # gone from the queue -> finished
  if [[ "$state" != "$last_state" ]]; then
    echo "[hpc-run]   state: ${state}"
    last_state="$state"
  fi
  # Surface the latest log line so progress is visible while RUNNING.
  if [[ "$state" == "RUNNING" ]]; then
    tail_line="$(ssh "$SSH_HOST" "tail -n 1 '${REMOTE_REPO}/${REMOTE_LOG}' 2>/dev/null" || true)"
    [[ -n "$tail_line" ]] && echo "[hpc-run]   · ${tail_line}"
  fi
  sleep "$POLL_SECONDS"
done

FINAL_STATE="$(ssh "$SSH_HOST" "bash -lc 'sacct -j ${JOB_ID} -n -P -o State 2>/dev/null'" | head -1 || true)"
echo "[hpc-run] job ${JOB_ID} finished: ${FINAL_STATE:-unknown}"
echo "[hpc-run] --- remote log tail -------------------------------------------"
ssh "$SSH_HOST" "tail -n 12 '${REMOTE_REPO}/${REMOTE_LOG}' 2>/dev/null" || true
echo "[hpc-run] ---------------------------------------------------------------"
if [[ "$FINAL_STATE" != COMPLETED* ]]; then
  die "remote job did not complete cleanly (state: ${FINAL_STATE:-unknown}). Outputs left on the HPC; inspect ${REMOTE_REPO}/${REMOTE_LOG}."
fi

# --- 4. pull EVERYTHING back -----------------------------------------------
if [[ "$DO_PULL" == 0 ]]; then
  echo "[hpc-run] --no-pull: leaving results on the HPC at ${REMOTE_REPO}/${OUT_REL}"
  echo "[hpc-run] view remotely with: ./scripts/hpc-serve.sh ${OUT_REL}"
  exit 0
fi

echo "[hpc-run] [4/4] pulling results -> ${LOCAL_OUT}/ ..."
mkdir -p "$LOCAL_OUT"
if command -v rsync >/dev/null 2>&1; then
  # --stats (not --info=progress2): the latter is rsync >=3.1 only, but macOS
  # ships rsync 2.6.9 — `--stats` prints a portable end-of-transfer summary and
  # is supported by both old and new rsync.
  rsync -az --stats -e ssh \
    "${SSH_HOST}:${REMOTE_REPO}/${OUT_REL}/" "${LOCAL_OUT}/"
else
  scp -q -r "${SSH_HOST}:${REMOTE_REPO}/${OUT_REL}/." "${LOCAL_OUT}/"
fi
# The Slurm log + config now live inside ${OUT_REL}/, so the pull above already
# brought them — nothing extra to fetch.

echo "[hpc-run] ✓ done — everything is local now: ${OUT_REL}/"
ls -1 "$LOCAL_OUT" | sed 's/^/[hpc-run]     /'

# --- view locally (no GPU, no tunnel) --------------------------------------
TRACE_JSON="${OUT_REL}/adaptive_token_trace.json"
VIEWER_URL="http://localhost:${FRONTEND_PORT}/?trace=http://localhost:${LOCAL_VIEW_PORT}/adaptive_token_trace.json"
echo
if [[ "$DO_SERVE" == 1 ]]; then
  LOCAL_BIN="token-heatmap"
  command -v "$LOCAL_BIN" >/dev/null 2>&1 || LOCAL_BIN="${LOCAL_REPO}/.venv/bin/token-heatmap"
  [[ -x "$LOCAL_BIN" ]] || command -v "$LOCAL_BIN" >/dev/null 2>&1 || die \
    "--serve: no local token-heatmap (run ./scripts/setup.sh). The files are pulled; serve them yourself."
  echo "[hpc-run] --serve: starting a local file server on :${LOCAL_VIEW_PORT} (Ctrl+C to stop)."
  echo "[hpc-run] open the viewer (start the frontend with 'cd web/frontend && npm run dev' if needed):"
  echo "            ${VIEWER_URL}"
  exec "$LOCAL_BIN" serve "$OUT_REL" --port "$LOCAL_VIEW_PORT"
else
  echo "[hpc-run] view it locally (no GPU needed) — either:"
  echo "  A) drag ${TRACE_JSON} onto http://localhost:${FRONTEND_PORT}"
  echo "  B) serve + open:"
  echo "       token-heatmap serve ${OUT_REL} --port ${LOCAL_VIEW_PORT}"
  echo "       ${VIEWER_URL}"
  echo "  (or re-run with --serve to do B automatically)"
fi
