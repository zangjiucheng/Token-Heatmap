#!/usr/bin/env bash
# From the LAPTOP: SSH into the HPC, start the token-heatmap file server there,
# and forward it to a local port so the local frontend can fetch the trace.
#
# One SSH session does both the tunnel and the remote server, so a single
# Ctrl+C tears down both (the remote process gets SIGHUP via the -t pty — no
# orphaned server left listening on the node).
#
# Usage:
#   ./scripts/hpc-serve.sh                       # serve outputs/example-run
#   ./scripts/hpc-serve.sh outputs/other-run     # serve a different dir
#   ./scripts/hpc-serve.sh --gen                 # regenerate trace+manifold first, then serve
#
# Override any default via env:
#   SSH_HOST=j7zang-gpu REMOTE_PORT=8000 LOCAL_PORT=8001 ./scripts/hpc-serve.sh
#
# The frontend is separate — run it on the laptop (scripts/dev.sh or
# `npm run dev` in web/frontend). This script only handles the remote server.

set -euo pipefail

# --- config (override via env) ---------------------------------------------
SSH_HOST="${SSH_HOST:-j7zang-gpu}"                          # ~/.ssh/config alias
REMOTE_REPO="${REMOTE_REPO:-/work/j7zang/Token-Heatmap}"    # repo checkout on HPC
REMOTE_BIN="${REMOTE_BIN:-/work/j7zang/.local/bin/token-heatmap}"  # full path: ~/.local/bin isn't on a non-interactive SSH PATH
REMOTE_PORT="${REMOTE_PORT:-8000}"                          # file-server port on the HPC
LOCAL_PORT="${LOCAL_PORT:-8001}"                            # laptop port (8000 is taken by WC2026)
FRONTEND_PORT="${FRONTEND_PORT:-5173}"                      # for the printed viewer URL
CONFIG="${CONFIG:-configs/example.yaml}"                    # only used with --gen

# --- args ------------------------------------------------------------------
REGEN=0
REMOTE_DIR="outputs/example-run"   # configs/example.yaml -> out: outputs/example-run
for arg in "$@"; do
  case "$arg" in
    --gen|-g) REGEN=1 ;;
    -*) echo "error: unknown flag '$arg'" >&2; exit 2 ;;
    *)  REMOTE_DIR="$arg" ;;
  esac
done

VIEWER_URL="http://localhost:${FRONTEND_PORT}/?trace=http://localhost:${LOCAL_PORT}/adaptive_token_trace.json"

# --- pre-flight: local port free? (/dev/tcp is built into bash) ------------
if (echo > "/dev/tcp/127.0.0.1/${LOCAL_PORT}") 2>/dev/null; then
  echo "error: local port ${LOCAL_PORT} is already in use — pick another with LOCAL_PORT=… ." >&2
  echo "       (port 8000 is your WC2026 service; that's why the default is 8001)" >&2
  exit 1
fi

# --- build the remote command ----------------------------------------------
# exec on the final serve so SIGHUP reaches token-heatmap directly when the
# SSH connection drops, instead of a wrapper shell swallowing it.
REMOTE_CMD="cd '${REMOTE_REPO}'"
if [[ "$REGEN" == "1" ]]; then
  # Capture the full set so every UI tab is populated: attention (forces eager
  # attention -- slower, but that's what lights up the Attention tab),
  # logit-lens (from the config), and activations. Then add manifold geometry.
  REMOTE_CMD="${REMOTE_CMD} && '${REMOTE_BIN}' trace --config '${CONFIG}'"
  REMOTE_CMD="${REMOTE_CMD} --capture-attention --attention-layers all --capture-full-attention"
  REMOTE_CMD="${REMOTE_CMD} --capture-activations --capture-full-activations"
  REMOTE_CMD="${REMOTE_CMD} && '${REMOTE_BIN}' manifold --trace '${REMOTE_DIR}/adaptive_token_trace.json'"
fi
REMOTE_CMD="${REMOTE_CMD} && exec '${REMOTE_BIN}' serve '${REMOTE_DIR}' --port ${REMOTE_PORT}"

echo "[hpc-serve] host=${SSH_HOST}"
echo "[hpc-serve] serving ${REMOTE_DIR} on remote :${REMOTE_PORT}  ->  forwarded to local :${LOCAL_PORT}"
[[ "$REGEN" == "1" ]] && echo "[hpc-serve] --gen: regenerating full trace (attention + logit-lens + activations) + manifold first (runs the model, eager attention; takes a while)"
echo "[hpc-serve] once it says 'Serving', open the viewer:"
echo "              ${VIEWER_URL}"
echo "[hpc-serve] Ctrl+C here stops the tunnel AND the remote server."
echo

# -t: pty so Ctrl+C/SIGHUP reaches the remote process and cleans it up.
# ExitOnForwardFailure: fail loudly if :LOCAL_PORT can't be bound instead of
#   silently connecting with no tunnel.
# ServerAlive*: drop the connection if the node goes away (keeps the tunnel honest).
exec ssh -t \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
  "${SSH_HOST}" \
  "${REMOTE_CMD}"
