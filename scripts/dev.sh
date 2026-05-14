#!/usr/bin/env bash
# Boot the FastAPI backend and the Vite frontend together for local dev.
#
# Usage:
#   ./scripts/dev.sh                          # defaults: backend :8000, frontend :5173
#   BACKEND_PORT=8765 ./scripts/dev.sh
#   FRONTEND_PORT=5180 ./scripts/dev.sh       # CORS origin is auto-adjusted
#
# Ctrl+C terminates both processes cleanly.

set -euo pipefail

# `wait -n` was added in bash 4.3. macOS ships bash 3.2; install a newer one
# with `brew install bash` and run via /opt/homebrew/bin/bash.
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
  echo "error: dev.sh requires bash >= 4.3 (you have ${BASH_VERSION})." >&2
  echo "       On macOS: brew install bash && /opt/homebrew/bin/bash ./scripts/dev.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/web/backend"
FRONTEND_DIR="$REPO_ROOT/web/frontend"

BACKEND_PORT="${BACKEND_PORT:-${LLM_HEATMAP_API_PORT:-8000}}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Keep the backend's CORS list in sync with whichever port the frontend
# actually lands on. If the caller already set LLM_HEATMAP_ALLOWED_ORIGINS,
# respect it; otherwise scope it to the chosen frontend port.
export LLM_HEATMAP_ALLOWED_ORIGINS="${LLM_HEATMAP_ALLOWED_ORIGINS:-http://localhost:$FRONTEND_PORT}"
export VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://localhost:$BACKEND_PORT}"

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "error: backend directory not found at $BACKEND_DIR" >&2
  exit 1
fi
if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "error: frontend directory not found at $FRONTEND_DIR" >&2
  exit 1
fi

# Auto-activate the project venv if uvicorn isn't already on PATH. Avoids
# the common "I forgot to source .venv/bin/activate" footgun.
if ! command -v uvicorn >/dev/null 2>&1; then
  if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
  fi
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "error: 'uvicorn' not found." >&2
  echo "       Run scripts/setup.sh first, then activate the venv with" >&2
  echo "       'source .venv/bin/activate' before running this script." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "error: 'npm' is not on PATH. Install Node.js 20+ and re-run." >&2
  exit 1
fi

# /dev/tcp is built into bash, so this works without lsof/netstat being
# installed. Connect attempt succeeds iff something is already listening.
port_in_use() {
  (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null
}
if port_in_use "$BACKEND_PORT"; then
  echo "error: backend port $BACKEND_PORT is already in use." >&2
  exit 1
fi
if port_in_use "$FRONTEND_PORT"; then
  echo "error: frontend port $FRONTEND_PORT is already in use." >&2
  exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup INT TERM EXIT

echo "[dev] starting backend on :$BACKEND_PORT (CORS origin: $LLM_HEATMAP_ALLOWED_ORIGINS)"
# Process substitution prefixes each line with a tag. `exec` replaces the
# subshell so $! refers to uvicorn itself, not a wrapper shell. When uvicorn
# exits the sed pipes see EOF and exit on their own.
(
  cd "$BACKEND_DIR"
  # `::` binds IPv6 with V4-mapped accepting (IPV6_V6ONLY=0 by default on
  # Linux and macOS). Without this, browsers that resolve `localhost` to ::1
  # hit ERR_CONNECTION_REFUSED while curl falls back to 127.0.0.1 and works.
  exec uvicorn llm_token_heatmap_api.main:app --reload \
    --host :: --port "$BACKEND_PORT" \
    > >(sed -u 's/^/[backend]  /') \
    2> >(sed -u 's/^/[backend]  /' >&2)
) &
BACKEND_PID=$!

echo "[dev] starting frontend on :$FRONTEND_PORT (API base: $VITE_API_BASE_URL)"
(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --port "$FRONTEND_PORT" --strictPort \
    > >(sed -u 's/^/[frontend] /') \
    2> >(sed -u 's/^/[frontend] /' >&2)
) &
FRONTEND_PID=$!

echo "[dev] backend pid=$BACKEND_PID, frontend pid=$FRONTEND_PID"
echo "[dev] open http://localhost:$FRONTEND_PORT (Ctrl+C to stop)"

# Exit as soon as either process exits, so a crashed backend takes the whole
# stack down rather than leaving an orphan frontend serving stale UI.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
