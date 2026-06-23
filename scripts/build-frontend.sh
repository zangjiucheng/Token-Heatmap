#!/usr/bin/env bash
# Build the Vite frontend for same-origin serving via the FastAPI backend.
#
# Run this on any machine that has Node.js 20+, then copy dist/ to the server.
#
# Usage
# -----
#   ./scripts/build-frontend.sh                              # relative API URLs (for same-origin)
#   VITE_API_BASE_URL=http://myhost:8000 ./scripts/build-frontend.sh
#
# After building, copy to the HPC / remote server:
#   rsync -av web/frontend/dist/ user@hpc:/path/to/Token-Heatmap/web/frontend/dist/
#
# Then on the HPC server, start the backend (Python only — no Node.js needed):
#   token-heatmap trace --config configs/my_run.yaml --serve
#   # or just start the backend directly:
#   cd web/backend && uvicorn llm_token_heatmap_api.main:app --host :: --port 8000
#
# Access the app (with SSH port-forwarding from your laptop):
#   ssh -L 8000:localhost:8000 user@hpc   # in a separate terminal
#   open http://localhost:8000             # on your laptop

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/web/frontend"

if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "error: frontend directory not found at $FRONTEND_DIR" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "error: 'npm' is not on PATH. Install Node.js 20+ and re-run." >&2
  exit 1
fi

# Default: empty base URL so all API calls use relative paths (same-origin).
export VITE_API_BASE_URL="${VITE_API_BASE_URL:-}"

echo "[build] installing node_modules (if needed)…"
(cd "$FRONTEND_DIR" && npm install --prefer-offline)

echo "[build] building frontend (VITE_API_BASE_URL='${VITE_API_BASE_URL}')…"
(cd "$FRONTEND_DIR" && npm run build)

DIST="$FRONTEND_DIR/dist"
echo ""
echo "[build] done — output at $DIST"
echo ""
echo "Next steps:"
echo "  1. Copy dist/ to the server:"
echo "       rsync -av $DIST/ user@hpc:\$(pwd)/web/frontend/dist/"
echo ""
echo "  2. On the server, start the backend:"
echo "       token-heatmap trace --config configs/my_run.yaml --serve"
echo "       # or:"
echo "       cd web/backend && uvicorn llm_token_heatmap_api.main:app --host :: --port 8000"
echo ""
echo "  3. Port-forward from your laptop (if on HPC):"
echo "       ssh -L 8000:localhost:8000 user@hpc"
echo ""
echo "  4. Open http://localhost:8000"
