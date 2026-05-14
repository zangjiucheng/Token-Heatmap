#!/bin/bash
# Set up the llm-token-heatmap development environment end-to-end:
# Python venv + core library + web backend + (if Node is available) the
# web frontend node_modules. Safe to re-run; each step is idempotent.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[setup] Setting up llm-token-heatmap development environment..."

if [ ! -d ".venv" ]; then
    echo "[setup] Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Installing core Python package..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

if [ -d "web/backend" ]; then
    echo "[setup] Installing web backend (web/backend) into the same venv..."
    pip install -e ./web/backend
else
    echo "[setup] web/backend not found; skipping backend install."
fi

if [ -d "web/frontend" ]; then
    if command -v npm >/dev/null 2>&1; then
        echo "[setup] Installing web frontend deps (npm install in web/frontend)..."
        (cd web/frontend && npm install)
    else
        echo "[setup] npm not found on PATH; skipping web/frontend install."
        echo "        Install Node.js 20+ and run 'cd web/frontend && npm install'"
        echo "        if you want to run the web app."
    fi
else
    echo "[setup] web/frontend not found; skipping frontend install."
fi

echo ""
echo "[setup] Setup complete."
echo "  Activate with:   source .venv/bin/activate"
echo "  Run example:     python examples/qwen_attention_inspect.py"
echo "  Start the app:   ./scripts/dev.sh"
