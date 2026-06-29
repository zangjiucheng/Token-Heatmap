#!/bin/bash
# Set up the llm-token-heatmap development environment end-to-end:
# Python venv + core library + (if Node is available) the web frontend
# node_modules. Safe to re-run; each step is idempotent.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[setup] Setting up llm-token-heatmap development environment..."

# Pick the interpreter for the venv. Torch only ships wheels for a range of
# Python versions (currently 3.10–3.13), so if the default python3 is too new
# (e.g. 3.14) the torch install fails. Allow PYTHON=… to override, and fall
# back to python3.13/3.12 if the default is unsupported.
PYTHON="${PYTHON:-python3}"
py_minor() { "$1" -c 'import sys; print(sys.version_info[1])' 2>/dev/null; }
if [ ! -d ".venv" ]; then
    if [ "${PYTHON}" = "python3" ]; then
        minor="$(py_minor python3)"
        if [ -n "$minor" ] && [ "$minor" -ge 14 ]; then
            for alt in python3.13 python3.12 python3.11 python3.10; do
                if command -v "$alt" >/dev/null 2>&1; then PYTHON="$alt"; break; fi
            done
            echo "[setup] python3 is 3.${minor} (no torch wheels yet); using ${PYTHON} instead."
            echo "        Override with: PYTHON=python3.12 ./scripts/setup.sh"
        fi
    fi
    echo "[setup] Creating virtual environment in .venv (${PYTHON})..."
    "${PYTHON}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Installing core Python package..."
pip install --upgrade pip
pip install -e ".[dev,models]"

if [ -d "app" ]; then
    if command -v npm >/dev/null 2>&1; then
        echo "[setup] Installing web frontend deps (npm install in app)..."
        (cd app && npm install)
    else
        echo "[setup] npm not found on PATH; skipping app install."
        echo "        Install Node.js 20+ and run 'cd app && npm install'"
        echo "        if you want to run the web app."
    fi
else
    echo "[setup] app not found; skipping frontend install."
fi

echo ""
echo "[setup] Setup complete."
echo "  Activate with:   source .venv/bin/activate"
echo "  Run example:     python examples/qwen_attention_inspect.py"
echo "  Produce:         token-heatmap trace --config configs/example.yaml"
echo "  View:            drag the JSON onto the viewer (cd app && npm run dev)"
