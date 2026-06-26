"""``token-heatmap web build`` — build the Vite frontend for production serving.

Replaces ``scripts/build-frontend.sh``. Runs ``npm install`` + ``npm run build``
in ``web/frontend`` and prints how to ship the resulting ``dist/`` to a server
that has no Node.js (e.g. an HPC login node) and serve it same-origin from the
FastAPI backend.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from llm_token_heatmap.commands._util import repo_root


def run_web_build(args: argparse.Namespace) -> int:
    """Execute ``token-heatmap web build``."""
    frontend_dir = repo_root() / "web" / "frontend"
    if not frontend_dir.is_dir():
        print(f"error: frontend directory not found at {frontend_dir}", file=sys.stderr)
        return 2
    npm = shutil.which("npm")
    if npm is None:
        print("error: 'npm' is not on PATH. Install Node.js 20+ and re-run.", file=sys.stderr)
        return 2

    api_base = args.api_base_url
    env = {**os.environ, "VITE_API_BASE_URL": api_base}

    print("[web build] installing node_modules (if needed)…")
    rc = subprocess.run([npm, "install", "--prefer-offline"], cwd=str(frontend_dir)).returncode
    if rc != 0:
        return rc

    print(f"[web build] building frontend (VITE_API_BASE_URL='{api_base}')…")
    rc = subprocess.run([npm, "run", "build"], cwd=str(frontend_dir), env=env).returncode
    if rc != 0:
        return rc

    dist = frontend_dir / "dist"
    print(f"\n[web build] done — output at {dist}\n")
    print("Next steps:")
    print("  1. Copy dist/ to the server:")
    print(f"       rsync -av {dist}/ user@hpc:$(pwd)/web/frontend/dist/")
    print("  2. On the server, start the backend (Python only — no Node.js needed):")
    print("       token-heatmap trace --config configs/my_run.yaml --serve")
    print("       # or: cd web/backend && uvicorn llm_token_heatmap_api.main:app --host :: --port 8000")
    print("  3. Port-forward from your laptop (if on HPC):")
    print("       ssh -L 8000:localhost:8000 user@hpc")
    print("  4. Open http://localhost:8000")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser(
        "web",
        help="Frontend tasks (build the production dist/).",
        description="Frontend build tasks for the web app.",
    )
    web_sub = web.add_subparsers(dest="web_command", required=True)
    build = web_sub.add_parser(
        "build",
        help="Build the Vite frontend (dist/) for same-origin serving by the backend.",
        description=(
            "Run `npm install` + `npm run build` in web/frontend. The default "
            "empty VITE_API_BASE_URL makes the SPA use relative API paths "
            "(same-origin), so you can serve dist/ straight from the FastAPI "
            "backend on a host with no Node.js."
        ),
    )
    build.add_argument(
        "--api-base-url",
        default=os.environ.get("VITE_API_BASE_URL", ""),
        help="API base URL baked into the build (default: empty = relative/same-origin).",
    )
    build.set_defaults(func=run_web_build)
