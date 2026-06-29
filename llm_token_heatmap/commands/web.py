"""``token-heatmap web build`` — build the Vite frontend for static hosting.

Replaces ``scripts/build-frontend.sh``. Runs ``npm install`` + ``npm run build``
in ``web/frontend`` and prints how to ship the resulting ``dist/`` to a host
that has no Node.js (e.g. an HPC login node) and serve it with any static file
server. The viewer is backend-free: it loads traces from a dropped file or the
bundled sample.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from llm_token_heatmap.commands._util import repo_root


def run_web_build(args: argparse.Namespace) -> int:  # noqa: ARG001 — uniform run(args) signature
    """Execute ``token-heatmap web build``."""
    frontend_dir = repo_root() / "web" / "frontend"
    if not frontend_dir.is_dir():
        print(f"error: frontend directory not found at {frontend_dir}", file=sys.stderr)
        return 2
    npm = shutil.which("npm")
    if npm is None:
        print("error: 'npm' is not on PATH. Install Node.js 20+ and re-run.", file=sys.stderr)
        return 2

    print("[web build] installing node_modules (if needed)…")
    rc = subprocess.run([npm, "install", "--prefer-offline"], cwd=str(frontend_dir)).returncode
    if rc != 0:
        return rc

    print("[web build] building frontend…")
    rc = subprocess.run([npm, "run", "build"], cwd=str(frontend_dir)).returncode
    if rc != 0:
        return rc

    dist = frontend_dir / "dist"
    print(f"\n[web build] done — output at {dist}\n")
    print("Next steps:")
    print("  1. Serve dist/ with any static file server, e.g.:")
    print(f"       python -m http.server -d {dist} 8080")
    print("  2. Open the viewer and drag a trace JSON onto the page:")
    print("       http://localhost:8080/")
    print("  (Produce traces with `token-heatmap trace`.)")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser(
        "web",
        help="Frontend tasks (build the static viewer dist/).",
        description="Frontend build tasks for the web app (a static, file-based trace viewer).",
    )
    web_sub = web.add_subparsers(dest="web_command", required=True)
    build = web_sub.add_parser(
        "build",
        help="Build the Vite frontend (dist/) for static hosting.",
        description=(
            "Run `npm install` + `npm run build` in web/frontend. The viewer is a "
            "static SPA with no backend — serve the resulting dist/ from any static "
            "file server on a host with no Node.js. Traces load from a dropped file "
            "or the bundled sample."
        ),
    )
    build.set_defaults(func=run_web_build)
