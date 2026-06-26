"""``token-heatmap dev`` — run the FastAPI backend + Vite frontend together.

Replaces the old ``scripts/dev.sh``. Starts uvicorn (``--reload``) for
``web/backend`` and ``npm run dev`` for ``web/frontend``, wires their ports +
CORS together, prefixes their output, and tears **both** down cleanly on Ctrl+C
or when either exits.

Two things it does that the shell script couldn't do reliably:

- Port probing is IPv4 **and** IPv6 aware (see :func:`._util.port_in_use`), so a
  requested port that is busy on ``::1`` is detected and skipped.
- Each child runs in its own process group (``start_new_session``) and is
  killed with ``killpg`` on shutdown, so uvicorn's ``--reload`` worker and
  Vite's node child can't survive as orphans.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from llm_token_heatmap.commands._util import next_free_port, repo_root


def _relay(proc: subprocess.Popen[str], prefix: str, lock: threading.Lock) -> None:
    """Stream a child's merged stdout/stderr to ours, line-prefixed."""
    stream = proc.stdout
    if stream is None:
        return
    for line in stream:
        with lock:
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()


def _spawn(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group → killpg reaps the whole tree
    )


def _terminate_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the child's process group, escalating to SIGKILL if it lingers."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)


def run_dev(args: argparse.Namespace) -> int:
    """Execute ``token-heatmap dev``."""
    repo = repo_root()
    backend_dir = repo / "web" / "backend"
    frontend_dir = repo / "web" / "frontend"
    if not backend_dir.is_dir():
        print(f"error: backend directory not found at {backend_dir}", file=sys.stderr)
        return 2
    if not frontend_dir.is_dir():
        print(f"error: frontend directory not found at {frontend_dir}", file=sys.stderr)
        return 2

    if importlib.util.find_spec("uvicorn") is None:
        print(
            "error: 'uvicorn' is not importable in this environment.\n"
            "       Run scripts/setup.sh and activate the venv "
            "('source .venv/bin/activate') first.",
            file=sys.stderr,
        )
        return 2
    npm = shutil.which("npm")
    if npm is None:
        print("error: 'npm' is not on PATH. Install Node.js 20+ and re-run.", file=sys.stderr)
        return 2

    # Auto-fall-back past busy ports so `dev` just works instead of erroring.
    backend_port = next_free_port(args.backend_port)
    if backend_port != args.backend_port:
        print(f"[dev] backend port {args.backend_port} in use -> using {backend_port}")
    frontend_port = next_free_port(args.frontend_port)
    if frontend_port != args.frontend_port:
        print(f"[dev] frontend port {args.frontend_port} in use -> using {frontend_port}")

    # Keep the backend's CORS list and the frontend's API base in sync with the
    # ports we landed on; respect explicit overrides if the caller set them.
    allowed_origins = os.environ.get(
        "LLM_HEATMAP_ALLOWED_ORIGINS", f"http://localhost:{frontend_port}"
    )
    api_base = os.environ.get("VITE_API_BASE_URL", f"http://localhost:{backend_port}")

    backend_env = {**os.environ, "LLM_HEATMAP_ALLOWED_ORIGINS": allowed_origins}
    frontend_env = {**os.environ, "VITE_API_BASE_URL": api_base}

    print(f"[dev] starting backend on :{backend_port} (CORS origin: {allowed_origins})")
    # `::` binds IPv6 with V4-mapped accepting, so a browser resolving localhost
    # to ::1 reaches it just like curl on 127.0.0.1.
    backend = _spawn(
        [
            sys.executable, "-m", "uvicorn",
            "llm_token_heatmap_api.main:app",
            "--reload", "--host", "::", "--port", str(backend_port),
        ],
        cwd=backend_dir,
        env=backend_env,
    )
    print(f"[dev] starting frontend on :{frontend_port} (API base: {api_base})")
    frontend = _spawn(
        [npm, "run", "dev", "--", "--port", str(frontend_port), "--strictPort"],
        cwd=frontend_dir,
        env=frontend_env,
    )

    lock = threading.Lock()
    threading.Thread(target=_relay, args=(backend, "[backend]  ", lock), daemon=True).start()
    threading.Thread(target=_relay, args=(frontend, "[frontend] ", lock), daemon=True).start()

    print(f"[dev] backend pid={backend.pid}, frontend pid={frontend.pid}")
    print(f"[dev] open http://localhost:{frontend_port} (Ctrl+C to stop)")

    # Route SIGTERM through the same KeyboardInterrupt path as Ctrl+C so a plain
    # `kill <pid>` (or a supervisor's polite stop) still runs the killpg cleanup
    # below instead of leaving orphaned uvicorn/vite behind.
    def _raise_interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    prev_term = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        # Exit as soon as either process exits, so a crashed backend takes the
        # whole stack down rather than leaving an orphan frontend.
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        _terminate_group(backend)
        _terminate_group(frontend)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dev",
        help="Run the FastAPI backend + Vite frontend together for local development.",
        description=(
            "Boot uvicorn (web/backend, --reload) and the Vite dev server "
            "(web/frontend) together, with their ports + CORS wired up. Busy "
            "ports auto-advance to the next free one. Ctrl+C stops both cleanly "
            "(process groups are killed, so no orphan uvicorn/vite is left "
            "listening). Needs the venv active (uvicorn) and Node.js 20+ (npm)."
        ),
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=int(os.environ.get("BACKEND_PORT", os.environ.get("LLM_HEATMAP_API_PORT", "8000"))),
        help="Backend (uvicorn) port; auto-advances if busy (default: 8000).",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=int(os.environ.get("FRONTEND_PORT", "5173")),
        help="Frontend (Vite) port; auto-advances if busy (default: 5173).",
    )
    parser.set_defaults(func=run_dev)
