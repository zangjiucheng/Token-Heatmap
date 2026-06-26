"""Small shared helpers for the operational sub-commands (stdlib only)."""

from __future__ import annotations

import socket
from pathlib import Path


def repo_root() -> Path:
    """Repo root, resolved from this file (``llm_token_heatmap/commands/_util.py``)."""
    return Path(__file__).resolve().parents[2]


def port_in_use(port: int) -> bool:
    """True if something is already listening on ``port``.

    Checks IPv4 ``127.0.0.1`` *and* IPv6 ``::1`` — the dev servers bind IPv6
    (uvicorn ``--host ::``, Vite ``[::1]``), so an IPv4-only probe (the old
    ``/dev/tcp/127.0.0.1`` check in ``dev.sh``) would falsely report the port
    free and let two servers fight over it.
    """
    for family, addr in (
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port)),
    ):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.3)
                if sock.connect_ex(addr) == 0:
                    return True
        except OSError:
            continue
    return False
