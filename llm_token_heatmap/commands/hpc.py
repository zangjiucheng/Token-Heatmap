"""``token-heatmap hpc {setup,run}`` — laptop-side HPC orchestration.

Thin Python wrappers around ssh / scp / rsync / sbatch that replace the old
``scripts/hpc-setup.sh`` and ``scripts/hpc-run.sh``. The remote GPU job is the
*only* step that touches the cluster:

    laptop                         HPC (Slurm)
    ------                         -----------
    config.yaml --- scp -->        outputs/<name>/config.yaml
                                   sbatch -> GPU node: trace + manifold
    outputs/<name>/ <-- rsync --   outputs/<name>/

After ``hpc run`` returns, the whole run lives in ``./outputs/<name>/`` and you
view it locally with no GPU and no tunnel.

ssh-quoting note: ssh joins everything after the host into ONE remote string and
runs it under the login shell. So a remote ``bash -lc <cmd>`` must be passed as a
single, fully-quoted argument (via :func:`shlex.quote`) — otherwise the login
shell runs ``bash -lc <first-word>`` with the rest as positional args (the bug
the old shell scripts warned about). Multi-line scripts go in over stdin with
``bash -l -s`` instead.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from llm_token_heatmap.commands._util import repo_root

# Defaults — every one overridable via its flag or the matching env var.
_SSH_HOST = os.environ.get("SSH_HOST", "j7zang-gpu")
_REMOTE_REPO = os.environ.get("REMOTE_REPO", "/work/j7zang/Token-Heatmap")
_REMOTE_VENV = os.environ.get("REMOTE_VENV", "/work/j7zang/th-gpu")
_REMOTE_BIN_GPU = os.environ.get("REMOTE_BIN_GPU", "/work/j7zang/th-gpu/bin/token-heatmap")
_ANACONDA_PYTHON = os.environ.get("ANACONDA_PYTHON", "/opt/uw/anaconda3/2025.06.1/bin/python3.13")
_POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "15"))


def _fail(msg: str) -> int:
    print(f"[hpc] error: {msg}", file=sys.stderr)
    return 2


def _ssh_capture(host: str, command: str, *, login: bool = True) -> str:
    """Run ``command`` on ``host`` and return its stdout (stderr discarded).

    ``login`` wraps it in a single quoted ``bash -lc`` so PATH-dependent tools
    (squeue/sacct) resolve; pass ``login=False`` for absolute-path commands.
    """
    remote = f"bash -lc {shlex.quote(command)}" if login else command
    try:
        res = subprocess.run(
            ["ssh", host, remote],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return res.stdout or ""


def _model_from_config(path: Path) -> str:
    """First ``model:`` value from a YAML config (no PyYAML dependency)."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*model\s*:\s*(.+)$", line)
            if match:
                return match.group(1).strip().strip("\"'").strip()
    except OSError:
        return ""
    return ""


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def _build_gpu_venv(
    host: str,
    remote_repo: str,
    remote_venv: str,
    anaconda_python: str,
    *,
    verify: bool,
) -> int:
    """Build (idempotently) the CUDA-12.4 GPU venv on the HPC; optionally verify."""
    print(f"[hpc-setup] host={host}  venv={remote_venv}  repo={remote_repo}")
    bin_path = f"{remote_venv}/bin/token-heatmap"
    script = f"""set -e
if [ ! -d {shlex.quote(remote_repo)} ]; then
  echo "[hpc-setup] ERROR: repo not found at {remote_repo}. Clone it there first." >&2
  exit 1
fi
if [ -x {shlex.quote(bin_path)} ]; then
  echo "[hpc-setup] GPU venv already present — skipping build."
else
  echo "[hpc-setup] creating venv with {anaconda_python}..."
  {shlex.quote(anaconda_python)} -m venv {shlex.quote(remote_venv)}
  source {shlex.quote(remote_venv + "/bin/activate")}
  pip install --upgrade pip
  echo "[hpc-setup] installing torch (cu124)..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  echo "[hpc-setup] installing token-heatmap (editable) + deps..."
  pip install -e {shlex.quote(remote_repo)}
  echo "[hpc-setup] built {remote_venv}"
fi
{shlex.quote(remote_venv + "/bin/python")} -c "import torch; print('[hpc-setup] torch', torch.__version__)"
"""
    rc = subprocess.run(["ssh", host, "bash", "-l", "-s"], input=script, text=True).returncode
    if rc != 0:
        return rc

    if verify:
        print("[hpc-setup] --verify: requesting a GPU node for a real CUDA check (may queue briefly)...")
        py = (
            'import torch; assert torch.cuda.is_available(), "CUDA not available"; '
            'x=torch.rand(1024,1024,device="cuda"); '
            'print("[hpc-setup] CUDA OK:", (x@x).sum().item() > 0)'
        )
        srun = (
            "srun --account=normal --qos=normal --gres=gpu:l40s:1 --mem=8G --time=00:05:00 "
            f"{shlex.quote(remote_venv + '/bin/python')} -c {shlex.quote(py)}"
        )
        rc = subprocess.run(["ssh", host, f"bash -lc {shlex.quote(srun)}"]).returncode
        if rc != 0:
            print(
                "[hpc-setup] GPU verification failed — the venv built but CUDA didn't light up. "
                "Check driver/torch versions.",
                file=sys.stderr,
            )
            return 1

    print("[hpc-setup] ✓ ready. Now run:  token-heatmap hpc run configs/wrap-text.yaml")
    return 0


def run_hpc_setup(args: argparse.Namespace) -> int:
    return _build_gpu_venv(
        args.ssh_host,
        args.remote_repo,
        args.remote_venv,
        args.anaconda_python,
        verify=args.verify,
    )


# --------------------------------------------------------------------------- #
# run  (the laptop -> HPC round-trip)
# --------------------------------------------------------------------------- #
def run_hpc_run(args: argparse.Namespace) -> int:
    host = args.ssh_host
    remote_repo = args.remote_repo
    remote_bin_gpu = args.remote_bin_gpu

    config_local = Path(args.config)
    if not config_local.is_file():
        return _fail(f"config file not found: {config_local}")

    gpu = args.gpu
    # rtx6000 cards here are RTX 6000 Ada (48 GB) on a 1 TB-RAM node; their qos
    # (qos_rtx6000_max) grants 200 G mem + 1-day walltime, far roomier than
    # qos=normal's 30 G / 12 h — so default rtx6000 runs onto it.
    qos = args.qos or ("qos_rtx6000_max" if gpu == "rtx6000" else "normal")
    mem = args.mem or ("64G" if qos == "qos_rtx6000_max" else "28G")

    name = args.name or config_local.stem
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return _fail(f"run name '{name}' has unsafe characters; pass --name with [A-Za-z0-9._-].")

    out_rel = f"outputs/{name}"
    remote_config = f"{out_rel}/config.yaml"
    local_out = repo_root() / out_rel

    extra = args.extra or ""
    if args.four_bit:
        extra = (extra + " --load-in-4bit").strip()
    manifold_extra = f"--components 6 --probe {args.probe}" if args.probe else ""

    # --- pre-flight VRAM heuristic ----------------------------------------- #
    effective_model = args.model or _model_from_config(config_local)
    if effective_model and not args.force:
        match = re.search(r"[0-9]+(?:\.[0-9]+)?[bB]", effective_model)
        if match:
            size_b = match.group(0)[:-1]
            int_part = int(size_b.split(".")[0])
            vram = 47 if gpu == "rtx6000" else 45  # both 48 GB cards
            bytes_per_param_x100 = 65 if args.four_bit else 200  # NF4 4-bit vs bf16
            est = int_part * bytes_per_param_x100 // 100  # integer GB; sub-1B floors to 0
            if est > vram * 92 // 100:
                if args.four_bit:
                    return _fail(
                        f"model '{effective_model}' (~{size_b}B) is ~{est} GB even in 4-bit — too big "
                        f"for one {gpu} (~{vram} GB). Pick a smaller --model or shard across GPUs. "
                        "(override: --force)"
                    )
                return _fail(
                    f"model '{effective_model}' (~{size_b}B) is ~{est} GB in bf16 — won't fit one "
                    f"{gpu} (~{vram} GB). Add --4bit (fits ~32B), pick a smaller --model, or override "
                    "with --force."
                )
            if est > vram * 80 // 100:
                print(
                    f"[hpc-run] ⚠ {effective_model} (~{est} GB) is close to the {gpu} limit "
                    f"(~{vram} GB) — may OOM under load."
                )

    print(f"[hpc-run] host={host}  name={name}  gpu={gpu}  qos={qos}  capture={args.capture}")
    print(
        f"[hpc-run] config(local)={config_local}  ->  {out_rel}  "
        f"(manifold={'on' if args.manifold else 'off'})"
    )

    # --- 0. one-time GPU env setup (optional) ------------------------------ #
    if args.setup:
        print("[hpc-run] --setup: ensuring the GPU venv exists on the HPC...")
        rc = _build_gpu_venv(host, remote_repo, args.remote_venv, args.anaconda_python, verify=False)
        if rc != 0:
            return rc

    # --- pre-flight: HPC reachable + GPU venv present ---------------------- #
    print("[hpc-run] checking the HPC...")
    rc = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", host, f"test -x {shlex.quote(remote_bin_gpu)}"]
    ).returncode
    if rc != 0:
        return _fail(
            f"GPU venv not found at {remote_bin_gpu} on {host}. Build it once: "
            f"token-heatmap hpc run {config_local} --setup  (or: token-heatmap hpc setup)."
        )

    # --- 1. ship the config into the run folder ---------------------------- #
    print(f"[hpc-run] [1/4] uploading config -> {host}:{remote_repo}/{remote_config}")
    rc = subprocess.run(
        ["ssh", host, f"mkdir -p {shlex.quote(remote_repo + '/' + out_rel)}"]
    ).returncode
    if rc != 0:
        return _fail("could not create the remote run folder.")
    rc = subprocess.run(
        ["scp", "-q", str(config_local), f"{host}:{remote_repo}/{remote_config}"]
    ).returncode
    if rc != 0:
        return _fail("scp of the config failed.")

    # --- 2. submit the GPU job (the only remote compute) ------------------- #
    print("[hpc-run] [2/4] submitting Slurm job (sbatch)...")
    sync_cmd = (
        f"git -C {shlex.quote(remote_repo)} pull --ff-only >&2 "
        "|| echo '[hpc-run] (remote git pull skipped/failed — using existing checkout)' >&2"
        if args.sync
        else ":"
    )
    env_assigns = [
        f"BIN={shlex.quote(remote_bin_gpu)}",
        f"CONFIG={shlex.quote(remote_config)}",
        f"OUT={shlex.quote(out_rel)}",
        f"CAPTURE={shlex.quote(args.capture)}",
        f"EXTRA={shlex.quote(extra)}",
    ]
    if args.model:
        env_assigns.append(f"MODEL={shlex.quote(args.model)}")
    if args.manifold:
        env_assigns.append(f"MANIFOLD_EXTRA={shlex.quote(manifold_extra)}")
    submit_script = f"""set -e
{sync_cmd}
cd {shlex.quote(remote_repo)}
{" ".join(env_assigns)} \\
sbatch --parsable \\
  --job-name={shlex.quote("th-" + name)} \\
  --qos={shlex.quote(qos)} --gres={shlex.quote("gpu:" + gpu + ":1")} \\
  --mem={shlex.quote(mem)} --time={shlex.quote(args.time)} \\
  --output={shlex.quote(out_rel + "/slurm-%j.log")} \\
  --export=ALL,BIN,CONFIG,OUT,CAPTURE,EXTRA,MODEL,MANIFOLD_EXTRA \\
  scripts/hpc-gen.slurm
"""
    res = subprocess.run(
        ["ssh", host, "bash", "-l", "-s"], input=submit_script, text=True, stdout=subprocess.PIPE
    )
    job_id = (res.stdout or "").strip()
    if not re.fullmatch(r"[0-9]+", job_id):
        return _fail(
            f"did not get a numeric Slurm job id back (got: '{job_id}'). Check the HPC repo / venv."
        )
    remote_log = f"{out_rel}/slurm-{job_id}.log"
    print(f"[hpc-run] submitted job {job_id}  (log: {remote_repo}/{remote_log})")

    # --- 3. wait for it (poll squeue; show the latest log line) ------------ #
    print(
        f"[hpc-run] [3/4] waiting for the GPU job to finish "
        f"(poll every {args.poll_seconds}s; Ctrl+C is safe — the job keeps running)..."
    )
    last_state = ""
    try:
        while True:
            out = _ssh_capture(host, f"squeue -h -j {job_id} -o %T 2>/dev/null")
            state = out.splitlines()[0].strip() if out.strip() else ""
            if not state:
                break  # gone from the queue -> finished
            if state != last_state:
                print(f"[hpc-run]   state: {state}")
                last_state = state
            if state == "RUNNING":
                tail = _ssh_capture(
                    host,
                    f"tail -n 1 {shlex.quote(remote_repo + '/' + remote_log)} 2>/dev/null",
                    login=False,
                )
                if tail.strip():
                    print(f"[hpc-run]   · {tail.strip()}")
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print(
            f"\n[hpc-run] interrupted — the Slurm job {job_id} keeps running. "
            f"Re-run `token-heatmap hpc run` once it finishes to pull the results."
        )
        return 130

    out = _ssh_capture(host, f"sacct -j {job_id} -n -P -o State 2>/dev/null")
    final_state = out.splitlines()[0].strip() if out.strip() else "unknown"
    print(f"[hpc-run] job {job_id} finished: {final_state}")
    print("[hpc-run] --- remote log tail -------------------------------------------")
    subprocess.run(["ssh", host, f"tail -n 12 {shlex.quote(remote_repo + '/' + remote_log)} 2>/dev/null"])
    print("[hpc-run] ---------------------------------------------------------------")
    if not final_state.startswith("COMPLETED"):
        return _fail(
            f"remote job did not complete cleanly (state: {final_state}). Outputs left on the HPC; "
            f"inspect {remote_repo}/{remote_log}."
        )

    # --- 4. pull EVERYTHING back ------------------------------------------- #
    if not args.pull:
        print(f"[hpc-run] --no-pull: leaving results on the HPC at {remote_repo}/{out_rel}")
        print(
            f"[hpc-run] rsync them back when ready: "
            f"rsync -az {host}:{remote_repo}/{out_rel}/ ./{out_rel}/"
        )
        return 0

    print(f"[hpc-run] [4/4] pulling results -> {local_out}/ ...")
    local_out.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        # --stats (not --info=progress2): macOS ships rsync 2.6.9; --stats is
        # the portable end-of-transfer summary supported by old and new rsync.
        rc = subprocess.run(
            ["rsync", "-az", "--stats", "-e", "ssh",
             f"{host}:{remote_repo}/{out_rel}/", f"{local_out}/"]
        ).returncode
    else:
        rc = subprocess.run(
            ["scp", "-q", "-r", f"{host}:{remote_repo}/{out_rel}/.", f"{local_out}/"]
        ).returncode
    if rc != 0:
        return _fail("pulling results back failed.")

    print(f"[hpc-run] ✓ done — everything is local now: {out_rel}/")
    for entry in sorted(p.name for p in local_out.iterdir()):
        print(f"[hpc-run]     {entry}")

    # --- view locally (no GPU) --------------------------------------------- #
    trace_json = f"{out_rel}/adaptive_token_trace.json"
    print()
    print(f"[hpc-run] view it locally (no GPU needed): open {trace_json} in the viewer —")
    print("[hpc-run] drag it onto the web app (cd app && npm run dev), or open")
    print("[hpc-run] it in the desktop app.")
    return 0


# --------------------------------------------------------------------------- #
# parser registration
# --------------------------------------------------------------------------- #
def _add_host_repo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ssh-host", default=_SSH_HOST, help=f"SSH host alias (default: {_SSH_HOST}).")
    parser.add_argument(
        "--remote-repo", default=_REMOTE_REPO, help=f"Repo checkout on the HPC (default: {_REMOTE_REPO})."
    )


def _add_venv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-venv", default=_REMOTE_VENV, help=f"GPU venv path (default: {_REMOTE_VENV}).")
    parser.add_argument(
        "--anaconda-python",
        default=_ANACONDA_PYTHON,
        help="Base interpreter used to create the venv.",
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    hpc = subparsers.add_parser(
        "hpc",
        help="Run GPU compute on an HPC (Slurm) and pull results back.",
        description="Laptop-side HPC orchestration: setup the GPU venv or run a round-trip.",
    )
    hpc_sub = hpc.add_subparsers(dest="hpc_command", required=True)

    # --- hpc setup --------------------------------------------------------- #
    setup = hpc_sub.add_parser(
        "setup",
        help="One-time: build/verify the CUDA-12.4 GPU venv on the HPC.",
        description=(
            "Build (idempotently) the dedicated cu124 torch venv on the HPC so "
            "token-heatmap runs on the GPU instead of silently falling back to CPU."
        ),
    )
    setup.add_argument(
        "--verify", action="store_true", help="Also run a real GPU matmul check (queues a short srun)."
    )
    _add_host_repo(setup)
    _add_venv_args(setup)
    setup.set_defaults(func=run_hpc_setup)

    # --- hpc run ----------------------------------------------------------- #
    run = hpc_sub.add_parser(
        "run",
        help="Laptop->HPC round-trip: submit a Slurm trace+manifold job and rsync it back.",
        description=(
            "Upload a config, submit a Slurm GPU job (trace + manifold) — the only "
            "remote step — poll it, then rsync the whole outputs/<name>/ folder back "
            "so viewing needs no GPU and no tunnel."
        ),
    )
    run.add_argument("config", help="Trace config YAML (its basename is the default run name).")
    run.add_argument("--name", help="Run name -> outputs/NAME locally + on the HPC (default: config basename).")
    run.add_argument("--model", help="Override the model id (e.g. Qwen/Qwen2.5-14B-Instruct).")
    run.add_argument(
        "--gpu", choices=("rtx6000", "l40s"), default="rtx6000",
        help="GPU type; both 48 GB (default: rtx6000 -> qos_rtx6000_max).",
    )
    run.add_argument("--qos", help="Slurm qos (default: qos_rtx6000_max for rtx6000, normal for l40s).")
    run.add_argument("--mem", help="Host memory (default: 64G under qos_rtx6000_max, else 28G).")
    run.add_argument("--time", default="01:00:00", help="Walltime HH:MM:SS (default: 01:00:00).")
    run.add_argument(
        "--capture", choices=("full", "activations"), default="full",
        help="full = +attention (slower); activations = manifold-only (default: full).",
    )
    run.add_argument("--probe", help="Add a supervised manifold probe scalar (e.g. line_position).")
    run.add_argument("--extra", help="Extra `trace` flags (e.g. '--max-new-tokens 320').")
    run.add_argument("--4bit", dest="four_bit", action="store_true", help="Load in 4-bit NF4 (for 32B+).")
    run.add_argument("--no-manifold", dest="manifold", action="store_false", help="Skip the manifold pass.")
    run.add_argument("--no-sync", dest="sync", action="store_false", help="Don't `git pull` the HPC repo first.")
    run.add_argument("--no-pull", dest="pull", action="store_false", help="Leave outputs on the HPC (no rsync back).")
    run.add_argument("--setup", action="store_true", help="Build/verify the GPU venv on the HPC first (one-time).")
    run.add_argument("--force", action="store_true", help="Skip the pre-flight 'won't fit in VRAM' size check.")
    _add_host_repo(run)
    _add_venv_args(run)
    run.add_argument("--remote-bin-gpu", default=_REMOTE_BIN_GPU, help="token-heatmap in the GPU venv on the HPC.")
    run.add_argument("--poll-seconds", type=int, default=_POLL_SECONDS, help="squeue poll interval (default: 15).")
    run.set_defaults(func=run_hpc_run)
