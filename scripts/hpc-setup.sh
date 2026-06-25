#!/usr/bin/env bash
# One-time, run from the LAPTOP: build the dedicated CUDA-12.4 GPU venv on the
# HPC so token-heatmap actually runs on the GPU. Idempotent — safe to re-run;
# it no-ops if the venv already works.
#
# Why a dedicated venv: the cluster's default torch is cu130, but the GPU
# drivers are CUDA 12.4, so the default install silently falls back to CPU.
# This venv (`/work/j7zang/th-gpu`, torch cu124) is isolated so it can't disturb
# your other research environments.
#
#   ./scripts/hpc-setup.sh            # build/verify the venv
#   ./scripts/hpc-setup.sh --verify   # also run a real GPU matmul check (uses a GPU node)
#
# Env overrides: SSH_HOST, REMOTE_REPO, REMOTE_VENV, ANACONDA_PYTHON.

set -euo pipefail

SSH_HOST="${SSH_HOST:-j7zang-gpu}"
REMOTE_REPO="${REMOTE_REPO:-/work/j7zang/Token-Heatmap}"
REMOTE_VENV="${REMOTE_VENV:-/work/j7zang/th-gpu}"
ANACONDA_PYTHON="${ANACONDA_PYTHON:-/opt/uw/anaconda3/2025.06.1/bin/python3.13}"

VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

echo "[hpc-setup] host=${SSH_HOST}  venv=${REMOTE_VENV}  repo=${REMOTE_REPO}"

# Build the venv if its CLI isn't already there. The heredoc expands local
# vars before sending; bash -l for a login shell.
ssh "$SSH_HOST" bash -l -s <<EOF
set -e
if [ ! -d '${REMOTE_REPO}' ]; then
  echo "[hpc-setup] ERROR: repo not found at ${REMOTE_REPO}. Clone it there first." >&2
  exit 1
fi
if [ -x '${REMOTE_VENV}/bin/token-heatmap' ]; then
  echo "[hpc-setup] GPU venv already present — skipping build."
else
  echo "[hpc-setup] creating venv with ${ANACONDA_PYTHON}..."
  '${ANACONDA_PYTHON}' -m venv '${REMOTE_VENV}'
  source '${REMOTE_VENV}/bin/activate'
  pip install --upgrade pip
  echo "[hpc-setup] installing torch (cu124)..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  echo "[hpc-setup] installing token-heatmap (editable) + deps..."
  pip install -e '${REMOTE_REPO}'
  echo "[hpc-setup] built ${REMOTE_VENV}"
fi
'${REMOTE_VENV}/bin/python' -c "import torch; print('[hpc-setup] torch', torch.__version__)"
EOF

if [[ "$VERIFY" == 1 ]]; then
  echo "[hpc-setup] --verify: requesting a GPU node for a real CUDA check (may queue briefly)..."
  ssh "$SSH_HOST" bash -lc "srun --account=normal --qos=normal --gres=gpu:l40s:1 --mem=8G --time=00:05:00 \
    '${REMOTE_VENV}/bin/python' -c 'import torch; assert torch.cuda.is_available(), \"CUDA not available\"; x=torch.rand(1024,1024,device=\"cuda\"); print(\"[hpc-setup] CUDA OK:\", (x@x).sum().item() > 0)'" \
    || { echo "[hpc-setup] GPU verification failed — the venv built but CUDA didn't light up. Check driver/torch versions." >&2; exit 1; }
fi

echo "[hpc-setup] ✓ ready. Now run:  ./scripts/hpc-run.sh configs/wrap-text.yaml"
