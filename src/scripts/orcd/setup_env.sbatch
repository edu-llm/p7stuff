#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J edullm-env
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail
umask 077

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
: "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"

mkdir -p "$EDULLM_SCRATCH/logs"
exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1
module load miniforge/24.3.0-0
test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
python -m venv "$EDULLM_VENV"
source "$EDULLM_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools uv
uv pip install torch --torch-backend=cu128
uv pip install "${EDULLM_REPO_ROOT}[wandb]"
python - <<'PY'
import platform

import edullm.ssh_helper
import torch
import wandb

from olmo_core.version import VERSION

print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print(f"W&B: {wandb.__version__}")
print(f"OLMo-core: {VERSION}")
PY
printf '%s\n' "$EDULLM_COMMIT_SHA" > "$EDULLM_VENV/.edullm-commit"
chmod 600 "$EDULLM_VENV/.edullm-commit"
