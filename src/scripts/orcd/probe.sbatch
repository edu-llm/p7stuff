#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -t 00:15:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -J edullm-probe
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
: "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"

mkdir -p "$EDULLM_SCRATCH/logs"
exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1
source "$EDULLM_VENV/bin/activate"
export PYTHONPATH="$EDULLM_REPO_ROOT/src"
test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
nvidia-smi
python "$EDULLM_REPO_ROOT/src/scripts/orcd/probe.py" \
  --scratch "$EDULLM_SCRATCH" \
  --output "$EDULLM_SCRATCH/probes/${SLURM_JOB_ID}.json"
