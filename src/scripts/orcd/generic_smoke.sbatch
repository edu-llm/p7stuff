#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -t 00:45:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -J edullm-smoke
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
: "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"
WANDB_ENV="${WANDB_ENV:-$HOME/.config/edullm/wandb.env}"

mkdir -p "$EDULLM_SCRATCH/logs"
exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1
test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
source "$EDULLM_VENV/bin/activate"
source "$WANDB_ENV"
export EDULLM_SCRATCH

EDULLM_DATA_MODE="${EDULLM_DATA_MODE:-synthetic}"
if [[ "$EDULLM_DATA_MODE" == "synthetic" ]]; then
  python "$EDULLM_REPO_ROOT/src/scripts/orcd/create_tiny_data.py" \
    --output "$EDULLM_SCRATCH/data/generic-smoke"
  export OLMO_DATA_ROOT="$EDULLM_SCRATCH/data/generic-smoke"
elif [[ "$EDULLM_DATA_MODE" == "staged" ]]; then
  : "${OLMO_DATA_ROOT:?staged mode requires OLMO_DATA_ROOT}"
  test -s "$OLMO_DATA_ROOT/c4-train.00000-00099.npy"
  test -s "$OLMO_DATA_ROOT/c4-validation.00000-00008.npy"
  export OLMO_DATA_ROOT
else
  echo "unknown EDULLM_DATA_MODE: $EDULLM_DATA_MODE" >&2
  exit 2
fi
bash "$EDULLM_REPO_ROOT/src/scripts/orcd/run_generic_smoke.sh"
