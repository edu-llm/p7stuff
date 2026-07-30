#!/bin/bash
set -euo pipefail

: "${EDULLM_REPO_ROOT:?missing EDULLM_REPO_ROOT}"
: "${EDULLM_SCRATCH:?missing EDULLM_SCRATCH}"
: "${EDULLM_COMMIT_SHA:?missing EDULLM_COMMIT_SHA}"
: "${WANDB_API_KEY:?missing WANDB_API_KEY}"

RUN_NAME="${RUN_NAME:-orcd-smoke-${SLURM_JOB_ID:-manual}}"
HARD_STOP_STEPS="${HARD_STOP_STEPS:-20}"
DATA_ROOT="${OLMO_DATA_ROOT:-$EDULLM_SCRATCH/data/generic-smoke}"
SAVE_FOLDER="${SAVE_FOLDER:-$EDULLM_SCRATCH/runs/$RUN_NAME}"

export OLMO_DATA_ROOT="$DATA_ROOT"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
export WANDB_RESUME=allow
if ! python -c 'import requests; raise SystemExit(0 if requests.get("https://api.wandb.ai", timeout=5).status_code < 500 else 1)'
then
  export WANDB_MODE=offline
fi
export PYTHONPATH="$EDULLM_REPO_ROOT/src"

test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
python -c 'import olmo_core, os; assert os.path.realpath(olmo_core.__file__).startswith(os.path.realpath(os.environ["EDULLM_REPO_ROOT"]))'

torchrun --standalone --nproc-per-node=1 \
  "$EDULLM_REPO_ROOT/src/examples/llm/train.py" "$RUN_NAME" \
  --model-factory=olmo2_190M \
  --sequence-length=512 \
  --save-folder="$SAVE_FOLDER" \
  --work-dir="$SAVE_FOLDER" \
  --data_loader.global_batch_size=8192 \
  --train_module.rank_microbatch_size=2048 \
  --train_module.max_sequence_length=512 \
  --trainer.hard_stop="{value: $HARD_STOP_STEPS, unit: steps}" \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.callbacks.checkpointer.save_interval=10 \
  --trainer.callbacks.checkpointer.ephemeral_save_interval=null \
  --trainer.callbacks.wandb.enabled=true \
  --trainer.callbacks.wandb.entity="$WANDB_ENTITY" \
  --trainer.callbacks.wandb.project="$WANDB_PROJECT" \
  --trainer.callbacks.wandb.group="$WANDB_GROUP" \
  --trainer.callbacks.wandb.tags="[orcd,generic-smoke,olmo2-190m]"
