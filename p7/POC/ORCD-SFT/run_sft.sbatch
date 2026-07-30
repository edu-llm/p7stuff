#!/bin/bash
#SBATCH -p mit_normal_gpu           # free base-tier GPU partition (L40S/H100/H200)
#SBATCH -G l40s:1                   # 1 L40S (44GB) fits a 1B LoRA easily; free tier allows up to 2
#SBATCH -c 16                       # <=16 CPUs per L40S GPU on mit_normal_gpu
#SBATCH --mem=64G
#SBATCH -t 04:00:00                 # well under the 6h partition limit
#SBATCH -J olmo2_sft_instruct
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ---- SFT OLMo-2-1B-Instruct into a Socratic math tutor (training only for now) ----
# Submit from the ORCD-SFT/ project dir:  sbatch run_sft.sbatch
# Monitor:  squeue -u $USER    tail -f logs/olmo2_sft_instruct_<jobid>.out
#
# GPU / memory sizing for the ORCD free tier:
#   - mit_normal_gpu base tier = up to 2 GPUs, 6h. A 1B LoRA needs only 1 L40S.
#   - Gradient checkpointing is ON with per_device_batch=8, grad_accum=4 (effective 32),
#     which fits comfortably in ~44GB. If you ever hit OOM, drop --per_device_batch to 4
#     and raise --grad_accum to 8 (same effective batch).

set -euo pipefail
mkdir -p logs

START_FROM="${START_FROM:-instruct}"

echo "Job $SLURM_JOB_ID on $(hostname) | start_from=$START_FROM"
nvidia-smi || true

# --- environment ---
# Adjust this to wherever you installed Miniforge (see setup_orcd_env.sh).
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate socrateach

# Cache HF models/datasets on large shared storage, not your small home quota.
# Point HF_HOME at your /orcd path before submitting, e.g.:
#   export HF_HOME=/orcd/pool/<yourpath>/hf_cache
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF_HOME=$HF_HOME"

# --- train ---
python train_sft.py \
    --start_from "$START_FROM" \
    --data_dir data \
    --per_device_batch 8 \
    --grad_accum 4 \
    --resume auto

echo "Done. Output: olmo2-1b-socratic-tutor-$START_FROM/"

# --- (optional, later) generate the 2x2 test results for eval ---
# Evals are out of scope for now. When you're ready to produce model outputs for
# scoring, run this separately (script is still in this folder):
#   python generate_test_results.py --start_from "$START_FROM" --data_dir data --n_eval_dialogues 50
