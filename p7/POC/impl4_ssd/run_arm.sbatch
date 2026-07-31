#!/bin/bash
#SBATCH -p mit_normal_gpu           # free base-tier GPU partition (L40S/H100/H200)
#SBATCH -G l40s:1                   # 1 L40S (44GB) fits a 1B LoRA easily
#SBATCH -c 16                       # <=16 CPUs per L40S GPU on mit_normal_gpu
#SBATCH --mem=64G
#SBATCH -t 04:00:00                 # well under the 6h partition limit
#SBATCH -J impl4_arm
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Impl 4 — build one arm's replay slot, mix + order it, check it, train it.
# Parameterized copy of ORCD-SFT/run_sft.sbatch (PLAN §8.5).
#
# Submit from impl4_ssd/:
#     ARM=A1 sbatch run_arm.sbatch
#     ARM=A3 sbatch --job-name impl4_A3 run_arm.sbatch
#     ARM=A3 STAGE=train sbatch run_arm.sbatch          # skip data build, just train
#     ARM=A3 POC=1 sbatch run_arm.sbatch                # smoke run
#
# Environment knobs:
#     ARM       required, one of A1 A2 A3 A4 T2 T3 T4 B2 (T1 is an alias of A3)
#     STAGE     all (default) | data | train
#     POC       1 to run the 63-block smoke version
#     BACKEND   auto (default) | vllm | hf
#     MIN_GOLD_WORDS  PLAN §3 filter 3; run `build_prompt_pool.py --scan_only` first
#     TOKEN_REFERENCE a1 (default) | superni_gold — see build_general_slot.py
#     SUPERNI_DIR  local clone of allenai/natural-instructions (recommended)
#     HF_HOME   point at /orcd storage, not your home quota

set -euo pipefail
mkdir -p logs

ARM="${ARM:?set ARM=A1|A2|A3|A4|T2|T3|T4|B2}"
STAGE="${STAGE:-all}"
BACKEND="${BACKEND:-auto}"
TOKEN_REFERENCE="${TOKEN_REFERENCE:-a1}"
POC_FLAG=""
[ "${POC:-0}" = "1" ] && POC_FLAG="--poc"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | arm=$ARM stage=$STAGE poc=${POC:-0}"
nvidia-smi || true

# --- environment ---
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate socrateach
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF_HOME=$HF_HOME"

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
RUN_DIR="runs/$ARM"
mkdir -p "$RUN_DIR"

POOL_ARGS=()
[ -n "${SUPERNI_DIR:-}" ] && POOL_ARGS+=(--superni_dir "$SUPERNI_DIR")
[ -n "${MIN_GOLD_WORDS:-}" ] && POOL_ARGS+=(--min_gold_words "$MIN_GOLD_WORDS")

if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
    # Shared, built once and reused by every arm. Both are no-ops if already present.
    python build_pedagogy_pool.py
    if [ "$ARM" != "A1" ]; then
        python build_prompt_pool.py "${POOL_ARGS[@]}"
    fi

    python build_general_slot.py --arm "$ARM" --backend "$BACKEND" \
        --token_reference "$TOKEN_REFERENCE" $POC_FLAG
    python mix_and_order.py --arm "$ARM" $POC_FLAG
    python acceptance_checks.py --arm "$ARM"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
    python train_sft_impl4.py --arm "$ARM" $POC_FLAG --resume auto \
        2>&1 | tee -a "$RUN_DIR/train.log"
fi

echo "Done. Deliverables in $RUN_DIR/"
ls -la "$RUN_DIR" | head -30
