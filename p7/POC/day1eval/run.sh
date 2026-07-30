#!/usr/bin/env bash
# Convenience launcher for the day1 tutor-generation eval.
#
#   ./run.sh                 # OLMo-2 base over MRBench V1 (all dialogues)
#   ./run.sh qwen            # Qwen3-1.7B over V1
#   ./run.sh olmo-instruct V1 20   # model, dataset, limit
#
# Any extra args after the first three pass straight through to generate.py:
#   ./run.sh qwen V1 0 --thinking --temperature 0.6
set -euo pipefail

MODEL="${1:-olmo}"
DATASET="${2:-V1}"
LIMIT="${3:-0}"
shift $(( $# < 3 ? $# : 3 )) || true

cd "$(dirname "$0")"

# Faster HF downloads if hf_transfer is installed.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

python generate.py --model "$MODEL" --dataset "$DATASET" --limit "$LIMIT" "$@"
