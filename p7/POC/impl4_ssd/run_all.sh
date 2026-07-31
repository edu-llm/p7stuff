#!/bin/bash
# Drive the whole Impl 4 run matrix (PLAN §9). Eight runs, ~40 min each on the L40S
# (~5.5 GPU-hours total). Only the replay slot differs; everything else is Impl 2.
#
#     ./run_all.sh                     # all eight arms, submitted to Slurm
#     ./run_all.sh --cut four          # PLAN §12 cut 1: A1 A2 A3 T4
#     ./run_all.sh --cut one           # PLAN §12 cut 2: A3 only
#     ./run_all.sh --local             # run sequentially here instead of sbatch
#     ./run_all.sh --poc --local       # 63-block smoke rehearsal of the full pipeline
#
# A1 must go first: it defines the token budget every other arm is matched to
# (data/tulu_reference.json), so its data stage is always run synchronously and the
# rest are only submitted after it lands.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

ARMS=(A1 A2 A3 A4 T2 T3 T4 B2)
LOCAL=0
POC=""
TOKEN_REFERENCE="${TOKEN_REFERENCE:-a1}"
MIN_GOLD_WORDS="${MIN_GOLD_WORDS:-}"
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --token_reference) TOKEN_REFERENCE="$2"; shift 2 ;;
        --min_gold_words)  MIN_GOLD_WORDS="$2";  shift 2 ;;
        --cut)
            case "$2" in
                four) ARMS=(A1 A2 A3 T4) ;;
                one)  ARMS=(A3) ;;
                *) echo "unknown cut '$2' (four|one)" >&2; exit 1 ;;
            esac
            shift 2 ;;
        --arms) IFS=' ' read -r -a ARMS <<< "$2"; shift 2 ;;
        --local) LOCAL=1; shift ;;
        --poc)   POC="1"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

POC_FLAG=""
[ -n "$POC" ] && POC_FLAG="--poc"
SLOT_ARGS=(--token_reference "$TOKEN_REFERENCE")

echo "Arms: ${ARMS[*]} | local=$LOCAL | poc=${POC:-0} | token_reference=$TOKEN_REFERENCE"

# --- shared, built once ---------------------------------------------------
python build_pedagogy_pool.py
if [ "${#ARMS[@]}" -gt 1 ] || [ "${ARMS[0]}" != "A1" ]; then
    python build_prompt_pool.py ${SUPERNI_DIR:+--superni_dir "$SUPERNI_DIR"} \
        ${MIN_GOLD_WORDS:+--min_gold_words "$MIN_GOLD_WORDS"} "${EXTRA[@]}"
fi

# --- the token references must exist before any arm that matches to them ---
# A1 always (it is the Tulu budget); A2 as well under --token_reference superni_gold,
# since the SSD arms are then matched to *its* realized total.
if [ ! -f data/tulu_reference.json ]; then
    echo "== building A1's Tulu reference slot (defines the token budget) =="
    python build_general_slot.py --arm A1 $POC_FLAG
fi
if [ "$TOKEN_REFERENCE" = "superni_gold" ] && [ ! -f data/superni_gold_reference.json ]; then
    echo "== building A2's SuperNI-gold reference slot =="
    python build_general_slot.py --arm A2 "${SLOT_ARGS[@]}" $POC_FLAG
fi

for arm in "${ARMS[@]}"; do
    echo "=============================== $arm ==============================="
    if [ "$LOCAL" = "1" ]; then
        python build_general_slot.py --arm "$arm" "${SLOT_ARGS[@]}" $POC_FLAG
        python mix_and_order.py --arm "$arm" $POC_FLAG
        python acceptance_checks.py --arm "$arm"
        mkdir -p "runs/$arm"
        python train_sft_impl4.py --arm "$arm" $POC_FLAG --resume auto \
            2>&1 | tee -a "runs/$arm/train.log"
    else
        ARM="$arm" POC="${POC:-0}" TOKEN_REFERENCE="$TOKEN_REFERENCE" \
            MIN_GOLD_WORDS="$MIN_GOLD_WORDS" \
            sbatch --job-name "impl4_$arm" run_arm.sbatch
    fi
done

# --- shared deliverables (PLAN §10) ---------------------------------------
python build_prompt_pool.py --split test ${SUPERNI_DIR:+--superni_dir "$SUPERNI_DIR"} || \
    echo "NOTE: held-out prompt build failed; it is optional (PLAN §10, shipped unused)."

echo
echo "Done. Per-arm deliverables in runs/<arm>/; shared files in shared/."
