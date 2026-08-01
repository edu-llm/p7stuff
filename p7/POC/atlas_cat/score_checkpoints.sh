#!/usr/bin/env bash
# Score a POC run's checkpoint grid with the ATLAS CAT diagnostic.
#
# One CAT per checkpoint, sequentially. This runs *alongside* the existing evals
# (math_eval/, general_eval/, llm_judge/, day1eval/) -- it replaces none of them. What it
# adds is an ability estimate with an error bar, cheap enough to run unattended on every
# point of a checkpoint grid. It says nothing about pedagogy, formatting, or
# instruction-following; README.md has the per-eval breakdown.
#
#   ./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 --s3-out s3://BUCKET/atlas_cat
#   ./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 --s3-out s3://BUCKET/atlas_cat --priority
#   ./score_checkpoints.sh --run-dir ../impl4_ssd/runs/A3 --s3-out s3://BUCKET/atlas_cat --steps 20,160,937
#
# Checkpoints are LoRA adapters, so each one is merged onto the base model before
# serving; the merged copy is deleted after its CAT. Sequential on purpose -- see
# CONVERSION.md on disk use.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR=""
S3_OUT=""
BASE_MODEL=""
STEPS=""
ARM_NAME=""
RUNNER=""
PRIORITY=0
FORCE=0
DRY_RUN=0
EXTRA=()

usage() {
    cat <<'EOF'
Score a POC checkpoint grid with the ATLAS CAT diagnostic.

Required:
  --run-dir PATH     a run dir holding ckpt-<step>/ adapters (e.g. impl4_ssd/runs/A3)
  --s3-out s3://B/P  results root; each checkpoint lands at <s3-out>/<arm>-step<N>/

Optional:
  --priority         only the manifest's priority_checkpoints (Block S: all 11;
                     Blocks T/G: {20,160,937}) instead of every ckpt-* on disk
  --steps 20,160     explicit comma-separated step list (overrides --priority)
  --base-model ID    base the adapters merge onto (default: from manifest.json,
                     else adapter_config.json)
  --arm-name NAME    run-id prefix (default: manifest.json arm, else dir name)
  --runner PATH      override the skill's run_cat_diagnostic.sh
  --force            re-score checkpoints that already have a _READY in S3
  --dry-run          print what would run, touch nothing
  --                 everything after this is passed through to run_cat_diagnostic.sh
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --run-dir)    RUN_DIR="$2"; shift 2 ;;
        --s3-out)     S3_OUT="${2%/}"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --arm-name)   ARM_NAME="$2"; shift 2 ;;
        --runner)     RUNNER="$2"; shift 2 ;;
        --priority)   PRIORITY=1; shift ;;
        --force)      FORCE=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$RUN_DIR" ] || { usage >&2; die "--run-dir is required"; }
[ -n "$S3_OUT" ] || { usage >&2; die "--s3-out is required"; }
[ -d "$RUN_DIR" ] || die "no such run dir: $RUN_DIR"

# --- locate the skill runner ------------------------------------------------
if [ -z "$RUNNER" ]; then
    REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)"
    for cand in \
        "${REPO_ROOT:-/nonexistent}/.claude/skills/add-cat-evals/scripts/run_cat_diagnostic.sh" \
        "${REPO_ROOT:-/nonexistent}/.cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh" \
        "$HERE/../../../.claude/skills/add-cat-evals/scripts/run_cat_diagnostic.sh"
    do
        if [ -f "$cand" ]; then RUNNER="$cand"; break; fi
    done
    [ -n "$RUNNER" ] || die "cannot find run_cat_diagnostic.sh; pass --runner"
fi

# --- what the run says about itself -----------------------------------------
MANIFEST="$RUN_DIR/manifest.json"
INDEX="$RUN_DIR/checkpoint_index.json"

read_json_field() {   # file, key -> value or empty
    [ -f "$1" ] || return 0
    python3 - "$1" "$2" <<'PY' || true
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
v = doc.get(sys.argv[2])
if isinstance(v, (list, tuple)):
    print(",".join(str(x) for x in v))
elif v is not None:
    print(v)
PY
}

[ -n "$ARM_NAME" ]   || ARM_NAME="$(read_json_field "$MANIFEST" arm)"
[ -n "$ARM_NAME" ]   || ARM_NAME="$(basename "$(cd "$RUN_DIR" && pwd)")"
[ -n "$BASE_MODEL" ] || BASE_MODEL="$(read_json_field "$MANIFEST" base_model)"

# --- which steps ------------------------------------------------------------
on_disk="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'ckpt-*' 2>/dev/null \
           | sed 's#.*/ckpt-##' | sort -n | tr '\n' ' ')"
[ -n "${on_disk// /}" ] || die "no ckpt-*/ dirs under $RUN_DIR — has this arm trained yet?"

if [ -n "$STEPS" ]; then
    want="${STEPS//,/ }"
elif [ "$PRIORITY" = "1" ]; then
    prio="$(read_json_field "$INDEX" priority_checkpoints)"
    [ -n "$prio" ] || die "--priority given but $INDEX has no priority_checkpoints"
    want="${prio//,/ }"
else
    want="$on_disk"
fi

selected=()
skipped_absent=()
for s in $want; do
    if [ -d "$RUN_DIR/ckpt-$s" ]; then
        selected+=("$s")
    else
        skipped_absent+=("$s")
    fi
done
[ ${#selected[@]} -gt 0 ] || die "none of the requested steps ($want) exist under $RUN_DIR"

echo "arm         : $ARM_NAME"
echo "run dir     : $RUN_DIR"
echo "base model  : ${BASE_MODEL:-(from adapter_config.json)}"
echo "runner      : $RUNNER"
echo "on disk     : $on_disk"
echo "scoring     : ${selected[*]}"
[ ${#skipped_absent[@]} -eq 0 ] || echo "not on disk  : ${skipped_absent[*]} (skipped)"
echo "destination : $S3_OUT/$ARM_NAME-step<N>/"
echo

# --- sweep ------------------------------------------------------------------
done_steps=()
failed_steps=()
skipped_steps=()

for s in "${selected[@]}"; do
    run_id="$ARM_NAME-step$s"
    ckpt="$RUN_DIR/ckpt-$s"

    if [ "$FORCE" = "0" ] && [ "$DRY_RUN" = "0" ] && command -v aws >/dev/null 2>&1 \
       && aws s3 ls "$S3_OUT/$run_id/_READY" >/dev/null 2>&1; then
        echo "== $run_id: already has _READY in S3, skipping (--force to redo)"
        skipped_steps+=("$s")
        continue
    fi

    cmd=(bash "$RUNNER" --checkpoint "$ckpt" --run-id "$run_id" --s3-out "$S3_OUT")
    [ -n "$BASE_MODEL" ] && cmd+=(--base-model "$BASE_MODEL")
    [ "$DRY_RUN" = "1" ] && cmd+=(--dry-run)
    [ ${#EXTRA[@]} -eq 0 ] || cmd+=("${EXTRA[@]}")

    echo "== $run_id"
    if "${cmd[@]}"; then
        done_steps+=("$s")
    else
        # A single bad checkpoint should not abandon the rest of an 11-point grid.
        echo "   FAILED (continuing) — its worker.log is in S3 without a _READY marker" >&2
        failed_steps+=("$s")
    fi
    echo
done

# --- report -----------------------------------------------------------------
echo "----------------------------------------------------------------------"
echo "$ARM_NAME: ${#done_steps[@]} scored, ${#failed_steps[@]} failed, ${#skipped_steps[@]} already done"
[ ${#done_steps[@]} -eq 0 ]    || echo "  scored : ${done_steps[*]}"
[ ${#failed_steps[@]} -eq 0 ]  || echo "  failed : ${failed_steps[*]}"
[ ${#skipped_steps[@]} -eq 0 ] || echo "  skipped: ${skipped_steps[*]}"
[ ${#failed_steps[@]} -eq 0 ] || exit 1
