#!/usr/bin/env bash
# ATLAS CAT (Computerized Adaptive Testing) checkpoint diagnostic.
#
#   checkpoint -> HF format -> vLLM -> olmo-eval atlas_arc -> S3
#
# Designed to be called from a training script right after a checkpoint is saved,
# usually backgrounded so it does not block training. Results land at
# <s3-out>/<run-id>/ with a _READY marker written last.
#
# The vLLM server is booted by `olmo-eval run-external --provider vllm_server`,
# not by this script -- that is the documented mapping in SKILL.md and it keeps the
# server lifecycle in one place. This script's jobs are: work out what kind of
# checkpoint it was handed, get it into HF format, invoke the CAT, normalise the
# result, and ship it.
#
# Exit codes: 0 = results uploaded and _READY written. Non-zero = something failed;
# worker.log is still uploaded (without _READY) so the failure is diagnosable in S3.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- defaults ---------------------------------------------------------------
CHECKPOINT=""
RUN_ID=""
S3_OUT=""
TOKENIZER=""
BASE_MODEL=""
SE_STOP="0.3"
MIN_ITEMS="8"
MAX_ITEMS="40"
TP="1"
SKIP_CONVERT=0
KEEP_HF=0
DRY_RUN=0
SKIP_PREFLIGHT=0
VALIDATE_CONVERSION=0
CONVERTER=""
OLMO_EVAL_CMD=""
WORK_DIR=""

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Required:
  --checkpoint PATH     OLMo-core checkpoint dir, HF dir, PEFT adapter dir, or HF id
  --run-id ID           results subdirectory; [A-Za-z0-9._-]+
  --s3-out s3://B/P     results root

Optional:
  --tokenizer ID        HF tokenizer id, when the checkpoint config does not resolve one
  --base-model ID       base model a PEFT adapter is merged onto (adapters only)
  --se-stop F           CAT early-stop standard error            (default 0.3)
  --min-items N         CAT floor                                 (default 8)
  --max-items N         CAT cap                                   (default 40)
  --tp N                vLLM tensor-parallel size                 (default 1)
  --skip-convert        checkpoint is already HF format / an HF id
  --skip-preflight      do not check that olmo-eval is runnable before converting
  --keep-hf             keep the converted -hf dir (default: temp, removed after)
  --validate-conversion run olmo-core's numerical validation (~2x RAM, much slower)
  --converter PATH      override convert_checkpoint_to_hf.py location
  --olmo-eval-cmd CMD   override how olmo-eval is invoked (e.g. "uv run olmo-eval")
  --work-dir PATH       staging dir for HF weights + results (default: mktemp)
  --dry-run             print the plan and exit
  -h, --help            this text
EOF
}

# --- args -------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint)          CHECKPOINT="$2"; shift 2 ;;
        --run-id)              RUN_ID="$2"; shift 2 ;;
        --s3-out)              S3_OUT="${2%/}"; shift 2 ;;
        --tokenizer)           TOKENIZER="$2"; shift 2 ;;
        --base-model)          BASE_MODEL="$2"; shift 2 ;;
        --se-stop)             SE_STOP="$2"; shift 2 ;;
        --min-items)           MIN_ITEMS="$2"; shift 2 ;;
        --max-items)           MAX_ITEMS="$2"; shift 2 ;;
        --tp)                  TP="$2"; shift 2 ;;
        --converter)           CONVERTER="$2"; shift 2 ;;
        --olmo-eval-cmd)       OLMO_EVAL_CMD="$2"; shift 2 ;;
        --work-dir)            WORK_DIR="$2"; shift 2 ;;
        --skip-convert)        SKIP_CONVERT=1; shift ;;
        --skip-preflight)      SKIP_PREFLIGHT=1; shift ;;
        --keep-hf)             KEEP_HF=1; shift ;;
        --validate-conversion) VALIDATE_CONVERSION=1; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$CHECKPOINT" ] || die "--checkpoint is required"
[ -n "$RUN_ID" ]     || die "--run-id is required"
[ -n "$S3_OUT" ]     || die "--s3-out is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id must match [A-Za-z0-9._-]+ (got '$RUN_ID')"
[[ "$S3_OUT" =~ ^s3://[^/]+(/.*)?$ ]] || die "--s3-out must be s3://bucket/prefix (got '$S3_OUT')"

S3_DEST="$S3_OUT/$RUN_ID"

# --- staging ----------------------------------------------------------------
CLEANUP_DIRS=()
if [ -n "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
    STAGE="$WORK_DIR"
else
    STAGE="$(mktemp -d -t cat_diag_XXXXXX)"
    CLEANUP_DIRS+=("$STAGE")
fi
LOG="$STAGE/worker.log"
RESULTS="$STAGE/atlas_arc_results.json"
PROVENANCE="$STAGE/pipeline_provenance.json"
EVAL_OUT="$STAGE/olmo_eval_out"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

UPLOADED=0
finish() {
    rc=$?
    if [ "$DRY_RUN" = "0" ] && [ "$UPLOADED" = "0" ] && [ -f "$LOG" ]; then
        # Ship the log even on failure -- without _READY, so no consumer treats it as done.
        aws s3 cp "$LOG" "$S3_DEST/worker.log" >/dev/null 2>&1 \
            && echo "uploaded failure log -> $S3_DEST/worker.log" >&2 \
            || echo "could not upload failure log to $S3_DEST" >&2
    fi
    if [ ${#CLEANUP_DIRS[@]} -gt 0 ]; then
        if [ "$KEEP_HF" = "0" ]; then
            for d in "${CLEANUP_DIRS[@]}"; do
                [ -d "$d" ] && rm -rf "$d"
            done
        else
            echo "kept: ${CLEANUP_DIRS[*]}" >&2
        fi
    fi
    exit $rc
}
trap finish EXIT

# --- 1. what kind of checkpoint is this? ------------------------------------
detect_kind() {
    local ckpt="$1"
    if [ ! -e "$ckpt" ]; then
        echo "hf_id"                        # not a path: assume a HuggingFace id
    elif [ -d "$ckpt/model_and_optim" ]; then
        echo "olmo_core"                    # native OLMo-core checkpoint
    elif [ -f "$ckpt/adapter_config.json" ]; then
        echo "peft"                         # LoRA/PEFT adapter -- merge onto a base
    elif [ -f "$ckpt/config.json" ]; then
        echo "hf"                           # already HF format
    else
        echo "unknown"
    fi
}

KIND="$(detect_kind "$CHECKPOINT")"
[ "$KIND" != "unknown" ] || die \
    "cannot tell what '$CHECKPOINT' is: no model_and_optim/, no adapter_config.json, no config.json"
if [ "$SKIP_CONVERT" = "1" ] && [ "$KIND" = "olmo_core" ]; then
    die "--skip-convert given, but '$CHECKPOINT' is a native OLMo-core checkpoint (has model_and_optim/). vLLM cannot load native weights; drop --skip-convert."
fi

# A PEFT adapter records its own base model; --base-model overrides it.
if [ "$KIND" = "peft" ] && [ -z "$BASE_MODEL" ]; then
    BASE_MODEL="$(python3 - "$CHECKPOINT" <<'PY' || true
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/adapter_config.json"))
print(cfg.get("base_model_name_or_path") or "")
PY
)"
    [ -n "$BASE_MODEL" ] || die \
        "'$CHECKPOINT' is a PEFT adapter with no base_model_name_or_path in adapter_config.json. Pass --base-model."
fi

# --- 2. resolve the tools we need -------------------------------------------
if [ -z "$CONVERTER" ] && [ "$KIND" = "olmo_core" ]; then
    REPO_ROOT="$(git -C "$SKILL_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    for cand in \
        "${REPO_ROOT:-/nonexistent}/src/examples/huggingface/convert_checkpoint_to_hf.py" \
        "$SKILL_DIR/../../../src/examples/huggingface/convert_checkpoint_to_hf.py"
    do
        if [ -f "$cand" ]; then CONVERTER="$cand"; break; fi
    done
    [ -n "$CONVERTER" ] || die \
        "cannot find convert_checkpoint_to_hf.py (looked under the OLMo-core repo root). Pass --converter."
fi

if [ -z "$OLMO_EVAL_CMD" ]; then
    if command -v olmo-eval >/dev/null 2>&1; then
        OLMO_EVAL_CMD="olmo-eval"
    elif command -v uv >/dev/null 2>&1; then
        OLMO_EVAL_CMD="uv run olmo-eval"
    else
        die "olmo-eval not found and uv is not installed. Install olmo-eval with vLLM extras (uv sync --extra vllm --extra hf) or pass --olmo-eval-cmd."
    fi
fi

# Resolving the command is not the same as it working: `uv run olmo-eval` resolves on
# any box with uv installed. A conversion costs minutes and gigabytes, so find out
# now rather than after paying for it.
# shellcheck disable=SC2086
preflight_ok() { $OLMO_EVAL_CMD --help >/dev/null 2>&1; }
if [ "$SKIP_PREFLIGHT" = "1" ]; then
    PREFLIGHT="skipped (--skip-preflight)"
elif preflight_ok; then
    PREFLIGHT="ok"
else
    PREFLIGHT="FAILING ('$OLMO_EVAL_CMD --help' is not runnable)"
fi

# --- 3. the plan ------------------------------------------------------------
case "$KIND" in
    hf|hf_id) HF_CKPT="$CHECKPOINT" ;;
    *)        HF_CKPT="$STAGE/$(basename "$CHECKPOINT")-hf" ;;
esac
[ "$SKIP_CONVERT" = "0" ] || HF_CKPT="$CHECKPOINT"

if [ "$SKIP_CONVERT" = "1" ]; then
    CONVERT_PLAN="no (--skip-convert)"
elif [ "$KIND" = "hf" ] || [ "$KIND" = "hf_id" ]; then
    CONVERT_PLAN="no (already HF)"
elif [ "$KIND" = "peft" ]; then
    CONVERT_PLAN="yes (merge adapter onto $BASE_MODEL)"
else
    CONVERT_PLAN="yes (olmo-core -> HF)"
fi

{
    echo "run_id       : $RUN_ID"
    echo "checkpoint   : $CHECKPOINT"
    echo "kind         : $KIND"
    [ "$KIND" = "peft" ] && echo "base_model   : $BASE_MODEL"
    echo "hf_checkpoint: $HF_CKPT"
    echo "convert      : $CONVERT_PLAN"
    echo "cat          : atlas_arc se_stop=$SE_STOP min_items=$MIN_ITEMS max_items=$MAX_ITEMS tp=$TP"
    echo "olmo-eval    : $OLMO_EVAL_CMD  [preflight: $PREFLIGHT]"
    [ -n "$CONVERTER" ] && echo "converter    : $CONVERTER"
    echo "stage        : $STAGE"
    echo "destination  : $S3_DEST/"
} | tee -a "$LOG"

if [ "$DRY_RUN" = "1" ]; then
    echo "--dry-run: nothing executed."
    exit 0
fi

command -v aws >/dev/null 2>&1 || die "aws CLI not found; results could not be uploaded"
case "$PREFLIGHT" in
    FAILING*) die "$OLMO_EVAL_CMD is not runnable, so the CAT would fail after conversion had already cost time and disk. Install olmo-eval with vLLM extras (uv sync --extra vllm --extra hf), pass --olmo-eval-cmd, or --skip-preflight to try anyway." ;;
esac

# --- 4. convert -------------------------------------------------------------
T_CONVERT_START=$(date +%s)
if [ "$SKIP_CONVERT" = "1" ] || [ "$KIND" = "hf" ] || [ "$KIND" = "hf_id" ]; then
    log "conversion skipped (kind=$KIND)"
elif [ "$KIND" = "olmo_core" ]; then
    log "converting native OLMo-core checkpoint -> HF ($HF_CKPT)"
    conv=("python3" "$CONVERTER" -i "$CHECKPOINT" -o "$HF_CKPT" --dtype bfloat16)
    [ -n "$TOKENIZER" ] && conv+=(-t "$TOKENIZER")
    [ "$VALIDATE_CONVERSION" = "1" ] || conv+=(--skip-validation)
    log "+ ${conv[*]}"
    "${conv[@]}" 2>&1 | tee -a "$LOG"
    CLEANUP_DIRS+=("$HF_CKPT")
elif [ "$KIND" = "peft" ]; then
    log "merging PEFT adapter onto $BASE_MODEL -> HF ($HF_CKPT)"
    python3 - "$CHECKPOINT" "$BASE_MODEL" "$HF_CKPT" "${TOKENIZER:-}" 2>&1 <<'PY' | tee -a "$LOG"
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter, base_id, out, tok_id = sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4] or None)
print(f"loading base {base_id}")
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
print(f"applying adapter {adapter}")
merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
merged.save_pretrained(out, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(tok_id or base_id)
if tok.chat_template is None:
    print("WARNING: tokenizer has no chat_template; vLLM chat formatting will differ from training")
tok.save_pretrained(out)
print(f"wrote merged model -> {out}")
PY
    CLEANUP_DIRS+=("$HF_CKPT")
fi
T_CONVERT=$(( $(date +%s) - T_CONVERT_START ))

# --- 5. the CAT -------------------------------------------------------------
mkdir -p "$EVAL_OUT"
log "running atlas_arc CAT"
T_EVAL_START=$(date +%s)
EVAL_ARGS=(-a "se_stop=$SE_STOP" -a "min_items=$MIN_ITEMS" -a "max_items=$MAX_ITEMS")
# Only forwarded when it is actually needed: `tensor_parallel_size` is the one arg
# name here not pinned by SKILL.md's documented invocation, so a single-GPU run
# (the common case) never depends on it being right.
[ "$TP" = "1" ] || EVAL_ARGS+=(-a "tensor_parallel_size=$TP")

# shellcheck disable=SC2086  # OLMO_EVAL_CMD may legitimately be "uv run olmo-eval"
set +e
$OLMO_EVAL_CMD run-external \
    -m "$HF_CKPT" -e atlas_arc --provider vllm_server \
    "${EVAL_ARGS[@]}" \
    -O "$EVAL_OUT" 2>&1 | tee -a "$LOG"
eval_rc=${PIPESTATUS[0]}
set -e
T_EVAL=$(( $(date +%s) - T_EVAL_START ))
[ "$eval_rc" = "0" ] || die "olmo-eval exited $eval_rc -- see $LOG"

# --- 6. normalise the result ------------------------------------------------
# olmo-eval's on-disk layout is version-dependent, so find the payload by content
# (a JSON object carrying a theta) rather than by a hardcoded filename. Failing
# loudly here beats uploading an empty result that reads as a finished run.
python3 - "$EVAL_OUT" "$RESULTS" 2>&1 <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path

out_dir, dest = Path(sys.argv[1]), Path(sys.argv[2])
FIELDS = ("theta", "se", "pirt_accuracy", "n_items", "selected_question_ids")


def candidates(obj):
    """Yield every dict in the document that looks like a CAT result."""
    if isinstance(obj, dict):
        if "theta" in obj:
            yield obj
        for v in obj.values():
            yield from candidates(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from candidates(v)


found = []
for p in sorted(out_dir.rglob("*.json")):
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    for c in candidates(doc):
        found.append((p, c))

if not found:
    listing = "\n  ".join(str(p.relative_to(out_dir)) for p in sorted(out_dir.rglob("*"))) or "(empty)"
    sys.exit(f"no atlas_arc result with a 'theta' field under {out_dir}. Contents:\n  {listing}")

src, res = found[0]
payload = {k: res.get(k) for k in FIELDS}
payload["source_file"] = str(src.relative_to(out_dir))
if len(found) > 1:
    payload["note"] = f"{len(found)} theta-bearing objects found; took {payload['source_file']}"
missing = [k for k in FIELDS if payload.get(k) is None]
if missing:
    payload["missing_fields"] = missing
    print(f"WARNING: fields absent from the olmo-eval payload: {missing}")
dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"theta={payload['theta']} se={payload['se']} n_items={payload['n_items']} <- {payload['source_file']}")
PY

# --- 7. provenance ----------------------------------------------------------
# Values go through the environment, not string interpolation, so a path containing
# a quote cannot produce broken JSON or inject Python.
CAT_RUN_ID="$RUN_ID" CAT_CKPT="$CHECKPOINT" CAT_KIND="$KIND" CAT_BASE="$BASE_MODEL" \
CAT_HF="$HF_CKPT" CAT_SE="$SE_STOP" CAT_MIN="$MIN_ITEMS" CAT_MAX="$MAX_ITEMS" \
CAT_TP="$TP" CAT_T_CONVERT="$T_CONVERT" CAT_T_EVAL="$T_EVAL" CAT_SKILL_DIR="$SKILL_DIR" \
python3 - > "$PROVENANCE" <<'PY'
import json, os, platform, shutil, subprocess

env = os.environ


def version(pkg):
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:
        return None


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None


print(json.dumps({
    "run_id": env["CAT_RUN_ID"],
    "checkpoint": env["CAT_CKPT"],
    "checkpoint_kind": env["CAT_KIND"],
    "base_model": env["CAT_BASE"] or None,
    "hf_checkpoint": env["CAT_HF"],
    "eval": {"name": "atlas_arc", "provider": "vllm_server",
             "se_stop": float(env["CAT_SE"]), "min_items": int(env["CAT_MIN"]),
             "max_items": int(env["CAT_MAX"]), "tensor_parallel_size": int(env["CAT_TP"])},
    "scope_note": ("ARC-Challenge only -- one ability signal, not the full 5-benchmark "
                   "ATLAS profile."),
    "timings_seconds": {"convert": int(env["CAT_T_CONVERT"]), "eval": int(env["CAT_T_EVAL"])},
    "git_sha": git_sha(env["CAT_SKILL_DIR"]),
    "versions": {p: version(p) for p in
                 ("ai2-olmo-core", "transformers", "peft", "vllm", "olmo-eval", "torch")},
    "host": platform.node(),
    "platform": platform.platform(),
    "gpu": gpu(),
}, indent=2))
PY

# --- 8. upload; _READY last -------------------------------------------------
log "uploading -> $S3_DEST/"
for f in "$RESULTS" "$PROVENANCE" "$LOG"; do
    aws s3 cp "$f" "$S3_DEST/$(basename "$f")" >/dev/null || die "upload failed: $f"
done
: > "$STAGE/_READY"
aws s3 cp "$STAGE/_READY" "$S3_DEST/_READY" >/dev/null || die "upload failed: _READY"
UPLOADED=1

log "done: $S3_DEST/ (convert ${T_CONVERT}s, eval ${T_EVAL}s)"
