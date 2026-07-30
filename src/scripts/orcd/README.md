# ORCD smoke jobs

## Pre-acceptance operator procedure

These commands are an operator procedure, not acceptance evidence. Run them only after the
local verification gate passes and the operator has Engaging access, one-L40S permission, and
W&B `eduLLM/test` membership.

The operator or review process must supply the full reviewed commit SHA. Never derive trust from
whatever commit happens to be checked out remotely. Official ORCD documentation places Engaging
Scratch at `/home/<username>/orcd/scratch` and Pool at `/home/<username>/orcd/pool`.
`$HOME/orcd/scratch` is a separate Engaging Scratch mount and quota despite its Home-shaped path;
it is for active jobs, not ordinary backed-up Home storage. The standard eduLLM candidate is
`$HOME/orcd/scratch/edullm`; set it explicitly and verify the mount before submission. See
https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/.

If the reviewed commit is not available from a remote, transfer the exact commit object through
an approved mechanism such as a verified Git bundle.

```bash
export EDULLM_REPO_ROOT="$HOME/OLMo-core"
: "${EDULLM_COMMIT_SHA:?set the known reviewed 40-character commit SHA}"
export EDULLM_SCRATCH="$HOME/orcd/scratch/edullm"

[[ "$EDULLM_COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "EDULLM_COMMIT_SHA must be a full lowercase commit SHA" >&2
  exit 2
}

verify_edullm_scratch() {
  local EDULLM_SCRATCH_ROOT="$HOME/orcd/scratch"
  local RESOLVED_SCRATCH_ROOT RESOLVED_SCRATCH RESOLVED_CREATED_SCRATCH
  local SCRATCH_MOUNT HOME_MOUNT SCRATCH_PROBE

  if [[ ! -d "$EDULLM_SCRATCH_ROOT" ]]; then
    echo "Engaging Scratch root does not exist: $EDULLM_SCRATCH_ROOT" >&2
    return 2
  fi
  if ! RESOLVED_SCRATCH_ROOT="$(realpath -e "$EDULLM_SCRATCH_ROOT")"; then
    echo "could not resolve Engaging Scratch root" >&2
    return 2
  fi
  if ! RESOLVED_SCRATCH="$(realpath -m "$EDULLM_SCRATCH")"; then
    echo "could not resolve EDULLM_SCRATCH" >&2
    return 2
  fi
  case "$RESOLVED_SCRATCH/" in
    "$RESOLVED_SCRATCH_ROOT/"*) ;;
    *) echo "EDULLM_SCRATCH does not resolve under Engaging Scratch" >&2; return 2 ;;
  esac

  if ! SCRATCH_MOUNT="$(findmnt -n -o TARGET -T "$RESOLVED_SCRATCH_ROOT")"; then
    echo "findmnt could not identify the Engaging Scratch filesystem" >&2
    return 2
  fi
  if ! HOME_MOUNT="$(findmnt -n -o TARGET -T "$HOME")"; then
    echo "findmnt could not identify the Home filesystem" >&2
    return 2
  fi
  if [[ -z "$SCRATCH_MOUNT" || "$SCRATCH_MOUNT" == "$HOME_MOUNT" ]]; then
    echo "Engaging Scratch is not a separate mounted filesystem" >&2
    return 2
  fi

  if ! mkdir -p "$EDULLM_SCRATCH"; then
    echo "could not create EDULLM_SCRATCH" >&2
    return 2
  fi
  if ! RESOLVED_CREATED_SCRATCH="$(realpath -e "$EDULLM_SCRATCH")"; then
    echo "could not resolve created EDULLM_SCRATCH" >&2
    return 2
  fi
  case "$RESOLVED_CREATED_SCRATCH/" in
    "$RESOLVED_SCRATCH_ROOT/"*) ;;
    *) echo "created EDULLM_SCRATCH escaped Engaging Scratch" >&2; return 2 ;;
  esac
  if [[ ! -w "$EDULLM_SCRATCH" ]]; then
    echo "EDULLM_SCRATCH is not writable" >&2
    return 2
  fi
  if ! SCRATCH_PROBE="$(mktemp "$EDULLM_SCRATCH/.edullm-preflight.XXXXXX")"; then
    echo "could not write an EDULLM_SCRATCH probe" >&2
    return 2
  fi
  rm -f "$SCRATCH_PROBE"
}
verify_edullm_scratch || exit $?

if ! cd "$EDULLM_REPO_ROOT"; then
  echo "could not enter EDULLM_REPO_ROOT" >&2
  exit 2
fi
if ! REMOTE_COMMIT_SHA="$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)"; then
  echo "could not resolve remote commit SHA" >&2
  exit 2
fi
if [[ "$REMOTE_COMMIT_SHA" != "$EDULLM_COMMIT_SHA" ]]; then
  echo "remote commit does not match reviewed EDULLM_COMMIT_SHA" >&2
  exit 2
fi
if ! REMOTE_STATUS="$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"; then
  echo "could not inspect remote checkout cleanliness" >&2
  exit 2
fi
if [[ -n "$REMOTE_STATUS" ]]; then
  echo "remote checkout is not clean" >&2
  exit 2
fi

# Setup environment
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH" \
  src/scripts/orcd/setup_env.sbatch

# GPU probe
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH" \
  src/scripts/orcd/probe.sbatch
```

Wait for setup to finish before submitting the probe, and wait for the probe to confirm one L40S,
CUDA, imports, W&B reachability status, and writable Scratch before training.

## Configure W&B privately

Provision the operator's own key without putting it in a command argument, repository, Slurm log,
or this file. Run the `printf` only from a shell where `WANDB_API_KEY` was populated without
shell-history exposure, then unset it.

```bash
cd "$EDULLM_REPO_ROOT"
mkdir -p "$HOME/.config/edullm"
umask 077
cp src/scripts/orcd/wandb.env.example "$HOME/.config/edullm/wandb.env"
printf '%s\n' "$WANDB_API_KEY" > "$HOME/.config/edullm/wandb.key"
unset WANDB_API_KEY
chmod 600 "$HOME/.config/edullm/wandb.env" "$HOME/.config/edullm/wandb.key"
```

Before accepting the run, have another `eduLLM` member confirm that the resulting `eduLLM/test`
run is visible. Never copy the key into `sbatch --export`; jobs source the private key file through
the secret-free environment file.

## Verify checkpoint and resume

Keep `EDULLM_COMMIT_SHA`, `RUN_NAME`, `WANDB_RUN_ID`, and `SAVE_FOLDER` unchanged between the
initial and resumed submissions.

```bash
# Initial run
export RUN_NAME=orcd-bootstrap
export WANDB_RUN_ID=orcd-bootstrap
export SAVE_FOLDER="$EDULLM_SCRATCH/runs/$RUN_NAME"
export HARD_STOP_STEPS=20
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",SAVE_FOLDER="$SAVE_FOLDER",HARD_STOP_STEPS="$HARD_STOP_STEPS" \
  src/scripts/orcd/generic_smoke.sbatch

# Resume run
export HARD_STOP_STEPS=25
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",SAVE_FOLDER="$SAVE_FOLDER",HARD_STOP_STEPS="$HARD_STOP_STEPS" \
  src/scripts/orcd/generic_smoke.sbatch
```

The first job writes durable checkpoints at steps 10 and 20; ephemeral checkpoints remain
disabled. The second job uses automatic checkpoint loading, resumes the same checkpoint and W&B
identity, and advances the run to step 25.

## Exercise forced-offline W&B

Use a separate deterministic run identity so the outage exercise cannot alter checkpoint/resume
evidence. The smoke passes its explicit `SAVE_FOLDER` as trainer work-dir, so OLMo calls
`wandb.init(dir="$SAVE_FOLDER/wandb")`. W&B therefore places offline run directories under
`$SAVE_FOLDER/wandb/wandb`.

```bash
# Forced-offline smoke
export RUN_NAME=orcd-bootstrap-offline
export WANDB_RUN_ID=orcd-bootstrap-offline
export SAVE_FOLDER="$EDULLM_SCRATCH/runs/$RUN_NAME"
export HARD_STOP_STEPS=20
export WANDB_MODE=offline
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",SAVE_FOLDER="$SAVE_FOLDER",HARD_STOP_STEPS="$HARD_STOP_STEPS",WANDB_MODE="$WANDB_MODE" \
  src/scripts/orcd/generic_smoke.sbatch
```

After the offline job finishes, identify exactly one run directory and sync only that directory.
Do not use `--sync-all`.

```bash
source "$HOME/venvs/edullm/bin/activate"
source "$HOME/.config/edullm/wandb.env"
mapfile -t OFFLINE_RUN_DIRS < <(
  find "$SAVE_FOLDER/wandb/wandb" -mindepth 1 -maxdepth 1 -type d -name 'offline-run-*'
)
if [[ "${#OFFLINE_RUN_DIRS[@]}" -ne 1 ]]; then
  echo "expected exactly one offline W&B run directory; found ${#OFFLINE_RUN_DIRS[@]}" >&2
  exit 2
fi
OFFLINE_RUN_DIR="${OFFLINE_RUN_DIRS[0]}"
wandb sync "$OFFLINE_RUN_DIR"
WANDB_SYNC_STATUS=$?
unset WANDB_MODE
if [[ "$WANDB_SYNC_STATUS" -ne 0 ]]; then
  exit "$WANDB_SYNC_STATUS"
fi
```

## Pilot a bounded S3 transfer

A live transfer is conditional on separate, explicit approval and does not block
Plan 1 exit. The sandbox bucket is pilot-only; measure AWS transfer charges before
scaling this workflow.

Use only a bounded token shard that is public or research-cleared and has a known
SHA-256 digest. Keep the short-lived presigned GET and PUT URLs in a temporary JSON
file outside the repository with mode `0600`. Never place either URL in Git, command
arguments, Slurm logs, W&B, reports, or other durable output. The JSON object has
`download_url` and `upload_url` keys, but documentation and committed fixtures must
not contain URL or signature values.

After approval, run the transfer tool from a clean checkout. Pass the private JSON
file with `--url-file`, the known shard digest with `--expected-sha256`, a strict
byte bound with `--max-download-bytes`, and an explicit Engaging Scratch directory
with `--work-dir`. Pass `step20/config.json` or a deliberately selected checkpoint
archive from the smoke run with `--result-file`; do not upload a synthetic digest
payload.

Copy the verified downloaded shard into the generic smoke data root, generate the
local validation shard there, and submit the staged-data training run with both
`EDULLM_SCRATCH` and `OLMO_DATA_ROOT` set to explicit Engaging Scratch paths. The
tool reports byte counts, SHA-256, elapsed time, and throughput without including
credentials or presigned URLs.

## Engaging acceptance evidence — 2026-07-22

Status: acceptance evidence complete; independent re-review pending.

- Accepted commit: `cf6118809ec135c66d471727d7ba34c82f8465f5`
- Scratch: `$HOME/orcd/scratch/edullm`, resolved under the separate
  `/orcd/scratch/orcd/008` mount and verified writable
- Environment: Python `3.10.14`, PyTorch `2.11.0+cu128`, W&B `0.28.1`,
  OLMo-core `2.5.0`
- Environment setup: Slurm `18575176`, `COMPLETED 0:0` in `00:01:56`, submitted
  with the accepted commit SHA and Scratch root
- Pre-execution review: the exact `8d20b59..cf61188` SDD review package received
  an independent Approved verdict before setup job `18575176` began
- GPU probe: Slurm `18576231`, `COMPLETED 0:0` in `00:00:12`, one NVIDIA L40S
- Initial smoke: Slurm `18576778`, `COMPLETED 0:0` in `00:04:19`, trained from
  scratch through step 20
- Resume smoke: Slurm `18577188`, `COMPLETED 0:0` in `00:01:42`, loaded step 20
  and advanced through step 25 with the same run identity
- Forced-offline smoke: Slurm `18578152`, `COMPLETED 0:0` in `00:01:21`; exactly
  one offline run directory was preserved and synced successfully
- W&B run: https://wandb.ai/eduLLM/test/runs/orcd-bootstrap
- W&B identity: run ID, name, and group `orcd-bootstrap`; state `finished`;
  25 `train/CE loss` points; final step `25`; final value
  `10.968331336975098`
- W&B evidence: required config, output, requirements, metadata, and summary files
  are present; system history includes CPU, memory, disk, network, process, and
  GPU metrics
- Second-member visibility: Frank Gonzalez requested a check at 22:24; Eric Wu
  confirmed at 22:25 that he could see `eduLLM/test/orcd-bootstrap`
- Checkpoints:
  `$HOME/orcd/scratch/edullm/runs/orcd-bootstrap/step20` and
  `$HOME/orcd/scratch/edullm/runs/orcd-bootstrap/step25`
- Logs:
  `$HOME/orcd/scratch/edullm/logs/edullm-env-18575176.log`,
  `$HOME/orcd/scratch/edullm/logs/edullm-probe-18576231.log`,
  `$HOME/orcd/scratch/edullm/logs/edullm-smoke-18576778.log`,
  `$HOME/orcd/scratch/edullm/logs/edullm-smoke-18577188.log`, and
  `$HOME/orcd/scratch/edullm/logs/edullm-smoke-18578152.log`
- S3 pilot: not run. The eligible result file existed, but no approved concrete
  mode-`0600` URL JSON, expected digest, or size bound existed; no transfer
  measurements were produced. This is conditional and non-blocking for Plan 1.
