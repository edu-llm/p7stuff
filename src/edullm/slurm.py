"""
Render reviewed Slurm jobs and perform one idempotent submission transaction.
"""

from __future__ import annotations

import base64
import dataclasses
import fcntl
import hashlib
import json
import os
import pwd
import re
import secrets
import shlex
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from edullm.data_manifest import verify_manifest
from edullm.models import ResolvedRequest
from edullm.secure_publish import (
    SecurePublishError,
    capture_file,
    compare_and_publish,
    directory_identity,
)

SBATCH_TIMEOUT_SECONDS = 30.0
MAX_SBATCH_OUTPUT_CHARS = 64
MAX_RECOVERY_OUTPUT_CHARS = 65_536
MAX_RECEIPT_CHARS = 16_384
MAX_SCRIPT_BYTES = 1_048_576

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_LOGIN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_REMOTE_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]{0,19}\Z")
_SAFE_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,99}\Z")
_CONDITION = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}\Z")
_MODEL = re.compile(r"[a-z0-9][a-z0-9._-]{0,99}\Z")
_PROJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_RECOVERABLE_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "COMPLETING",
        "CONFIGURING",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PENDING",
        "PREEMPTED",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "REVOKED",
        "RUNNING",
        "SPECIAL_EXIT",
        "STAGE_OUT",
        "SUSPENDED",
        "TIMEOUT",
    }
)
_SUBMISSION_FIELDS = frozenset(
    {
        "allowed_data_kinds",
        "attempt_number",
        "issue",
        "log_pattern",
        "manifest_sha256",
        "manifest_uri",
        "operator",
        "remote_user",
        "request_digest",
        "script_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "attempt_number",
        "issue",
        "log_path",
        "manifest_sha256",
        "operator",
        "remote_user",
        "request_digest",
        "script_sha256",
        "slurm_job_id",
        "submitted_at",
    }
)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class SubmissionError(RuntimeError):
    """A sanitized render, staging, or submission failure."""


@dataclass(frozen=True)
class SubmissionSpec:
    """Immutable identities required by the remote submission transaction."""

    issue: int
    request_digest: str
    attempt_number: int
    operator: str
    remote_user: str
    script_sha256: str
    manifest_uri: str
    manifest_sha256: str
    allowed_data_kinds: tuple[str, ...]
    log_pattern: str

    def canonical_json(self) -> str:
        """Return strict canonical JSON for a durable pre-submission intent."""
        _validate_spec(self)
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SubmissionReceipt:
    """Canonical durable evidence that Slurm accepted one attempt."""

    issue: int
    request_digest: str
    attempt_number: int
    operator: str
    remote_user: str
    script_sha256: str
    manifest_sha256: str
    slurm_job_id: str
    log_path: str
    submitted_at: datetime

    def canonical_json(self) -> str:
        """Return the exact bounded receipt representation."""
        _validate_receipt(self)
        payload = {
            "attempt_number": self.attempt_number,
            "issue": self.issue,
            "log_path": self.log_path,
            "manifest_sha256": self.manifest_sha256,
            "operator": self.operator,
            "remote_user": self.remote_user,
            "request_digest": self.request_digest,
            "script_sha256": self.script_sha256,
            "slurm_job_id": self.slurm_job_id,
            "submitted_at": _format_time(self.submitted_at),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _safe_text(value: object, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SubmissionError(f"{name} is invalid")
    return cast(str, value)


def _safe_argument(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SubmissionError("training argument is invalid")
    return cast(str, value)


def _safe_repository_path(value: object) -> str:
    if type(value) is not str or len(value) > 512 or "\\" in value:
        raise SubmissionError("repository script path is invalid")
    path = PurePosixPath(cast(str, value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            not part or any(ord(char) < 33 or ord(char) == 127 for char in part)
            for part in path.parts
        )
    ):
        raise SubmissionError("repository script path is invalid")
    return cast(str, value)


def _duration(minutes: int) -> str:
    if type(minutes) is not int or not 1 <= minutes <= 360:
        raise SubmissionError("runtime is invalid")
    hours, remaining = divmod(minutes, 60)
    return f"{hours:02d}:{remaining:02d}:00"


def _attempt_number(resolved: ResolvedRequest) -> int:
    request = resolved.request
    match = re.fullmatch(
        rf"issue-{request.issue_number}-attempt-([1-9][0-9]{{0,5}})", resolved.wandb_run_prefix
    )
    if match is None:
        raise SubmissionError("W&B run prefix is invalid")
    return int(match.group(1))


def _validate_resolved(resolved: object) -> tuple[int, list[str]]:
    if not isinstance(resolved, ResolvedRequest):
        raise SubmissionError("resolved request is invalid")
    request = resolved.request
    if type(request.issue_number) is not int or request.issue_number <= 0:
        raise SubmissionError("Issue number is invalid")
    _safe_text(request.commit_sha, _SHA, "commit SHA")
    _safe_repository_path(request.script_path)
    if request.launcher not in {"python", "torchrun", "bash"}:
        raise SubmissionError("launcher is invalid")
    if type(request.argv) is not tuple:
        raise SubmissionError("training arguments are invalid")
    arguments = [_safe_argument(value) for value in request.argv]
    if type(request.gpu_count) is not int or not 1 <= request.gpu_count <= 2:
        raise SubmissionError("GPU count is invalid")
    if request.gpu_preference not in {"any", "l40s", "h100", "h200"}:
        raise SubmissionError("GPU preference is invalid")
    _duration(request.max_runtime_minutes)
    if type(request.seed) is not int or not 0 <= request.seed <= 2_147_483_647:
        raise SubmissionError("seed is invalid")
    _safe_text(request.digest, _SHA256, "request digest")
    _safe_text(request.data_manifest_sha256, _SHA256, "manifest digest")
    _safe_text(resolved.operator, _LOGIN, "operator")
    _safe_text(resolved.wandb_entity, _PROJECT, "W&B entity")
    _safe_text(request.wandb_project, _PROJECT, "W&B project")
    _safe_text(request.entrypoint_profile, _SAFE_SLUG, "entrypoint profile")
    _safe_text(request.condition, _CONDITION, "condition")
    _safe_text(request.request_name, _SAFE_SLUG, "request name")
    _safe_text(resolved.model_identity, _MODEL, "model identity")
    _safe_text(resolved.slurm_job_name, _SAFE_SLUG, "Slurm job name")
    if resolved.slurm_job_name != request.request_name:
        raise SubmissionError("Slurm job name does not match the request")
    if (
        type(request.data_manifest) is not str
        or not request.data_manifest
        or len(request.data_manifest) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in request.data_manifest)
        or (
            request.data_manifest != "builtin://generic-smoke-v1"
            and not request.data_manifest.startswith("/orcd/pool/")
        )
    ):
        raise SubmissionError("data manifest is invalid")
    if resolved.repository_url != "https://github.com/edu-llm/OLMo-core.git":
        raise SubmissionError("repository URL is invalid")
    if resolved.slurm_partition != "mit_normal_gpu":
        raise SubmissionError("Slurm partition is invalid")
    if resolved.slurm_memory != "64G":
        raise SubmissionError("Slurm memory is invalid")
    if type(resolved.slurm_cpus_per_gpu) is not int or resolved.slurm_cpus_per_gpu != 4:
        raise SubmissionError("Slurm CPU policy is invalid")
    if resolved.scratch_root != "$HOME/orcd/scratch/edullm":
        raise SubmissionError("Scratch root is invalid")
    attempt = _attempt_number(resolved)
    expected_log = f"logs/issue-{request.issue_number}-attempt-{attempt}-%j.log"
    if resolved.log_pattern != expected_log:
        raise SubmissionError("Slurm log pattern is invalid")
    if (
        type(resolved.allowed_data_kinds) is not tuple
        or not resolved.allowed_data_kinds
        or len(resolved.allowed_data_kinds) > 8
        or any(
            type(kind) is not str or _SAFE_SLUG.fullmatch(kind) is None
            for kind in resolved.allowed_data_kinds
        )
        or len(set(resolved.allowed_data_kinds)) != len(resolved.allowed_data_kinds)
    ):
        raise SubmissionError("allowed data kinds are invalid")
    if (
        type(resolved.fixed_launcher_arguments) is not tuple
        or type(resolved.fixed_arguments) is not tuple
        or type(resolved.fixed_option_names) is not tuple
        or type(resolved.derived_path_options) is not tuple
        or type(resolved.fixed_wandb_tags) is not tuple
    ):
        raise SubmissionError("protected fixed arguments are invalid")
    launcher_arguments = [_safe_argument(value) for value in resolved.fixed_launcher_arguments]
    if request.launcher == "torchrun":
        expected_launcher = {"--standalone", f"--nproc-per-node={request.gpu_count}"}
        if launcher_arguments and (
            set(launcher_arguments) != expected_launcher
            or len(launcher_arguments) != len(expected_launcher)
        ):
            raise SubmissionError("protected launcher arguments are invalid")
    elif launcher_arguments:
        raise SubmissionError("protected launcher arguments are invalid")
    fixed_arguments = [_safe_argument(value) for value in resolved.fixed_arguments]
    fixed_wandb_tags = [_safe_argument(value) for value in resolved.fixed_wandb_tags]
    if len(set(fixed_wandb_tags)) != len(fixed_wandb_tags):
        raise SubmissionError("protected W&B tags are invalid")
    names = list(resolved.fixed_option_names)
    derived = list(resolved.derived_path_options)
    if (
        not all(
            type(name) is str and re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", name) for name in names
        )
        or len(set(names)) != len(names)
        or any(name not in names or name not in {"save-folder", "work-dir"} for name in derived)
        or len(set(derived)) != len(derived)
    ):
        raise SubmissionError("protected fixed options are invalid")
    rendered_names = {
        argument.removeprefix("--").split("=", 1)[0]
        for argument in fixed_arguments
        if "=" in argument
    }
    special_names = {"trainer.callbacks.wandb.tags"} if fixed_wandb_tags else set()
    if rendered_names | set(derived) | special_names != set(names) or len(rendered_names) != len(
        fixed_arguments
    ):
        raise SubmissionError("protected fixed options are invalid")
    return attempt, arguments


def render_sbatch(resolved: ResolvedRequest) -> str:
    """
    Render one deterministic shell-safe Slurm script from typed trusted values.

    Every field is revalidated at this boundary. Slurm directives use exact
    allowlisted syntax; shell arguments are quoted once and never evaluated.
    """
    attempt, request_arguments = _validate_resolved(resolved)
    request = resolved.request
    gpu = str(request.gpu_count)
    if request.gpu_preference != "any":
        gpu = f"{request.gpu_preference}:{request.gpu_count}"
    request_b64 = base64.b64encode(request.canonical_json().encode("utf-8")).decode("ascii")
    data_kind_arguments = " ".join(
        f"--allowed-kind {shlex.quote(kind)}" for kind in resolved.allowed_data_kinds
    )
    dynamic_tags = [
        f"issue-{request.issue_number}",
        f"attempt-{attempt}",
        request.entrypoint_profile,
        request.condition,
        request.gpu_preference,
        "engaging",
        request.commit_sha,
        request.commit_sha[:12],
        f"seed-{request.seed}",
        request.digest,
        request.data_manifest_sha256,
        resolved.model_identity,
    ]
    tags = list(dict.fromkeys((*resolved.fixed_wandb_tags, *dynamic_tags)))
    fixed_names = set(resolved.fixed_option_names)
    callback_values = {
        "trainer.callbacks.wandb.enabled": "true",
        "trainer.callbacks.wandb.entity": resolved.wandb_entity,
        "trainer.callbacks.wandb.project": request.wandb_project,
        "trainer.callbacks.wandb.group": request.study,
        "trainer.callbacks.wandb.tags": json.dumps(tags, sort_keys=True, separators=(",", ":")),
    }
    callback_arguments = [
        f"--{name}={value}"
        for name, value in callback_values.items()
        if name not in fixed_names or name == "trainer.callbacks.wandb.tags"
    ]
    callback_arguments.append(f"--trainer.callbacks.wandb.notes=request_digest:{request.digest}")
    command_arguments = [
        _safe_repository_path(request.script_path),
        *request_arguments,
        *resolved.fixed_arguments,
        *callback_arguments,
    ]
    if request.launcher == "python":
        command_arguments = ["python", *command_arguments]
    elif request.launcher == "torchrun":
        launcher_arguments = resolved.fixed_launcher_arguments or (
            "--standalone",
            f"--nproc-per-node={request.gpu_count}",
        )
        command_arguments = [
            "torchrun",
            *launcher_arguments,
            *command_arguments,
        ]
    else:
        command_arguments = ["bash", *command_arguments]
    command = shlex.join(command_arguments)
    derived_arguments = "\n".join(
        f'TRAIN_COMMAND+=("--{name}=$EDULLM_RUN_DIR")' for name in resolved.derived_path_options
    )

    return f"""#!/bin/bash
#SBATCH -p {resolved.slurm_partition}
#SBATCH -G {gpu}
#SBATCH -t {_duration(request.max_runtime_minutes)}
#SBATCH -c {request.gpu_count * resolved.slurm_cpus_per_gpu}
#SBATCH --mem={resolved.slurm_memory}
#SBATCH -J {resolved.slurm_job_name}
#SBATCH --export=NONE
#SBATCH -o {resolved.log_pattern}
#SBATCH -e {resolved.log_pattern}
set -euo pipefail
umask 077
source "$HOME/venvs/edullm/bin/activate"
source "$HOME/.config/edullm/wandb.env"
EDULLM_SCRATCH="$HOME/orcd/scratch/edullm"
WORKTREE="$EDULLM_SCRATCH/work/{request.request_name}/${{SLURM_JOB_ID}}"
EDULLM_RUN_DIR="$EDULLM_SCRATCH/runs/{request.request_name}"
mkdir -p "$(dirname "$WORKTREE")"
chmod 700 "$(dirname "$WORKTREE")"
git clone --no-checkout {resolved.repository_url} "$WORKTREE"
cd "$WORKTREE"
git fetch --no-tags origin {request.commit_sha}
git checkout --detach {request.commit_sha}
test "$(git rev-parse HEAD)" = {request.commit_sha}
if ! GIT_STATUS="$(git status --porcelain --untracked-files=all)"; then
  exit 2
fi
test -z "$GIT_STATUS"
export PYTHONPATH="$WORKTREE/src"
python -c 'import olmo_core, os; assert os.path.realpath(olmo_core.__file__).startswith(os.path.realpath(os.environ["PYTHONPATH"]) + os.sep)'
export WANDB_ENTITY={shlex.quote(resolved.wandb_entity)}
export WANDB_PROJECT={shlex.quote(request.wandb_project)}
export WANDB_GROUP={shlex.quote(request.study)}
export WANDB_RUN_PREFIX={shlex.quote(resolved.wandb_run_prefix)}
export WANDB_RUN_ID="${{WANDB_RUN_PREFIX}}-${{SLURM_JOB_ID}}"
export WANDB_RUN_URL="https://wandb.ai/${{WANDB_ENTITY}}/${{WANDB_PROJECT}}/runs/${{WANDB_RUN_ID}}"
export WANDB_DIR="$EDULLM_SCRATCH/wandb"
mkdir -p "$WANDB_DIR"
chmod 700 "$WANDB_DIR"
if timeout 10 python -c 'import wandb; wandb.Api(timeout=8).viewer.username' >/dev/null 2>&1; then
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi
export EDULLM_REQUEST_ID={request.issue_number}
export EDULLM_ATTEMPT={attempt}
export EDULLM_COMMIT_SHA={request.commit_sha}
export EDULLM_REQUEST_DIGEST={request.digest}
export EDULLM_DATA_DIGEST={request.data_manifest_sha256}
export EDULLM_MODEL_IDENTITY={shlex.quote(resolved.model_identity)}
export EDULLM_PROFILE={shlex.quote(request.entrypoint_profile)}
export EDULLM_STUDY={shlex.quote(request.study)}
export EDULLM_CONDITION={shlex.quote(request.condition)}
export EDULLM_GPU={shlex.quote(request.gpu_preference)}
export EDULLM_SEED={request.seed}
export EDULLM_DATA_MANIFEST={shlex.quote(request.data_manifest)}
export EDULLM_DATA_MANIFEST_SHA256={request.data_manifest_sha256}
export EDULLM_REQUEST_PATH="$WORKTREE/edullm_request.json"
REQUEST_TMP="$(mktemp "$WORKTREE/.edullm-request.XXXXXX")"
printf %s {shlex.quote(request_b64)} | base64 --decode > "$REQUEST_TMP"
chmod 600 "$REQUEST_TMP"
mv -f "$REQUEST_TMP" "$EDULLM_REQUEST_PATH"
DATA_ENV="$WORKTREE/edullm_data.env"
DATA_ENV_TMP="$(mktemp "$WORKTREE/.edullm-data.XXXXXX")"
python -m edullm.data_manifest render-env \
  "$EDULLM_DATA_MANIFEST" "$EDULLM_DATA_MANIFEST_SHA256" \
  {data_kind_arguments} > "$DATA_ENV_TMP"
chmod 600 "$DATA_ENV_TMP"
mv -f "$DATA_ENV_TMP" "$DATA_ENV"
source "$DATA_ENV"
if [[ "${{EDULLM_DATA_MODE:-}}" == "synthetic" ]]; then
  python src/scripts/orcd/create_tiny_data.py --output "$OLMO_DATA_ROOT"
fi
set +e
TRAIN_COMMAND=({command})
{derived_arguments}
"${{TRAIN_COMMAND[@]}}"
TRAIN_STATUS=$?
set -e
if [[ "${{WANDB_MODE}}" == "offline" ]]; then
  timeout 120 wandb sync --include-offline "$WANDB_DIR" >/dev/null 2>&1 || true
fi
exit "$TRAIN_STATUS"
"""


def build_submission_key(
    issue: int,
    request_digest: str,
    attempt_number: int,
    operator: str,
) -> str:
    """Derive a path-safe opaque key from the complete submission identity."""
    if type(issue) is not int or issue <= 0:
        raise SubmissionError("submission Issue is invalid")
    _safe_text(request_digest, _SHA256, "request digest")
    if type(attempt_number) is not int or not 1 <= attempt_number <= 999_999:
        raise SubmissionError("attempt number is invalid")
    _safe_text(operator, _LOGIN, "operator")
    identity = f"{issue}\0{request_digest}\0{attempt_number}\0{operator}".encode("ascii")
    return hashlib.sha256(identity).hexdigest()


def _validate_spec(spec: object) -> None:
    if not isinstance(spec, SubmissionSpec):
        raise SubmissionError("submission spec is invalid")
    build_submission_key(spec.issue, spec.request_digest, spec.attempt_number, spec.operator)
    _safe_text(spec.remote_user, _REMOTE_USER, "remote user")
    _safe_text(spec.script_sha256, _SHA256, "script digest")
    _safe_text(spec.manifest_sha256, _SHA256, "manifest digest")
    if (
        type(spec.manifest_uri) is not str
        or not spec.manifest_uri
        or len(spec.manifest_uri) > 4096
        or any(ord(char) < 32 or ord(char) == 127 for char in spec.manifest_uri)
    ):
        raise SubmissionError("manifest URI is invalid")
    if (
        type(spec.allowed_data_kinds) is not tuple
        or not spec.allowed_data_kinds
        or len(spec.allowed_data_kinds) > 8
        or any(_SAFE_SLUG.fullmatch(kind) is None for kind in spec.allowed_data_kinds)
        or len(set(spec.allowed_data_kinds)) != len(spec.allowed_data_kinds)
    ):
        raise SubmissionError("allowed data kinds are invalid")
    expected = f"logs/issue-{spec.issue}-attempt-{spec.attempt_number}-%j.log"
    if spec.log_pattern != expected:
        raise SubmissionError("log pattern is invalid")


def _validate_time(value: object) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.utcoffset() != timedelta(0)
        or value.microsecond != 0
        or not 2000 <= value.year <= 9999
    ):
        raise SubmissionError("submission timestamp is invalid")
    return value


def _format_time(value: datetime) -> str:
    return _validate_time(value).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: object) -> datetime:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        raise SubmissionError("submission receipt timestamp is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise SubmissionError("submission receipt timestamp is invalid") from None
    return _validate_time(parsed)


def _valid_log_path(value: object, issue: int, attempt: int, job_id: str) -> bool:
    if type(value) is not str or not value.startswith("/") or len(value) > 4096:
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    path = PurePosixPath(value)
    expected_name = f"issue-{issue}-attempt-{attempt}-{job_id}.log"
    return path.name == expected_name and path.parent.name == "logs" and ".." not in path.parts


def _validate_receipt(receipt: object) -> None:
    if not isinstance(receipt, SubmissionReceipt):
        raise SubmissionError("submission receipt is invalid")
    build_submission_key(
        receipt.issue,
        receipt.request_digest,
        receipt.attempt_number,
        receipt.operator,
    )
    _safe_text(receipt.remote_user, _REMOTE_USER, "remote user")
    _safe_text(receipt.script_sha256, _SHA256, "script digest")
    _safe_text(receipt.manifest_sha256, _SHA256, "manifest digest")
    _safe_text(receipt.slurm_job_id, _JOB_ID, "Slurm job ID")
    if not _valid_log_path(
        receipt.log_path,
        receipt.issue,
        receipt.attempt_number,
        receipt.slurm_job_id,
    ):
        raise SubmissionError("submission receipt log path is invalid")
    _validate_time(receipt.submitted_at)


def parse_submission_receipt(text: str) -> SubmissionReceipt:
    """Parse an exact canonical private submission receipt."""
    if type(text) is not str or not text or len(text) > MAX_RECEIPT_CHARS:
        raise SubmissionError("submission receipt is invalid")
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        raise SubmissionError("submission receipt is invalid") from None
    if (
        type(payload) is not dict
        or set(payload) != _RECEIPT_FIELDS
        or text != json.dumps(payload, sort_keys=True, separators=(",", ":"))
    ):
        raise SubmissionError("submission receipt is invalid")
    data = cast(dict[str, object], payload)
    receipt = SubmissionReceipt(
        issue=cast(int, data["issue"]),
        request_digest=cast(str, data["request_digest"]),
        attempt_number=cast(int, data["attempt_number"]),
        operator=cast(str, data["operator"]),
        remote_user=cast(str, data["remote_user"]),
        script_sha256=cast(str, data["script_sha256"]),
        manifest_sha256=cast(str, data["manifest_sha256"]),
        slurm_job_id=cast(str, data["slurm_job_id"]),
        log_path=cast(str, data["log_path"]),
        submitted_at=_parse_time(data["submitted_at"]),
    )
    _validate_receipt(receipt)
    return receipt


def _ensure_private_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        try:
            parent = path.parent.lstat()
            if (
                not stat.S_ISDIR(parent.st_mode)
                or stat.S_ISLNK(parent.st_mode)
                or parent.st_uid != os.getuid()
            ):
                raise OSError
            path.mkdir(mode=0o700)
            path.chmod(0o700)
            status = path.lstat()
        except OSError:
            raise SubmissionError("submission state directory is unsafe") from None
    except OSError:
        raise SubmissionError("submission state directory is unsafe") from None
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
    ):
        raise SubmissionError("submission state directory is unsafe")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino)
        ):
            raise OSError
        os.fchmod(descriptor, 0o700)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
            raise OSError
    except OSError:
        raise SubmissionError("submission state directory is unsafe") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short write")
        remaining = remaining[written:]


def _publish_private(path: Path, content: bytes) -> None:
    if len(content) > max(MAX_SCRIPT_BYTES, MAX_RECEIPT_CHARS):
        raise OSError("content too large")
    directory_fd: int | None = None
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        directory_fd = os.open(path.parent, _DIRECTORY_FLAGS)
        parent_identity = directory_identity(directory_fd)
        expected = capture_file(directory_fd, path.name, exact_mode=0o600)
        temporary_name = f".{path.name}.edullm-{secrets.token_hex(12)}"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        prepared = capture_file(directory_fd, temporary_name, exact_mode=0o600)
        if prepared is None:
            raise OSError("prepared file disappeared")
        publishing_name = temporary_name
        temporary_name = None
        compare_and_publish(
            directory_fd,
            path.parent,
            parent_identity,
            path.name,
            publishing_name,
            expected,
            prepared,
        )
    except (OSError, SecurePublishError):
        raise OSError("private publication failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)


def stage_submission(state_root: Path, key: str, script: str) -> Path:
    """Atomically stage one private rendered script below a fixed state root."""
    if (
        not isinstance(state_root, Path)
        or not state_root.is_absolute()
        or type(key) is not str
        or _SHA256.fullmatch(key) is None
        or type(script) is not str
        or not script.startswith("#!/bin/bash\n")
        or len(script.encode("utf-8")) > MAX_SCRIPT_BYTES
    ):
        raise SubmissionError("submission staging input is invalid")
    try:
        _ensure_private_directory(state_root)
        directory = state_root / key
        _ensure_private_directory(directory)
        path = directory / "request.sbatch"
        _publish_private(path, script.encode("utf-8"))
        status = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.getuid()
            or status.st_nlink != 1
            or stat.S_IMODE(status.st_mode) != 0o600
        ):
            raise SubmissionError("staged submission script is unsafe")
        return path
    except SubmissionError:
        raise
    except OSError:
        raise SubmissionError("submission script could not be staged safely") from None


def _read_private(path: Path, *, maximum: int) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError:
        raise SubmissionError("private submission state is unsafe") from None
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.getuid()
            or status.st_nlink != 1
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_size > maximum
        ):
            raise SubmissionError("private submission state is unsafe")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > maximum:
            raise SubmissionError("private submission state is unsafe")
        return content
    finally:
        os.close(descriptor)


def _receipt_matches_spec(receipt: SubmissionReceipt, spec: SubmissionSpec) -> bool:
    return (
        receipt.issue == spec.issue
        and receipt.request_digest == spec.request_digest
        and receipt.attempt_number == spec.attempt_number
        and receipt.operator == spec.operator
        and receipt.remote_user == spec.remote_user
        and receipt.script_sha256 == spec.script_sha256
        and receipt.manifest_sha256 == spec.manifest_sha256
    )


def _remove_intent(path: Path) -> None:
    try:
        path.unlink()
        descriptor = os.open(path.parent, _DIRECTORY_FLAGS)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise SubmissionError("failed submission intent could not be cleared safely") from None


def _submission_job_name(key: str) -> str:
    _safe_text(key, _SHA256, "submission key")
    return f"edullm-{key}"


def _discover_submission_receipt(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    *,
    key: str,
    spec: SubmissionSpec,
    cwd: Path,
) -> SubmissionReceipt | None:
    name = _submission_job_name(key)
    comment = f"edullm:{key}"
    argv = [
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        f"--name={name}",
        "--format=JobIDRaw,JobName,State,User,Comment,Submit",
    ]
    try:
        result = runner(
            argv,
            check=False,
            text=True,
            capture_output=True,
            timeout=SBATCH_TIMEOUT_SECONDS,
            cwd=str(cwd),
        )
    except (OSError, subprocess.SubprocessError):
        raise SubmissionError("submission recovery evidence is unavailable") from None
    if (
        not isinstance(result, subprocess.CompletedProcess)
        or type(result.returncode) is not int
        or type(result.stdout) is not str
        or type(result.stderr) is not str
        or result.returncode != 0
        or len(result.stdout) > MAX_RECOVERY_OUTPUT_CHARS
    ):
        raise SubmissionError("submission recovery evidence is unavailable")
    lines = [line for line in result.stdout.splitlines() if line]
    if not lines:
        return None
    if len(lines) != 1:
        raise SubmissionError("submission recovery evidence is ambiguous")
    fields = lines[0].split("|")
    if len(fields) != 6:
        raise SubmissionError("submission recovery evidence is invalid")
    job_id, job_name, raw_state, user, job_comment, submitted = fields
    state = raw_state.partition("+")[0].partition(" ")[0]
    if (
        _JOB_ID.fullmatch(job_id) is None
        or job_name != name
        or state not in _RECOVERABLE_SLURM_STATES
        or user != spec.remote_user
        or job_comment != comment
    ):
        raise SubmissionError("submission recovery evidence is invalid")
    if not submitted.endswith("Z"):
        submitted += "Z"
    submitted_at = _parse_time(submitted)
    return SubmissionReceipt(
        issue=spec.issue,
        request_digest=spec.request_digest,
        attempt_number=spec.attempt_number,
        operator=spec.operator,
        remote_user=spec.remote_user,
        script_sha256=spec.script_sha256,
        manifest_sha256=spec.manifest_sha256,
        slurm_job_id=job_id,
        log_path=str(cwd / spec.log_pattern.replace("%j", job_id)),
        submitted_at=submitted_at,
    )


def _effective_remote_user() -> str:
    try:
        return pwd.getpwuid(os.geteuid()).pw_name
    except (KeyError, OSError):
        raise SubmissionError("authenticated remote user is unavailable") from None


def submission_transaction(
    state_root: Path,
    key: str,
    spec: SubmissionSpec,
    *,
    sbatch_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    manifest_verifier: Callable[..., object] = verify_manifest,
    remote_user_getter: Callable[[], str] = _effective_remote_user,
    now: datetime | None = None,
) -> SubmissionReceipt:
    """
    Submit exactly once under a private filesystem lock and durable receipt.

    A durable intent is fsynced before ``sbatch``. If output or receipt
    persistence is ambiguous, retries fail closed instead of submitting again.
    """
    del manifest_verifier  # Remote preparation verifies the manifest before the final GitHub gate.
    _validate_spec(spec)
    try:
        authenticated_remote_user = remote_user_getter()
    except SubmissionError:
        raise
    except Exception:
        raise SubmissionError("authenticated remote user is unavailable") from None
    if (
        type(authenticated_remote_user) is not str
        or _REMOTE_USER.fullmatch(authenticated_remote_user) is None
        or authenticated_remote_user != spec.remote_user
    ):
        raise SubmissionError("authenticated remote user does not match submission identity")
    expected_key = build_submission_key(
        spec.issue,
        spec.request_digest,
        spec.attempt_number,
        spec.operator,
    )
    if key != expected_key:
        raise SubmissionError("submission receipt/key does not match its identity")
    _ensure_private_directory(state_root)
    _ensure_private_directory(state_root.parent / "logs")
    directory = state_root / key
    _ensure_private_directory(directory)
    lock_path = directory / "transaction.lock"
    try:
        lock_fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError:
        raise SubmissionError("submission lock is unsafe") from None
    try:
        os.fchmod(lock_fd, 0o600)
        lock_status = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_status.st_mode) or lock_status.st_uid != os.getuid():
            raise SubmissionError("submission lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        receipt_path = directory / "receipt.json"
        receipt_bytes = _read_private(receipt_path, maximum=MAX_RECEIPT_CHARS)
        if receipt_bytes is not None:
            try:
                receipt = parse_submission_receipt(receipt_bytes.decode("utf-8"))
            except (UnicodeDecodeError, SubmissionError):
                raise SubmissionError("submission receipt is malformed or tampered") from None
            if not _receipt_matches_spec(receipt, spec):
                raise SubmissionError("submission receipt does not match this attempt")
            return receipt

        script_path = directory / "request.sbatch"
        script = _read_private(script_path, maximum=MAX_SCRIPT_BYTES)
        if script is None or hashlib.sha256(script).hexdigest() != spec.script_sha256:
            raise SubmissionError("staged submission script identity is invalid")

        intent_path = directory / "intent.json"
        intent_bytes = _read_private(intent_path, maximum=MAX_RECEIPT_CHARS)
        if intent_bytes is not None:
            if intent_bytes != spec.canonical_json().encode("utf-8"):
                raise SubmissionError("submission intent is malformed or tampered")
            recovered = _discover_submission_receipt(
                sbatch_runner,
                key=key,
                spec=spec,
                cwd=state_root.parent,
            )
            if recovered is None:
                raise SubmissionError(
                    "submission outcome is ambiguous; operator repair is required"
                )
            try:
                _publish_private(receipt_path, recovered.canonical_json().encode("utf-8"))
            except OSError:
                raise SubmissionError(
                    "submission receipt persistence failed; operator repair is required"
                ) from None
            persisted = _read_private(receipt_path, maximum=MAX_RECEIPT_CHARS)
            if persisted != recovered.canonical_json().encode("utf-8"):
                raise SubmissionError(
                    "submission receipt persistence failed; operator repair is required"
                )
            return recovered

        try:
            _publish_private(intent_path, spec.canonical_json().encode("utf-8"))
        except OSError:
            raise SubmissionError("submission intent could not be persisted") from None

        argv = [
            "sbatch",
            "--export=NONE",
            "--parsable",
            f"--job-name={_submission_job_name(key)}",
            f"--comment=edullm:{key}",
            str(script_path),
        ]
        try:
            result = sbatch_runner(
                argv,
                check=False,
                text=True,
                capture_output=True,
                timeout=SBATCH_TIMEOUT_SECONDS,
                cwd=str(state_root.parent),
            )
        except (OSError, subprocess.SubprocessError):
            raise SubmissionError(
                "submission outcome is ambiguous; operator repair is required"
            ) from None
        if (
            not isinstance(result, subprocess.CompletedProcess)
            or type(result.returncode) is not int
            or type(result.stdout) is not str
            or type(result.stderr) is not str
        ):
            raise SubmissionError("submission outcome is ambiguous; operator repair is required")
        if result.returncode != 0:
            _remove_intent(intent_path)
            raise SubmissionError("Slurm submission failed")
        if len(result.stdout) > MAX_SBATCH_OUTPUT_CHARS:
            raise SubmissionError("submission outcome is ambiguous; operator repair is required")
        output = result.stdout.strip()
        if _JOB_ID.fullmatch(output) is None or result.stdout not in {output, output + "\n"}:
            raise SubmissionError("submission outcome is ambiguous; operator repair is required")

        submitted_at = _validate_time(now or datetime.now(timezone.utc).replace(microsecond=0))
        expanded = spec.log_pattern.replace("%j", output)
        log_path = str(state_root.parent / expanded)
        receipt = SubmissionReceipt(
            issue=spec.issue,
            request_digest=spec.request_digest,
            attempt_number=spec.attempt_number,
            operator=spec.operator,
            remote_user=spec.remote_user,
            script_sha256=spec.script_sha256,
            manifest_sha256=spec.manifest_sha256,
            slurm_job_id=output,
            log_path=log_path,
            submitted_at=submitted_at,
        )
        try:
            _publish_private(receipt_path, receipt.canonical_json().encode("utf-8"))
        except OSError:
            raise SubmissionError(
                "submission receipt persistence failed; operator repair is required"
            ) from None
        persisted = _read_private(receipt_path, maximum=MAX_RECEIPT_CHARS)
        if persisted != receipt.canonical_json().encode("utf-8"):
            raise SubmissionError(
                "submission receipt persistence failed; operator repair is required"
            )
        return receipt
    finally:
        os.close(lock_fd)


class SSHSubmissionRemote:
    """Production SSH adapter for staging, verifying, and reconciling submissions."""

    def __init__(
        self,
        ssh_client: Any | None = None,
        *,
        remote_user: str | None = None,
        local_recovery_root: Path | None = None,
    ) -> None:
        from edullm.ssh import SSHClient

        self.ssh_client = SSHClient() if ssh_client is None else ssh_client
        self.remote_user = (
            None if remote_user is None else _safe_text(remote_user, _REMOTE_USER, "remote user")
        )
        self.local_recovery_root = (
            Path.home() / ".config" / "edullm" / "recovery"
            if local_recovery_root is None
            else local_recovery_root
        )

    def _validate_identity(self, spec: SubmissionSpec) -> None:
        if self.remote_user is None or spec.remote_user != self.remote_user:
            raise SubmissionError("remote submission user identity is invalid")

    def stage(self, key: str, script: str, spec: SubmissionSpec) -> None:
        """Upload one private script through the constrained stdin helper."""
        _validate_spec(spec)
        self._validate_identity(spec)
        expected = build_submission_key(
            spec.issue,
            spec.request_digest,
            spec.attempt_number,
            spec.operator,
        )
        if (
            key != expected
            or hashlib.sha256(script.encode("utf-8")).hexdigest() != spec.script_sha256
        ):
            raise SubmissionError("remote staging identity is invalid")
        try:
            self.ssh_client.write_remote(
                f"submission/{key}/request.sbatch",
                script,
                timeout=SBATCH_TIMEOUT_SECONDS,
            )
        except Exception:
            raise SubmissionError("remote submission staging failed") from None

    @staticmethod
    def _environment_command(arguments: Sequence[str]) -> str:
        return 'set -euo pipefail\nsource "$HOME/venvs/edullm/bin/activate"\n' + shlex.join(
            arguments
        )

    def verify_manifest(self, spec: SubmissionSpec) -> None:
        """Reverify the complete remote manifest immediately before submission."""
        _validate_spec(spec)
        self._validate_identity(spec)
        arguments = [
            "python",
            "-m",
            "edullm.data_manifest",
            "verify",
            spec.manifest_uri,
            spec.manifest_sha256,
        ]
        for kind in spec.allowed_data_kinds:
            arguments.extend(["--allowed-kind", kind])
        try:
            result = self.ssh_client.run_remote(
                ["bash", "-lc", self._environment_command(arguments)],
                check=False,
                timeout=SBATCH_TIMEOUT_SECONDS,
            )
        except Exception:
            raise SubmissionError("remote data manifest verification failed") from None
        if getattr(result, "returncode", None) != 0:
            raise SubmissionError("remote data manifest verification failed")

    def submit(self, key: str, spec: SubmissionSpec) -> SubmissionReceipt:
        """Run or reconcile the single remote transaction and save recovery state."""
        _validate_spec(spec)
        self._validate_identity(spec)
        expected = build_submission_key(
            spec.issue,
            spec.request_digest,
            spec.attempt_number,
            spec.operator,
        )
        if key != expected:
            raise SubmissionError("remote transaction identity is invalid")
        spec_b64 = base64.b64encode(spec.canonical_json().encode("utf-8")).decode("ascii")
        command = (
            'set -euo pipefail\nsource "$HOME/venvs/edullm/bin/activate"\n'
            "python -m edullm.slurm transaction "
            '--state-root "$HOME/orcd/scratch/edullm/state" '
            f"--key {key} --spec-b64 {spec_b64}"
        )
        try:
            result = self.ssh_client.run_remote(
                ["bash", "-lc", command],
                check=False,
                timeout=SBATCH_TIMEOUT_SECONDS * 2,
            )
        except Exception:
            raise SubmissionError(
                "remote submission outcome is unknown; retry to reconcile"
            ) from None
        if (
            getattr(result, "returncode", None) != 0
            or type(getattr(result, "stdout", None)) is not str
            or len(result.stdout) > MAX_RECEIPT_CHARS + 1
        ):
            raise SubmissionError("remote submission transaction failed")
        output = result.stdout
        if not output.endswith("\n") or output.count("\n") != 1:
            raise SubmissionError("remote submission receipt is malformed")
        receipt = parse_submission_receipt(output[:-1])
        if not _receipt_matches_spec(receipt, spec):
            raise SubmissionError("remote submission receipt does not match this attempt")
        receipt_path = PurePosixPath(receipt.log_path)
        if tuple(receipt_path.parts[-5:-1]) != ("orcd", "scratch", "edullm", "logs"):
            raise SubmissionError("remote submission receipt log path is invalid")
        try:
            _ensure_private_directory(self.local_recovery_root)
            _publish_private(
                self.local_recovery_root / f"{key}.json",
                receipt.canonical_json().encode("utf-8"),
            )
        except (OSError, SubmissionError):
            raise SubmissionError("local recovery receipt could not be persisted") from None
        return receipt


def submit(
    resolved: ResolvedRequest,
    *,
    ssh_client: Any | None = None,
    remote_user: str | None = None,
    local_recovery_root: Path | None = None,
) -> str:
    """
    Stage, remotely reverify, and idempotently submit one resolved request.

    Operator orchestration normally uses :class:`SSHSubmissionRemote` directly
    so its mandatory second GitHub gate can run between ``stage`` and ``submit``.
    """
    attempt, _ = _validate_resolved(resolved)
    script = render_sbatch(resolved)
    request = resolved.request
    spec = SubmissionSpec(
        issue=request.issue_number,
        request_digest=request.digest,
        attempt_number=attempt,
        operator=resolved.operator,
        remote_user=cast(str, remote_user),
        script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
        manifest_uri=request.data_manifest,
        manifest_sha256=request.data_manifest_sha256,
        allowed_data_kinds=resolved.allowed_data_kinds,
        log_pattern=resolved.log_pattern,
    )
    key = build_submission_key(
        spec.issue,
        spec.request_digest,
        spec.attempt_number,
        spec.operator,
    )
    remote = SSHSubmissionRemote(
        ssh_client,
        remote_user=remote_user,
        local_recovery_root=local_recovery_root,
    )
    remote.stage(key, script, spec)
    remote.verify_manifest(spec)
    return remote.submit(key, spec).slurm_job_id


def _spec_from_payload(payload: object) -> SubmissionSpec:
    if type(payload) is not dict or set(payload) != _SUBMISSION_FIELDS:
        raise SubmissionError("submission spec is invalid")
    data = cast(dict[str, object], payload)
    kinds = data["allowed_data_kinds"]
    if type(kinds) is not list or any(type(kind) is not str for kind in kinds):
        raise SubmissionError("submission spec is invalid")
    spec = SubmissionSpec(
        issue=cast(int, data["issue"]),
        request_digest=cast(str, data["request_digest"]),
        attempt_number=cast(int, data["attempt_number"]),
        operator=cast(str, data["operator"]),
        remote_user=cast(str, data["remote_user"]),
        script_sha256=cast(str, data["script_sha256"]),
        manifest_uri=cast(str, data["manifest_uri"]),
        manifest_sha256=cast(str, data["manifest_sha256"]),
        allowed_data_kinds=tuple(cast(list[str], kinds)),
        log_pattern=cast(str, data["log_pattern"]),
    )
    _validate_spec(spec)
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed remote transaction entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m edullm.slurm")
    commands = parser.add_subparsers(dest="command", required=True)
    transaction = commands.add_parser("transaction")
    transaction.add_argument("--state-root", required=True)
    transaction.add_argument("--key", required=True)
    transaction.add_argument("--spec-b64", required=True)
    arguments = parser.parse_args(argv)
    try:
        encoded = base64.b64decode(arguments.spec_b64, validate=True)
        if len(encoded) > MAX_RECEIPT_CHARS:
            raise SubmissionError("submission spec is invalid")
        text = encoded.decode("utf-8")
        payload = json.loads(text)
        if text != json.dumps(payload, sort_keys=True, separators=(",", ":")):
            raise SubmissionError("submission spec is invalid")
        spec = _spec_from_payload(payload)
        receipt = submission_transaction(
            Path(arguments.state_root),
            arguments.key,
            spec,
        )
    except (OSError, UnicodeDecodeError, ValueError, SubmissionError):
        print("submission transaction failed", file=sys.stderr)
        return 1
    print(receipt.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
