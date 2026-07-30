"""
Operator and internal automation CLI for the eduLLM ORCD job pool.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

from edullm.assignment import (
    AUTOMATION_ACTOR,
    AssignmentResult,
    assign_ready_issues,
    process_assignment_timeouts,
)
from edullm.automation import AutomationResult, validate_issue
from edullm.github import GitHubClient, GitHubError
from edullm.jobs import (
    GateConfiguration,
    JobOperationError,
    OperatorJobSummary,
    SSHSlurm,
    deliver_terminal_notifications,
)
from edullm.jobs import jobs as list_operator_jobs
from edullm.jobs import load_gate_configuration
from edullm.jobs import logs as read_operator_logs
from edullm.jobs import run_assigned
from edullm.jobs import stop as stop_operator_job
from edullm.notifications import SlackNotifier
from edullm.policy import load_operators, load_policy
from edullm.secure_publish import SecurePublishError, capture_file, compare_and_publish
from edullm.slurm import SSHSubmissionRemote
from edullm.ssh import (
    COMMAND_TIMEOUT_SECONDS,
    SSHClient,
    SSHConfigError,
    SSHError,
    SSHSessionError,
    apply_control_config,
    plan_control_config,
    read_control_config,
)

ValidationRunner = Callable[..., AutomationResult]
AssignmentRunner = Callable[..., tuple[AssignmentResult, ...]]
TerminalRunner = Callable[..., tuple[Any, ...]]
LocalRunner = Callable[..., subprocess.CompletedProcess[str]]

SETUP_POLL_INTERVAL_SECONDS = 5.0
SETUP_POLL_TIMEOUT_SECONDS = 3600.0
REMOTE_REPO_ROOT = "$HOME/OLMo-core"
REMOTE_SCRATCH = "$HOME/orcd/scratch/edullm"
_CANONICAL_REPOSITORY = "edu-llm/OLMo-core"
_SSH_CONFIGURATION_PLANNING_ERROR = "operator setup failed while planning SSH configuration"
_SSH_SESSION_RECOVERY = "ORCD SSH login failed; run ssh -MNf orcd-login and retry."
_GITHUB_LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_ORCD_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"([0-9]+)(?:;[A-Za-z0-9._-]+)?\Z")
_TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)

SCRATCH_CHECK_SCRIPT = r"""set -euo pipefail
EDULLM_SCRATCH_ROOT="$HOME/orcd/scratch"
EDULLM_SCRATCH="$HOME/orcd/scratch/edullm"
test -d "$EDULLM_SCRATCH_ROOT"
RESOLVED_SCRATCH_ROOT="$(realpath -e "$EDULLM_SCRATCH_ROOT")"
RESOLVED_SCRATCH="$(realpath -m "$EDULLM_SCRATCH")"
case "$RESOLVED_SCRATCH/" in
  "$RESOLVED_SCRATCH_ROOT/"*) ;;
  *) exit 2 ;;
esac
SCRATCH_MOUNT="$(findmnt -n -o TARGET -T "$RESOLVED_SCRATCH_ROOT")"
HOME_MOUNT="$(findmnt -n -o TARGET -T "$HOME")"
test -n "$SCRATCH_MOUNT"
test "$SCRATCH_MOUNT" != "$HOME_MOUNT"
mkdir -p "$EDULLM_SCRATCH"
RESOLVED_CREATED_SCRATCH="$(realpath -e "$EDULLM_SCRATCH")"
case "$RESOLVED_CREATED_SCRATCH/" in
  "$RESOLVED_SCRATCH_ROOT/"*) ;;
  *) exit 2 ;;
esac
test -w "$EDULLM_SCRATCH"
SCRATCH_PROBE="$(mktemp "$EDULLM_SCRATCH/.edullm-preflight.XXXXXX")"
trap 'rm -f "$SCRATCH_PROBE"' EXIT
printf '%s\n' edullm-probe > "$SCRATCH_PROBE"
rm -f "$SCRATCH_PROBE"
trap - EXIT"""

ENVIRONMENT_CHECK_SCRIPT = r"""set -euo pipefail
EDULLM_REPO_ROOT="$HOME/OLMo-core"
EDULLM_VENV="$HOME/venvs/edullm"
test -x "$EDULLM_VENV/bin/python"
"$EDULLM_VENV/bin/python" -c "import edullm.ssh_helper"
test -f "$EDULLM_VENV/.edullm-commit"
test "$(cat "$EDULLM_VENV/.edullm-commit")" = \
  "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" """

SUBMIT_ENV_SCRIPT = r"""set -euo pipefail
EDULLM_REPO_ROOT="$HOME/OLMo-core"
EDULLM_SCRATCH="$HOME/orcd/scratch/edullm"
SETUP_SCRIPT="$HOME/OLMo-core/src/scripts/orcd/setup_env.sbatch"
test -f "$SETUP_SCRIPT"
EDULLM_COMMIT_SHA="$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)"
case "$EDULLM_COMMIT_SHA" in
  ""|*[!0-9a-f]*) exit 2 ;;
esac
test "${#EDULLM_COMMIT_SHA}" -eq 40
if ! EDULLM_GIT_STATUS="$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"; then
  exit 2
fi
test -z "$EDULLM_GIT_STATUS"
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --parsable \
  --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH" \
  "$HOME/OLMo-core/src/scripts/orcd/setup_env.sbatch" """

IMPORT_CHECK_SCRIPT = r"""set -euo pipefail
source "$HOME/venvs/edullm/bin/activate"
python -c "import torch, wandb, olmo_core, edullm.ssh_helper" """

FINGERPRINT_SCRIPT = r"""set -euo pipefail
source "$HOME/venvs/edullm/bin/activate"
python -m pip freeze --all | LC_ALL=C sort | sha256sum"""

REMOTE_KEY_CHECK_SCRIPT = r"""set -euo pipefail
test -s "$HOME/.config/edullm/wandb.key"
test "$(stat -c '%a' "$HOME/.config/edullm/wandb.key")" = 600"""

REMOTE_WANDB_CHECK_SCRIPT = r"""set -euo pipefail
source "$HOME/venvs/edullm/bin/activate"
source "$HOME/.config/edullm/wandb.env"
python -c "import sys, wandb; api=wandb.Api(timeout=10); username=api.viewer.username; projects={project.name for project in api.projects(entity='eduLLM')}; 'test' in projects or sys.exit(2); print(username)" """

WANDB_ENV = (
    'export WANDB_API_KEY="$(cat "$HOME/.config/edullm/wandb.key")"\n'
    'export WANDB_ENTITY="eduLLM"\n'
    'export WANDB_PROJECT="test"\n'
)


class SetupError(RuntimeError):
    """A sanitized operator setup failure."""


class SetupDeclined(SetupError):
    """The operator declined the displayed SSH configuration change."""


class _MissingWandbDependency(SetupError):
    pass


@dataclass(frozen=True)
class SetupResult:
    """Public, secret-free values recorded by successful operator setup."""

    github: str
    wandb_username: str
    environment_fingerprint: str


@dataclass(frozen=True)
class OperatorServices:
    """Authenticated local operator dependencies for one focused CLI action."""

    operator: str
    remote_user: str
    github: Any
    root: Path
    remote: Any
    slurm: Any

    def load_configuration(self) -> GateConfiguration:
        """Reload all protected controls for each mandatory gate."""
        return load_gate_configuration(self.root)


def _default_wandb_api() -> Any:
    try:
        import wandb
    except ImportError:
        raise _MissingWandbDependency from None

    return wandb.Api(timeout=10)


def _default_confirm(prompt: str) -> bool:
    return input(prompt).strip().casefold() == "yes"


@dataclass
class SetupDependencies:
    """Injectable local, remote, API, prompt, and clock boundaries for setup."""

    local_runner: LocalRunner = subprocess.run
    ssh_client: Any = field(default_factory=SSHClient)
    wandb_api_factory: Callable[[], Any] = _default_wandb_api
    confirm: Callable[[str], bool] = _default_confirm
    get_secret: Callable[[str], str] = getpass.getpass
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


def setup_operator(
    *,
    root: Path,
    home: Path,
    orcd_username: str,
    dependencies: SetupDependencies | None = None,
    output: TextIO | None = None,
    poll_timeout: float = SETUP_POLL_TIMEOUT_SECONDS,
) -> SetupResult:
    """
    Perform ordered, fail-closed setup for one ORCD operator.

    No service is contacted until this function is explicitly called. All
    subprocess, W&B API, prompt, SSH, and clock boundaries are injectable.

    :param root: Trusted local checkout containing protected operator config.
    :param home: Local operator home directory.
    :param orcd_username: Personal Engaging login.
    :param dependencies: Optional injectable setup boundaries.
    :param output: Destination for the exact redacted SSH diff.
    :param poll_timeout: Maximum environment-setup polling duration.

    :returns: The public, secret-free setup result.

    :raises SetupError: If any ordered preflight or setup transition fails.
    :raises SSHSessionError: If the project SSH session cannot be established.
    """
    dependencies = dependencies or SetupDependencies()
    output = output or sys.stdout
    if poll_timeout <= 0:
        raise ValueError("poll timeout must be positive")

    _run_local(dependencies.local_runner, ["gh", "auth", "status"], "GitHub authentication")
    github_result = _run_local(
        dependencies.local_runner,
        ["gh", "api", "user", "--jq", ".login"],
        "GitHub identity",
    )
    github_login = github_result.stdout.strip()
    if _GITHUB_LOGIN.fullmatch(github_login) is None:
        raise SetupError("operator setup failed during GitHub identity verification")

    wandb_username = _verify_local_wandb(dependencies.wandb_api_factory)

    ssh_config = home / ".ssh" / "config"
    try:
        original = read_control_config(ssh_config)
        ssh_plan = plan_control_config(original, orcd_username)
    except (SSHConfigError, ValueError):
        raise SetupError(_SSH_CONFIGURATION_PLANNING_ERROR) from None
    if ssh_plan.changed:
        output.write(ssh_plan.redacted_diff)
        output.flush()
        try:
            confirmed = dependencies.confirm(
                "Apply the displayed change to ~/.ssh/config? Type yes: "
            )
        except Exception:
            raise SetupError("operator setup failed during the SSH confirmation prompt") from None
        if not confirmed:
            raise SetupDeclined("SSH configuration change was declined")
        try:
            apply_control_config(ssh_config, ssh_plan)
        except SSHConfigError:
            raise SetupError("operator setup failed while applying SSH configuration") from None

    dependencies.ssh_client.ensure_master()
    for command, label in (
        (["hostname"], "remote hostname verification"),
        (["command", "-v", "sbatch"], "remote sbatch verification"),
        (["command", "-v", "squeue"], "remote squeue verification"),
    ):
        _run_required_remote(dependencies.ssh_client, command, label)
    _run_required_remote(
        dependencies.ssh_client,
        ["bash", "-lc", SCRATCH_CHECK_SCRIPT],
        "Engaging Scratch verification",
    )
    _ensure_environment(dependencies, poll_timeout=poll_timeout)
    _run_required_remote(
        dependencies.ssh_client,
        ["bash", "-lc", IMPORT_CHECK_SCRIPT],
        "remote Python import verification",
    )
    fingerprint = _environment_fingerprint(dependencies.ssh_client)

    try:
        dependencies.ssh_client.write_remote(
            "~/.config/edullm/wandb.env",
            WANDB_ENV,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (SSHError, ValueError):
        raise SetupError("operator setup failed while writing remote W&B environment") from None

    key_exists = _remote_key_exists(dependencies.ssh_client)
    remote_username = (
        _try_remote_wandb(dependencies.ssh_client, expected_failure=key_exists)
        if key_exists
        else None
    )
    if remote_username is None:
        try:
            key = dependencies.get_secret("W&B API key: ")
        except Exception:
            raise SetupError("operator setup failed during the W&B key prompt") from None
        if not key or "\n" in key or "\r" in key:
            raise SetupError("operator setup failed because the W&B key is invalid")
        try:
            dependencies.ssh_client.write_remote(
                "~/.config/edullm/wandb.key",
                key + "\n",
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except (SSHError, ValueError):
            raise SetupError("operator setup failed while writing the remote W&B key") from None
        finally:
            key = ""
        remote_username = _try_remote_wandb(dependencies.ssh_client, expected_failure=False)
    if remote_username != wandb_username:
        raise SetupError("operator setup failed because the remote W&B identity does not match")

    try:
        operators = load_operators(root / "config" / "edullm" / "operators.yaml")
    except (OSError, ValueError):
        raise SetupError("operator setup failed while reading protected operators") from None
    if not any(operator.github == github_login for operator in operators):
        raise SetupError("GitHub login is not present in the protected operator roster")

    config = {
        "environment_fingerprint": fingerprint,
        "github": github_login,
        "orcd_username": orcd_username,
        "remote_repo_root": REMOTE_REPO_ROOT,
        "scratch": REMOTE_SCRATCH,
        "version": 1,
        "wandb_username": wandb_username,
    }
    write_operator_config(home / ".config" / "edullm" / "config.yaml", config)
    return SetupResult(github_login, wandb_username, fingerprint)


def write_operator_config(path: Path, config: Mapping[str, object]) -> None:
    """
    Atomically write a secret-free operator config with mode ``0600``.

    :param path: Local operator config path.
    :param config: Public setup values to serialize.

    :raises SetupError: If path safety, permissions, or atomic replacement fail.
    """
    try:
        serialized = yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")
    except yaml.YAMLError:
        raise SetupError("operator config could not be serialized safely") from None
    parent = path.parent
    _reject_operator_symlink_ancestors(parent)
    _ensure_operator_directory(parent)
    directory_fd: int | None = None
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        directory_fd = _open_operator_directory(parent)
        directory_identity = _operator_directory_identity(directory_fd)
        _validate_operator_directory(parent, directory_fd, directory_identity)
        original_state = capture_file(directory_fd, path.name, exact_mode=0o600)
        temporary_name = f".{path.name}.edullm-{secrets.token_hex(8)}"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise SetupError("operator config could not be written safely")
        _write_descriptor(descriptor, serialized)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        prepared = capture_file(directory_fd, temporary_name, exact_mode=0o600)
        if prepared is None:
            raise SetupError("operator config could not be written safely")
        publishing_name = temporary_name
        temporary_name = None
        compare_and_publish(
            directory_fd,
            parent,
            directory_identity,
            path.name,
            publishing_name,
            original_state,
            prepared,
        )
    except SetupError:
        raise
    except SecurePublishError:
        raise SetupError("operator config changed while it was being updated") from None
    except OSError:
        raise SetupError("operator config could not be written safely") from None
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


def _run_local(
    runner: LocalRunner, argv: list[str], label: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            argv,
            check=False,
            text=True,
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        raise SetupError(f"operator setup failed during {label}") from None
    if result.returncode != 0:
        raise SetupError(f"operator setup failed during {label}")
    return result


def _verify_local_wandb(api_factory: Callable[[], Any]) -> str:
    try:
        api = api_factory()
        username = api.viewer.username
        project_names = {project.name for project in api.projects(entity="eduLLM")}
    except _MissingWandbDependency:
        raise
    except Exception:
        raise SetupError("operator setup failed during local W&B verification") from None
    if type(username) is not str or not username.strip():
        raise SetupError("operator setup failed during local W&B identity verification")
    if "test" not in project_names:
        raise SetupError("verified eduLLM/test access is required")
    return username.strip()


def _run_required_remote(
    ssh_client: Any, argv: list[str], label: str
) -> subprocess.CompletedProcess[str]:
    try:
        return ssh_client.run_remote(
            argv,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (SSHError, ValueError):
        raise SetupError(f"operator setup failed during {label}") from None


def _ensure_environment(dependencies: SetupDependencies, *, poll_timeout: float) -> None:
    try:
        existing = dependencies.ssh_client.run_remote(
            ["bash", "-lc", ENVIRONMENT_CHECK_SCRIPT],
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (SSHError, ValueError):
        raise SetupError("operator setup failed while checking the remote environment") from None
    if existing.returncode == 0:
        return
    if existing.returncode != 1:
        raise SetupError("operator setup failed while checking the remote environment")

    submitted = _run_required_remote(
        dependencies.ssh_client,
        ["bash", "-lc", SUBMIT_ENV_SCRIPT],
        "environment setup submission",
    )
    match = _JOB_ID.fullmatch(submitted.stdout.strip())
    if match is None:
        raise SetupError("operator setup failed because sbatch returned an invalid job ID")
    job_id = match.group(1)
    deadline = dependencies.monotonic() + poll_timeout
    while True:
        if dependencies.monotonic() >= deadline:
            raise SetupError("environment setup timed out")
        state_result = _run_required_remote(
            dependencies.ssh_client,
            ["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=State%32"],
            "environment setup status",
        )
        state = _parse_slurm_state(state_result.stdout)
        if state in _TERMINAL_SLURM_STATES:
            if state != "COMPLETED":
                raise SetupError("environment setup job did not complete successfully")
            return
        dependencies.sleep(
            min(SETUP_POLL_INTERVAL_SECONDS, max(0.0, deadline - dependencies.monotonic()))
        )


def _parse_slurm_state(output: str) -> str:
    states = []
    for line in output.splitlines():
        value = line.partition("|")[0].strip()
        if not value:
            continue
        states.append(value.split(maxsplit=1)[0])
    if not states:
        return ""
    return states[0]


def _environment_fingerprint(ssh_client: Any) -> str:
    result = _run_required_remote(
        ssh_client,
        ["bash", "-lc", FINGERPRINT_SCRIPT],
        "environment fingerprint",
    )
    fingerprint = result.stdout.strip().partition(" ")[0]
    if _FINGERPRINT.fullmatch(fingerprint) is None:
        raise SetupError("remote environment fingerprint is invalid")
    return fingerprint


def _remote_key_exists(ssh_client: Any) -> bool:
    try:
        result = ssh_client.run_remote(
            ["bash", "-lc", REMOTE_KEY_CHECK_SCRIPT],
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (SSHError, ValueError):
        raise SetupError("operator setup failed while checking the remote W&B key") from None
    if result.returncode not in {0, 1}:
        raise SetupError("operator setup failed while checking the remote W&B key")
    return result.returncode == 0


def _try_remote_wandb(ssh_client: Any, *, expected_failure: bool) -> str | None:
    try:
        result = ssh_client.run_remote(
            ["bash", "-lc", REMOTE_WANDB_CHECK_SCRIPT],
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (SSHError, ValueError):
        raise SetupError("operator setup failed during remote W&B verification") from None
    if result.returncode == 1 and expected_failure:
        return None
    if result.returncode != 0:
        raise SetupError("operator setup failed during remote W&B verification")
    username = result.stdout.strip()
    if not username or "\n" in username:
        raise SetupError("operator setup failed during remote W&B identity verification")
    return username


def _ensure_operator_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        missing = []
        current = path
        while True:
            try:
                current.lstat()
                break
            except FileNotFoundError:
                missing.append(current)
                current = current.parent
            except OSError:
                raise SetupError("operator config directory could not be created safely") from None
        try:
            for directory in reversed(missing):
                directory.mkdir(mode=0o700)
                directory.chmod(0o700)
            status = path.lstat()
        except OSError:
            raise SetupError("operator config directory could not be created safely") from None
    except OSError:
        raise SetupError("operator config directory could not be inspected safely") from None
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
    ):
        raise SetupError("operator config directory is unsafe")
    try:
        path.chmod(0o700)
    except OSError:
        raise SetupError("operator config directory permissions could not be secured") from None


def _reject_operator_symlink_ancestors(path: Path) -> None:
    current = path
    while current != current.parent:
        try:
            status = current.lstat()
        except FileNotFoundError:
            pass
        except OSError:
            raise SetupError("operator config path could not be inspected safely") from None
        else:
            if stat.S_ISLNK(status.st_mode):
                raise SetupError("operator config path contains a symbolic link")
        current = current.parent


def _open_operator_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
            raise SetupError("operator config directory is unsafe")
        return descriptor
    except SetupError:
        if "descriptor" in locals():
            os.close(descriptor)
        raise
    except OSError:
        raise SetupError("operator config directory could not be opened safely") from None


def _operator_directory_identity(descriptor: int) -> tuple[int, int, int]:
    status = os.fstat(descriptor)
    return status.st_dev, status.st_ino, status.st_uid


def _validate_operator_directory(
    path: Path,
    descriptor: int,
    expected: tuple[int, int, int],
) -> None:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        raise SetupError("operator config changed while it was being updated") from None
    if (
        _operator_directory_identity(descriptor) != expected
        or (status.st_dev, status.st_ino, status.st_uid) != expected
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise SetupError("operator config changed while it was being updated")


def _write_descriptor(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if str(parsed) != value or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _orcd_username(value: str) -> str:
    if _ORCD_USERNAME.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a valid ORCD username")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the complete public and internal eduLLM command parser."""
    parser = argparse.ArgumentParser(prog="edullm")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{setup,jobs,run,logs,stop,logout}",
    )
    setup = commands.add_parser("setup", help="configure this operator")
    setup.add_argument("--orcd-username", type=_orcd_username)
    jobs = commands.add_parser("jobs", help="show job requests")
    jobs.add_argument("--mine", action="store_true", help="show only requests assigned to you")
    commands.add_parser("run", help="run the next assigned request")
    logs = commands.add_parser("logs", help="show logs for an Issue")
    logs.add_argument("issue", type=_positive_integer)
    stop = commands.add_parser("stop", help="stop an Issue job")
    stop.add_argument("issue", type=_positive_integer)
    commands.add_parser("logout", help="close the project SSH session")
    automation = commands.add_parser("automation")
    commands._choices_actions = [
        action for action in commands._choices_actions if action.dest != "automation"
    ]
    automation_commands = automation.add_subparsers(
        dest="automation_command",
        required=True,
    )
    validate = automation_commands.add_parser("validate")
    validate.add_argument("--issue", type=_positive_integer, required=True)
    automation_commands.add_parser("assign")
    automation_commands.add_parser("reminders")
    automation_commands.add_parser("terminal")
    return parser


def _parser() -> argparse.ArgumentParser:
    return build_parser()


def handle_setup(orcd_username: str | None = None) -> int:
    """
    Run focused operator setup handling.

    :param orcd_username: Optional personal Engaging login. Defaults to the local
        operating-system username.
    """
    try:
        result = setup_operator(
            root=Path(__file__).resolve().parents[2],
            home=Path.home(),
            orcd_username=orcd_username or getpass.getuser(),
        )
    except _MissingWandbDependency:
        print(
            "eduLLM operator setup failed: local W&B SDK is unavailable; "
            "run python -m pip install -e '.[wandb]'",
            file=sys.stderr,
        )
        return 1
    except SetupDeclined:
        print("eduLLM operator setup cancelled; no SSH change was applied", file=sys.stderr)
        return 2
    except SSHSessionError:
        print(_SSH_SESSION_RECOVERY, file=sys.stderr)
        return 1
    except SetupError as error:
        if str(error) == _SSH_CONFIGURATION_PLANNING_ERROR:
            print(f"eduLLM {error}", file=sys.stderr)
        else:
            print("eduLLM operator setup failed", file=sys.stderr)
        return 1
    print(f"eduLLM operator setup complete for {result.github}")
    return 0


def _read_operator_document(path: Path) -> object:
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        _reject_operator_symlink_ancestors(path)
        directory_fd = _open_operator_directory(path.parent)
        directory_before = os.fstat(directory_fd)
        if stat.S_IMODE(directory_before.st_mode) != 0o700:
            raise OSError
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > 65_536
        ):
            raise OSError
        content = os.read(descriptor, 65_537)
        after = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        directory_after = os.fstat(directory_fd)

        def identity(status: os.stat_result) -> tuple[int, ...]:
            return (
                status.st_dev,
                status.st_ino,
                status.st_mode,
                status.st_uid,
                status.st_nlink,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )

        if (
            len(content) > 65_536
            or identity(before) != identity(after)
            or identity(after) != identity(named)
            or identity(directory_before) != identity(directory_after)
        ):
            raise OSError
        return yaml.safe_load(content.decode("utf-8"))
    except (OSError, SetupError, UnicodeDecodeError, yaml.YAMLError):
        raise JobOperationError("local operator configuration is unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


def _load_operator_services() -> OperatorServices:
    root = Path(__file__).resolve().parents[2]
    configuration = load_gate_configuration(root)
    enabled = {operator.github for operator in configuration.operators if operator.enabled}
    if not enabled:
        raise JobOperationError("operator execution is disabled by protected configuration")
    path = Path.home() / ".config" / "edullm" / "config.yaml"
    document = _read_operator_document(path)
    expected_fields = {
        "environment_fingerprint",
        "github",
        "orcd_username",
        "remote_repo_root",
        "scratch",
        "version",
        "wandb_username",
    }
    if type(document) is not dict or set(document) != expected_fields:
        raise JobOperationError("local operator configuration is invalid")
    operator = document.get("github")
    if type(operator) is not str or operator not in enabled:
        raise JobOperationError("local identity is not an enabled protected operator")
    remote_user = document.get("orcd_username")
    if type(remote_user) is not str or _ORCD_USERNAME.fullmatch(remote_user) is None:
        raise JobOperationError("local ORCD identity is invalid")
    try:
        login = _run_local(
            subprocess.run,
            ["gh", "api", "user", "--jq", ".login"],
            "GitHub identity",
        ).stdout.strip()
    except SetupError:
        raise JobOperationError("local GitHub identity is unavailable") from None
    if login != operator:
        raise JobOperationError("local GitHub identity does not match operator configuration")
    try:
        token = _run_local(
            subprocess.run,
            ["gh", "auth", "token"],
            "GitHub authentication",
        ).stdout.strip()
    except SetupError:
        raise JobOperationError("local GitHub authentication is unavailable") from None
    try:
        github = GitHubClient(token, "edu-llm/OLMo-core")
    finally:
        token = ""
    try:
        identity = github.get("/user")
        if (
            type(identity) is not dict
            or identity.get("login") != operator
            or identity.get("type") != "User"
        ):
            raise JobOperationError("authenticated GitHub identity does not match the operator")
    except JobOperationError:
        raise
    except Exception:
        raise JobOperationError("authenticated GitHub identity is unavailable") from None
    ssh_client = SSHClient()
    ssh_client.ensure_master()
    return OperatorServices(
        operator=operator,
        remote_user=remote_user,
        github=github,
        root=root,
        remote=SSHSubmissionRemote(ssh_client, remote_user=remote_user),
        slurm=SSHSlurm(ssh_client),
    )


def _format_operator_jobs(
    operator: str,
    summaries: Sequence[OperatorJobSummary],
) -> str:
    """Render validated job summaries as the deterministic operator dashboard."""
    if not summaries:
        return f"No eduLLM jobs assigned to {operator}."

    issue_width = max(3, max(len(str(summary.issue)) for summary in summaries))
    continuation = " " * (issue_width + 4)
    ready = sorted(
        (summary for summary in summaries if summary.status == "assigned"),
        key=lambda summary: summary.issue,
    )
    recent = [summary for summary in summaries if summary.status != "assigned"]
    lines = [
        f"eduLLM jobs for {operator}",
        "",
        f"Ready to run ({len(ready)})",
    ]
    for index, summary in enumerate(ready):
        action = "Next: edullm run" if index == 0 else f"Waiting behind #{ready[0].issue}"
        lines.extend(
            (
                f"  #{summary.issue:<{issue_width}} assigned",
                f"{continuation}{action}",
            )
        )
    lines.extend(("", f"Submitted and recent ({len(recent)})"))
    for summary in recent:
        assert summary.slurm_job_id is not None and summary.wandb_url is not None
        lines.extend(
            (
                f"  #{summary.issue:<{issue_width}} {summary.status:<11} "
                f"Slurm {summary.slurm_job_id}",
                f"{continuation}W&B: {summary.wandb_url}",
            )
        )
    return "\n".join(lines)


def handle_jobs(mine: bool, *, services: OperatorServices | None = None) -> int:
    """List authorized jobs and monotonically repair stale lifecycle state."""
    try:
        services = services or _load_operator_services()
        summaries = list_operator_jobs(
            mine=mine,
            operator=services.operator,
            github=services.github,
            configuration=services.load_configuration(),
            slurm=services.slurm,
            now=datetime.now(timezone.utc).replace(microsecond=0),
        )
    except SSHSessionError:
        print(_SSH_SESSION_RECOVERY, file=sys.stderr)
        return 1
    except (GitHubError, JobOperationError, OSError, ValueError):
        print("edullm jobs failed", file=sys.stderr)
        return 1
    print(_format_operator_jobs(services.operator, summaries))
    return 0


def handle_run(*, services: OperatorServices | None = None) -> int:
    """Submit the next assigned request after both mandatory fresh gates."""
    try:
        services = services or _load_operator_services()
        state = run_assigned(
            operator=services.operator,
            github=services.github,
            load_configuration=services.load_configuration,
            remote=services.remote,
            now=datetime.now(timezone.utc).replace(microsecond=0),
        )
    except SSHSessionError:
        print(_SSH_SESSION_RECOVERY, file=sys.stderr)
        return 1
    except (GitHubError, JobOperationError, OSError, ValueError):
        print("edullm run failed before a safe submission could be confirmed", file=sys.stderr)
        return 1
    attempt = state.attempts[-1]
    print(
        f"Submitted Issue #{state.issue} as Slurm job {attempt.slurm_job_id}. "
        f"W&B: {attempt.wandb_url}"
    )
    return 0


def handle_logs(issue: int, *, services: OperatorServices | None = None) -> int:
    """Print only the authorized canonical bounded redacted Issue log."""
    try:
        services = services or _load_operator_services()
        text = read_operator_logs(
            issue,
            operator=services.operator,
            github=services.github,
            configuration=services.load_configuration(),
            slurm=services.slurm,
        )
    except SSHSessionError:
        print(_SSH_SESSION_RECOVERY, file=sys.stderr)
        return 1
    except (GitHubError, JobOperationError, OSError, ValueError):
        print("edullm logs failed", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def handle_stop(issue: int, *, services: OperatorServices | None = None) -> int:
    """Cancel only the current operator's canonical Issue job binding."""
    try:
        services = services or _load_operator_services()
        state = stop_operator_job(
            issue,
            operator=services.operator,
            github=services.github,
            configuration=services.load_configuration(),
            slurm=services.slurm,
            now=datetime.now(timezone.utc).replace(microsecond=0),
        )
    except SSHSessionError:
        print(_SSH_SESSION_RECOVERY, file=sys.stderr)
        return 1
    except (GitHubError, JobOperationError, OSError, ValueError):
        print("edullm stop failed", file=sys.stderr)
        return 1
    print(f"Issue #{issue}: {state.current_state}")
    return 0


def handle_logout(*, ssh_client: Any | None = None) -> int:
    """Close only the project ControlMaster, tolerating an absent master."""
    client = ssh_client or SSHClient()
    try:
        closed = client.close_master()
    except SSHError:
        print("eduLLM could not close the project SSH session", file=sys.stderr)
        return 1
    if closed:
        print("eduLLM project SSH session closed")
    else:
        print("eduLLM project SSH session was already closed")
    return 0


def automation_validate(
    issue_number: int,
    *,
    token: str,
    repository: str,
    root: Path,
) -> AutomationResult:
    """
    Load tracked controls and validate one GitHub Issue.

    :param issue_number: The positive Issue number.
    :param token: The workflow-provided GitHub token.
    :param repository: The workflow-provided ``owner/name`` repository.
    :param root: The checked-out repository root.

    :returns: The validation automation result.
    """
    if repository != _CANONICAL_REPOSITORY:
        raise ValueError("GitHub repository is not supported")
    config = root / "config/edullm"
    policy = load_policy(
        config / "policy.yaml",
        config / "entrypoints.yaml",
    )
    github = GitHubClient(token, repository)
    validated_at = datetime.now(timezone.utc).replace(microsecond=0)
    return validate_issue(
        issue_number,
        github=github,
        policy=policy,
        validated_at=validated_at,
    )


def automation_assign(
    *,
    token: str,
    repository: str,
    webhook: str | None,
    root: Path,
) -> tuple[AssignmentResult, ...]:
    """
    Load protected controls and scan current ready Issues for assignment.

    :param token: The workflow-provided GitHub token.
    :param repository: The workflow-provided ``owner/name`` repository.
    :param webhook: The optional protected Slack incoming-webhook URL.
    :param root: The checked-out repository root.

    :returns: Sanitized assignment results, or an empty closed-roster no-op.
    """
    config = root / "config/edullm"
    operators = load_operators(config / "operators.yaml")
    if not any(operator.enabled for operator in operators):
        return ()
    if webhook is None:
        raise ValueError("Slack notification configuration is unavailable")
    policy = load_policy(config / "policy.yaml", config / "entrypoints.yaml")
    github = GitHubClient(token, repository)
    notifier = SlackNotifier(webhook)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return assign_ready_issues(
        github=github,
        operators=operators,
        policy=policy,
        now=now,
        notifier=notifier,
        automation_actor=AUTOMATION_ACTOR,
    )


def automation_reminders(
    *,
    token: str,
    repository: str,
    webhook: str | None,
    root: Path,
) -> tuple[AssignmentResult, ...]:
    """
    Load protected controls and scan current assignments for timeouts.

    :param token: The workflow-provided GitHub token.
    :param repository: The workflow-provided ``owner/name`` repository.
    :param webhook: The optional protected Slack incoming-webhook URL.
    :param root: The checked-out repository root.

    :returns: Sanitized reminder and reassignment results.
    """
    config = root / "config/edullm"
    operators = load_operators(config / "operators.yaml")
    if not any(operator.enabled for operator in operators):
        return ()
    if webhook is None:
        raise ValueError("Slack notification configuration is unavailable")
    policy = load_policy(config / "policy.yaml", config / "entrypoints.yaml")
    github = GitHubClient(token, repository)
    notifier = SlackNotifier(webhook)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return process_assignment_timeouts(
        github=github,
        operators=operators,
        policy=policy,
        now=now,
        notifier=notifier,
        automation_actor=AUTOMATION_ACTOR,
    )


def automation_terminal(
    *,
    token: str,
    repository: str,
    webhook: str | None,
    root: Path,
) -> tuple[Any, ...]:
    """Deliver hard-disabled workflow terminal events from canonical lifecycle state."""
    configuration = load_gate_configuration(root.resolve())
    if not any(operator.enabled for operator in configuration.operators):
        return ()
    if webhook is None:
        raise ValueError("Slack notification configuration is unavailable")
    return deliver_terminal_notifications(
        github=GitHubClient(token, repository),
        configuration=configuration,
        notifier=SlackNotifier(webhook),
        now=datetime.now(timezone.utc).replace(microsecond=0),
        automation_actor=AUTOMATION_ACTOR,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    validation_runner: ValidationRunner = automation_validate,
    assignment_runner: AssignmentRunner = automation_assign,
    reminder_runner: AssignmentRunner = automation_reminders,
    terminal_runner: TerminalRunner = automation_terminal,
) -> int:
    """
    Run the ``edullm`` console entry point and internal automation commands.

    Public commands are ``setup``, ``jobs``, ``run``, ``logs``, ``stop``, and
    ``logout``. Internal workflow commands are ``automation validate``,
    ``automation assign``, ``automation reminders``, and ``automation terminal``.

    :param argv: Optional command arguments without the module name.
    :param environ: Optional environment mapping for tests.
    :param validation_runner: Optional validation dependency for tests.
    :param assignment_runner: Optional ready-Issue assignment dependency for tests.
    :param reminder_runner: Optional reminder and reassignment dependency for tests.
    :param terminal_runner: Optional terminal-notification dependency for tests.

    :returns: A process exit status.
    """
    arguments = _parser().parse_args(argv)
    if arguments.command == "setup":
        return handle_setup(arguments.orcd_username)
    if arguments.command == "jobs":
        return handle_jobs(arguments.mine)
    if arguments.command == "run":
        return handle_run()
    if arguments.command == "logs":
        return handle_logs(arguments.issue)
    if arguments.command == "stop":
        return handle_stop(arguments.issue)
    if arguments.command == "logout":
        return handle_logout()

    environment = os.environ if environ is None else environ
    token = environment.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    repository = environment.get("GITHUB_REPOSITORY")
    if not repository:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2

    if arguments.automation_command == "validate":
        try:
            result = validation_runner(
                arguments.issue,
                token=token,
                repository=repository,
                root=Path.cwd(),
            )
        except (GitHubError, OSError, ValueError):
            print(
                "eduLLM validation configuration or GitHub access failed",
                file=sys.stderr,
            )
            return 1

        if result.operational_error:
            print(result.errors[0], file=sys.stderr)
            return 1
        print(f"eduLLM Issue #{arguments.issue}: {result.status}")
        return 0

    webhook = environment.get("SLACK_WEBHOOK_URL")
    if arguments.automation_command == "assign":
        runner: Callable[..., tuple[Any, ...]] = assignment_runner
    elif arguments.automation_command == "reminders":
        runner = reminder_runner
    else:
        runner = terminal_runner
    try:
        results = runner(
            token=token,
            repository=repository,
            webhook=webhook,
            root=Path.cwd(),
        )
    except (GitHubError, OSError, ValueError):
        print("eduLLM automation operation failed", file=sys.stderr)
        return 1
    if arguments.automation_command != "terminal" and any(
        result.operational_error for result in results
    ):
        print("eduLLM automation operation failed", file=sys.stderr)
        return 1
    label = {
        "assign": "assignment",
        "reminders": "reminder",
        "terminal": "terminal notification",
    }[arguments.automation_command]
    print(f"eduLLM {label} scan: {len(results)} result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
