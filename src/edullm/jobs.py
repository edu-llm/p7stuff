"""
Authorize, record, reconcile, inspect, and stop eduLLM Slurm jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from edullm.assignment import (
    ASSIGNMENT_MARKER,
    AUTOMATION_ACTOR,
    parse_assignment_comment,
)
from edullm.automation import validation_decision
from edullm.github import GitHubError, GitHubIssue, IssueComment, normalize_actor_login
from edullm.models import JobRequest, JobStatus, Operator, ResolvedRequest
from edullm.notifications import SlackNotificationError
from edullm.policy import Policy, load_operators, load_policy
from edullm.request_parser import IssueParseError, parse_issue
from edullm.slurm import (
    SubmissionReceipt,
    SubmissionSpec,
    build_submission_key,
    render_sbatch,
)
from edullm.validation import (
    MAX_INTEGER_TOKEN_CHARS,
    STATUS_MARKER,
    validate_request,
    validated_status_for_request,
)

JOB_MARKER = "<!-- edullm-job:v1 -->"
MAX_JOB_COMMENT_CHARS = 65_536
MAX_ATTEMPTS = 32
MAX_SLURM_OUTPUT_CHARS = 1_048_576
MAX_SLURM_JOBS = 256
DEFAULT_LOG_BYTES = 262_144

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOGIN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_REMOTE_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]{0,19}\Z")
_ATTEMPT_ID = re.compile(r"attempt-([1-9][0-9]{0,5})\Z")
_RUN_ID = re.compile(r"issue-([1-9][0-9]*)-attempt-([1-9][0-9]{0,5})-([1-9][0-9]{0,19})\Z")
_SAFE_TEXT = re.compile(r"[^\x00-\x1f\x7f]{1,4096}\Z")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_JOB_MARKER_LINE = re.compile(rf"^{re.escape(JOB_MARKER)}$", re.MULTILINE)
_STATUS_MARKER_LINE = re.compile(rf"^{re.escape(STATUS_MARKER)}$", re.MULTILINE)
_ASSIGNMENT_MARKER_LINE = re.compile(rf"^{re.escape(ASSIGNMENT_MARKER)}$", re.MULTILINE)
_LIFECYCLE_STATES = frozenset(
    {"assigned", "submitted", "running", "completed", "failed", "cancelled", "preempted"}
)
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "preempted"})
_NOTIFICATION_STATUSES = frozenset({"none", "pending", "ambiguous", "sent"})
_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_id",
        "attempt_number",
        "log_path",
        "operator",
        "request_digest",
        "slurm_job_id",
        "state",
        "submitted_at",
        "updated_at",
        "wandb_run_id",
        "wandb_url",
    }
)
_NOTIFICATION_FIELDS = frozenset({"event", "status", "updated_at"})
_JOB_FIELDS = frozenset(
    {
        "assignment_version",
        "attempts",
        "current_state",
        "issue",
        "notification",
        "operator",
        "request_digest",
        "updated_at",
    }
)
_MANAGED_STATUS_LABELS = frozenset(f"status:{status.value}" for status in JobStatus)
_SLURM_TO_LIFECYCLE = {
    "PENDING": "submitted",
    "CONFIGURING": "submitted",
    "SUSPENDED": "submitted",
    "RESIZING": "submitted",
    "REQUEUED": "submitted",
    "REQUEUE_FED": "submitted",
    "REQUEUE_HOLD": "submitted",
    "RUNNING": "running",
    "COMPLETING": "running",
    "STAGE_OUT": "running",
    "COMPLETED": "completed",
    "CANCELLED": "cancelled",
    "PREEMPTED": "preempted",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "REVOKED": "failed",
    "SPECIAL_EXIT": "failed",
    "TIMEOUT": "failed",
}


class JobOperationError(RuntimeError):
    """A sanitized fail-closed lifecycle or operator action failure."""


class _OversizedInteger(ValueError):
    pass


@dataclass(frozen=True)
class NotificationRecord:
    """Honest delivery state for the latest terminal notification."""

    event: str
    status: str
    updated_at: datetime


@dataclass(frozen=True)
class JobAttempt:
    """Public bounded audit fields for one immutable Slurm attempt."""

    attempt_id: str
    attempt_number: int
    request_digest: str
    operator: str
    slurm_job_id: str
    wandb_run_id: str
    wandb_url: str
    log_path: str
    state: str
    submitted_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class LifecycleState:
    """Canonical public lifecycle state kept in one assigned-operator-owned comment."""

    issue: int
    request_digest: str
    operator: str
    assignment_version: int
    attempts: tuple[JobAttempt, ...]
    current_state: str
    updated_at: datetime
    notification: NotificationRecord


@dataclass(frozen=True)
class OperatorJobSummary:
    """One validated operator-facing queue record."""

    issue: int
    status: str
    slurm_job_id: str | None
    wandb_url: str | None

    def __post_init__(self) -> None:
        if (
            type(self.issue) is not int
            or self.issue <= 0
            or type(self.status) is not str
            or self.status not in _LIFECYCLE_STATES
        ):
            raise JobOperationError("operator job summary is invalid")
        if self.status == "assigned":
            if self.slurm_job_id is not None or self.wandb_url is not None:
                raise JobOperationError("assigned job summary is invalid")
            return
        if (
            type(self.slurm_job_id) is not str
            or _JOB_ID.fullmatch(self.slurm_job_id) is None
            or type(self.wandb_url) is not str
            or not _wandb_project(self.wandb_url)
        ):
            raise JobOperationError("Slurm-backed job summary is invalid")
        run = _RUN_ID.fullmatch(self.wandb_url.rpartition("/")[2])
        if run is None or int(run.group(1)) != self.issue or run.group(3) != self.slurm_job_id:
            raise JobOperationError("operator job summary binding is invalid")


@dataclass(frozen=True)
class GateConfiguration:
    """Strict protected controls and their immutable evidence digest."""

    policy: Policy
    operators: tuple[Operator, ...]
    digest: str


@dataclass(frozen=True)
class SubmissionGateSnapshot:
    """All fresh evidence that must remain identical across both run gates."""

    issue: int
    request: JobRequest
    request_digest: str
    operator: str
    validated_at: datetime
    status_comment_id: int
    assignment_comment_id: int
    assignment_binding: str
    assignment_version: int
    config_digest: str
    lifecycle: LifecycleState | None
    profile: Mapping[str, object]
    repository_url: str
    scratch_root: str
    slurm_partition: str
    slurm_memory: str
    slurm_cpus_per_gpu: int


@dataclass(frozen=True)
class SlurmJob:
    """One strictly parsed scheduler row."""

    job_id: str
    name: str
    state: str
    user: str
    lifecycle_state: str


def _capture_protected_file(path: Path) -> tuple[tuple[int, int, int, int, int], bytes]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise JobOperationError("protected configuration cannot be read safely") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1_048_576:
            raise JobOperationError("protected configuration is unsafe")
        chunks: list[bytes] = []
        remaining = 1_048_577
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(content) > 1_048_576 or identity != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise JobOperationError("protected configuration changed while it was read")
        return identity, content
    finally:
        os.close(descriptor)


def load_gate_configuration(root: Path) -> GateConfiguration:
    """
    Strictly load every protected run-gate file and bind it to one digest.

    Files are captured before and after the existing strict schema loaders run;
    any symlink, byte edit, or identity swap aborts.
    """
    if not isinstance(root, Path) or not root.is_absolute():
        raise JobOperationError("protected configuration root is invalid")
    config = root / "config" / "edullm"
    names = ("policy.yaml", "entrypoints.yaml", "operators.yaml")
    paths = {name: config / name for name in names}
    before = {name: _capture_protected_file(path) for name, path in paths.items()}
    try:
        policy = load_policy(paths["policy.yaml"], paths["entrypoints.yaml"])
        operators = load_operators(paths["operators.yaml"])
    except (OSError, ValueError):
        raise JobOperationError("protected configuration schema is invalid") from None
    after = {name: _capture_protected_file(path) for name, path in paths.items()}
    if before != after:
        raise JobOperationError("protected configuration changed while it was loaded")
    digest = hashlib.sha256()
    for name in names:
        content = before[name][1]
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
    return GateConfiguration(policy, operators, digest.hexdigest())


def _bounded_integer(value: str) -> int:
    if len(value.lstrip("-")) > MAX_INTEGER_TOKEN_CHARS:
        raise _OversizedInteger
    return int(value)


def _valid_time(value: object) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.utcoffset() != timedelta(0)
        or value.microsecond != 0
        or not 2000 <= value.year <= 9999
    ):
        raise JobOperationError("lifecycle timestamp is invalid")
    return value


def _format_time(value: datetime) -> str:
    return _valid_time(value).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: object) -> datetime:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        raise JobOperationError("lifecycle timestamp is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise JobOperationError("lifecycle timestamp is invalid") from None
    return _valid_time(parsed)


def _valid_login(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 39
        and _LOGIN.fullmatch(cast(str, value)) is not None
    )


def _validate_notification(value: object, current_state: str) -> None:
    if not isinstance(value, NotificationRecord):
        raise JobOperationError("lifecycle notification is invalid")
    _valid_time(value.updated_at)
    if value.status not in _NOTIFICATION_STATUSES:
        raise JobOperationError("lifecycle notification is invalid")
    if current_state in _TERMINAL_STATES:
        if value.event != current_state or value.status == "none":
            raise JobOperationError("terminal notification state is invalid")
    elif value.event != "none" or value.status != "none":
        raise JobOperationError("non-terminal notification state is invalid")


def _valid_log_path(value: object, attempt: int, job_id: str) -> bool:
    if type(value) is not str or not value.startswith("/") or len(value) > 4096:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    path = PurePosixPath(value)
    return (
        ".." not in path.parts
        and tuple(path.parts[-5:-1]) == ("orcd", "scratch", "edullm", "logs")
        and re.fullmatch(
            rf"issue-[1-9][0-9]*-attempt-{attempt}-{job_id}\.log",
            path.name,
        )
        is not None
    )


def _validate_attempt(value: object) -> None:
    if not isinstance(value, JobAttempt):
        raise JobOperationError("lifecycle attempt is invalid")
    match = _ATTEMPT_ID.fullmatch(value.attempt_id)
    if (
        match is None
        or type(value.attempt_number) is not int
        or not 1 <= value.attempt_number <= 999_999
        or int(match.group(1)) != value.attempt_number
        or type(value.request_digest) is not str
        or _SHA256.fullmatch(value.request_digest) is None
        or not _valid_login(value.operator)
        or type(value.slurm_job_id) is not str
        or _JOB_ID.fullmatch(value.slurm_job_id) is None
        or value.state not in _LIFECYCLE_STATES - {"assigned"}
    ):
        raise JobOperationError("lifecycle attempt is invalid")
    run_match = _RUN_ID.fullmatch(value.wandb_run_id)
    if (
        run_match is None
        or int(run_match.group(2)) != value.attempt_number
        or run_match.group(3) != value.slurm_job_id
        or value.wandb_url
        != f"https://wandb.ai/eduLLM/{_wandb_project(value.wandb_url)}/runs/{value.wandb_run_id}"
        or not _valid_log_path(value.log_path, value.attempt_number, value.slurm_job_id)
    ):
        raise JobOperationError("lifecycle attempt is invalid")
    submitted = _valid_time(value.submitted_at)
    updated = _valid_time(value.updated_at)
    if updated < submitted:
        raise JobOperationError("lifecycle attempt timestamp is invalid")


def _wandb_project(url: object) -> str:
    if type(url) is not str or len(url) > 4096:
        return ""
    match = re.fullmatch(
        r"https://wandb\.ai/eduLLM/([A-Za-z0-9][A-Za-z0-9._-]{0,99})/runs/"
        r"issue-[1-9][0-9]*-attempt-[1-9][0-9]{0,5}-[1-9][0-9]{0,19}",
        url,
    )
    return "" if match is None else match.group(1)


def _validate_lifecycle(state: object) -> None:
    if not isinstance(state, LifecycleState):
        raise JobOperationError("lifecycle state is invalid")
    if (
        type(state.issue) is not int
        or state.issue <= 0
        or type(state.request_digest) is not str
        or _SHA256.fullmatch(state.request_digest) is None
        or not _valid_login(state.operator)
        or type(state.assignment_version) is not int
        or not 0 <= state.assignment_version <= 32
        or type(state.attempts) is not tuple
        or len(state.attempts) > MAX_ATTEMPTS
        or state.current_state not in _LIFECYCLE_STATES
    ):
        raise JobOperationError("lifecycle state is invalid")
    _valid_time(state.updated_at)
    if state.current_state == "assigned":
        if state.attempts:
            raise JobOperationError("assigned lifecycle cannot contain an attempt")
    elif not state.attempts:
        raise JobOperationError("active lifecycle must contain an attempt")

    seen_jobs: set[str] = set()
    seen_runs: set[str] = set()
    for expected, attempt in enumerate(state.attempts, start=1):
        _validate_attempt(attempt)
        if (
            attempt.attempt_number != expected
            or attempt.request_digest != state.request_digest
            or attempt.operator != state.operator
            or attempt.slurm_job_id in seen_jobs
            or attempt.wandb_run_id in seen_runs
        ):
            raise JobOperationError("lifecycle attempt binding is invalid")
        seen_jobs.add(attempt.slurm_job_id)
        seen_runs.add(attempt.wandb_run_id)
    if state.attempts:
        latest = state.attempts[-1]
        if latest.state != state.current_state or state.updated_at < latest.updated_at:
            raise JobOperationError("lifecycle state and latest attempt disagree")
    _validate_notification(state.notification, state.current_state)


def build_job_comment(state: LifecycleState) -> str:
    """Build the exact canonical v1 public lifecycle comment."""
    _validate_lifecycle(state)
    payload = {
        "assignment_version": state.assignment_version,
        "attempts": [
            {
                "attempt_id": attempt.attempt_id,
                "attempt_number": attempt.attempt_number,
                "log_path": attempt.log_path,
                "operator": attempt.operator,
                "request_digest": attempt.request_digest,
                "slurm_job_id": attempt.slurm_job_id,
                "state": attempt.state,
                "submitted_at": _format_time(attempt.submitted_at),
                "updated_at": _format_time(attempt.updated_at),
                "wandb_run_id": attempt.wandb_run_id,
                "wandb_url": attempt.wandb_url,
            }
            for attempt in state.attempts
        ],
        "current_state": state.current_state,
        "issue": state.issue,
        "notification": {
            "event": state.notification.event,
            "status": state.notification.status,
            "updated_at": _format_time(state.notification.updated_at),
        },
        "operator": state.operator,
        "request_digest": state.request_digest,
        "updated_at": _format_time(state.updated_at),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    comment = f"{JOB_MARKER}\n{encoded}"
    if len(comment) > MAX_JOB_COMMENT_CHARS:
        raise JobOperationError("lifecycle comment is too large")
    return comment


def parse_job_comment(comment: str) -> LifecycleState:
    """Parse an exact strict canonical v1 public lifecycle comment."""
    if (
        type(comment) is not str
        or len(comment) > MAX_JOB_COMMENT_CHARS
        or len(_JOB_MARKER_LINE.findall(comment)) != 1
        or not comment.startswith(JOB_MARKER + "\n")
    ):
        raise JobOperationError("lifecycle marker is missing or duplicated")
    encoded = comment[len(JOB_MARKER) + 1 :]
    try:
        payload = json.loads(encoded, parse_int=_bounded_integer)
    except (_OversizedInteger, ValueError, RecursionError):
        raise JobOperationError("lifecycle payload is invalid") from None
    if (
        type(payload) is not dict
        or set(payload) != _JOB_FIELDS
        or encoded != json.dumps(payload, sort_keys=True, separators=(",", ":"))
    ):
        raise JobOperationError("lifecycle payload is not strict canonical JSON")
    data = cast(dict[str, object], payload)
    attempts_data = data["attempts"]
    notification_data = data["notification"]
    if (
        type(attempts_data) is not list
        or len(attempts_data) > MAX_ATTEMPTS
        or type(notification_data) is not dict
        or set(notification_data) != _NOTIFICATION_FIELDS
    ):
        raise JobOperationError("lifecycle payload fields are invalid")
    attempts: list[JobAttempt] = []
    for value in cast(list[object], attempts_data):
        if type(value) is not dict or set(value) != _ATTEMPT_FIELDS:
            raise JobOperationError("lifecycle attempt fields are invalid")
        row = cast(dict[str, object], value)
        attempts.append(
            JobAttempt(
                attempt_id=cast(str, row["attempt_id"]),
                attempt_number=cast(int, row["attempt_number"]),
                request_digest=cast(str, row["request_digest"]),
                operator=cast(str, row["operator"]),
                slurm_job_id=cast(str, row["slurm_job_id"]),
                wandb_run_id=cast(str, row["wandb_run_id"]),
                wandb_url=cast(str, row["wandb_url"]),
                log_path=cast(str, row["log_path"]),
                state=cast(str, row["state"]),
                submitted_at=_parse_time(row["submitted_at"]),
                updated_at=_parse_time(row["updated_at"]),
            )
        )
    notice = cast(dict[str, object], notification_data)
    state = LifecycleState(
        issue=cast(int, data["issue"]),
        request_digest=cast(str, data["request_digest"]),
        operator=cast(str, data["operator"]),
        assignment_version=cast(int, data["assignment_version"]),
        attempts=tuple(attempts),
        current_state=cast(str, data["current_state"]),
        updated_at=_parse_time(data["updated_at"]),
        notification=NotificationRecord(
            event=cast(str, notice["event"]),
            status=cast(str, notice["status"]),
            updated_at=_parse_time(notice["updated_at"]),
        ),
    )
    _validate_lifecycle(state)
    return state


def _exact_marker(
    comments: Iterable[IssueComment],
    *,
    marker: re.Pattern[str],
    kind: str,
    required: bool,
    actor: str,
    actor_is_bot: bool,
) -> IssueComment | None:
    matches: list[IssueComment] = []
    for comment in comments:
        count = len(marker.findall(comment.body))
        if count > 1:
            raise JobOperationError(f"duplicate {kind} marker")
        if count == 1:
            matches.append(comment)
    if len(matches) > 1:
        raise JobOperationError(f"duplicate {kind} comments")
    if not matches:
        if required:
            raise JobOperationError(f"{kind} comment is missing")
        return None
    selected = matches[0]
    if selected.author_is_bot is not actor_is_bot or selected.author != actor:
        raise JobOperationError(f"{kind} comment ownership is invalid")
    return selected


def _managed_status(issue: GitHubIssue) -> str:
    labels = _managed_status_labels(issue)
    if len(labels) != 1:
        raise JobOperationError("managed status labels are malformed")
    return next(iter(labels)).removeprefix("status:")


def _managed_status_labels(issue: GitHubIssue) -> set[str]:
    labels = {label for label in issue.labels if label.startswith("status:")}
    if not labels or not labels <= _MANAGED_STATUS_LABELS or len(labels) > 2:
        raise JobOperationError("managed status labels are malformed")
    return labels


def _valid_label_transition(old: str, new: str) -> bool:
    if old == "assigned":
        return new == "submitted"
    if old == "submitted":
        return new == "running" or new in _TERMINAL_STATES
    if old == "running":
        return new in _TERMINAL_STATES
    return False


def _enabled_operators(operators: Sequence[Operator]) -> dict[str, Operator]:
    enabled: dict[str, Operator] = {}
    all_logins: set[str] = set()
    for operator in operators:
        if (
            not isinstance(operator, Operator)
            or not _valid_login(operator.github)
            or operator.github in all_logins
            or type(operator.enabled) is not bool
        ):
            raise JobOperationError("operator configuration is invalid")
        all_logins.add(operator.github)
        if operator.enabled:
            enabled[operator.github] = operator
    return enabled


def _validate_configuration(configuration: object) -> GateConfiguration:
    if (
        not isinstance(configuration, GateConfiguration)
        or not isinstance(configuration.policy, Policy)
        or type(configuration.operators) is not tuple
        or type(configuration.digest) is not str
        or _SHA256.fullmatch(configuration.digest) is None
    ):
        raise JobOperationError("protected gate configuration is invalid")
    _enabled_operators(configuration.operators)
    return configuration


def _issue_snapshot(
    github: Any,
    issue_number: int,
    *,
    expected_issue: GitHubIssue | None = None,
) -> GitHubIssue:
    try:
        if expected_issue is None:
            listed = github.list_active_queue_issues()
            matches = [issue for issue in listed if issue.number == issue_number]
            if len(matches) != 1:
                raise JobOperationError("exact queue Issue was not found")
            expected_issue = matches[0]
        elif not isinstance(expected_issue, GitHubIssue) or expected_issue.number != issue_number:
            raise JobOperationError("exact queue Issue was not found")
        current = github.fetch_issue(issue_number)
    except JobOperationError:
        raise
    except Exception:
        raise JobOperationError("fresh GitHub Issue evidence is unavailable") from None
    if current.number != expected_issue.number or "edullm-job" not in current.labels:
        raise JobOperationError("exact queue Issue was not found")
    return current


def _gate(
    issue_number: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    allowed_statuses: Set[str],
    revalidate_commit: bool,
    automation_actor: str,
    expected_issue: GitHubIssue | None = None,
) -> SubmissionGateSnapshot:
    configuration = _validate_configuration(configuration)
    if not _valid_login(operator):
        raise JobOperationError("local GitHub operator identity is invalid")
    enabled = _enabled_operators(configuration.operators)
    if operator not in enabled:
        raise JobOperationError("local identity is not an enabled protected operator")
    issue = _issue_snapshot(github, issue_number, expected_issue=expected_issue)
    managed_labels = _managed_status_labels(issue)
    enabled_assignees = set(issue.assignees) & set(enabled)
    if operator not in issue.assignees or enabled_assignees != {operator}:
        raise JobOperationError("Issue assignee does not match the protected operator")
    try:
        request = parse_issue(
            issue.body,
            issue_number=issue.number,
            requester=issue.requester,
        )
        comments = github.list_issue_comments(issue.number)
    except (IssueParseError, Exception) as error:
        if isinstance(error, JobOperationError):
            raise
        raise JobOperationError("current Issue request or comments are unavailable") from None
    status_comment = _exact_marker(
        comments,
        marker=_STATUS_MARKER_LINE,
        kind="validated status",
        required=True,
        actor=automation_actor,
        actor_is_bot=True,
    )
    assignment_comment = _exact_marker(
        comments,
        marker=_ASSIGNMENT_MARKER_LINE,
        kind="assignment",
        required=True,
        actor=automation_actor,
        actor_is_bot=True,
    )
    assert status_comment is not None and assignment_comment is not None
    try:
        validated = validated_status_for_request(status_comment.body, request)
        assignment = parse_assignment_comment(assignment_comment.body)
    except Exception:
        raise JobOperationError("validated request or assignment binding is invalid") from None
    if (
        assignment.issue != issue.number
        or assignment.request_digest != request.digest
        or assignment.operator_github != operator
        or assignment.operator_github not in enabled
    ):
        raise JobOperationError("assignment does not match the current request and operator")

    profile = configuration.policy.entrypoints.get(request.entrypoint_profile)
    if not isinstance(profile, Mapping):
        raise JobOperationError("protected entrypoint profile is invalid")
    if profile.get("wandb_callback") is not True:
        raise JobOperationError("protected entrypoint must enable W&B auditing")
    model_identity = profile.get("model_identity")
    allowed_kinds = profile.get("allowed_data_kinds")
    if (
        type(model_identity) is not str
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,99}", model_identity) is None
        or type(allowed_kinds) not in {tuple, list}
    ):
        raise JobOperationError("protected entrypoint profile is incomplete")
    kinds = cast(tuple[object, ...] | list[object], allowed_kinds)
    if not kinds or any(type(kind) is not str for kind in kinds):
        raise JobOperationError("protected entrypoint profile is incomplete")
    if revalidate_commit:
        try:
            decision = validation_decision(
                request,
                policy=configuration.policy,
                github=github,
            )
        except Exception:
            raise JobOperationError("commit or script evidence is unavailable") from None
        if decision.status != "ready" or decision.errors:
            raise JobOperationError("commit or request policy is no longer authorized")
    elif validate_request(request, configuration.policy):
        raise JobOperationError("current request no longer satisfies protected policy")

    lifecycle_comment = _exact_marker(
        comments,
        marker=_JOB_MARKER_LINE,
        kind="job lifecycle",
        required=False,
        actor=operator,
        actor_is_bot=False,
    )
    lifecycle = None if lifecycle_comment is None else parse_job_comment(lifecycle_comment.body)
    if lifecycle is not None and (
        lifecycle.issue != issue.number
        or lifecycle.request_digest != request.digest
        or lifecycle.operator != operator
        or lifecycle.assignment_version != len(assignment.history)
    ):
        raise JobOperationError("job lifecycle does not match its assignment")
    managed_states = {label.removeprefix("status:") for label in managed_labels}
    if len(managed_states) == 1:
        if next(iter(managed_states)) not in allowed_statuses:
            raise JobOperationError("Issue lifecycle status is not authorized for this operation")
    else:
        if lifecycle is None or lifecycle.current_state not in managed_states:
            raise JobOperationError("managed status labels are malformed")
        stale_states = managed_states - {lifecycle.current_state}
        if (
            len(stale_states) != 1
            or not stale_states <= allowed_statuses
            or not _valid_label_transition(next(iter(stale_states)), lifecycle.current_state)
        ):
            raise JobOperationError("managed status labels are malformed")

    return SubmissionGateSnapshot(
        issue=issue.number,
        request=request,
        request_digest=request.digest,
        operator=operator,
        validated_at=validated.validated_at,
        status_comment_id=status_comment.id,
        assignment_comment_id=assignment_comment.id,
        assignment_binding=hashlib.sha256(assignment_comment.body.encode("utf-8")).hexdigest(),
        assignment_version=len(assignment.history),
        config_digest=configuration.digest,
        lifecycle=lifecycle,
        profile=profile,
        repository_url=configuration.policy.repository_url,
        scratch_root=configuration.policy.scratch_root,
        slurm_partition=configuration.policy.slurm_partition,
        slurm_memory=configuration.policy.slurm_memory,
        slurm_cpus_per_gpu=configuration.policy.slurm_cpus_per_gpu,
    )


def full_submission_gate(
    issue_number: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    automation_actor: str = AUTOMATION_ACTOR,
) -> SubmissionGateSnapshot:
    """Perform the complete fresh pre-submission gate."""
    return _gate(
        issue_number,
        operator=operator,
        github=github,
        configuration=configuration,
        allowed_statuses={"assigned"},
        revalidate_commit=True,
        automation_actor=automation_actor,
    )


_FIXED_OPTION_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


def _fixed_option_value(value: object, request: JobRequest) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if value is None:
        return "null"
    if type(value) is int:
        return str(value)
    if type(value) is str:
        if _SAFE_TEXT.fullmatch(value) is None:
            raise JobOperationError("protected fixed option is invalid")
        return value
    if type(value) is tuple:
        items = cast(tuple[object, ...], value)
        if not items or any(
            type(item) is not str or _SAFE_TEXT.fullmatch(item) is None for item in items
        ):
            raise JobOperationError("protected fixed option is invalid")
        return json.dumps(items, separators=(",", ":"))
    if isinstance(value, Mapping):
        fields = set(value)
        if fields == {"type", "field"} and value.get("type") == "request_field":
            field = value.get("field")
            if field != "study" or _SAFE_TEXT.fullmatch(request.study) is None:
                raise JobOperationError("protected request-derived option is invalid")
            return request.study
        if fields == {"value", "unit"}:
            steps = value.get("value")
            unit = value.get("unit")
            if type(steps) is not int or not 1 <= steps <= 100 or unit != "steps":
                raise JobOperationError("protected duration option is invalid")
            return json.dumps({"value": steps, "unit": unit}, separators=(",", ":"))
    raise JobOperationError("protected fixed option is invalid")


def _resolve_fixed_arguments(
    profile: Mapping[str, object],
    request: JobRequest,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...],]:
    launcher_values = profile.get("fixed_launcher_arguments", ())
    option_values = profile.get("fixed_options", {})
    if type(launcher_values) is not tuple or not isinstance(option_values, Mapping):
        raise JobOperationError("protected fixed arguments are invalid")
    launcher = tuple(cast(Sequence[str], launcher_values))
    if any(
        type(argument) is not str or _SAFE_TEXT.fullmatch(argument) is None for argument in launcher
    ):
        raise JobOperationError("protected fixed launcher argument is invalid")

    arguments: list[str] = []
    names: list[str] = []
    derived: list[str] = []
    wandb_tags: tuple[str, ...] = ()
    for name, value in option_values.items():
        if type(name) is not str or _FIXED_OPTION_NAME.fullmatch(name) is None or name in names:
            raise JobOperationError("protected fixed option name is invalid")
        names.append(name)
        if name == "trainer.callbacks.wandb.tags":
            if type(value) is not tuple:
                raise JobOperationError("protected W&B tags are invalid")
            _fixed_option_value(value, request)
            wandb_tags = tuple(cast(Sequence[str], value))
            continue
        if isinstance(value, Mapping) and value.get("type") == "derived_path":
            if (
                set(value) != {"type", "root_env", "relative"}
                or value.get("root_env") != "EDULLM_SCRATCH"
                or value.get("relative") != "runs/{run_name}"
                or name not in {"save-folder", "work-dir"}
            ):
                raise JobOperationError("protected derived path option is invalid")
            derived.append(name)
            continue
        rendered = _fixed_option_value(value, request)
        arguments.append(f"--{name}={rendered}")
    return launcher, tuple(arguments), tuple(names), tuple(derived), wandb_tags


def build_resolved_request(
    snapshot: SubmissionGateSnapshot,
    *,
    attempt_number: int,
) -> ResolvedRequest:
    """Build trusted execution metadata from one successful gate snapshot."""
    if not isinstance(snapshot, SubmissionGateSnapshot):
        raise JobOperationError("submission gate snapshot is invalid")
    profile = snapshot.profile
    model_identity = profile.get("model_identity")
    kinds = profile.get("allowed_data_kinds")
    if type(model_identity) is not str or type(kinds) not in {tuple, list}:
        raise JobOperationError("protected entrypoint profile is invalid")
    request = snapshot.request
    launcher, fixed, fixed_names, derived, wandb_tags = _resolve_fixed_arguments(profile, request)
    return ResolvedRequest(
        request=request,
        operator=snapshot.operator,
        wandb_entity="eduLLM",
        wandb_run_prefix=f"issue-{request.issue_number}-attempt-{attempt_number}",
        slurm_job_name=request.request_name,
        log_pattern=f"logs/issue-{request.issue_number}-attempt-{attempt_number}-%j.log",
        allowed_data_kinds=tuple(cast(Sequence[str], kinds)),
        model_identity=model_identity,
        repository_url=snapshot.repository_url,
        scratch_root=snapshot.scratch_root,
        slurm_partition=snapshot.slurm_partition,
        slurm_memory=snapshot.slurm_memory,
        slurm_cpus_per_gpu=snapshot.slurm_cpus_per_gpu,
        fixed_launcher_arguments=launcher,
        fixed_arguments=fixed,
        fixed_option_names=fixed_names,
        derived_path_options=derived,
        fixed_wandb_tags=wandb_tags,
    )


def _persist_lifecycle(
    github: Any,
    issue: int,
    state: LifecycleState,
    *,
    operator: str,
) -> IssueComment:
    if operator != state.operator or not _valid_login(operator):
        raise JobOperationError("lifecycle operator binding is invalid")
    body = build_job_comment(state)
    comments = github.list_issue_comments(issue)
    existing = _exact_marker(
        comments,
        marker=_JOB_MARKER_LINE,
        kind="job lifecycle",
        required=False,
        actor=operator,
        actor_is_bot=False,
    )
    try:
        persisted = (
            github.create_issue_comment(issue, body)
            if existing is None
            else github.update_issue_comment(existing.id, body)
        )
    except GitHubError:
        comments = github.list_issue_comments(issue)
        reconciled = _exact_marker(
            comments,
            marker=_JOB_MARKER_LINE,
            kind="job lifecycle",
            required=True,
            actor=operator,
            actor_is_bot=False,
        )
        if (
            reconciled is None
            or reconciled.body != body
            or (existing is not None and reconciled.id != existing.id)
        ):
            raise JobOperationError("GitHub lifecycle write outcome is ambiguous") from None
        persisted = reconciled
    if persisted.body != body or persisted.author_is_bot or persisted.author != operator:
        raise JobOperationError("persisted lifecycle ownership is invalid")
    confirmed = _exact_marker(
        github.list_issue_comments(issue),
        marker=_JOB_MARKER_LINE,
        kind="job lifecycle",
        required=True,
        actor=operator,
        actor_is_bot=False,
    )
    if confirmed is None or confirmed.id != persisted.id or confirmed.body != body:
        raise JobOperationError("lifecycle postcondition failed")
    return confirmed


def _transition_labels(github: Any, issue: int, old: str, new: str) -> None:
    old_label = f"status:{old}"
    new_label = f"status:{new}"
    current = github.fetch_issue(issue)
    managed = _managed_status_labels(current)
    allowed = {old_label, new_label}
    if not managed <= allowed:
        raise JobOperationError("lifecycle label transition is malformed")
    if managed == {new_label}:
        return
    if old == new or managed not in ({old_label}, {old_label, new_label}):
        raise JobOperationError("lifecycle label transition is malformed")
    if managed == {old_label}:
        try:
            github.add_issue_status_label(issue, new_label)
        except GitHubError:
            current = github.fetch_issue(issue)
            if new_label not in current.labels:
                raise JobOperationError("lifecycle label write failed") from None
        current = github.fetch_issue(issue)
        managed = _managed_status_labels(current)
    if managed != {old_label, new_label}:
        raise JobOperationError("lifecycle label transition is malformed")
    try:
        github.remove_issue_status_label(issue, old_label)
    except GitHubError:
        current = github.fetch_issue(issue)
        if old_label in current.labels:
            raise JobOperationError("lifecycle label write failed") from None
    final = github.fetch_issue(issue)
    if _managed_status(final) != new:
        raise JobOperationError("lifecycle label postcondition failed")


def _repair_labels_to_lifecycle(github: Any, issue: int, state: str) -> None:
    current = github.fetch_issue(issue)
    managed = _managed_status_labels(current)
    expected = f"status:{state}"
    if managed == {expected}:
        return
    if len(managed) == 1:
        stale = next(iter(managed)).removeprefix("status:")
    elif expected in managed and len(managed) == 2:
        stale = next(iter(managed - {expected})).removeprefix("status:")
    else:
        raise JobOperationError("lifecycle label transition is malformed")
    if not _valid_label_transition(stale, state):
        raise JobOperationError("lifecycle label transition is malformed")
    _transition_labels(github, issue, stale, state)


def _ensure_notice(
    github: Any,
    issue: int,
    body: str,
    *,
    actor: str,
    actor_is_bot: bool,
) -> None:
    comments = github.list_issue_comments(issue)
    matches = [comment for comment in comments if comment.body == body]
    if len(matches) > 1 or (
        matches and (matches[0].author_is_bot is not actor_is_bot or matches[0].author != actor)
    ):
        raise JobOperationError("requester notice ownership is invalid")
    if matches:
        return
    try:
        persisted = github.create_issue_comment(issue, body)
    except GitHubError:
        matches = [comment for comment in github.list_issue_comments(issue) if comment.body == body]
        if (
            len(matches) != 1
            or matches[0].author_is_bot is not actor_is_bot
            or matches[0].author != actor
        ):
            raise JobOperationError("requester notice write outcome is ambiguous") from None
        return
    if (
        persisted.body != body
        or persisted.author_is_bot is not actor_is_bot
        or persisted.author != actor
    ):
        raise JobOperationError("requester notice ownership is invalid")


def _receipt_matches(receipt: SubmissionReceipt, spec: SubmissionSpec) -> bool:
    return (
        receipt.issue == spec.issue
        and receipt.request_digest == spec.request_digest
        and receipt.attempt_number == spec.attempt_number
        and receipt.operator == spec.operator
        and receipt.remote_user == spec.remote_user
        and receipt.script_sha256 == spec.script_sha256
        and receipt.manifest_sha256 == spec.manifest_sha256
    )


def _submission_notice(requester: str, issue: int, attempt: JobAttempt) -> str:
    return (
        f"@{requester} eduLLM job #{issue} was submitted as Slurm job "
        f"{attempt.slurm_job_id}. W&B: {attempt.wandb_url}"
    )


def run_assigned(
    *,
    operator: str,
    github: Any,
    load_configuration: Callable[[], GateConfiguration],
    remote: Any,
    now: datetime,
    automation_actor: str = AUTOMATION_ACTOR,
) -> LifecycleState:
    """Select, gate twice, submit once, and publish one assigned request."""
    _valid_time(now)
    try:
        listed = github.list_active_queue_issues()
    except Exception:
        raise JobOperationError("assigned Issue scan failed") from None
    candidates = []
    for issue in listed:
        try:
            managed = _managed_status_labels(issue)
            if "status:assigned" in managed and operator in issue.assignees:
                candidates.append(issue.number)
            elif len(managed) != 1:
                raise JobOperationError("managed status labels are malformed")
        except JobOperationError:
            raise
    if not candidates:
        raise JobOperationError("no assigned request is available")
    issue_number = min(candidates)

    first_config = load_configuration()
    first = full_submission_gate(
        issue_number,
        operator=operator,
        github=github,
        configuration=first_config,
        automation_actor=automation_actor,
    )
    recovery_state: LifecycleState | None = None
    if first.lifecycle is not None:
        if first.lifecycle.current_state == "assigned":
            if first.lifecycle.attempts:
                raise JobOperationError("assigned request has an invalid submission attempt")
            attempt_number = 1
        elif not first.lifecycle.attempts:
            raise JobOperationError("assigned request already has a submission attempt")
        else:
            recovery_state = first.lifecycle
            attempt_number = recovery_state.attempts[-1].attempt_number
    else:
        attempt_number = 1
    resolved = build_resolved_request(first, attempt_number=attempt_number)
    script = render_sbatch(resolved)
    remote_user = getattr(remote, "remote_user", None)
    if type(remote_user) is not str or _REMOTE_USER.fullmatch(remote_user) is None:
        raise JobOperationError("trusted remote user identity is unavailable")
    spec = SubmissionSpec(
        issue=issue_number,
        request_digest=first.request_digest,
        attempt_number=attempt_number,
        operator=operator,
        remote_user=remote_user,
        script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
        manifest_uri=first.request.data_manifest,
        manifest_sha256=first.request.data_manifest_sha256,
        allowed_data_kinds=resolved.allowed_data_kinds,
        log_pattern=resolved.log_pattern,
    )
    key = build_submission_key(issue_number, first.request_digest, attempt_number, operator)
    try:
        remote.stage(key, script, spec)
        remote.verify_manifest(spec)
    except Exception:
        raise JobOperationError("remote submission preparation failed") from None

    second_config = load_configuration()
    second = full_submission_gate(
        issue_number,
        operator=operator,
        github=github,
        configuration=second_config,
        automation_actor=automation_actor,
    )
    if second != first:
        raise JobOperationError("submission evidence changed between mandatory gates")
    try:
        receipt = remote.submit(key, spec)
    except Exception:
        raise JobOperationError("remote submission transaction failed") from None
    if not isinstance(receipt, SubmissionReceipt) or not _receipt_matches(receipt, spec):
        raise JobOperationError("remote submission receipt does not match the request")

    wandb_run_id = f"{resolved.wandb_run_prefix}-{receipt.slurm_job_id}"
    wandb_url = (
        f"https://wandb.ai/{resolved.wandb_entity}/{first.request.wandb_project}"
        f"/runs/{wandb_run_id}"
    )
    if recovery_state is not None:
        recorded = recovery_state.attempts[-1]
        if (
            recorded.attempt_number != attempt_number
            or recorded.request_digest != first.request_digest
            or recorded.operator != operator
            or recorded.slurm_job_id != receipt.slurm_job_id
            or recorded.wandb_run_id != wandb_run_id
            or recorded.wandb_url != wandb_url
            or recorded.log_path != receipt.log_path
            or recorded.submitted_at != receipt.submitted_at
        ):
            raise JobOperationError("recovery receipt does not match canonical lifecycle state")
        _persist_lifecycle(github, issue_number, recovery_state, operator=operator)
        _transition_labels(
            github,
            issue_number,
            "assigned",
            recovery_state.current_state,
        )
        requester = normalize_actor_login(first.request.requester)
        _ensure_notice(
            github,
            issue_number,
            _submission_notice(requester, issue_number, recorded),
            actor=operator,
            actor_is_bot=False,
        )
        return recovery_state

    attempt = JobAttempt(
        attempt_id=f"attempt-{attempt_number}",
        attempt_number=attempt_number,
        request_digest=first.request_digest,
        operator=operator,
        slurm_job_id=receipt.slurm_job_id,
        wandb_run_id=wandb_run_id,
        wandb_url=wandb_url,
        log_path=receipt.log_path,
        state="submitted",
        submitted_at=receipt.submitted_at,
        updated_at=receipt.submitted_at,
    )
    state = LifecycleState(
        issue=issue_number,
        request_digest=first.request_digest,
        operator=operator,
        assignment_version=first.assignment_version,
        attempts=(attempt,),
        current_state="submitted",
        updated_at=receipt.submitted_at,
        notification=NotificationRecord("none", "none", receipt.submitted_at),
    )
    _persist_lifecycle(github, issue_number, state, operator=operator)
    _transition_labels(github, issue_number, "assigned", "submitted")
    requester = normalize_actor_login(first.request.requester)
    _ensure_notice(
        github,
        issue_number,
        _submission_notice(requester, issue_number, attempt),
        actor=operator,
        actor_is_bot=False,
    )
    return state


def _clean_scheduler_text(value: object, name: str) -> str:
    if type(value) is not str or len(value) > 256 or _SAFE_TEXT.fullmatch(value) is None:
        raise JobOperationError(f"Slurm {name} is invalid")
    return cast(str, value)


def _scheduler_state(value: object) -> tuple[str, str]:
    if type(value) is list:
        if len(value) != 1 or type(value[0]) is not str:
            raise JobOperationError("Slurm state is invalid")
        value = value[0]
    if type(value) is not str:
        raise JobOperationError("Slurm state is invalid")
    normalized = value.partition("+")[0].partition(" ")[0]
    lifecycle = _SLURM_TO_LIFECYCLE.get(normalized)
    if lifecycle is None:
        raise JobOperationError("Slurm state is unknown")
    return normalized, lifecycle


def parse_squeue_json(output: str) -> tuple[SlurmJob, ...]:
    """Parse bounded ``squeue --json`` evidence and reject unknown states."""
    if type(output) is not str or not output or len(output) > MAX_SLURM_OUTPUT_CHARS:
        raise JobOperationError("squeue output is invalid")
    try:
        payload = json.loads(output, parse_int=_bounded_integer)
    except (_OversizedInteger, ValueError, RecursionError):
        raise JobOperationError("squeue output is invalid") from None
    if type(payload) is not dict or type(payload.get("jobs")) is not list:
        raise JobOperationError("squeue output is invalid")
    rows = cast(list[object], payload["jobs"])
    if len(rows) > MAX_SLURM_JOBS:
        raise JobOperationError("squeue output contains too many jobs")
    jobs: list[SlurmJob] = []
    seen: set[str] = set()
    for value in rows:
        if type(value) is not dict:
            raise JobOperationError("squeue row is invalid")
        row = cast(dict[str, object], value)
        if not {"job_id", "job_state", "name", "user_name"} <= set(row):
            raise JobOperationError("squeue row is invalid")
        identifier = row["job_id"]
        if type(identifier) is not int or identifier <= 0:
            raise JobOperationError("squeue job ID is invalid")
        job_id = str(identifier)
        if _JOB_ID.fullmatch(job_id) is None or job_id in seen:
            raise JobOperationError("squeue job ID is invalid")
        state, lifecycle = _scheduler_state(row["job_state"])
        jobs.append(
            SlurmJob(
                job_id=job_id,
                name=_clean_scheduler_text(row["name"], "job name"),
                state=state,
                user=_clean_scheduler_text(row["user_name"], "user"),
                lifecycle_state=lifecycle,
            )
        )
        seen.add(job_id)
    return tuple(jobs)


def parse_sacct(output: str) -> tuple[SlurmJob, ...]:
    """Parse exact ``JobIDRaw,JobName,State,User`` parsable2 evidence."""
    if type(output) is not str or not output or len(output) > MAX_SLURM_OUTPUT_CHARS:
        raise JobOperationError("sacct output is invalid")
    jobs: list[SlurmJob] = []
    seen: set[str] = set()
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise JobOperationError("sacct row is invalid")
        job_id, name, raw_state, user = fields
        if _JOB_ID.fullmatch(job_id) is None or job_id in seen:
            raise JobOperationError("sacct job ID is invalid")
        state, lifecycle = _scheduler_state(raw_state)
        jobs.append(
            SlurmJob(
                job_id=job_id,
                name=_clean_scheduler_text(name, "job name"),
                state=state,
                user=_clean_scheduler_text(user, "user"),
                lifecycle_state=lifecycle,
            )
        )
        seen.add(job_id)
    if not jobs or len(jobs) > MAX_SLURM_JOBS:
        raise JobOperationError("sacct output is invalid")
    return tuple(jobs)


def reconcile_lifecycle(
    state: LifecycleState,
    evidence: str,
    *,
    now: datetime,
) -> LifecycleState:
    """Advance lifecycle monotonically from authoritative Slurm evidence."""
    _validate_lifecycle(state)
    _valid_time(now)
    if evidence not in _LIFECYCLE_STATES - {"assigned"}:
        raise JobOperationError("Slurm lifecycle evidence is invalid")
    current = state.current_state
    if current in _TERMINAL_STATES:
        return state
    if evidence == "submitted" and current == "running":
        return state
    if evidence == current:
        return state
    if current == "submitted" and evidence not in {"running", *_TERMINAL_STATES}:
        raise JobOperationError("Slurm lifecycle transition is invalid")
    if current == "running" and evidence not in _TERMINAL_STATES:
        raise JobOperationError("Slurm lifecycle transition is invalid")
    latest = replace(state.attempts[-1], state=evidence, updated_at=now)
    notification = (
        NotificationRecord(evidence, "pending", now)
        if evidence in _TERMINAL_STATES
        else NotificationRecord("none", "none", now)
    )
    updated = replace(
        state,
        attempts=state.attempts[:-1] + (latest,),
        current_state=evidence,
        updated_at=now,
        notification=notification,
    )
    _validate_lifecycle(updated)
    return updated


_SECRET_PATTERNS = (
    re.compile(r"(?i)(WANDB_API_KEY\s*=\s*)\S+"),
    re.compile(r"(?i)(Authorization:\s*Bearer\s+)\S+"),
    re.compile(r"(?i)(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})"),
    re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+"),
    re.compile(r"(?i)(token\s*=\s*)\S+"),
)


def _redact_log(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted>", redacted)
    return redacted


def read_authorized_log(
    path_value: str,
    *,
    logs_root: Path,
    max_bytes: int = DEFAULT_LOG_BYTES,
) -> str:
    """Read one recorded direct-child log without following links and redact it."""
    if (
        type(path_value) is not str
        or not isinstance(logs_root, Path)
        or not logs_root.is_absolute()
        or type(max_bytes) is not int
        or not 1 <= max_bytes <= 1_048_576
    ):
        raise JobOperationError("log request is invalid")
    path = Path(path_value)
    if (
        not path.is_absolute()
        or path.parent != logs_root
        or ".." in PurePosixPath(path_value).parts
    ):
        raise JobOperationError("recorded log path is outside the fixed log root")
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        root_before = logs_root.stat(follow_symlinks=False)
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
            raise JobOperationError("fixed log root is unsafe")
        directory_fd = os.open(
            logs_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_uid != os.getuid():
            raise JobOperationError("recorded log file is unsafe")
        content = os.read(descriptor, max_bytes + 1)
        named = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        root_after = logs_root.stat(follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino) or (
            root_after.st_dev,
            root_after.st_ino,
        ) != (root_before.st_dev, root_before.st_ino):
            raise JobOperationError("recorded log changed during reading")
        truncated = len(content) > max_bytes
        content = content[:max_bytes]
        text = content.decode("utf-8", errors="replace")
        return _redact_log(text) + ("\n[log truncated]\n" if truncated else "")
    except JobOperationError:
        raise
    except OSError:
        raise JobOperationError("recorded log cannot be read safely") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


def _authorized_lifecycle_gate(
    issue: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    automation_actor: str,
    expected_issue: GitHubIssue | None = None,
) -> SubmissionGateSnapshot:
    snapshot = _gate(
        issue,
        operator=operator,
        github=github,
        configuration=configuration,
        allowed_statuses={"submitted", "running", *_TERMINAL_STATES},
        revalidate_commit=False,
        automation_actor=automation_actor,
        expected_issue=expected_issue,
    )
    if snapshot.lifecycle is None or not snapshot.lifecycle.attempts:
        raise JobOperationError("authorized Issue has no canonical attempt binding")
    return snapshot


def stop_job(
    issue: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    slurm: Any,
    now: datetime,
    automation_actor: str = AUTOMATION_ACTOR,
) -> LifecycleState:
    """Cancel only the current operator's canonical numeric job, then reconcile."""
    _valid_time(now)
    first = _authorized_lifecycle_gate(
        issue,
        operator=operator,
        github=github,
        configuration=configuration,
        automation_actor=automation_actor,
    )
    assert first.lifecycle is not None
    state = first.lifecycle
    job_id = state.attempts[-1].slurm_job_id
    if _JOB_ID.fullmatch(job_id) is None:
        raise JobOperationError("canonical Slurm job binding is invalid")
    try:
        evidence = slurm.status(job_id)
    except Exception:
        raise JobOperationError("authoritative Slurm state is unavailable") from None
    if evidence in _TERMINAL_STATES:
        repaired = reconcile_lifecycle(state, evidence, now=now)
        if repaired != state:
            _persist_lifecycle(github, issue, repaired, operator=operator)
        _repair_labels_to_lifecycle(github, issue, repaired.current_state)
        requester = normalize_actor_login(first.request.requester)
        _ensure_notice(
            github,
            issue,
            _terminal_notice(requester, issue, repaired),
            actor=operator,
            actor_is_bot=False,
        )
        return repaired
    if evidence not in {"submitted", "running"}:
        raise JobOperationError("authoritative Slurm state is invalid")

    second = _authorized_lifecycle_gate(
        issue,
        operator=operator,
        github=github,
        configuration=configuration,
        automation_actor=automation_actor,
    )
    if second != first:
        raise JobOperationError("job binding changed before cancellation")
    try:
        slurm.cancel(job_id)
        after = slurm.status(job_id)
    except Exception:
        raise JobOperationError("Slurm cancellation outcome is unavailable") from None
    if after not in {"submitted", "running", *_TERMINAL_STATES}:
        raise JobOperationError("authoritative Slurm state is invalid")
    repaired = reconcile_lifecycle(state, after, now=now)
    if repaired != state:
        _persist_lifecycle(github, issue, repaired, operator=operator)
    _repair_labels_to_lifecycle(github, issue, repaired.current_state)
    if repaired.current_state == "cancelled":
        requester = normalize_actor_login(first.request.requester)
        _ensure_notice(
            github,
            issue,
            f"@{requester} eduLLM job #{issue} was cancelled.",
            actor=operator,
            actor_is_bot=False,
        )
    return repaired


class SSHSlurm:
    """Bounded SSH boundary for authorized scheduler, log, and offline-sync work."""

    def __init__(self, ssh_client: Any | None = None) -> None:
        from edullm.ssh import SSHClient

        self.ssh_client = SSHClient() if ssh_client is None else ssh_client

    @staticmethod
    def _remote_python(arguments: Sequence[str]) -> list[str]:
        command = 'set -euo pipefail\nsource "$HOME/venvs/edullm/bin/activate"\n' + shlex.join(
            arguments
        )
        return ["bash", "-lc", command]

    def query(self, job_ids: Sequence[str]) -> dict[str, SlurmJob]:
        """Query only explicit canonical job IDs, using squeue then sacct."""
        if (
            not isinstance(job_ids, Sequence)
            or not job_ids
            or len(job_ids) > MAX_ATTEMPTS
            or any(
                type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None for job_id in job_ids
            )
            or len(set(job_ids)) != len(job_ids)
        ):
            raise JobOperationError("authorized Slurm query IDs are invalid")
        joined = ",".join(job_ids)
        try:
            active_result = self.ssh_client.run_remote(
                ["squeue", "--json", "--jobs", joined],
                check=False,
                timeout=30,
            )
        except Exception:
            raise JobOperationError("squeue query failed") from None
        active: dict[str, SlurmJob] = {}
        if getattr(active_result, "returncode", None) == 0:
            for job in parse_squeue_json(active_result.stdout):
                if job.job_id not in job_ids:
                    raise JobOperationError("squeue returned an unauthorized job")
                active[job.job_id] = job
        missing = [job_id for job_id in job_ids if job_id not in active]
        if missing:
            try:
                accounting_result = self.ssh_client.run_remote(
                    [
                        "sacct",
                        "-X",
                        "--noheader",
                        "--parsable2",
                        "--jobs",
                        ",".join(missing),
                        "--format=JobIDRaw,JobName,State,User",
                    ],
                    check=False,
                    timeout=30,
                )
            except Exception:
                raise JobOperationError("sacct query failed") from None
            if getattr(accounting_result, "returncode", None) != 0:
                raise JobOperationError("sacct query failed")
            for job in parse_sacct(accounting_result.stdout):
                if job.job_id not in missing or job.job_id in active:
                    raise JobOperationError("sacct returned an unauthorized or duplicate job")
                active[job.job_id] = job
        if set(active) != set(job_ids):
            raise JobOperationError("Slurm did not return exact authoritative evidence")
        return active

    def status(self, job_id: str) -> str:
        """Return normalized lifecycle evidence for one authorized numeric ID."""
        return self.query((job_id,))[job_id].lifecycle_state

    def cancel(self, job_id: str) -> None:
        """Issue one numeric scancel with no user-controlled options."""
        if type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None:
            raise JobOperationError("Slurm cancellation ID is invalid")
        try:
            result = self.ssh_client.run_remote(
                ["scancel", job_id],
                check=False,
                timeout=30,
            )
        except Exception:
            raise JobOperationError("Slurm cancellation outcome is unavailable") from None
        if getattr(result, "returncode", None) != 0:
            # A cancellation race is reconciled by the caller's authoritative
            # status query; do not expose scheduler diagnostics.
            return

    def read_log(self, path: str) -> str:
        """Read one canonical remote log through the reviewed bounded helper."""
        if type(path) is not str or not path.startswith("/") or len(path) > 4096:
            raise JobOperationError("recorded log path is invalid")
        try:
            result = self.ssh_client.run_remote(
                self._remote_python(["python", "-m", "edullm.jobs", "read-log", path]),
                check=False,
                timeout=30,
            )
        except Exception:
            raise JobOperationError("remote log read failed") from None
        if (
            getattr(result, "returncode", None) != 0
            or type(getattr(result, "stdout", None)) is not str
            or len(result.stdout.encode("utf-8")) > DEFAULT_LOG_BYTES + 256
        ):
            raise JobOperationError("remote log read failed")
        return result.stdout

    def reconcile_offline_tracking(self) -> None:
        """Best-effort bounded sync for retained offline W&B files."""
        command = (
            'set -euo pipefail\nsource "$HOME/venvs/edullm/bin/activate"\n'
            "timeout 120 wandb sync --include-offline "
            '"$HOME/orcd/scratch/edullm/wandb" >/dev/null 2>&1 || true'
        )
        try:
            self.ssh_client.run_remote(
                ["bash", "-lc", command],
                check=False,
                timeout=130,
            )
        except Exception:
            pass


def _terminal_notice(requester: str, issue: int, state: LifecycleState) -> str:
    latest = state.attempts[-1]
    if state.current_state == "completed":
        detail = f"completed. W&B: {latest.wandb_url}"
    else:
        detail = f"ended with status {state.current_state}."
    return f"@{requester} eduLLM job #{issue} {detail}"


def reconcile_operator_jobs(
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    slurm: Any,
    now: datetime,
    mine: bool = False,
    automation_actor: str = AUTOMATION_ACTOR,
) -> tuple[OperatorJobSummary, ...]:
    """Query only this operator's canonical jobs and repair GitHub monotonically."""
    del mine  # The one-operator pilot authorizes the same bounded view for both forms.
    _valid_time(now)
    try:
        issues = github.list_active_queue_issues()
    except Exception:
        raise JobOperationError("job Issue scan failed") from None
    if not isinstance(issues, Sequence):
        raise JobOperationError("job Issue scan is invalid")
    assigned: list[OperatorJobSummary] = []
    snapshots: list[SubmissionGateSnapshot] = []
    seen: set[int] = set()
    for issue in issues:
        if not isinstance(issue, GitHubIssue) or issue.number in seen:
            raise JobOperationError("job Issue scan is invalid")
        seen.add(issue.number)
        if operator not in issue.assignees:
            continue
        status = _managed_status(issue)
        if status == "assigned":
            snapshot = _gate(
                issue.number,
                operator=operator,
                github=github,
                configuration=configuration,
                allowed_statuses={"assigned"},
                revalidate_commit=False,
                automation_actor=automation_actor,
                expected_issue=issue,
            )
            assigned.append(OperatorJobSummary(snapshot.issue, status, None, None))
            continue
        snapshot = _authorized_lifecycle_gate(
            issue.number,
            operator=operator,
            github=github,
            configuration=configuration,
            automation_actor=automation_actor,
            expected_issue=issue,
        )
        snapshots.append(snapshot)
    assigned.sort(key=lambda summary: summary.issue)
    if not snapshots:
        return tuple(assigned)
    ids = [
        snapshot.lifecycle.attempts[-1].slurm_job_id for snapshot in snapshots if snapshot.lifecycle
    ]
    try:
        evidence = slurm.query(tuple(ids))
    except Exception:
        raise JobOperationError("authoritative Slurm query failed") from None
    summaries: list[OperatorJobSummary] = []
    for snapshot in snapshots:
        assert snapshot.lifecycle is not None
        current = snapshot.lifecycle
        job_id = current.attempts[-1].slurm_job_id
        job = evidence.get(job_id)
        if not isinstance(job, SlurmJob):
            raise JobOperationError("authoritative Slurm evidence is incomplete")
        repaired = reconcile_lifecycle(current, job.lifecycle_state, now=now)
        if repaired != current:
            _persist_lifecycle(
                github,
                snapshot.issue,
                repaired,
                operator=operator,
            )
        _repair_labels_to_lifecycle(github, snapshot.issue, repaired.current_state)
        requester = normalize_actor_login(snapshot.request.requester)
        if repaired.current_state == "submitted":
            notice = _submission_notice(
                requester,
                snapshot.issue,
                repaired.attempts[-1],
            )
        elif repaired.current_state == "running":
            notice = f"@{requester} eduLLM job #{snapshot.issue} started."
        else:
            notice = _terminal_notice(requester, snapshot.issue, repaired)
        _ensure_notice(
            github,
            snapshot.issue,
            notice,
            actor=operator,
            actor_is_bot=False,
        )
        latest = repaired.attempts[-1]
        summaries.append(
            OperatorJobSummary(
                issue=repaired.issue,
                status=repaired.current_state,
                slurm_job_id=latest.slurm_job_id,
                wandb_url=latest.wandb_url,
            )
        )
    try:
        slurm.reconcile_offline_tracking()
    except Exception:
        pass
    summaries.sort(key=lambda summary: summary.issue, reverse=True)
    return tuple(assigned + summaries)


def authorized_logs(
    issue: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    slurm: Any,
    automation_actor: str = AUTOMATION_ACTOR,
) -> str:
    """Authorize an Issue binding and return only its recorded redacted log."""
    snapshot = _authorized_lifecycle_gate(
        issue,
        operator=operator,
        github=github,
        configuration=configuration,
        automation_actor=automation_actor,
    )
    assert snapshot.lifecycle is not None
    path = snapshot.lifecycle.attempts[-1].log_path
    try:
        return slurm.read_log(path)
    except Exception:
        raise JobOperationError("authorized log retrieval failed") from None


def deliver_terminal_notifications(
    *,
    github: Any,
    configuration: GateConfiguration,
    notifier: Any,
    now: datetime,
    automation_actor: str = AUTOMATION_ACTOR,
) -> tuple[LifecycleState, ...]:
    """
    Deliver pending terminal Slack events with durable honest outcome state.

    ``ambiguous`` is persisted before the webhook call, so a lost response is
    never blindly retried. A definite non-delivery returns the event to
    ``pending``; success is persisted as ``sent``.
    """
    _valid_time(now)
    _validate_configuration(configuration)
    operators = {
        operator.github: operator for operator in configuration.operators if operator.enabled
    }
    try:
        issues = github.list_active_queue_issues()
    except Exception:
        raise JobOperationError("terminal notification Issue scan failed") from None
    if not isinstance(issues, Sequence) or len(issues) > MAX_SLURM_JOBS:
        raise JobOperationError("terminal notification Issue scan is invalid")

    results: list[LifecycleState] = []
    seen: set[int] = set()
    for issue in issues:
        if not isinstance(issue, GitHubIssue) or issue.number in seen:
            raise JobOperationError("terminal notification Issue scan is invalid")
        seen.add(issue.number)
        status = _managed_status(issue)
        if status not in _TERMINAL_STATES:
            continue
        if len(issue.assignees) > 64:
            raise JobOperationError("terminal notification Issue binding is invalid")
        operator_names = [name for name in issue.assignees if name in operators]
        if len(operator_names) != 1:
            raise JobOperationError("terminal notification operator binding is invalid")
        operator = operators[operator_names[0]]
        snapshot = _authorized_lifecycle_gate(
            issue.number,
            operator=operator.github,
            github=github,
            configuration=configuration,
            automation_actor=automation_actor,
        )
        assert snapshot.lifecycle is not None
        state = snapshot.lifecycle
        if state.notification.status in {"sent", "ambiguous"}:
            results.append(state)
            continue
        if state.notification.status != "pending" or state.notification.event != status:
            raise JobOperationError("terminal notification state is invalid")

        in_flight = replace(
            state,
            updated_at=max(state.updated_at, now),
            notification=NotificationRecord(status, "ambiguous", now),
        )
        _persist_lifecycle(
            github,
            issue.number,
            in_flight,
            operator=state.operator,
        )
        try:
            notifier.terminal(
                issue=issue.number,
                operator_slack_id=operator.slack_user_id,
                state=status,
            )
        except SlackNotificationError as error:
            if not error.ambiguous:
                pending = replace(
                    in_flight,
                    notification=NotificationRecord(status, "pending", now),
                )
                _persist_lifecycle(
                    github,
                    issue.number,
                    pending,
                    operator=state.operator,
                )
            raise JobOperationError("terminal Slack notification failed") from None
        except Exception:
            raise JobOperationError("terminal Slack notification failed") from None
        sent = replace(
            in_flight,
            notification=NotificationRecord(status, "sent", now),
        )
        _persist_lifecycle(
            github,
            issue.number,
            sent,
            operator=state.operator,
        )
        results.append(sent)
    return tuple(results)


def jobs(
    *,
    mine: bool,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    slurm: Any,
    now: datetime,
) -> tuple[OperatorJobSummary, ...]:
    """Public operator service for authorized status listing and repair."""
    return reconcile_operator_jobs(
        operator=operator,
        github=github,
        configuration=configuration,
        slurm=slurm,
        now=now,
        mine=mine,
    )


def logs(
    issue: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    slurm: Any,
) -> str:
    """Public operator service for one authorized recorded log."""
    return authorized_logs(
        issue,
        operator=operator,
        github=github,
        configuration=configuration,
        slurm=slurm,
    )


def stop(
    issue: int,
    *,
    operator: str,
    github: Any,
    configuration: GateConfiguration,
    slurm: Any,
    now: datetime,
) -> LifecycleState:
    """Public operator service for one authorized idempotent cancellation."""
    return stop_job(
        issue,
        operator=operator,
        github=github,
        configuration=configuration,
        slurm=slurm,
        now=now,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Remote-only helper for bounded redacted log access."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m edullm.jobs")
    commands = parser.add_subparsers(dest="command", required=True)
    read_log = commands.add_parser("read-log")
    read_log.add_argument("path")
    arguments = parser.parse_args(argv)
    try:
        logs_root = Path.home() / "orcd" / "scratch" / "edullm" / "logs"
        text = read_authorized_log(arguments.path, logs_root=logs_root)
    except JobOperationError:
        print("log read failed", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
