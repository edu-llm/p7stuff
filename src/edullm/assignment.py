"""
Fail-closed least-loaded assignment and timeout automation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from edullm.github import GitHubError, GitHubIssue, IssueComment
from edullm.models import JobRequest, JobStatus, Operator
from edullm.notifications import SlackNotificationError
from edullm.policy import Policy
from edullm.request_parser import IssueParseError, parse_issue
from edullm.validation import (
    MAX_INTEGER_TOKEN_CHARS,
    STATUS_MARKER,
    StatusCommentError,
    validated_status_for_request,
)

ASSIGNMENT_MARKER = "<!-- edullm-assignment:v1 -->"
AUTOMATION_ACTOR = "github-actions[bot]"
MAX_ASSIGNMENT_COMMENT_CHARS = 65_536
MAX_ASSIGNMENT_HISTORY = 32

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_LOGIN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_BOT_LOGIN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\[bot\]\Z")
_SLACK_USER_ID = re.compile(r"[UW][A-Z0-9]{8,20}\Z")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_ASSIGNMENT_MARKER_LINE = re.compile(rf"^{re.escape(ASSIGNMENT_MARKER)}$", re.MULTILINE)
_STATUS_MARKER_LINE = re.compile(rf"^{re.escape(STATUS_MARKER)}$", re.MULTILINE)
_SLACK_STATUSES = frozenset({"pending", "sent", "failed", "ambiguous"})
_ACTIVE_STATUSES = frozenset({"assigned", "submitted", "running"})
_MANAGED_STATUS_LABELS = frozenset(f"status:{status.value}" for status in JobStatus)
_ASSIGNMENT_FIELDS = {
    "assigned_at",
    "history",
    "issue",
    "operator_github",
    "reminder_at",
    "request_digest",
    "slack_status",
}
_HISTORY_FIELDS = {"previous_operator", "reassigned_at"}


class AssignmentStateError(RuntimeError):
    """A sanitized fail-closed assignment state or orchestration error."""


class NoEligibleOperatorError(ValueError):
    """Raised when a valid load snapshot has no operator capacity."""


class _OversizedIntegerToken(ValueError):
    """Internal signal raised before converting an oversized JSON integer."""


@dataclass(frozen=True)
class OperatorLoad:
    """Current active capacity attributed to one enabled operator."""

    github: str
    active_gpus: int
    active_jobs: int
    rotation: int


@dataclass(frozen=True)
class ActiveJob:
    """The operator and requested GPU count for one active queue job."""

    operator_github: str
    gpu_count: int


@dataclass(frozen=True)
class AssignmentHistory:
    """One prior operator displaced by a timed-out reassignment."""

    previous_operator: str
    reassigned_at: datetime


@dataclass(frozen=True)
class AssignmentState:
    """Canonical assignment metadata bound to one validated request digest."""

    issue: int
    request_digest: str
    operator_github: str
    assigned_at: datetime
    reminder_at: datetime | None
    history: tuple[AssignmentHistory, ...]
    slack_status: str


@dataclass(frozen=True)
class AssignmentResult:
    """Sanitized result of one assignment, reminder, or reassignment attempt."""

    issue: int
    action: str
    operator: str | None
    operational_error: bool


@dataclass(frozen=True)
class _IssueContext:
    issue: GitHubIssue
    request: JobRequest
    comments: tuple[IssueComment, ...]
    automation_actor: str
    status: str
    assignment_comment: IssueComment | None
    assignment: AssignmentState | None


def select_operator(
    loads: Sequence[OperatorLoad],
    *,
    incoming_gpus: int,
    max_gpus: int,
    exclude: set[str] | None = None,
) -> OperatorLoad:
    """
    Select an eligible operator by GPU load, job load, rotation, then login.

    :param loads: A complete enabled-operator load snapshot.
    :param incoming_gpus: The positive requested GPU count.
    :param max_gpus: The positive per-operator active GPU limit.
    :param exclude: Canonical logins that cannot receive this assignment.

    :returns: The least-loaded eligible operator.

    :raises ValueError: If inputs or load state are malformed.
    :raises NoEligibleOperatorError: If no operator has capacity.
    """
    if (
        type(incoming_gpus) is not int
        or incoming_gpus <= 0
        or type(max_gpus) is not int
        or max_gpus <= 0
        or incoming_gpus > max_gpus
    ):
        raise ValueError("assignment capacity values must be positive integers")
    if not isinstance(loads, Sequence) or not loads:
        raise ValueError("operator loads must be a non-empty sequence")
    excluded = set() if exclude is None else exclude
    if type(excluded) is not set or any(not _valid_login(login) for login in excluded):
        raise ValueError("operator exclusions are malformed")

    seen: set[str] = set()
    validated: list[OperatorLoad] = []
    for load in loads:
        if (
            not isinstance(load, OperatorLoad)
            or not _valid_login(load.github)
            or load.github in seen
            or type(load.active_gpus) is not int
            or load.active_gpus < 0
            or load.active_gpus > max_gpus
            or type(load.active_jobs) is not int
            or load.active_jobs < 0
            or type(load.rotation) is not int
            or load.rotation < 0
        ):
            raise ValueError("operator load state is malformed")
        seen.add(load.github)
        validated.append(load)
    if not excluded <= seen:
        raise ValueError("operator exclusions are not in the load snapshot")

    eligible = [
        load
        for load in validated
        if load.github not in excluded and load.active_gpus + incoming_gpus <= max_gpus
    ]
    if not eligible:
        raise NoEligibleOperatorError("no eligible operators")
    return min(
        eligible,
        key=lambda load: (
            load.active_gpus,
            load.active_jobs,
            load.rotation,
            load.github,
        ),
    )


def derive_operator_loads(
    operators: Sequence[Operator],
    active_jobs: Iterable[ActiveJob],
) -> tuple[OperatorLoad, ...]:
    """
    Derive immutable loads for every enabled operator.

    :param operators: The protected operator roster.
    :param active_jobs: Current validated assigned/submitted/running jobs.

    :returns: Loads in protected roster order.

    :raises ValueError: If roster or active state is malformed.
    """
    enabled = _enabled_operator_map(operators)
    loads = {login: [0, 0, operator.rotation_order] for login, operator in enabled.items()}
    for job in active_jobs:
        if (
            not isinstance(job, ActiveJob)
            or not _valid_login(job.operator_github)
            or type(job.gpu_count) is not int
            or job.gpu_count <= 0
        ):
            raise ValueError("active job state is malformed")
        if job.operator_github not in loads:
            raise ValueError("active job references an unknown or disabled operator")
        current = loads[job.operator_github]
        current[0] += job.gpu_count
        current[1] += 1
    return tuple(
        OperatorLoad(login, values[0], values[1], values[2]) for login, values in loads.items()
    )


def build_assignment_comment(state: AssignmentState) -> str:
    """
    Build the exact canonical v1 assignment machine comment.

    :param state: Validated assignment metadata.

    :returns: Marker plus canonical JSON.

    :raises AssignmentStateError: If any state field is malformed or unbounded.
    """
    _validate_assignment_state(state)
    payload = {
        "assigned_at": _format_timestamp(state.assigned_at),
        "history": [
            {
                "previous_operator": item.previous_operator,
                "reassigned_at": _format_timestamp(item.reassigned_at),
            }
            for item in state.history
        ],
        "issue": state.issue,
        "operator_github": state.operator_github,
        "reminder_at": (
            None if state.reminder_at is None else _format_timestamp(state.reminder_at)
        ),
        "request_digest": state.request_digest,
        "slack_status": state.slack_status,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    comment = f"{ASSIGNMENT_MARKER}\n{encoded}"
    if len(comment) > MAX_ASSIGNMENT_COMMENT_CHARS:
        raise AssignmentStateError("assignment comment is too large")
    return comment


def parse_assignment_comment(comment: str) -> AssignmentState:
    """
    Parse and integrity-check one exact canonical v1 assignment comment.

    :param comment: The complete comment body.

    :returns: Canonical immutable assignment state.

    :raises AssignmentStateError: If marker, JSON, schema, or values are invalid.
    """
    if type(comment) is not str or len(comment) > MAX_ASSIGNMENT_COMMENT_CHARS:
        raise AssignmentStateError("assignment marker is missing")
    marker_count = len(_ASSIGNMENT_MARKER_LINE.findall(comment))
    if marker_count != 1 or not comment.startswith(ASSIGNMENT_MARKER + "\n"):
        raise AssignmentStateError("assignment marker must appear exactly once")
    encoded = comment[len(ASSIGNMENT_MARKER) + 1 :]
    try:
        payload = json.loads(encoded, parse_int=_bounded_json_integer)
    except (_OversizedIntegerToken, ValueError, RecursionError):
        raise AssignmentStateError("assignment payload is not valid JSON") from None
    if type(payload) is not dict:
        raise AssignmentStateError("assignment payload fields are invalid")
    if encoded != json.dumps(payload, sort_keys=True, separators=(",", ":")):
        raise AssignmentStateError("assignment payload must use canonical JSON")
    data = cast(dict[str, object], payload)
    if set(data) != _ASSIGNMENT_FIELDS:
        raise AssignmentStateError("assignment payload fields are invalid")
    history_data = data["history"]
    if type(history_data) is not list or len(history_data) > MAX_ASSIGNMENT_HISTORY:
        raise AssignmentStateError("assignment history is invalid")
    history: list[AssignmentHistory] = []
    for value in cast(list[object], history_data):
        if type(value) is not dict or set(value) != _HISTORY_FIELDS:
            raise AssignmentStateError("assignment history is invalid")
        row = cast(dict[str, object], value)
        history.append(
            AssignmentHistory(
                previous_operator=cast(str, row["previous_operator"]),
                reassigned_at=_parse_timestamp(row["reassigned_at"]),
            )
        )
    reminder_value = data["reminder_at"]
    state = AssignmentState(
        issue=cast(int, data["issue"]),
        request_digest=cast(str, data["request_digest"]),
        operator_github=cast(str, data["operator_github"]),
        assigned_at=_parse_timestamp(data["assigned_at"]),
        reminder_at=(None if reminder_value is None else _parse_timestamp(reminder_value)),
        history=tuple(history),
        slack_status=cast(str, data["slack_status"]),
    )
    _validate_assignment_state(state)
    return state


def assign_ready_issues(
    *,
    github: Any,
    operators: Sequence[Operator],
    policy: Policy,
    now: datetime,
    notifier: Any,
    automation_actor: str = AUTOMATION_ACTOR,
) -> tuple[AssignmentResult, ...]:
    """
    Scan and assign every currently ready queue Issue.

    An empty enabled roster returns without reading GitHub. Workflow-level
    repository concurrency is required around this function.

    :param github: A GitHub client-compatible boundary.
    :param operators: Protected operator configuration.
    :param policy: Trusted queue policy and capacity.
    :param now: Canonical current UTC whole-second timestamp.
    :param notifier: An injectable Slack notifier.
    :param automation_actor: The exact trusted GitHub machine-comment author.

    :returns: One sanitized result per attempted ready Issue.
    """
    enabled = _enabled_operator_map(operators)
    _validate_now(now)
    _validate_policy_capacity(policy)
    _validate_automation_actor(automation_actor)
    if not enabled:
        return ()
    try:
        snapshots = github.list_active_queue_issues()
    except GitHubError:
        return (AssignmentResult(0, "error", None, True),)

    results: list[AssignmentResult] = []
    for snapshot in snapshots:
        if "status:ready" not in snapshot.labels:
            continue
        try:
            results.append(
                _assign_one(
                    snapshot.number,
                    github=github,
                    enabled=enabled,
                    policy=policy,
                    now=now,
                    notifier=notifier,
                    automation_actor=automation_actor,
                )
            )
        except (AssignmentStateError, GitHubError, IssueParseError, StatusCommentError, ValueError):
            results.append(AssignmentResult(snapshot.number, "error", None, True))
    return tuple(results)


def process_assignment_timeouts(
    *,
    github: Any,
    operators: Sequence[Operator],
    policy: Policy,
    now: datetime,
    notifier: Any,
    automation_actor: str = AUTOMATION_ACTOR,
) -> tuple[AssignmentResult, ...]:
    """
    Scan assigned Issues for a one-time reminder or timed-out reassignment.

    Submitted and later statuses are never regressed. Workflow-level repository
    concurrency must be the same group used for initial assignment.

    :param github: A GitHub client-compatible boundary.
    :param operators: Protected operator configuration.
    :param policy: Trusted reminder, reassignment, and capacity policy.
    :param now: Canonical current UTC whole-second timestamp.
    :param notifier: An injectable Slack notifier.
    :param automation_actor: The exact trusted GitHub machine-comment author.

    :returns: One sanitized result per currently assigned Issue.
    """
    enabled = _enabled_operator_map(operators)
    _validate_now(now)
    _validate_policy_capacity(policy)
    _validate_automation_actor(automation_actor)
    if not enabled:
        return ()
    try:
        snapshots = github.list_active_queue_issues()
    except GitHubError:
        return (AssignmentResult(0, "error", None, True),)

    results: list[AssignmentResult] = []
    for snapshot in snapshots:
        try:
            managed = _managed_statuses(snapshot)
            if managed != {"assigned"}:
                continue
            context = _current_context(
                github,
                snapshot.number,
                allowed_statuses=({"assigned"},),
                automation_actor=automation_actor,
            )
            state = _require_assignment(context, enabled)
            _reject_future_assignment_time(state, now)
            _reconcile_current_assignees(github, context, state)
            context = _refresh_context(
                github,
                context,
                allowed_statuses=({"assigned"},),
            )
            state = _require_assignment(context, enabled)
            _reject_future_assignment_time(state, now)
            _ensure_notice(
                github,
                context,
                _assignment_notice(state.operator_github, len(state.history)),
            )
            context = _refresh_context(
                github,
                context,
                allowed_statuses=({"assigned"},),
            )
            state = _require_assignment(context, enabled)
            _reject_future_assignment_time(state, now)
            age = now - state.assigned_at
            if age >= timedelta(minutes=policy.reassign_after_minutes):
                results.append(
                    _reassign(
                        context,
                        state,
                        github=github,
                        enabled=enabled,
                        policy=policy,
                        now=now,
                        notifier=notifier,
                    )
                )
            elif (
                age >= timedelta(minutes=policy.reminder_after_minutes)
                and state.reminder_at is None
            ):
                results.append(
                    _remind(
                        context,
                        state,
                        github=github,
                        now=now,
                    )
                )
            elif state.slack_status == "failed":
                results.append(
                    _send_notification(
                        github,
                        context,
                        state,
                        enabled[state.operator_github],
                        notifier=notifier,
                        kind="reassignment" if state.history else "assignment",
                        success_action="notification_retried",
                        send_pending=False,
                    )
                )
            else:
                results.append(
                    AssignmentResult(
                        snapshot.number,
                        "unchanged",
                        state.operator_github,
                        False,
                    )
                )
        except (AssignmentStateError, GitHubError, IssueParseError, StatusCommentError, ValueError):
            results.append(AssignmentResult(snapshot.number, "error", None, True))
    return tuple(results)


def _assign_one(
    issue_number: int,
    *,
    github: Any,
    enabled: dict[str, Operator],
    policy: Policy,
    now: datetime,
    notifier: Any,
    automation_actor: str,
) -> AssignmentResult:
    context = _current_context(
        github,
        issue_number,
        allowed_statuses=({"ready"}, {"ready", "assigned"}),
        automation_actor=automation_actor,
    )
    if _managed_statuses(context.issue) == {"ready", "assigned"}:
        state = _require_assignment(context, enabled)
        _require_in_progress_assignment(context, state)
        _reject_future_assignment_time(state, now)
    send_pending = context.assignment is None
    if context.assignment is None:
        loads = _fresh_loads(
            github,
            enabled,
            automation_actor=context.automation_actor,
        )
        try:
            selected = select_operator(
                loads,
                incoming_gpus=context.request.gpu_count,
                max_gpus=policy.max_gpu_count,
            )
        except NoEligibleOperatorError:
            return AssignmentResult(issue_number, "no_capacity", None, False)
        operator = enabled[selected.github]
        state = AssignmentState(
            issue=issue_number,
            request_digest=context.request.digest,
            operator_github=operator.github,
            assigned_at=now,
            reminder_at=None,
            history=(),
            slack_status="pending",
        )
        context = _refresh_context(
            github,
            context,
            allowed_statuses=({"ready"}, {"ready", "assigned"}),
        )
        if context.assignment is not None:
            raise AssignmentStateError("assignment changed during scoring")
        _persist_assignment_state(github, context, state)
    else:
        state = _require_assignment(context, enabled)
        _reject_future_assignment_time(state, now)
        operator = enabled[state.operator_github]

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"ready"}, {"ready", "assigned"}),
    )
    state = _require_assignment(context, enabled)
    _reject_future_assignment_time(state, now)
    original_assignees = set(context.issue.assignees)
    if operator.github not in context.issue.assignees:
        try:
            github.add_issue_assignee(issue_number, operator.github)
        except GitHubError:
            pass
        confirmed = github.fetch_issue(issue_number)
        if operator.github not in confirmed.assignees or not original_assignees <= set(
            confirmed.assignees
        ):
            raise AssignmentStateError("assignment assignee postcondition failed")

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"ready"}, {"ready", "assigned"}),
    )
    current = _require_assignment(context, enabled)
    if current != state:
        raise AssignmentStateError("assignment changed during assignment")
    _ensure_notice(
        github,
        context,
        _assignment_notice(operator.github, len(state.history)),
    )

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"ready"}, {"ready", "assigned"}),
    )
    current = _require_assignment(context, enabled)
    if current != state:
        raise AssignmentStateError("assignment changed during assignment")
    _set_assigned(github, context)
    context = _refresh_context(github, context, allowed_statuses=({"assigned"},))
    current = _require_assignment(context, enabled)
    if current != state:
        raise AssignmentStateError("assignment changed during assignment")
    return _send_notification(
        github,
        context,
        current,
        operator,
        notifier=notifier,
        kind="assignment",
        success_action="assigned",
        send_pending=send_pending,
    )


def _remind(
    context: _IssueContext,
    state: AssignmentState,
    *,
    github: Any,
    now: datetime,
) -> AssignmentResult:
    body = _reminder_notice(state.operator_github, len(state.history))
    _ensure_notice(github, context, body)
    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current = _require_assignment(context)
    if current != state:
        raise AssignmentStateError("assignment changed during reminder")
    if current.reminder_at is None:
        updated = AssignmentState(
            issue=current.issue,
            request_digest=current.request_digest,
            operator_github=current.operator_github,
            assigned_at=current.assigned_at,
            reminder_at=now,
            history=current.history,
            slack_status=current.slack_status,
        )
        _persist_assignment_state(github, context, updated)
    return AssignmentResult(context.issue.number, "reminded", state.operator_github, False)


def _reassign(
    context: _IssueContext,
    state: AssignmentState,
    *,
    github: Any,
    enabled: dict[str, Operator],
    policy: Policy,
    now: datetime,
    notifier: Any,
) -> AssignmentResult:
    loads = _fresh_loads(
        github,
        enabled,
        automation_actor=context.automation_actor,
    )
    try:
        selected = select_operator(
            loads,
            incoming_gpus=context.request.gpu_count,
            max_gpus=policy.max_gpu_count,
            exclude={state.operator_github},
        )
    except NoEligibleOperatorError:
        return AssignmentResult(
            context.issue.number,
            "no_capacity",
            state.operator_github,
            False,
        )
    operator = enabled[selected.github]

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current = _require_assignment(context, enabled)
    if current != state:
        raise AssignmentStateError("assignment changed during reassignment")
    preserved = set(context.issue.assignees)
    unrelated_assignees = preserved - {state.operator_github, operator.github}
    if operator.github not in context.issue.assignees:
        try:
            github.add_issue_assignee(context.issue.number, operator.github)
        except GitHubError:
            pass
        confirmed = github.fetch_issue(context.issue.number)
        if operator.github not in confirmed.assignees or not preserved <= set(confirmed.assignees):
            _best_effort_restore_assignees(
                github,
                context,
                state,
                preserved,
            )
            raise AssignmentStateError("reassignment assignee postcondition failed")

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current = _require_assignment(context, enabled)
    if current != state:
        raise AssignmentStateError("assignment changed during reassignment")
    updated = AssignmentState(
        issue=state.issue,
        request_digest=state.request_digest,
        operator_github=operator.github,
        assigned_at=now,
        reminder_at=None,
        history=state.history + (AssignmentHistory(state.operator_github, now),),
        slack_status="pending",
    )
    _persist_assignment_state(github, context, updated)

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current = _require_assignment(context, enabled)
    if current != updated:
        raise AssignmentStateError("assignment changed during reassignment")
    _ensure_notice(
        github,
        context,
        _assignment_notice(operator.github, len(updated.history)),
    )

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current = _require_assignment(context, enabled)
    if current != updated:
        raise AssignmentStateError("assignment changed during reassignment")
    if state.operator_github in context.issue.assignees:
        try:
            github.remove_issue_assignee(context.issue.number, state.operator_github)
        except GitHubError:
            pass
        confirmed = github.fetch_issue(context.issue.number)
        if (
            state.operator_github in confirmed.assignees
            or operator.github not in confirmed.assignees
            or not unrelated_assignees <= set(confirmed.assignees)
        ):
            _best_effort_restore_assignees(
                github,
                context,
                updated,
                unrelated_assignees,
            )
            raise AssignmentStateError("reassignment assignee postcondition failed")

    context = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current = _require_assignment(context, enabled)
    if current != updated:
        raise AssignmentStateError("assignment changed during reassignment")
    if not unrelated_assignees <= set(context.issue.assignees):
        _best_effort_restore_assignees(
            github,
            context,
            updated,
            unrelated_assignees,
        )
        raise AssignmentStateError("reassignment assignee postcondition failed")
    return _send_notification(
        github,
        context,
        current,
        operator,
        notifier=notifier,
        kind="reassignment",
        success_action="reassigned",
        send_pending=True,
    )


def _send_notification(
    github: Any,
    context: _IssueContext,
    state: AssignmentState,
    operator: Operator,
    *,
    notifier: Any,
    kind: str,
    success_action: str,
    send_pending: bool,
) -> AssignmentResult:
    if state.slack_status == "sent":
        return AssignmentResult(context.issue.number, success_action, operator.github, False)
    if state.slack_status == "ambiguous":
        return AssignmentResult(
            context.issue.number,
            f"{success_action}_notification_pending",
            operator.github,
            False,
        )
    if state.slack_status == "pending" and not send_pending:
        return AssignmentResult(
            context.issue.number,
            f"{success_action}_notification_pending",
            operator.github,
            False,
        )
    try:
        notifier.assignment(
            issue=context.issue.number,
            title=context.issue.title,
            operator_slack_id=operator.slack_user_id,
            kind=kind,
        )
        slack_status = "sent"
    except SlackNotificationError as error:
        slack_status = "ambiguous" if error.ambiguous else "failed"

    current = _refresh_context(
        github,
        context,
        allowed_statuses=({"assigned"},),
    )
    current_state = _require_assignment(current)
    if (
        current_state.request_digest != state.request_digest
        or current_state.operator_github != state.operator_github
        or current_state.assigned_at != state.assigned_at
        or current_state.history != state.history
    ):
        raise AssignmentStateError("assignment changed during notification")
    updated = AssignmentState(
        issue=current_state.issue,
        request_digest=current_state.request_digest,
        operator_github=current_state.operator_github,
        assigned_at=current_state.assigned_at,
        reminder_at=current_state.reminder_at,
        history=current_state.history,
        slack_status=slack_status,
    )
    _persist_assignment_state(github, current, updated)
    action = success_action if slack_status == "sent" else f"{success_action}_notification_pending"
    return AssignmentResult(context.issue.number, action, operator.github, False)


def _fresh_loads(
    github: Any,
    enabled: dict[str, Operator],
    *,
    automation_actor: str,
) -> tuple[OperatorLoad, ...]:
    active_jobs: list[ActiveJob] = []
    snapshots = github.list_active_queue_issues()
    for snapshot in snapshots:
        current = github.fetch_issue(snapshot.number)
        managed = _managed_statuses(current)
        if managed == {"ready", "assigned"}:
            context = _context_from_issue(
                github,
                current,
                allowed_statuses=({"ready", "assigned"},),
                automation_actor=automation_actor,
            )
            state = _require_assignment(context, enabled)
            _require_in_progress_assignment(context, state)
            active_jobs.append(ActiveJob(state.operator_github, context.request.gpu_count))
            continue
        if len(managed) != 1:
            raise AssignmentStateError("managed status labels are malformed")
        status = next(iter(managed))
        if status == "ready":
            context = _context_from_issue(
                github,
                current,
                allowed_statuses=({"ready"},),
                automation_actor=automation_actor,
            )
            if context.assignment is not None:
                state = _require_assignment(context, enabled)
                active_jobs.append(ActiveJob(state.operator_github, context.request.gpu_count))
            continue
        if status not in _ACTIVE_STATUSES:
            continue
        context = _context_from_issue(
            github,
            current,
            allowed_statuses=({status},),
            automation_actor=automation_actor,
        )
        state = _require_assignment(context, enabled)
        active_jobs.append(ActiveJob(state.operator_github, context.request.gpu_count))
    return derive_operator_loads(tuple(enabled.values()), active_jobs)


def _current_context(
    github: Any,
    issue_number: int,
    *,
    allowed_statuses: tuple[set[str], ...],
    automation_actor: str = AUTOMATION_ACTOR,
) -> _IssueContext:
    issue = github.fetch_issue(issue_number)
    return _context_from_issue(
        github,
        issue,
        allowed_statuses=allowed_statuses,
        automation_actor=automation_actor,
    )


def _refresh_context(
    github: Any,
    context: _IssueContext,
    *,
    allowed_statuses: tuple[set[str], ...],
) -> _IssueContext:
    return _current_context(
        github,
        context.issue.number,
        allowed_statuses=allowed_statuses,
        automation_actor=context.automation_actor,
    )


def _context_from_issue(
    github: Any,
    issue: GitHubIssue,
    *,
    allowed_statuses: tuple[set[str], ...],
    automation_actor: str = AUTOMATION_ACTOR,
) -> _IssueContext:
    if "edullm-job" not in issue.labels:
        raise AssignmentStateError("Issue is not in the eduLLM queue")
    managed = _managed_statuses(issue)
    if managed not in allowed_statuses:
        raise AssignmentStateError("managed status labels are malformed")
    try:
        request = parse_issue(
            issue.body,
            issue_number=issue.number,
            requester=issue.requester,
        )
    except IssueParseError:
        raise AssignmentStateError("current Issue request is malformed") from None
    comments = github.list_issue_comments(issue.number)
    status_comment = _exact_marker_comment(
        comments,
        marker_line=_STATUS_MARKER_LINE,
        kind="status",
        required=True,
        expected_author=automation_actor,
    )
    assert status_comment is not None
    try:
        validated_status_for_request(status_comment.body, request)
    except StatusCommentError:
        raise AssignmentStateError("validated status is stale or malformed") from None

    assignment_comment = _exact_marker_comment(
        comments,
        marker_line=_ASSIGNMENT_MARKER_LINE,
        kind="assignment",
        required=False,
        expected_author=automation_actor,
    )
    assignment = None
    if assignment_comment is not None:
        assignment = parse_assignment_comment(assignment_comment.body)
        if assignment.issue != issue.number or assignment.request_digest != request.digest:
            raise AssignmentStateError("assignment is stale for the current request")
    status = next(iter(managed - {"ready"}), "ready") if len(managed) == 1 else "transition"
    return _IssueContext(
        issue=issue,
        request=request,
        comments=comments,
        automation_actor=automation_actor,
        status=status,
        assignment_comment=assignment_comment,
        assignment=assignment,
    )


def _persist_assignment_state(
    github: Any,
    context: _IssueContext,
    state: AssignmentState,
) -> IssueComment:
    if state.issue != context.issue.number or state.request_digest != context.request.digest:
        raise AssignmentStateError("assignment is stale for the current request")
    body = build_assignment_comment(state)
    existing = _exact_marker_comment(
        context.comments,
        marker_line=_ASSIGNMENT_MARKER_LINE,
        kind="assignment",
        required=False,
        expected_author=context.automation_actor,
    )
    if existing is None:
        try:
            persisted = github.create_issue_comment(context.issue.number, body)
        except GitHubError:
            comments = github.list_issue_comments(context.issue.number)
            reconciled = _exact_marker_comment(
                comments,
                marker_line=_ASSIGNMENT_MARKER_LINE,
                kind="assignment",
                required=True,
                expected_author=context.automation_actor,
            )
            if reconciled is None or reconciled.body != body:
                raise
            persisted = reconciled
    else:
        try:
            persisted = github.update_issue_comment(existing.id, body)
        except GitHubError:
            comments = github.list_issue_comments(context.issue.number)
            reconciled = _exact_marker_comment(
                comments,
                marker_line=_ASSIGNMENT_MARKER_LINE,
                kind="assignment",
                required=True,
                expected_author=context.automation_actor,
            )
            if reconciled is None or reconciled.id != existing.id or reconciled.body != body:
                raise
            persisted = reconciled
    if (
        not persisted.author_is_bot
        or persisted.author != context.automation_actor
        or persisted.body != body
    ):
        raise AssignmentStateError("persisted assignment comment is invalid")
    comments = github.list_issue_comments(context.issue.number)
    confirmed = _exact_marker_comment(
        comments,
        marker_line=_ASSIGNMENT_MARKER_LINE,
        kind="assignment",
        required=True,
        expected_author=context.automation_actor,
    )
    if confirmed is None or confirmed.id != persisted.id or confirmed.body != body:
        raise AssignmentStateError("persisted assignment comment postcondition failed")
    return confirmed


def _set_assigned(github: Any, context: _IssueContext) -> None:
    issue_number = context.issue.number
    try:
        github.add_issue_status_label(issue_number, "status:assigned")
    except GitHubError:
        current = github.fetch_issue(issue_number)
        if "status:assigned" not in current.labels:
            raise
    transitional = _refresh_context(
        github,
        context,
        allowed_statuses=({"ready", "assigned"}, {"assigned"}),
    )
    _require_assignment(transitional)
    try:
        github.remove_issue_status_label(issue_number, "status:ready")
    except GitHubError:
        current = github.fetch_issue(issue_number)
        if "status:ready" in current.labels:
            raise
    final = _refresh_context(github, context, allowed_statuses=({"assigned"},))
    _require_assignment(final)
    if "status:assigned" not in final.issue.labels or "status:ready" in final.issue.labels:
        raise AssignmentStateError("assigned status postcondition failed")


def _reconcile_current_assignees(
    github: Any,
    context: _IssueContext,
    state: AssignmentState,
) -> None:
    preserved = set(context.issue.assignees)
    if state.operator_github not in context.issue.assignees:
        try:
            github.add_issue_assignee(context.issue.number, state.operator_github)
        except GitHubError:
            pass
        confirmed = github.fetch_issue(context.issue.number)
        if state.operator_github not in confirmed.assignees or not preserved <= set(
            confirmed.assignees
        ):
            _best_effort_restore_assignees(
                github,
                context,
                state,
                preserved,
            )
            raise AssignmentStateError("assignment assignee postcondition failed")
        context = _refresh_context(
            github,
            context,
            allowed_statuses=({"assigned"},),
        )
    if state.history:
        prior = state.history[-1].previous_operator
        unrelated_assignees = set(context.issue.assignees) - {
            prior,
            state.operator_github,
        }
        if prior != state.operator_github and prior in context.issue.assignees:
            try:
                github.remove_issue_assignee(context.issue.number, prior)
            except GitHubError:
                pass
            confirmed = github.fetch_issue(context.issue.number)
            if (
                prior in confirmed.assignees
                or state.operator_github not in confirmed.assignees
                or not unrelated_assignees <= set(confirmed.assignees)
            ):
                _best_effort_restore_assignees(
                    github,
                    context,
                    state,
                    unrelated_assignees,
                )
                raise AssignmentStateError("reassignment assignee postcondition failed")


def _best_effort_restore_assignees(
    github: Any,
    context: _IssueContext,
    state: AssignmentState,
    expected: set[str],
) -> None:
    for login in sorted(expected):
        try:
            current = _refresh_context(
                github,
                context,
                allowed_statuses=({"assigned"},),
            )
            if _require_assignment(current) != state:
                return
            if login not in current.issue.assignees:
                github.add_issue_assignee(current.issue.number, login)
        except (AssignmentStateError, GitHubError, IssueParseError, StatusCommentError):
            continue


def _require_in_progress_assignment(
    context: _IssueContext,
    state: AssignmentState,
) -> None:
    if (
        state.slack_status != "pending"
        or state.reminder_at is not None
        or state.history
        or state.operator_github not in context.issue.assignees
    ):
        raise AssignmentStateError("label transition is not a proven in-progress assignment")
    body = _assignment_notice(state.operator_github, 0)
    matches = [comment for comment in context.comments if comment.body == body]
    if (
        len(matches) != 1
        or not matches[0].author_is_bot
        or matches[0].author != context.automation_actor
    ):
        raise AssignmentStateError("label transition assignment notice is invalid")


def _ensure_notice(github: Any, context: _IssueContext, body: str) -> IssueComment:
    matches = [comment for comment in context.comments if comment.body == body]
    if len(matches) > 1:
        raise AssignmentStateError("duplicate assignment notice comments were found")
    if matches:
        if not matches[0].author_is_bot or matches[0].author != context.automation_actor:
            raise AssignmentStateError(
                "assignment notice is not owned by the expected automation actor"
            )
        return matches[0]
    try:
        persisted = github.create_issue_comment(context.issue.number, body)
    except GitHubError:
        comments = github.list_issue_comments(context.issue.number)
        matches = [comment for comment in comments if comment.body == body]
        if (
            len(matches) != 1
            or not matches[0].author_is_bot
            or matches[0].author != context.automation_actor
        ):
            raise
        persisted = matches[0]
    if (
        not persisted.author_is_bot
        or persisted.author != context.automation_actor
        or persisted.body != body
    ):
        raise AssignmentStateError("persisted assignment notice is invalid")
    comments = github.list_issue_comments(context.issue.number)
    matches = [comment for comment in comments if comment.body == body]
    if (
        len(matches) != 1
        or not matches[0].author_is_bot
        or matches[0].author != context.automation_actor
        or matches[0].id != persisted.id
    ):
        raise AssignmentStateError("persisted assignment notice postcondition failed")
    return matches[0]


def _exact_marker_comment(
    comments: Iterable[IssueComment],
    *,
    marker_line: re.Pattern[str],
    kind: str,
    required: bool,
    expected_author: str,
) -> IssueComment | None:
    matches: list[IssueComment] = []
    for comment in comments:
        marker_count = len(marker_line.findall(comment.body))
        if marker_count > 1:
            raise AssignmentStateError(f"multiple eduLLM {kind} markers were found")
        if marker_count == 1:
            matches.append(comment)
    if len(matches) > 1:
        raise AssignmentStateError(f"multiple eduLLM {kind} comments were found")
    if not matches:
        if required:
            raise AssignmentStateError(f"eduLLM {kind} comment is missing")
        return None
    if not matches[0].author_is_bot or matches[0].author != expected_author:
        raise AssignmentStateError(
            f"eduLLM {kind} marker is not owned by the expected automation actor"
        )
    return matches[0]


def _require_assignment(
    context: _IssueContext,
    enabled: dict[str, Operator] | None = None,
) -> AssignmentState:
    state = context.assignment
    if state is None:
        raise AssignmentStateError("eduLLM assignment comment is missing")
    if enabled is not None and state.operator_github not in enabled:
        raise AssignmentStateError("assignment references an unknown or disabled operator")
    return state


def _enabled_operator_map(operators: Sequence[Operator]) -> dict[str, Operator]:
    if not isinstance(operators, Sequence):
        raise ValueError("operator roster must be a sequence")
    enabled: dict[str, Operator] = {}
    rotations: set[int] = set()
    all_logins: set[str] = set()
    all_slack: set[str] = set()
    for operator in operators:
        if (
            not isinstance(operator, Operator)
            or not _valid_login(operator.github)
            or type(operator.enabled) is not bool
            or type(operator.rotation_order) is not int
            or operator.rotation_order < 0
            or operator.github in all_logins
            or type(operator.slack_user_id) is not str
            or _SLACK_USER_ID.fullmatch(operator.slack_user_id) is None
            or operator.slack_user_id in all_slack
        ):
            raise ValueError("operator roster is malformed")
        all_logins.add(operator.github)
        all_slack.add(operator.slack_user_id)
        if not operator.enabled:
            continue
        if operator.rotation_order in rotations:
            raise ValueError("enabled operator rotations must be unique")
        rotations.add(operator.rotation_order)
        enabled[operator.github] = operator
    return enabled


def _managed_statuses(issue: GitHubIssue) -> set[str]:
    status_labels = {label for label in issue.labels if label.startswith("status:")}
    if not status_labels <= _MANAGED_STATUS_LABELS:
        raise AssignmentStateError("managed status labels are malformed")
    return {label.removeprefix("status:") for label in status_labels}


def _validate_assignment_state(state: object) -> None:
    if not isinstance(state, AssignmentState):
        raise AssignmentStateError("assignment state is invalid")
    if type(state.history) is not tuple or len(state.history) > MAX_ASSIGNMENT_HISTORY:
        raise AssignmentStateError("assignment history is invalid")
    if (
        type(state.issue) is not int
        or state.issue <= 0
        or type(state.request_digest) is not str
        or _DIGEST.fullmatch(state.request_digest) is None
        or not _valid_login(state.operator_github)
        or type(state.slack_status) is not str
        or state.slack_status not in _SLACK_STATUSES
    ):
        raise AssignmentStateError("assignment state is invalid")
    _validate_now(state.assigned_at)
    if state.reminder_at is not None:
        _validate_now(state.reminder_at)
        if state.reminder_at < state.assigned_at:
            raise AssignmentStateError("assignment reminder timestamp is invalid")
    seen_times: list[datetime] = []
    for item in state.history:
        if not isinstance(item, AssignmentHistory) or not _valid_login(item.previous_operator):
            raise AssignmentStateError("assignment history is invalid")
        _validate_now(item.reassigned_at)
        seen_times.append(item.reassigned_at)
    if (
        seen_times != sorted(seen_times)
        or len(set(seen_times)) != len(seen_times)
        or (
            state.history
            and (
                state.history[-1].reassigned_at != state.assigned_at
                or state.history[-1].previous_operator == state.operator_github
            )
        )
    ):
        raise AssignmentStateError("assignment history is invalid")


def _valid_login(value: object) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 39
        and _LOGIN.fullmatch(cast(str, value)) is not None
    )


def _validate_automation_actor(value: object) -> None:
    if type(value) is not str or len(value) > 100 or _BOT_LOGIN.fullmatch(value) is None:
        raise ValueError("automation actor is malformed")


def _validate_now(value: object) -> None:
    if (
        not isinstance(value, datetime)
        or value.utcoffset() != timedelta(0)
        or value.microsecond != 0
        or not 2000 <= value.year <= 9999
    ):
        raise AssignmentStateError("assignment timestamp must be UTC whole-second time")


def _format_timestamp(value: datetime) -> str:
    _validate_now(value)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: object) -> datetime:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        raise AssignmentStateError("assignment timestamp is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise AssignmentStateError("assignment timestamp is invalid") from None
    _validate_now(parsed)
    return parsed


def _bounded_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > MAX_INTEGER_TOKEN_CHARS:
        raise _OversizedIntegerToken
    return int(value)


def _validate_policy_capacity(policy: object) -> None:
    if (
        not isinstance(policy, Policy)
        or type(policy.max_gpu_count) is not int
        or policy.max_gpu_count <= 0
        or type(policy.reminder_after_minutes) is not int
        or policy.reminder_after_minutes <= 0
        or type(policy.reassign_after_minutes) is not int
        or policy.reassign_after_minutes <= policy.reminder_after_minutes
    ):
        raise ValueError("assignment policy capacity is malformed")


def _reject_future_assignment_time(state: AssignmentState, now: datetime) -> None:
    if state.assigned_at > now or (state.reminder_at is not None and state.reminder_at > now):
        raise AssignmentStateError("assignment timestamp is in the future")


def _assignment_notice(operator: str, generation: int) -> str:
    return (
        f"@{operator} eduLLM job assignment is ready for submission "
        f"(assignment generation {generation})."
    )


def _reminder_notice(operator: str, generation: int) -> str:
    return (
        f"@{operator} 15-minute reminder: this eduLLM job is still assigned "
        f"and unsubmitted (assignment generation {generation})."
    )
