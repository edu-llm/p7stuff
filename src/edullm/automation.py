"""
Fail-closed validation orchestration for eduLLM Issue requests.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from edullm.github import (
    GitHubError,
    GitHubValidationError,
    IssueComment,
    normalize_actor_login,
)
from edullm.models import JobRequest
from edullm.policy import Policy
from edullm.request_parser import IssueParseError, parse_issue
from edullm.validation import (
    STATUS_MARKER,
    StatusCommentError,
    build_status_comment,
    validate_request,
)

VALIDATION_MARKER = "<!-- edullm-validation:v1 -->"

_TEAM_LEADS_FIELD = "team_leads"
_SAFE_COMMIT_REASONS = frozenset(
    {
        "commit does not exist at the requested SHA",
        "malformed GitHub commit evidence",
        "malformed GitHub contents evidence",
        "script does not exist at the requested SHA",
    }
)


class AutomationStateError(RuntimeError):
    """A sanitized fail-closed machine-comment state error."""


@dataclass(frozen=True)
class ValidationDecision:
    """Immutable result of local policy and executable-commit validation."""

    status: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class AutomationResult:
    """Immutable result returned by one Issue validation attempt."""

    status: str
    errors: tuple[str, ...]
    operational_error: bool


def load_team_leads(path: Path) -> frozenset[str]:
    """
    Load the protected, explicit team-lead login allowlist.

    The exact schema is ``team_leads: [<login>, ...]``. An empty list remains
    a valid disabled configuration for reviewer-gated submission paths.

    :param path: The tracked team-lead YAML path.

    :returns: Case-normalized immutable GitHub actor logins.

    :raises ValueError: If YAML, schema, a login, or uniqueness is invalid.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("team-leads: invalid YAML") from None
    except OSError:
        raise ValueError("team-leads: configuration cannot be read") from None
    if type(document) is not dict or set(document) != {_TEAM_LEADS_FIELD}:
        raise ValueError("team-leads: expected only the team_leads field")
    values = cast(dict[object, object], document)[_TEAM_LEADS_FIELD]
    if type(values) is not list:
        raise ValueError("team-leads: team_leads must be a list")

    normalized: set[str] = set()
    for index, value in enumerate(cast(list[object], values)):
        try:
            login = normalize_actor_login(value)
        except GitHubValidationError:
            raise ValueError(f"team-leads: team_leads[{index}] is invalid") from None
        if login in normalized:
            raise ValueError("team-leads: team_leads contains duplicate logins")
        normalized.add(login)
    return frozenset(normalized)


def validation_decision(
    request: JobRequest,
    *,
    policy: Policy,
    github: Any,
) -> ValidationDecision:
    """
    Validate local request safety before consulting executable-commit evidence.

    :param request: The parsed immutable request.
    :param policy: The trusted queue policy.
    :param github: A :class:`~edullm.github.GitHubClient`-compatible client.

    :returns: A deterministic immutable ready/requested decision.
    """
    local_errors = validate_request(request, policy)
    if local_errors:
        return ValidationDecision("requested", tuple(local_errors))

    evidence = github.executable_commit(
        request.commit_sha,
        script_path=request.script_path,
    )
    if evidence.available:
        return ValidationDecision("ready", ())
    return ValidationDecision("requested", (_sanitized_commit_reason(evidence.reason),))


def validate_issue(
    issue_number: int,
    *,
    github: Any,
    policy: Policy,
    validated_at: datetime,
) -> AutomationResult:
    """
    Validate one current Issue and persist its fail-closed machine state.

    ``status:ready`` is invalidated before parsing or remote evidence. A canonical
    status comment is persisted only after the current Issue is re-fetched and
    re-parsed to the same digest, and the ready label is written only after that
    comment succeeds. GitHub has no transaction or compare-and-swap spanning
    Issue bodies, comments, and labels, so before/after postconditions narrow but
    cannot eliminate every external race. Per-Issue workflow concurrency and
    later submission-time digest revalidation remain required safety layers.

    :param issue_number: The trusted positive Issue number.
    :param github: A :class:`~edullm.github.GitHubClient`-compatible client.
    :param policy: The trusted queue policy.
    :param validated_at: The canonical validation timestamp.

    :returns: The immutable validation attempt result.
    """
    try:
        current = _set_requested(github, issue_number)

        try:
            request = parse_issue(
                current.body,
                issue_number=issue_number,
                requester=current.requester,
            )
        except IssueParseError as error:
            errors = tuple(error.errors)
            _persist_requested_errors(github, issue_number, errors)
            return AutomationResult("requested", errors, False)

        decision = validation_decision(
            request,
            policy=policy,
            github=github,
        )
        if decision.errors:
            _persist_requested_errors(github, issue_number, decision.errors)
            return AutomationResult("requested", decision.errors, False)

        status_body = build_status_comment(request, validated_at=validated_at)
        current = github.fetch_issue(issue_number)
        if not _issue_matches_request(current, request, issue_number):
            errors = ("Issue changed during validation; submit or save the current request again",)
            _persist_requested_errors(github, issue_number, errors)
            return AutomationResult("requested", errors, False)

        persisted_status = _persist_machine_comment(
            github,
            issue_number,
            marker=STATUS_MARKER,
            kind="status",
            body=status_body,
        )
        current = github.fetch_issue(issue_number)
        if not _issue_matches_request(current, request, issue_number):
            errors = ("Issue changed during validation; submit or save the current request again",)
            _persist_requested_errors(github, issue_number, errors)
            return AutomationResult("requested", errors, False)

        pre_ready_comments = github.list_issue_comments(issue_number)
        _validate_ready_comment_snapshot(
            pre_ready_comments,
            status_body=status_body,
            status_id=persisted_status.id,
        )
        final_issue = _set_ready(github, issue_number)
        if not _issue_matches_request(final_issue, request, issue_number):
            raise AutomationStateError(
                "Issue changed during validation; submit or save the current request again"
            )
        published_comments = github.list_issue_comments(issue_number)
        _validate_ready_comment_snapshot(
            published_comments,
            status_body=status_body,
            status_id=persisted_status.id,
        )
        confirmed_issue = github.fetch_issue(issue_number)
        if (
            not _issue_matches_request(confirmed_issue, request, issue_number)
            or "status:ready" not in confirmed_issue.labels
            or "status:requested" in confirmed_issue.labels
        ):
            raise AutomationStateError(
                "Issue changed during validation; submit or save the current request again"
            )
        return AutomationResult("ready", (), False)
    except (AutomationStateError, GitHubError, StatusCommentError) as error:
        if _force_requested(github, issue_number) is None:
            return AutomationResult(
                "requested",
                ("GitHub validation reconciliation failed",),
                True,
            )
        if isinstance(error, AutomationStateError):
            return AutomationResult("requested", (str(error),), True)
        return AutomationResult(
            "requested",
            ("GitHub validation operation failed",),
            True,
        )


def _sanitized_commit_reason(reason: object) -> str:
    if type(reason) is not str:
        return "commit evidence was not accepted"
    if reason in _SAFE_COMMIT_REASONS:
        return reason
    return "commit evidence was not accepted"


def _set_requested(github: Any, issue_number: int) -> Any:
    issue = _force_requested(github, issue_number)
    if issue is None:
        raise AutomationStateError("failed to establish requested status")
    return issue


def _force_requested(github: Any, issue_number: int) -> Any | None:
    try:
        github.add_issue_status_label(issue_number, "status:requested")
    except GitHubError:
        pass
    try:
        github.remove_issue_status_label(issue_number, "status:ready")
    except GitHubError:
        pass
    try:
        issue = github.fetch_issue(issue_number)
    except GitHubError:
        return None
    if "status:requested" not in issue.labels or "status:ready" in issue.labels:
        return None
    return issue


def _set_ready(github: Any, issue_number: int) -> Any:
    github.add_issue_status_label(issue_number, "status:ready")
    github.remove_issue_status_label(issue_number, "status:requested")
    issue = github.fetch_issue(issue_number)
    if "status:ready" not in issue.labels or "status:requested" in issue.labels:
        raise AutomationStateError("failed to establish ready status")
    return issue


def _persist_validation_errors(
    github: Any,
    issue_number: int,
    errors: tuple[str, ...],
) -> None:
    escaped = tuple(html.escape(error, quote=False) for error in errors)
    body = (
        f"{VALIDATION_MARKER}\n"
        "### eduLLM validation\n\n"
        "This request is not ready:\n" + "\n".join(f"- {error}" for error in escaped)
    )
    _persist_machine_comment(
        github,
        issue_number,
        marker=VALIDATION_MARKER,
        kind="validation",
        body=body,
    )


def _persist_requested_errors(
    github: Any,
    issue_number: int,
    errors: tuple[str, ...],
) -> None:
    _persist_validation_errors(github, issue_number, errors)
    _set_requested(github, issue_number)


def _persist_machine_comment(
    github: Any,
    issue_number: int,
    *,
    marker: str,
    kind: str,
    body: str,
) -> IssueComment:
    comments = github.list_issue_comments(issue_number)
    existing = _validate_marker_namespace(comments, marker=marker, kind=kind)
    if existing is None:
        persisted = github.create_issue_comment(issue_number, body)
    else:
        persisted = github.update_issue_comment(existing.id, body)
    if not persisted.author_is_bot or persisted.body != body:
        raise AutomationStateError(f"persisted eduLLM {kind} comment is invalid")
    comments = github.list_issue_comments(issue_number)
    return _require_exact_machine_comment(
        comments,
        marker=marker,
        kind=kind,
        body=body,
        expected_id=persisted.id,
    )


def _require_exact_machine_comment(
    comments: Iterable[IssueComment],
    *,
    marker: str,
    kind: str,
    body: str,
    expected_id: int,
) -> IssueComment:
    persisted = _validate_marker_namespace(comments, marker=marker, kind=kind)
    if persisted is None:
        raise AutomationStateError(f"persisted eduLLM {kind} comment is missing")
    if persisted.id != expected_id:
        raise AutomationStateError(f"persisted eduLLM {kind} comment identity changed")
    if persisted.body != body:
        raise AutomationStateError(f"persisted eduLLM {kind} comment body changed")
    return persisted


def _validate_ready_comment_snapshot(
    comments: tuple[IssueComment, ...],
    *,
    status_body: str,
    status_id: int,
) -> None:
    _validate_marker_namespace(
        comments,
        marker=VALIDATION_MARKER,
        kind="validation",
    )
    _require_exact_machine_comment(
        comments,
        marker=STATUS_MARKER,
        kind="status",
        body=status_body,
        expected_id=status_id,
    )


def _issue_matches_request(issue: Any, request: JobRequest, issue_number: int) -> bool:
    try:
        current_request = parse_issue(
            issue.body,
            issue_number=issue_number,
            requester=issue.requester,
        )
    except IssueParseError:
        return False
    return (
        current_request.digest == request.digest
        and current_request.canonical_json() == request.canonical_json()
    )


def _validate_marker_namespace(
    comments: Iterable[IssueComment],
    *,
    marker: str,
    kind: str,
) -> IssueComment | None:
    marker_line = re.compile(rf"^{re.escape(marker)}$", re.MULTILINE)
    matches: list[IssueComment] = []
    for comment in comments:
        count = len(marker_line.findall(comment.body))
        if count > 1:
            raise AutomationStateError(f"multiple eduLLM {kind} markers were found")
        if count == 1:
            matches.append(comment)
    if len(matches) > 1:
        raise AutomationStateError(f"multiple eduLLM {kind} comments were found")
    if not matches:
        return None
    if not matches[0].author_is_bot:
        raise AutomationStateError(f"eduLLM {kind} marker is not bot-authored")
    return matches[0]
