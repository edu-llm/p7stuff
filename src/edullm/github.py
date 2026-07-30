"""
Constrained GitHub REST access and reviewed-commit authorization.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import parse_qsl, quote, unquote, urlsplit

import requests

_API_VERSION = "2026-03-10"
_DEFAULT_TIMEOUT = (5, 20)
_DEFAULT_MAX_PAGES = 100
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_OWNER_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_REPO_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9_.-])?\Z")
_ACTOR_NAME_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\Z")
_MAX_USER_LOGIN_LENGTH = 39
_MAX_ACTOR_LOGIN_LENGTH = 100
_PATH_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9._~-]+\Z")
_QUERY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_QUERY_VALUE_PATTERN = re.compile(r"[A-Za-z0-9._~-]*\Z")
_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING"})
_SUBSTANTIVE_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})
_CHECK_STATUSES = frozenset(
    {"queued", "in_progress", "completed", "waiting", "requested", "pending"}
)
_CHECK_CONCLUSIONS = frozenset(
    {
        "action_required",
        "cancelled",
        "failure",
        "neutral",
        "success",
        "skipped",
        "stale",
        "timed_out",
        "startup_failure",
    }
)
_MAX_COMMENT_CHARS = 65_536
_MAX_LABEL_CHARS = 50
_WRITABLE_STATUS_LABELS = frozenset(
    {
        "status:requested",
        "status:validating",
        "status:ready",
        "status:assigned",
        "status:submitted",
        "status:running",
        "status:completed",
        "status:failed",
        "status:cancelled",
        "status:preempted",
    }
)


class GitHubError(RuntimeError):
    """Base class for sanitized GitHub client failures."""


class GitHubValidationError(GitHubError, ValueError):
    """Raised when caller-controlled GitHub input is unsafe or malformed."""


class GitHubAPIError(GitHubError):
    """Raised when GitHub cannot provide a successful response."""


class GitHubDataError(GitHubError):
    """Raised when a successful GitHub response has an invalid shape."""


@dataclass(frozen=True)
class ReviewResult:
    """Immutable reviewed-commit authorization result."""

    approved: bool
    pr_number: int | None
    reason: str


@dataclass(frozen=True)
class CommitResult:
    """Immutable commit-and-script evidence result."""

    available: bool
    reason: str


@dataclass(frozen=True)
class GitHubIssue:
    """Validated Issue state used by the queue automation."""

    number: int
    body: str
    requester: str
    labels: tuple[str, ...]
    title: str = ""
    assignees: tuple[str, ...] = ()


@dataclass(frozen=True)
class IssueComment:
    """Validated Issue comment state used by the queue automation."""

    id: int
    body: str
    author: str
    author_is_bot: bool


@dataclass(frozen=True)
class _ReviewState:
    state: str
    commit_id: str
    submitted_at: datetime


class GitHubClient:
    """A small client for the GitHub evidence and Issue state used by the queue."""

    def __init__(
        self,
        token: str,
        repo: str,
        *,
        base_url: str = "https://api.github.com",
        session: Any | None = None,
        timeout: tuple[float, float] = _DEFAULT_TIMEOUT,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        """
        Initialize the client with a bearer token and ``owner/repository`` name.

        :param token: A GitHub token. It is installed directly on the session and
            is never retained in a printable client field.
        :param repo: The repository in ``owner/name`` form.
        :param base_url: The HTTPS GitHub REST API origin.
        :param session: An optional requests-compatible session.
        :param timeout: Connect and read timeouts in seconds.
        :param max_pages: The finite pagination safety bound.

        :raises GitHubValidationError: If any constructor input is unsafe.
        """
        _validate_token(token)
        _validate_repo(repo)
        self.base_url = _validate_base_url(base_url)
        self.repo = repo
        self._timeout = _validate_timeout(timeout)
        if type(max_pages) is not int or not 1 <= max_pages <= 1_000:
            raise GitHubValidationError("max_pages must be an integer from 1 to 1000")
        self._max_pages = max_pages
        self._session = requests.Session() if session is None else session
        headers = getattr(self._session, "headers", None)
        if not hasattr(headers, "update"):
            raise GitHubValidationError("session must provide mutable headers")
        cast(Any, headers).update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": _API_VERSION,
            }
        )

    def __repr__(self) -> str:
        """Return a token-free representation."""
        return (
            f"{type(self).__name__}(repo={self.repo!r}, "
            f"base_url={self.base_url!r}, max_pages={self._max_pages!r})"
        )

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> object:
        """
        GET and JSON-decode a relative GitHub API path.

        :param path: A validated relative API path beginning with ``/``.
        :param params: Optional scalar query parameters.

        :returns: The JSON-decoded payload.

        :raises GitHubAPIError: If the request fails.
        :raises GitHubDataError: If GitHub returns malformed JSON.
        :raises GitHubValidationError: If the path or parameters are unsafe.
        """
        response = self._request(path, params=params)
        return self._decode_json(response)

    def paginated_get(
        self,
        path: str,
        *,
        key: str | None = None,
        total_count_key: str | None = None,
    ) -> list[dict[str, object]]:
        """
        Read all object rows from a list or keyed-list GitHub endpoint.

        :param path: A relative API path, optionally with an existing query string.
        :param key: The list field for endpoints that return an object envelope.
        :param total_count_key: The required count field for a counted object
            envelope. Bare-list endpoints should leave this unset.

        :returns: All rows in server order.

        :raises GitHubDataError: If an envelope, row, or page count is malformed.
        """
        _validate_api_path(path)
        query_fields = {name for name, _ in parse_qsl(urlsplit(path).query)}
        if query_fields & {"page", "per_page"}:
            raise GitHubValidationError("paginated path must not set page or per_page")
        if key is not None and (type(key) is not str or _QUERY_NAME_PATTERN.fullmatch(key) is None):
            raise GitHubValidationError("pagination key is invalid")
        if total_count_key is not None and (
            key is None
            or type(total_count_key) is not str
            or _QUERY_NAME_PATTERN.fullmatch(total_count_key) is None
        ):
            raise GitHubValidationError("pagination total-count key is invalid")

        rows: list[dict[str, object]] = []
        expected_total: int | None = None
        for page in range(1, self._max_pages + 1):
            payload = self.get(path, params={"per_page": 100, "page": page})
            if key is None:
                batch = payload
            elif type(payload) is dict:
                envelope = cast(dict[object, object], payload)
                batch = envelope.get(key)
                if total_count_key is not None:
                    total_count = envelope.get(total_count_key)
                    if type(total_count) is not int or total_count < 0:
                        raise GitHubDataError("GitHub API returned malformed paginated response")
                    if expected_total is None:
                        expected_total = total_count
                    elif total_count != expected_total:
                        raise GitHubDataError("GitHub API paginated response count is inconsistent")
            else:
                raise GitHubDataError("GitHub API returned malformed paginated response")
            if type(batch) is not list or len(batch) > 100:
                raise GitHubDataError("GitHub API returned malformed paginated response")
            typed_batch = cast(list[object], batch)
            if any(type(row) is not dict for row in typed_batch):
                raise GitHubDataError("GitHub API returned malformed paginated response")
            rows.extend(cast(list[dict[str, object]], typed_batch))
            if len(typed_batch) < 100:
                if expected_total is not None and expected_total != len(rows):
                    raise GitHubDataError("GitHub API paginated response count is inconsistent")
                return rows

        raise GitHubDataError("GitHub pagination limit exceeded")

    def fetch_issue(self, issue_number: int) -> GitHubIssue:
        """
        Fetch and validate the current body, requester, and labels for an Issue.

        :param issue_number: The positive Issue number.

        :returns: Immutable validated Issue state.
        """
        _validate_positive_identifier(issue_number, "issue number")
        payload = self.get(f"/repos/{self.repo}/issues/{issue_number}")
        return _issue_from_payload(payload, expected_number=issue_number)

    def list_active_queue_issues(self) -> tuple[GitHubIssue, ...]:
        """
        List every current open Issue carrying the queue discriminator label.

        Pull-request rows are rejected rather than silently treated as Issues.

        :returns: Validated immutable Issue snapshots in server order.
        """
        rows = self.paginated_get(f"/repos/{self.repo}/issues?labels=edullm-job&state=open")
        issues = tuple(_issue_from_payload(row, reject_pull=True) for row in rows)
        numbers = [issue.number for issue in issues]
        if len(set(numbers)) != len(numbers):
            raise GitHubDataError("GitHub API returned malformed Issue response")
        return issues

    def list_issue_comments(self, issue_number: int) -> tuple[IssueComment, ...]:
        """
        Fetch every validated comment for an Issue.

        :param issue_number: The positive Issue number.

        :returns: Immutable comments in GitHub's server order.
        """
        _validate_positive_identifier(issue_number, "issue number")
        rows = self.paginated_get(f"/repos/{self.repo}/issues/{issue_number}/comments")
        comments = tuple(_issue_comment_from_payload(row) for row in rows)
        ids = [comment.id for comment in comments]
        if len(set(ids)) != len(ids):
            raise GitHubDataError("GitHub API returned malformed Issue comment response")
        return comments

    def create_issue_comment(self, issue_number: int, body: str) -> IssueComment:
        """
        Create one Issue comment using a JSON request body.

        :param issue_number: The positive Issue number.
        :param body: The complete machine comment body.

        :returns: The validated created comment.
        """
        _validate_positive_identifier(issue_number, "issue number")
        _validate_comment_body(body)
        response = self._write_request(
            "post",
            f"/repos/{self.repo}/issues/{issue_number}/comments",
            json_body={"body": body},
        )
        comment = _issue_comment_from_payload(self._decode_json(response))
        if comment.body != body:
            raise GitHubDataError("GitHub API returned malformed Issue comment response")
        return comment

    def update_issue_comment(self, comment_id: int, body: str) -> IssueComment:
        """
        Replace one Issue comment using a JSON request body.

        :param comment_id: The positive repository comment identifier.
        :param body: The complete replacement body.

        :returns: The validated updated comment.
        """
        _validate_positive_identifier(comment_id, "comment identifier")
        _validate_comment_body(body)
        response = self._write_request(
            "patch",
            f"/repos/{self.repo}/issues/comments/{comment_id}",
            json_body={"body": body},
        )
        comment = _issue_comment_from_payload(self._decode_json(response))
        if comment.id != comment_id or comment.body != body:
            raise GitHubDataError("GitHub API returned malformed Issue comment response")
        return comment

    def add_issue_status_label(
        self,
        issue_number: int,
        label: str,
    ) -> tuple[str, ...]:
        """
        Add one managed status label without replacing unrelated labels.

        :param issue_number: The positive Issue number.
        :param label: One exact managed :class:`~edullm.models.JobStatus` label.

        :returns: The complete validated server label sequence after the add.
        """
        _validate_positive_identifier(issue_number, "issue number")
        _validate_status_label(label)
        response = self._write_request(
            "post",
            f"/repos/{self.repo}/issues/{issue_number}/labels",
            json_body={"labels": [label]},
        )
        labels = _label_names_from_payload(self._decode_json(response))
        if label not in labels:
            raise GitHubDataError("GitHub API returned malformed Issue labels response")
        return labels

    def remove_issue_status_label(self, issue_number: int, label: str) -> bool:
        """
        Idempotently remove one managed status label without touching other labels.

        :param issue_number: The positive Issue number.
        :param label: One exact managed :class:`~edullm.models.JobStatus` label.

        :returns: ``True`` when removed, or ``False`` only when GitHub returns 404.
        """
        _validate_positive_identifier(issue_number, "issue number")
        _validate_status_label(label)
        encoded_label = quote(label, safe="")
        response = self._write_request(
            "delete",
            f"/repos/{self.repo}/issues/{issue_number}/labels/{encoded_label}",
            allow_not_found=True,
            encoded_status_label=True,
        )
        if response.status_code == 404:
            return False
        labels = _label_names_from_payload(self._decode_json(response))
        if label in labels:
            raise GitHubDataError("GitHub API returned malformed Issue labels response")
        return True

    def add_issue_assignee(self, issue_number: int, login: str) -> tuple[str, ...]:
        """
        Add one operator assignee without replacing unrelated assignees.

        :param issue_number: The positive Issue number.
        :param login: A canonical lowercase GitHub user login.

        :returns: The complete validated assignee sequence after the write.
        """
        _validate_positive_identifier(issue_number, "issue number")
        _validate_assignee_login(login)
        response = self._write_request(
            "post",
            f"/repos/{self.repo}/issues/{issue_number}/assignees",
            json_body={"assignees": [login]},
        )
        issue = _issue_from_payload(self._decode_json(response), expected_number=issue_number)
        if login not in issue.assignees:
            raise GitHubDataError("GitHub API returned malformed Issue assignee response")
        return issue.assignees

    def remove_issue_assignee(self, issue_number: int, login: str) -> bool:
        """
        Idempotently remove one operator without replacing unrelated assignees.

        :param issue_number: The positive Issue number.
        :param login: A canonical lowercase GitHub user login.

        :returns: ``True`` when removed, or ``False`` only for a 404 response.
        """
        _validate_positive_identifier(issue_number, "issue number")
        _validate_assignee_login(login)
        response = self._write_request(
            "delete",
            f"/repos/{self.repo}/issues/{issue_number}/assignees",
            json_body={"assignees": [login]},
            allow_not_found=True,
        )
        if response.status_code == 404:
            return False
        issue = _issue_from_payload(self._decode_json(response), expected_number=issue_number)
        if login in issue.assignees:
            raise GitHubDataError("GitHub API returned malformed Issue assignee response")
        return True

    def file_exists(self, path: str, *, ref: str) -> bool:
        """
        Check that a file exists at an exact full commit SHA.

        :param path: A safe repository-relative file path.
        :param ref: The exact 40-character commit SHA.

        :returns: ``False`` only when GitHub returns 404, otherwise ``True`` for a
            valid file response.

        :raises GitHubAPIError: For authorization, server, or network failures.
        :raises GitHubDataError: For malformed successful responses.
        """
        _validate_script_path(path)
        _validate_sha(ref)
        response = self._request(
            f"/repos/{self.repo}/contents/{path}",
            params={"ref": ref},
            allow_not_found=True,
        )
        if response.status_code == 404:
            return False
        payload = self._decode_json(response)
        if (
            type(payload) is not dict
            or payload.get("type") != "file"
            or payload.get("path") != path
        ):
            raise GitHubDataError("GitHub API returned malformed contents response")
        return True

    def executable_commit(self, sha: str, *, script_path: str) -> CommitResult:
        """
        Prove that an exact pushed commit and script exist.

        :param sha: The requested full commit SHA.
        :param script_path: The protected script selected by policy.

        :returns: A deterministic, fail-closed evidence result.

        :raises GitHubValidationError: If caller-controlled input is malformed.
        :raises GitHubAPIError: If GitHub evidence cannot be retrieved.
        """
        _validate_sha(sha)
        _validate_script_path(script_path)
        response = self._request(
            f"/repos/{self.repo}/commits/{sha}",
            allow_not_found=True,
        )
        if response.status_code == 404:
            return CommitResult(False, "commit does not exist at the requested SHA")
        payload = self._decode_json(response)
        if type(payload) is not dict:
            return CommitResult(False, "malformed GitHub commit evidence")
        commit_sha = cast(dict[object, object], payload).get("sha")
        if type(commit_sha) is not str or commit_sha != sha:
            return CommitResult(False, "malformed GitHub commit evidence")
        try:
            exists = self.file_exists(script_path, ref=sha)
        except GitHubDataError:
            return CommitResult(False, "malformed GitHub contents evidence")
        if not exists:
            return CommitResult(False, "script does not exist at the requested SHA")
        return CommitResult(True, "commit and script exist")

    def reviewed_commit(
        self,
        sha: str,
        *,
        script_path: str,
        allowed_reviewers: set[str],
        required_checks: set[str],
    ) -> ReviewResult:
        """
        Authorize an exact PR head reviewed by a protected reviewer.

        The selected pull request must be open or merged, an authorized reviewer's
        current substantive review must approve this exact SHA, no current
        authorized review on that pull request may request changes, every required
        check must be uniquely completed with success, and the script must exist at
        the same SHA.

        :param sha: The requested full commit SHA.
        :param script_path: The protected script selected by policy.
        :param allowed_reviewers: The explicit non-empty protected reviewer set.
        :param required_checks: The explicit non-empty protected check-name set.

        :returns: A deterministic, fail-closed authorization result.

        :raises GitHubValidationError: If caller-controlled input is malformed.
        :raises GitHubAPIError: If GitHub evidence cannot be retrieved.
        """
        _validate_sha(sha)
        _validate_script_path(script_path)
        reviewers = _normalize_reviewers(allowed_reviewers)
        checks_required = _normalize_required_checks(required_checks)

        try:
            qualifying_pulls = self._qualifying_pull_numbers(sha)
        except GitHubDataError:
            return ReviewResult(False, None, "malformed GitHub pull evidence")
        if not qualifying_pulls:
            return ReviewResult(
                False,
                None,
                "no qualifying pull request has the requested SHA as its head",
            )

        approved_pulls: list[int] = []
        blocked = False
        try:
            for number in qualifying_pulls:
                reviews = self.paginated_get(f"/repos/{self.repo}/pulls/{number}/reviews")
                approved, pull_blocked = _effective_review_decision(reviews, reviewers, sha)
                if approved and not pull_blocked:
                    approved_pulls.append(number)
                blocked = blocked or pull_blocked
        except GitHubDataError:
            return ReviewResult(False, None, "malformed GitHub review evidence")

        if blocked:
            return ReviewResult(
                False,
                None,
                "an authorized reviewer currently requests changes",
            )
        if not approved_pulls:
            return ReviewResult(
                False,
                None,
                "no authorized reviewer approved the requested SHA",
            )
        selected_pr = min(approved_pulls)

        try:
            check_runs = self.paginated_get(
                f"/repos/{self.repo}/commits/{sha}/check-runs?filter=latest",
                key="check_runs",
                total_count_key="total_count",
            )
            unsuccessful = _unsuccessful_required_checks(check_runs, checks_required)
        except GitHubDataError:
            return ReviewResult(False, None, "malformed GitHub check evidence")
        if unsuccessful:
            return ReviewResult(
                False,
                selected_pr,
                "required checks are not uniquely successful: " + ", ".join(unsuccessful),
            )

        try:
            exists = self.file_exists(script_path, ref=sha)
        except GitHubDataError:
            return ReviewResult(False, selected_pr, "malformed GitHub contents evidence")
        if not exists:
            return ReviewResult(
                False,
                selected_pr,
                "script does not exist at the requested SHA",
            )
        try:
            final_pull = self.get(f"/repos/{self.repo}/pulls/{selected_pr}")
            still_qualifying = _is_qualifying_pull(final_pull, sha)
        except GitHubDataError:
            return ReviewResult(False, selected_pr, "malformed GitHub pull evidence")
        if not still_qualifying:
            return ReviewResult(
                False,
                selected_pr,
                "selected pull request head changed during authorization",
            )
        return ReviewResult(
            True,
            selected_pr,
            "approved PR with successful checks and script",
        )

    def _qualifying_pull_numbers(self, sha: str) -> list[int]:
        pulls = self.paginated_get(f"/repos/{self.repo}/commits/{sha}/pulls")
        numbers: list[int] = []
        seen: set[int] = set()
        for pull in pulls:
            number = pull.get("number")
            if type(number) is not int or number <= 0 or number in seen:
                raise GitHubDataError("GitHub API returned malformed pull evidence")
            seen.add(number)
            details = self.get(f"/repos/{self.repo}/pulls/{number}")
            if _is_qualifying_pull(details, sha):
                numbers.append(number)
        return sorted(numbers)

    def _request(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        _validate_api_path(path)
        _validate_query_params(params)
        try:
            response = self._session.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self._timeout,
            )
        except requests.RequestException:
            raise GitHubAPIError("GitHub API request failed") from None
        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise GitHubDataError("GitHub API returned malformed HTTP response")
        if allow_not_found and status_code == 404:
            return response
        try:
            response.raise_for_status()
        except requests.RequestException:
            raise GitHubAPIError("GitHub API request failed") from None
        return response

    def _write_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, object] | None = None,
        allow_not_found: bool = False,
        encoded_status_label: bool = False,
    ) -> Any:
        if encoded_status_label:
            _validate_encoded_status_label_path(path, self.repo)
        else:
            _validate_api_path(path)
        requester = getattr(self._session, method, None)
        if not callable(requester):
            raise GitHubDataError("GitHub session cannot perform write requests")
        request_arguments: dict[str, object] = {"timeout": self._timeout}
        if json_body is not None:
            request_arguments["json"] = dict(json_body)
        try:
            response = requester(f"{self.base_url}{path}", **request_arguments)
        except requests.RequestException:
            raise GitHubAPIError("GitHub API request failed") from None
        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise GitHubDataError("GitHub API returned malformed HTTP response")
        if allow_not_found and status_code == 404:
            return response
        try:
            response.raise_for_status()
        except requests.RequestException:
            raise GitHubAPIError("GitHub API request failed") from None
        return response

    @staticmethod
    def _decode_json(response: Any) -> object:
        try:
            return response.json()
        except (TypeError, ValueError):
            raise GitHubDataError("GitHub API returned malformed JSON") from None


def _validate_token(token: object) -> None:
    if (
        type(token) is not str
        or not token
        or token != token.strip()
        or any(ord(character) < 33 or ord(character) == 127 for character in token)
    ):
        raise GitHubValidationError("GitHub token is invalid")


def _validate_repo(repo: object) -> None:
    if type(repo) is not str:
        raise GitHubValidationError("repository must use owner/name form")
    parts = repo.split("/")
    if (
        len(parts) != 2
        or _OWNER_PATTERN.fullmatch(parts[0]) is None
        or _REPO_PATTERN.fullmatch(parts[1]) is None
        or parts[1] in {".", ".."}
    ):
        raise GitHubValidationError("repository must use safe owner/name form")


def _validate_base_url(base_url: object) -> str:
    if type(base_url) is not str or not base_url:
        raise GitHubValidationError("base_url must be a valid HTTPS URL")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GitHubValidationError("base_url must be a valid HTTPS URL")
    _validate_url_path(parsed.path or "/")
    return base_url.rstrip("/")


def _validate_timeout(timeout: object) -> tuple[float, float]:
    if type(timeout) is not tuple or len(timeout) != 2:
        raise GitHubValidationError("timeout must contain connect and read seconds")
    values = cast(tuple[object, object], timeout)
    if any(
        type(value) not in {int, float}
        or not math.isfinite(cast(float, value))
        or cast(float, value) <= 0
        for value in values
    ):
        raise GitHubValidationError("timeout values must be finite and positive")
    return cast(tuple[float, float], timeout)


def _validate_api_path(path: object) -> None:
    if type(path) is not str or not path.startswith("/") or path.startswith("//"):
        raise GitHubValidationError("GitHub API path must be relative")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise GitHubValidationError("GitHub API path is invalid")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise GitHubValidationError("GitHub API path must be relative")
    _validate_url_path(parsed.path)
    try:
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise GitHubValidationError("GitHub API query is invalid") from None
    for name, value in query:
        if (
            _QUERY_NAME_PATTERN.fullmatch(name) is None
            or _QUERY_VALUE_PATTERN.fullmatch(value) is None
        ):
            raise GitHubValidationError("GitHub API query is invalid")


def _validate_url_path(path: str) -> None:
    if "\\" in path or "%" in path:
        raise GitHubValidationError("URL path is invalid")
    if path == "/":
        return
    segments = path.split("/")[1:]
    if any(
        not segment
        or segment in {".", ".."}
        or _PATH_SEGMENT_PATTERN.fullmatch(unquote(segment)) is None
        for segment in segments
    ):
        raise GitHubValidationError("URL path is invalid")


def _validate_query_params(params: Mapping[str, object] | None) -> None:
    if params is None:
        return
    if not isinstance(params, Mapping):
        raise GitHubValidationError("GitHub API parameters must be a mapping")
    for name, value in params.items():
        if type(name) is not str or _QUERY_NAME_PATTERN.fullmatch(name) is None:
            raise GitHubValidationError("GitHub API parameter name is invalid")
        if type(value) is int:
            if value < 0:
                raise GitHubValidationError("GitHub API parameter value is invalid")
        elif type(value) is str:
            if not value or _QUERY_VALUE_PATTERN.fullmatch(value) is None:
                raise GitHubValidationError("GitHub API parameter value is invalid")
        else:
            raise GitHubValidationError("GitHub API parameter value is invalid")


def _validate_positive_identifier(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise GitHubValidationError(f"{name} must be a positive integer")


def _is_valid_label_name(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value) <= _MAX_LABEL_CHARS
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _validate_status_label(label: object) -> None:
    if type(label) is not str or label not in _WRITABLE_STATUS_LABELS:
        raise GitHubValidationError("status label must be an exact managed lifecycle label")


def _validate_encoded_status_label_path(path: object, repo: str) -> None:
    expected_prefix = f"/repos/{repo}/issues/"
    if type(path) is not str or not path.startswith(expected_prefix):
        raise GitHubValidationError("GitHub API path must be relative")
    suffix = path[len(expected_prefix) :]
    issue, separator, label_path = suffix.partition("/labels/")
    if (
        not separator
        or not issue.isascii()
        or not issue.isdecimal()
        or issue.startswith("0")
        or label_path not in {quote(label, safe="") for label in _WRITABLE_STATUS_LABELS}
    ):
        raise GitHubValidationError("GitHub API path is invalid")


def _label_names_from_payload(payload: object) -> tuple[str, ...]:
    if type(payload) is not list:
        raise GitHubDataError("GitHub API returned malformed Issue labels response")
    labels: list[str] = []
    for row in cast(list[object], payload):
        if type(row) is not dict:
            raise GitHubDataError("GitHub API returned malformed Issue labels response")
        name = cast(dict[object, object], row).get("name")
        if not _is_valid_label_name(name):
            raise GitHubDataError("GitHub API returned malformed Issue labels response")
        labels.append(cast(str, name))
    if len(set(labels)) != len(labels):
        raise GitHubDataError("GitHub API returned malformed Issue labels response")
    return tuple(labels)


def _validate_comment_body(body: object) -> None:
    if (
        type(body) is not str
        or not body
        or len(body) > _MAX_COMMENT_CHARS
        or any(
            ord(character) < 32 and character not in {"\t", "\n", "\r"} or ord(character) == 127
            for character in body
        )
    ):
        raise GitHubValidationError("comment body is invalid")


def _issue_comment_from_payload(payload: object) -> IssueComment:
    if type(payload) is not dict:
        raise GitHubDataError("GitHub API returned malformed Issue comment response")
    row = cast(dict[object, object], payload)
    comment_id = row.get("id")
    body = row.get("body")
    user = row.get("user")
    if (
        type(comment_id) is not int
        or comment_id <= 0
        or type(body) is not str
        or len(body) > _MAX_COMMENT_CHARS
        or type(user) is not dict
    ):
        raise GitHubDataError("GitHub API returned malformed Issue comment response")
    author = cast(dict[object, object], user).get("login")
    author_type = cast(dict[object, object], user).get("type")
    if not _is_valid_actor_login(author) or author_type not in {"User", "Bot"}:
        raise GitHubDataError("GitHub API returned malformed Issue comment response")
    return IssueComment(
        id=comment_id,
        body=body,
        author=cast(str, author),
        author_is_bot=author_type == "Bot",
    )


def _issue_from_payload(
    payload: object,
    *,
    expected_number: int | None = None,
    reject_pull: bool = False,
) -> GitHubIssue:
    if type(payload) is not dict:
        raise GitHubDataError("GitHub API returned malformed Issue response")
    issue = cast(dict[object, object], payload)
    number = issue.get("number")
    body = issue.get("body")
    title = issue.get("title")
    user = issue.get("user")
    labels = issue.get("labels")
    assignees = issue.get("assignees")
    if (
        type(number) is not int
        or number <= 0
        or (expected_number is not None and number != expected_number)
        or (body is not None and type(body) is not str)
        or type(title) is not str
        or not title
        or len(title) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in title)
        or type(user) is not dict
        or type(labels) is not list
        or type(assignees) is not list
        or (reject_pull and "pull_request" in issue)
    ):
        raise GitHubDataError("GitHub API returned malformed Issue response")
    requester = cast(dict[object, object], user).get("login")
    if not _is_valid_actor_login(requester):
        raise GitHubDataError("GitHub API returned malformed Issue response")

    label_names: list[str] = []
    for row in cast(list[object], labels):
        if type(row) is not dict:
            raise GitHubDataError("GitHub API returned malformed Issue response")
        name = cast(dict[object, object], row).get("name")
        if not _is_valid_label_name(name):
            raise GitHubDataError("GitHub API returned malformed Issue response")
        label_names.append(cast(str, name))
    if len(set(label_names)) != len(label_names):
        raise GitHubDataError("GitHub API returned malformed Issue response")

    assignee_names: list[str] = []
    for row in cast(list[object], assignees):
        if type(row) is not dict:
            raise GitHubDataError("GitHub API returned malformed Issue response")
        login = cast(dict[object, object], row).get("login")
        if not _is_valid_actor_login(login):
            raise GitHubDataError("GitHub API returned malformed Issue response")
        normalized = cast(str, login).casefold()
        if normalized.endswith("[bot]"):
            raise GitHubDataError("GitHub API returned malformed Issue response")
        assignee_names.append(normalized)
    if len(set(assignee_names)) != len(assignee_names):
        raise GitHubDataError("GitHub API returned malformed Issue response")
    return GitHubIssue(
        number=cast(int, number),
        body="" if body is None else cast(str, body),
        requester=cast(str, requester),
        labels=tuple(label_names),
        title=cast(str, title),
        assignees=tuple(assignee_names),
    )


def _validate_assignee_login(value: object) -> None:
    if (
        type(value) is not str
        or not _is_valid_actor_login(value)
        or value != value.casefold()
        or value.endswith("[bot]")
    ):
        raise GitHubValidationError("assignee must be a canonical GitHub user login")


def _validate_sha(sha: object) -> None:
    if type(sha) is not str or _SHA_PATTERN.fullmatch(sha) is None:
        raise GitHubValidationError("SHA must be a lowercase 40-character hexadecimal value")


def _validate_script_path(path: object) -> None:
    if type(path) is not str or not path or path.startswith("/"):
        raise GitHubValidationError("script path must be repository-relative")
    if "\\" in path or "%" in path or "?" in path or "#" in path:
        raise GitHubValidationError("script path is invalid")
    segments = path.split("/")
    if any(
        not segment or segment in {".", ".."} or _PATH_SEGMENT_PATTERN.fullmatch(segment) is None
        for segment in segments
    ):
        raise GitHubValidationError("script path is invalid")


def _normalize_reviewers(reviewers: object) -> frozenset[str]:
    if type(reviewers) not in {set, frozenset} or not reviewers:
        raise GitHubValidationError("allowed_reviewers must be an explicit non-empty set")
    normalized: set[str] = set()
    for reviewer in cast(set[object] | frozenset[object], reviewers):
        try:
            login = normalize_actor_login(reviewer)
        except GitHubValidationError:
            raise GitHubValidationError(
                "allowed_reviewers contains an invalid GitHub login"
            ) from None
        if login in normalized:
            raise GitHubValidationError("allowed_reviewers contains duplicate GitHub logins")
        normalized.add(login)
    return frozenset(normalized)


def normalize_actor_login(value: object) -> str:
    """
    Validate and case-normalize a GitHub user or ``[bot]`` actor login.

    :param value: The candidate GitHub actor login.

    :returns: Its case-folded canonical form.

    :raises GitHubValidationError: If the actor login is malformed.
    """
    if not _is_valid_actor_login(value):
        raise GitHubValidationError("GitHub actor login is invalid")
    return cast(str, value).casefold()


def _is_valid_actor_login(value: object) -> bool:
    if type(value) is not str or not value or len(value) > _MAX_ACTOR_LOGIN_LENGTH:
        return False
    normalized = value.casefold()
    if normalized.endswith("[bot]"):
        name = value[:-5]
        max_name_length = _MAX_ACTOR_LOGIN_LENGTH - 5
    else:
        name = value
        max_name_length = _MAX_USER_LOGIN_LENGTH
    return 1 <= len(name) <= max_name_length and _ACTOR_NAME_PATTERN.fullmatch(name) is not None


def _normalize_required_checks(checks: object) -> frozenset[str]:
    if type(checks) not in {set, frozenset} or not checks:
        raise GitHubValidationError("required_checks must be an explicit non-empty set")
    normalized: set[str] = set()
    for check in cast(set[object] | frozenset[object], checks):
        if not _is_valid_check_name(check):
            raise GitHubValidationError("required_checks contains an invalid check name")
        normalized.add(cast(str, check))
    return frozenset(normalized)


def _is_valid_check_name(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value) <= 100
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _parse_timestamp(value: object) -> datetime:
    if type(value) is not str or not value:
        raise GitHubDataError("GitHub API returned malformed timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise GitHubDataError("GitHub API returned malformed timestamp") from None
    if parsed.tzinfo is None:
        raise GitHubDataError("GitHub API returned malformed timestamp")
    return parsed.astimezone(timezone.utc)


def _is_qualifying_pull(payload: object, sha: str) -> bool:
    if type(payload) is not dict:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    pull = cast(dict[object, object], payload)
    head = pull.get("head")
    if type(head) is not dict:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    head_sha = cast(dict[object, object], head).get("sha")
    if type(head_sha) is not str or _SHA_PATTERN.fullmatch(head_sha) is None:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    state = pull.get("state")
    merged_at = pull.get("merged_at")
    if state not in {"open", "closed"}:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    if state == "open":
        if merged_at is not None:
            raise GitHubDataError("GitHub API returned malformed pull evidence")
        qualifies_by_state = True
    elif merged_at is None:
        qualifies_by_state = False
    else:
        _parse_timestamp(merged_at)
        qualifies_by_state = True
    return head_sha == sha and qualifies_by_state


def _effective_review_decision(
    rows: list[dict[str, object]],
    allowed_reviewers: frozenset[str],
    sha: str,
) -> tuple[bool, bool]:
    substantive: dict[str, list[_ReviewState]] = {}
    for row in rows:
        state = row.get("state")
        commit_id = row.get("commit_id")
        submitted_at = row.get("submitted_at")
        user = row.get("user")
        if type(state) is not str or state not in _REVIEW_STATES or type(user) is not dict:
            raise GitHubDataError("GitHub API returned malformed review evidence")
        login = cast(dict[object, object], user).get("login")
        if not _is_valid_actor_login(login):
            raise GitHubDataError("GitHub API returned malformed review evidence")
        if state == "PENDING":
            raise GitHubDataError("GitHub API returned pending review evidence")
        if type(commit_id) is not str or _SHA_PATTERN.fullmatch(commit_id) is None:
            raise GitHubDataError("GitHub API returned malformed review evidence")
        timestamp = _parse_timestamp(submitted_at)
        normalized_login = cast(str, login).casefold()
        if normalized_login not in allowed_reviewers or state == "COMMENTED":
            continue
        substantive.setdefault(normalized_login, []).append(
            _ReviewState(state, commit_id, timestamp)
        )

    latest: list[_ReviewState] = []
    for reviewer_rows in substantive.values():
        latest_time = max(row.submitted_at for row in reviewer_rows)
        tied = [row for row in reviewer_rows if row.submitted_at == latest_time]
        if len({(row.state, row.commit_id) for row in tied}) != 1:
            raise GitHubDataError("GitHub API returned ambiguous review ordering")
        latest.append(tied[0])
    blocked = any(row.state == "CHANGES_REQUESTED" for row in latest)
    approved = any(row.state == "APPROVED" and row.commit_id == sha for row in latest)
    return approved, blocked


def _unsuccessful_required_checks(
    rows: list[dict[str, object]],
    required_checks: frozenset[str],
) -> list[str]:
    matching: dict[str, list[tuple[str, object]]] = {name: [] for name in required_checks}
    for row in rows:
        name = row.get("name")
        status = row.get("status")
        conclusion = row.get("conclusion")
        if (
            not _is_valid_check_name(name)
            or type(status) is not str
            or status not in _CHECK_STATUSES
            or (
                conclusion is not None
                and (type(conclusion) is not str or conclusion not in _CHECK_CONCLUSIONS)
            )
        ):
            raise GitHubDataError("GitHub API returned malformed check evidence")
        if cast(str, name) in matching:
            matching[cast(str, name)].append((status, conclusion))
    return sorted(
        name
        for name, evidence in matching.items()
        if len(evidence) != 1 or evidence[0] != ("completed", "success")
    )
