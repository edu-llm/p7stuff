from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest
import requests

from edullm.github import (
    GitHubAPIError,
    GitHubClient,
    GitHubDataError,
    GitHubIssue,
    GitHubValidationError,
    IssueComment,
)

REPO = "edu-llm/OLMo-core"
BASE_URL = "https://api.github.test"


class FakeResponse:
    def __init__(
        self,
        payload,
        *,
        status_code=200,
        error_message=None,
        json_error=None,
    ):
        self.payload = payload
        self.status_code = status_code
        self.error_message = error_message
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.error_message or f"HTTP {self.status_code}")

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.responses = {}
        self.failure = None

    def add(self, method, path, payload, *, params=None, **response_kwargs):
        key = (method, f"{BASE_URL}{path}", _params_key(params))
        self.responses.setdefault(key, []).append(FakeResponse(payload, **response_kwargs))

    def get(self, url, *, params=None, timeout=None):
        return self._call("GET", url, params=params, json=None, timeout=timeout)

    def post(self, url, *, params=None, json=None, timeout=None):
        return self._call("POST", url, params=params, json=json, timeout=timeout)

    def patch(self, url, *, params=None, json=None, timeout=None):
        return self._call("PATCH", url, params=params, json=json, timeout=timeout)

    def delete(self, url, *, params=None, json=None, timeout=None):
        return self._call("DELETE", url, params=params, json=json, timeout=timeout)

    def _call(self, method, url, *, params, json, timeout):
        copied_params = dict(params) if params is not None else None
        self.calls.append((method, url, copied_params, json, timeout))
        if self.failure is not None:
            raise self.failure
        key = (method, url, _params_key(params))
        responses = self.responses.get(key)
        if not responses:
            raise AssertionError(f"unexpected request: {key!r}")
        return responses.pop(0)


def _params_key(
    params: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((params or {}).items()))


def _issue_payload(
    *,
    body="body",
    labels=None,
    number=42,
    login="student",
    title="Queue title",
    assignees=None,
):
    return {
        "number": number,
        "body": body,
        "title": title,
        "user": {"login": login},
        "labels": [{"name": label} for label in (labels or ["edullm-job"])],
        "assignees": [{"login": assignee} for assignee in (assignees or [])],
    }


def _comment_payload(
    *,
    comment_id=7,
    body="comment",
    login="github-actions[bot]",
    user_type="Bot",
):
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": login, "type": user_type},
    }


def _client(session):
    return GitHubClient("token", REPO, base_url=BASE_URL, session=session)


def test_fetch_issue_returns_validated_immutable_record():
    session = FakeSession()
    session.add(
        "GET",
        f"/repos/{REPO}/issues/42",
        _issue_payload(labels=["status:ready", "edullm-job"]),
    )

    issue = _client(session).fetch_issue(42)

    assert issue == GitHubIssue(
        number=42,
        body="body",
        requester="student",
        labels=("status:ready", "edullm-job"),
        title="Queue title",
        assignees=(),
    )
    with pytest.raises(FrozenInstanceError):
        issue.body = "edited"  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        _issue_payload(number=41),
        {**_issue_payload(), "user": {}},
        {**_issue_payload(), "title": None},
        {**_issue_payload(), "labels": ["edullm-job"]},
        {**_issue_payload(), "labels": [{"name": ""}]},
        {**_issue_payload(), "assignees": [{"login": ""}]},
        {**_issue_payload(), "assignees": [{"login": "alice"}, {"login": "Alice"}]},
        {
            **_issue_payload(),
            "labels": [{"name": "edullm-job"}, {"name": "edullm-job"}],
        },
    ],
)
def test_fetch_issue_rejects_malformed_response_shapes(payload):
    session = FakeSession()
    session.add("GET", f"/repos/{REPO}/issues/42", payload)

    with pytest.raises(GitHubDataError, match="malformed Issue response"):
        _client(session).fetch_issue(42)


def test_fetch_issue_normalizes_nullable_github_body_to_empty_text():
    session = FakeSession()
    session.add(
        "GET",
        f"/repos/{REPO}/issues/42",
        _issue_payload(body=None),
    )

    assert _client(session).fetch_issue(42).body == ""


def test_list_issue_comments_uses_pagination_and_validates_bot_authorship():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/comments"
    session.add(
        "GET",
        path,
        [_comment_payload()],
        params={"page": 1, "per_page": 100},
    )

    comments = _client(session).list_issue_comments(42)

    assert comments == (
        IssueComment(
            id=7,
            body="comment",
            author="github-actions[bot]",
            author_is_bot=True,
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        [{"id": 7, "body": "comment", "user": {"login": "student"}}],
        [_comment_payload(comment_id=0)],
        [_comment_payload(body=None)],
        [_comment_payload(user_type="Organization")],
        [
            _comment_payload(comment_id=7),
            _comment_payload(comment_id=7),
        ],
    ],
)
def test_list_issue_comments_rejects_malformed_rows(payload):
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/comments"
    session.add(
        "GET",
        path,
        payload,
        params={"page": 1, "per_page": 100},
    )

    with pytest.raises(GitHubDataError, match="malformed Issue comment response"):
        _client(session).list_issue_comments(42)


def test_create_issue_comment_json_encodes_body_and_validates_response():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/comments"
    body = "<!-- edullm-validation:v1 -->\nRequest needs changes."
    session.add("POST", path, _comment_payload(body=body))

    comment = _client(session).create_issue_comment(42, body)

    assert comment.body == body
    assert session.calls == [
        (
            "POST",
            f"{BASE_URL}{path}",
            None,
            {"body": body},
            (5, 20),
        )
    ]


def test_update_issue_comment_json_encodes_body_and_requires_matching_id_and_body():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/comments/7"
    body = "<!-- edullm-status:v1 -->\n{}"
    session.add("PATCH", path, _comment_payload(body=body))

    comment = _client(session).update_issue_comment(7, body)

    assert comment.id == 7
    assert session.calls[0][3] == {"body": body}


@pytest.mark.parametrize(
    "method,payload",
    [
        ("create", _comment_payload(body="different")),
        ("update", _comment_payload(comment_id=8, body="expected")),
    ],
)
def test_comment_writes_reject_mismatched_success_payloads(method, payload):
    session = FakeSession()
    if method == "create":
        path = f"/repos/{REPO}/issues/42/comments"
        session.add("POST", path, payload)
    else:
        path = f"/repos/{REPO}/issues/comments/7"
        session.add("PATCH", path, payload)

    with pytest.raises(GitHubDataError, match="malformed Issue comment response"):
        if method == "create":
            _client(session).create_issue_comment(42, "expected")
        else:
            _client(session).update_issue_comment(7, "expected")


def test_add_issue_status_label_uses_targeted_json_and_preserves_server_labels():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/labels"
    labels = ("edullm-job", "concurrent", "status:requested")
    session.add(
        "POST",
        path,
        [{"name": label} for label in labels],
    )

    result = _client(session).add_issue_status_label(42, "status:requested")

    assert result == labels
    assert session.calls == [
        (
            "POST",
            f"{BASE_URL}{path}",
            None,
            {"labels": ["status:requested"]},
            (5, 20),
        )
    ]


def test_task_5_can_target_the_assigned_status_label():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/labels"
    session.add(
        "POST",
        path,
        [{"name": "edullm-job"}, {"name": "status:ready"}, {"name": "status:assigned"}],
    )

    labels = _client(session).add_issue_status_label(42, "status:assigned")

    assert "status:assigned" in labels
    assert session.calls[0][3] == {"labels": ["status:assigned"]}


def test_remove_issue_status_label_encodes_path_and_validates_success():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/labels/status%3Aready"
    session.add(
        "DELETE",
        path,
        [{"name": "edullm-job"}, {"name": "concurrent"}],
    )

    removed = _client(session).remove_issue_status_label(42, "status:ready")

    assert removed is True
    assert session.calls == [
        (
            "DELETE",
            f"{BASE_URL}{path}",
            None,
            None,
            (5, 20),
        )
    ]


def test_remove_issue_status_label_treats_only_404_as_already_absent():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/labels/status%3Aready"
    session.add("DELETE", path, {"message": "Not Found"}, status_code=404)

    assert _client(session).remove_issue_status_label(42, "status:ready") is False


@pytest.mark.parametrize("status_code", [401, 403, 500])
def test_remove_issue_status_label_keeps_non_404_failures_as_errors(status_code):
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/labels/status%3Aready"
    session.add(
        "DELETE",
        path,
        {"message": "failure"},
        status_code=status_code,
    )

    with pytest.raises(GitHubAPIError, match="GitHub API request failed"):
        _client(session).remove_issue_status_label(42, "status:ready")


@pytest.mark.parametrize(
    "issue_number",
    [0, -1, True, "42", "1/comments"],
)
def test_issue_operations_reject_unsafe_issue_identifiers(issue_number):
    client = _client(FakeSession())

    with pytest.raises(GitHubValidationError, match="positive integer"):
        client.fetch_issue(issue_number)


@pytest.mark.parametrize(
    "label",
    [
        "",
        "status:unknown",
        "status:ready/other",
        "status%3Aready",
        " status:ready",
        7,
    ],
)
def test_targeted_status_label_operations_reject_other_labels(label):
    client = _client(FakeSession())

    with pytest.raises(GitHubValidationError, match="status label"):
        client.add_issue_status_label(42, label)
    with pytest.raises(GitHubValidationError, match="status label"):
        client.remove_issue_status_label(42, label)


def test_general_full_label_replacement_is_not_exposed():
    assert not hasattr(_client(FakeSession()), "replace_issue_labels")


def test_list_active_queue_issues_scans_current_open_issues_and_rejects_pull_rows():
    session = FakeSession()
    path = f"/repos/{REPO}/issues?labels=edullm-job&state=open"
    session.add(
        "GET",
        path,
        [
            _issue_payload(
                number=42,
                labels=["edullm-job", "status:ready"],
                assignees=["alice"],
            ),
            _issue_payload(
                number=43,
                labels=["edullm-job", "status:assigned"],
                login="other",
            ),
        ],
        params={"page": 1, "per_page": 100},
    )

    issues = _client(session).list_active_queue_issues()

    assert [issue.number for issue in issues] == [42, 43]
    assert issues[0].assignees == ("alice",)
    assert session.calls[0][1].endswith("?labels=edullm-job&state=open")

    bad_session = FakeSession()
    bad_session.add(
        "GET",
        path,
        [{**_issue_payload(), "pull_request": {"url": "https://api.github.test/pulls/42"}}],
        params={"page": 1, "per_page": 100},
    )
    with pytest.raises(GitHubDataError, match="malformed Issue response"):
        _client(bad_session).list_active_queue_issues()


def test_targeted_assignee_add_preserves_existing_assignees_and_validates_response():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/assignees"
    session.add(
        "POST",
        path,
        _issue_payload(assignees=["reviewer", "alice"]),
    )

    result = _client(session).add_issue_assignee(42, "alice")

    assert result == ("reviewer", "alice")
    assert session.calls[0][3] == {"assignees": ["alice"]}


def test_targeted_assignee_remove_preserves_others_and_treats_only_404_as_absent():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/assignees"
    session.add(
        "DELETE",
        path,
        _issue_payload(assignees=["reviewer"]),
    )

    assert _client(session).remove_issue_assignee(42, "alice") is True
    assert session.calls[0][3] == {"assignees": ["alice"]}

    absent = FakeSession()
    absent.add("DELETE", path, {"message": "Not Found"}, status_code=404)
    assert _client(absent).remove_issue_assignee(42, "alice") is False


@pytest.mark.parametrize("login", ["", "Alice", "alice/other", "review-app[bot]", 7])
def test_targeted_assignee_operations_reject_unsafe_or_noncanonical_logins(login):
    client = _client(FakeSession())

    with pytest.raises(GitHubValidationError, match="assignee"):
        client.add_issue_assignee(42, login)
    with pytest.raises(GitHubValidationError, match="assignee"):
        client.remove_issue_assignee(42, login)


def test_targeted_assignee_write_requires_requested_postcondition():
    session = FakeSession()
    path = f"/repos/{REPO}/issues/42/assignees"
    session.add("POST", path, _issue_payload(assignees=["reviewer"]))

    with pytest.raises(GitHubDataError, match="assignee"):
        _client(session).add_issue_assignee(42, "alice")


@pytest.mark.parametrize("body", ["", "x" * 65_537, "bad\x00body", 7])
def test_comment_writes_reject_invalid_bodies(body):
    client = _client(FakeSession())

    with pytest.raises(GitHubValidationError, match="comment body"):
        client.create_issue_comment(42, body)


@pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE"])
def test_write_failures_are_sanitized(method):
    token = "top-secret-token"
    session = FakeSession()
    session.failure = requests.ConnectionError(f"failure contains {token}")
    client = GitHubClient(token, REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubAPIError) as raised:
        if method == "POST":
            client.create_issue_comment(42, "body")
        elif method == "PATCH":
            client.update_issue_comment(7, "body")
        else:
            client.remove_issue_status_label(42, "status:ready")

    assert str(raised.value) == "GitHub API request failed"
    assert token not in repr(raised.value)
