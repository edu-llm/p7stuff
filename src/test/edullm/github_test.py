from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError

import pytest
import requests

from edullm.github import (
    CommitResult,
    GitHubAPIError,
    GitHubClient,
    GitHubDataError,
    GitHubValidationError,
    ReviewResult,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40
REPO = "edu-llm/OLMo-core"
BASE_URL = "https://api.github.test"
SCRIPT_PATH = "src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py"
REQUIRED_CHECKS = {"Lint", "Test", "Test scripts"}


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        error_message: str | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.error_message = error_message
        self.json_error = json_error

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(self.error_message or f"HTTP {self.status_code}")

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(
        self,
        handler: Callable[[str, Mapping[str, object] | None], FakeResponse] | None = None,
    ) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, object] | None, object]] = []
        self._handler = handler
        self._responses: dict[tuple[str, tuple[tuple[str, object], ...]], list[FakeResponse]] = {}

    def add(
        self,
        path: str,
        payload: object,
        *,
        params: Mapping[str, object] | None = None,
        status_code: int = 200,
        error_message: str | None = None,
        json_error: Exception | None = None,
    ) -> None:
        key = (f"{BASE_URL}{path}", _params_key(params))
        self._responses.setdefault(key, []).append(
            FakeResponse(
                payload,
                status_code=status_code,
                error_message=error_message,
                json_error=json_error,
            )
        )

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        timeout: object = None,
    ) -> FakeResponse:
        copied_params = dict(params) if params is not None else None
        self.calls.append((url, copied_params, timeout))
        if self._handler is not None:
            return self._handler(url, params)
        key = (url, _params_key(params))
        responses = self._responses.get(key)
        if not responses:
            raise AssertionError(f"unexpected request: {key!r}")
        return responses.pop(0)


def _params_key(params: Mapping[str, object] | None) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((params or {}).items()))


def _review(
    state: str = "APPROVED",
    *,
    sha: str = SHA,
    login: object = "operator",
    submitted_at: object = "2026-07-22T10:00:00Z",
) -> dict[str, object]:
    return {
        "state": state,
        "commit_id": sha,
        "submitted_at": submitted_at,
        "user": {"login": login},
    }


def _successful_checks() -> list[dict[str, object]]:
    return [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in sorted(REQUIRED_CHECKS)
    ]


def _open_pull(sha: str = SHA) -> dict[str, object]:
    return {"head": {"sha": sha}, "state": "open", "merged_at": None}


def _merged_pull(sha: str = SHA) -> dict[str, object]:
    return {
        "head": {"sha": sha},
        "state": "closed",
        "merged_at": "2026-07-22T12:00:00Z",
    }


def _gate_client(
    *,
    pulls: object = None,
    details: Mapping[int, object] | None = None,
    final_details: Mapping[int, object] | None = None,
    reviews: Mapping[int, object] | None = None,
    checks: object = None,
    check_payload: object = None,
    file_status: int = 200,
    file_payload: object = None,
) -> tuple[GitHubClient, FakeSession]:
    pulls = [{"number": 7}] if pulls is None else pulls
    details = {7: _open_pull()} if details is None else details
    final_details = details if final_details is None else final_details
    reviews = {7: [_review()]} if reviews is None else reviews
    checks = _successful_checks() if checks is None else checks
    check_payload = (
        {"total_count": len(checks) if isinstance(checks, list) else 0, "check_runs": checks}
        if check_payload is None
        else check_payload
    )
    file_payload = {"type": "file", "path": SCRIPT_PATH} if file_payload is None else file_payload
    session = FakeSession()
    session.add(
        f"/repos/{REPO}/commits/{SHA}/pulls",
        pulls,
        params={"page": 1, "per_page": 100},
    )
    for number, payload in details.items():
        session.add(f"/repos/{REPO}/pulls/{number}", payload)
        if number in final_details:
            session.add(f"/repos/{REPO}/pulls/{number}", final_details[number])
    for number, payload in reviews.items():
        session.add(
            f"/repos/{REPO}/pulls/{number}/reviews",
            payload,
            params={"page": 1, "per_page": 100},
        )
    session.add(
        f"/repos/{REPO}/commits/{SHA}/check-runs?filter=latest",
        check_payload,
        params={"page": 1, "per_page": 100},
    )
    session.add(
        f"/repos/{REPO}/contents/{SCRIPT_PATH}",
        file_payload,
        params={"ref": SHA},
        status_code=file_status,
    )
    return GitHubClient("token", REPO, base_url=BASE_URL, session=session), session


def _result(client: GitHubClient, **overrides: object) -> ReviewResult:
    arguments: dict[str, object] = {
        "script_path": SCRIPT_PATH,
        "allowed_reviewers": {"operator"},
        "required_checks": REQUIRED_CHECKS,
    }
    arguments.update(overrides)
    return client.reviewed_commit(SHA, **arguments)  # type: ignore[arg-type]


def test_client_sets_current_versioned_json_headers_and_finite_timeout():
    session = FakeSession()
    session.add("/rate_limit", {"resources": {}})
    client = GitHubClient("secret-token", REPO, base_url=BASE_URL, session=session)

    assert client.get("/rate_limit") == {"resources": {}}
    assert session.headers == {
        "Authorization": "Bearer secret-token",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    assert session.calls == [(f"{BASE_URL}/rate_limit", None, (5, 20))]


def test_client_repr_does_not_expose_token():
    client = GitHubClient("top-secret-token", REPO, base_url=BASE_URL, session=FakeSession())

    assert "top-secret-token" not in repr(client)
    assert REPO in repr(client)


@pytest.mark.parametrize(
    "token, repo",
    [
        ("", REPO),
        (" token", REPO),
        ("token", ""),
        ("token", "owner"),
        ("token", "owner/repo/extra"),
        ("token", "../repo"),
        ("token", "owner/../repo"),
        ("token", "owner/repo?ref=main"),
        ("token", "review-app[bot]/repo"),
    ],
)
def test_client_rejects_invalid_credentials_or_repository(token, repo):
    with pytest.raises(GitHubValidationError):
        GitHubClient(token, repo, base_url=BASE_URL, session=FakeSession())


@pytest.mark.parametrize(
    "path",
    [
        "repos/owner/repo",
        "https://evil.example/repos/owner/repo",
        "//evil.example/repos/owner/repo",
        "/repos/owner/../secret",
        "/repos/owner/%2e%2e/secret",
        "/repos/owner/repo#fragment",
        "/repos/owner/repo\nX-Evil: true",
    ],
)
def test_get_rejects_unsafe_api_paths(path):
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=FakeSession())

    with pytest.raises(GitHubValidationError):
        client.get(path)


def test_get_redacts_http_response_and_token_from_errors():
    token = "top-secret-token"
    session = FakeSession()
    session.add(
        "/user",
        {"message": token},
        status_code=500,
        error_message=f"response body contains {token}",
    )
    client = GitHubClient(token, REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubAPIError) as raised:
        client.get("/user")

    assert str(raised.value) == "GitHub API request failed"
    assert token not in repr(raised.value)


def test_get_redacts_json_decode_errors():
    session = FakeSession()
    session.add("/user", None, json_error=ValueError("sensitive response body"))
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="GitHub API returned malformed JSON") as raised:
        client.get("/user")

    assert "sensitive response body" not in repr(raised.value)


def test_get_redacts_network_failures():
    token = "top-secret-token"

    def fail_request(url: str, params: Mapping[str, object] | None) -> FakeResponse:  # noqa: ARG001
        raise requests.ConnectionError(f"connection failed with {token}")

    client = GitHubClient(
        token,
        REPO,
        base_url=BASE_URL,
        session=FakeSession(handler=fail_request),
    )

    with pytest.raises(GitHubAPIError) as raised:
        client.get("/user")

    assert str(raised.value) == "GitHub API request failed"
    assert token not in repr(raised.value)


def test_paginated_get_preserves_existing_query_and_reads_second_page():
    session = FakeSession()
    first_page = [{"id": index} for index in range(100)]
    session.add(
        "/reviews?filter=latest",
        first_page,
        params={"page": 1, "per_page": 100},
    )
    session.add(
        "/reviews?filter=latest",
        [{"id": 100}],
        params={"page": 2, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    assert client.paginated_get("/reviews?filter=latest") == first_page + [{"id": 100}]


def test_paginated_get_extracts_a_keyed_envelope():
    session = FakeSession()
    session.add(
        "/rows",
        {"rows": [{"id": 1}]},
        params={"page": 1, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    assert client.paginated_get("/rows", key="rows") == [{"id": 1}]


@pytest.mark.parametrize(
    "payload",
    [
        {"check_runs": []},
        {"total_count": True, "check_runs": []},
        {"total_count": "0", "check_runs": []},
        {"total_count": -1, "check_runs": []},
    ],
)
def test_counted_pagination_rejects_missing_or_malformed_total_count(payload):
    session = FakeSession()
    session.add(
        "/check-runs?filter=latest",
        payload,
        params={"page": 1, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="malformed paginated response"):
        client.paginated_get(
            "/check-runs?filter=latest",
            key="check_runs",
            total_count_key="total_count",
        )


@pytest.mark.parametrize("total_count", [0, 2])
def test_counted_pagination_rejects_one_page_count_mismatch(total_count):
    session = FakeSession()
    session.add(
        "/check-runs?filter=latest",
        {"total_count": total_count, "check_runs": [{"id": 1}]},
        params={"page": 1, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="paginated response count is inconsistent"):
        client.paginated_get(
            "/check-runs?filter=latest",
            key="check_runs",
            total_count_key="total_count",
        )


@pytest.mark.parametrize("total_count", [100, 102])
def test_counted_pagination_rejects_multiple_page_count_mismatch(total_count):
    session = FakeSession()
    first_page = [{"id": index} for index in range(100)]
    path = "/check-runs?filter=latest"
    session.add(
        path,
        {"total_count": total_count, "check_runs": first_page},
        params={"page": 1, "per_page": 100},
    )
    session.add(
        path,
        {"total_count": total_count, "check_runs": [{"id": 100}]},
        params={"page": 2, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="paginated response count is inconsistent"):
        client.paginated_get(
            path,
            key="check_runs",
            total_count_key="total_count",
        )


def test_counted_pagination_rejects_a_changing_total_count():
    session = FakeSession()
    first_page = [{"id": index} for index in range(100)]
    path = "/check-runs?filter=latest"
    session.add(
        path,
        {"total_count": 101, "check_runs": first_page},
        params={"page": 1, "per_page": 100},
    )
    session.add(
        path,
        {"total_count": 102, "check_runs": [{"id": 100}]},
        params={"page": 2, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="paginated response count is inconsistent"):
        client.paginated_get(
            path,
            key="check_runs",
            total_count_key="total_count",
        )


def test_counted_pagination_accepts_an_exact_stable_paginated_count():
    session = FakeSession()
    first_page = [{"id": index} for index in range(100)]
    path = "/check-runs?filter=latest"
    session.add(
        path,
        {"total_count": 101, "check_runs": first_page},
        params={"page": 1, "per_page": 100},
    )
    session.add(
        path,
        {"total_count": 101, "check_runs": [{"id": 100}]},
        params={"page": 2, "per_page": 100},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    assert client.paginated_get(
        path,
        key="check_runs",
        total_count_key="total_count",
    ) == first_page + [{"id": 100}]
    assert session.calls == [
        (f"{BASE_URL}{path}", {"page": 1, "per_page": 100}, (5, 20)),
        (f"{BASE_URL}{path}", {"page": 2, "per_page": 100}, (5, 20)),
    ]


@pytest.mark.parametrize(
    "payload, key",
    [
        ({"not_rows": []}, None),
        ([{"id": 1}, "bad-row"], None),
        ([], "check_runs"),
        ({"check_runs": "not-a-list"}, "check_runs"),
    ],
)
def test_paginated_get_rejects_malformed_envelopes_and_rows(payload, key):
    session = FakeSession()
    session.add("/rows", payload, params={"page": 1, "per_page": 100})
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="malformed paginated response"):
        client.paginated_get("/rows", key=key)


def test_paginated_get_has_a_finite_page_bound():
    def full_page(url: str, params: Mapping[str, object] | None) -> FakeResponse:  # noqa: ARG001
        return FakeResponse([{"id": index} for index in range(100)])

    client = GitHubClient(
        "token",
        REPO,
        base_url=BASE_URL,
        session=FakeSession(handler=full_page),
        max_pages=3,
    )

    with pytest.raises(GitHubDataError, match="GitHub pagination limit exceeded"):
        client.paginated_get("/rows")


def test_file_exists_uses_the_exact_sha_and_accepts_only_a_file_payload():
    session = FakeSession()
    session.add(
        f"/repos/{REPO}/contents/{SCRIPT_PATH}",
        {"type": "file", "path": SCRIPT_PATH},
        params={"ref": SHA},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    assert client.file_exists(SCRIPT_PATH, ref=SHA) is True


def test_file_exists_returns_false_only_for_404():
    session = FakeSession()
    session.add(
        f"/repos/{REPO}/contents/{SCRIPT_PATH}",
        {"message": "Not Found"},
        params={"ref": SHA},
        status_code=404,
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    assert client.file_exists(SCRIPT_PATH, ref=SHA) is False


@pytest.mark.parametrize("status_code", [401, 403, 500])
def test_file_exists_keeps_non_404_failures_as_errors(status_code):
    session = FakeSession()
    session.add(
        f"/repos/{REPO}/contents/{SCRIPT_PATH}",
        {"message": "failure"},
        params={"ref": SHA},
        status_code=status_code,
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubAPIError, match="GitHub API request failed"):
        client.file_exists(SCRIPT_PATH, ref=SHA)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute.py",
        "../train.py",
        "src/../train.py",
        "src/%2e%2e/train.py",
        r"src\train.py",
        "src/train.py?ref=main",
    ],
)
def test_file_exists_rejects_unsafe_script_paths(path):
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=FakeSession())

    with pytest.raises(GitHubValidationError):
        client.file_exists(path, ref=SHA)


def test_file_exists_rejects_malformed_success_payload():
    session = FakeSession()
    session.add(
        f"/repos/{REPO}/contents/{SCRIPT_PATH}",
        [{"type": "file"}],
        params={"ref": SHA},
    )
    client = GitHubClient("token", REPO, base_url=BASE_URL, session=session)

    with pytest.raises(GitHubDataError, match="malformed contents response"):
        client.file_exists(SCRIPT_PATH, ref=SHA)


def _commit_client(
    *,
    commit_status: int = 200,
    commit_payload: object = None,
    file_status: int = 200,
    file_payload: object = None,
) -> tuple[GitHubClient, FakeSession]:
    commit_payload = {"sha": SHA} if commit_payload is None else commit_payload
    file_payload = {"type": "file", "path": SCRIPT_PATH} if file_payload is None else file_payload
    session = FakeSession()
    session.add(
        f"/repos/{REPO}/commits/{SHA}",
        commit_payload,
        status_code=commit_status,
    )
    session.add(
        f"/repos/{REPO}/contents/{SCRIPT_PATH}",
        file_payload,
        params={"ref": SHA},
        status_code=file_status,
    )
    return GitHubClient("token", REPO, base_url=BASE_URL, session=session), session


def test_executable_commit_accepts_exact_pushed_commit_and_script():
    client, session = _commit_client()

    result = client.executable_commit(SHA, script_path=SCRIPT_PATH)

    assert result == CommitResult(True, "commit and script exist")
    assert all("/pulls" not in call[0] for call in session.calls)
    assert all("/reviews" not in call[0] for call in session.calls)
    assert all("/check-runs" not in call[0] for call in session.calls)


def test_executable_commit_rejects_missing_commit():
    client, _ = _commit_client(
        commit_status=404,
        commit_payload={"message": "Not Found"},
    )

    assert client.executable_commit(SHA, script_path=SCRIPT_PATH) == CommitResult(
        False,
        "commit does not exist at the requested SHA",
    )


@pytest.mark.parametrize(
    "commit_payload",
    [
        {},
        {"sha": OTHER_SHA},
        {"sha": "not-a-sha"},
        {"not_sha": SHA},
    ],
)
def test_executable_commit_rejects_malformed_commit_payload(commit_payload):
    client, _ = _commit_client(commit_payload=commit_payload)

    assert client.executable_commit(SHA, script_path=SCRIPT_PATH) == CommitResult(
        False,
        "malformed GitHub commit evidence",
    )


def test_executable_commit_rejects_missing_script():
    client, _ = _commit_client(file_status=404, file_payload={"message": "Not Found"})

    assert client.executable_commit(SHA, script_path=SCRIPT_PATH) == CommitResult(
        False,
        "script does not exist at the requested SHA",
    )


def test_executable_commit_rejects_malformed_contents_payload():
    client, _ = _commit_client(file_payload=[{"type": "file"}])

    assert client.executable_commit(SHA, script_path=SCRIPT_PATH) == CommitResult(
        False,
        "malformed GitHub contents evidence",
    )


def test_commit_result_is_immutable():
    result = CommitResult(False, "rejected")

    with pytest.raises(FrozenInstanceError):
        result.available = True  # type: ignore[misc]


@pytest.mark.parametrize("details", [_open_pull(), _merged_pull()])
def test_reviewed_commit_accepts_exact_head_open_or_merged_pull(details):
    client, session = _gate_client(details={7: details})

    result = _result(client)

    assert result == ReviewResult(
        approved=True,
        pr_number=7,
        reason="approved PR with successful checks and script",
    )
    assert (
        f"{BASE_URL}/repos/{REPO}/commits/{SHA}/check-runs?filter=latest",
        {"page": 1, "per_page": 100},
        (5, 20),
    ) in session.calls


def test_reviewed_commit_rejects_closed_unmerged_pull():
    client, _ = _gate_client(
        details={7: {"head": {"sha": SHA}, "state": "closed", "merged_at": None}}
    )

    assert _result(client) == ReviewResult(
        False,
        None,
        "no qualifying pull request has the requested SHA as its head",
    )


def test_reviewed_commit_refetches_selected_pr_head_after_checks_and_file_evidence():
    client, session = _gate_client(final_details={7: _open_pull(OTHER_SHA)})

    result = _result(client)

    assert result == ReviewResult(
        False,
        7,
        "selected pull request head changed during authorization",
    )
    pull_url = f"{BASE_URL}/repos/{REPO}/pulls/7"
    assert [call[0] for call in session.calls].count(pull_url) == 2
    assert session.calls[-1][0] == pull_url


@pytest.mark.parametrize("pulls", [[], [{"number": 7}]])
def test_reviewed_commit_rejects_unreviewed_or_non_head_commit(pulls):
    details = {} if not pulls else {7: _open_pull(OTHER_SHA)}
    client, _ = _gate_client(pulls=pulls, details=details, reviews={})

    assert _result(client).reason == (
        "no qualifying pull request has the requested SHA as its head"
    )


def test_reviewed_commit_rejects_approval_for_previous_sha():
    client, _ = _gate_client(reviews={7: [_review(sha=OTHER_SHA)]})

    assert _result(client) == ReviewResult(
        False,
        None,
        "no authorized reviewer approved the requested SHA",
    )


def test_latest_changes_requested_overrides_approval():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(submitted_at="2026-07-22T10:00:00Z"),
                _review("CHANGES_REQUESTED", submitted_at="2026-07-22T11:00:00Z"),
            ]
        }
    )

    assert _result(client) == ReviewResult(
        False,
        None,
        "an authorized reviewer currently requests changes",
    )


def test_commented_review_does_not_erase_current_approval():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(submitted_at="2026-07-22T10:00:00Z"),
                _review("COMMENTED", submitted_at="2026-07-22T11:00:00Z"),
            ]
        }
    )

    assert _result(client).approved is True


def test_dismissed_review_does_not_count_as_approval():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(submitted_at="2026-07-22T10:00:00Z"),
                _review("DISMISSED", submitted_at="2026-07-22T11:00:00Z"),
            ]
        }
    )

    assert _result(client).reason == "no authorized reviewer approved the requested SHA"


def test_authorized_reviewer_logins_are_case_insensitive():
    client, _ = _gate_client(reviews={7: [_review(login="Operator")]})

    assert _result(client, allowed_reviewers={"oPeRaToR"}).approved is True


def test_authorized_github_app_bot_can_approve_case_insensitively():
    client, _ = _gate_client(reviews={7: [_review(login="Review-App[bot]")]})

    assert _result(client, allowed_reviewers={"review-app[BOT]"}).approved is True


def test_unauthorized_bot_comment_does_not_invalidate_an_authorized_approval():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(login="operator", submitted_at="2026-07-22T10:00:00Z"),
                _review(
                    "COMMENTED",
                    login="review-app[bot]",
                    submitted_at="2026-07-22T11:00:00Z",
                ),
            ]
        }
    )

    assert _result(client).approved is True


def test_authorized_bot_changes_requested_blocks_approval():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(login="operator", submitted_at="2026-07-22T10:00:00Z"),
                _review(
                    "CHANGES_REQUESTED",
                    login="review-app[bot]",
                    submitted_at="2026-07-22T11:00:00Z",
                ),
            ]
        }
    )

    assert _result(
        client,
        allowed_reviewers={"operator", "Review-App[BOT]"},
    ) == ReviewResult(
        False,
        None,
        "an authorized reviewer currently requests changes",
    )


def test_authorized_bot_dismissal_does_not_count_as_approval():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(login="review-app[bot]", submitted_at="2026-07-22T10:00:00Z"),
                _review(
                    "DISMISSED",
                    login="Review-App[BOT]",
                    submitted_at="2026-07-22T11:00:00Z",
                ),
            ]
        }
    )

    assert _result(client, allowed_reviewers={"review-app[bot]"}) == ReviewResult(
        False,
        None,
        "no authorized reviewer approved the requested SHA",
    )


def test_unauthorized_approval_does_not_pass():
    client, _ = _gate_client(reviews={7: [_review(login="outsider")]})

    assert _result(client).reason == "no authorized reviewer approved the requested SHA"


@pytest.mark.parametrize(
    "allowed_reviewers",
    [
        set(),
        {""},
        {" operator"},
        {"operator/other"},
        {"Operator", "operator"},
        {"Review-App[bot]", "review-app[BOT]"},
        {"operator", 7},
    ],
)
def test_reviewed_commit_rejects_empty_or_malformed_reviewer_sets(allowed_reviewers):
    client, _ = _gate_client()

    with pytest.raises(GitHubValidationError, match="allowed_reviewers"):
        _result(client, allowed_reviewers=allowed_reviewers)


@pytest.mark.parametrize(
    "login",
    [
        "[bot]",
        "-review-app[bot]",
        "review-app-[bot]",
        "review--app[bot]",
        "review-app[bot",
        "review-appbot]",
        "review-app[robot]",
        "review-app[bot]suffix",
        "review/app[bot]",
        r"review\app[bot]",
        "review?app[bot]",
        "review#app[bot]",
        "review&app[bot]",
        "review=app[bot]",
        "review\napp[bot]",
        ("a" * 101) + "[bot]",
    ],
)
def test_reviewed_commit_rejects_malformed_bot_reviewer_identities(login):
    client, _ = _gate_client()

    with pytest.raises(GitHubValidationError, match="allowed_reviewers"):
        _result(client, allowed_reviewers={login})


@pytest.mark.parametrize(
    "login",
    [
        "[bot]",
        "-review-app[bot]",
        "review-app-[bot]",
        "review--app[bot]",
        "review-app[robot]",
        "review/app[bot]",
        r"review\app[bot]",
        "review?app[bot]",
        "review\napp[bot]",
        ("a" * 101) + "[bot]",
    ],
)
def test_malformed_bot_identity_in_review_payload_fails_closed(login):
    client, _ = _gate_client(reviews={7: [_review(login=login)]})

    assert _result(client) == ReviewResult(False, None, "malformed GitHub review evidence")


def test_pending_review_fails_closed():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(),
                _review("PENDING", submitted_at=None),
            ]
        }
    )

    assert _result(client) == ReviewResult(False, None, "malformed GitHub review evidence")


def test_conflicting_substantive_reviews_with_tied_timestamps_fail_closed():
    client, _ = _gate_client(
        reviews={
            7: [
                _review(),
                _review("CHANGES_REQUESTED"),
            ]
        }
    )

    assert _result(client) == ReviewResult(False, None, "malformed GitHub review evidence")


@pytest.mark.parametrize(
    "review",
    [
        {"state": "APPROVED"},
        _review(login=""),
        _review(login="operator/name"),
        _review(submitted_at="not-a-timestamp"),
        _review(state="UNKNOWN"),
    ],
)
def test_malformed_review_rows_fail_closed(review):
    client, _ = _gate_client(reviews={7: [review]})

    assert _result(client) == ReviewResult(False, None, "malformed GitHub review evidence")


def test_reviewed_commit_rejects_check_runs_without_total_count():
    client, _ = _gate_client(check_payload={"check_runs": _successful_checks()})

    assert _result(client) == ReviewResult(False, None, "malformed GitHub check evidence")


@pytest.mark.parametrize(
    "checks, missing",
    [
        (
            [
                {"name": "Lint", "status": "completed", "conclusion": "success"},
                {"name": "Test", "status": "completed", "conclusion": "success"},
            ],
            "Test scripts",
        ),
        (
            [
                {"name": "Lint", "status": "completed", "conclusion": "failure"},
                {"name": "Test", "status": "completed", "conclusion": "success"},
                {
                    "name": "Test scripts",
                    "status": "completed",
                    "conclusion": "success",
                },
            ],
            "Lint",
        ),
        (
            [
                {"name": "Lint", "status": "in_progress", "conclusion": None},
                {"name": "Test", "status": "completed", "conclusion": "success"},
                {
                    "name": "Test scripts",
                    "status": "completed",
                    "conclusion": "success",
                },
            ],
            "Lint",
        ),
        (
            _successful_checks()
            + [{"name": "Lint", "status": "completed", "conclusion": "success"}],
            "Lint",
        ),
    ],
)
def test_missing_failing_pending_or_duplicate_required_checks_do_not_pass(checks, missing):
    client, _ = _gate_client(checks=checks)

    assert _result(client) == ReviewResult(
        False,
        7,
        f"required checks are not uniquely successful: {missing}",
    )


@pytest.mark.parametrize(
    "check",
    [
        {"name": "", "status": "completed", "conclusion": "success"},
        {"name": "Lint", "status": 7, "conclusion": "success"},
        {"name": "Lint", "status": "completed", "conclusion": 7},
        {"status": "completed", "conclusion": "success"},
    ],
)
def test_malformed_check_rows_fail_closed(check):
    client, _ = _gate_client(checks=[check])

    assert _result(client) == ReviewResult(False, None, "malformed GitHub check evidence")


@pytest.mark.parametrize(
    "required_checks",
    [
        set(),
        {""},
        {" Lint"},
        {"Lint\nOther"},
        {"Lint", 7},
    ],
)
def test_reviewed_commit_rejects_empty_or_malformed_required_checks(required_checks):
    client, _ = _gate_client()

    with pytest.raises(GitHubValidationError, match="required_checks"):
        _result(client, required_checks=required_checks)


def test_reviewed_commit_selects_a_later_qualifying_associated_pull():
    client, _ = _gate_client(
        pulls=[{"number": 7}, {"number": 8}],
        details={7: _open_pull(OTHER_SHA), 8: _merged_pull()},
        reviews={8: [_review()]},
    )

    assert _result(client).pr_number == 8


def test_changes_requested_on_any_qualifying_pull_blocks_the_sha():
    client, _ = _gate_client(
        pulls=[{"number": 7}, {"number": 8}],
        details={7: _open_pull(), 8: _merged_pull()},
        reviews={
            7: [_review("CHANGES_REQUESTED")],
            8: [_review(login="second-operator")],
        },
    )

    assert _result(
        client,
        allowed_reviewers={"operator", "second-operator"},
    ) == ReviewResult(
        False,
        None,
        "an authorized reviewer currently requests changes",
    )


def test_missing_script_at_exact_sha_is_rejected():
    client, _ = _gate_client(file_status=404)

    assert _result(client) == ReviewResult(
        False,
        7,
        "script does not exist at the requested SHA",
    )


def test_script_authorization_or_server_failure_is_not_treated_as_absence():
    client, _ = _gate_client(file_status=403)

    with pytest.raises(GitHubAPIError, match="GitHub API request failed"):
        _result(client)


@pytest.mark.parametrize(
    "pulls, details",
    [
        ([{"number": True}], {}),
        ([{"number": 0}], {}),
        ([{"number": "7"}], {}),
        ([{"number": 7}], {7: {"state": "open"}}),
        (
            [{"number": 7}],
            {7: {"head": {"sha": SHA}, "state": "closed", "merged_at": "not-a-time"}},
        ),
    ],
)
def test_malformed_pull_evidence_fails_closed(pulls, details):
    client, _ = _gate_client(pulls=pulls, details=details, reviews={})

    assert _result(client) == ReviewResult(False, None, "malformed GitHub pull evidence")


@pytest.mark.parametrize("sha", ["abc", "A" * 40, "g" * 40, "../" + ("a" * 37)])
def test_reviewed_commit_rejects_malformed_full_sha(sha):
    client, _ = _gate_client()

    with pytest.raises(GitHubValidationError, match="SHA"):
        client.reviewed_commit(
            sha,
            script_path=SCRIPT_PATH,
            allowed_reviewers={"operator"},
            required_checks=REQUIRED_CHECKS,
        )


def test_review_result_is_immutable():
    result = ReviewResult(False, None, "rejected")

    with pytest.raises(FrozenInstanceError):
        result.approved = True  # type: ignore[misc]


def test_reviewed_github_fixture_uses_task_4_call_contract(reviewed_github):
    result = reviewed_github.reviewed_commit(
        SHA,
        script_path=SCRIPT_PATH,
        allowed_reviewers={"operator"},
        required_checks=REQUIRED_CHECKS,
    )

    assert result == ReviewResult(True, 7, "approved")
