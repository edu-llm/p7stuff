from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from edullm.assignment import (
    ASSIGNMENT_MARKER,
    ActiveJob,
    AssignmentHistory,
    AssignmentState,
    AssignmentStateError,
    NoEligibleOperatorError,
    OperatorLoad,
    assign_ready_issues,
    build_assignment_comment,
    derive_operator_loads,
    parse_assignment_comment,
    process_assignment_timeouts,
    select_operator,
)
from edullm.github import GitHubAPIError, GitHubIssue, IssueComment
from edullm.models import Operator
from edullm.notifications import SlackNotificationError
from edullm.validation import build_status_comment

NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)
BODY = Path("src/test/edullm/fixtures/valid_issue.md").read_text(encoding="utf-8")


def _operator(
    github: str,
    *,
    slack: str = "U12345678",
    rotation: int = 0,
    enabled: bool = True,
) -> Operator:
    return Operator(github, slack, rotation, enabled)


def _state(
    valid_request,
    *,
    operator: str = "alice",
    assigned_at: datetime = NOW,
    reminder_at: datetime | None = None,
    history: tuple[AssignmentHistory, ...] = (),
    slack_status: str = "sent",
) -> AssignmentState:
    return AssignmentState(
        issue=valid_request.issue_number,
        request_digest=valid_request.digest,
        operator_github=operator,
        assigned_at=assigned_at,
        reminder_at=reminder_at,
        history=history,
        slack_status=slack_status,
    )


def test_selects_by_gpus_then_jobs_then_rotation_then_login():
    loads = [
        OperatorLoad("zeta", active_gpus=1, active_jobs=1, rotation=2),
        OperatorLoad("bob", active_gpus=1, active_jobs=1, rotation=1),
        OperatorLoad("alice", active_gpus=1, active_jobs=1, rotation=0),
        OperatorLoad("carol", active_gpus=1, active_jobs=2, rotation=0),
        OperatorLoad("dave", active_gpus=2, active_jobs=0, rotation=0),
    ]

    assert select_operator(loads, incoming_gpus=1, max_gpus=3).github == "alice"


def test_select_operator_excludes_timeout_and_capacity():
    loads = [
        OperatorLoad("alice", 0, 0, 0),
        OperatorLoad("bob", 1, 1, 1),
        OperatorLoad("carol", 2, 1, 2),
    ]

    selected = select_operator(
        loads,
        incoming_gpus=1,
        max_gpus=2,
        exclude={"alice"},
    )

    assert selected.github == "bob"


@pytest.mark.parametrize(
    "loads,incoming,max_gpus,exclude",
    [
        ([], 1, 2, set()),
        ([OperatorLoad("alice", -1, 0, 0)], 1, 2, set()),
        ([OperatorLoad("alice", 0, -1, 0)], 1, 2, set()),
        ([OperatorLoad("alice", 0, 0, -1)], 1, 2, set()),
        ([OperatorLoad("Alice", 0, 0, 0)], 1, 2, set()),
        (
            [OperatorLoad("alice", 0, 0, 0), OperatorLoad("alice", 0, 0, 1)],
            1,
            2,
            set(),
        ),
        ([OperatorLoad("alice", 0, 0, 0)], 0, 2, set()),
        ([OperatorLoad("alice", 0, 0, 0)], True, 2, set()),
        ([OperatorLoad("alice", 0, 0, 0)], 3, 2, set()),
        ([OperatorLoad("alice", 0, 0, 0)], 1, 0, set()),
        ([OperatorLoad("alice", 3, 1, 0)], 1, 2, set()),
        ([OperatorLoad("alice", 0, 0, 0)], 1, 2, {"unknown"}),
    ],
)
def test_select_operator_rejects_malformed_state(loads, incoming, max_gpus, exclude):
    with pytest.raises(ValueError):
        select_operator(
            loads,
            incoming_gpus=incoming,
            max_gpus=max_gpus,
            exclude=exclude,
        )


def test_select_operator_reports_no_capacity_without_forcing_assignment():
    with pytest.raises(NoEligibleOperatorError, match="no eligible operators"):
        select_operator(
            [OperatorLoad("alice", 2, 1, 0)],
            incoming_gpus=1,
            max_gpus=2,
        )


@pytest.mark.parametrize(
    "loads,incoming",
    [
        ([OperatorLoad("alice", 0, 0, 0)], 3),
        ([OperatorLoad("alice", 3, 1, 0)], 1),
    ],
)
def test_policy_capacity_violations_are_malformed_not_no_capacity(loads, incoming):
    with pytest.raises(ValueError) as raised:
        select_operator(loads, incoming_gpus=incoming, max_gpus=2)

    assert not isinstance(raised.value, NoEligibleOperatorError)


def test_derives_loads_for_enabled_operators_and_rejects_unknown_assignment():
    operators = (_operator("alice"), _operator("bob", slack="U22222222", rotation=1))
    jobs = (
        ActiveJob("alice", 2),
        ActiveJob("alice", 1),
        ActiveJob("bob", 1),
    )

    assert derive_operator_loads(operators, jobs) == (
        OperatorLoad("alice", 3, 2, 0),
        OperatorLoad("bob", 1, 1, 1),
    )

    with pytest.raises(ValueError, match="unknown or disabled operator"):
        derive_operator_loads(operators, jobs + (ActiveJob("carol", 1),))


@pytest.mark.parametrize(
    "jobs",
    [
        (ActiveJob("alice", 0),),
        (ActiveJob("alice", -1),),
        (ActiveJob("Alice", 1),),
    ],
)
def test_derive_loads_rejects_malformed_active_jobs(jobs):
    with pytest.raises(ValueError):
        derive_operator_loads((_operator("alice"),), jobs)


def test_assignment_state_round_trips_as_strict_canonical_json(valid_request):
    history = (
        AssignmentHistory(
            previous_operator="carol",
            reassigned_at=NOW,
        ),
    )
    state = _state(
        valid_request,
        reminder_at=NOW + timedelta(minutes=15),
        history=history,
    )

    comment = build_assignment_comment(state)

    assert comment.startswith(ASSIGNMENT_MARKER + "\n")
    encoded = comment.split("\n", 1)[1]
    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"))
    assert parse_assignment_comment(comment) == state
    with pytest.raises(FrozenInstanceError):
        state.operator_github = "bob"  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace(ASSIGNMENT_MARKER, "<!-- wrong -->"),
        lambda text: text + "\n" + ASSIGNMENT_MARKER,
        lambda text: text.replace(',"slack_status":"sent"', ""),
        lambda text: text.replace('"alice"', '"Alice"'),
        lambda text: text.replace('"issue":42', '"issue":0'),
        lambda text: text.replace('"request_digest":"', '"request_digest":"g'),
        lambda text: text.replace("2026-07-23T12:00:00Z", "2026-07-23T12:00:00.1Z"),
        lambda text: text.replace('"history":[]', '"history":[1]'),
        lambda text: text.replace('{"assigned_at"', '{ "assigned_at"'),
    ],
)
def test_assignment_state_rejects_tampered_noncanonical_or_malformed_payload(
    valid_request,
    mutator,
):
    comment = mutator(build_assignment_comment(_state(valid_request)))

    with pytest.raises(AssignmentStateError):
        parse_assignment_comment(comment)


def test_assignment_state_rejects_unbounded_history(valid_request):
    history = tuple(
        AssignmentHistory("alice", NOW + timedelta(seconds=index)) for index in range(33)
    )

    with pytest.raises(AssignmentStateError, match="history"):
        build_assignment_comment(_state(valid_request, history=history))


@pytest.mark.parametrize(
    "history",
    [
        (AssignmentHistory("alice", NOW),),
        (AssignmentHistory("carol", NOW - timedelta(seconds=1)),),
    ],
)
def test_assignment_state_rejects_inconsistent_current_history(valid_request, history):
    with pytest.raises(AssignmentStateError, match="history"):
        build_assignment_comment(_state(valid_request, history=history))


class RecordingNotifier:
    def __init__(self, error: SlackNotificationError | None = None):
        self.error = error
        self.calls: list[tuple[int, str, str, str]] = []

    def assignment(self, *, issue, title, operator_slack_id, kind):
        self.calls.append((issue, title, operator_slack_id, kind))
        if self.error is not None:
            raise self.error


class QueueGitHub:
    def __init__(
        self,
        valid_request,
        *,
        labels=("edullm-job", "research", "status:ready"),
        assignees=(),
        assignment_state=None,
    ):
        self.request = valid_request
        self.issue = GitHubIssue(
            number=valid_request.issue_number,
            body=BODY,
            requester=valid_request.requester,
            labels=tuple(labels),
            title='Untrusted <@U99999999> & <script> "title"',
            assignees=tuple(assignees),
        )
        status = build_status_comment(valid_request, validated_at=NOW - timedelta(minutes=5))
        self.comments = [
            IssueComment(1, status, "github-actions[bot]", True),
        ]
        if assignment_state is not None:
            self.comments.append(
                IssueComment(
                    2,
                    build_assignment_comment(assignment_state),
                    "github-actions[bot]",
                    True,
                )
            )
        self.next_comment_id = 10
        self.events: list[tuple] = []
        self.other_issues: list[tuple[GitHubIssue, list[IssueComment]]] = []

    def add_active_assignment(self, valid_request, *, operator, gpu_count, status="assigned"):
        request = replace(
            valid_request, issue_number=100 + len(self.other_issues), gpu_count=gpu_count
        )
        body = BODY.replace("### GPU count\n1", f"### GPU count\n{gpu_count}", 1)
        issue = GitHubIssue(
            request.issue_number,
            body,
            request.requester,
            ("edullm-job", f"status:{status}"),
            title="active",
            assignees=(operator,),
        )
        state = AssignmentState(
            issue=request.issue_number,
            request_digest=request.digest,
            operator_github=operator,
            assigned_at=NOW - timedelta(minutes=5),
            reminder_at=None,
            history=(),
            slack_status="sent",
        )
        comments = [
            IssueComment(
                1000 + request.issue_number,
                build_status_comment(request, validated_at=NOW - timedelta(hours=1)),
                "github-actions[bot]",
                True,
            ),
            IssueComment(
                2000 + request.issue_number,
                build_assignment_comment(state),
                "github-actions[bot]",
                True,
            ),
        ]
        self.other_issues.append((issue, comments))

    def list_active_queue_issues(self):
        self.events.append(("list-active",))
        return (self.issue,) + tuple(issue for issue, _ in self.other_issues)

    def fetch_issue(self, issue_number):
        self.events.append(("fetch", issue_number))
        if issue_number == self.issue.number:
            return self.issue
        return next(issue for issue, _ in self.other_issues if issue.number == issue_number)

    def list_issue_comments(self, issue_number):
        self.events.append(("comments", issue_number))
        if issue_number == self.issue.number:
            return tuple(self.comments)
        return tuple(
            comments for issue, comments in self.other_issues if issue.number == issue_number
        )[0]

    def create_issue_comment(self, issue_number, body):
        self.events.append(("create-comment", issue_number, body))
        comment = IssueComment(
            self.next_comment_id,
            body,
            "github-actions[bot]",
            True,
        )
        self.next_comment_id += 1
        self.comments.append(comment)
        return comment

    def update_issue_comment(self, comment_id, body):
        self.events.append(("update-comment", comment_id, body))
        updated = IssueComment(comment_id, body, "github-actions[bot]", True)
        self.comments = [
            updated if comment.id == comment_id else comment for comment in self.comments
        ]
        return updated

    def add_issue_status_label(self, issue_number, label):
        self.events.append(("add-label", issue_number, label))
        self.issue = replace(
            self.issue,
            labels=tuple(sorted(set(self.issue.labels) | {label})),
        )
        return self.issue.labels

    def remove_issue_status_label(self, issue_number, label):
        self.events.append(("remove-label", issue_number, label))
        present = label in self.issue.labels
        self.issue = replace(
            self.issue,
            labels=tuple(current for current in self.issue.labels if current != label),
        )
        return present

    def add_issue_assignee(self, issue_number, login):
        self.events.append(("add-assignee", issue_number, login))
        self.issue = replace(
            self.issue,
            assignees=tuple(sorted(set(self.issue.assignees) | {login})),
        )
        return self.issue.assignees

    def remove_issue_assignee(self, issue_number, login):
        self.events.append(("remove-assignee", issue_number, login))
        present = login in self.issue.assignees
        self.issue = replace(
            self.issue,
            assignees=tuple(current for current in self.issue.assignees if current != login),
        )
        return present


def _assignment_comments(github):
    return [comment for comment in github.comments if ASSIGNMENT_MARKER in comment.body]


def test_assignment_revalidates_current_ready_request_and_scores_fresh_active_loads(
    valid_request,
    policy,
):
    github = QueueGitHub(valid_request)
    github.add_active_assignment(valid_request, operator="alice", gpu_count=1)
    notifier = RecordingNotifier()
    operators = (
        _operator("alice", rotation=0),
        _operator("bob", slack="U22222222", rotation=1),
    )

    results = assign_ready_issues(
        github=github,
        operators=operators,
        policy=policy,
        now=NOW,
        notifier=notifier,
    )

    assert results[0].action == "assigned"
    assert results[0].operator == "bob"
    assert github.events.count(("list-active",)) >= 2
    assert "bob" in github.issue.assignees
    assert "status:assigned" in github.issue.labels
    assert "status:ready" not in github.issue.labels
    assert "research" in github.issue.labels
    assert any("@bob" in comment.body for comment in github.comments)
    assert notifier.calls == [
        (
            42,
            github.issue.title,
            "U22222222",
            "assignment",
        )
    ]
    state = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert state.operator_github == "bob"
    assert state.slack_status == "sent"


def test_empty_enabled_operator_roster_is_a_closed_noop(valid_request, policy):
    github = QueueGitHub(valid_request)

    results = assign_ready_issues(
        github=github,
        operators=(_operator("alice", enabled=False),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )

    assert results == ()
    assert github.events == []
    assert github.issue.labels[-1] == "status:ready"


def test_runtime_rejects_malformed_protected_operator_mapping(valid_request, policy):
    github = QueueGitHub(valid_request)

    with pytest.raises(ValueError, match="operator roster"):
        assign_ready_issues(
            github=github,
            operators=(_operator("alice", slack="not-a-slack-id"),),
            policy=policy,
            now=NOW,
            notifier=RecordingNotifier(),
        )

    assert github.events == []


def test_no_capacity_leaves_issue_ready_and_unassigned(valid_request, policy):
    github = QueueGitHub(valid_request)
    github.add_active_assignment(valid_request, operator="alice", gpu_count=2)

    results = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )

    assert results[0].action == "no_capacity"
    assert "status:ready" in github.issue.labels
    assert github.issue.assignees == ()
    assert _assignment_comments(github) == []


@pytest.mark.parametrize(
    "comments",
    [
        [
            IssueComment(
                50,
                f"{ASSIGNMENT_MARKER}\n{{}}",
                "student",
                False,
            )
        ],
        [
            IssueComment(50, f"{ASSIGNMENT_MARKER}\n{{}}", "github-actions[bot]", True),
            IssueComment(51, f"{ASSIGNMENT_MARKER}\n{{}}", "github-actions[bot]", True),
        ],
    ],
)
def test_assignment_marker_spoof_or_duplicate_fails_closed(
    valid_request,
    policy,
    comments,
):
    github = QueueGitHub(valid_request)
    github.comments.extend(comments)

    results = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )

    assert results[0].action == "error"
    assert results[0].operational_error is True
    assert "status:ready" in github.issue.labels
    assert github.issue.assignees == ()


@pytest.mark.parametrize("mode", ["human", "duplicate", "missing"])
def test_downstream_ready_revalidation_requires_one_owned_status_comment(
    valid_request,
    policy,
    mode,
):
    github = QueueGitHub(valid_request)
    if mode == "human":
        github.comments[0] = replace(
            github.comments[0],
            author="student",
            author_is_bot=False,
        )
    elif mode == "duplicate":
        github.comments.append(replace(github.comments[0], id=99))
    else:
        github.comments.clear()

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert github.issue.assignees == ()
    assert "status:ready" in github.issue.labels


def test_downstream_revalidation_rejects_unrelated_bot_status_owner(
    valid_request,
    policy,
):
    github = QueueGitHub(valid_request)
    github.comments[0] = replace(
        github.comments[0],
        author="other-automation[bot]",
        author_is_bot=True,
    )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert github.issue.assignees == ()
    assert "status:ready" in github.issue.labels


@pytest.mark.parametrize(
    "author,author_is_bot",
    [
        ("student", False),
        ("other-automation[bot]", True),
    ],
)
def test_partial_assignment_rejects_wrong_machine_comment_owner(
    valid_request,
    policy,
    author,
    author_is_bot,
):
    github = QueueGitHub(
        valid_request,
        assignment_state=_state(valid_request, slack_status="pending"),
    )
    github.comments[1] = replace(
        github.comments[1],
        author=author,
        author_is_bot=author_is_bot,
    )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert github.issue.assignees == ()
    assert "status:ready" in github.issue.labels


def test_partial_assignment_rejects_unrelated_bot_notice_owner(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        assignees=("alice",),
        assignment_state=_state(valid_request, slack_status="pending"),
    )
    github.comments.append(
        IssueComment(
            9,
            "@alice eduLLM job assignment is ready for submission " "(assignment generation 0).",
            "other-automation[bot]",
            True,
        )
    )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert "status:ready" in github.issue.labels


def test_exact_automation_comment_owner_is_accepted(valid_request, policy):
    github = QueueGitHub(valid_request)

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "assigned"
    machine_comments = [
        comment
        for comment in github.comments
        if "<!-- edullm-status:v1 -->" in comment.body or ASSIGNMENT_MARKER in comment.body
    ]
    assert machine_comments
    assert {comment.author for comment in machine_comments} == {"github-actions[bot]"}


def test_malformed_other_ready_issue_blocks_repository_load_snapshot(
    valid_request,
    policy,
):
    github = QueueGitHub(valid_request)
    other_request = replace(valid_request, issue_number=100)
    other_issue = GitHubIssue(
        100,
        BODY,
        valid_request.requester,
        ("edullm-job", "status:ready"),
        title="other ready",
    )
    github.other_issues.append(
        (
            other_issue,
            [
                IssueComment(
                    1001,
                    build_status_comment(
                        other_request,
                        validated_at=NOW - timedelta(minutes=5),
                    ),
                    "github-actions[bot]",
                    True,
                ),
                IssueComment(
                    1002,
                    f"{ASSIGNMENT_MARKER}\n{{}}",
                    "student",
                    False,
                ),
            ],
        )
    )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert _assignment_comments(github) == []
    assert github.issue.assignees == ()


def test_edited_issue_or_stale_validated_digest_fails_closed(valid_request, policy):
    github = QueueGitHub(valid_request)
    github.issue = replace(
        github.issue,
        body=github.issue.body.replace("Skill-DAG smoke", "edited after validation", 1),
    )

    results = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )

    assert results[0].action == "error"
    assert github.issue.assignees == ()
    assert "status:ready" in github.issue.labels


def test_issue_edit_after_scoring_is_refetched_before_assignment_state_write(
    valid_request,
    policy,
):
    class EditAfterLoadFetch(QueueGitHub):
        fetches = 0

        def fetch_issue(self, issue_number):
            issue = super().fetch_issue(issue_number)
            self.fetches += 1
            if self.fetches == 2:
                self.issue = replace(
                    self.issue,
                    body=self.issue.body.replace("Skill-DAG smoke", "edited during scoring", 1),
                )
            return issue

    github = EditAfterLoadFetch(valid_request)

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert _assignment_comments(github) == []
    assert github.issue.assignees == ()
    assert "status:ready" in github.issue.labels


def test_future_partial_assignment_state_does_not_consume_ready(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        assignment_state=_state(
            valid_request,
            assigned_at=NOW + timedelta(minutes=1),
            slack_status="pending",
        ),
    )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert github.issue.assignees == ()
    assert "status:ready" in github.issue.labels
    assert not any(event[0] == "create-comment" for event in github.events)


@pytest.mark.parametrize(
    "labels",
    [
        ("edullm-job", "status:ready", "status:assigned"),
        ("edullm-job", "status:ready", "status:unknown"),
    ],
)
def test_managed_status_labels_must_be_mutually_exclusive(valid_request, policy, labels):
    github = QueueGitHub(
        valid_request,
        labels=labels,
    )

    results = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )

    assert results[0].action == "error"
    assert github.issue.assignees == ()


def test_assignment_retry_is_idempotent_after_authoritative_github_state(
    valid_request,
    policy,
):
    github = QueueGitHub(valid_request)
    notifier = RecordingNotifier()
    operators = (_operator("alice"),)

    first = assign_ready_issues(
        github=github,
        operators=operators,
        policy=policy,
        now=NOW,
        notifier=notifier,
    )
    second = assign_ready_issues(
        github=github,
        operators=operators,
        policy=policy,
        now=NOW + timedelta(minutes=1),
        notifier=notifier,
    )

    assert first[0].action == "assigned"
    assert second == ()
    assert len(_assignment_comments(github)) == 1
    assert len([comment for comment in github.comments if "@alice" in comment.body]) == 1
    assert len(notifier.calls) == 1


def test_resumed_pending_assignment_does_not_blindly_repeat_slack_send(
    valid_request,
    policy,
):
    github = QueueGitHub(
        valid_request,
        assignment_state=_state(
            valid_request,
            slack_status="pending",
        ),
    )
    notifier = RecordingNotifier()

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "assigned_notification_pending"
    assert notifier.calls == []
    assert "status:assigned" in github.issue.labels
    assert github.issue.assignees == ("alice",)
    assert parse_assignment_comment(_assignment_comments(github)[0].body).slack_status == "pending"


@pytest.mark.parametrize("operation", ["state-comment", "assignee"])
def test_ambiguous_assignment_writes_reconcile_before_claiming_success(
    valid_request,
    policy,
    operation,
):
    class CommitThenLoseResponse(QueueGitHub):
        failed = False

        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            if operation == "state-comment" and ASSIGNMENT_MARKER in body and not self.failed:
                self.failed = True
                raise GitHubAPIError("response body with secret")
            return persisted

        def add_issue_assignee(self, issue_number, login):
            assignees = super().add_issue_assignee(issue_number, login)
            if operation == "assignee" and not self.failed:
                self.failed = True
                raise GitHubAPIError("response body with secret")
            return assignees

    github = CommitThenLoseResponse(valid_request)

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "assigned"
    assert "status:assigned" in github.issue.labels
    assert github.issue.assignees == ("alice",)
    assert len(_assignment_comments(github)) == 1


def test_interrupted_ready_to_assigned_transition_resumes_on_restart(
    valid_request,
    policy,
):
    class InterruptReadyRemovalOnce(QueueGitHub):
        interrupted = False

        def remove_issue_status_label(self, issue_number, label):
            if label == "status:ready" and not self.interrupted:
                self.interrupted = True
                self.events.append(("remove-label-interrupted", issue_number, label))
                raise GitHubAPIError("interrupted label removal")
            return super().remove_issue_status_label(issue_number, label)

    github = InterruptReadyRemovalOnce(valid_request)
    notifier = RecordingNotifier()
    kwargs = {
        "github": github,
        "operators": (_operator("alice"),),
        "policy": policy,
        "notifier": notifier,
    }

    first = assign_ready_issues(now=NOW, **kwargs)[0]

    assert first.action == "error"
    assert set(github.issue.labels) >= {"status:ready", "status:assigned", "research"}
    assert github.issue.assignees == ("alice",)
    assert parse_assignment_comment(_assignment_comments(github)[0].body).slack_status == "pending"
    assert notifier.calls == []

    second = assign_ready_issues(now=NOW + timedelta(minutes=1), **kwargs)[0]

    assert second.action == "assigned_notification_pending"
    assert "status:assigned" in github.issue.labels
    assert "status:ready" not in github.issue.labels
    assert "research" in github.issue.labels
    assert github.issue.assignees == ("alice",)
    assert len(_assignment_comments(github)) == 1
    assert notifier.calls == []


def test_ambiguous_ready_label_removal_reconciles_committed_write(
    valid_request,
    policy,
):
    class LoseReadyRemovalResponseOnce(QueueGitHub):
        lost_response = False

        def remove_issue_status_label(self, issue_number, label):
            result = super().remove_issue_status_label(issue_number, label)
            if label == "status:ready" and not self.lost_response:
                self.lost_response = True
                raise GitHubAPIError("lost label removal response")
            return result

    github = LoseReadyRemovalResponseOnce(valid_request)

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "assigned"
    assert "status:assigned" in github.issue.labels
    assert "status:ready" not in github.issue.labels
    assert "research" in github.issue.labels


@pytest.mark.parametrize("mode", ["missing-assignment", "stale-digest", "sent-state"])
def test_both_label_state_is_rejected_without_proven_in_progress_assignment(
    valid_request,
    policy,
    mode,
):
    state = None
    if mode != "missing-assignment":
        state = _state(
            valid_request,
            slack_status="sent" if mode == "sent-state" else "pending",
        )
        if mode == "stale-digest":
            state = replace(state, request_digest="0" * 64)
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "research", "status:ready", "status:assigned"),
        assignees=("alice",),
        assignment_state=state,
    )
    if state is not None:
        github.comments.append(
            IssueComment(
                9,
                "@alice eduLLM job assignment is ready for submission "
                "(assignment generation 0).",
                "github-actions[bot]",
                True,
            )
        )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert {"status:ready", "status:assigned"} <= set(github.issue.labels)
    assert "research" in github.issue.labels


def test_both_label_state_requires_owned_assignment_notice(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "research", "status:ready", "status:assigned"),
        assignees=("alice",),
        assignment_state=_state(valid_request, slack_status="pending"),
    )

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert {"status:ready", "status:assigned"} <= set(github.issue.labels)


def test_assignment_aborts_if_state_changes_after_assignee_write(valid_request, policy):
    class ConcurrentAssignment(QueueGitHub):
        changed = False

        def add_issue_assignee(self, issue_number, login):
            assignees = super().add_issue_assignee(issue_number, login)
            if not self.changed:
                self.changed = True
                concurrent = _state(
                    valid_request,
                    operator="bob",
                    assigned_at=NOW,
                    slack_status="pending",
                )
                replacement = IssueComment(
                    10,
                    build_assignment_comment(concurrent),
                    "github-actions[bot]",
                    True,
                )
                self.comments = [
                    replacement if ASSIGNMENT_MARKER in comment.body else comment
                    for comment in self.comments
                ]
                self.issue = replace(self.issue, assignees=("alice", "bob"))
            return assignees

    github = ConcurrentAssignment(valid_request)

    result = assign_ready_issues(
        github=github,
        operators=(
            _operator("alice"),
            _operator("bob", slack="U22222222", rotation=1),
        ),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    current = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert current.operator_github == "bob"
    assert "status:ready" in github.issue.labels


@pytest.mark.parametrize(
    "error,expected_status",
    [
        (SlackNotificationError("Slack notification failed", ambiguous=False), "failed"),
        (
            SlackNotificationError("Slack notification outcome is unknown", ambiguous=True),
            "ambiguous",
        ),
    ],
)
def test_slack_failure_keeps_github_assignment_authoritative_and_sanitized(
    valid_request,
    policy,
    error,
    expected_status,
):
    secret = "https://hooks.slack.com/services/T/SECRET/VALUE"
    github = QueueGitHub(valid_request)
    notifier = RecordingNotifier(error)

    result = assign_ready_issues(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "assigned_notification_pending"
    assert result.operator == "alice"
    assert "status:assigned" in github.issue.labels
    assert "alice" in github.issue.assignees
    state = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert state.slack_status == expected_status
    assert secret not in repr(result)
    assert secret not in _assignment_comments(github)[0].body


def test_definitive_slack_failure_is_retryable_without_reassignment(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW - timedelta(minutes=5),
            slack_status="failed",
        ),
    )
    notifier = RecordingNotifier()

    result = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "notification_retried"
    assert github.issue.assignees == ("alice",)
    assert parse_assignment_comment(_assignment_comments(github)[0].body).slack_status == "sent"
    assert notifier.calls[-1][-1] == "assignment"


def test_ambiguous_slack_send_is_not_blindly_retried(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW - timedelta(minutes=5),
            slack_status="ambiguous",
        ),
    )
    notifier = RecordingNotifier()

    result = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "unchanged"
    assert notifier.calls == []
    assert (
        parse_assignment_comment(_assignment_comments(github)[0].body).slack_status == "ambiguous"
    )


def test_future_assignment_timestamp_fails_closed_without_timeout_write(
    valid_request,
    policy,
):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW + timedelta(minutes=1),
        ),
    )

    result = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    assert len(_assignment_comments(github)) == 1
    assert github.issue.assignees == ("alice",)
    assert not any(event[0] == "create-comment" for event in github.events)


def test_timeout_scan_repairs_partial_reassignment_without_double_assignment(
    valid_request,
    policy,
):
    assigned_at = NOW - timedelta(minutes=5)
    state = _state(
        valid_request,
        operator="bob",
        assigned_at=assigned_at,
        history=(AssignmentHistory("alice", assigned_at),),
    )
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice", "bob", "reviewer"),
        assignment_state=state,
    )

    result = process_assignment_timeouts(
        github=github,
        operators=(
            _operator("alice"),
            _operator("bob", slack="U22222222", rotation=1),
        ),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "unchanged"
    assert set(github.issue.assignees) == {"bob", "reviewer"}
    assert len([comment for comment in github.comments if "@bob" in comment.body]) == 1
    assert parse_assignment_comment(_assignment_comments(github)[0].body) == state


def test_reminder_after_fifteen_minutes_is_recorded_exactly_once(valid_request, policy):
    state = _state(valid_request, assigned_at=NOW - timedelta(minutes=15))
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "research", "status:assigned"),
        assignees=("alice", "reviewer"),
        assignment_state=state,
    )
    notifier = RecordingNotifier()

    first = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )
    second = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW + timedelta(minutes=1),
        notifier=notifier,
    )

    assert first[0].action == "reminded"
    assert second[0].action == "unchanged"
    assert (
        len(
            [comment for comment in github.comments if "15-minute reminder" in comment.body.lower()]
        )
        == 1
    )
    assert parse_assignment_comment(_assignment_comments(github)[0].body).reminder_at == NOW
    assert "reviewer" in github.issue.assignees


def test_submitted_or_running_issue_is_never_reminded_or_reassigned(valid_request, policy):
    for status in ("submitted", "running"):
        github = QueueGitHub(
            valid_request,
            labels=("edullm-job", f"status:{status}"),
            assignees=("alice",),
            assignment_state=_state(
                valid_request,
                assigned_at=NOW - timedelta(hours=1),
            ),
        )

        results = process_assignment_timeouts(
            github=github,
            operators=(
                _operator("alice"),
                _operator("bob", slack="U22222222", rotation=1),
            ),
            policy=policy,
            now=NOW,
            notifier=RecordingNotifier(),
        )

        assert results == ()
        assert github.issue.assignees == ("alice",)
        assert f"status:{status}" in github.issue.labels


def test_reassigns_after_thirty_minutes_excluding_timed_out_operator(
    valid_request,
    policy,
):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "research", "status:assigned"),
        assignees=("alice", "reviewer"),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW - timedelta(minutes=30),
            reminder_at=NOW - timedelta(minutes=15),
        ),
    )
    notifier = RecordingNotifier()

    result = process_assignment_timeouts(
        github=github,
        operators=(
            _operator("alice", rotation=0),
            _operator("bob", slack="U22222222", rotation=1),
        ),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "reassigned"
    assert result.operator == "bob"
    assert "alice" not in github.issue.assignees
    assert set(github.issue.assignees) == {"bob", "reviewer"}
    assert "status:assigned" in github.issue.labels
    state = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert state.operator_github == "bob"
    assert state.assigned_at == NOW
    assert state.reminder_at is None
    assert state.history == (AssignmentHistory("alice", NOW),)
    assert notifier.calls[-1][-1] == "reassignment"


@pytest.mark.parametrize("operation", ["add-new", "remove-old"])
def test_reassignment_detects_and_repairs_unrelated_assignee_loss(
    valid_request,
    policy,
    operation,
):
    class DropsUnrelatedAssignee(QueueGitHub):
        def add_issue_assignee(self, issue_number, login):
            result = super().add_issue_assignee(issue_number, login)
            if operation == "add-new" and login == "bob":
                self.issue = replace(
                    self.issue,
                    assignees=tuple(
                        assignee for assignee in self.issue.assignees if assignee != "reviewer"
                    ),
                )
            return result

        def remove_issue_assignee(self, issue_number, login):
            result = super().remove_issue_assignee(issue_number, login)
            if operation == "remove-old" and login == "alice":
                self.issue = replace(
                    self.issue,
                    assignees=tuple(
                        assignee for assignee in self.issue.assignees if assignee != "reviewer"
                    ),
                )
            return result

    github = DropsUnrelatedAssignee(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice", "reviewer"),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW - timedelta(minutes=30),
            reminder_at=NOW - timedelta(minutes=15),
        ),
    )
    notifier = RecordingNotifier()

    result = process_assignment_timeouts(
        github=github,
        operators=(
            _operator("alice"),
            _operator("bob", slack="U22222222", rotation=1),
        ),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "error"
    assert "reviewer" in github.issue.assignees
    assert ("add-assignee", 42, "reviewer") in github.events
    assert notifier.calls == []
    state = parse_assignment_comment(_assignment_comments(github)[0].body)
    if operation == "add-new":
        assert state.operator_github == "alice"
        assert "alice" in github.issue.assignees
    else:
        assert state.operator_github == "bob"
        assert "alice" not in github.issue.assignees
    assert "bob" in github.issue.assignees


def test_reassignment_deadline_takes_priority_over_old_slack_retry(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW - timedelta(minutes=30),
            reminder_at=NOW - timedelta(minutes=15),
            slack_status="failed",
        ),
    )
    notifier = RecordingNotifier()

    result = process_assignment_timeouts(
        github=github,
        operators=(
            _operator("alice"),
            _operator("bob", slack="U22222222", rotation=1),
        ),
        policy=policy,
        now=NOW,
        notifier=notifier,
    )[0]

    assert result.action == "reassigned"
    assert result.operator == "bob"
    assert notifier.calls == [(42, github.issue.title, "U22222222", "reassignment")]


def test_reassignment_aborts_if_state_changes_after_assignee_write(valid_request, policy):
    original = _state(
        valid_request,
        assigned_at=NOW - timedelta(minutes=30),
        reminder_at=NOW - timedelta(minutes=15),
    )

    class ConcurrentReassignment(QueueGitHub):
        changed = False

        def add_issue_assignee(self, issue_number, login):
            assignees = super().add_issue_assignee(issue_number, login)
            if login == "bob" and not self.changed:
                self.changed = True
                reassigned = _state(
                    valid_request,
                    operator="carol",
                    assigned_at=NOW,
                    history=(AssignmentHistory("alice", NOW),),
                )
                replacement = IssueComment(
                    2,
                    build_assignment_comment(reassigned),
                    "github-actions[bot]",
                    True,
                )
                self.comments = [
                    replacement if comment.id == 2 else comment for comment in self.comments
                ]
                self.issue = replace(self.issue, assignees=("alice", "bob", "carol"))
            return assignees

    github = ConcurrentReassignment(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=original,
    )

    result = process_assignment_timeouts(
        github=github,
        operators=(
            _operator("alice"),
            _operator("bob", slack="U22222222", rotation=1),
            _operator("carol", slack="U33333333", rotation=2),
        ),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    current = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert current.operator_github == "carol"
    assert github.issue.assignees == ("alice", "bob", "carol")


def test_reassignment_without_alternate_capacity_preserves_single_assignment(
    valid_request,
    policy,
):
    original = _state(
        valid_request,
        assigned_at=NOW - timedelta(minutes=30),
        reminder_at=NOW - timedelta(minutes=15),
    )
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=original,
    )

    result = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "no_capacity"
    assert github.issue.assignees == ("alice",)
    assert parse_assignment_comment(_assignment_comments(github)[0].body) == original


def test_reminder_revalidates_digest_before_write(valid_request, policy):
    github = QueueGitHub(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=_state(
            valid_request,
            assigned_at=NOW - timedelta(minutes=15),
        ),
    )
    github.issue = replace(
        github.issue,
        body=github.issue.body.replace("Skill-DAG smoke", "edited", 1),
    )

    result = process_assignment_timeouts(
        github=github,
        operators=(_operator("alice"),),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    state = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert state.reminder_at is None


def test_reminder_aborts_if_assignment_changes_after_notice(valid_request, policy):
    original = _state(
        valid_request,
        assigned_at=NOW - timedelta(minutes=15),
    )

    class ReassignDuringReminderNotice(QueueGitHub):
        changed = False

        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            if "15-minute reminder" in body and not self.changed:
                self.changed = True
                reassigned = _state(
                    valid_request,
                    operator="bob",
                    assigned_at=NOW,
                    history=(AssignmentHistory("alice", NOW),),
                )
                replacement = IssueComment(
                    2,
                    build_assignment_comment(reassigned),
                    "github-actions[bot]",
                    True,
                )
                self.comments = [
                    replacement if comment.id == 2 else comment for comment in self.comments
                ]
                self.issue = replace(self.issue, assignees=("bob",))
            return persisted

    github = ReassignDuringReminderNotice(
        valid_request,
        labels=("edullm-job", "status:assigned"),
        assignees=("alice",),
        assignment_state=original,
    )

    result = process_assignment_timeouts(
        github=github,
        operators=(
            _operator("alice"),
            _operator("bob", slack="U22222222", rotation=1),
        ),
        policy=policy,
        now=NOW,
        notifier=RecordingNotifier(),
    )[0]

    assert result.action == "error"
    current = parse_assignment_comment(_assignment_comments(github)[0].body)
    assert current.operator_github == "bob"
    assert current.reminder_at is None
