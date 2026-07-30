from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

import edullm.jobs as jobs_module
from edullm.assignment import (
    ASSIGNMENT_MARKER,
    AssignmentState,
    build_assignment_comment,
)
from edullm.github import (
    CommitResult,
    GitHubError,
    GitHubIssue,
    IssueComment,
    ReviewResult,
)
from edullm.jobs import (
    JOB_MARKER,
    GateConfiguration,
    JobAttempt,
    JobOperationError,
    LifecycleState,
    NotificationRecord,
    SlurmJob,
    build_job_comment,
    build_resolved_request,
    deliver_terminal_notifications,
    full_submission_gate,
    jobs,
    parse_job_comment,
    parse_sacct,
    parse_squeue_json,
    read_authorized_log,
    reconcile_lifecycle,
    run_assigned,
    stop_job,
)
from edullm.models import Operator
from edullm.policy import Policy
from edullm.slurm import SubmissionReceipt
from edullm.validation import build_status_comment

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)
BODY = Path("src/test/edullm/fixtures/valid_issue.md").read_text(encoding="utf-8")


def _policy() -> Policy:
    return Policy(
        "eduLLM",
        ("test", "pretraining"),
        required_checks=("Lint",),
        entrypoints={
            "hypothesis-smoke": {
                "allowed_data_kinds": ("skill-dag", "curriculum"),
                "launcher": "python",
                "model_identity": "olmo2-190m",
                "positionals": 3,
                "script": "src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
                "wandb_callback": True,
                "allowed_positionals": {
                    0: ("dry_run", "train_single", "train"),
                    2: ("local",),
                },
                "fixed_options": {"trainer.callbacks.wandb.enabled": True},
                "allowed_options": {
                    "seed": {
                        "type": "integer",
                        "min": 0,
                        "max": 2147483647,
                        "required": True,
                        "request_field": "seed",
                    }
                },
            }
        },
    )


def _operator(github: str = "operator", rotation_order: int = 0) -> Operator:
    return Operator(github, "U12345678", rotation_order, True)


def _three_enabled_operators() -> tuple[Operator, ...]:
    return (
        _operator("philote-dev", 0),
        _operator("meric233", 1),
        _operator("alsy7009", 2),
    )


def _issue(valid_request, *, operator: str = "operator", **changes) -> GitHubIssue:
    return replace(
        GitHubIssue(
            number=42,
            body=BODY,
            requester="student",
            labels=("edullm-job", "status:assigned", "research"),
            title="Untrusted <script>alert(1)</script>",
            assignees=(operator, "faculty"),
        ),
        **changes,
    )


def _assignment(valid_request, *, operator: str = "operator") -> AssignmentState:
    return AssignmentState(
        issue=42,
        request_digest=valid_request.digest,
        operator_github=operator,
        assigned_at=NOW,
        reminder_at=None,
        history=(),
        slack_status="sent",
    )


def _comments(
    valid_request,
    lifecycle: LifecycleState | None = None,
    *,
    operator: str = "operator",
):
    comments = [
        IssueComment(
            1,
            build_status_comment(valid_request, validated_at=NOW),
            "github-actions[bot]",
            True,
        ),
        IssueComment(
            2,
            build_assignment_comment(_assignment(valid_request, operator=operator)),
            "github-actions[bot]",
            True,
        ),
    ]
    if lifecycle is not None:
        comments.append(IssueComment(3, build_job_comment(lifecycle), lifecycle.operator, False))
    return comments


class StatefulGitHub:
    def __init__(
        self,
        valid_request,
        *,
        lifecycle=None,
        write_author="operator",
        write_author_is_bot=False,
        stored_write_author=None,
        stored_write_author_is_bot=None,
        fail_add_once=False,
        fail_remove_once=False,
        operator="operator",
        commit_evidence=CommitResult(True, "commit and script exist"),
    ):
        self.valid_request = valid_request
        self.issue = _issue(valid_request, operator=operator)
        if lifecycle is not None:
            self.issue = replace(
                self.issue,
                labels=("edullm-job", f"status:{lifecycle.current_state}", "research"),
            )
        self.comments = _comments(valid_request, lifecycle, operator=operator)
        self.review = ReviewResult(True, 7, "approved")
        self.commit_evidence = commit_evidence
        self.commit_evidence_calls = []
        self.review_calls = []
        self.events = []
        self.next_comment_id = 10
        self.write_author = write_author
        self.write_author_is_bot = write_author_is_bot
        self.stored_write_author = stored_write_author
        self.stored_write_author_is_bot = stored_write_author_is_bot
        self.fail_add_once = fail_add_once
        self.fail_remove_once = fail_remove_once

    def list_active_queue_issues(self):
        self.events.append(("list",))
        return (self.issue,)

    def fetch_issue(self, number):
        self.events.append(("fetch", number))
        return self.issue

    def list_issue_comments(self, number):
        self.events.append(("comments", number))
        return tuple(self.comments)

    def reviewed_commit(self, sha, **kwargs):
        self.review_calls.append((sha, kwargs))
        self.events.append(("review", sha))
        return self.review

    def executable_commit(self, sha, *, script_path):
        self.commit_evidence_calls.append((sha, script_path))
        self.events.append(("commit-evidence", sha, script_path))
        if isinstance(self.commit_evidence, Exception):
            raise self.commit_evidence
        return self.commit_evidence

    def create_issue_comment(self, number, body):
        self.events.append(("create-comment", number, body))
        comment = IssueComment(
            self.next_comment_id,
            body,
            self.write_author,
            self.write_author_is_bot,
        )
        self.next_comment_id += 1
        self.comments.append(
            replace(
                comment,
                author=(
                    comment.author if self.stored_write_author is None else self.stored_write_author
                ),
                author_is_bot=(
                    comment.author_is_bot
                    if self.stored_write_author_is_bot is None
                    else self.stored_write_author_is_bot
                ),
            )
        )
        return comment

    def update_issue_comment(self, comment_id, body):
        self.events.append(("update-comment", comment_id, body))
        existing = next(comment for comment in self.comments if comment.id == comment_id)
        updated = replace(existing, body=body)
        self.comments = [
            updated if comment.id == comment_id else comment for comment in self.comments
        ]
        return updated

    def add_issue_status_label(self, number, label):
        self.events.append(("add-label", number, label))
        self.issue = replace(
            self.issue,
            labels=tuple(sorted(set(self.issue.labels) | {label})),
        )
        if self.fail_add_once:
            self.fail_add_once = False
            raise GitHubError("ambiguous add")
        return self.issue.labels

    def remove_issue_status_label(self, number, label):
        self.events.append(("remove-label", number, label))
        if self.fail_remove_once:
            self.fail_remove_once = False
            raise GitHubError("ambiguous remove")
        self.issue = replace(
            self.issue,
            labels=tuple(item for item in self.issue.labels if item != label),
        )
        return True


def _config() -> GateConfiguration:
    return GateConfiguration(
        policy=_policy(),
        operators=(_operator(),),
        digest="c" * 64,
    )


def _multi_operator_config() -> GateConfiguration:
    return GateConfiguration(
        policy=_policy(),
        operators=_three_enabled_operators(),
        digest="c" * 64,
    )


def _attempt(state: str = "submitted") -> JobAttempt:
    return JobAttempt(
        attempt_id="attempt-1",
        attempt_number=1,
        request_digest="a" * 64,
        operator="operator",
        slurm_job_id="12345",
        wandb_run_id="issue-42-attempt-1-12345",
        wandb_url="https://wandb.ai/eduLLM/pretraining/runs/issue-42-attempt-1-12345",
        log_path="/home/operator/orcd/scratch/edullm/logs/issue-42-attempt-1-12345.log",
        state=state,
        submitted_at=NOW,
        updated_at=NOW,
    )


def _lifecycle(state: str = "submitted") -> LifecycleState:
    return LifecycleState(
        issue=42,
        request_digest="a" * 64,
        operator="operator",
        assignment_version=0,
        attempts=(_attempt(state),),
        current_state=state,
        updated_at=NOW,
        notification=NotificationRecord(
            event=state if state in {"completed", "failed", "cancelled", "preempted"} else "none",
            status=(
                "pending" if state in {"completed", "failed", "cancelled", "preempted"} else "none"
            ),
            updated_at=NOW,
        ),
    )


def test_job_lifecycle_comment_round_trips_as_strict_bounded_canonical_json():
    state = _lifecycle()

    comment = build_job_comment(state)

    assert comment.startswith(JOB_MARKER + "\n")
    encoded = comment.split("\n", 1)[1]
    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"))
    assert parse_job_comment(comment) == state


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace(JOB_MARKER, "<!-- wrong -->"),
        lambda text: text + "\n" + JOB_MARKER,
        lambda text: text.replace('"issue":42', '"issue":0'),
        lambda text: text.replace('"attempt_number":1', '"attempt_number":true'),
        lambda text: text.replace('"operator":"operator"', '"operator":"Operator"'),
        lambda text: text.replace('"slurm_job_id":"12345"', '"slurm_job_id":"12345;cluster"'),
        lambda text: text.replace('"current_state":"submitted"', '"current_state":"assigned"'),
        lambda text: text.replace('{"assignment_version"', '{ "assignment_version"'),
        lambda text: text.replace('"notification":', '"secret":"never","notification":'),
    ],
)
def test_job_lifecycle_rejects_tampered_noncanonical_or_regressive_state(mutator):
    with pytest.raises(JobOperationError):
        parse_job_comment(mutator(build_job_comment(_lifecycle())))


def test_gate_proves_fresh_issue_status_assignment_identity_policy_and_commit_evidence(
    valid_request,
):
    github = StatefulGitHub(valid_request)

    snapshot = full_submission_gate(
        42,
        operator="operator",
        github=github,
        configuration=_config(),
    )

    assert snapshot.issue == 42
    assert snapshot.request_digest == valid_request.digest
    assert snapshot.operator == "operator"
    assert snapshot.validated_at == NOW
    assert snapshot.assignment_version == 0
    assert snapshot.config_digest == "c" * 64
    assert github.commit_evidence_calls == [(valid_request.commit_sha, valid_request.script_path)]
    assert github.review_calls == []


@pytest.mark.parametrize("operator", ["philote-dev", "meric233", "alsy7009"])
def test_gate_accepts_each_enabled_assigned_operator_without_review_calls(
    valid_request,
    operator,
):
    github = StatefulGitHub(
        valid_request,
        operator=operator,
        commit_evidence=CommitResult(True, "commit and script exist"),
    )

    snapshot = full_submission_gate(
        42,
        operator=operator,
        github=github,
        configuration=_multi_operator_config(),
    )

    assert snapshot.operator == operator
    assert github.commit_evidence_calls == [(valid_request.commit_sha, valid_request.script_path)]
    assert github.review_calls == []


def test_gate_rejects_enabled_operator_when_assignee_is_different(valid_request):
    github = StatefulGitHub(valid_request, operator="philote-dev")

    with pytest.raises(JobOperationError, match="assignee"):
        full_submission_gate(
            42,
            operator="meric233",
            github=github,
            configuration=_multi_operator_config(),
        )

    assert github.commit_evidence_calls == []
    assert github.review_calls == []


@pytest.mark.parametrize(
    "commit_evidence",
    [
        CommitResult(False, "commit does not exist at the requested SHA"),
        CommitResult(False, "script does not exist at the requested SHA"),
    ],
)
def test_gate_rejects_missing_commit_or_script_evidence_without_review_calls(
    valid_request,
    commit_evidence,
):
    github = StatefulGitHub(valid_request, commit_evidence=commit_evidence)

    with pytest.raises(JobOperationError, match="commit or request policy"):
        full_submission_gate(
            42,
            operator="operator",
            github=github,
            configuration=_config(),
        )

    assert github.commit_evidence_calls == [(valid_request.commit_sha, valid_request.script_path)]
    assert github.review_calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda github: setattr(github, "issue", replace(github.issue, body=BODY + "\nchanged")),
        lambda github: setattr(
            github, "issue", replace(github.issue, requester="different-student")
        ),
        lambda github: setattr(
            github,
            "issue",
            replace(github.issue, labels=("edullm-job", "status:assigned", "status:ready")),
        ),
        lambda github: setattr(github, "issue", replace(github.issue, assignees=("faculty",))),
        lambda github: github.comments.append(replace(github.comments[0], id=99)),
        lambda github: setattr(
            github,
            "comments",
            [
                github.comments[0],
                replace(
                    github.comments[1],
                    body=github.comments[1].body.replace('"operator"', '"other"'),
                ),
            ],
        ),
        lambda github: setattr(
            github,
            "comments",
            [replace(github.comments[0], author="student", author_is_bot=False)]
            + github.comments[1:],
        ),
        lambda github: setattr(
            github,
            "commit_evidence",
            CommitResult(False, "commit does not exist at the requested SHA"),
        ),
    ],
)
def test_gate_fails_closed_for_each_fresh_resource_mismatch(valid_request, mutate):
    github = StatefulGitHub(valid_request)
    mutate(github)

    with pytest.raises(JobOperationError):
        full_submission_gate(
            42,
            operator="operator",
            github=github,
            configuration=_config(),
        )


def test_resolved_request_uses_only_protected_model_and_profile_values(valid_request):
    github = StatefulGitHub(valid_request)
    snapshot = full_submission_gate(
        42,
        operator="operator",
        github=github,
        configuration=_config(),
    )

    resolved = build_resolved_request(snapshot, attempt_number=1)

    assert resolved.model_identity == "olmo2-190m"
    assert resolved.allowed_data_kinds == ("skill-dag", "curriculum")
    assert resolved.slurm_partition == "mit_normal_gpu"
    assert resolved.log_pattern == "logs/issue-42-attempt-1-%j.log"


class StatefulRemote:
    def __init__(self, mutate=None, verify_mutate=None, remote_user="orcd-user"):
        self.mutate = mutate
        self.verify_mutate = verify_mutate
        self.remote_user = remote_user
        self.staged = []
        self.submissions = []
        self.verified = []

    def stage(self, key, script, spec):
        self.staged.append((key, script, spec))
        if self.mutate:
            self.mutate()

    def verify_manifest(self, spec):
        self.verified.append(spec)
        if self.verify_mutate:
            self.verify_mutate()

    def submit(self, key, spec):
        self.submissions.append((key, spec))
        return SubmissionReceipt(
            issue=spec.issue,
            request_digest=spec.request_digest,
            attempt_number=spec.attempt_number,
            operator=spec.operator,
            remote_user=spec.remote_user,
            script_sha256=spec.script_sha256,
            manifest_sha256=spec.manifest_sha256,
            slurm_job_id="12345",
            log_path="/home/operator/orcd/scratch/edullm/logs/issue-42-attempt-1-12345.log",
            submitted_at=NOW,
        )


class ReverseOrderGitHub(StatefulGitHub):
    def __init__(self, valid_request):
        super().__init__(valid_request)
        request_43 = replace(valid_request, issue_number=43)
        self.issue_43 = _issue(request_43, number=43)
        assignment_43 = AssignmentState(
            issue=43,
            request_digest=request_43.digest,
            operator_github="operator",
            assigned_at=NOW,
            reminder_at=None,
            history=(),
            slack_status="sent",
        )
        self.comments_43 = [
            IssueComment(
                431,
                build_status_comment(request_43, validated_at=NOW),
                "github-actions[bot]",
                True,
            ),
            IssueComment(
                432,
                build_assignment_comment(assignment_43),
                "github-actions[bot]",
                True,
            ),
        ]

    def list_active_queue_issues(self):
        self.events.append(("list",))
        return (self.issue_43, self.issue)

    def fetch_issue(self, number):
        self.events.append(("fetch", number))
        return self.issue_43 if number == 43 else self.issue

    def list_issue_comments(self, number):
        self.events.append(("comments", number))
        return tuple(self.comments_43 if number == 43 else self.comments)


def test_run_selects_oldest_assigned_issue_and_submits_once(valid_request):
    github = ReverseOrderGitHub(valid_request)
    remote = StatefulRemote()

    state = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=remote,
        now=NOW,
    )

    assert state.issue == 42
    assert len(remote.staged) == 1
    assert len(remote.verified) == 1
    assert len(remote.submissions) == 1
    assert state.attempts[-1].wandb_run_id == "issue-42-attempt-1-12345"
    assert state.attempts[-1].wandb_url == (
        "https://wandb.ai/eduLLM/pretraining/runs/issue-42-attempt-1-12345"
    )


def test_run_publishes_lifecycle_as_the_authenticated_operator_user(valid_request):
    github = StatefulGitHub(
        valid_request,
        write_author="operator",
        write_author_is_bot=False,
    )
    remote = StatefulRemote()

    state = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=remote,
        now=NOW,
    )

    lifecycle_comments = [comment for comment in github.comments if JOB_MARKER in comment.body]
    assert state.current_state == "submitted"
    assert remote.submissions[0][1].remote_user == "orcd-user"
    assert [(comment.author, comment.author_is_bot) for comment in lifecycle_comments] == [
        ("operator", False)
    ]


def test_run_fails_before_staging_without_trusted_remote_user_identity(valid_request):
    github = StatefulGitHub(
        valid_request,
        write_author="operator",
        write_author_is_bot=False,
    )
    remote = StatefulRemote(remote_user=None)

    with pytest.raises(JobOperationError, match="remote user identity"):
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=_config,
            remote=remote,
            now=NOW,
        )

    assert remote.staged == []
    assert remote.submissions == []


@pytest.mark.parametrize(
    "author,author_is_bot",
    [
        ("unrelated-human", False),
        ("operator", True),
        ("github-actions[bot]", True),
    ],
)
def test_gate_rejects_lifecycle_owned_by_unrelated_human_or_bot(
    valid_request,
    author,
    author_is_bot,
):
    lifecycle = replace(
        _lifecycle("submitted"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("submitted"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    github.issue = replace(
        github.issue,
        labels=("edullm-job", "status:assigned", "research"),
    )
    github.comments[2] = replace(
        github.comments[2],
        author=author,
        author_is_bot=author_is_bot,
    )

    with pytest.raises(JobOperationError, match="ownership"):
        full_submission_gate(
            42,
            operator="operator",
            github=github,
            configuration=_config(),
        )


def test_run_fails_closed_when_post_write_comment_author_changes(valid_request):
    github = StatefulGitHub(
        valid_request,
        write_author="operator",
        write_author_is_bot=False,
        stored_write_author="github-actions[bot]",
        stored_write_author_is_bot=True,
    )

    with pytest.raises(JobOperationError, match="ownership"):
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=_config,
            remote=StatefulRemote(),
            now=NOW,
        )


def test_run_sanitizes_unavailable_commit_evidence_without_submission(valid_request):
    secret = "ghp_private_token_forbidden"
    github = StatefulGitHub(
        valid_request,
        commit_evidence=GitHubError(
            f"{secret} failed for {valid_request.commit_sha} {valid_request.data_manifest}"
        ),
    )
    remote = StatefulRemote()

    with pytest.raises(JobOperationError) as raised:
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=_config,
            remote=remote,
            now=NOW,
        )

    message = str(raised.value)
    assert "commit or script evidence is unavailable" in message
    assert secret not in message
    assert valid_request.commit_sha not in message
    assert valid_request.data_manifest not in message
    assert github.commit_evidence_calls == [(valid_request.commit_sha, valid_request.script_path)]
    assert remote.staged == []
    assert remote.verified == []
    assert remote.submissions == []


@pytest.mark.parametrize(
    "resource",
    [
        "body",
        "requester",
        "labels",
        "assignees",
        "status",
        "assignment",
        "commit",
        "config",
    ],
)
def test_run_revalidates_every_resource_between_stage_and_submit_with_zero_sbatch(
    valid_request,
    resource,
):
    github = StatefulGitHub(valid_request)
    configs = [_config(), _config()]

    def mutate():
        if resource == "body":
            github.issue = replace(github.issue, body=BODY + "\nchanged")
        elif resource == "requester":
            github.issue = replace(github.issue, requester="other")
        elif resource == "labels":
            github.issue = replace(
                github.issue,
                labels=("edullm-job", "status:assigned", "status:running"),
            )
        elif resource == "assignees":
            github.issue = replace(github.issue, assignees=("faculty",))
        elif resource == "status":
            github.comments[0] = replace(
                github.comments[0],
                body=github.comments[0].body.replace(
                    "2026-07-23T14:00:00Z", "2026-07-23T14:00:01Z"
                ),
            )
        elif resource == "assignment":
            github.comments[1] = replace(
                github.comments[1],
                body=github.comments[1].body.replace(
                    '"slack_status":"sent"', '"slack_status":"pending"'
                ),
            )
        elif resource == "commit":
            github.commit_evidence = CommitResult(
                False, "script does not exist at the requested SHA"
            )
        elif resource == "config":
            configs[1] = replace(configs[1], digest="d" * 64)

    remote = StatefulRemote(mutate)
    loads = 0

    def load_config():
        nonlocal loads
        value = configs[min(loads, 1)]
        loads += 1
        return value

    with pytest.raises(JobOperationError):
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=load_config,
            remote=remote,
            now=NOW,
        )

    assert len(remote.staged) == 1
    assert remote.submissions == []


def test_run_reverifies_manifest_then_submits_once_and_publishes_monotonic_state(
    valid_request,
):
    github = StatefulGitHub(valid_request)
    remote = StatefulRemote()

    state = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=remote,
        now=NOW,
    )

    assert state.current_state == "submitted"
    assert state.attempts[0].slurm_job_id == "12345"
    assert len(remote.staged) == 1
    assert len(remote.verified) == 1
    assert len(remote.submissions) == 1
    assert "status:submitted" in github.issue.labels
    assert "status:assigned" not in github.issue.labels
    machine = [comment for comment in github.comments if JOB_MARKER in comment.body]
    assert len(machine) == 1
    assert parse_job_comment(machine[0].body) == state
    notices = [comment.body for comment in github.comments if comment.body.startswith("@student")]
    assert notices == [
        "@student eduLLM job #42 was submitted as Slurm job 12345. "
        "W&B: https://wandb.ai/eduLLM/pretraining/runs/issue-42-attempt-1-12345"
    ]
    assert "<script>" not in "\n".join(notices)


def test_run_places_final_authorization_after_remote_manifest_verification(valid_request):
    github = StatefulGitHub(valid_request)

    def change_head_authorization():
        github.commit_evidence = CommitResult(False, "commit does not exist at the requested SHA")

    remote = StatefulRemote(verify_mutate=change_head_authorization)

    with pytest.raises(JobOperationError, match="commit or request policy"):
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=_config,
            remote=remote,
            now=NOW,
        )

    assert len(remote.verified) == 1
    assert remote.submissions == []


def test_run_repairs_interrupted_add_before_remove_label_transition(valid_request):
    github = StatefulGitHub(valid_request, fail_remove_once=True)
    remote = StatefulRemote()

    with pytest.raises(JobOperationError, match="label write"):
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=_config,
            remote=remote,
            now=NOW,
        )

    assert {"status:assigned", "status:submitted"} <= set(github.issue.labels)
    assert "research" in github.issue.labels

    repaired = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=remote,
        now=NOW,
    )

    assert repaired.current_state == "submitted"
    assert set(github.issue.labels) == {"edullm-job", "status:submitted", "research"}


def test_run_reconciles_ambiguous_label_add_and_rejects_later_state_pair(valid_request):
    github = StatefulGitHub(valid_request, fail_add_once=True)

    state = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=StatefulRemote(),
        now=NOW,
    )

    assert state.current_state == "submitted"
    assert set(github.issue.labels) == {"edullm-job", "status:submitted", "research"}

    github.issue = replace(
        github.issue,
        labels=("edullm-job", "status:submitted", "status:running", "research"),
    )
    with pytest.raises(JobOperationError, match="malformed"):
        run_assigned(
            operator="operator",
            github=github,
            load_configuration=_config,
            remote=StatefulRemote(),
            now=NOW,
        )


def test_run_repairs_github_outage_after_receipt_without_new_attempt(valid_request):
    lifecycle = replace(
        _lifecycle("submitted"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("submitted"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    github.issue = replace(
        github.issue,
        labels=("edullm-job", "status:assigned", "research"),
    )
    remote = StatefulRemote()

    repaired = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=remote,
        now=NOW,
    )

    assert repaired == lifecycle
    assert [attempt.attempt_number for attempt in repaired.attempts] == [1]
    assert len(remote.submissions) == 1
    assert "status:submitted" in github.issue.labels
    assert "status:assigned" not in github.issue.labels


class CompletedSlurm:
    def __init__(self):
        self.queries = []

    def query(self, job_ids):
        self.queries.append(tuple(job_ids))
        return {
            "12345": SlurmJob(
                job_id="12345",
                name="issue-42-skill-dag-v1-natural",
                state="COMPLETED",
                user="operator",
                lifecycle_state="completed",
            )
        }

    def reconcile_offline_tracking(self):
        return None


class MixedIssueGitHub:
    def __init__(self, valid_request):
        self.issues = {}
        self.comments = {}
        self.next_comment_id = 1000
        self.list_calls = 0
        self.fetch_calls = []
        self.comment_calls = []
        self.order = (46, 44, 41, 49)
        lifecycle_specs = {
            44: ("running", "44001"),
            49: ("submitted", "49001"),
        }
        for number in self.order:
            request = replace(valid_request, issue_number=number)
            lifecycle = None
            status = "assigned"
            if number in lifecycle_specs:
                status, job_id = lifecycle_specs[number]
                attempt = JobAttempt(
                    attempt_id="attempt-1",
                    attempt_number=1,
                    request_digest=request.digest,
                    operator="operator",
                    slurm_job_id=job_id,
                    wandb_run_id=f"issue-{number}-attempt-1-{job_id}",
                    wandb_url=(
                        "https://wandb.ai/eduLLM/pretraining/runs/"
                        f"issue-{number}-attempt-1-{job_id}"
                    ),
                    log_path=(
                        "/home/operator/orcd/scratch/edullm/logs/"
                        f"issue-{number}-attempt-1-{job_id}.log"
                    ),
                    state=status,
                    submitted_at=NOW,
                    updated_at=NOW,
                )
                lifecycle = LifecycleState(
                    issue=number,
                    request_digest=request.digest,
                    operator="operator",
                    assignment_version=0,
                    attempts=(attempt,),
                    current_state=status,
                    updated_at=NOW,
                    notification=NotificationRecord("none", "none", NOW),
                )
            self.issues[number] = _issue(
                request,
                number=number,
                labels=("edullm-job", f"status:{status}", "research"),
            )
            assignment = AssignmentState(
                issue=number,
                request_digest=request.digest,
                operator_github="operator",
                assigned_at=NOW,
                reminder_at=None,
                history=(),
                slack_status="sent",
            )
            comments = [
                IssueComment(
                    number * 10 + 1,
                    build_status_comment(request, validated_at=NOW),
                    "github-actions[bot]",
                    True,
                ),
                IssueComment(
                    number * 10 + 2,
                    build_assignment_comment(assignment),
                    "github-actions[bot]",
                    True,
                ),
            ]
            if lifecycle is not None:
                comments.append(
                    IssueComment(
                        number * 10 + 3,
                        build_job_comment(lifecycle),
                        "operator",
                        False,
                    )
                )
            self.comments[number] = comments

    def list_active_queue_issues(self):
        self.list_calls += 1
        return tuple(self.issues[number] for number in self.order)

    def fetch_issue(self, number):
        self.fetch_calls.append(number)
        return self.issues[number]

    def list_issue_comments(self, number):
        self.comment_calls.append(number)
        return tuple(self.comments[number])

    def create_issue_comment(self, number, body):
        comment = IssueComment(self.next_comment_id, body, "operator", False)
        self.next_comment_id += 1
        self.comments[number].append(comment)
        return comment

    def update_issue_comment(self, comment_id, body):
        for number, comments in self.comments.items():
            for index, comment in enumerate(comments):
                if comment.id == comment_id:
                    updated = replace(comment, body=body)
                    comments[index] = updated
                    return updated
        raise AssertionError(f"unknown comment {comment_id}")

    def add_issue_status_label(self, number, label):
        issue = self.issues[number]
        self.issues[number] = replace(
            issue,
            labels=tuple(sorted(set(issue.labels) | {label})),
        )
        return self.issues[number].labels

    def remove_issue_status_label(self, number, label):
        issue = self.issues[number]
        self.issues[number] = replace(
            issue,
            labels=tuple(existing for existing in issue.labels if existing != label),
        )
        return True


class MixedJobSlurm:
    def __init__(self):
        self.queries = []
        self.offline_reconciliations = 0

    def query(self, job_ids):
        self.queries.append(tuple(job_ids))
        return {
            "49001": SlurmJob(
                job_id="49001",
                name="issue-49-skill-dag-v1-natural",
                state="RUNNING",
                user="operator",
                lifecycle_state="running",
            ),
            "44001": SlurmJob(
                job_id="44001",
                name="issue-44-skill-dag-v1-natural",
                state="COMPLETED",
                user="operator",
                lifecycle_state="completed",
            ),
        }

    def reconcile_offline_tracking(self):
        self.offline_reconciliations += 1


def test_jobs_lists_assigned_request_without_querying_slurm(valid_request):
    github = StatefulGitHub(valid_request)
    slurm = CompletedSlurm()

    summaries = jobs(
        mine=True,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )

    assert summaries == (
        jobs_module.OperatorJobSummary(
            issue=42,
            status="assigned",
            slurm_job_id=None,
            wandb_url=None,
        ),
    )
    assert slurm.queries == []
    assert ("fetch", 42) in github.events
    assert ("comments", 42) in github.events
    assert github.commit_evidence_calls == []


@pytest.mark.parametrize("assignment_evidence", ["malformed", "mismatched"])
def test_jobs_rejects_invalid_assigned_request_evidence(valid_request, assignment_evidence):
    github = StatefulGitHub(valid_request)
    if assignment_evidence == "malformed":
        body = f"{ASSIGNMENT_MARKER}\n{{"
    else:
        body = github.comments[1].body.replace(valid_request.digest, "b" * 64)
    github.comments[1] = replace(github.comments[1], body=body)
    slurm = CompletedSlurm()

    with pytest.raises(JobOperationError):
        jobs(
            mine=True,
            operator="operator",
            github=github,
            configuration=_config(),
            slurm=slurm,
            now=NOW,
        )

    assert slurm.queries == []
    assert github.commit_evidence_calls == []


def test_jobs_rejects_assigned_request_with_multiple_enabled_operator_assignees(valid_request):
    github = StatefulGitHub(valid_request, operator="philote-dev")
    github.issue = replace(github.issue, assignees=("philote-dev", "meric233"))
    slurm = CompletedSlurm()

    with pytest.raises(JobOperationError, match="assignee"):
        jobs(
            mine=True,
            operator="philote-dev",
            github=github,
            configuration=_multi_operator_config(),
            slurm=slurm,
            now=NOW,
        )

    assert slurm.queries == []
    assert github.commit_evidence_calls == []


def test_jobs_orders_assigned_requests_oldest_first(valid_request):
    github = ReverseOrderGitHub(valid_request)
    slurm = CompletedSlurm()

    summaries = jobs(
        mine=True,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )

    assert summaries == (
        jobs_module.OperatorJobSummary(42, "assigned", None, None),
        jobs_module.OperatorJobSummary(43, "assigned", None, None),
    )
    assert slurm.queries == []


def test_jobs_orders_mixed_summaries_and_queries_only_lifecycle_job_ids(valid_request):
    github = MixedIssueGitHub(valid_request)
    slurm = MixedJobSlurm()

    summaries = jobs(
        mine=True,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )

    assert summaries == (
        jobs_module.OperatorJobSummary(41, "assigned", None, None),
        jobs_module.OperatorJobSummary(46, "assigned", None, None),
        jobs_module.OperatorJobSummary(
            49,
            "running",
            "49001",
            "https://wandb.ai/eduLLM/pretraining/runs/issue-49-attempt-1-49001",
        ),
        jobs_module.OperatorJobSummary(
            44,
            "completed",
            "44001",
            "https://wandb.ai/eduLLM/pretraining/runs/issue-44-attempt-1-44001",
        ),
    )
    assert github.list_calls == 1
    assert github.fetch_calls[:4] == list(github.order)
    assert github.comment_calls[:4] == list(github.order)
    assert slurm.queries == [("44001", "49001")]
    assert slurm.offline_reconciliations == 1


def test_jobs_repairs_terminal_issue_without_submitting(valid_request):
    lifecycle = replace(
        _lifecycle("running"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("running"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    slurm = CompletedSlurm()

    summaries = jobs(
        mine=True,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )

    assert summaries == (
        jobs_module.OperatorJobSummary(
            issue=42,
            status="completed",
            slurm_job_id="12345",
            wandb_url="https://wandb.ai/eduLLM/pretraining/runs/issue-42-attempt-1-12345",
        ),
    )
    assert set(github.issue.labels) == {"edullm-job", "research", "status:completed"}
    assert slurm.queries == [("12345",)]
    assert github.commit_evidence_calls == []
    assert github.review_calls == []


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {"jobs": [{"job_id": 12345, "job_state": "PENDING", "name": "job", "user_name": "u"}]},
            "submitted",
        ),
        (
            {
                "jobs": [
                    {
                        "job_id": 12345,
                        "job_state": ["PENDING"],
                        "name": "job",
                        "user_name": "u",
                    }
                ]
            },
            "submitted",
        ),
        (
            {"jobs": [{"job_id": 12345, "job_state": "RUNNING", "name": "job", "user_name": "u"}]},
            "running",
        ),
        (
            {
                "jobs": [
                    {"job_id": 12345, "job_state": "COMPLETED", "name": "job", "user_name": "u"}
                ]
            },
            "completed",
        ),
        (
            {
                "jobs": [
                    {"job_id": 12345, "job_state": "CANCELLED", "name": "job", "user_name": "u"}
                ]
            },
            "cancelled",
        ),
        (
            {
                "jobs": [
                    {"job_id": 12345, "job_state": "PREEMPTED", "name": "job", "user_name": "u"}
                ]
            },
            "preempted",
        ),
        (
            {
                "jobs": [
                    {"job_id": 12345, "job_state": "OUT_OF_MEMORY", "name": "job", "user_name": "u"}
                ]
            },
            "failed",
        ),
    ],
)
def test_squeue_json_maps_only_known_bounded_states(payload, expected):
    jobs = parse_squeue_json(json.dumps(payload))

    assert len(jobs) == 1
    assert jobs[0].lifecycle_state == expected


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{}",
        '{"jobs":{}}',
        '{"jobs":[{"job_id":"123","job_state":"RUNNING","name":"x","user_name":"u"}]}',
        '{"jobs":[{"job_id":123,"job_state":"UNKNOWN","name":"x","user_name":"u"}]}',
        '{"jobs":[{"job_id":123,"job_state":[],"name":"x","user_name":"u"}]}',
        '{"jobs":[{"job_id":123,"job_state":["RUNNING","PENDING"],"name":"x","user_name":"u"}]}',
        '{"jobs":[{"job_id":123,"job_state":["UNKNOWN"],"name":"x","user_name":"u"}]}',
        '{"jobs":[{"job_id":123,"job_state":[1],"name":"x","user_name":"u"}]}',
        '{"jobs":[{"job_id":123,"job_state":"RUNNING","name":"x\\n","user_name":"u"}]}',
        '{"jobs":[' + '{"job_id":1,"job_state":"RUNNING","name":"x","user_name":"u"},' * 257 + "]}",
    ],
)
def test_squeue_json_rejects_malformed_unknown_or_unbounded_output(payload):
    with pytest.raises(JobOperationError):
        parse_squeue_json(payload)


def test_sacct_parses_exact_field_set_and_rejects_steps_or_unknown_state():
    jobs = parse_sacct("12345|job|COMPLETED|operator\n")
    assert jobs[0].lifecycle_state == "completed"

    for output in (
        "12345.batch|job|COMPLETED|operator\n",
        "12345|job|MYSTERY|operator\n",
        "12345|job|RUNNING|operator|extra\n",
        "12345|job\n",
    ):
        with pytest.raises(JobOperationError):
            parse_sacct(output)


@pytest.mark.parametrize(
    "current,evidence,expected",
    [
        ("submitted", "submitted", "submitted"),
        ("submitted", "running", "running"),
        ("submitted", "completed", "completed"),
        ("running", "completed", "completed"),
        ("completed", "running", "completed"),
        ("failed", "completed", "failed"),
        ("preempted", "submitted", "preempted"),
    ],
)
def test_repair_is_monotonic_and_terminal_never_regresses(current, evidence, expected):
    state = _lifecycle(current)

    repaired = reconcile_lifecycle(state, evidence, now=NOW)

    assert repaired.current_state == expected
    assert reconcile_lifecycle(repaired, evidence, now=NOW) == repaired


def test_logs_read_only_recorded_root_contained_regular_file_and_redact_secrets(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    log = root / "issue-42-attempt-1-12345.log"
    log.write_text(
        "step=1 loss=2\nWANDB_API_KEY=secret-value\n"
        "https://hooks.slack.com/services/T1/B1/SECRET\n",
        encoding="utf-8",
    )

    text = read_authorized_log(str(log), logs_root=root, max_bytes=1024)

    assert "step=1 loss=2" in text
    assert "secret-value" not in text
    assert "hooks.slack.com" not in text
    assert "<redacted>" in text

    outside = tmp_path / "outside"
    outside.write_text("private", encoding="utf-8")
    link = root / "issue-42-attempt-1-99999.log"
    link.symlink_to(outside)
    with pytest.raises(JobOperationError):
        read_authorized_log(str(link), logs_root=root)
    with pytest.raises(JobOperationError):
        read_authorized_log(str(outside), logs_root=root)


class StopSlurm:
    def __init__(self, states):
        self.states = list(states)
        self.cancelled = []

    def status(self, job_id):
        return self.states.pop(0)

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def test_stop_revalidates_binding_cancels_numeric_job_and_waits_for_authoritative_state(
    valid_request,
):
    lifecycle = replace(
        _lifecycle("running"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("running"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    slurm = StopSlurm(["running", "cancelled"])

    result = stop_job(
        42,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )

    assert slurm.cancelled == ["12345"]
    assert result.current_state == "cancelled"


def test_stop_is_idempotent_for_terminal_and_never_cancels_cross_operator(
    valid_request,
):
    terminal = replace(
        _lifecycle("completed"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("completed"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=terminal)
    slurm = StopSlurm(["completed"])
    assert (
        stop_job(
            42,
            operator="operator",
            github=github,
            configuration=_config(),
            slurm=slurm,
            now=NOW,
        ).current_state
        == "completed"
    )
    assert slurm.cancelled == []

    github.issue = replace(github.issue, assignees=("other",))
    with pytest.raises(JobOperationError):
        stop_job(
            42,
            operator="operator",
            github=github,
            configuration=_config(),
            slurm=StopSlurm(["running"]),
            now=NOW,
        )


class TerminalNotifier:
    def __init__(self, github, error=None):
        self.github = github
        self.error = error
        self.calls = []

    def terminal(self, **event):
        lifecycle = parse_job_comment(self.github.comments[2].body)
        assert lifecycle.notification.status == "ambiguous"
        self.calls.append(event)
        if self.error is not None:
            raise self.error


def test_terminal_notification_records_ambiguity_before_delivery_and_sent_after(valid_request):
    lifecycle = replace(
        _lifecycle("completed"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("completed"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    notifier = TerminalNotifier(github)

    results = deliver_terminal_notifications(
        github=github,
        configuration=_config(),
        notifier=notifier,
        now=NOW,
    )

    assert notifier.calls == [{"issue": 42, "operator_slack_id": "U12345678", "state": "completed"}]
    assert results[0].notification.status == "sent"
    assert parse_job_comment(github.comments[2].body).notification.status == "sent"
    assert len([event for event in github.events if event[0] == "update-comment"]) == 2

    assert (
        deliver_terminal_notifications(
            github=github,
            configuration=_config(),
            notifier=notifier,
            now=NOW,
        )[0].notification.status
        == "sent"
    )
    assert len(notifier.calls) == 1


def test_terminal_notification_never_blindly_retries_ambiguous_state(valid_request):
    lifecycle = replace(
        _lifecycle("failed"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("failed"), request_digest=valid_request.digest),),
        notification=NotificationRecord("failed", "ambiguous", NOW),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    notifier = TerminalNotifier(github)

    results = deliver_terminal_notifications(
        github=github,
        configuration=_config(),
        notifier=notifier,
        now=NOW,
    )

    assert results == (lifecycle,)
    assert notifier.calls == []
