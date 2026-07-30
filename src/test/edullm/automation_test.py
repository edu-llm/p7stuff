from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from edullm.automation import (
    VALIDATION_MARKER,
    AutomationResult,
    ValidationDecision,
    _force_requested,
    load_team_leads,
    validate_issue,
    validation_decision,
)
from edullm.github import (
    CommitResult,
    GitHubAPIError,
    GitHubDataError,
    GitHubIssue,
    IssueComment,
)
from edullm.validation import STATUS_MARKER, parse_status_comment

VALIDATED_AT = datetime(2026, 7, 23, 6, 0, 0, tzinfo=timezone.utc)
BODY = Path("src/test/edullm/fixtures/valid_issue.md").read_text(encoding="utf-8")


class RecordingCommitGitHub:
    def __init__(self, result=None):
        self.result = result or CommitResult(True, "commit and script exist")
        self.calls = []

    def executable_commit(self, sha, *, script_path):
        self.calls.append((sha, script_path))
        return self.result


class AutomationGitHub(RecordingCommitGitHub):
    def __init__(
        self,
        issues,
        *,
        comments=(),
        result=None,
        evidence_error=None,
        create_error=None,
        update_error=None,
        labels_error=None,
        concurrent_label_on_add=None,
    ):
        super().__init__(result)
        self.issues = list(issues)
        self.comments = list(comments)
        self.evidence_error = evidence_error
        self.create_error = create_error
        self.update_error = update_error
        self.labels_error = labels_error
        self.concurrent_label_on_add = concurrent_label_on_add
        self.events = []

    def fetch_issue(self, issue_number):
        self.events.append(("fetch", issue_number))
        if len(self.issues) > 1:
            return self.issues.pop(0)
        return self.issues[0]

    def add_issue_status_label(self, issue_number, label):
        self.events.append(("add-label", issue_number, label))
        if self.labels_error is not None:
            raise self.labels_error
        additions = {label}
        if self.concurrent_label_on_add is not None:
            additions.add(self.concurrent_label_on_add)
            self.concurrent_label_on_add = None
        self.issues = [
            replace(issue, labels=tuple(sorted(set(issue.labels) | additions)))
            for issue in self.issues
        ]
        return self.issues[-1].labels

    def remove_issue_status_label(self, issue_number, label):
        self.events.append(("remove-label", issue_number, label))
        if self.labels_error is not None:
            raise self.labels_error
        was_present = any(label in issue.labels for issue in self.issues)
        self.issues = [
            replace(
                issue,
                labels=tuple(existing for existing in issue.labels if existing != label),
            )
            for issue in self.issues
        ]
        return was_present

    def executable_commit(self, *args, **kwargs):
        self.events.append(("evidence", args[0]))
        if self.evidence_error is not None:
            raise self.evidence_error
        return super().executable_commit(*args, **kwargs)

    def list_issue_comments(self, issue_number):
        self.events.append(("comments", issue_number))
        return tuple(self.comments)

    def create_issue_comment(self, issue_number, body):
        self.events.append(("create", issue_number, body))
        if self.create_error is not None:
            raise self.create_error
        comment = IssueComment(
            id=99,
            body=body,
            author="github-actions[bot]",
            author_is_bot=True,
        )
        self.comments.append(comment)
        return comment

    def update_issue_comment(self, comment_id, body):
        self.events.append(("update", comment_id, body))
        if self.update_error is not None:
            raise self.update_error
        comment = IssueComment(
            id=comment_id,
            body=body,
            author="github-actions[bot]",
            author_is_bot=True,
        )
        self.comments = [
            comment if existing.id == comment_id else existing for existing in self.comments
        ]
        return comment


def _issue(
    *,
    body=BODY,
    requester="student",
    labels=("edullm-job", "status:ready", "research"),
):
    return GitHubIssue(
        number=42,
        body=body,
        requester=requester,
        labels=labels,
    )


def _labels_events(github):
    return [event for event in github.events if event[0] in {"add-label", "remove-label"}]


def test_load_team_leads_normalizes_users_and_supported_bots(tmp_path):
    path = tmp_path / "team-leads.yaml"
    path.write_text(
        "team_leads:\n  - Team-Lead\n  - Review-App[bot]\n",
        encoding="utf-8",
    )

    assert load_team_leads(path) == frozenset({"team-lead", "review-app[bot]"})


def test_production_team_lead_allowlist_matches_configured_roster():
    assert load_team_leads(Path("config/edullm/team-leads.yaml")) == frozenset(
        {
            "ericrcwu001",
            "pianomaster99",
            "philote-dev",
            "syz2026",
            "hiyasvyas",
            "meric233",
            "alsy7009",
            "gorpyshortlegs",
        }
    )


@pytest.mark.parametrize(
    "document",
    [
        "",
        "[]\n",
        "unknown: []\n",
        "team_leads: {}\n",
        "team_leads: [operator]\nextra: true\n",
        "team_leads: ['']\n",
        "team_leads: [' operator']\n",
        "team_leads: ['operator/other']\n",
        "team_leads: ['[bot]']\n",
        "team_leads: [operator, Operator]\n",
        "team_leads: [review-app[bot], Review-App[BOT]]\n",
        "team_leads: [7]\n",
    ],
)
def test_load_team_leads_rejects_malformed_or_duplicate_configuration(tmp_path, document):
    path = tmp_path / "team-leads.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="team-leads"):
        load_team_leads(path)


def test_load_team_leads_redacts_yaml_parser_details(tmp_path):
    secret = "ghp_DO_NOT_ECHO_THIS_SECRET"
    path = tmp_path / "team-leads.yaml"
    path.write_text(f"team_leads: [operator\n{secret}", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        load_team_leads(path)

    assert secret not in str(raised.value)


def test_valid_request_uses_immutable_commit_and_script_evidence(valid_request, policy):
    github = RecordingCommitGitHub()

    decision = validation_decision(
        valid_request,
        policy=policy,
        github=github,
    )

    assert decision == ValidationDecision(status="ready", errors=())
    assert github.calls == [(valid_request.commit_sha, valid_request.script_path)]


def test_missing_script_evidence_stays_requested_with_safe_reason(valid_request, policy):
    github = RecordingCommitGitHub(
        CommitResult(False, "script does not exist at the requested SHA")
    )

    decision = validation_decision(
        valid_request,
        policy=policy,
        github=github,
    )

    assert decision == ValidationDecision(
        "requested",
        ("script does not exist at the requested SHA",),
    )


def test_private_commit_evidence_reason_is_sanitized(valid_request, policy):
    github = RecordingCommitGitHub(CommitResult(False, f"private reason: {valid_request.purpose}"))

    decision = validation_decision(
        valid_request,
        policy=policy,
        github=github,
    )

    assert decision == ValidationDecision(
        "requested",
        ("commit evidence was not accepted",),
    )


def test_local_request_errors_short_circuit_commit_evidence(valid_request, policy):
    github = RecordingCommitGitHub()
    request = replace(valid_request, commit_sha="main")

    decision = validation_decision(
        request,
        policy=policy,
        github=github,
    )

    assert decision.status == "requested"
    assert decision.errors == ("commit SHA must be 40 lowercase hexadecimal characters",)
    assert github.calls == []


def test_restricted_request_is_rejected_before_commit_evidence(valid_request, policy):
    github = RecordingCommitGitHub()
    request = replace(valid_request, data_classification="restricted")

    decision = validation_decision(
        request,
        policy=policy,
        github=github,
    )

    assert decision.errors == ("restricted data is not accepted by the public pilot queue",)
    assert github.calls == []


def test_decision_and_automation_results_are_immutable():
    decision = ValidationDecision("requested", ("not ready",))
    result = AutomationResult("requested", ("not ready",), False)

    with pytest.raises(FrozenInstanceError):
        decision.status = "ready"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status = "ready"  # type: ignore[misc]


def test_invalid_issue_is_requested_and_updates_one_sanitized_validation_comment(
    policy,
):
    existing = IssueComment(
        id=5,
        body=f"{VALIDATION_MARKER}\nold",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub(
        [_issue(body=BODY.replace("### Purpose", "## Purpose", 1))],
        comments=[existing],
    )

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is False
    assert github.calls == []
    assert _labels_events(github) == [
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
    ]
    updates = [event for event in github.events if event[0] == "update"]
    assert len(updates) == 1
    assert updates[0][1] == 5
    assert updates[0][2].startswith(VALIDATION_MARKER + "\n")
    assert "missing heading: Purpose" in updates[0][2]
    assert BODY not in updates[0][2]


def test_ready_added_during_invalid_validation_comment_is_removed(policy):
    class AddReadyDuringValidationComment(AutomationGitHub):
        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            super().add_issue_status_label(issue_number, "status:ready")
            return persisted

    github = AddReadyDuringValidationComment(
        [_issue(body=BODY.replace("### Purpose", "## Purpose", 1))]
    )

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_initial_invalidation_targets_only_managed_labels_and_verifies_postcondition(
    policy,
):
    github = AutomationGitHub(
        [_issue(body=BODY.replace("### Purpose", "## Purpose", 1))],
        concurrent_label_on_add="concurrent-review",
    )

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert _labels_events(github)[:2] == [
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
    ]
    assert set(github.issues[0].labels) == {
        "edullm-job",
        "research",
        "concurrent-review",
        "status:requested",
    }
    assert github.events[:3] == [
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
        ("fetch", 42),
    ]


def test_valid_issue_persists_canonical_status_before_ready_and_preserves_labels(
    policy,
):
    github = AutomationGitHub([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult("ready", (), False)
    creates = [event for event in github.events if event[0] == "create"]
    assert len(creates) == 1
    status = parse_status_comment(creates[0][2])
    assert status.request.requester == "student"
    assert status.validated_at == VALIDATED_AT
    assert github.events.index(("evidence", "a" * 40)) > github.events.index(
        ("remove-label", 42, "status:ready")
    )
    assert github.events.index(creates[0]) < github.events.index(("add-label", 42, "status:ready"))
    assert set(github.issues[0].labels) == {
        "edullm-job",
        "research",
        "status:ready",
    }


@pytest.mark.parametrize(
    "second_issue",
    [
        _issue(body=BODY.replace("Skill-DAG smoke", "edited purpose", 1)),
        _issue(requester="different-user"),
    ],
)
def test_issue_edit_or_requester_race_fails_closed(policy, second_issue):
    github = AutomationGitHub([_issue(), second_issue])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.errors == (
        "Issue changed during validation; submit or save the current request again",
    )
    assert not any(event[0] == "create" and STATUS_MARKER in event[2] for event in github.events)
    assert _labels_events(github) == [
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
    ]


def test_duplicate_status_comments_are_rejected_without_selecting_one(policy, valid_request):
    comments = [
        IssueComment(
            id=index,
            body=f"{STATUS_MARKER}\n{{}}",
            author="github-actions[bot]",
            author_is_bot=True,
        )
        for index in (5, 6)
    ]
    github = AutomationGitHub([_issue(), _issue()], comments=comments)

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM status comments were found",)
    assert not any(event[0] in {"create", "update"} for event in github.events)
    assert "status:ready" not in github.issues[0].labels
    assert "status:requested" in github.issues[0].labels


def test_duplicate_status_markers_in_one_comment_are_rejected(policy):
    comment = IssueComment(
        id=5,
        body=f"{STATUS_MARKER}\n{{}}\n{STATUS_MARKER}",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[comment])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM status markers were found",)


def test_human_authored_machine_marker_is_rejected(policy):
    comment = IssueComment(
        id=5,
        body=f"{STATUS_MARKER}\n{{}}",
        author="student",
        author_is_bot=False,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[comment])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("eduLLM status marker is not bot-authored",)


@pytest.mark.parametrize(
    "comments,expected_error",
    [
        (
            [
                IssueComment(
                    id=5,
                    body=f"{VALIDATION_MARKER}\nold",
                    author="student",
                    author_is_bot=False,
                )
            ],
            "eduLLM validation marker is not bot-authored",
        ),
        (
            [
                IssueComment(
                    id=index,
                    body=f"{VALIDATION_MARKER}\nold",
                    author="github-actions[bot]",
                    author_is_bot=True,
                )
                for index in (5, 6)
            ],
            "multiple eduLLM validation comments were found",
        ),
        (
            [
                IssueComment(
                    id=5,
                    body=f"{VALIDATION_MARKER}\nold\n{VALIDATION_MARKER}",
                    author="github-actions[bot]",
                    author_is_bot=True,
                )
            ],
            "multiple eduLLM validation markers were found",
        ),
    ],
)
def test_invalid_validation_marker_namespace_blocks_ready(policy, comments, expected_error):
    github = AutomationGitHub([_issue(), _issue()], comments=comments)

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult("requested", (expected_error,), True)
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_validation_marker_inserted_after_status_persistence_blocks_ready(policy):
    class InsertValidationAfterStatusCreate(AutomationGitHub):
        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            if STATUS_MARKER in body:
                self.comments.append(
                    IssueComment(
                        id=100,
                        body=f"{VALIDATION_MARKER}\nspoofed",
                        author="student",
                        author_is_bot=False,
                    )
                )
            return persisted

    github = InsertValidationAfterStatusCreate([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_validation_marker_inserted_during_ready_write_blocks_ready(policy):
    class InsertValidationDuringReady(AutomationGitHub):
        def remove_issue_status_label(self, issue_number, label):
            removed = super().remove_issue_status_label(issue_number, label)
            if label == "status:requested":
                self.comments.append(
                    IssueComment(
                        id=100,
                        body=f"{VALIDATION_MARKER}\nspoofed",
                        author="student",
                        author_is_bot=False,
                    )
                )
            return removed

    github = InsertValidationDuringReady([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_validation_marker_inserted_after_ready_publication_blocks_ready(policy):
    class InsertValidationOnPublishedCommentRead(AutomationGitHub):
        inserted = False

        def list_issue_comments(self, issue_number):
            labels = set(self.issues[0].labels)
            if "status:ready" in labels and "status:requested" not in labels and not self.inserted:
                self.inserted = True
                self.comments.extend(
                    [
                        IssueComment(
                            id=index,
                            body=f"{VALIDATION_MARKER}\nold",
                            author="github-actions[bot]",
                            author_is_bot=True,
                        )
                        for index in (100, 101)
                    ]
                )
            return super().list_issue_comments(issue_number)

    github = InsertValidationOnPublishedCommentRead([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_single_bot_validation_comment_may_coexist_with_ready(policy):
    validation_comment = IssueComment(
        id=5,
        body=f"{VALIDATION_MARKER}\nprior validation",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[validation_comment])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "ready"
    assert len(github.comments) == 2
    assert "status:ready" in github.issues[0].labels
    assert "status:requested" not in github.issues[0].labels


def test_idempotent_retry_updates_existing_status_comment_without_duplication(
    policy,
):
    existing = IssueComment(
        id=5,
        body=f"{STATUS_MARKER}\n{{}}",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[existing])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "ready"
    assert len([event for event in github.events if event[0] == "update"]) == 1
    assert not any(event[0] == "create" for event in github.events)
    assert len(github.comments) == 1
    parse_status_comment(github.comments[0].body)


def test_repeated_unchanged_validation_converges_to_one_comment_and_ready(policy):
    github = AutomationGitHub([_issue(labels=("edullm-job", "research", "status:ready"))])

    first = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )
    second = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert first.status == second.status == "ready"
    assert len(github.comments) == 1
    parse_status_comment(github.comments[0].body)
    assert set(github.issues[0].labels) == {
        "edullm-job",
        "research",
        "status:ready",
    }


def test_concurrent_unrelated_label_during_ready_is_preserved(policy):
    class AddUnrelatedWithReady(AutomationGitHub):
        def add_issue_status_label(self, issue_number, label):
            result = super().add_issue_status_label(issue_number, label)
            if label == "status:ready":
                self.issues = [
                    replace(
                        issue,
                        labels=tuple(sorted(set(issue.labels) | {"concurrent-review"})),
                    )
                    for issue in self.issues
                ]
            return result

    github = AddUnrelatedWithReady([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "ready"
    assert "concurrent-review" in github.issues[0].labels
    assert "status:ready" in github.issues[0].labels
    assert "status:requested" not in github.issues[0].labels


def test_concurrent_duplicate_after_status_create_fails_closed(policy):
    class DuplicateAfterCreate(AutomationGitHub):
        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            self.comments.append(
                IssueComment(
                    id=100,
                    body=body,
                    author="other-bot[bot]",
                    author_is_bot=True,
                )
            )
            return persisted

    github = DuplicateAfterCreate([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM status comments were found",)
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_concurrent_human_marker_after_status_create_fails_closed(policy):
    class HumanMarkerAfterCreate(AutomationGitHub):
        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            self.comments.append(
                IssueComment(
                    id=100,
                    body=body,
                    author="student",
                    author_is_bot=False,
                )
            )
            return persisted

    github = HumanMarkerAfterCreate([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_status_write_requires_relisted_persisted_comment_identity(policy):
    class MismatchedCreateIdentity(AutomationGitHub):
        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            return replace(persisted, id=persisted.id + 1)

    github = MismatchedCreateIdentity([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert result.errors == ("persisted eduLLM status comment identity changed",)
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


@pytest.mark.parametrize("field", ["body", "requester"])
def test_issue_identity_change_after_comment_before_ready_converges_requested(policy, field):
    class EditAfterCreate(AutomationGitHub):
        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            if field == "body":
                self.issues = [
                    replace(issue, body=issue.body.replace("Skill-DAG smoke", "edited", 1))
                    for issue in self.issues
                ]
            else:
                self.issues = [replace(issue, requester="different-user") for issue in self.issues]
            return persisted

    github = EditAfterCreate([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.errors == (
        "Issue changed during validation; submit or save the current request again",
    )
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_concurrent_duplicate_during_ready_write_reconciles_requested(policy):
    class DuplicateDuringReady(AutomationGitHub):
        def remove_issue_status_label(self, issue_number, label):
            removed = super().remove_issue_status_label(issue_number, label)
            if label == "status:requested":
                self.comments.append(
                    IssueComment(
                        id=100,
                        body=self.comments[0].body,
                        author="other-bot[bot]",
                        author_is_bot=True,
                    )
                )
            return removed

    github = DuplicateDuringReady([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


@pytest.mark.parametrize("field", ["body", "requester"])
def test_issue_identity_change_during_ready_write_reconciles_requested(policy, field):
    class EditDuringReady(AutomationGitHub):
        def remove_issue_status_label(self, issue_number, label):
            removed = super().remove_issue_status_label(issue_number, label)
            if label == "status:requested":
                if field == "body":
                    self.issues = [
                        replace(issue, body=issue.body.replace("Skill-DAG smoke", "edited", 1))
                        for issue in self.issues
                    ]
                else:
                    self.issues = [
                        replace(issue, requester="different-user") for issue in self.issues
                    ]
            return removed

    github = EditDuringReady([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


@pytest.mark.parametrize("field", ["body", "requester"])
def test_issue_identity_change_after_ready_postcondition_reconciles_requested(policy, field):
    class EditAfterReadyFetch(AutomationGitHub):
        edited = False

        def fetch_issue(self, issue_number):
            issue = super().fetch_issue(issue_number)
            if "status:ready" in issue.labels and not self.edited:
                self.edited = True
                if field == "body":
                    self.issues = [
                        replace(current, body=current.body.replace("Skill-DAG smoke", "edited", 1))
                        for current in self.issues
                    ]
                else:
                    self.issues = [
                        replace(current, requester="different-user") for current in self.issues
                    ]
            return issue

    github = EditAfterReadyFetch([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_duplicate_validation_comments_fail_closed_on_invalid_request(policy):
    comments = [
        IssueComment(
            id=index,
            body=f"{VALIDATION_MARKER}\nold",
            author="github-actions[bot]",
            author_is_bot=True,
        )
        for index in (5, 6)
    ]
    github = AutomationGitHub(
        [_issue(body=BODY.replace("### Purpose", "## Purpose", 1))],
        comments=comments,
    )

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM validation comments were found",)
    assert not any(event[0] in {"create", "update"} for event in github.events)


@pytest.mark.parametrize(
    "failure_field",
    ["evidence_error", "create_error", "update_error"],
)
def test_github_failures_never_transition_to_ready(policy, failure_field):
    kwargs = {failure_field: GitHubAPIError("GitHub API request failed")}
    comments: tuple[IssueComment, ...] = ()
    if failure_field == "update_error":
        comments = (
            IssueComment(
                id=5,
                body=f"{STATUS_MARKER}\n{{}}",
                author="github-actions[bot]",
                author_is_bot=True,
            ),
        )
    github = AutomationGitHub([_issue(), _issue()], comments=comments, **kwargs)

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert result.errors == ("GitHub validation operation failed",)
    assert "status:ready" not in github.issues[0].labels
    assert "status:requested" in github.issues[0].labels


def test_ready_label_failure_reports_operational_error_after_status_persistence(
    policy,
):
    class FailSecondLabelUpdate(AutomationGitHub):
        def add_issue_status_label(self, issue_number, label):
            if label == "status:ready":
                raise GitHubAPIError("sensitive API response")
            return super().add_issue_status_label(issue_number, label)

    github = FailSecondLabelUpdate([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult(
        "requested",
        ("GitHub validation operation failed",),
        True,
    )
    assert any(event[0] == "create" and STATUS_MARKER in event[2] for event in github.events)
    assert _labels_events(github) == [
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
        ("add-label", 42, "status:requested"),
        ("remove-label", 42, "status:ready"),
    ]


@pytest.mark.parametrize(
    "error",
    [
        GitHubAPIError("request timed out"),
        GitHubDataError("malformed write response"),
    ],
)
def test_ready_add_committed_before_error_is_reconciled_requested(policy, error):
    class CommitReadyThenFail(AutomationGitHub):
        failed = False

        def add_issue_status_label(self, issue_number, label):
            result = super().add_issue_status_label(issue_number, label)
            if label == "status:ready" and not self.failed:
                self.failed = True
                self.issues = [
                    replace(
                        issue,
                        labels=tuple(sorted(set(issue.labels) | {"concurrent-review"})),
                    )
                    for issue in self.issues
                ]
                raise error
            return result

    github = CommitReadyThenFail([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult(
        "requested",
        ("GitHub validation operation failed",),
        True,
    )
    assert set(github.issues[0].labels) == {
        "edullm-job",
        "research",
        "concurrent-review",
        "status:requested",
    }


def test_requested_remove_committed_before_timeout_is_reconciled(policy):
    class CommitRemoveThenTimeout(AutomationGitHub):
        failed = False

        def remove_issue_status_label(self, issue_number, label):
            result = super().remove_issue_status_label(issue_number, label)
            if label == "status:requested" and not self.failed:
                self.failed = True
                raise GitHubAPIError("request timed out")
            return result

    github = CommitRemoveThenTimeout([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


@pytest.mark.parametrize(
    "error",
    [
        GitHubAPIError("request timed out"),
        GitHubDataError("malformed write response"),
    ],
)
def test_comment_committed_before_error_is_reconciled_requested(policy, error):
    class CommitCommentThenFail(AutomationGitHub):
        failed = False

        def create_issue_comment(self, issue_number, body):
            persisted = super().create_issue_comment(issue_number, body)
            if not self.failed:
                self.failed = True
                raise error
            return persisted

    github = CommitCommentThenFail([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert len(github.comments) == 1
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_failed_ready_postcondition_is_reconciled_requested(policy):
    class StaleReadyPostcondition(AutomationGitHub):
        injected = False

        def fetch_issue(self, issue_number):
            issue = super().fetch_issue(issue_number)
            if "status:ready" in issue.labels and not self.injected:
                self.injected = True
                return replace(
                    issue,
                    labels=tuple(sorted(set(issue.labels) | {"status:requested"})),
                )
            return issue

    github = StaleReadyPostcondition([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.errors == ("failed to establish ready status",)
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels


def test_unavailable_reconciliation_never_claims_ready_and_is_sanitized(policy):
    secret = "ghp_DO_NOT_ECHO_THIS_SECRET"

    class TotalOutageAfterReadyCommit(AutomationGitHub):
        ready_failed = False

        def add_issue_status_label(self, issue_number, label):
            if label == "status:requested" and self.ready_failed:
                raise GitHubAPIError(f"outage contains {secret}")
            result = super().add_issue_status_label(issue_number, label)
            if label == "status:ready" and not self.ready_failed:
                self.ready_failed = True
                raise GitHubAPIError(f"timeout contains {secret}")
            return result

        def fetch_issue(self, issue_number):
            if self.ready_failed:
                raise GitHubAPIError(f"outage contains {secret}")
            return super().fetch_issue(issue_number)

    github = TotalOutageAfterReadyCommit([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult(
        "requested",
        ("GitHub validation reconciliation failed",),
        True,
    )
    assert secret not in "\n".join(result.errors)


class ReconciliationGitHub(AutomationGitHub):
    def __init__(
        self,
        issue,
        *,
        add_error=False,
        remove_error=False,
        fetch_error=False,
    ):
        super().__init__([issue])
        self.add_error = add_error
        self.remove_error = remove_error
        self.fetch_error = fetch_error

    def add_issue_status_label(self, issue_number, label):
        if self.add_error:
            self.events.append(("add-error", issue_number, label))
            raise GitHubAPIError("add failed")
        return super().add_issue_status_label(issue_number, label)

    def remove_issue_status_label(self, issue_number, label):
        if self.remove_error:
            self.events.append(("remove-error", issue_number, label))
            raise GitHubAPIError("remove failed")
        return super().remove_issue_status_label(issue_number, label)

    def fetch_issue(self, issue_number):
        if self.fetch_error:
            self.events.append(("fetch-error", issue_number))
            raise GitHubAPIError("fetch failed")
        return super().fetch_issue(issue_number)


def test_force_requested_attempts_remove_when_add_fails_and_accepts_postcondition():
    github = ReconciliationGitHub(
        _issue(labels=("edullm-job", "research", "status:requested", "status:ready")),
        add_error=True,
    )

    issue = _force_requested(github, 42)

    assert issue is not None
    assert set(github.issues[0].labels) == {
        "edullm-job",
        "research",
        "status:requested",
    }
    assert ("remove-label", 42, "status:ready") in github.events


def test_force_requested_accepts_remove_error_when_ready_is_already_absent():
    github = ReconciliationGitHub(
        _issue(labels=("edullm-job", "research")),
        remove_error=True,
    )

    issue = _force_requested(github, 42)

    assert issue is not None
    assert set(github.issues[0].labels) == {
        "edullm-job",
        "research",
        "status:requested",
    }
    assert ("fetch", 42) in github.events


def test_force_requested_accepts_both_write_errors_when_state_is_already_safe():
    github = ReconciliationGitHub(
        _issue(labels=("edullm-job", "research", "status:requested")),
        add_error=True,
        remove_error=True,
    )

    issue = _force_requested(github, 42)

    assert issue is not None
    assert ("add-error", 42, "status:requested") in github.events
    assert ("remove-error", 42, "status:ready") in github.events
    assert ("fetch", 42) in github.events


def test_force_requested_attempts_both_writes_before_unavailable_fetch():
    github = ReconciliationGitHub(
        _issue(labels=("edullm-job", "research", "status:ready")),
        fetch_error=True,
    )

    assert _force_requested(github, 42) is None
    assert ("add-label", 42, "status:requested") in github.events
    assert ("remove-label", 42, "status:ready") in github.events
    assert ("fetch-error", 42) in github.events


def test_ready_commit_then_error_removes_ready_even_if_reconciliation_add_fails(policy):
    class CommitReadyThenFailWithAddOutage(AutomationGitHub):
        ready_failed = False

        def add_issue_status_label(self, issue_number, label):
            if label == "status:requested" and self.ready_failed:
                self.events.append(("add-error", issue_number, label))
                raise GitHubAPIError("add endpoint unavailable")
            result = super().add_issue_status_label(issue_number, label)
            if label == "status:ready" and not self.ready_failed:
                self.ready_failed = True
                raise GitHubAPIError("ready response lost")
            return result

    github = CommitReadyThenFailWithAddOutage([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult(
        "requested",
        ("GitHub validation operation failed",),
        True,
    )
    assert "status:requested" in github.issues[0].labels
    assert "status:ready" not in github.issues[0].labels
    assert ("remove-label", 42, "status:ready") in github.events
