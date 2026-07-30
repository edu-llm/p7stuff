import io
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from edullm.request_parser import (
    ISSUE_HEADINGS,
    fields_from_markdown,
    issue_body_from_fields,
)

SKILL = Path(".cursor/skills/submit-edullm-job")
FIXTURE = Path("src/test/edullm/fixtures/valid_issue.md")


class _FlushFailure:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        raise OSError("raw output failure")


def _run_adapter_arguments(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts/validate_request.py"),
            *arguments,
        ],
        check=False,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src", "WANDB_API_KEY": "must-not-appear"},
    )


def _run_adapter(source: Path) -> subprocess.CompletedProcess[str]:
    return _run_adapter_arguments(
        "--input-json",
        str(source),
        "--requester",
        "student",
    )


def test_submission_skill_is_real_agent_facing_workflow():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter == {
        "name": "submit-edullm-job",
        "description": (
            "Use when a submitting team explicitly invokes /submit-edullm-job "
            "to request an eduLLM job on MIT Engaging."
        ),
        "disable-model-invocation": True,
    }
    assert "/submit-edullm-job" in text
    assert "[request reference](request-reference.md)" in text
    assert "[validation adapter](scripts/validate_request.py)" in text
    assert "gh issue create" in text
    assert "--body-file" in text
    assert "explicit confirmation" in text
    assert "Issue form is not a substitute" in text
    assert "Never request or handle credentials" in text
    assert "ssh orcd-login" not in text
    assert "The assigned operator authorizes a job by running `edullm run`." in text
    assert "sbatch " not in text
    assert 'STATUS="$(git status --porcelain=v1)" || exit 2' in text
    assert 'test -z "$STATUS"' in text
    assert 'test -z "$(git status' not in text
    assert len(text.splitlines()) < 500


def test_skill_fails_closed_on_exact_pushed_commit_and_cleans_private_files():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert 'test "$BRANCH" != main' in text
    assert 'COMMIT_SHA="$(git rev-parse HEAD)" || exit 2' in text
    assert 'test "${#COMMIT_SHA}" -eq 40' in text
    assert 'case "$COMMIT_SHA" in' in text
    assert '(*[!0-9a-f]*|"") exit 2 ;;' in text
    assert "gh repo view edu-llm/OLMo-core --json nameWithOwner --jq .nameWithOwner" in text
    assert 'test "$CANONICAL_OWNER" = edu-llm/OLMo-core' in text
    assert 'REMOTE_URL="$(git remote get-url origin)" || exit 2' in text
    assert '"$CANONICAL_HTTPS.git"' in text
    assert '"$CANONICAL_SSH"' in text
    assert "ssh://git@github.com/edu-llm/OLMo-core.git" in text
    assert (
        'REMOTE_SHA="$(gh api "repos/edu-llm/OLMo-core/commits/$COMMIT_SHA" --jq .sha)" || exit 2'
        in text
    )
    assert 'test "$REMOTE_SHA" = "$COMMIT_SHA"' in text
    assert "gh pr view" not in text
    assert "approval" not in lowered
    assert "reviewer" not in lowered
    assert "check state" not in lowered
    assert "pr url" not in lowered
    assert 'REQUEST_DIR="$(mktemp -d)"' in text
    assert 'chmod 700 "$REQUEST_DIR"' in text
    assert "trap 'rm -rf \"$REQUEST_DIR\"' EXIT" in text
    assert "umask 077" in text
    assert 'chmod 600 "$REQUEST_DIR/request.json" "$REQUEST_DIR/issue.md"' in text


def test_skill_passes_the_unchanged_validated_body_as_a_file():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert '> "$REQUEST_DIR/issue.md"' in text
    assert '--title "[eduLLM job]: ${STUDY}-${CONDITION}"' in text
    assert '--body-file "$REQUEST_DIR/issue.md"' in text
    assert "--label edullm-job" in text
    assert "--label status:requested" in text
    assert "Do not edit, re-render, copy, or transform `issue.md`" in text
    assert "Actions is authoritative" in text
    assert "The request Issue is not a compute submission." in text
    assert "edullm approve" not in text
    assert "edullm reject" not in text


def test_skill_suppresses_external_diagnostics_and_checks_requester_lookup():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "exec 3>&2\nexec 2>/dev/null" in text
    assert "exec 2>&3 3>&-" in text
    assert 'REQUESTER="$(gh api user --jq .login 2>/dev/null)" || exit 2' in text
    assert 'test -n "$REQUESTER"' in text
    assert '--requester "$REQUESTER"' in text
    assert '--body-file "$REQUEST_DIR/issue.md" 2>/dev/null)" || exit 2' in text
    assert "--json number,url,labels,assignees,comments 2>/dev/null" in text


def test_request_reference_tracks_the_authoritative_heading_order():
    text = (SKILL / "request-reference.md").read_text(encoding="utf-8")
    offsets = [text.index(f"`{heading}`") for heading in ISSUE_HEADINGS]
    assert offsets == sorted(offsets)
    assert "exact full pushed commit SHA" in text
    assert "pull request" not in text.lower()
    assert "approval" not in text.lower()
    assert "ordered JSON array of strings" in text
    assert "`builtin://generic-smoke-v1`" in text
    assert "policy-controlled" in text
    assert "`/orcd/pool/...`" in text
    assert "no S3 URI" in text
    assert "names actually emitted by the selected code" in text
    assert "`generic-smoke`" in text
    assert "`src/examples/llm/train.py`" in text
    assert "`torchrun`" in text
    assert '`["orcd-bootstrap"]`' in text
    assert "`1c82abfc35b17e8a15eae8e0e1afa3dee6696aeb213d46799f204e1c4fc093d7`" in text
    assert "one L40S" in text
    assert "30 minutes" in text
    assert "W&B project `test`" in text
    assert "operator-only" in text


def test_adapter_reuses_parser_policy_and_emits_exact_validated_body(tmp_path):
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    source = tmp_path / "request.json"
    source.write_text(json.dumps(fields), encoding="utf-8")

    result = _run_adapter(source)

    assert result.returncode == 0
    assert result.stdout == issue_body_from_fields(fields) + "\n"
    assert fields_from_markdown(result.stdout) == fields
    assert result.stderr == ""


@pytest.mark.parametrize(
    "raw_argument",
    [
        "--unknown=ghp_SENTINEL_SECRET",
        "/private/credentials/ghp_SENTINEL_SECRET/wandb.key",
    ],
)
def test_adapter_sanitizes_unknown_or_malformed_command_arguments(tmp_path, raw_argument):
    source = tmp_path / "request.json"

    result = _run_adapter_arguments(
        "--input-json",
        str(source),
        "--requester",
        "student",
        raw_argument,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "invalid command arguments\n"
    assert "ghp_SENTINEL_SECRET" not in result.stdout + result.stderr


def test_adapter_rejects_unsafe_argv_without_echoing_credentials(tmp_path):
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    fields["Arguments JSON"] = '["train_single", "x; env"]'
    source = tmp_path / "request.json"
    source.write_text(json.dumps(fields), encoding="utf-8")

    result = _run_adapter(source)

    assert result.returncode == 2
    assert "unsafe argument" in result.stderr
    assert "x; env" not in result.stdout + result.stderr
    assert "must-not-appear" not in result.stdout + result.stderr


def test_adapter_sanitizes_file_and_document_errors(tmp_path):
    secret = "ghp_DO_NOT_ECHO"
    missing = tmp_path / secret

    missing_result = _run_adapter(missing)

    assert missing_result.returncode == 2
    assert missing_result.stderr == "request input could not be read\n"
    assert secret not in missing_result.stdout + missing_result.stderr

    malformed = tmp_path / "request.json"
    malformed.write_text(f'{{"Purpose": "{secret}"', encoding="utf-8")
    malformed_result = _run_adapter(malformed)

    assert malformed_result.returncode == 2
    assert malformed_result.stderr == "request input is not valid JSON\n"
    assert secret not in malformed_result.stdout + malformed_result.stderr


def test_adapter_sanitizes_output_encoding_errors(tmp_path):
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    fields["Purpose"] = "\ud800"
    source = tmp_path / "request.json"
    source.write_text(json.dumps(fields), encoding="utf-8")

    result = _run_adapter(source)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "validated request could not be written\n"


def test_adapter_sanitizes_delayed_output_flush_errors(tmp_path, monkeypatch):
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    source = tmp_path / "request.json"
    source.write_text(json.dumps(fields), encoding="utf-8")
    namespace = runpy.run_path(str(SKILL / "scripts/validate_request.py"))
    main = namespace["main"]
    errors = io.StringIO()
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_request.py", "--input-json", str(source), "--requester", "student"],
    )
    monkeypatch.setattr(sys, "stdout", _FlushFailure())
    monkeypatch.setattr(sys, "stderr", errors)

    result = main()

    assert result == 2
    assert errors.getvalue() == "validated request could not be written\n"
