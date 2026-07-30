from pathlib import Path

import yaml

WORKFLOWS = Path(".github/workflows")


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _trigger(document):
    return document.get("on", document.get(True))


def test_core_assignment_handoff_requires_only_the_slack_secret():
    assign = _load(WORKFLOWS / "edullm-assign.yml")
    validate = _load(WORKFLOWS / "edullm-validate.yml")

    assert _trigger(assign)["workflow_call"]["secrets"] == {"SLACK_WEBHOOK_URL": {"required": True}}
    assert validate["jobs"]["assign"]["needs"] == "validate"
    assert validate["jobs"]["assign"]["uses"] == "./.github/workflows/edullm-assign.yml"
    assert validate["jobs"]["assign"]["secrets"] == {
        "SLACK_WEBHOOK_URL": "${{ secrets.SLACK_WEBHOOK_URL }}"
    }


def test_only_core_workflows_are_enabled():
    validate = yaml.load(
        (WORKFLOWS / "edullm-validate.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assign = yaml.load(
        (WORKFLOWS / "edullm-assign.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    reminders = (WORKFLOWS / "edullm-reminders.yml").read_text(encoding="utf-8")
    terminal = (WORKFLOWS / "edullm-terminal-notify.yml").read_text(encoding="utf-8")

    assert validate["jobs"]["validate"]["if"] == (
        "${{ contains(github.event.issue.labels.*.name, 'edullm-job') }}"
    )
    assert validate["jobs"]["assign"]["if"] == "${{ needs.validate.result == 'success' }}"
    assert assign["jobs"]["assign"]["if"] == "${{ github.repository == 'edu-llm/OLMo-core' }}"
    assert "${{ false &&" in reminders
    assert "${{ false &&" in terminal

    paths = sorted(WORKFLOWS.glob("edullm-*.yml"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "persist-credentials: false" in text
        for line in text.splitlines():
            if line.strip().startswith("- uses:"):
                reference = line.split("@", 1)[1]
                assert len(reference) == 40
                assert all(character in "0123456789abcdef" for character in reference)


def test_terminal_notification_workflow_has_least_privilege_and_scoped_secret():
    path = WORKFLOWS / "edullm-terminal-notify.yml"
    text = path.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert document["permissions"] == {"contents": "read", "issues": "write"}
    assert document["concurrency"] == {
        "group": "edullm-terminal-notification",
        "cancel-in-progress": "false",
    }
    steps = document["jobs"]["notify"]["steps"]
    secret_steps = [step for step in steps if "SLACK_WEBHOOK_URL" in step.get("env", {})]
    assert len(secret_steps) == 1
    assert secret_steps[0]["run"] == "python -m edullm.cli automation terminal"
    assert "SLACK_WEBHOOK_URL" not in "\n".join(step.get("run", "") for step in steps[:-1])
