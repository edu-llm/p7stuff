import re
from pathlib import Path

import yaml

ASSIGN = Path(".github/workflows/edullm-assign.yml")
REMINDERS = Path(".github/workflows/edullm-reminders.yml")
VALIDATE = Path(".github/workflows/edullm-validate.yml")


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _trigger(document):
    return document.get("on", document.get(True))


def test_assignment_workflow_has_explicit_handoff_and_retry_triggers():
    workflow = _load(ASSIGN)
    trigger = _trigger(workflow)

    assert set(trigger) == {"workflow_call", "workflow_dispatch", "schedule"}
    assert trigger["schedule"] == [{"cron": "*/5 * * * *"}]
    assert trigger["workflow_call"]["secrets"] == {"SLACK_WEBHOOK_URL": {"required": True}}


def test_validation_workflow_has_enabled_reusable_assignment_handoff():
    workflow = _load(VALIDATE)
    handoff = workflow["jobs"]["assign"]

    assert handoff["needs"] == "validate"
    assert handoff["if"] == "${{ needs.validate.result == 'success' }}"
    assert handoff["uses"] == "./.github/workflows/edullm-assign.yml"
    assert handoff["secrets"] == {"SLACK_WEBHOOK_URL": "${{ secrets.SLACK_WEBHOOK_URL }}"}
    assert "status:ready" not in _trigger(workflow)["issues"]["types"]


def test_task_5_workflows_are_globally_serialized_and_reminders_stay_disabled():
    for path in (ASSIGN, REMINDERS):
        workflow = _load(path)
        assert workflow["concurrency"] == {
            "group": "edullm-assignment",
            "cancel-in-progress": False,
        }

    text = REMINDERS.read_text(encoding="utf-8")
    job = next(iter(_load(REMINDERS)["jobs"].values()))
    assert "${{ false" in job["if"]
    assert "vars." not in job["if"]
    assert "secrets." not in job["if"]
    assert "${{ false" in text


def test_reminder_workflow_has_only_scheduled_and_manual_scanning():
    workflow = _load(REMINDERS)

    assert _trigger(workflow) == {
        "schedule": [{"cron": "*/5 * * * *"}],
        "workflow_dispatch": None,
    }


def test_task_5_workflows_use_least_privilege_pins_and_secret_scoping():
    for path, command in (
        (ASSIGN, "automation assign"),
        (REMINDERS, "automation reminders"),
    ):
        workflow = _load(path)
        assert workflow["permissions"] == {"contents": "read", "issues": "write"}
        steps = next(iter(workflow["jobs"].values()))["steps"]
        checkout, setup = steps[:2]
        execution = steps[-1]

        assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout["uses"])
        assert re.fullmatch(r"actions/setup-python@[0-9a-f]{40}", setup["uses"])
        assert checkout["with"]["persist-credentials"] is False
        assert setup["with"]["python-version"] == "3.11"
        assert command in execution["run"]
        assert "${{" not in execution["run"]
        assert execution["env"] == {
            "GITHUB_TOKEN": "${{ github.token }}",
            "GITHUB_REPOSITORY": "${{ github.repository }}",
            "SLACK_WEBHOOK_URL": "${{ secrets.SLACK_WEBHOOK_URL }}",
        }
        assert all("SLACK_WEBHOOK_URL" not in str(step) for step in steps[:-1])
        assert not any("gh " in step.get("run", "") for step in steps)
        assert "id-token" not in workflow["permissions"]
        assert "actions" not in workflow["permissions"]
        assert "packages" not in workflow["permissions"]
