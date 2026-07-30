import builtins
import inspect
import os
import stat
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import edullm.cli as cli
import edullm.secure_publish as secure_publish
from edullm.assignment import AssignmentResult
from edullm.automation import AutomationResult
from edullm.jobs import GateConfiguration, OperatorJobSummary
from edullm.policy import Policy
from edullm.ssh import SSHSessionError


class RestrictedEnvironment(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        if key not in {"GITHUB_TOKEN", "GITHUB_REPOSITORY"}:
            raise AssertionError(f"unexpected environment read: {key}")
        return super().get(key, default)


def test_module_cli_accepts_automation_validate_and_positive_issue(tmp_path, monkeypatch, capsys):
    environment = RestrictedEnvironment(
        {
            "GITHUB_TOKEN": "secret-token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        }
    )
    calls = []

    def runner(issue_number, *, token, repository, root):
        calls.append((issue_number, token, repository, root))
        return AutomationResult("ready", (), False)

    monkeypatch.chdir(tmp_path)
    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ=environment,
        validation_runner=runner,
    )

    assert exit_code == 0
    assert calls == [(42, "secret-token", "edu-llm/OLMo-core", tmp_path)]
    assert environment.reads == ["GITHUB_TOKEN", "GITHUB_REPOSITORY"]
    output = capsys.readouterr()
    assert output.out == "eduLLM Issue #42: ready\n"
    assert "secret-token" not in output.out + output.err


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["validate", "--issue", "42"],
        ["automation", "validate"],
        ["automation", "validate", "--issue", "0"],
        ["automation", "validate", "--issue", "-1"],
        ["automation", "validate", "--issue", "1.5"],
        ["automation", "submit", "--issue", "42"],
    ],
)
def test_module_cli_rejects_missing_commands_or_nonpositive_issue(arguments):
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments, environ={})

    assert raised.value.code == 2


@pytest.mark.parametrize(
    "environment,message",
    [
        (
            {"GITHUB_REPOSITORY": "edu-llm/OLMo-core"},
            "GITHUB_TOKEN is required",
        ),
        (
            {"GITHUB_TOKEN": "token"},
            "GITHUB_REPOSITORY is required",
        ),
    ],
)
def test_cli_requires_only_expected_github_environment(environment, message, capsys):
    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ=RestrictedEnvironment(environment),
    )

    assert exit_code == 2
    output = capsys.readouterr()
    assert message in output.err


def test_cli_operational_failure_is_sanitized_and_nonzero(capsys):
    token = "top-secret-token"

    def runner(issue_number, *, token, repository, root):  # noqa: ARG001
        return AutomationResult(
            "requested",
            ("GitHub validation operation failed",),
            True,
        )

    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ={
            "GITHUB_TOKEN": token,
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        validation_runner=runner,
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert "GitHub validation operation failed" in output.err
    assert token not in output.out + output.err


def test_invalid_request_is_a_handled_requested_result(capsys):
    def runner(issue_number, *, token, repository, root):  # noqa: ARG001
        return AutomationResult(
            "requested",
            ("missing heading: Purpose",),
            False,
        )

    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        validation_runner=runner,
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.out == "eduLLM Issue #42: requested\n"


def test_default_runner_loads_only_tracked_configuration_for_canonical_repository(
    tmp_path, monkeypatch
):
    loaded = []
    fake_policy = object()
    fake_client = object()
    expected_result = AutomationResult("ready", (), False)

    def fake_load_policy(policy_path, entrypoints_path=None):
        loaded.append((policy_path, entrypoints_path))
        return fake_policy

    def fake_client_factory(token, repository):
        assert token == "token"
        assert repository == "edu-llm/OLMo-core"
        return fake_client

    def fake_validate_issue(
        issue_number,
        *,
        github,
        policy,
        validated_at,
    ):
        assert issue_number == 42
        assert github is fake_client
        assert policy is fake_policy
        assert validated_at.tzinfo is not None
        return expected_result

    monkeypatch.setattr(cli, "load_policy", fake_load_policy)
    monkeypatch.setattr(cli, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(cli, "validate_issue", fake_validate_issue)

    result = cli.automation_validate(
        42,
        token="token",
        repository="edu-llm/OLMo-core",
        root=tmp_path,
    )

    assert result is expected_result
    assert loaded == [
        (
            tmp_path / "config/edullm/policy.yaml",
            tmp_path / "config/edullm/entrypoints.yaml",
        ),
    ]


def test_cli_rejects_noncanonical_repository_before_authorization(tmp_path, monkeypatch, capsys):
    raw_repository = "attacker/fork"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_policy", lambda *args: object())
    monkeypatch.setattr(
        cli,
        "GitHubClient",
        lambda *args, **kwargs: pytest.fail("GitHub client must not be constructed"),
    )
    monkeypatch.setattr(
        cli,
        "validate_issue",
        lambda *args, **kwargs: pytest.fail("commit evidence must not be authorized"),
    )

    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": raw_repository,
        },
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "eduLLM validation configuration or GitHub access failed" in output.err
    assert raw_repository not in output.out + output.err


def test_package_registers_user_facing_edullm_console_entrypoint():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    scripts_section = pyproject.partition("[project.scripts]\n")[2].partition("\n[")[0]

    assert scripts_section.strip() == 'edullm = "edullm.cli:main"'
    assert 'requires-python = ">=3.10"' in pyproject


def test_cli_tests_do_not_import_python_311_only_tomllib():
    source = Path(__file__).read_text(encoding="utf-8")

    assert "\nimport tomllib\n" not in source


def test_main_docstring_documents_public_and_internal_commands_and_runners():
    documentation = inspect.getdoc(cli.main)

    assert documentation is not None
    assert "setup" in documentation
    assert "jobs" in documentation
    assert "logout" in documentation
    assert "automation validate" in documentation
    assert "automation assign" in documentation
    assert "automation reminders" in documentation
    assert "automation terminal" in documentation
    assert ":param validation_runner:" in documentation
    assert ":param assignment_runner:" in documentation
    assert ":param reminder_runner:" in documentation
    assert ":param terminal_runner:" in documentation
    assert "console entry point" in documentation


@pytest.mark.parametrize(
    "command,runner_name,expected_output",
    [
        (
            "assign",
            "assignment_runner",
            "eduLLM assignment scan: 1 result(s)\n",
        ),
        (
            "reminders",
            "reminder_runner",
            "eduLLM reminder scan: 1 result(s)\n",
        ),
        (
            "terminal",
            "terminal_runner",
            "eduLLM terminal notification scan: 1 result(s)\n",
        ),
    ],
)
def test_internal_cli_exposes_only_hard_disabled_assignment_automation(
    tmp_path,
    monkeypatch,
    capsys,
    command,
    runner_name,
    expected_output,
):
    calls = []

    def runner(*, token, repository, webhook, root):
        calls.append((token, repository, webhook, root))
        return (AssignmentResult(42, "assigned", "alice", False),)

    kwargs: dict[str, Any] = {
        "assignment_runner": lambda **unused: (),
        "reminder_runner": lambda **unused: (),
        "terminal_runner": lambda **unused: (),
        runner_name: runner,
    }
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        ["automation", command],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T1/B1/secret",
        },
        **kwargs,
    )

    assert exit_code == 0
    assert calls == [
        (
            "token",
            "edu-llm/OLMo-core",
            "https://hooks.slack.com/services/T1/B1/secret",
            tmp_path,
        )
    ]
    output = capsys.readouterr()
    assert output.out == expected_output
    assert "hooks.slack.com" not in output.out + output.err


@pytest.mark.parametrize("command", ["assign", "reminders", "terminal"])
def test_assignment_cli_allows_absent_webhook_for_closed_operator_roster(command, capsys):
    def runner(*, token, repository, webhook, root):  # noqa: ARG001
        assert webhook is None
        return ()

    kwargs = {
        "assignment_runner": runner,
        "reminder_runner": runner,
        "terminal_runner": runner,
    }
    exit_code = cli.main(
        ["automation", command],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        **kwargs,
    )

    assert exit_code == 0
    assert "0 result(s)" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["assign", "reminders"])
def test_assignment_cli_returns_nonzero_for_sanitized_operational_result(command, capsys):
    def runner(**unused):
        return (AssignmentResult(42, "error", None, True),)

    exit_code = cli.main(
        ["automation", command],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        assignment_runner=runner,
        reminder_runner=runner,
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert "automation operation failed" in output.err


class StatefulSetupSSH:
    def __init__(
        self,
        events,
        *,
        fail_at=None,
        env_ready=True,
        states=("COMPLETED",),
        key_exists=True,
        existing_key_valid=True,
        remote_wandb_username="wandb-user",
    ):
        self.events = events
        self.fail_at = fail_at
        self.env_ready = env_ready
        self.states = list(states)
        self.key_exists = key_exists
        self.existing_key_valid = existing_key_valid
        self.remote_wandb_username = remote_wandb_username
        self.writes = []
        self.remote_wandb_attempts = 0

    def _result(self, label, *, stdout="", returncode=0):
        self.events.append(label)
        if self.fail_at == label:
            returncode = 19
            stdout = "sensitive captured output"
        return subprocess.CompletedProcess(["ssh"], returncode, stdout, "sensitive stderr")

    def run_direct(self, username, argv, **kwargs):
        assert username == "orcd-user"
        assert argv == ["hostname"]
        assert kwargs["timeout"] > 0
        result = self._result("ssh-direct")
        if result.returncode:
            raise cli.SSHError("SSH command failed")
        return result

    def ensure_master(self):
        self.events.append("ensure-master")
        if self.fail_at == "ensure-master":
            raise SSHSessionError("ORCD SSH login failed")

    def run_remote(self, argv, *, check=True, timeout=30):
        assert timeout > 0
        command = " ".join(argv)
        if argv == ["hostname"]:
            label = "tool-hostname"
            result = self._result(label, stdout="host\n")
        elif argv == ["command", "-v", "sbatch"]:
            label = "tool-sbatch"
            result = self._result(label, stdout="/bin/sbatch\n")
        elif argv == ["command", "-v", "squeue"]:
            label = "tool-squeue"
            result = self._result(label, stdout="/bin/squeue\n")
        elif "findmnt -n -o TARGET" in command:
            label = "scratch"
            result = self._result(label)
        elif 'test -x "$EDULLM_VENV/bin/python"' in command:
            label = "env-check"
            result = self._result(label, returncode=0 if self.env_ready else 1)
        elif "sbatch --parsable" in command:
            label = "env-submit"
            result = self._result(label, stdout="12345\n")
        elif "sacct -j" in command:
            label = "sacct"
            state = self.states.pop(0) if self.states else "PENDING"
            result = self._result(label, stdout=state + "|\n")
        elif "import torch, wandb, olmo_core" in command:
            label = "imports"
            result = self._result(label)
        elif "pip freeze --all" in command:
            label = "fingerprint"
            result = self._result(label, stdout="a" * 64 + "  -\n")
        elif 'test -s "$HOME/.config/edullm/wandb.key"' in command:
            label = "key-check"
            result = self._result(label, returncode=0 if self.key_exists else 1)
        elif "api.projects(entity=" in command:
            label = "remote-wandb"
            self.remote_wandb_attempts += 1
            valid = self.existing_key_valid or self.remote_wandb_attempts > 1
            result = self._result(
                label,
                stdout=self.remote_wandb_username + "\n",
                returncode=0 if valid else 1,
            )
        else:
            raise AssertionError(f"unexpected remote command: {command}")
        if check and result.returncode:
            raise cli.SSHError("SSH command failed")
        return result

    def write_remote(self, path, content, **kwargs):
        assert kwargs["timeout"] > 0
        label = "write-key" if path.endswith("wandb.key") else "write-env"
        self.events.append(label)
        if self.fail_at == label:
            raise cli.SSHError("SSH command failed")
        self.writes.append((path, content))

    def close_master(self):
        self.events.append("logout")
        return True


class StatefulWandbAPI:
    def __init__(self, events, *, project_access=True, fail_at=None):
        self.events = events
        self.project_access = project_access
        self.fail_at = fail_at

    @property
    def viewer(self):
        self.events.append("wandb-viewer")
        if self.fail_at == "wandb-viewer":
            raise RuntimeError("sensitive W&B failure")
        return SimpleNamespace(username="wandb-user")

    def projects(self, *, entity):
        assert entity == "eduLLM"
        self.events.append("wandb-projects")
        if self.fail_at == "wandb-projects":
            raise RuntimeError("sensitive W&B failure")
        names = ["test"] if self.project_access else ["other"]
        return [SimpleNamespace(name=name) for name in names]


class SetupClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _write_operator_roster(root, github="operator"):
    config = root / "config" / "edullm"
    config.mkdir(parents=True)
    (config / "operators.yaml").write_text(
        yaml.safe_dump(
            {
                "operators": [
                    {
                        "github": github,
                        "slack_user_id": "U12345678",
                        "rotation_order": 0,
                        "enabled": True,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _setup_dependencies(
    events,
    ssh_client,
    *,
    fail_at=None,
    project_access=True,
    secret="literal-wandb-key",
    clock=None,
):
    def local_runner(argv, **kwargs):
        assert kwargs["timeout"] > 0
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        label = "gh-auth" if argv == ["gh", "auth", "status"] else "gh-login"
        events.append(label)
        return subprocess.CompletedProcess(
            argv,
            1 if fail_at == label else 0,
            "operator\n" if label == "gh-login" else "",
            "sensitive local stderr",
        )

    def api_factory():
        return StatefulWandbAPI(
            events,
            project_access=project_access,
            fail_at=fail_at,
        )

    def confirm(prompt):
        assert "~/.ssh/config" in prompt
        events.append("confirm")
        return True

    def get_secret(prompt):
        assert "W&B API key" in prompt
        events.append("secret-prompt")
        return secret

    clock = clock or SetupClock()
    return cli.SetupDependencies(
        local_runner=local_runner,
        ssh_client=ssh_client,
        wandb_api_factory=api_factory,
        confirm=confirm,
        get_secret=get_secret,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_setup_runs_checks_in_safe_alias_order_and_writes_secret_free_config(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)
    dependencies = _setup_dependencies(events, ssh_client)
    output = StringIO()
    apply_control_config = cli.apply_control_config

    def apply_and_record(path, plan):
        events.append("apply-ssh-config")
        return apply_control_config(path, plan)

    monkeypatch.setattr(cli, "apply_control_config", apply_and_record)

    result = cli.setup_operator(
        root=root,
        home=home,
        orcd_username="orcd-user",
        dependencies=dependencies,
        output=output,
    )

    assert result.github == "operator"
    assert events == [
        "gh-auth",
        "gh-login",
        "wandb-viewer",
        "wandb-projects",
        "confirm",
        "apply-ssh-config",
        "ensure-master",
        "tool-hostname",
        "tool-sbatch",
        "tool-squeue",
        "scratch",
        "env-check",
        "imports",
        "fingerprint",
        "write-env",
        "key-check",
        "remote-wandb",
    ]
    config_path = home / ".config" / "edullm" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config == {
        "environment_fingerprint": "a" * 64,
        "github": "operator",
        "orcd_username": "orcd-user",
        "remote_repo_root": "$HOME/OLMo-core",
        "scratch": "$HOME/orcd/scratch/edullm",
        "version": 1,
        "wandb_username": "wandb-user",
    }
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config_path.parent.stat().st_mode) == 0o700
    assert "literal-wandb-key" not in config_path.read_text(encoding="utf-8")
    assert "Host orcd-login" in (home / ".ssh" / "config").read_text(encoding="utf-8")
    assert "--- ~/.ssh/config" in output.getvalue()


@pytest.mark.parametrize(
    "fail_at",
    [
        "gh-auth",
        "gh-login",
        "wandb-viewer",
        "wandb-projects",
        "tool-hostname",
        "tool-sbatch",
        "tool-squeue",
        "scratch",
        "imports",
        "fingerprint",
        "write-env",
        "remote-wandb",
    ],
)
def test_setup_stops_on_first_failed_transition_and_sanitizes_failure(tmp_path, fail_at):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, fail_at=fail_at)
    dependencies = _setup_dependencies(events, ssh_client, fail_at=fail_at)

    with pytest.raises(cli.SetupError, match="setup failed") as raised:
        cli.setup_operator(
            root=root,
            home=home,
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=StringIO(),
        )

    assert events[-1] == fail_at
    assert events.count(fail_at) == 1
    assert "sensitive" not in str(raised.value)
    assert not (home / ".config" / "edullm" / "config.yaml").exists()


@pytest.mark.parametrize(
    ("fail_at", "ssh_options", "forbidden_later"),
    [
        ("env-check", {}, "imports"),
        ("env-submit", {"env_ready": False}, "sacct"),
        ("sacct", {"env_ready": False}, "imports"),
        ("key-check", {}, "remote-wandb"),
        ("write-key", {"key_exists": False}, "remote-wandb"),
    ],
)
def test_setup_late_failure_transitions_stop_without_insecure_artifacts(
    tmp_path, fail_at, ssh_options, forbidden_later
):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, fail_at=fail_at, **ssh_options)

    with pytest.raises(cli.SetupError, match="setup failed") as raised:
        cli.setup_operator(
            root=root,
            home=home,
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )

    assert events[-1] == fail_at
    assert forbidden_later not in events
    assert "sensitive" not in str(raised.value)
    assert not (home / ".config" / "edullm" / "config.yaml").exists()
    assert not any(path.endswith("wandb.key") for path, _ in ssh_client.writes)


def test_setup_requires_actual_local_wandb_project_access(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)
    dependencies = _setup_dependencies(events, ssh_client, project_access=False)

    with pytest.raises(cli.SetupError, match="eduLLM/test access"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=StringIO(),
        )

    assert events == ["gh-auth", "gh-login", "wandb-viewer", "wandb-projects"]


def test_setup_decline_does_not_modify_ssh_or_continue(tmp_path):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)
    dependencies = _setup_dependencies(events, ssh_client)

    def decline(prompt: str) -> bool:
        events.append("decline")
        return False

    dependencies.confirm = decline

    with pytest.raises(cli.SetupDeclined):
        cli.setup_operator(
            root=root,
            home=home,
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=StringIO(),
        )

    assert events[-1] == "decline"
    assert not (home / ".ssh").exists()


def test_setup_rejects_unsafe_target_block_before_output_or_confirmation(tmp_path):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir(parents=True)
    config = ssh_dir / "config"
    unsafe = (
        "Host orcd-login\n" "  Hostname old.example\n" "  SetEnv WANDB_API_KEY=never-display-this\n"
    )
    config.write_text(unsafe, encoding="utf-8")
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)
    dependencies = _setup_dependencies(events, ssh_client)
    dependencies.confirm = lambda prompt: pytest.fail("unsafe config reached confirmation")
    output = StringIO()

    with pytest.raises(cli.SetupError, match="planning SSH configuration") as raised:
        cli.setup_operator(
            root=root,
            home=home,
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=output,
        )

    assert output.getvalue() == ""
    assert "never-display-this" not in str(raised.value)
    assert config.read_text(encoding="utf-8") == unsafe
    assert "ssh-direct" not in events


def test_setup_sanitizes_confirmation_prompt_failure_without_modifying_ssh(tmp_path):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)
    dependencies = _setup_dependencies(events, ssh_client)

    def fail_confirmation(prompt):
        raise RuntimeError("sensitive prompt diagnostics")

    dependencies.confirm = fail_confirmation

    with pytest.raises(cli.SetupError, match="confirmation prompt") as raised:
        cli.setup_operator(
            root=root,
            home=home,
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=StringIO(),
        )

    assert "sensitive" not in str(raised.value)
    assert not (home / ".ssh").exists()


def test_setup_is_idempotent_and_does_not_confirm_or_backup_twice(tmp_path):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    first_events: list[str] = []
    first_ssh = StatefulSetupSSH(first_events)
    cli.setup_operator(
        root=root,
        home=home,
        orcd_username="orcd-user",
        dependencies=_setup_dependencies(first_events, first_ssh),
        output=StringIO(),
    )
    backups = list((home / ".ssh").glob("config.edullm-backup*"))
    second_events: list[str] = []
    second_ssh = StatefulSetupSSH(second_events)
    second_dependencies = _setup_dependencies(second_events, second_ssh)
    second_dependencies.confirm = lambda prompt: pytest.fail("idempotent setup asked to confirm")

    cli.setup_operator(
        root=root,
        home=home,
        orcd_username="orcd-user",
        dependencies=second_dependencies,
        output=StringIO(),
    )

    assert list((home / ".ssh").glob("config.edullm-backup*")) == backups
    assert "confirm" not in second_events
    assert "ssh-direct" not in second_events
    assert second_events.index("ensure-master") < second_events.index("tool-hostname")


def test_setup_session_bootstrap_failure_stops_before_alias_remote_checks(tmp_path):
    root = tmp_path / "checkout"
    home = tmp_path / "home"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, fail_at="ensure-master")

    with pytest.raises(SSHSessionError, match="ORCD SSH login failed") as raised:
        cli.setup_operator(
            root=root,
            home=home,
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )

    assert events[-1] == "ensure-master"
    assert "tool-hostname" not in events
    assert "sensitive" not in str(raised.value)


def test_setup_submits_reviewed_environment_script_and_polls_to_completed(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(
        events,
        env_ready=False,
        states=("PENDING", "RUNNING", "COMPLETED"),
    )
    clock = SetupClock()

    cli.setup_operator(
        root=root,
        home=tmp_path / "home",
        orcd_username="orcd-user",
        dependencies=_setup_dependencies(events, ssh_client, clock=clock),
        output=StringIO(),
    )

    assert events.index("env-check") < events.index("env-submit")
    assert events[events.index("env-submit") + 1 : events.index("imports")] == [
        "sacct",
        "sacct",
        "sacct",
    ]
    assert clock.now == 2 * cli.SETUP_POLL_INTERVAL_SECONDS


@pytest.mark.parametrize(
    "state",
    ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"],
)
def test_setup_requires_exact_completed_terminal_state(tmp_path, state):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, env_ready=False, states=(state,))

    with pytest.raises(cli.SetupError, match="did not complete"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )

    assert events[-1] == "sacct"
    assert "imports" not in events


def test_setup_polling_is_bounded(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, env_ready=False, states=())
    clock = SetupClock()

    with pytest.raises(cli.SetupError, match="timed out"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client, clock=clock),
            output=StringIO(),
            poll_timeout=10,
        )

    assert clock.now == 10


def test_setup_rejects_malformed_fingerprint(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)
    original_result = ssh_client._result

    def malformed(label, **kwargs):
        if label == "fingerprint":
            kwargs["stdout"] = "not-a-fingerprint\n"
        return original_result(label, **kwargs)

    monkeypatch.setattr(ssh_client, "_result", malformed)

    with pytest.raises(cli.SetupError, match="fingerprint"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )


def test_setup_wandb_environment_is_secret_free_and_reads_mode_0600_key(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)

    cli.setup_operator(
        root=root,
        home=tmp_path / "home",
        orcd_username="orcd-user",
        dependencies=_setup_dependencies(events, ssh_client),
        output=StringIO(),
    )

    assert ssh_client.writes == [
        (
            "~/.config/edullm/wandb.env",
            'export WANDB_API_KEY="$(cat "$HOME/.config/edullm/wandb.key")"\n'
            'export WANDB_ENTITY="eduLLM"\n'
            'export WANDB_PROJECT="test"\n',
        )
    ]
    assert "api.viewer.username" in cli.REMOTE_WANDB_CHECK_SCRIPT
    assert "api.projects(entity='eduLLM')" in cli.REMOTE_WANDB_CHECK_SCRIPT
    assert "assert 'test'" not in cli.REMOTE_WANDB_CHECK_SCRIPT


def test_setup_prompts_and_writes_key_only_when_verified_remote_key_is_absent(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, key_exists=False)
    secret = "literal-wandb-key"
    output = StringIO()

    cli.setup_operator(
        root=root,
        home=tmp_path / "home",
        orcd_username="orcd-user",
        dependencies=_setup_dependencies(events, ssh_client, secret=secret),
        output=output,
    )

    assert events[events.index("key-check") :] == [
        "key-check",
        "secret-prompt",
        "write-key",
        "remote-wandb",
    ]
    key_write = ssh_client.writes[-1]
    assert key_write == ("~/.config/edullm/wandb.key", secret + "\n")
    non_stdin_surfaces = (
        output.getvalue()
        + (tmp_path / "home" / ".config" / "edullm" / "config.yaml").read_text()
        + repr(events)
    )
    assert secret not in non_stdin_surfaces


def test_setup_sanitizes_secret_prompt_failure(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, key_exists=False)
    dependencies = _setup_dependencies(events, ssh_client)

    def fail_prompt(prompt):
        raise RuntimeError("sensitive prompt diagnostics")

    dependencies.get_secret = fail_prompt

    with pytest.raises(cli.SetupError, match="W&B key prompt") as raised:
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=StringIO(),
        )

    assert "sensitive" not in str(raised.value)


def test_setup_replaces_unverified_existing_remote_key_without_exposing_it(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, key_exists=True, existing_key_valid=False)

    cli.setup_operator(
        root=root,
        home=tmp_path / "home",
        orcd_username="orcd-user",
        dependencies=_setup_dependencies(events, ssh_client),
        output=StringIO(),
    )

    assert events[events.index("key-check") :] == [
        "key-check",
        "remote-wandb",
        "secret-prompt",
        "write-key",
        "remote-wandb",
    ]


def test_setup_does_not_prompt_when_existing_key_verifies(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, key_exists=True, existing_key_valid=True)
    dependencies = _setup_dependencies(events, ssh_client)
    dependencies.get_secret = lambda prompt: pytest.fail("unexpected W&B key prompt")

    cli.setup_operator(
        root=root,
        home=tmp_path / "home",
        orcd_username="orcd-user",
        dependencies=dependencies,
        output=StringIO(),
    )

    assert not any(path.endswith("wandb.key") for path, _ in ssh_client.writes)


def test_setup_requires_remote_wandb_identity_to_match_local_identity(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, remote_wandb_username="different-user")

    with pytest.raises(cli.SetupError, match="W&B identity"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )


def test_setup_fails_closed_when_github_login_is_not_protected_operator(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root, github="someone-else")
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)

    with pytest.raises(cli.SetupError, match="protected operator"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )

    assert not (tmp_path / "home" / ".config" / "edullm" / "config.yaml").exists()


def test_production_empty_roster_fails_closed(tmp_path):
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events)

    with pytest.raises(cli.SetupError, match="protected operator"):
        cli.setup_operator(
            root=Path.cwd(),
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )


def test_scratch_check_matches_reviewed_mount_containment_and_write_probe():
    script = cli.SCRATCH_CHECK_SCRIPT
    required = [
        'realpath -e "$EDULLM_SCRATCH_ROOT"',
        'realpath -m "$EDULLM_SCRATCH"',
        '"$RESOLVED_SCRATCH_ROOT/"*',
        'findmnt -n -o TARGET -T "$RESOLVED_SCRATCH_ROOT"',
        'findmnt -n -o TARGET -T "$HOME"',
        '"$SCRATCH_MOUNT" != "$HOME_MOUNT"',
        'mkdir -p "$EDULLM_SCRATCH"',
        'realpath -e "$EDULLM_SCRATCH"',
        'test -w "$EDULLM_SCRATCH"',
        'mktemp "$EDULLM_SCRATCH/.edullm-preflight.XXXXXX"',
        "printf '%s\\n' edullm-probe",
        'rm -f "$SCRATCH_PROBE"',
    ]
    positions = [
        script.rindex(item) if item == 'rm -f "$SCRATCH_PROBE"' else script.index(item)
        for item in required
    ]

    assert positions == sorted(positions)
    assert "$HOME/orcd/scratch/edullm" in script
    assert '"$HOME"|"$HOME"/*)' not in script


def test_environment_commands_use_reviewed_checkout_and_sorted_freeze():
    assert "$HOME/OLMo-core/src/scripts/orcd/setup_env.sbatch" in cli.SUBMIT_ENV_SCRIPT
    assert "git -C" in cli.SUBMIT_ENV_SCRIPT
    assert "status --porcelain" in cli.SUBMIT_ENV_SCRIPT
    assert "*[!0-9a-f]*" in cli.SUBMIT_ENV_SCRIPT
    assert 'test "${#EDULLM_COMMIT_SHA}" -eq 40' in cli.SUBMIT_ENV_SCRIPT
    assert "sbatch --parsable" in cli.SUBMIT_ENV_SCRIPT
    assert "python -m pip freeze --all | LC_ALL=C sort | sha256sum" in cli.FINGERPRINT_SCRIPT
    assert "import torch, wandb, olmo_core" in cli.IMPORT_CHECK_SCRIPT


def test_environment_readiness_requires_remote_private_write_helper():
    setup_script = Path("src/scripts/orcd/setup_env.sbatch").read_text(encoding="utf-8")

    assert 'python" -c "import edullm.ssh_helper"' in cli.ENVIRONMENT_CHECK_SCRIPT
    assert 'git -C "$EDULLM_REPO_ROOT" rev-parse HEAD' in cli.ENVIRONMENT_CHECK_SCRIPT
    assert '"$EDULLM_VENV/.edullm-commit"' in cli.ENVIRONMENT_CHECK_SCRIPT
    assert "import torch, wandb, olmo_core, edullm.ssh_helper" in cli.IMPORT_CHECK_SCRIPT
    assert "\nimport edullm.ssh_helper\n" in setup_script
    assert 'printf \'%s\\n\' "$EDULLM_COMMIT_SHA" > "$EDULLM_VENV/.edullm-commit"' in setup_script


@pytest.mark.parametrize(
    ("failing_tool", "forbidden_later"),
    [
        ("tool-hostname", "tool-sbatch"),
        ("tool-sbatch", "tool-squeue"),
        ("tool-squeue", "scratch"),
    ],
)
def test_setup_remote_tool_checks_fail_immediately_in_order(
    tmp_path, failing_tool, forbidden_later
):
    root = tmp_path / "checkout"
    _write_operator_roster(root)
    events: list[str] = []
    ssh_client = StatefulSetupSSH(events, fail_at=failing_tool)

    with pytest.raises(cli.SetupError, match="setup failed"):
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=_setup_dependencies(events, ssh_client),
            output=StringIO(),
        )

    assert events[-1] == failing_tool
    assert forbidden_later not in events


def _run_submit_env_script(tmp_path, status_mode):
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setup_script = home / "OLMo-core" / "src" / "scripts" / "orcd" / "setup_env.sbatch"
    setup_script.parent.mkdir(parents=True)
    setup_script.write_text("#!/bin/bash\n", encoding="utf-8")
    sbatch_log = tmp_path / "sbatch.log"
    (fake_bin / "git").write_text(
        """#!/bin/bash
case "$*" in
  *"rev-parse HEAD"*) printf '%040d\n' 0 ;;
  *"status --porcelain"*)
    case "$GIT_STATUS_MODE" in
      fail) exit 17 ;;
      tracked) printf ' M tracked.py\n' ;;
      untracked) printf '?? untracked.py\n' ;;
      clean) exit 0 ;;
    esac
    ;;
esac
""",
        encoding="utf-8",
    )
    (fake_bin / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" > "$SBATCH_LOG"\nprintf "12345\\n"\n',
        encoding="utf-8",
    )
    (fake_bin / "git").chmod(0o755)
    (fake_bin / "sbatch").chmod(0o755)
    environment = {
        **os.environ,
        "GIT_STATUS_MODE": status_mode,
        "HOME": str(home),
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "SBATCH_LOG": str(sbatch_log),
    }
    result = subprocess.run(
        ["bash", "-c", cli.SUBMIT_ENV_SCRIPT],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
        timeout=10,
    )
    return result, sbatch_log


@pytest.mark.parametrize("status_mode", ["fail", "tracked", "untracked"])
def test_environment_submit_rejects_failed_or_dirty_git_status(tmp_path, status_mode):
    result, sbatch_log = _run_submit_env_script(tmp_path, status_mode)

    assert result.returncode != 0
    assert not sbatch_log.exists()


def test_environment_submit_accepts_clean_git_status(tmp_path):
    result, sbatch_log = _run_submit_env_script(tmp_path, "clean")

    assert result.returncode == 0
    assert sbatch_log.exists()


def test_write_operator_config_rejects_symlink_and_preserves_target(tmp_path):
    config = tmp_path / "config.yaml"
    target = tmp_path / "target"
    target.write_text("unchanged", encoding="utf-8")
    config.symlink_to(target)

    with pytest.raises(cli.SetupError, match="operator config"):
        cli.write_operator_config(config, {"version": 1})

    assert target.read_text(encoding="utf-8") == "unchanged"


def test_write_operator_config_rejects_symlinked_ancestor(tmp_path):
    real_config = tmp_path / "real-config"
    real_config.mkdir()
    linked_config = tmp_path / ".config"
    linked_config.symlink_to(real_config, target_is_directory=True)

    with pytest.raises(cli.SetupError, match="operator config"):
        cli.write_operator_config(linked_config / "edullm" / "config.yaml", {"version": 1})

    assert not (real_config / "edullm").exists()


def test_write_operator_config_is_atomic_and_mode_0600(tmp_path, monkeypatch):
    config = tmp_path / "config" / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    original = config.read_bytes()

    real_link = secure_publish.os.link

    def fail_publish(source, target, **kwargs):
        if target == config.name and source.startswith(f".{config.name}.edullm-"):
            raise OSError("simulated failure")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", fail_publish)
    with pytest.raises(cli.SetupError, match="operator config"):
        cli.write_operator_config(config, {"version": 2})

    assert config.read_bytes() == original
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert not list(config.parent.glob(".config.yaml.edullm-*"))


def test_write_operator_config_rejects_unsafe_existing_mode_before_replace(tmp_path):
    config = tmp_path / "config" / "config.yaml"
    config.parent.mkdir()
    config.write_text("version: 1\n", encoding="utf-8")
    config.chmod(0o644)

    with pytest.raises(cli.SetupError, match="operator config"):
        cli.write_operator_config(config, {"version": 2})

    assert config.read_text(encoding="utf-8") == "version: 1\n"
    assert stat.S_IMODE(config.stat().st_mode) == 0o644
    assert not list(config.parent.glob(".config.yaml.edullm-*"))


def test_write_operator_config_commits_with_same_open_directory_fd(tmp_path, monkeypatch):
    config = tmp_path / "config" / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    real_link = secure_publish.os.link
    calls = []

    def record_publish(source, target, **kwargs):
        if target == config.name and source.startswith(f".{config.name}.edullm-"):
            calls.append(((source, target), kwargs))
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", record_publish)

    cli.write_operator_config(config, {"version": 2})

    assert len(calls) == 1
    _, options = calls[0]
    assert options["src_dir_fd"] == options["dst_dir_fd"]
    assert yaml.safe_load(config.read_text(encoding="utf-8")) == {"version": 2}


def test_write_operator_config_establishes_exact_mode_before_write_under_umask(
    tmp_path, monkeypatch
):
    config = tmp_path / "home" / ".config" / "edullm" / "config.yaml"
    config.parents[2].mkdir()
    observed_modes = []
    real_write = cli._write_descriptor

    def record_mode(descriptor, content):
        observed_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        real_write(descriptor, content)

    monkeypatch.setattr(cli, "_write_descriptor", record_mode)
    previous_umask = os.umask(0o777)
    try:
        cli.write_operator_config(config, {"version": 1})
    finally:
        os.umask(previous_umask)

    assert observed_modes == [0o600]
    assert stat.S_IMODE(config.stat().st_mode) == 0o600


def test_write_operator_config_restores_edit_at_actual_publish_boundary(tmp_path, monkeypatch):
    config = tmp_path / "config" / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    concurrent_fd = os.open(config, os.O_WRONLY)
    real_link = secure_publish.os.link

    def edit_then_publish(source, target, **kwargs):
        if target == config.name and source.startswith(f".{config.name}.edullm-"):
            os.lseek(concurrent_fd, 0, os.SEEK_SET)
            os.write(concurrent_fd, b"version: concurrent\n")
            os.ftruncate(concurrent_fd, len(b"version: concurrent\n"))
            os.fsync(concurrent_fd)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", edit_then_publish)
    try:
        with pytest.raises(cli.SetupError, match="operator config"):
            cli.write_operator_config(config, {"version": 2})
    finally:
        os.close(concurrent_fd)

    assert config.read_bytes() == b"version: concurrent\n"
    assert not list(config.parent.glob(f".{config.name}.edullm-*"))


def test_write_operator_config_never_clobbers_boundary_path_replacement(tmp_path, monkeypatch):
    config = tmp_path / "config" / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    original = config.read_bytes()
    real_link = secure_publish.os.link
    replaced = False

    def replace_then_publish(source, target, **kwargs):
        nonlocal replaced
        if target == config.name and source.startswith(f".{config.name}.edullm-") and not replaced:
            replaced = True
            descriptor = os.open(
                config.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=kwargs["dst_dir_fd"],
            )
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, b"version: concurrent\n")
            os.close(descriptor)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", replace_then_publish)

    with pytest.raises(cli.SetupError, match="operator config"):
        cli.write_operator_config(config, {"version": 2})

    assert config.read_bytes() == b"version: concurrent\n"
    recovery = list(config.parent.glob(f".{config.name}.edullm-recovery-*/original"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == original


def test_write_operator_config_rejects_parent_swap_at_actual_publish_boundary(
    tmp_path, monkeypatch
):
    parent = tmp_path / "config"
    config = parent / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    original = config.read_bytes()
    moved = tmp_path / "config-moved"
    real_link = secure_publish.os.link
    swapped = False

    def swap_then_publish(source, target, **kwargs):
        nonlocal swapped
        if target == config.name and source.startswith(f".{config.name}.edullm-") and not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", swap_then_publish)

    with pytest.raises(cli.SetupError, match="operator config"):
        cli.write_operator_config(config, {"version": 2})

    assert not config.exists()
    assert (moved / config.name).read_bytes() == original
    assert not list(moved.glob(f".{config.name}.edullm-*"))


def test_write_operator_config_rejects_parent_swap_and_cleans_old_temp(tmp_path, monkeypatch):
    parent = tmp_path / "config"
    config = parent / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    moved = tmp_path / "config-moved"
    real_write = cli._write_descriptor
    swapped = False

    def swap_parent_after_write(descriptor, content):
        nonlocal swapped
        real_write(descriptor, content)
        if not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()

    monkeypatch.setattr(cli, "_write_descriptor", swap_parent_after_write)

    with pytest.raises(cli.SetupError, match="changed"):
        cli.write_operator_config(config, {"version": 2})

    assert yaml.safe_load((moved / "config.yaml").read_text(encoding="utf-8")) == {"version": 1}
    assert not list(moved.glob(".config.yaml.edullm-*"))
    assert not config.exists()


def test_write_operator_config_rejects_target_edit_after_validation(tmp_path, monkeypatch):
    config = tmp_path / "config" / "config.yaml"
    cli.write_operator_config(config, {"version": 1})
    real_write = cli._write_descriptor
    edited = False

    def edit_target_after_write(descriptor, content):
        nonlocal edited
        real_write(descriptor, content)
        if not edited:
            edited = True
            config.write_text("version: concurrent\n", encoding="utf-8")
            config.chmod(0o600)

    monkeypatch.setattr(cli, "_write_descriptor", edit_target_after_write)

    with pytest.raises(cli.SetupError, match="changed"):
        cli.write_operator_config(config, {"version": 2})

    assert config.read_text(encoding="utf-8") == "version: concurrent\n"
    assert not list(config.parent.glob(".config.yaml.edullm-*"))


def test_local_subprocess_timeout_is_sanitized(tmp_path):
    root = tmp_path / "checkout"
    _write_operator_roster(root)

    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="sensitive output")

    dependencies = cli.SetupDependencies(
        local_runner=timed_out,
        ssh_client=StatefulSetupSSH([]),
        wandb_api_factory=lambda: pytest.fail("must stop before W&B"),
        confirm=lambda prompt: True,
        get_secret=lambda prompt: pytest.fail("must not prompt"),
        sleep=lambda seconds: None,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(cli.SetupError, match="setup failed") as raised:
        cli.setup_operator(
            root=root,
            home=tmp_path / "home",
            orcd_username="orcd-user",
            dependencies=dependencies,
            output=StringIO(),
        )

    assert "sensitive" not in str(raised.value)


def test_slurm_state_parser_normalizes_reason_but_preserves_nonexact_state():
    assert cli._parse_slurm_state("CANCELLED by 12345|\n") == "CANCELLED"
    assert cli._parse_slurm_state("COMPLETED+|\n") == "COMPLETED+"
    assert cli._parse_slurm_state("\n|\n") == ""


def test_commands_are_plain_language_and_internal_automation_remains_available():
    parser = cli.build_parser()
    text = parser.format_help()

    for command in ("setup", "jobs", "run", "logs", "stop", "logout"):
        assert command in text
    for forbidden in ("claim", "run-next", "sync", "tail"):
        assert forbidden not in text
    assert "automation" not in text
    automation = parser.parse_args(["automation", "validate", "--issue", "42"])
    assert automation.command == "automation"
    assert automation.automation_command == "validate"
    assert automation.issue == 42


def test_public_parser_exposes_exact_commands_and_hides_automation_metavar():
    parser = cli.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")

    assert command_action.metavar == "{setup,jobs,run,logs,stop,logout}"
    assert "automation" not in parser.format_usage()
    assert command_action.choices is not None
    assert set(command_action.choices) == {
        "setup",
        "jobs",
        "run",
        "logs",
        "stop",
        "logout",
        "automation",
    }


def test_handle_run_calls_run_assigned_without_manual_confirmation(monkeypatch, capsys):
    calls = []
    configuration = GateConfiguration(
        policy=Policy(wandb_entity="eduLLM", allowed_wandb_projects=("test",)),
        operators=(),
        digest="a" * 64,
    )
    services = cli.OperatorServices(
        operator="operator",
        remote_user="orcd-user",
        github=object(),
        root=Path.cwd(),
        remote=object(),
        slurm=object(),
    )
    state = SimpleNamespace(
        issue=42,
        attempts=(
            SimpleNamespace(
                slurm_job_id="12345",
                wandb_url="https://wandb.ai/eduLLM/test/runs/issue-42-attempt-1-12345",
            ),
        ),
    )

    def run_assigned(**kwargs):
        calls.append(kwargs)
        return state

    monkeypatch.setattr(cli, "run_assigned", run_assigned)
    monkeypatch.setattr(cli, "load_gate_configuration", lambda root: configuration)

    assert cli.handle_run(services=services) == 0
    assert len(calls) == 1
    assert calls[0]["operator"] == "operator"
    assert calls[0]["github"] is services.github
    assert calls[0]["load_configuration"]() is configuration
    assert calls[0]["remote"] is services.remote
    assert "Submitted Issue #42" in capsys.readouterr().out


def test_operator_services_reject_matching_login_with_bot_actor_type(monkeypatch):
    configuration = SimpleNamespace(
        operators=(SimpleNamespace(github="operator", enabled=True),),
    )
    document = {
        "environment_fingerprint": "a" * 64,
        "github": "operator",
        "orcd_username": "operator",
        "remote_repo_root": "$HOME/OLMo-core",
        "scratch": "$HOME/orcd/scratch/edullm",
        "version": 1,
        "wandb_username": "operator",
    }
    local_results = iter(
        [
            subprocess.CompletedProcess([], 0, "operator\n", ""),
            subprocess.CompletedProcess([], 0, "token\n", ""),
        ]
    )

    class BotClient:
        def __init__(self, token, repository):
            assert token == "token"
            assert repository == "edu-llm/OLMo-core"

        @staticmethod
        def get(path):
            assert path == "/user"
            return {"login": "operator", "type": "Bot"}

    monkeypatch.setattr(cli, "load_gate_configuration", lambda root: configuration)
    monkeypatch.setattr(cli, "_read_operator_document", lambda path: document)
    monkeypatch.setattr(cli, "_run_local", lambda *args: next(local_results))
    monkeypatch.setattr(cli, "GitHubClient", BotClient)
    monkeypatch.setattr(cli, "SSHClient", lambda: object())

    with pytest.raises(cli.JobOperationError, match="does not match"):
        cli._load_operator_services()


def test_operator_services_carry_private_orcd_user_separately_from_github_login(monkeypatch):
    configuration = SimpleNamespace(
        operators=(SimpleNamespace(github="github-operator", enabled=True),),
    )
    document = {
        "environment_fingerprint": "a" * 64,
        "github": "github-operator",
        "orcd_username": "orcd-user",
        "remote_repo_root": "$HOME/OLMo-core",
        "scratch": "$HOME/orcd/scratch/edullm",
        "version": 1,
        "wandb_username": "wandb-user",
    }
    local_results = iter(
        [
            subprocess.CompletedProcess([], 0, "github-operator\n", ""),
            subprocess.CompletedProcess([], 0, "token\n", ""),
        ]
    )
    captured = {}
    events = []

    class HumanClient:
        def __init__(self, token, repository):
            assert token == "token"
            assert repository == "edu-llm/OLMo-core"

        @staticmethod
        def get(path):
            assert path == "/user"
            return {"login": "github-operator", "type": "User"}

    class SubmissionRemote:
        def __init__(self, ssh_client, *, remote_user):
            events.append("remote")
            captured["ssh_client"] = ssh_client
            captured["remote_user"] = remote_user

    class Slurm:
        def __init__(self, ssh_client):
            events.append("slurm")
            assert ssh_client is session_client

    class SessionClient:
        @staticmethod
        def ensure_master():
            events.append("ensure-master")

    session_client = SessionClient()
    monkeypatch.setattr(cli, "load_gate_configuration", lambda root: configuration)
    monkeypatch.setattr(cli, "_read_operator_document", lambda path: document)
    monkeypatch.setattr(cli, "_run_local", lambda *args: next(local_results))
    monkeypatch.setattr(cli, "GitHubClient", HumanClient)
    monkeypatch.setattr(cli, "SSHClient", lambda: session_client)
    monkeypatch.setattr(cli, "SSHSubmissionRemote", SubmissionRemote)
    monkeypatch.setattr(cli, "SSHSlurm", Slurm)

    services = cli._load_operator_services()

    assert services.operator == "github-operator"
    assert services.remote_user == "orcd-user"
    assert captured == {"ssh_client": session_client, "remote_user": "orcd-user"}
    assert events == ["ensure-master", "remote", "slurm"]


def test_public_parser_has_complete_arguments_without_task_7_behavior():
    parser = cli.build_parser()

    assert parser.parse_args(["setup"]).command == "setup"
    jobs = parser.parse_args(["jobs", "--mine"])
    assert jobs.command == "jobs"
    assert jobs.mine is True
    assert parser.parse_args(["run"]).command == "run"
    assert parser.parse_args(["logs", "42"]).issue == 42
    assert parser.parse_args(["stop", "42"]).issue == 42
    assert parser.parse_args(["logout"]).command == "logout"


def test_format_operator_jobs_renders_mixed_groups_exactly():
    summaries = (
        OperatorJobSummary(20, "assigned", None, None),
        OperatorJobSummary(
            12,
            "completed",
            "18653501",
            "https://wandb.ai/eduLLM/test/runs/issue-12-attempt-1-18653501",
        ),
    )

    assert cli._format_operator_jobs("operator", summaries) == (
        "eduLLM jobs for operator\n"
        "\n"
        "Ready to run (1)\n"
        "  #20  assigned\n"
        "       Next: edullm run\n"
        "\n"
        "Submitted and recent (1)\n"
        "  #12  completed   Slurm 18653501\n"
        "       W&B: https://wandb.ai/eduLLM/test/runs/issue-12-attempt-1-18653501"
    )


def test_format_operator_jobs_identifies_only_lowest_assigned_issue_as_next_exactly():
    summaries = (
        OperatorJobSummary(23, "assigned", None, None),
        OperatorJobSummary(20, "assigned", None, None),
    )

    assert cli._format_operator_jobs("operator", summaries) == (
        "eduLLM jobs for operator\n"
        "\n"
        "Ready to run (2)\n"
        "  #20  assigned\n"
        "       Next: edullm run\n"
        "  #23  assigned\n"
        "       Waiting behind #20\n"
        "\n"
        "Submitted and recent (0)"
    )


def test_format_operator_jobs_aligns_mixed_long_issue_ids_exactly():
    summaries = (
        OperatorJobSummary(20, "assigned", None, None),
        OperatorJobSummary(
            1234,
            "completed",
            "18653501",
            "https://wandb.ai/eduLLM/test/runs/issue-1234-attempt-1-18653501",
        ),
    )

    assert cli._format_operator_jobs("operator", summaries) == (
        "eduLLM jobs for operator\n"
        "\n"
        "Ready to run (1)\n"
        "  #20   assigned\n"
        "        Next: edullm run\n"
        "\n"
        "Submitted and recent (1)\n"
        "  #1234 completed   Slurm 18653501\n"
        "        W&B: https://wandb.ai/eduLLM/test/runs/issue-1234-attempt-1-18653501"
    )


def test_format_operator_jobs_renders_empty_ready_group_exactly():
    summaries = (
        OperatorJobSummary(
            12,
            "completed",
            "18653501",
            "https://wandb.ai/eduLLM/test/runs/issue-12-attempt-1-18653501",
        ),
    )

    assert cli._format_operator_jobs("operator", summaries) == (
        "eduLLM jobs for operator\n"
        "\n"
        "Ready to run (0)\n"
        "\n"
        "Submitted and recent (1)\n"
        "  #12  completed   Slurm 18653501\n"
        "       W&B: https://wandb.ai/eduLLM/test/runs/issue-12-attempt-1-18653501"
    )


def test_format_operator_jobs_renders_no_jobs_message_only():
    assert cli._format_operator_jobs("operator", ()) == "No eduLLM jobs assigned to operator."


def test_handle_jobs_prints_formatted_operator_summaries(monkeypatch, capsys):
    summaries = (OperatorJobSummary(20, "assigned", None, None),)
    calls = []
    services = cli.OperatorServices(
        operator="operator",
        remote_user="orcd-user",
        github=object(),
        root=Path.cwd(),
        remote=object(),
        slurm=object(),
    )

    def list_jobs(**kwargs):
        calls.append(kwargs)
        return summaries

    monkeypatch.setattr(cli, "list_operator_jobs", list_jobs)
    monkeypatch.setattr(cli, "load_gate_configuration", lambda root: object())

    assert cli.handle_jobs(True, services=services) == 0
    assert len(calls) == 1
    assert calls[0]["mine"] is True
    assert calls[0]["operator"] == "operator"
    assert capsys.readouterr().out == (
        "eduLLM jobs for operator\n"
        "\n"
        "Ready to run (1)\n"
        "  #20  assigned\n"
        "       Next: edullm run\n"
        "\n"
        "Submitted and recent (0)\n"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["jobs", "extra"],
        ["run", "extra"],
        ["logs"],
        ["logs", "0"],
        ["stop", "-1"],
        ["claim"],
        ["run-next"],
        ["sync"],
        ["tail", "42"],
    ],
)
def test_public_parser_rejects_invalid_or_unpublished_surface(arguments):
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(arguments)

    assert raised.value.code == 2


def test_main_dispatches_each_public_command_to_focused_handler(monkeypatch):
    calls: list[tuple[object, ...]] = []

    def handle_setup(orcd_username=None) -> int:
        calls.append(("setup", orcd_username))
        return 11

    def handle_jobs(mine: bool) -> int:
        calls.append(("jobs", mine))
        return 12

    def handle_run() -> int:
        calls.append(("run",))
        return 13

    def handle_logs(issue: int) -> int:
        calls.append(("logs", issue))
        return 14

    def handle_stop(issue: int) -> int:
        calls.append(("stop", issue))
        return 15

    def handle_logout() -> int:
        calls.append(("logout",))
        return 16

    monkeypatch.setattr(cli, "handle_setup", handle_setup)
    monkeypatch.setattr(cli, "handle_jobs", handle_jobs)
    monkeypatch.setattr(cli, "handle_run", handle_run)
    monkeypatch.setattr(cli, "handle_logs", handle_logs)
    monkeypatch.setattr(cli, "handle_stop", handle_stop)
    monkeypatch.setattr(cli, "handle_logout", handle_logout)

    assert cli.main(["setup"], environ=RestrictedEnvironment({})) == 11
    assert cli.main(["jobs", "--mine"], environ=RestrictedEnvironment({})) == 12
    assert cli.main(["run"], environ=RestrictedEnvironment({})) == 13
    assert cli.main(["logs", "42"], environ=RestrictedEnvironment({})) == 14
    assert cli.main(["stop", "42"], environ=RestrictedEnvironment({})) == 15
    assert cli.main(["logout"], environ=RestrictedEnvironment({})) == 16
    assert calls == [
        ("setup", None),
        ("jobs", True),
        ("run",),
        ("logs", 42),
        ("stop", 42),
        ("logout",),
    ]


@pytest.mark.parametrize(
    "arguments,command",
    [
        (["jobs"], "jobs"),
        (["jobs", "--mine"], "jobs"),
        (["run"], "run"),
        (["logs", "42"], "logs"),
        (["stop", "42"], "stop"),
    ],
)
def test_task_7_commands_fail_closed_with_empty_protected_roster(
    arguments,
    command,
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(cli, "load_gate_configuration", lambda root: SimpleNamespace(operators=()))

    exit_code = cli.main(arguments, environ=RestrictedEnvironment({}))

    assert exit_code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert f"edullm {command} failed" in output.err
    assert "Task 7" not in output.err


def test_setup_handler_reports_decline_without_traceback(monkeypatch, capsys):
    def declined(**kwargs):
        raise cli.SetupDeclined("declined")

    monkeypatch.setattr(cli, "setup_operator", declined)

    assert cli.handle_setup() == 2
    output = capsys.readouterr()
    assert "cancelled" in output.err
    assert "declined" not in output.err


def test_setup_handler_sanitizes_operational_failure(monkeypatch, capsys):
    def failed(**kwargs):
        raise cli.SetupError("private implementation detail")

    monkeypatch.setattr(cli, "setup_operator", failed)

    assert cli.handle_setup() == 1
    output = capsys.readouterr()
    assert "operator setup failed" in output.err
    assert "private implementation detail" not in output.err


def test_setup_handler_reports_missing_local_wandb_dependency_action(monkeypatch, capsys):
    original_import = builtins.__import__

    def import_without_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ModuleNotFoundError("sensitive import diagnostics", name=name)
        return original_import(name, *args, **kwargs)

    def setup_with_missing_wandb(**kwargs):
        cli._verify_local_wandb(cli._default_wandb_api)

    monkeypatch.setattr(builtins, "__import__", import_without_wandb)
    monkeypatch.setattr(cli, "setup_operator", setup_with_missing_wandb)

    assert cli.handle_setup() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "python -m pip install -e '.[wandb]'" in output.err
    assert "sensitive import diagnostics" not in output.err


def test_setup_handler_reports_session_bootstrap_recovery(monkeypatch, capsys):
    def failed(**kwargs):
        raise SSHSessionError("private SSH diagnostics")

    monkeypatch.setattr(cli, "setup_operator", failed)

    assert cli.handle_setup("orcd-user") == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "ORCD SSH login failed; run ssh -MNf orcd-login and retry.\n"
    assert "private SSH diagnostics" not in output.err


def test_setup_handler_reports_ssh_configuration_planning_stage(monkeypatch, capsys):
    def failed(**kwargs):
        raise cli.SetupError("operator setup failed while planning SSH configuration")

    monkeypatch.setattr(cli, "setup_operator", failed)

    assert cli.handle_setup("orcd-user") == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "failed while planning SSH configuration" in output.err


@pytest.mark.parametrize(
    "handler",
    [
        lambda: cli.handle_jobs(False),
        cli.handle_run,
        lambda: cli.handle_logs(42),
        lambda: cli.handle_stop(42),
    ],
)
def test_operator_commands_report_fixed_session_bootstrap_recovery(handler, monkeypatch, capsys):
    def failed():
        raise SSHSessionError("private SSH diagnostics")

    monkeypatch.setattr(cli, "_load_operator_services", failed)

    assert handler() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "ORCD SSH login failed; run ssh -MNf orcd-login and retry.\n"
    assert "private SSH diagnostics" not in output.err


@pytest.mark.parametrize(
    ("handler", "generic_message"),
    [
        (lambda: cli.handle_jobs(False), "edullm jobs failed\n"),
        (cli.handle_run, "edullm run failed before a safe submission could be confirmed\n"),
        (lambda: cli.handle_logs(42), "edullm logs failed\n"),
        (lambda: cli.handle_stop(42), "edullm stop failed\n"),
    ],
)
def test_operator_commands_keep_other_failures_generic(
    handler, generic_message, monkeypatch, capsys
):
    def failed():
        raise cli.JobOperationError("private non-session diagnostics")

    monkeypatch.setattr(cli, "_load_operator_services", failed)

    assert handler() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == generic_message
    assert "private non-session diagnostics" not in output.err
    assert "ssh -MNf" not in output.err


def test_setup_cli_passes_explicit_orcd_username_to_handler(monkeypatch):
    calls = []

    def handle_setup(orcd_username=None):
        calls.append(orcd_username)
        return 17

    monkeypatch.setattr(cli, "handle_setup", handle_setup)

    assert (
        cli.main(
            ["setup", "--orcd-username", "orcd-user"],
            environ=RestrictedEnvironment({}),
        )
        == 17
    )
    assert calls == ["orcd-user"]


def test_setup_cli_rejects_invalid_orcd_username_before_handler(monkeypatch):
    monkeypatch.setattr(
        cli,
        "handle_setup",
        lambda orcd_username=None: pytest.fail("invalid ORCD username reached handler"),
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(
            ["setup", "--orcd-username", "invalid username"],
            environ=RestrictedEnvironment({}),
        )

    assert raised.value.code == 2


def test_logout_closes_only_project_master_and_reports_closed(capsys):
    class Client:
        def __init__(self):
            self.calls = 0

        def close_master(self):
            self.calls += 1
            return True

    client = Client()

    assert cli.handle_logout(ssh_client=client) == 0
    assert client.calls == 1
    assert "closed" in capsys.readouterr().out


def test_logout_handles_already_closed_master_cleanly(capsys):
    class Client:
        @staticmethod
        def close_master():
            return False

    assert cli.handle_logout(ssh_client=Client()) == 0
    assert "already closed" in capsys.readouterr().out


def test_logout_does_not_hide_other_sanitized_errors(capsys):
    class Client:
        @staticmethod
        def close_master():
            raise cli.SSHError("private SSH diagnostics")

    assert cli.handle_logout(ssh_client=Client()) == 1
    output = capsys.readouterr()
    assert "could not close" in output.err
    assert "private SSH diagnostics" not in output.err
