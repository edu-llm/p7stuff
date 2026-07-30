import io
import os
import pty
import select
import shlex
import signal
import stat
import subprocess
import sys
import termios
from pathlib import Path

import pytest

import edullm.secure_publish as secure_publish
import edullm.ssh as ssh


class RecordingRunner:
    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return subprocess.CompletedProcess(argv, 0, "", "")


class RecordingLoginRunner:
    def __init__(self, result=0):
        self.calls = 0
        self.result = result

    def __call__(self):
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class RecordingProcess:
    def __init__(self, wait_results):
        self.wait_results = list(wait_results)
        self.wait_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        result = self.wait_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


def test_control_block_uses_project_alias_and_one_hour_persist():
    block = ssh.control_block("philote")

    assert block == "\n".join(
        [
            "Host orcd-login",
            "    Hostname orcd-login.mit.edu",
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/edullm-%C",
            "    ControlPersist 1h",
            "    User philote",
        ]
    )


@pytest.mark.parametrize("username", ["", "two words", "name;touch /tmp/x", "-option"])
def test_control_block_rejects_unsafe_usernames(username):
    with pytest.raises(ValueError, match="username"):
        ssh.control_block(username)


def test_remote_argv_is_shell_quoted_once_and_has_finite_timeout():
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    client.run_remote(["python", "-c", "print('two words; $HOME')"])

    argv, options = runner.calls[0]
    assert argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "orcd-login",
        "python -c 'print('\"'\"'two words; $HOME'\"'\"')'",
    ]
    assert options["timeout"] == ssh.COMMAND_TIMEOUT_SECONDS
    assert options["capture_output"] is True
    assert options["text"] is True
    assert options["check"] is False


def test_direct_check_does_not_depend_on_project_alias():
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    client.run_direct("philote", ["hostname"])

    argv, _ = runner.calls[0]
    assert argv == [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-l",
        "philote",
        "orcd-login.mit.edu",
        "hostname",
    ]
    assert "orcd-login" not in argv[:-1]


def test_write_remote_sends_file_content_only_via_stdin():
    secret = "literal-super-secret"
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    client.write_remote("~/.config/edullm/wandb.key", secret + "\n")

    argv, options = runner.calls[0]
    assert secret not in repr(argv)
    assert options["input"] == secret + "\n"
    assert options["timeout"] == ssh.COMMAND_TIMEOUT_SECONDS
    assert argv[:4] == ["ssh", "-o", "BatchMode=yes", "orcd-login"]
    assert shlex.split(argv[4]) == [
        "$HOME/venvs/edullm/bin/python",
        "-m",
        "edullm.ssh_helper",
        "--target",
        "wandb.key",
    ]
    assert "cat >" not in argv[4]


def test_write_remote_rejects_paths_outside_fixed_private_targets():
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    with pytest.raises(ValueError, match="remote path"):
        client.write_remote("~/.ssh/authorized_keys", "content")

    assert runner.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["ssh", "orcd-login"], 30, output="private-output"),
        OSError("private local path"),
    ],
)
def test_subprocess_failures_are_sanitized(failure):
    runner = RecordingRunner([failure])
    client = ssh.SSHClient(runner=runner)

    with pytest.raises(ssh.SSHError, match="SSH command failed") as raised:
        client.run_remote(["printf", "private-argument"])

    text = str(raised.value)
    assert "private-output" not in text
    assert "private local path" not in text
    assert "private-argument" not in text


def test_nonzero_remote_failure_is_sanitized():
    result = subprocess.CompletedProcess(
        ["ssh"], 23, "captured-private-stdout", "captured-private-stderr"
    )
    client = ssh.SSHClient(runner=RecordingRunner([result]))

    with pytest.raises(ssh.SSHError, match="SSH command failed") as raised:
        client.run_remote(["false"])

    assert "private" not in str(raised.value)


def test_interactive_login_uses_fixed_argv_inherited_stdin_and_devnull_output():
    process = RecordingProcess([0])
    calls = []

    def popen_factory(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return process

    result = ssh._run_interactive_login(popen_factory=popen_factory)

    assert result == 0
    assert calls == [
        (
            ["ssh", "-MNf", "orcd-login"],
            {
                "stdin": None,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            },
        )
    ]
    assert process.wait_calls == [ssh.LOGIN_TIMEOUT_SECONDS]


def test_interactive_login_discards_diagnostics_and_returns_only_status(capfd):
    secret = "private-ssh-diagnostic"

    def popen_factory(argv, **kwargs):
        assert argv == ["ssh", "-MNf", "orcd-login"]
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"print({secret!r}); "
                    f"print({secret!r}, file=sys.stderr); "
                    "raise SystemExit(23)"
                ),
            ],
            **kwargs,
        )

    result = ssh._run_interactive_login(popen_factory=popen_factory)

    output = capfd.readouterr()
    assert result == 23
    assert output.out == ""
    assert output.err == ""
    assert secret not in repr(result)


def test_interactive_login_timeout_terminates_then_kills_and_reaps():
    command = ["ssh", "-MNf", "orcd-login"]
    process = RecordingProcess(
        [
            subprocess.TimeoutExpired(command, ssh.LOGIN_TIMEOUT_SECONDS),
            subprocess.TimeoutExpired(command, ssh.LOGIN_TERMINATE_GRACE_SECONDS),
            -9,
        ]
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        ssh._run_interactive_login(popen_factory=lambda argv, **kwargs: process)

    assert raised.value.output is None
    assert raised.value.stderr is None
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == [
        ssh.LOGIN_TIMEOUT_SECONDS,
        ssh.LOGIN_TERMINATE_GRACE_SECONDS,
        None,
    ]


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX controlling terminal")
def test_interactive_login_restores_controlling_terminal_after_timeout():
    child_pid, master_fd = pty.fork()
    if child_pid == 0:
        try:
            terminal_fd = os.open("/dev/tty", os.O_RDWR)
            original = termios.tcgetattr(terminal_fd)
            original[3] |= termios.ECHO
            termios.tcsetattr(terminal_fd, termios.TCSANOW, original)
            ready_read, ready_write = os.pipe()
            disable_echo = (
                "import os, signal, sys, termios, time; "
                "ready = int(sys.argv[1]); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "fd = os.open('/dev/tty', os.O_RDWR); "
                "state = termios.tcgetattr(fd); "
                "state[3] &= ~termios.ECHO; "
                "termios.tcsetattr(fd, termios.TCSANOW, state); "
                "os.write(ready, b'1'); "
                "os.close(ready); "
                "time.sleep(5)"
            )
            login_process = None

            def popen_factory(argv, **kwargs):
                nonlocal login_process
                assert argv == ["ssh", "-MNf", "orcd-login"]
                login_process = subprocess.Popen(
                    [sys.executable, "-c", disable_echo, str(ready_write)],
                    pass_fds=(ready_write,),
                    **kwargs,
                )
                os.close(ready_write)
                readable, _, _ = select.select([ready_read], [], [], 2.0)
                ready = os.read(ready_read, 1) if readable else b""
                os.close(ready_read)
                if ready != b"1":
                    login_process.kill()
                    login_process.wait()
                    raise AssertionError("login child did not become ready")
                return login_process

            with pytest.raises(subprocess.TimeoutExpired):
                ssh._run_interactive_login(
                    popen_factory=popen_factory,
                    timeout=0.25,
                    terminate_grace=0.25,
                )
            restored = termios.tcgetattr(terminal_fd)
            os.close(terminal_fd)
            killed_and_reaped = (
                login_process is not None and login_process.returncode == -signal.SIGKILL
            )
            os._exit(0 if restored[3] & termios.ECHO and killed_and_reaped else 1)
        except BaseException:
            os._exit(2)

    try:
        _, status = os.waitpid(child_pid, 0)
    finally:
        os.close(master_fd)

    assert os.waitstatus_to_exitcode(status) == 0


def test_ensure_master_reuses_healthy_batch_session_without_login():
    healthy = subprocess.CompletedProcess(["ssh"], 0, "", "")
    runner = RecordingRunner([healthy])
    login_runner = RecordingLoginRunner()

    ssh.SSHClient(runner=runner, login_runner=login_runner).ensure_master()

    assert len(runner.calls) == 1
    assert login_runner.calls == 0
    argv, options = runner.calls[0]
    assert argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "orcd-login",
        "true",
    ]
    assert options == {
        "check": False,
        "text": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": ssh.HEALTH_TIMEOUT_SECONDS,
    }


@pytest.mark.parametrize(
    "close_failure",
    [
        subprocess.TimeoutExpired(["ssh", "-O", "exit", "orcd-login"], 30),
        OSError("private cleanup failure"),
    ],
)
def test_ensure_master_continues_when_stale_close_fails(close_failure):
    unavailable = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    healthy = subprocess.CompletedProcess(["ssh"], 0, "", "")
    runner = RecordingRunner([unavailable, close_failure, healthy])
    login_runner = RecordingLoginRunner()

    ssh.SSHClient(runner=runner, login_runner=login_runner).ensure_master()

    close_argv, close_options = runner.calls[1]
    assert close_argv == ["ssh", "-O", "exit", "orcd-login"]
    assert close_options == {
        "check": False,
        "text": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": ssh.COMMAND_TIMEOUT_SECONDS,
    }
    assert runner.calls[2][0][-2:] == ["orcd-login", "true"]
    assert login_runner.calls == 1
    assert "private" not in repr(runner.calls)


def test_ensure_master_opens_and_verifies_missing_session_interactively():
    unavailable = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    closed = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    healthy = subprocess.CompletedProcess(["ssh"], 0, "", "")
    runner = RecordingRunner([unavailable, closed, healthy])
    login_runner = RecordingLoginRunner()

    ssh.SSHClient(runner=runner, login_runner=login_runner).ensure_master()

    assert login_runner.calls == 1
    assert runner.calls[2][0][-2:] == ["orcd-login", "true"]


def test_ensure_master_raises_sanitized_error_on_login_timeout():
    unavailable = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    closed = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    private_timeout = subprocess.TimeoutExpired(
        ["ssh", "-MNf", "orcd-login"],
        300,
        output="private stdout",
        stderr="private stderr",
    )
    client = ssh.SSHClient(
        runner=RecordingRunner([unavailable, closed]),
        login_runner=RecordingLoginRunner(private_timeout),
    )

    with pytest.raises(ssh.SSHSessionError) as raised:
        client.ensure_master()

    assert str(raised.value) == "ORCD SSH login failed"
    assert raised.value.__cause__ is None
    assert "private" not in repr(raised.value)


def test_ensure_master_raises_sanitized_error_on_login_nonzero():
    unavailable = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    closed = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    runner = RecordingRunner([unavailable, closed])
    login_runner = RecordingLoginRunner(255)
    client = ssh.SSHClient(runner=runner, login_runner=login_runner)

    with pytest.raises(ssh.SSHSessionError, match="ORCD SSH login failed") as raised:
        client.ensure_master()

    assert login_runner.calls == 1
    assert "private" not in str(raised.value)


def test_ensure_master_raises_sanitized_error_on_failed_post_login_verification():
    unavailable = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    closed = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    still_unhealthy = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    runner = RecordingRunner([unavailable, closed, still_unhealthy])
    client = ssh.SSHClient(runner=runner, login_runner=RecordingLoginRunner())

    with pytest.raises(ssh.SSHSessionError, match="ORCD SSH login failed") as raised:
        client.ensure_master()

    assert "private" not in str(raised.value)


def test_close_master_treats_missing_control_socket_as_already_closed():
    result = subprocess.CompletedProcess(
        ["ssh"], 255, "", "Control socket connect(/tmp/socket): No such file or directory"
    )
    runner = RecordingRunner([result])

    assert ssh.SSHClient(runner=runner).close_master() is False
    assert runner.calls[0][0] == ["ssh", "-O", "exit", "orcd-login"]


def test_close_master_does_not_hide_other_errors():
    result = subprocess.CompletedProcess(["ssh"], 255, "", "Permission denied")

    with pytest.raises(ssh.SSHError, match="could not close"):
        ssh.SSHClient(runner=RecordingRunner([result])).close_master()


def test_plan_adds_control_block_without_changing_unrelated_bytes():
    original = b"# personal config\nHost github.com\n  User git\n"

    plan = ssh.plan_control_config(original, "philote")

    assert plan.changed is True
    assert plan.proposed.startswith(original)
    assert plan.proposed[len(original) :] == b"\n" + ssh.control_block("philote").encode() + b"\n"
    assert "--- ~/.ssh/config" in plan.redacted_diff
    assert "+++ ~/.ssh/config (proposed)" in plan.redacted_diff
    assert "personal config" not in plan.redacted_diff
    assert "github.com" not in plan.redacted_diff
    assert "User git" not in plan.redacted_diff


def test_plan_replaces_one_exact_alias_and_preserves_neighbor_blocks():
    before = (
        b"Host first\n  User one\n\n"
        b"Host orcd-login\n  Hostname old.example\n  User old\n\n"
        b"Host last\n  User three\n"
    )

    plan = ssh.plan_control_config(before, "new-user")

    assert plan.proposed.startswith(b"Host first\n  User one\n\n")
    assert plan.proposed.endswith(b"\nHost last\n  User three\n")
    assert plan.proposed.count(b"Host orcd-login") == 1
    assert b"old.example" not in plan.proposed
    assert b"User new-user" in plan.proposed
    assert "old.example" in plan.redacted_diff
    assert "User old" in plan.redacted_diff
    assert "Host first" not in plan.redacted_diff
    assert "Host last" not in plan.redacted_diff


def test_plan_recognizes_tab_separated_host_directive_without_duplicate():
    original = b"Host\torcd-login\n\tHostname old.example\n"

    plan = ssh.plan_control_config(original, "philote")

    assert plan.proposed.count(b"Host orcd-login") == 1
    assert b"Host\torcd-login" not in plan.proposed


def test_plan_is_idempotent_for_existing_managed_block():
    original = (ssh.control_block("philote") + "\n").encode()

    plan = ssh.plan_control_config(original, "philote")

    assert plan.changed is False
    assert plan.proposed == original
    assert plan.redacted_diff == ""


@pytest.mark.parametrize(
    "original",
    [
        b"Host orcd-login\n  User one\nHost orcd-login\n  User two\n",
        b"Host orcd-login other\n  User one\n",
        b"Host orcd-*\n  User one\n",
        b"Host *\n  ServerAliveInterval 60\n",
        b"ControlMaster no\nHost github.com\n  User git\n",
        b"Match all\n  ControlPersist no\n",
    ],
)
def test_plan_rejects_duplicate_or_ambiguous_matching_host_blocks(original):
    with pytest.raises(ssh.SSHConfigError, match="safely"):
        ssh.plan_control_config(original, "philote")


def test_plan_allows_unrelated_multi_pattern_blocks():
    original = b"Host github.com gitlab.com\n  User git\n"

    plan = ssh.plan_control_config(original, "philote")

    assert plan.proposed.startswith(original)


def test_plan_rejects_include_that_could_hide_conflicting_alias():
    original = b"Include ~/.ssh/config.d/*\nHost github.com\n  User git\n"

    with pytest.raises(ssh.SSHConfigError, match="safely"):
        ssh.plan_control_config(original, "philote")


@pytest.mark.parametrize(
    ("unsafe_line", "secret"),
    [
        ("  SetEnv AWS_SECRET_ACCESS_KEY=aws-secret", "aws-secret"),
        ("  SetEnv GITHUB_TOKEN=github-secret", "github-secret"),
        ("  SetEnv WANDB_API_KEY=wandb-secret", "wandb-secret"),
        ("  UnknownDirective arbitrary-secret", "arbitrary-secret"),
        ("  ProxyCommand sh -c 'send proxy-secret'", "proxy-secret"),
        ("  ProxyCommand=proxy-secret", "proxy-secret"),
    ],
)
def test_plan_rejects_unsafe_target_lines_without_displaying_secrets(unsafe_line, secret):
    original = f"Host orcd-login\n  Hostname old.example\n{unsafe_line}\n".encode()

    with pytest.raises(ssh.SSHConfigError, match="safely") as raised:
        ssh.plan_control_config(original, "philote")

    assert secret not in str(raised.value)


def test_plan_preserves_neighboring_comments_without_displaying_them():
    original = (
        b"Host orcd-login\n"
        b"  Hostname old.example\n"
        b"  User old\n"
        b"\n"
        b"# next-token=neighbor-secret\n"
        b"Host next\n"
        b"  User next\n"
    )

    plan = ssh.plan_control_config(original, "philote")

    assert b"# next-token=neighbor-secret\nHost next\n" in plan.proposed
    assert "neighbor-secret" not in plan.redacted_diff
    assert "Host next" not in plan.redacted_diff


def test_plan_preserves_trailing_comments_at_eof_without_displaying_them():
    original = (
        b"Host orcd-login\n"
        b"  Hostname old.example\n"
        b"  User old\n"
        b"\n"
        b"# eof-token=neighbor-secret\n"
    )

    plan = ssh.plan_control_config(original, "philote")

    assert plan.proposed.endswith(b"\n# eof-token=neighbor-secret\n")
    assert "neighbor-secret" not in plan.redacted_diff


def test_plan_rejects_comment_inside_target_block_before_safe_directive():
    original = b"Host orcd-login\n" b"  # token=inside-secret\n" b"  Hostname old.example\n"

    with pytest.raises(ssh.SSHConfigError, match="safely") as raised:
        ssh.plan_control_config(original, "philote")

    assert "inside-secret" not in str(raised.value)


def test_secure_apply_creates_exact_mode_0600_backup_and_config(tmp_path):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n  User git\n"
    config.write_bytes(original)
    config.chmod(0o644)
    plan = ssh.plan_control_config(original, "philote")

    backup = ssh.apply_control_config(config, plan)

    assert backup is not None
    assert backup.read_bytes() == original
    assert config.read_bytes() == plan.proposed
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.parent.stat().st_mode) == 0o700
    assert not list(config.parent.glob(".config.edullm-*"))


def test_secure_apply_is_noop_when_plan_is_unchanged(tmp_path):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = (ssh.control_block("philote") + "\n").encode()
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")

    assert ssh.apply_control_config(config, plan) is None
    assert list(config.parent.iterdir()) == [config]


def test_secure_apply_rejects_symlink_without_touching_target(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_bytes(b"untouched")
    config = ssh_dir / "config"
    config.symlink_to(target)
    plan = ssh.plan_control_config(b"", "philote")

    with pytest.raises(ssh.SSHConfigError, match="symbolic link"):
        ssh.apply_control_config(config, plan)

    assert target.read_bytes() == b"untouched"


def test_secure_apply_detects_stale_plan(tmp_path):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    config.write_bytes(b"Host first\n")
    plan = ssh.plan_control_config(config.read_bytes(), "philote")
    config.write_bytes(b"Host changed\n")

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert config.read_bytes() == b"Host changed\n"


def test_failed_atomic_replace_leaves_original_and_secure_backup(tmp_path, monkeypatch):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o640)
    plan = ssh.plan_control_config(original, "philote")

    real_link = secure_publish.os.link

    def fail_publish(source, target, **kwargs):
        if target == config.name and source.startswith(f".{config.name}.edullm-"):
            raise OSError("simulated publish failure")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", fail_publish)

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert config.read_bytes() == original
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    backups = list(config.parent.glob("config.edullm-backup*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert not list(config.parent.glob(".config.edullm-*"))


def test_read_control_config_rejects_unsafe_parent_symlink(tmp_path):
    real_dir = tmp_path / "real-ssh"
    real_dir.mkdir()
    linked_dir = tmp_path / ".ssh"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ssh.SSHConfigError, match="symbolic link"):
        ssh.read_control_config(linked_dir / "config")


def test_created_config_and_backup_never_have_group_or_other_permissions(tmp_path):
    config = tmp_path / ".ssh" / "config"
    plan = ssh.plan_control_config(b"", "philote")
    observed_modes = []
    real_write = ssh._write_all

    def record_mode_before_write(descriptor, content):
        observed_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        real_write(descriptor, content)

    previous_umask = os.umask(0o777)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(ssh, "_write_all", record_mode_before_write)
            backup = ssh.apply_control_config(config, plan)
    finally:
        os.umask(previous_umask)

    assert backup is not None
    assert observed_modes and set(observed_modes) == {0o600}
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_secure_apply_restores_target_edited_at_actual_publish_boundary(tmp_path, monkeypatch):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
    concurrent_fd = os.open(config, os.O_WRONLY)
    real_link = secure_publish.os.link

    def edit_then_publish(source, target, **kwargs):
        if target == config.name and source.startswith(f".{config.name}.edullm-"):
            os.lseek(concurrent_fd, 0, os.SEEK_SET)
            os.write(concurrent_fd, b"Host concurrent.example\n")
            os.ftruncate(concurrent_fd, len(b"Host concurrent.example\n"))
            os.fsync(concurrent_fd)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", edit_then_publish)
    try:
        with pytest.raises(ssh.SSHConfigError, match="changed|safely"):
            ssh.apply_control_config(config, plan)
    finally:
        os.close(concurrent_fd)

    assert config.read_bytes() == b"Host concurrent.example\n"
    assert not list(config.parent.glob(f".{config.name}.edullm-*"))


def test_secure_apply_never_clobbers_path_replaced_at_publish_boundary(tmp_path, monkeypatch):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
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
            os.write(descriptor, b"Host concurrent.example\n")
            os.close(descriptor)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", replace_then_publish)

    with pytest.raises(ssh.SSHConfigError, match="changed|safely"):
        ssh.apply_control_config(config, plan)

    assert config.read_bytes() == b"Host concurrent.example\n"
    recovery = list(config.parent.glob(f".{config.name}.edullm-recovery-*/original"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == original


def test_secure_apply_rejects_parent_swap_at_actual_publish_boundary(tmp_path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    config = ssh_dir / "config"
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
    moved = tmp_path / ".ssh-moved"
    real_link = secure_publish.os.link
    swapped = False

    def swap_then_publish(source, target, **kwargs):
        nonlocal swapped
        if target == config.name and source.startswith(f".{config.name}.edullm-") and not swapped:
            swapped = True
            ssh_dir.rename(moved)
            ssh_dir.mkdir(mode=0o700)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", swap_then_publish)

    with pytest.raises(ssh.SSHConfigError, match="changed|safely"):
        ssh.apply_control_config(config, plan)

    assert not config.exists()
    assert (moved / "config").read_bytes() == original
    assert not list(moved.glob(f".{config.name}.edullm-*"))


def test_secure_apply_commits_with_same_open_directory_fd(tmp_path, monkeypatch):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
    real_link = secure_publish.os.link
    calls = []

    def record_publish(source, target, **kwargs):
        if target == config.name and source.startswith(f".{config.name}.edullm-"):
            calls.append(((source, target), kwargs))
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", record_publish)

    ssh.apply_control_config(config, plan)

    assert len(calls) == 1
    _, options = calls[0]
    assert options["src_dir_fd"] == options["dst_dir_fd"]


def test_secure_apply_rejects_parent_swap_and_cleans_old_directory_temp(tmp_path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    config = ssh_dir / "config"
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
    moved = tmp_path / ".ssh-moved"
    real_write = ssh._write_all
    swapped = False

    def swap_parent_after_write(descriptor, content):
        nonlocal swapped
        real_write(descriptor, content)
        if not swapped and content == plan.proposed:
            swapped = True
            ssh_dir.rename(moved)
            ssh_dir.mkdir(mode=0o700)

    monkeypatch.setattr(ssh, "_write_all", swap_parent_after_write)

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert (moved / "config").read_bytes() == original
    assert not list(moved.glob(".config.edullm-*"))
    assert not (ssh_dir / "config").exists()


def test_secure_apply_rejects_target_swap_after_validation(tmp_path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    config = ssh_dir / "config"
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    destination = tmp_path / "destination"
    destination.write_bytes(b"untouched")
    plan = ssh.plan_control_config(original, "philote")
    real_write = ssh._write_all
    swapped = False

    def swap_target_after_write(descriptor, content):
        nonlocal swapped
        real_write(descriptor, content)
        if not swapped and content == plan.proposed:
            swapped = True
            config.unlink()
            config.symlink_to(destination)

    monkeypatch.setattr(ssh, "_write_all", swap_target_after_write)

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert destination.read_bytes() == b"untouched"
    assert not list(ssh_dir.glob(".config.edullm-*"))


def _private_parent(home: Path) -> Path:
    return home / ".config" / "edullm"


@pytest.mark.parametrize("target_name", ["wandb.key", "wandb.env"])
def test_remote_helper_atomically_creates_mode_0600_file(tmp_path, target_name):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    ssh_helper.atomic_write_private(home, target_name, io.BytesIO(b"private-content\n"))

    target = _private_parent(home) / target_name
    assert target.read_bytes() == b"private-content\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert not list(target.parent.glob(".edullm-write-*"))


def test_remote_helper_uses_descriptor_chmod_for_private_directories(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    def unsupported_path_chmod(*args, **kwargs):
        raise ValueError("dir_fd and follow_symlinks are incompatible")

    monkeypatch.setattr(ssh_helper.os, "chmod", unsupported_path_chmod)

    ssh_helper.atomic_write_private(home, "wandb.env", io.BytesIO(b"private-content\n"))

    target = _private_parent(home) / "wandb.env"
    assert target.read_bytes() == b"private-content\n"
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("target_name", ["wandb.key", "wandb.env"])
def test_remote_helper_establishes_exact_mode_before_read_under_umask(tmp_path, target_name):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()
    observed_modes = []

    class InspectingInput(io.BytesIO):
        def read(self, size=-1):
            temporary = list(_private_parent(home).glob(".edullm-write-*"))
            assert len(temporary) == 1
            observed_modes.append(stat.S_IMODE(temporary[0].stat().st_mode))
            return super().read(size)

    previous_umask = os.umask(0o777)
    try:
        ssh_helper.atomic_write_private(
            home,
            target_name,
            InspectingInput(b"private-content\n"),
        )
    finally:
        os.umask(previous_umask)

    target = _private_parent(home) / target_name
    assert observed_modes and set(observed_modes) == {0o600}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_remote_helper_rejects_existing_0644_before_reading_input(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o644)

    class Unreadable(io.BytesIO):
        def read(self, size=-1):
            raise AssertionError("input must not be read")

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", Unreadable())

    assert target.read_bytes() == b"existing"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_rejects_target_symlink_without_touching_destination(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    destination = tmp_path / "destination"
    destination.write_bytes(b"untouched")
    (parent / "wandb.key").symlink_to(destination)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"replacement"))

    assert destination.read_bytes() == b"untouched"
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_rejects_parent_symlink(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()
    real_config = tmp_path / "real-config"
    real_config.mkdir()
    (home / ".config").symlink_to(real_config, target_is_directory=True)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"private-content"))

    assert not (real_config / "edullm").exists()


def test_remote_helper_input_failure_is_sanitized_and_cleans_temp(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    class FailedInput(io.BytesIO):
        def read(self, size=-1):
            assert list(_private_parent(home).glob(".edullm-write-*"))
            raise OSError("private-content")

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed") as raised:
        ssh_helper.atomic_write_private(home, "wandb.key", FailedInput())

    assert "private-content" not in str(raised.value)
    assert not list(_private_parent(home).glob(".edullm-write-*"))
    assert not (_private_parent(home) / "wandb.key").exists()


def test_remote_helper_sanitizes_non_os_input_failure(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    class FailedInput(io.BytesIO):
        def read(self, size=-1):
            raise RuntimeError("private-content")

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed") as raised:
        ssh_helper.atomic_write_private(home, "wandb.key", FailedInput())

    assert "private-content" not in str(raised.value)
    assert not list(_private_parent(home).glob(".edullm-write-*"))


def test_remote_helper_handles_short_writes(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()
    real_write = ssh_helper.os.write
    calls = 0

    def short_write(descriptor, content):
        nonlocal calls
        calls += 1
        return real_write(descriptor, content[:1])

    monkeypatch.setattr(ssh_helper.os, "write", short_write)

    ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"complete"))

    assert calls > 1
    assert (_private_parent(home) / "wandb.key").read_bytes() == b"complete"


def test_remote_helper_replace_failure_cleans_temp_and_preserves_target(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o600)

    real_link = secure_publish.os.link

    def fail_publish(source, destination, **kwargs):
        if destination == target.name and source.startswith(".edullm-write-"):
            raise OSError("publish failed")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", fail_publish)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"replacement"))

    assert target.read_bytes() == b"existing"
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_restores_edit_at_actual_publish_boundary(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o600)
    concurrent_fd = os.open(target, os.O_WRONLY)
    real_link = secure_publish.os.link

    def edit_then_publish(source, destination, **kwargs):
        if destination == target.name and source.startswith(".edullm-write-"):
            os.lseek(concurrent_fd, 0, os.SEEK_SET)
            os.write(concurrent_fd, b"concurrent")
            os.ftruncate(concurrent_fd, len(b"concurrent"))
            os.fsync(concurrent_fd)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", edit_then_publish)
    try:
        with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
            ssh_helper.atomic_write_private(home, target.name, io.BytesIO(b"replacement"))
    finally:
        os.close(concurrent_fd)

    assert target.read_bytes() == b"concurrent"
    assert not list(parent.glob(".edullm-write-*"))
    assert not list(parent.glob(f".{target.name}.edullm-recovery-*"))


def test_remote_helper_never_clobbers_boundary_path_replacement(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o600)
    real_link = secure_publish.os.link
    replaced = False

    def replace_then_publish(source, destination, **kwargs):
        nonlocal replaced
        if destination == target.name and source.startswith(".edullm-write-") and not replaced:
            replaced = True
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=kwargs["dst_dir_fd"],
            )
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, b"concurrent")
            os.close(descriptor)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", replace_then_publish)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, target.name, io.BytesIO(b"replacement"))

    assert target.read_bytes() == b"concurrent"
    recovery = list(parent.glob(f".{target.name}.edullm-recovery-*/original"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == b"existing"


def test_remote_helper_rejects_parent_swap_at_actual_publish_boundary(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o600)
    moved = parent.with_name("edullm-moved-at-publish")
    real_link = secure_publish.os.link
    swapped = False

    def swap_then_publish(source, destination, **kwargs):
        nonlocal swapped
        if destination == target.name and source.startswith(".edullm-write-") and not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(secure_publish.os, "link", swap_then_publish)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, target.name, io.BytesIO(b"replacement"))

    assert not target.exists()
    assert (moved / target.name).read_bytes() == b"existing"
    assert not list(moved.glob(".edullm-write-*"))
    assert not list(moved.glob(f".{target.name}.edullm-recovery-*"))
