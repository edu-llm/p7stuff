# eduLLM SSH Session Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make operator commands automatically establish a healthy SSH ControlMaster without
counting password and Duo interaction against the 30-second Slurm command timeout.

**Architecture:** `SSHClient` owns a tested `ensure_master()` transition that probes, refreshes,
opens, and verifies the fixed `orcd-login` session. CLI setup and operational service loading invoke
that transition once, while bounded remote operations use `BatchMode=yes` and never prompt.

**Tech Stack:** Python 3.10+, stdlib `subprocess`, OpenSSH ControlMaster, pytest, Ruff

## Global Constraints

- Authentication may take up to 300 seconds; ordinary remote commands remain bounded to 30 seconds.
- Password, Duo, private-key, and raw SSH diagnostics must never be captured, persisted, or printed.
- SSH host aliases and command-line options are fixed project values, not user-controlled text.
- `ControlPersist 1h` and `edullm logout` behavior remain unchanged.
- Production behavior must be preceded by a focused failing regression test.
- No commit or push is included because the user requested working-tree changes for local testing,
  not publication.

---

### Task 1: Add a reliable SSH session transition

**Files:**
- Modify: `src/edullm/ssh.py`
- Test: `src/test/edullm/ssh_test.py`

**Interfaces:**
- Produces: `SSHClient.ensure_master() -> None`
- Produces: `SSHSessionError(SSHError)` for sanitized authentication/session failures
- Changes: `SSHClient.run_remote()` and `SSHClient.write_remote()` always pass `BatchMode=yes`

- [ ] **Step 1: Write failing tests for healthy-session reuse and safe interactive login**

Use `RecordingRunner` for batch probes and stale-master cleanup, plus a separate
`RecordingLoginRunner` for the injectable interactive boundary:

```python
def test_ensure_master_reuses_healthy_batch_session_without_login():
    healthy = subprocess.CompletedProcess(["ssh"], 0, "", "")
    runner = RecordingRunner([healthy])
    login_runner = RecordingLoginRunner()

    ssh.SSHClient(runner=runner, login_runner=login_runner).ensure_master()

    assert runner.calls[0][0][:5] == [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"
    ]
    assert login_runner.calls == 0


def test_ensure_master_opens_and_verifies_missing_session_interactively():
    unavailable = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    closed = subprocess.CompletedProcess(["ssh"], 255, "", "private")
    healthy = subprocess.CompletedProcess(["ssh"], 0, "", "")
    runner = RecordingRunner([unavailable, closed, healthy])
    login_runner = RecordingLoginRunner()

    ssh.SSHClient(runner=runner, login_runner=login_runner).ensure_master()

    assert login_runner.calls == 1
    assert runner.calls[2][0][-2:] == ["orcd-login", "true"]
```

Add focused `_run_interactive_login()` tests proving that its `Popen` call uses only
`["ssh", "-MNf", "orcd-login"]`, leaves stdin inherited with `stdin=None`, and sends ordinary
stdout/stderr to `subprocess.DEVNULL`. Add a POSIX PTY-backed regression whose real child disables
terminal echo, signals that startup is complete, ignores `SIGTERM`, and remains alive long enough
to force the helper's `SIGKILL` and reap fallback. After timeout, assert both the real child's
`-SIGKILL` return code and restoration of the original controlling-terminal attributes.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q \
  src/test/edullm/ssh_test.py::test_ensure_master_reuses_healthy_batch_session_without_login \
  src/test/edullm/ssh_test.py::test_ensure_master_opens_and_verifies_missing_session_interactively \
  src/test/edullm/ssh_test.py::test_interactive_login_uses_fixed_argv_inherited_stdin_and_devnull_output \
  src/test/edullm/ssh_test.py::test_interactive_login_timeout_terminates_then_kills_and_reaps \
  src/test/edullm/ssh_test.py::test_interactive_login_restores_controlling_terminal_after_timeout
```

Expected: FAIL because the injectable login runner and Popen-based helper do not exist.

- [ ] **Step 3: Implement the minimal session lifecycle**

Add fixed constants, the sanitized error subtype, a Popen-based login helper, a batch health probe,
and the injectable `ensure_master()` boundary:

```python
LOGIN_TIMEOUT_SECONDS = 300.0
LOGIN_TERMINATE_GRACE_SECONDS = 2.0
HEALTH_TIMEOUT_SECONDS = 15.0


class SSHSessionError(SSHError):
    """A sanitized SSH authentication or ControlMaster failure."""


def _run_interactive_login(
    *,
    popen_factory=subprocess.Popen,
    timeout=LOGIN_TIMEOUT_SECONDS,
    terminate_grace=LOGIN_TERMINATE_GRACE_SECONDS,
):
    terminal_fd = None
    terminal_state = None
    try:
        try:
            terminal_fd = os.open("/dev/tty", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
            terminal_state = termios.tcgetattr(terminal_fd)
        except (OSError, termios.error):
            if terminal_fd is not None:
                os.close(terminal_fd)
            terminal_fd = None
            terminal_state = None

        process = popen_factory(
            ["ssh", "-MNf", _ALIAS],
            stdin=None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=terminate_grace)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    finally:
        if terminal_fd is not None:
            try:
                termios.tcsetattr(terminal_fd, termios.TCSANOW, terminal_state)
            finally:
                os.close(terminal_fd)


def __init__(self, runner=subprocess.run, login_runner=_run_interactive_login):
    self._runner = runner
    self._login_runner = login_runner


def ensure_master(self) -> None:
    if self._master_healthy():
        return
    try:
        self._runner(
            ["ssh", "-O", "exit", _ALIAS],
            check=False,
            text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        returncode = self._login_runner()
    except (OSError, subprocess.SubprocessError):
        raise SSHSessionError("ORCD SSH login failed") from None
    if returncode != 0 or not self._master_healthy():
        raise SSHSessionError("ORCD SSH login failed")
```

The production helper snapshots `/dev/tty` termios state when a controlling terminal is available
and restores it in `finally`, including timeout and process-launch failures. It never pipes stdin or
captures ordinary diagnostics. On timeout it sends `SIGTERM`, waits for the fixed grace period,
escalates to `SIGKILL` when necessary, and performs an unbounded final `wait()` only after kill so
the child is reaped.

The private `_master_healthy()` probe uses:

```python
["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", _ALIAS, "true"]
```

and returns `False` for nonzero results, timeouts, or OS errors without exposing diagnostics.
Both the health probe and stale-master close send stdout and stderr directly to `DEVNULL`; they
must not capture, print, or persist diagnostic bytes.

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Add RED tests for sanitized failures and non-interactive operations**

Add tests proving:

```python
with pytest.raises(ssh.SSHSessionError, match="ORCD SSH login failed"):
    client.ensure_master()
```

for login timeout, login nonzero, and failed post-login verification. Update remote-command and
remote-write assertions to require `["ssh", "-o", "BatchMode=yes", "orcd-login", ...]`.

- [ ] **Step 6: Implement and verify strict batch operation mode**

Add `-o BatchMode=yes` to `run_remote()` and `write_remote()`. Run:

```bash
pytest -q src/test/edullm/ssh_test.py
```

Expected: all SSH tests pass.

---

### Task 2: Wire session bootstrap into setup and operator commands

**Files:**
- Modify: `src/edullm/cli.py`
- Modify: `src/test/edullm/cli_test.py`
- Modify: `docs/edullm-team-workflow.md`

**Interfaces:**
- Consumes: `SSHClient.ensure_master() -> None`
- Consumes: `SSHSessionError`
- Changes: `_load_operator_services()` establishes SSH before constructing Slurm adapters
- Changes: `setup_operator()` establishes SSH before its first alias-based remote check

- [ ] **Step 1: Write a failing setup-order test**

Extend `StatefulSetupSSH`:

```python
def ensure_master(self):
    self.events.append("ssh-master")
    if self.fail_at == "ssh-master":
        raise cli.SSHSessionError("ORCD SSH login failed")
```

Update the setup success assertion to require `ssh-master` immediately before `tool-hostname`, and
add a failure test proving no remote tool check occurs when `ssh-master` fails.

- [ ] **Step 2: Run the focused setup tests and verify RED**

Run:

```bash
pytest -q src/test/edullm/cli_test.py -k "setup_runs_checks or ssh_master"
```

Expected: FAIL because setup does not call `ensure_master()`.

- [ ] **Step 3: Bootstrap SSH during setup**

After the managed SSH block is confirmed/applied and before the remote tool loop:

```python
dependencies.ssh_client.ensure_master()
```

Allow `SSHSessionError` to propagate directly to `handle_setup()`, which catches it and prints the
fixed recovery message. Remove the redundant direct reachability login so setup produces only one
password/Duo flow.

- [ ] **Step 4: Verify setup GREEN**

Run the command from Step 2. Expected: selected tests pass.

- [ ] **Step 5: Write a failing operator-service bootstrap test**

Use a fake SSH client recording `ensure_master()` and assert `_load_operator_services()` invokes it
once before passing the same client to `SSHSubmissionRemote` and `SSHSlurm`. Add a failure case that
raises `SSHSessionError` and verifies the CLI prints:

```text
ORCD SSH login failed; run ssh -MNf orcd-login and retry.
```

- [ ] **Step 6: Wire and expose the actionable sanitized error**

Call `ssh_client.ensure_master()` in `_load_operator_services()` and allow `SSHSessionError` to
propagate directly. Have `jobs`, `run`, `logs`, and `stop` handlers catch `SSHSessionError` and print
the fixed recovery message; retain existing generic messages for every other failure class.

- [ ] **Step 7: Verify operator CLI GREEN**

Run:

```bash
pytest -q src/test/edullm/cli_test.py -k "operator_services or handle_jobs or handle_run or handle_logs or handle_stop"
```

Expected: selected tests pass.

- [ ] **Step 8: Update the teammate instructions**

Document that operator commands automatically request password/Duo when the one-hour session is
missing, reuse the master afterward, and that `ssh -MNf orcd-login` is the manual fallback.

- [ ] **Step 9: Run focused verification**

Run:

```bash
pytest -q src/test/edullm/ssh_test.py src/test/edullm/cli_test.py
ruff check src/edullm/ssh.py src/edullm/cli.py \
  src/test/edullm/ssh_test.py src/test/edullm/cli_test.py
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 10: Perform the real acceptance test**

Close the current project master:

```bash
edullm logout
```

Then run:

```bash
edullm jobs
```

Expected: password/Duo appears with no manual `ssh -MNf`; after approval, the command continues and
prints the operator's authorized jobs.
