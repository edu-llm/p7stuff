"""
Safe SSH and OpenSSH configuration boundaries for eduLLM operators.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import secrets
import shlex
import stat
import subprocess
import termios
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from edullm.secure_publish import SecurePublishError, capture_file, compare_and_publish

COMMAND_TIMEOUT_SECONDS = 30.0
LOGIN_TIMEOUT_SECONDS = 300.0
LOGIN_TERMINATE_GRACE_SECONDS = 2.0
HEALTH_TIMEOUT_SECONDS = 15.0
_ALIAS = "orcd-login"
_HOSTNAME = "orcd-login.mit.edu"
_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_REMOTE_PRIVATE_TARGETS = {
    "~/.config/edullm/wandb.env": "wandb.env",
    "~/.config/edullm/wandb.key": "wandb.key",
}
_REMOTE_SUBMISSION_TARGET = re.compile(r"submission/([0-9a-f]{64})/request\.sbatch\Z")
_SAFE_HOST_LINE = re.compile(r"(?i)^\s*Host\s+orcd-login\s*$")
_SAFE_HOSTNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z")
_SAFE_CONTROL_PATH = re.compile(r"[A-Za-z0-9_./~%+-]+\Z")
_SAFE_CONTROL_PERSIST = re.compile(r"(?:yes|no|[0-9]+[smhdw]?)\Z", re.IGNORECASE)
_SAFE_CONTROL_MASTER = frozenset({"auto", "autoask", "ask", "no", "yes"})


class SSHError(RuntimeError):
    """A sanitized SSH operation failure."""


class SSHSessionError(SSHError):
    """A sanitized SSH authentication or ControlMaster failure."""


class SSHConfigError(RuntimeError):
    """An unsafe or failed OpenSSH configuration operation."""


class _LoginProcess(Protocol):
    def wait(self, timeout: float | None = None) -> int:
        ...

    def terminate(self) -> None:
        ...

    def kill(self) -> None:
        ...


@dataclass(frozen=True)
class SSHConfigPlan:
    """A proposed byte-preserving OpenSSH configuration update."""

    original: bytes
    proposed: bytes
    redacted_diff: str

    @property
    def changed(self) -> bool:
        """Return whether this plan changes the configuration."""
        return self.original != self.proposed


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    mode: int
    owner: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, status: os.stat_result) -> "_FileSnapshot":
        return cls(
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_uid,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )


def _run_interactive_login(
    *,
    popen_factory: Callable[..., _LoginProcess] = subprocess.Popen,
    timeout: float = LOGIN_TIMEOUT_SECONDS,
    terminate_grace: float = LOGIN_TERMINATE_GRACE_SECONDS,
) -> int:
    terminal_fd: int | None = None
    terminal_state = None
    try:
        try:
            terminal_fd = os.open(
                "/dev/tty",
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            )
            terminal_state = termios.tcgetattr(terminal_fd)
        except (OSError, termios.error):
            if terminal_fd is not None:
                try:
                    os.close(terminal_fd)
                except OSError:
                    pass
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
                if terminal_state is not None:
                    termios.tcsetattr(terminal_fd, termios.TCSANOW, terminal_state)
            finally:
                os.close(terminal_fd)


class SSHClient:
    """Injectable subprocess boundary for project SSH operations."""

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        login_runner: Callable[[], int] = _run_interactive_login,
    ):
        self._runner = runner
        self._login_runner = login_runner

    def _run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
        operation: str = "SSH command failed",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                list(argv),
                input=input_text,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            raise SSHError(operation) from None
        if check and result.returncode != 0:
            raise SSHError(operation)
        return result

    def _master_healthy(self) -> bool:
        try:
            result = self._runner(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    _ALIAS,
                    "true",
                ],
                check=False,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=HEALTH_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def ensure_master(self) -> None:
        """
        Ensure a healthy project ControlMaster is available.

        Probes ``orcd-login`` in batch mode, closes any stale master on a
        best-effort basis, opens an interactive ``ssh -MNf orcd-login`` session
        when needed, and verifies the master afterward.

        :raises SSHSessionError: If interactive login or post-login verification
            fails.
        """
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

    def run_remote(
        self,
        argv: Sequence[str],
        *,
        check: bool = True,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        """Run one argv-safe command through the project alias."""
        command = _remote_command(argv)
        return self._run(
            ["ssh", "-o", "BatchMode=yes", _ALIAS, command],
            check=check,
            timeout=timeout,
        )

    def run_direct(
        self,
        username: str,
        argv: Sequence[str],
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        """Run one argv-safe command against Engaging without the alias."""
        _validate_username(username)
        command = _remote_command(argv)
        return self._run(
            [
                "ssh",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                "-l",
                username,
                _HOSTNAME,
                command,
            ],
            timeout=timeout,
        )

    def write_remote(
        self,
        path: str,
        content: str,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        """Write private content remotely using standard input."""
        target = _REMOTE_PRIVATE_TARGETS.get(path)
        if target is not None:
            command = (
                '"$HOME/venvs/edullm/bin/python" -m edullm.ssh_helper '
                f"--target {shlex.quote(target)}"
            )
        else:
            match = _REMOTE_SUBMISSION_TARGET.fullmatch(path) if type(path) is str else None
            if match is None:
                raise ValueError("remote path is unsafe")
            command = (
                '"$HOME/venvs/edullm/bin/python" -m edullm.ssh_helper '
                f"--target submission --key {match.group(1)}"
            )
        self._run(
            ["ssh", "-o", "BatchMode=yes", _ALIAS, command],
            input_text=content,
            timeout=timeout,
        )

    def close_master(self) -> bool:
        """
        Close only the project ControlMaster.

        :returns: ``True`` when a master was closed and ``False`` when it was
            already absent.
        """
        result = self._run(
            ["ssh", "-O", "exit", _ALIAS],
            operation="could not close eduLLM SSH session",
            check=False,
        )
        if result.returncode == 0:
            return True
        stderr = result.stderr.lower()
        already_closed = "control socket" in stderr and (
            "no such file" in stderr or "does not exist" in stderr
        )
        if already_closed:
            return False
        raise SSHError("could not close eduLLM SSH session")


def control_block(username: str) -> str:
    """Render the managed ``Host orcd-login`` block."""
    _validate_username(username)
    return "\n".join(
        [
            f"Host {_ALIAS}",
            f"    Hostname {_HOSTNAME}",
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/edullm-%C",
            "    ControlPersist 1h",
            f"    User {username}",
        ]
    )


def plan_control_config(original: bytes, username: str) -> SSHConfigPlan:
    """Plan a safe managed-block addition or replacement."""
    block = control_block(username)
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        raise SSHConfigError("SSH config cannot be updated safely") from None

    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    starts: list[tuple[int, list[str]]] = []
    block_started = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        directive = stripped.rstrip("\r\n").split(None, maxsplit=1)
        keyword = directive[0]
        normalized_keyword = keyword.casefold()
        if normalized_keyword in {"include", "match"}:
            raise SSHConfigError("SSH config cannot be updated safely")
        if not block_started and normalized_keyword in {
            "hostname",
            "user",
            "controlmaster",
            "controlpath",
            "controlpersist",
        }:
            raise SSHConfigError("SSH config cannot be updated safely")
        if normalized_keyword != "host":
            continue
        block_started = True
        remainder = directive[1] if len(directive) == 2 else ""
        try:
            patterns = shlex.split(remainder, comments=True, posix=True)
        except ValueError:
            raise SSHConfigError("SSH config cannot be updated safely") from None
        if not patterns:
            raise SSHConfigError("SSH config cannot be updated safely")
        starts.append((index, patterns))

    exact_starts: list[int] = []
    for start, patterns in starts:
        exact = len(patterns) == 1 and patterns[0].casefold() == _ALIAS
        if exact:
            exact_starts.append(start)
            continue
        if any(_host_pattern_matches(pattern, _ALIAS) for pattern in patterns):
            raise SSHConfigError("SSH config cannot be updated safely")
    if len(exact_starts) > 1:
        raise SSHConfigError("SSH config cannot be updated safely")

    rendered = block.replace("\n", newline) + newline
    old_display = ""
    if exact_starts:
        start = exact_starts[0]
        stanza_end = len(lines)
        for index in range(start + 1, len(lines)):
            stripped = lines[index].lstrip()
            directive = stripped.rstrip("\r\n").split(None, maxsplit=1)
            keyword = directive[0].casefold() if directive else ""
            if keyword in {"host", "match"}:
                stanza_end = index
                break
        content_end = stanza_end
        while content_end > start + 1 and _is_comment_or_blank(lines[content_end - 1]):
            content_end -= 1
        _validate_safe_target_block(lines[start:content_end])
        old_display = "".join(lines[start:content_end])
        proposed_text = "".join(lines[:start]) + rendered + "".join(lines[content_end:])
    else:
        separator = newline if text else ""
        proposed_text = text + separator + rendered

    proposed = proposed_text.encode("utf-8")
    return SSHConfigPlan(
        original=original,
        proposed=proposed,
        redacted_diff=_safe_block_diff(old_display, rendered) if proposed != original else "",
    )


def read_control_config(path: Path) -> bytes:
    """Read an OpenSSH configuration without following unsafe links."""
    if not _inspect_private_directory(path.parent):
        return b""
    try:
        path_status = path.lstat()
    except FileNotFoundError:
        return b""
    except OSError:
        raise SSHConfigError("SSH config cannot be read safely") from None
    if stat.S_ISLNK(path_status.st_mode):
        raise SSHConfigError("SSH config must not be a symbolic link")
    if not stat.S_ISREG(path_status.st_mode) or path_status.st_uid != os.getuid():
        raise SSHConfigError("SSH config cannot be read safely")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise SSHConfigError("SSH config cannot be read safely") from None
    try:
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or opened_status.st_uid != os.getuid()
            or (opened_status.st_dev, opened_status.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise SSHConfigError("SSH config changed while it was being read")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def apply_control_config(path: Path, plan: SSHConfigPlan) -> Path | None:
    """Apply a confirmed plan atomically and return its secure backup."""
    _ensure_private_directory(path.parent)
    directory_fd: int | None = None
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        directory_fd = _open_private_directory(path.parent)
        directory_identity = _directory_identity(directory_fd)
        _validate_directory_identity(path.parent, directory_fd, directory_identity)
        current, _ = _read_config_at(directory_fd, path.name)
        target_state = capture_file(directory_fd, path.name, reject_write_bits=True)
        if current != plan.original:
            raise SSHConfigError("SSH config changed after the proposed diff")
        if (target_state.content if target_state is not None else b"") != plan.original:
            raise SSHConfigError("SSH config changed after the proposed diff")
        if not plan.changed:
            return None

        backup = _create_backup_at(
            path,
            directory_fd,
            current,
        )
        temporary_name = f".{path.name}.edullm-{secrets.token_hex(8)}"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise SSHConfigError("SSH config could not update safely")
        _write_all(descriptor, plan.proposed)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        prepared = capture_file(directory_fd, temporary_name, exact_mode=0o600)
        if prepared is None:
            raise SSHConfigError("SSH config could not update safely")
        publishing_name = temporary_name
        temporary_name = None
        compare_and_publish(
            directory_fd,
            path.parent,
            directory_identity,
            path.name,
            publishing_name,
            target_state,
            prepared,
        )
    except SSHConfigError:
        raise
    except SecurePublishError:
        raise SSHConfigError("SSH config changed after the proposed diff") from None
    except OSError:
        raise SSHConfigError("SSH config could not update safely") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)
    return backup


def run_remote(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one command through the project SSH alias."""
    return SSHClient().run_remote(argv)


def write_remote(path: str, content: str) -> None:
    """Write content remotely through standard input."""
    SSHClient().write_remote(path, content)


def close_master() -> bool:
    """Close only the project ControlMaster."""
    return SSHClient().close_master()


def _validate_username(username: str) -> None:
    if _USERNAME.fullmatch(username) is None or username.startswith("-"):
        raise ValueError("Engaging username is invalid")


def _remote_command(argv: Sequence[str]) -> str:
    if not argv or any(not isinstance(argument, str) or "\0" in argument for argument in argv):
        raise ValueError("remote argv is invalid")
    return shlex.join(argv)


def _host_pattern_matches(pattern: str, hostname: str) -> bool:
    if pattern.startswith("!"):
        pattern = pattern[1:]
    return fnmatch.fnmatchcase(hostname.casefold(), pattern.casefold())


def _safe_block_diff(original: str, proposed: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(),
        proposed.splitlines(),
        fromfile="~/.ssh/config",
        tofile="~/.ssh/config (proposed)",
        n=0,
        lineterm="",
    )
    return "\n".join(lines) + "\n"


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _validate_safe_target_block(lines: Sequence[str]) -> None:
    if not lines or _SAFE_HOST_LINE.fullmatch(lines[0].rstrip("\r\n")) is None:
        raise SSHConfigError("SSH config cannot be updated safely")
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            raise SSHConfigError("SSH config cannot be updated safely")
        match = re.fullmatch(r"([A-Za-z]+)(?:\s+|=)(\S+)", stripped)
        if match is None:
            raise SSHConfigError("SSH config cannot be updated safely")
        directive, value = match.groups()
        normalized = directive.casefold()
        if normalized == "hostname":
            safe = _SAFE_HOSTNAME.fullmatch(value) is not None
        elif normalized == "user":
            safe = _USERNAME.fullmatch(value) is not None and not value.startswith("-")
        elif normalized == "controlmaster":
            safe = value.casefold() in _SAFE_CONTROL_MASTER
        elif normalized == "controlpath":
            safe = _SAFE_CONTROL_PATH.fullmatch(value) is not None
        elif normalized == "controlpersist":
            safe = _SAFE_CONTROL_PERSIST.fullmatch(value) is not None
        else:
            safe = False
        if not safe:
            raise SSHConfigError("SSH config cannot be updated safely")


def _ensure_private_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700, parents=True)
            status = path.lstat()
        except OSError:
            raise SSHConfigError("SSH directory cannot be created safely") from None
    except OSError:
        raise SSHConfigError("SSH directory cannot be inspected safely") from None
    if stat.S_ISLNK(status.st_mode):
        raise SSHConfigError("SSH directory must not be a symbolic link")
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise SSHConfigError("SSH directory is unsafe")
    try:
        path.chmod(0o700)
    except OSError:
        raise SSHConfigError("SSH directory permissions cannot be secured") from None


def _inspect_private_directory(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise SSHConfigError("SSH directory cannot be inspected safely") from None
    if stat.S_ISLNK(status.st_mode):
        raise SSHConfigError("SSH directory must not be a symbolic link")
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise SSHConfigError("SSH directory is unsafe")
    return True


def _open_private_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
            raise SSHConfigError("SSH directory is unsafe")
        return descriptor
    except SSHConfigError:
        if "descriptor" in locals():
            os.close(descriptor)
        raise
    except OSError:
        raise SSHConfigError("SSH directory cannot be opened safely") from None


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    status = os.fstat(descriptor)
    return status.st_dev, status.st_ino, status.st_uid


def _validate_directory_identity(
    path: Path,
    descriptor: int,
    expected: tuple[int, int, int],
) -> None:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        raise SSHConfigError("SSH config changed after the proposed diff") from None
    if (
        _directory_identity(descriptor) != expected
        or (status.st_dev, status.st_ino, status.st_uid) != expected
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise SSHConfigError("SSH config changed after the proposed diff")


def _read_config_at(directory_fd: int, name: str) -> tuple[bytes, _FileSnapshot | None]:
    try:
        status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return b"", None
    except OSError:
        raise SSHConfigError("SSH config cannot be read safely") from None
    mode = stat.S_IMODE(status.st_mode)
    if stat.S_ISLNK(status.st_mode):
        raise SSHConfigError("SSH config must not be a symbolic link")
    if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid() or mode & 0o022:
        raise SSHConfigError("SSH config is unsafe")
    expected = _FileSnapshot.from_stat(status)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        opened = _FileSnapshot.from_stat(os.fstat(descriptor))
        if opened != expected:
            raise SSHConfigError("SSH config changed after the proposed diff")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks), expected
    except SSHConfigError:
        raise
    except OSError:
        raise SSHConfigError("SSH config cannot be read safely") from None
    finally:
        if "descriptor" in locals():
            os.close(descriptor)


def _create_backup_at(
    path: Path,
    directory_fd: int,
    content: bytes,
) -> Path:
    index = 0
    while True:
        suffix = "" if index == 0 else f".{index}"
        candidate_name = f"{path.name}.edullm-backup{suffix}"
        try:
            descriptor = os.open(
                candidate_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            index += 1
            continue
        except OSError:
            raise SSHConfigError("SSH config backup could not be created safely") from None
        try:
            os.fchmod(descriptor, 0o600)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                raise OSError("unsafe backup mode")
            _write_all(descriptor, content)
            os.fsync(descriptor)
        except OSError:
            try:
                os.unlink(candidate_name, dir_fd=directory_fd)
            except OSError:
                pass
            raise SSHConfigError("SSH config backup could not be created safely") from None
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
        return path.with_name(candidate_name)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]
