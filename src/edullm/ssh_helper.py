"""
Trusted remote helper for atomic private-file writes.

This module is invoked through SSH only after the reviewed eduLLM environment
exists. File content is read exclusively from standard input.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from edullm.secure_publish import SecurePublishError, capture_file, compare_and_publish

_TARGETS = frozenset({"wandb.env", "wandb.key"})
_SUBMISSION_KEY = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SUBMISSION_BYTES = 1_048_576
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class PrivateWriteError(RuntimeError):
    """A sanitized private-file write failure."""


@dataclass(frozen=True)
class _Snapshot:
    device: int
    inode: int
    mode: int
    owner: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, status: os.stat_result) -> "_Snapshot":
        return cls(
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_uid,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )


def atomic_write_private(home: Path, target_name: str, source: BinaryIO) -> None:
    """
    Atomically replace one supported private operator file.

    :param home: Trusted operator home directory.
    :param target_name: One of the fixed eduLLM private-file basenames.
    :param source: Binary content stream, normally standard input.

    :raises PrivateWriteError: If any path, input, or commit operation is unsafe.
    """
    if target_name not in _TARGETS:
        raise PrivateWriteError("private write failed")

    directory_fd: int | None = None
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        parent = home / ".config" / "edullm"
        directory_fd, directory_snapshot = _open_private_parent(home)
        _validate_directory_identity(parent, directory_fd, directory_snapshot)
        target_state = capture_file(directory_fd, target_name, exact_mode=0o600)

        temporary_name = f".edullm-write-{secrets.token_hex(12)}"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise PrivateWriteError("private write failed")
        while True:
            chunk = source.read(1024 * 1024)
            if chunk == b"":
                break
            if not isinstance(chunk, bytes):
                raise OSError("invalid private input")
            _write_all(descriptor, chunk)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        prepared = capture_file(directory_fd, temporary_name, exact_mode=0o600)
        if prepared is None:
            raise PrivateWriteError("private write failed")
        publishing_name = temporary_name
        temporary_name = None
        compare_and_publish(
            directory_fd,
            parent,
            (
                directory_snapshot.device,
                directory_snapshot.inode,
                directory_snapshot.owner,
            ),
            target_name,
            publishing_name,
            target_state,
            prepared,
        )
    except PrivateWriteError:
        raise
    except SecurePublishError:
        raise PrivateWriteError("private write failed") from None
    except Exception:
        raise PrivateWriteError("private write failed") from None
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


def _open_private_parent(home: Path) -> tuple[int, _Snapshot]:
    current_fd: int | None = None
    try:
        current_fd = os.open(home, _DIRECTORY_FLAGS)
        _validate_owned_directory(current_fd)
        for component in (".config", "edullm"):
            next_fd = _open_or_create_directory(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        status = os.fstat(current_fd)
        return current_fd, _Snapshot.from_stat(status)
    except Exception:
        if current_fd is not None:
            os.close(current_fd)
        raise


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        previous_umask = os.umask(0o077)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        finally:
            os.umask(previous_umask)
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    try:
        _validate_owned_directory(descriptor)
        os.fchmod(descriptor, 0o700)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _validate_owned_directory(descriptor: int) -> None:
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise PrivateWriteError("private write failed")


def _validate_directory_identity(path: Path, directory_fd: int, expected: _Snapshot) -> None:
    opened = _Snapshot.from_stat(os.fstat(directory_fd))
    try:
        current = _Snapshot.from_stat(path.stat(follow_symlinks=False))
    except (FileNotFoundError, OSError):
        raise PrivateWriteError("private write failed") from None
    expected_identity = (expected.device, expected.inode, expected.owner)
    if (
        (opened.device, opened.inode, opened.owner) != expected_identity
        or (current.device, current.inode, current.owner) != expected_identity
        or not stat.S_ISDIR(opened.mode)
        or not stat.S_ISDIR(current.mode)
    ):
        raise PrivateWriteError("private write failed")


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short write")
        remaining = remaining[written:]


def main(argv: list[str] | None = None) -> int:
    """Read stdin and atomically write one fixed private target."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target", choices=sorted(_TARGETS | {"submission"}), required=True)
    parser.add_argument("--key")
    arguments = parser.parse_args(argv)
    try:
        if arguments.target == "submission":
            if type(arguments.key) is not str or _SUBMISSION_KEY.fullmatch(arguments.key) is None:
                raise PrivateWriteError("private write failed")
            content = sys.stdin.buffer.read(_MAX_SUBMISSION_BYTES + 1)
            if len(content) > _MAX_SUBMISSION_BYTES:
                raise PrivateWriteError("private write failed")
            try:
                script = content.decode("utf-8")
            except UnicodeDecodeError:
                raise PrivateWriteError("private write failed") from None
            from edullm.slurm import SubmissionError, stage_submission

            try:
                stage_submission(
                    Path.home() / "orcd" / "scratch" / "edullm" / "state",
                    arguments.key,
                    script,
                )
            except SubmissionError:
                raise PrivateWriteError("private write failed") from None
        else:
            if arguments.key is not None:
                raise PrivateWriteError("private write failed")
            atomic_write_private(Path.home(), arguments.target, sys.stdin.buffer)
    except PrivateWriteError:
        print("private write failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
