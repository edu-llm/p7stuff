"""
Portable compare-and-publish primitives for private regular files.

Publication uses only directory-fd-relative operations. Existing targets are
atomically displaced into a private recovery directory, prepared files are
hard-linked into an absent target name, and rollback never overwrites a name
that another actor recreated.
"""

from __future__ import annotations

import errno
import fcntl
import os
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


class SecurePublishError(RuntimeError):
    """A sanitized compare-and-publish failure."""


@dataclass(frozen=True)
class FileState:
    """Stable identity, mode, ownership, and bytes for one regular file."""

    device: int
    inode: int
    mode: int
    owner: int
    content: bytes = field(repr=False)


def directory_identity(directory_fd: int) -> tuple[int, int, int]:
    """Return the stable identity of an open owned directory."""
    status = os.fstat(directory_fd)
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise SecurePublishError("safe publication failed")
    return status.st_dev, status.st_ino, status.st_uid


def capture_file(
    directory_fd: int,
    name: str,
    *,
    exact_mode: int | None = None,
    reject_write_bits: bool = False,
) -> FileState | None:
    """Capture an owned regular file without following links."""
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError:
        raise SecurePublishError("safe publication failed") from None
    try:
        state = _capture_descriptor(descriptor)
    finally:
        os.close(descriptor)
    mode = stat.S_IMODE(state.mode)
    if exact_mode is not None and mode != exact_mode:
        raise SecurePublishError("safe publication failed")
    if reject_write_bits and mode & 0o022:
        raise SecurePublishError("safe publication failed")
    return state


def compare_and_publish(
    directory_fd: int,
    parent_path: Path,
    expected_parent: tuple[int, int, int],
    target_name: str,
    temporary_name: str,
    expected: FileState | None,
    prepared: FileState,
) -> None:
    """
    Publish ``temporary_name`` without clobbering concurrent target changes.

    Any displaced file that cannot be restored without overwriting a
    concurrently recreated name remains in an explicit private recovery
    directory.
    """
    stage_name: str | None = None
    stage_fd: int | None = None
    displaced_fd: int | None = None
    lock_fd: int | None = None
    lock_name = f".edullm-lock-{target_name}"
    published = False
    try:
        lock_fd = _acquire_lock(directory_fd, lock_name)
        _validate_parent(parent_path, directory_fd, expected_parent)
        if capture_file(directory_fd, target_name) != expected:
            raise SecurePublishError("safe publication failed")
        if capture_file(directory_fd, temporary_name, exact_mode=0o600) != prepared:
            raise SecurePublishError("safe publication failed")

        if expected is not None:
            stage_name, stage_fd = _create_stage(directory_fd, target_name)
            os.rename(
                target_name,
                "original",
                src_dir_fd=directory_fd,
                dst_dir_fd=stage_fd,
            )
            displaced_fd = os.open("original", _FILE_FLAGS, dir_fd=stage_fd)
            if _capture_descriptor(displaced_fd) != expected:
                raise SecurePublishError("safe publication failed")

        _validate_parent(parent_path, directory_fd, expected_parent)
        _validate_lock(directory_fd, lock_name, lock_fd)
        if displaced_fd is not None and _capture_descriptor(displaced_fd) != expected:
            raise SecurePublishError("safe publication failed")

        os.link(
            temporary_name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True

        if capture_file(directory_fd, target_name, exact_mode=0o600) != prepared:
            raise SecurePublishError("safe publication failed")
        _validate_parent(parent_path, directory_fd, expected_parent)
        _validate_lock(directory_fd, lock_name, lock_fd)
        if displaced_fd is not None and _capture_descriptor(displaced_fd) != expected:
            raise SecurePublishError("safe publication failed")

        os.fsync(directory_fd)
        _remove_name_if_state(directory_fd, target_name, temporary_name, prepared)
        if stage_fd is not None:
            assert displaced_fd is not None
            assert stage_name is not None
            os.unlink("original", dir_fd=stage_fd)
            os.close(displaced_fd)
            displaced_fd = None
            os.close(stage_fd)
            stage_fd = None
            os.rmdir(stage_name, dir_fd=directory_fd)
            stage_name = None
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        return
    except Exception:
        if published:
            stage_name, stage_fd = _rollback_published(
                directory_fd,
                target_name,
                prepared,
                stage_name,
                stage_fd,
            )
        if stage_fd is not None:
            _restore_stage_entry(directory_fd, target_name, stage_fd, "original")
        _remove_name_if_state(directory_fd, target_name, temporary_name, prepared)
        if displaced_fd is not None:
            os.close(displaced_fd)
            displaced_fd = None
        if stage_fd is not None:
            os.close(stage_fd)
            stage_fd = None
        if lock_fd is not None:
            os.close(lock_fd)
            lock_fd = None
        if stage_name is not None:
            try:
                os.rmdir(stage_name, dir_fd=directory_fd)
            except OSError:
                pass
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        raise SecurePublishError("safe publication failed") from None
    finally:
        if displaced_fd is not None:
            os.close(displaced_fd)
        if stage_fd is not None:
            os.close(stage_fd)
        if lock_fd is not None:
            os.close(lock_fd)


def _capture_descriptor(descriptor: int) -> FileState:
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
        raise SecurePublishError("safe publication failed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return FileState(
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_uid,
        b"".join(chunks),
    )


def _acquire_lock(directory_fd: int, name: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
            raise SecurePublishError("safe publication failed")
        os.fchmod(descriptor, 0o600)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise SecurePublishError("safe publication failed")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_lock(directory_fd, name, descriptor)
        return descriptor
    except Exception:
        if "descriptor" in locals():
            os.close(descriptor)
        raise SecurePublishError("safe publication failed") from None


def _validate_lock(directory_fd: int, name: str, descriptor: int) -> None:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        raise SecurePublishError("safe publication failed") from None
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_uid != os.getuid()
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        or stat.S_IMODE(current.st_mode) != 0o600
    ):
        raise SecurePublishError("safe publication failed")


def _validate_parent(
    path: Path,
    directory_fd: int,
    expected: tuple[int, int, int],
) -> None:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        raise SecurePublishError("safe publication failed") from None
    if (
        directory_identity(directory_fd) != expected
        or (status.st_dev, status.st_ino, status.st_uid) != expected
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise SecurePublishError("safe publication failed")


def _mkdir_private(directory_fd: int, name: str) -> None:
    previous_umask = os.umask(0o077)
    try:
        os.mkdir(name, 0o700, dir_fd=directory_fd)
    finally:
        os.umask(previous_umask)


def _create_stage(directory_fd: int, target_name: str) -> tuple[str, int]:
    for _ in range(100):
        name = f".{target_name}.edullm-recovery-{secrets.token_hex(12)}"
        try:
            _mkdir_private(directory_fd, name)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.fchmod(descriptor, 0o700)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
                raise SecurePublishError("safe publication failed")
            return name, descriptor
        except Exception:
            try:
                os.rmdir(name, dir_fd=directory_fd)
            except OSError:
                pass
            raise
    raise SecurePublishError("safe publication failed")


def _rollback_published(
    directory_fd: int,
    target_name: str,
    prepared: FileState,
    stage_name: str | None,
    stage_fd: int | None,
) -> tuple[str | None, int | None]:
    if stage_fd is None:
        stage_name, stage_fd = _create_stage(directory_fd, target_name)
    try:
        os.rename(
            target_name,
            "published",
            src_dir_fd=directory_fd,
            dst_dir_fd=stage_fd,
        )
    except FileNotFoundError:
        return stage_name, stage_fd
    current = capture_file(stage_fd, "published")
    if current == prepared:
        os.unlink("published", dir_fd=stage_fd)
    else:
        _restore_stage_entry(directory_fd, target_name, stage_fd, "published")
    return stage_name, stage_fd


def _restore_stage_entry(
    directory_fd: int,
    target_name: str,
    stage_fd: int,
    entry_name: str,
) -> bool:
    try:
        os.link(
            entry_name,
            target_name,
            src_dir_fd=stage_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return False
    except OSError:
        return False
    os.unlink(entry_name, dir_fd=stage_fd)
    return True


def _remove_name_if_state(
    directory_fd: int,
    target_name: str,
    name: str,
    expected: FileState,
) -> None:
    stage_name: str | None = None
    stage_fd: int | None = None
    try:
        stage_name, stage_fd = _create_stage(directory_fd, target_name)
        try:
            os.rename(
                name,
                "cleanup",
                src_dir_fd=directory_fd,
                dst_dir_fd=stage_fd,
            )
        except FileNotFoundError:
            return
        current = capture_file(stage_fd, "cleanup")
        if current == expected:
            os.unlink("cleanup", dir_fd=stage_fd)
        else:
            _restore_stage_entry(directory_fd, name, stage_fd, "cleanup")
    except OSError as error:
        if error.errno not in {errno.ENOENT, errno.EEXIST}:
            raise
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        if stage_name is not None:
            try:
                os.rmdir(stage_name, dir_fd=directory_fd)
            except OSError:
                pass
