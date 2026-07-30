import importlib
import os
import stat
from pathlib import Path

import pytest


def _module():
    return importlib.import_module("edullm.secure_publish")


def _open_directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))


def _write_at(directory_fd: int, name: str, content: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
        dir_fd=directory_fd,
    )
    try:
        os.fchmod(descriptor, mode)
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepared_publish(tmp_path, *, existing=b"original", proposed=b"proposed"):
    publish = _module()
    parent = tmp_path / "private"
    parent.mkdir()
    directory_fd = _open_directory(parent)
    if existing is not None:
        _write_at(directory_fd, "target", existing)
    expected = publish.capture_file(directory_fd, "target", exact_mode=0o600)
    _write_at(directory_fd, "temporary", proposed)
    prepared = publish.capture_file(directory_fd, "temporary", exact_mode=0o600)
    assert prepared is not None
    return publish, parent, directory_fd, expected, prepared


def test_file_state_representations_never_include_private_content():
    publish = _module()
    private_content = b"edullm-distinctive-private-repr-regression"
    state = publish.FileState(
        device=11,
        inode=22,
        mode=stat.S_IFREG | 0o600,
        owner=33,
        content=private_content,
    )
    error = publish.SecurePublishError({"state": state})
    renderings = (
        repr(state),
        str(state),
        repr([state]),
        repr({"captured": state}),
        repr(error),
        str(error),
        repr({"error": error}),
    )

    private_text = private_content.decode("ascii")
    if any(private_text in rendering for rendering in renderings):
        pytest.fail("private file content appeared in a state representation", pytrace=False)
    assert all("device=11" in rendering for rendering in renderings)
    assert "mode=33152" in repr(state)
    assert "content=" not in repr(state)


@pytest.mark.parametrize("existing", [None, b"original"])
def test_compare_and_publish_succeeds_for_new_and_existing_target(tmp_path, existing):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(
        tmp_path, existing=existing
    )
    try:
        publish.compare_and_publish(
            directory_fd,
            parent,
            publish.directory_identity(directory_fd),
            "target",
            "temporary",
            expected,
            prepared,
        )
    finally:
        os.close(directory_fd)

    assert (parent / "target").read_bytes() == b"proposed"
    assert stat.S_IMODE((parent / "target").stat().st_mode) == 0o600
    assert not (parent / "temporary").exists()
    assert not list(parent.glob(".target.edullm-recovery-*"))


def test_compare_and_publish_uses_descriptor_chmod_for_recovery_directory(tmp_path, monkeypatch):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(tmp_path)

    def unsupported_path_chmod(*args, **kwargs):
        raise ValueError("dir_fd and follow_symlinks are incompatible")

    monkeypatch.setattr(publish.os, "chmod", unsupported_path_chmod)
    try:
        publish.compare_and_publish(
            directory_fd,
            parent,
            publish.directory_identity(directory_fd),
            "target",
            "temporary",
            expected,
            prepared,
        )
    finally:
        os.close(directory_fd)

    assert (parent / "target").read_bytes() == b"proposed"
    assert stat.S_IMODE((parent / "target").stat().st_mode) == 0o600
    assert not list(parent.glob(".target.edullm-recovery-*"))


def test_compare_and_publish_restores_content_edited_at_publish_boundary(tmp_path, monkeypatch):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(tmp_path)
    concurrent_fd = os.open("target", os.O_WRONLY, dir_fd=directory_fd)
    real_link = publish.os.link

    def edit_then_publish(source, target, **kwargs):
        if source == "temporary" and target == "target":
            os.lseek(concurrent_fd, 0, os.SEEK_SET)
            os.write(concurrent_fd, b"concurrent")
            os.ftruncate(concurrent_fd, len(b"concurrent"))
            os.fsync(concurrent_fd)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(publish.os, "link", edit_then_publish)
    try:
        with pytest.raises(publish.SecurePublishError, match="safe publication failed"):
            publish.compare_and_publish(
                directory_fd,
                parent,
                publish.directory_identity(directory_fd),
                "target",
                "temporary",
                expected,
                prepared,
            )
    finally:
        os.close(concurrent_fd)
        os.close(directory_fd)

    assert (parent / "target").read_bytes() == b"concurrent"
    assert not list(parent.glob(".target.edullm-recovery-*"))


def test_compare_and_publish_never_clobbers_boundary_path_replacement(tmp_path, monkeypatch):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(tmp_path)
    real_link = publish.os.link
    replaced = False

    def replace_then_publish(source, target, **kwargs):
        nonlocal replaced
        if source == "temporary" and target == "target" and not replaced:
            replaced = True
            _write_at(directory_fd, "target", b"concurrent")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(publish.os, "link", replace_then_publish)
    previous_umask = os.umask(0o777)
    try:
        with pytest.raises(publish.SecurePublishError, match="safe publication failed"):
            publish.compare_and_publish(
                directory_fd,
                parent,
                publish.directory_identity(directory_fd),
                "target",
                "temporary",
                expected,
                prepared,
            )
    finally:
        os.umask(previous_umask)
        os.close(directory_fd)

    assert (parent / "target").read_bytes() == b"concurrent"
    recovery = list(parent.glob(".target.edullm-recovery-*/original"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == b"original"
    assert stat.S_IMODE(recovery[0].parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(recovery[0].stat().st_mode) == 0o600
    lock = parent / ".edullm-lock-target"
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    assert not (parent / "temporary").exists()


def test_compare_and_publish_rejects_parent_swap_at_publish_boundary(tmp_path, monkeypatch):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(tmp_path)
    moved = tmp_path / "private-moved"
    real_link = publish.os.link
    swapped = False

    def swap_then_publish(source, target, **kwargs):
        nonlocal swapped
        if source == "temporary" and target == "target" and not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(publish.os, "link", swap_then_publish)
    try:
        with pytest.raises(publish.SecurePublishError, match="safe publication failed"):
            publish.compare_and_publish(
                directory_fd,
                parent,
                publish.directory_identity(directory_fd),
                "target",
                "temporary",
                expected,
                prepared,
            )
    finally:
        os.close(directory_fd)

    assert not (parent / "target").exists()
    assert (moved / "target").read_bytes() == b"original"
    assert not list(moved.glob(".target.edullm-recovery-*"))
    assert not (moved / "temporary").exists()


def test_compare_and_publish_rollback_never_clobbers_concurrent_recreation(tmp_path, monkeypatch):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(tmp_path)
    concurrent_fd = os.open("target", os.O_WRONLY, dir_fd=directory_fd)
    real_link = publish.os.link
    real_rename = publish.os.rename
    published = False
    recreated = False

    def publish_then_edit_original(source, target, **kwargs):
        nonlocal published
        result = real_link(source, target, **kwargs)
        if source == "temporary" and target == "target" and not published:
            published = True
            os.lseek(concurrent_fd, 0, os.SEEK_SET)
            os.write(concurrent_fd, b"changed-original")
            os.ftruncate(concurrent_fd, len(b"changed-original"))
            os.fsync(concurrent_fd)
        return result

    def recreate_during_rollback(source, target, **kwargs):
        nonlocal recreated
        result = real_rename(source, target, **kwargs)
        if source == "target" and target == "published" and not recreated:
            recreated = True
            _write_at(directory_fd, "target", b"concurrent-recreation")
        return result

    monkeypatch.setattr(publish.os, "link", publish_then_edit_original)
    monkeypatch.setattr(publish.os, "rename", recreate_during_rollback)
    try:
        with pytest.raises(publish.SecurePublishError, match="safe publication failed"):
            publish.compare_and_publish(
                directory_fd,
                parent,
                publish.directory_identity(directory_fd),
                "target",
                "temporary",
                expected,
                prepared,
            )
    finally:
        os.close(concurrent_fd)
        os.close(directory_fd)

    assert (parent / "target").read_bytes() == b"concurrent-recreation"
    recovery = list(parent.glob(".target.edullm-recovery-*/original"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == b"changed-original"


def test_compare_and_publish_restores_original_when_precommit_fsync_fails(tmp_path, monkeypatch):
    publish, parent, directory_fd, expected, prepared = _prepared_publish(tmp_path)

    def fail_fsync(descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(publish.os, "fsync", fail_fsync)
    try:
        with pytest.raises(publish.SecurePublishError, match="safe publication failed"):
            publish.compare_and_publish(
                directory_fd,
                parent,
                publish.directory_identity(directory_fd),
                "target",
                "temporary",
                expected,
                prepared,
            )
    finally:
        os.close(directory_fd)

    assert (parent / "target").read_bytes() == b"original"
    assert not list(parent.glob(".target.edullm-recovery-*"))
    assert not (parent / "temporary").exists()
