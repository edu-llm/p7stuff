from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast

import pytest

from edullm.data_manifest import (
    BUILTIN_GENERIC_SMOKE_SHA256,
    DataManifestError,
    runtime_environment,
    verify_manifest,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_manifest(
    root: Path,
    *,
    kind: str = "skill-dag",
    shard_names: tuple[str, ...] = ("data/a.npy", "data/b.npy"),
    extra: dict[str, object] | None = None,
) -> tuple[Path, str, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, name in enumerate(shard_names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"shard-{index}".encode()
        path.write_bytes(content)
        rows.append({"path": str(path), "sha256": _sha256(content), "size": len(content)})
    data: dict[str, object] = {
        "data_dir": str(root / "data"),
        "files": rows,
        "kind": kind,
        "mix_file": str(root / shard_names[0]),
    }
    if kind == "curriculum":
        data.pop("mix_file")
        data["order_file"] = str(root / shard_names[0])
    elif kind == "generic-smoke":
        data.pop("mix_file")
    if extra:
        data.update(extra)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path = root / "manifest.json"
    path.write_bytes(encoded)
    return path, _sha256(encoded), data


def test_builtin_requires_exact_fixed_canonical_identity():
    verified = verify_manifest(
        "builtin://generic-smoke-v1",
        BUILTIN_GENERIC_SMOKE_SHA256,
        {"generic-smoke"},
    )

    assert verified == {"generate_tiny_data": True, "kind": "generic-smoke"}
    with pytest.raises(DataManifestError, match="digest"):
        verify_manifest("builtin://generic-smoke-v1", "b" * 64, {"generic-smoke"})
    with pytest.raises(DataManifestError, match="unknown"):
        verify_manifest("builtin://other", "0" * 64, {"generic-smoke"})


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/manifest.json",
        "https://example.invalid/manifest.json",
        "relative/manifest.json",
        "../manifest.json",
        "/tmp/manifest.json",
    ],
)
def test_rejects_network_relative_and_unapproved_manifest_uris(tmp_path, uri):
    with pytest.raises(DataManifestError):
        verify_manifest(uri, "0" * 64, {"skill-dag"}, allowed_roots=(tmp_path / "pool",))


def test_verifies_valid_multi_shard_manifest_and_runtime_environment(tmp_path):
    root = tmp_path / "pool"
    path, digest, data = _write_manifest(root)

    verified = verify_manifest(
        str(path),
        digest,
        {"skill-dag"},
        allowed_roots=(root,),
    )

    assert verified == data
    assert runtime_environment(verified) == {
        "SMOKE_DATA_DIR": str(root / "data"),
        "SMOKE_MIX_FILE": str(root / "data/a.npy"),
        "SMOKE_MODE": "skill_dag",
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda data: data.update(extra=True), "fields"),
        (lambda data: data.pop("files"), "fields"),
        (lambda data: data.update(kind="unknown"), "kind"),
        (lambda data: data.update(files={}), "files"),
        (lambda data: data["files"][0].update(extra=True), "file"),
        (lambda data: data["files"][0].pop("size"), "file"),
        (lambda data: data["files"][0].update(size=True), "size"),
        (lambda data: data["files"][0].update(sha256="A" * 64), "digest"),
        (lambda data: data.update(data_dir=1), "data_dir"),
        (lambda data: data.update(mix_file=1), "mix_file"),
    ],
)
def test_rejects_malformed_extra_missing_and_type_confused_fields(
    tmp_path,
    mutate,
    match,
):
    root = tmp_path / "pool"
    path, _, data = _write_manifest(root)
    mutate(data)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    with pytest.raises(DataManifestError, match=match):
        verify_manifest(str(path), _sha256(encoded), {"skill-dag"}, allowed_roots=(root,))


def test_rejects_duplicate_json_fields_and_duplicate_shards(tmp_path):
    root = tmp_path / "pool"
    path, _, data = _write_manifest(root)
    duplicate_json = b'{"kind":"skill-dag","kind":"skill-dag"}'
    path.write_bytes(duplicate_json)
    with pytest.raises(DataManifestError, match="JSON"):
        verify_manifest(
            str(path),
            _sha256(duplicate_json),
            {"skill-dag"},
            allowed_roots=(root,),
        )

    path, _, data = _write_manifest(root)
    files = cast(list[dict[str, object]], data["files"])
    files.append(dict(files[0]))
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(DataManifestError, match="duplicate"):
        verify_manifest(str(path), _sha256(encoded), {"skill-dag"}, allowed_roots=(root,))


def test_rejects_wrong_manifest_shard_size_and_hash(tmp_path):
    root = tmp_path / "pool"
    path, digest, data = _write_manifest(root)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(DataManifestError, match="manifest digest"):
        verify_manifest(str(path), digest, {"skill-dag"}, allowed_roots=(root,))

    path, _, data = _write_manifest(root)
    files = cast(list[dict[str, object]], data["files"])
    files[0]["size"] = cast(int, files[0]["size"]) + 1
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(DataManifestError, match="size"):
        verify_manifest(str(path), _sha256(encoded), {"skill-dag"}, allowed_roots=(root,))

    path, _, data = _write_manifest(root)
    files = cast(list[dict[str, object]], data["files"])
    files[0]["sha256"] = "0" * 64
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(DataManifestError, match="shard digest"):
        verify_manifest(str(path), _sha256(encoded), {"skill-dag"}, allowed_roots=(root,))


def test_required_mix_or_order_file_must_bind_to_verified_shard(tmp_path):
    root = tmp_path / "pool"
    path, _, data = _write_manifest(root)
    unbound = root / "data/unbound.json"
    unbound.write_text("{}", encoding="utf-8")
    data["mix_file"] = str(unbound)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    with pytest.raises(DataManifestError, match="bound"):
        verify_manifest(str(path), _sha256(encoded), {"skill-dag"}, allowed_roots=(root,))


def test_rejects_traversal_symlink_escape_and_hardlinks(tmp_path):
    root = tmp_path / "pool"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret"
    secret.write_bytes(b"secret")
    root.mkdir()

    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    data: dict[str, object] = {
        "data_dir": str(root),
        "files": [
            {
                "path": str(link / "secret"),
                "sha256": _sha256(b"secret"),
                "size": 6,
            }
        ],
        "kind": "generic-smoke",
    }
    path = root / "manifest.json"
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(DataManifestError):
        verify_manifest(str(path), _sha256(encoded), {"generic-smoke"}, allowed_roots=(root,))

    hardlink = root / "hardlink"
    os.link(secret, hardlink)
    files = cast(list[dict[str, object]], data["files"])
    files[0]["path"] = str(hardlink)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(DataManifestError, match="link"):
        verify_manifest(str(path), _sha256(encoded), {"generic-smoke"}, allowed_roots=(root,))


def test_rejects_manifest_symlink_and_hardlink(tmp_path):
    root = tmp_path / "pool"
    path, digest, _ = _write_manifest(root)
    symlink = root / "manifest-link.json"
    symlink.symlink_to(path)
    with pytest.raises(DataManifestError):
        verify_manifest(str(symlink), digest, {"skill-dag"}, allowed_roots=(root,))

    hardlink = root / "manifest-hardlink.json"
    os.link(path, hardlink)
    with pytest.raises(DataManifestError, match="link"):
        verify_manifest(str(hardlink), digest, {"skill-dag"}, allowed_roots=(root,))


def test_rejects_swapped_shard_during_descriptor_verification(tmp_path, monkeypatch):
    root = tmp_path / "pool"
    path, digest, _ = _write_manifest(root)
    import edullm.data_manifest as module

    real_hash = module._hash_descriptor
    swapped = False

    def hash_then_swap(descriptor, *, maximum, retain_content=True):
        nonlocal swapped
        result = real_hash(
            descriptor,
            maximum=maximum,
            retain_content=retain_content,
        )
        if not swapped and os.fstat(descriptor).st_size == len(b"shard-0"):
            swapped = True
            shard = root / "data/a.npy"
            replacement = root / "data/replacement"
            replacement.write_bytes(b"changed")
            replacement.replace(shard)
        return result

    monkeypatch.setattr(module, "_hash_descriptor", hash_then_swap)

    with pytest.raises(DataManifestError, match="changed"):
        verify_manifest(str(path), digest, {"skill-dag"}, allowed_roots=(root,))


def test_bounded_manifest_bytes_file_count_depth_and_shard_sizes(tmp_path):
    root = tmp_path / "pool"
    path, digest, _ = _write_manifest(root)
    with pytest.raises(DataManifestError, match="large"):
        verify_manifest(
            str(path),
            digest,
            {"skill-dag"},
            allowed_roots=(root,),
            max_manifest_bytes=8,
        )

    path, _, data = _write_manifest(root)
    data["files"] = cast(list[dict[str, object]], data["files"]) * 65
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(DataManifestError, match="many"):
        verify_manifest(str(path), _sha256(encoded), {"skill-dag"}, allowed_roots=(root,))

    bomb = b'{"a":' * 40 + b"0" + b"}" * 40
    path.write_bytes(bomb)
    with pytest.raises(DataManifestError, match="JSON"):
        verify_manifest(str(path), _sha256(bomb), {"skill-dag"}, allowed_roots=(root,))

    integer_bomb = (
        b'{"kind":"skill-dag","data_dir":"/orcd/pool","files":[],"mix_file":'
        + (b"9" * 100_000)
        + b"}"
    )
    path.write_bytes(integer_bomb)
    with pytest.raises(DataManifestError, match="JSON"):
        verify_manifest(
            str(path),
            _sha256(integer_bomb),
            {"skill-dag"},
            allowed_roots=(root,),
        )


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "skill-dag", "data_dir": "x\nBAD=1", "mix_file": "/orcd/pool/a"},
        {"kind": "curriculum", "data_dir": "/orcd/pool/a", "order_file": "\x00"},
        {"kind": "unknown"},
    ],
)
def test_runtime_environment_revalidates_typed_values(data):
    with pytest.raises(DataManifestError):
        runtime_environment(data)
