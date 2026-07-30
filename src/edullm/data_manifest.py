"""
Verify immutable eduLLM data manifests without following filesystem links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

ALLOWED_ROOTS = (Path("/orcd/pool"),)
MAX_MANIFEST_BYTES = 1_048_576
MAX_MANIFEST_FILES = 128
MAX_SHARD_BYTES = 8 * 1024**4
MAX_JSON_DEPTH = 20
MAX_INTEGER_TOKEN_CHARS = 20

_BUILTIN_GENERIC_SMOKE = {"generate_tiny_data": True, "kind": "generic-smoke"}
_BUILTIN_GENERIC_SMOKE_BYTES = json.dumps(
    _BUILTIN_GENERIC_SMOKE,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
BUILTIN_GENERIC_SMOKE_SHA256 = hashlib.sha256(_BUILTIN_GENERIC_SMOKE_BYTES).hexdigest()
BUILTINS = {
    "builtin://generic-smoke-v1": (
        BUILTIN_GENERIC_SMOKE_SHA256,
        _BUILTIN_GENERIC_SMOKE,
    )
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KIND = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_ENVIRONMENT_VALUE = re.compile(r"[^\x00-\x1f\x7f]{1,4096}\Z")
_MANIFEST_FIELDS = {
    "generic-smoke": frozenset({"data_dir", "files", "kind"}),
    "skill-dag": frozenset({"data_dir", "files", "kind", "mix_file"}),
    "curriculum": frozenset({"data_dir", "files", "kind", "order_file"}),
}
_FILE_FIELDS = frozenset({"path", "sha256", "size"})
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


class DataManifestError(ValueError):
    """A sanitized data-manifest validation failure."""


class _DuplicateJSONField(ValueError):
    pass


class _OversizedInteger(ValueError):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONField
        result[key] = value
    return result


def _validate_digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataManifestError(f"{name} digest is invalid")
    return cast(str, value)


def _validate_allowed_kinds(value: object) -> frozenset[str]:
    if type(value) not in {set, frozenset} or not value:
        raise DataManifestError("allowed dataset kinds are invalid")
    kinds = cast(set[object] | frozenset[object], value)
    if any(type(kind) is not str or _KIND.fullmatch(cast(str, kind)) is None for kind in kinds):
        raise DataManifestError("allowed dataset kinds are invalid")
    return frozenset(cast(Iterable[str], kinds))


def _validated_roots(roots: Sequence[Path]) -> tuple[Path, ...]:
    if not isinstance(roots, Sequence) or not roots:
        raise DataManifestError("approved data roots are invalid")
    result: list[Path] = []
    for root in roots:
        if not isinstance(root, Path) or not root.is_absolute():
            raise DataManifestError("approved data roots are invalid")
        if root in result:
            raise DataManifestError("approved data roots are invalid")
        result.append(root)
    return tuple(result)


def _validated_path(
    value: object,
    roots: Sequence[Path],
    name: str,
    *,
    allow_root: bool = False,
) -> tuple[Path, Path]:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DataManifestError(f"{name} path is invalid")
    pure = PurePosixPath(cast(str, value))
    if not pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts[1:]):
        raise DataManifestError(f"{name} path is outside approved roots")
    path = Path(cast(str, value))
    matches: list[Path] = []
    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts or allow_root:
            matches.append(root)
    if len(matches) != 1:
        raise DataManifestError(f"{name} path is outside approved roots")
    return path, matches[0]


def _snapshot(status: os.stat_result) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_uid,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _same_open_name(parent_fd: int, name: str, descriptor: int) -> bool:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return _snapshot(named) == _snapshot(os.fstat(descriptor))


@contextmanager
def _open_rooted(
    path: Path,
    root: Path,
    *,
    directory: bool,
    single_link: bool,
) -> Iterator[int]:
    root_fd: int | None = None
    descriptors: list[int] = []
    links: list[tuple[int, str, int]] = []
    try:
        try:
            root_before = root.stat(follow_symlinks=False)
            root_fd = os.open(root, _DIRECTORY_FLAGS)
        except OSError:
            raise DataManifestError("approved data root cannot be opened safely") from None
        if not stat.S_ISDIR(root_before.st_mode) or _snapshot(root_before) != _snapshot(
            os.fstat(root_fd)
        ):
            raise DataManifestError("approved data root changed during verification")

        current = root_fd
        relative = path.relative_to(root)
        parts = relative.parts
        if not parts:
            if not directory:
                raise DataManifestError("manifest path has an invalid file type")
            yield root_fd
            try:
                root_after = root.stat(follow_symlinks=False)
            except OSError:
                raise DataManifestError("approved data root changed during verification") from None
            if _snapshot(root_after) != _snapshot(os.fstat(root_fd)):
                raise DataManifestError("approved data root changed during verification")
            return
        for component in parts[:-1]:
            try:
                descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except OSError:
                raise DataManifestError("manifest path contains an unsafe link") from None
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(descriptor)
                raise DataManifestError("manifest path contains a non-directory")
            descriptors.append(descriptor)
            links.append((current, component, descriptor))
            current = descriptor

        final_flags = _DIRECTORY_FLAGS if directory else _FILE_FLAGS
        try:
            descriptor = os.open(parts[-1], final_flags, dir_fd=current)
        except OSError:
            raise DataManifestError("manifest path cannot be opened safely") from None
        descriptors.append(descriptor)
        links.append((current, parts[-1], descriptor))
        opened = os.fstat(descriptor)
        expected_type = stat.S_ISDIR(opened.st_mode) if directory else stat.S_ISREG(opened.st_mode)
        if not expected_type:
            raise DataManifestError("manifest path has an invalid file type")
        if single_link and opened.st_nlink != 1:
            raise DataManifestError("manifest or shard hardlink is not allowed")

        yield descriptor

        if any(not _same_open_name(parent, name, child) for parent, name, child in links):
            raise DataManifestError("manifest or shard changed during verification")
        try:
            root_after = root.stat(follow_symlinks=False)
        except OSError:
            raise DataManifestError("approved data root changed during verification") from None
        if _snapshot(root_after) != _snapshot(os.fstat(root_fd)):
            raise DataManifestError("approved data root changed during verification")
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if root_fd is not None:
            os.close(root_fd)


def _hash_descriptor(
    descriptor: int,
    *,
    maximum: int,
    retain_content: bool = True,
) -> tuple[str, bytes]:
    if type(maximum) is not int or maximum < 1:
        raise DataManifestError("verification size bound is invalid")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        content = bytearray()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if retain_content:
                content.extend(chunk)
            if total > maximum:
                raise DataManifestError("manifest or shard is too large")
        return digest.hexdigest(), bytes(content)
    except DataManifestError:
        raise
    except OSError:
        raise DataManifestError("manifest or shard cannot be read safely") from None


def _read_manifest(path: Path, root: Path, maximum: int) -> tuple[str, bytes]:
    with _open_rooted(path, root, directory=False, single_link=True) as descriptor:
        before = _snapshot(os.fstat(descriptor))
        digest, content = _hash_descriptor(descriptor, maximum=maximum)
        after = _snapshot(os.fstat(descriptor))
        if before != after:
            raise DataManifestError("manifest changed during verification")
        return digest, content


def _json_depth(value: object) -> int:
    maximum = 1
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        if maximum > MAX_JSON_DEPTH:
            return maximum
        if type(current) is dict:
            pending.extend(
                (item, depth + 1) for item in cast(dict[object, object], current).values()
            )
        elif type(current) is list:
            pending.extend((item, depth + 1) for item in cast(list[object], current))
    return maximum


def _decode_manifest(content: bytes) -> dict[str, object]:
    def bounded_integer(value: str) -> int:
        if len(value.lstrip("-")) > MAX_INTEGER_TOKEN_CHARS:
            raise _OversizedInteger
        return int(value)

    try:
        text = content.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_int=bounded_integer,
            parse_constant=lambda unused: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, ValueError, RecursionError, _OversizedInteger):
        raise DataManifestError("manifest is not valid bounded JSON") from None
    if type(value) is not dict or _json_depth(value) > MAX_JSON_DEPTH:
        raise DataManifestError("manifest is not valid bounded JSON")
    return cast(dict[str, object], value)


def _validate_manifest_schema(data: dict[str, object], allowed_kinds: frozenset[str]) -> str:
    kind = data.get("kind")
    if type(kind) is not str or kind not in _MANIFEST_FIELDS or kind not in allowed_kinds:
        raise DataManifestError("dataset kind is not allowed")
    if set(data) != _MANIFEST_FIELDS[kind]:
        raise DataManifestError("manifest fields are invalid")
    files = data["files"]
    if type(files) is not list:
        raise DataManifestError("manifest files must be a list")
    if not files or len(files) > MAX_MANIFEST_FILES:
        raise DataManifestError("manifest contains too many or too few files")
    return kind


def _verify_directory(path_value: object, roots: Sequence[Path], name: str) -> str:
    path, root = _validated_path(path_value, roots, name, allow_root=True)
    with _open_rooted(path, root, directory=True, single_link=False):
        pass
    return str(path)


def _verify_shards(
    rows: list[object],
    roots: Sequence[Path],
) -> set[str]:
    verified: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != _FILE_FIELDS:
            raise DataManifestError("manifest file fields are invalid")
        fields = cast(dict[str, object], row)
        path, root = _validated_path(fields["path"], roots, "shard")
        path_text = str(path)
        if path_text in verified:
            raise DataManifestError("manifest contains duplicate shard paths")
        size = fields["size"]
        if type(size) is not int or not 0 <= size <= MAX_SHARD_BYTES:
            raise DataManifestError("shard size is invalid")
        expected_digest = _validate_digest(fields["sha256"], "shard")
        with _open_rooted(path, root, directory=False, single_link=True) as descriptor:
            before = _snapshot(os.fstat(descriptor))
            if before[5] != size:
                raise DataManifestError("shard size mismatch")
            actual_digest, _ = _hash_descriptor(
                descriptor,
                maximum=max(size, 1),
                retain_content=False,
            )
            after = _snapshot(os.fstat(descriptor))
            if before != after:
                raise DataManifestError("shard changed during verification")
            if actual_digest != expected_digest:
                raise DataManifestError("shard digest mismatch")
        verified.add(path_text)
    return verified


def verify_manifest(
    uri: str,
    expected_sha256: str,
    allowed_kinds: set[str] | frozenset[str],
    *,
    allowed_roots: Sequence[Path] = ALLOWED_ROOTS,
    max_manifest_bytes: int = MAX_MANIFEST_BYTES,
) -> dict[str, object]:
    """
    Verify a fixed builtin or a strict manifest and every referenced shard.

    :param uri: Exact builtin URI or absolute manifest path below an approved root.
    :param expected_sha256: Canonical manifest identity supplied by the request.
    :param allowed_kinds: Protected entrypoint data-kind allowlist.
    :param allowed_roots: Trusted roots, injectable only for isolated tests.
    :param max_manifest_bytes: Finite manifest byte bound.

    :returns: A detached, strictly validated manifest mapping.

    :raises DataManifestError: If identity, schema, paths, or bytes are unsafe.
    """
    expected = _validate_digest(expected_sha256, "manifest")
    kinds = _validate_allowed_kinds(allowed_kinds)
    if type(uri) is not str:
        raise DataManifestError("manifest URI is invalid")
    if uri.startswith("builtin://"):
        builtin = BUILTINS.get(uri)
        if builtin is None:
            raise DataManifestError("unknown built-in dataset")
        digest, data = builtin
        if expected != digest:
            raise DataManifestError("built-in manifest digest mismatch")
        if data["kind"] not in kinds:
            raise DataManifestError("dataset kind is not allowed")
        return dict(data)
    if "://" in uri or not uri.startswith("/"):
        raise DataManifestError("manifest URI must use an approved local path")

    roots = _validated_roots(allowed_roots)
    path, root = _validated_path(uri, roots, "manifest")
    actual, content = _read_manifest(path, root, max_manifest_bytes)
    if actual != expected:
        raise DataManifestError("manifest digest mismatch")
    data = _decode_manifest(content)
    kind = _validate_manifest_schema(data, kinds)
    data_dir = _verify_directory(data["data_dir"], roots, "data_dir")
    verified = _verify_shards(cast(list[object], data["files"]), roots)
    required_key = {"skill-dag": "mix_file", "curriculum": "order_file"}.get(kind)
    if required_key is not None:
        required, _ = _validated_path(data[required_key], roots, required_key)
        if str(required) not in verified:
            raise DataManifestError(f"{required_key} is not bound to a verified shard")
    if not all(Path(path).is_relative_to(Path(data_dir)) for path in verified):
        raise DataManifestError("verified shard is outside data_dir")
    return json.loads(json.dumps(data, sort_keys=True, separators=(",", ":")))


def _runtime_value(value: object, name: str) -> str:
    if type(value) is not str or _ENVIRONMENT_VALUE.fullmatch(value) is None:
        raise DataManifestError(f"{name} runtime value is invalid")
    return cast(str, value)


def runtime_environment(data: Mapping[str, object]) -> dict[str, str]:
    """
    Convert verified typed manifest values into a small runtime environment.

    :param data: A manifest returned by :func:`verify_manifest`.

    :returns: Environment names and injection-safe values.
    """
    if not isinstance(data, Mapping):
        raise DataManifestError("runtime manifest is invalid")
    kind = data.get("kind")
    if kind == "generic-smoke":
        if data.get("generate_tiny_data") is True and set(data) == {
            "generate_tiny_data",
            "kind",
        }:
            return {
                "EDULLM_DATA_MODE": "synthetic",
                "OLMO_DATA_ROOT": str(Path.home() / "orcd/scratch/edullm/data/generic-smoke"),
            }
        required = {"data_dir", "files", "kind"}
        if set(data) != required:
            raise DataManifestError("runtime manifest is invalid")
        return {
            "EDULLM_DATA_MODE": "staged",
            "OLMO_DATA_ROOT": _runtime_value(data["data_dir"], "data_dir"),
        }
    if kind == "skill-dag":
        return {
            "SMOKE_MODE": "skill_dag",
            "SMOKE_DATA_DIR": _runtime_value(data.get("data_dir"), "data_dir"),
            "SMOKE_MIX_FILE": _runtime_value(data.get("mix_file"), "mix_file"),
        }
    if kind == "curriculum":
        return {
            "SMOKE_MODE": "curriculum",
            "SMOKE_DATA_DIR": _runtime_value(data.get("data_dir"), "data_dir"),
            "SMOKE_ORDER_FILE": _runtime_value(data.get("order_file"), "order_file"),
        }
    raise DataManifestError("runtime manifest is invalid")


def _render_environment(environment: Mapping[str, str]) -> str:
    names = {
        "EDULLM_DATA_MODE",
        "OLMO_DATA_ROOT",
        "SMOKE_DATA_DIR",
        "SMOKE_MIX_FILE",
        "SMOKE_MODE",
        "SMOKE_ORDER_FILE",
    }
    if any(name not in names for name in environment):
        raise DataManifestError("runtime environment name is invalid")
    return "".join(
        f"export {name}={shlex.quote(_runtime_value(value, name))}\n"
        for name, value in sorted(environment.items())
    )


def main(argv: Sequence[str] | None = None, *, output: BinaryIO | None = None) -> int:
    """Verify a manifest and optionally emit a private shell environment."""
    parser = argparse.ArgumentParser(prog="python -m edullm.data_manifest")
    parser.add_argument("command", choices=("verify", "render-env"))
    parser.add_argument("uri")
    parser.add_argument("sha256")
    parser.add_argument("--allowed-kind", action="append", required=True)
    arguments = parser.parse_args(argv)
    try:
        data = verify_manifest(
            arguments.uri,
            arguments.sha256,
            set(arguments.allowed_kind),
        )
        if arguments.command == "render-env":
            destination = sys.stdout.buffer if output is None else output
            destination.write(_render_environment(runtime_environment(data)).encode("utf-8"))
    except (DataManifestError, OSError):
        print("data manifest verification failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
