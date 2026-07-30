"""
Validate eduLLM requests and their machine-maintained status payloads.
"""

import dataclasses
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import cast

from edullm.data_manifest import BUILTINS
from edullm.models import JobRequest, JobStatus
from edullm.policy import Policy

STATUS_MARKER = "<!-- edullm-status:v1 -->"
# GitHub comments are bounded at 65,536 characters.
MAX_STATUS_COMMENT_CHARS = 65_536
# Ten decimal digits cover the reviewed signed-value domains without unbounded int conversion.
MAX_INTEGER_TOKEN_CHARS = 10
MAX_SEED = 2_147_483_647

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_UNSAFE_ARGUMENT = re.compile(r"[;&|`$<>\x00\n\r]")
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")
_BUILTIN_MANIFEST = re.compile(r"builtin://[a-z0-9][a-z0-9-]{0,63}")
_POOL_MANIFEST = re.compile(r"/orcd/pool/[A-Za-z0-9._/-]+")
_METRIC = re.compile(r"\S+")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_STATUS_MARKER_LINE = re.compile(rf"^{re.escape(STATUS_MARKER)}$", re.MULTILINE)
_DURATION = re.compile(
    r"\{\s*value:\s*(?P<value>-?(?:0|[1-9][0-9]*))\s*," r"\s*unit:\s*(?P<unit>[a-z]+)\s*\}"
)

_STATUS_FIELDS = {"request", "request_digest", "validated_at"}
_REQUEST_FIELDS = {field.name for field in dataclasses.fields(JobRequest)}
_REQUEST_INTEGER_FIELDS = {
    "issue_number",
    "seed",
    "gpu_count",
    "max_runtime_minutes",
}
_REQUEST_TUPLE_FIELDS = {"argv", "success_metrics"}
_REQUEST_STRING_FIELDS = (
    _REQUEST_FIELDS - _REQUEST_INTEGER_FIELDS - _REQUEST_TUPLE_FIELDS - {"status"}
)
_INVALID_OPTION_VALUE = object()


class _OversizedIntegerToken(ValueError):
    """Internal signal raised before converting an oversized JSON integer."""


class StatusCommentError(ValueError):
    """A fail-closed status-comment format or integrity error."""


@dataclass(frozen=True)
class ValidatedStatus:
    """Canonical request identity recorded after validation."""

    request: JobRequest
    request_digest: str
    validated_at: datetime


def _valid_repository_path(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    path = PurePosixPath(cast(str, value))
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and "\\" not in cast(str, value)
    )


def _parse_profile_arguments(
    arguments: tuple[str, ...],
    profile: Mapping[str, object],
    request: JobRequest,
    errors: list[str],
) -> None:
    fixed_launcher_arguments = tuple(
        cast(tuple[object, ...], profile.get("fixed_launcher_arguments", ()))
    )
    fixed_options = cast(Mapping[str, object], profile.get("fixed_options", {}))
    allowed_options = cast(Mapping[str, Mapping[str, object]], profile.get("allowed_options", {}))
    fixed_wandb_project = fixed_options.get("trainer.callbacks.wandb.project")
    if type(fixed_wandb_project) is str and request.wandb_project != fixed_wandb_project:
        errors.append("W&B project is fixed by entrypoint policy")
    fixed_seed = fixed_options.get("seed")
    if type(fixed_seed) is int and request.seed != fixed_seed:
        errors.append("seed is fixed by entrypoint policy")

    expected_positionals = profile.get("positionals")
    if type(expected_positionals) is not int or len(arguments) < expected_positionals:
        errors.append("positional arguments do not match the entrypoint profile")
        positional_count = 0 if type(expected_positionals) is not int else expected_positionals
    else:
        positional_count = expected_positionals
        if any(argument.startswith("--") for argument in arguments[:positional_count]):
            errors.append(f"the first {positional_count} arguments must be positional")

    for index, argument in enumerate(arguments[positional_count:], start=positional_count):
        if not argument.startswith("--"):
            errors.append(
                f"positional argument at index {index} is not allowed after options begin"
            )

    allowed_positionals = cast(Mapping[int, object], profile.get("allowed_positionals", {}))
    for position, rule in sorted(allowed_positionals.items()):
        if position >= positional_count or position >= len(arguments):
            continue
        value = arguments[position]
        if value.startswith("--"):
            continue
        if isinstance(rule, Mapping):
            if rule.get("type") != "slug" or _SLUG.fullmatch(value) is None:
                errors.append(f"positional argument {position} must be a lowercase slug")
        elif value not in cast(tuple[object, ...], rule):
            errors.append(f"positional argument {position} is not allowed")

    supplied_options: list[tuple[int, str, str]] = []
    for index, argument in enumerate(arguments):
        if not argument.startswith("--"):
            continue
        if argument in fixed_launcher_arguments:
            errors.append(f"argument at index {index} is fixed by launcher policy")
            continue

        candidate = argument[2:].split("=", 1)[0]
        known_name = (
            candidate if candidate in fixed_options or candidate in allowed_options else None
        )
        if "=" not in argument:
            if known_name is None:
                errors.append(f"option at index {index} must use --name=value form")
            else:
                errors.append(f"option --{known_name} at index {index} must use --name=value form")
            continue

        name, value = argument[2:].split("=", 1)
        if not name or not value:
            if known_name is None:
                errors.append(f"option at index {index} must use --name=value form")
            else:
                errors.append(f"option --{known_name} at index {index} must use --name=value form")
            continue
        supplied_options.append((index, name, value))

    counts = Counter(name for _, name, _ in supplied_options)
    for name in sorted(
        name
        for name, count in counts.items()
        if count > 1 and (name in fixed_options or name in allowed_options)
    ):
        errors.append(f"option may be supplied only once: --{name}")

    fixed_names = sorted({name for _, name, _ in supplied_options if name in fixed_options})
    for name in fixed_names:
        errors.append(f"option is fixed by policy and cannot be supplied: --{name}")

    for index, name, _ in supplied_options:
        if name not in fixed_options and name not in allowed_options:
            errors.append(f"option at index {index} is not allowed for this entrypoint")

    for name, rule in sorted(allowed_options.items()):
        if rule.get("required") is True and counts[name] == 0:
            errors.append(f"required option is missing: --{name}")

    for _, name, value in supplied_options:
        if name in fixed_options:
            continue
        rule = allowed_options.get(name)
        if rule is None:
            continue
        parsed = _validate_option(name, value, rule, errors)
        request_field = rule.get("request_field")
        if (
            parsed is not _INVALID_OPTION_VALUE
            and type(request_field) is str
            and parsed != getattr(request, request_field, _INVALID_OPTION_VALUE)
        ):
            errors.append(f"value for --{name} must match request.{request_field}")


def _validate_option(
    name: str,
    value: str,
    rule: Mapping[str, object],
    errors: list[str],
) -> object:
    allowed = True
    allowed_values = rule.get("values")
    if allowed_values is not None and value not in {
        str(item) for item in cast(tuple[object, ...], allowed_values)
    }:
        errors.append(f"value for --{name} is not allowed")
        allowed = False

    rule_type = rule.get("type")
    if rule_type == "integer":
        if _INTEGER.fullmatch(value) is None:
            errors.append(f"value for --{name} must be an integer")
            return _INVALID_OPTION_VALUE
        if len(value.lstrip("-")) > MAX_INTEGER_TOKEN_CHARS:
            errors.append(
                f"value for --{name} integer exceeds {MAX_INTEGER_TOKEN_CHARS} characters"
            )
            return _INVALID_OPTION_VALUE
        try:
            integer = int(value)
        except ValueError:
            errors.append(f"value for --{name} must be an integer")
            return _INVALID_OPTION_VALUE
        minimum = rule.get("min", integer)
        maximum = rule.get("max", integer)
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or integer < minimum
            or integer > maximum
        ):
            errors.append(f"value for --{name} is outside its allowed range")
            return _INVALID_OPTION_VALUE
        return integer if allowed else _INVALID_OPTION_VALUE
    elif rule_type == "path":
        roots = cast(tuple[object, ...], rule.get("roots", ()))
        path = PurePosixPath(value)
        within_root = False
        if ".." not in path.parts and "$" not in value and "\\" not in value:
            for root_value in roots:
                if type(root_value) is not str:
                    continue
                try:
                    path.relative_to(PurePosixPath(root_value))
                except ValueError:
                    continue
                within_root = True
                break
        if not within_root:
            errors.append(f"path for --{name} is outside allowed roots")
            return _INVALID_OPTION_VALUE
        return value if allowed else _INVALID_OPTION_VALUE
    elif rule_type == "boolean":
        if value not in {"true", "false"}:
            errors.append(f"value for --{name} must be true or false")
            return _INVALID_OPTION_VALUE
        return (value == "true") if allowed else _INVALID_OPTION_VALUE
    elif rule_type == "slug":
        if _SLUG.fullmatch(value) is None:
            errors.append(f"value for --{name} must be a lowercase slug")
            return _INVALID_OPTION_VALUE
        return value if allowed else _INVALID_OPTION_VALUE
    elif rule_type == "choice":
        if (
            type(allowed_values) not in {tuple, list}
            or not allowed_values
            or any(type(item) is not str for item in cast(tuple[object, ...], allowed_values))
        ):
            errors.append(f"validation rule for --{name} is invalid")
            return _INVALID_OPTION_VALUE
        return value if allowed else _INVALID_OPTION_VALUE
    elif rule_type == "duration":
        match = _DURATION.fullmatch(value)
        if match is None:
            errors.append(f"value for --{name} must be a duration mapping")
            return _INVALID_OPTION_VALUE
        integer_token = match.group("value")
        if len(integer_token.lstrip("-")) > MAX_INTEGER_TOKEN_CHARS:
            errors.append(
                f"value for --{name} integer exceeds {MAX_INTEGER_TOKEN_CHARS} characters"
            )
            return _INVALID_OPTION_VALUE
        try:
            steps = int(integer_token)
        except ValueError:
            errors.append(f"value for --{name} must be a duration mapping")
            return _INVALID_OPTION_VALUE
        unit = match.group("unit")
        max_steps = rule.get("max_steps")
        if unit != "steps" or type(max_steps) is not int or steps < 1 or steps > max_steps:
            errors.append(f"value for --{name} exceeds the allowed smoke duration")
            return _INVALID_OPTION_VALUE
        return {"value": steps, "unit": unit} if allowed else _INVALID_OPTION_VALUE
    else:
        errors.append(f"validation rule for --{name} is invalid")
        return _INVALID_OPTION_VALUE


def _valid_manifest_location(value: object) -> bool:
    if type(value) is not str:
        return False
    location = cast(str, value)
    if _BUILTIN_MANIFEST.fullmatch(location):
        return True
    if _POOL_MANIFEST.fullmatch(location) is None or "//" in location[1:]:
        return False
    if any(part in {"", ".", ".."} for part in location.removeprefix("/orcd/pool/").split("/")):
        return False
    path = PurePosixPath(location)
    try:
        relative = path.relative_to(PurePosixPath("/orcd/pool"))
    except ValueError:
        return False
    return bool(relative.parts)


def validate_request(request: JobRequest, policy: Policy) -> list[str]:
    """
    Return deterministic policy and safety errors for a job request.

    This validates only commit-SHA syntax. GitHub repository evidence is a
    separate automation and operator-submission gate that must bind the exact
    commit and protected script before execution.

    :param request: The immutable request to validate.
    :param policy: Trusted, recursively immutable queue policy.

    :returns: Actionable errors in stable validation order.
    """
    errors: list[str] = []

    if type(request.commit_sha) is not str or _COMMIT_SHA.fullmatch(request.commit_sha) is None:
        errors.append("commit SHA must be 40 lowercase hexadecimal characters")

    profile = (
        policy.entrypoints.get(request.entrypoint_profile)
        if type(request.entrypoint_profile) is str
        else None
    )
    if profile is None:
        errors.append("entrypoint profile is not allowed")
    elif request.script_path != profile.get("script") or request.launcher != profile.get(
        "launcher"
    ):
        errors.append("script and launcher do not match the entrypoint profile")
    if profile is not None:
        if type(profile.get("gpu_count")) is int and request.gpu_count != profile["gpu_count"]:
            errors.append("GPU count is fixed by entrypoint policy")
        if (
            type(profile.get("gpu_preference")) is str
            and request.gpu_preference != profile["gpu_preference"]
        ):
            errors.append("GPU preference is fixed by entrypoint policy")
        if (
            type(profile.get("max_runtime_minutes")) is int
            and request.max_runtime_minutes != profile["max_runtime_minutes"]
        ):
            errors.append("runtime is fixed by entrypoint policy")

    if not _valid_repository_path(request.script_path):
        errors.append("script path must be repository-relative without traversal")

    if type(request.launcher) is not str or request.launcher not in {"python", "torchrun", "bash"}:
        errors.append("launcher must be python, torchrun, or bash")

    valid_arguments = type(request.argv) is tuple
    if not valid_arguments:
        errors.append("arguments must be an immutable array of strings")
    else:
        for index, argument in enumerate(request.argv):
            if type(argument) is not str:
                errors.append(f"argument {index} must be a string")
                valid_arguments = False
        if valid_arguments:
            if any(not argument for argument in request.argv):
                errors.append("argument values must not be empty")
            for index, argument in enumerate(request.argv):
                if _UNSAFE_ARGUMENT.search(argument):
                    errors.append(f"unsafe argument at index {index}")
            if profile is not None:
                _parse_profile_arguments(request.argv, profile, request, errors)

    if (
        type(request.data_manifest_sha256) is not str
        or _DIGEST.fullmatch(request.data_manifest_sha256) is None
    ):
        errors.append("data manifest SHA-256 must be 64 lowercase hexadecimal characters")
    if not _valid_manifest_location(request.data_manifest):
        errors.append("data manifest location is not allowed")
    elif request.data_manifest.startswith("builtin://"):
        builtin = BUILTINS.get(request.data_manifest)
        if builtin is None:
            errors.append("built-in data manifest is unknown")
        else:
            builtin_digest, builtin_data = builtin
            if request.data_manifest_sha256 != builtin_digest:
                errors.append("built-in data manifest digest does not match its fixed identity")
            if profile is not None:
                allowed_kinds = profile.get("allowed_data_kinds")
                if type(allowed_kinds) not in {tuple, list}:
                    errors.append("built-in data kind is not allowed for this entrypoint")
                elif builtin_data["kind"] not in cast(
                    tuple[object, ...] | list[object],
                    allowed_kinds,
                ):
                    errors.append("built-in data kind is not allowed for this entrypoint")

    if (
        type(request.gpu_count) is not int
        or request.gpu_count < 1
        or request.gpu_count > policy.max_gpu_count
    ):
        errors.append(f"GPU count must be an integer from 1 to {policy.max_gpu_count}")
    if (
        type(request.max_runtime_minutes) is not int
        or request.max_runtime_minutes < 1
        or request.max_runtime_minutes > policy.max_runtime_minutes
    ):
        errors.append(f"runtime must be an integer from 1 to {policy.max_runtime_minutes} minutes")
    if (
        type(request.gpu_preference) is not str
        or request.gpu_preference not in policy.allowed_gpu_preferences
    ):
        errors.append("GPU preference is not allowed")
    if (
        type(request.wandb_project) is not str
        or request.wandb_project not in policy.allowed_wandb_projects
    ):
        errors.append("W&B project is not allowed")

    if type(request.data_classification) is not str or request.data_classification not in {
        "public",
        "research-cleared",
        "restricted",
    }:
        errors.append("data classification is invalid")
    elif request.data_classification == "restricted":
        errors.append("restricted data is not accepted by the public pilot queue")

    if type(request.seed) is not int or not 0 <= request.seed <= MAX_SEED:
        errors.append(f"seed must be an integer from 0 to {MAX_SEED}")

    for field_name in ("purpose", "study", "condition", "comparison", "success_signal"):
        value = getattr(request, field_name)
        if type(value) is not str or not value.strip():
            errors.append(f"{field_name.replace('_', ' ')} must not be empty")

    if type(request.success_metrics) is not tuple or any(
        type(metric) is not str for metric in request.success_metrics
    ):
        errors.append("success metrics must be an immutable array of strings")
    elif not request.success_metrics:
        errors.append("at least one emitted success metric is required")
    else:
        if any(_METRIC.fullmatch(metric) is None for metric in request.success_metrics):
            errors.append("success metric names must be non-empty and contain no whitespace")
        if len(set(request.success_metrics)) != len(request.success_metrics):
            errors.append("success metric names must not be duplicated")

    return errors


def _format_validation_timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.utcoffset() != timedelta(0)
        or value.microsecond != 0
    ):
        raise StatusCommentError("validation timestamp must be UTC with whole-second precision")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bounded_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > MAX_INTEGER_TOKEN_CHARS:
        raise _OversizedIntegerToken
    try:
        return int(value)
    except ValueError as error:
        raise ValueError("invalid JSON integer") from error


def build_status_comment(request: JobRequest, *, validated_at: datetime) -> str:
    """
    Build the canonical v1 status comment for a successfully validated request.

    The caller must first validate the request against trusted policy. This
    function owns only the deterministic status data contract.

    :param request: A successfully validated immutable request.
    :param validated_at: Its timezone-aware, whole-second validation time.

    :returns: The exact marker and canonical JSON payload.

    :raises StatusCommentError: If the timestamp is not canonicalizable.
    """
    try:
        request_json = request.canonical_json()
        request_data = json.loads(request_json, parse_int=_bounded_json_integer)
    except _OversizedIntegerToken as error:
        raise StatusCommentError(
            f"validated status contains an integer longer than {MAX_INTEGER_TOKEN_CHARS} characters"
        ) from error
    except (ValueError, RecursionError) as error:
        raise StatusCommentError("validated request cannot be serialized") from error
    payload = {
        "request": request_data,
        "request_digest": hashlib.sha256(request_json.encode()).hexdigest(),
        "validated_at": _format_validation_timestamp(validated_at),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    comment = f"{STATUS_MARKER}\n{encoded}"
    if len(comment) > MAX_STATUS_COMMENT_CHARS:
        raise StatusCommentError(
            f"validated status comment exceeds {MAX_STATUS_COMMENT_CHARS} characters"
        )
    return comment


def _request_from_payload(value: object) -> JobRequest:
    if type(value) is not dict:
        raise StatusCommentError("validated request is malformed")
    data = cast(dict[str, object], value)
    if set(data) != _REQUEST_FIELDS:
        raise StatusCommentError("validated request is malformed")

    if any(type(data[field]) is not int for field in _REQUEST_INTEGER_FIELDS):
        raise StatusCommentError("validated request is malformed")
    if any(type(data[field]) is not str for field in _REQUEST_STRING_FIELDS):
        raise StatusCommentError("validated request is malformed")
    for field in _REQUEST_TUPLE_FIELDS:
        items = data[field]
        if type(items) is not list or any(
            type(item) is not str for item in cast(list[object], items)
        ):
            raise StatusCommentError("validated request is malformed")

    status_value = data["status"]
    if type(status_value) is not str:
        raise StatusCommentError("validated request is malformed")
    try:
        status = JobStatus(status_value)
    except ValueError as error:
        raise StatusCommentError("validated request is malformed") from error

    try:
        return JobRequest(
            issue_number=cast(int, data["issue_number"]),
            requester=cast(str, data["requester"]),
            purpose=cast(str, data["purpose"]),
            study=cast(str, data["study"]),
            condition=cast(str, data["condition"]),
            comparison=cast(str, data["comparison"]),
            commit_sha=cast(str, data["commit_sha"]),
            entrypoint_profile=cast(str, data["entrypoint_profile"]),
            script_path=cast(str, data["script_path"]),
            launcher=cast(str, data["launcher"]),
            argv=tuple(cast(list[str], data["argv"])),
            data_manifest=cast(str, data["data_manifest"]),
            data_manifest_sha256=cast(str, data["data_manifest_sha256"]),
            data_classification=cast(str, data["data_classification"]),
            seed=cast(int, data["seed"]),
            wandb_project=cast(str, data["wandb_project"]),
            success_signal=cast(str, data["success_signal"]),
            success_metrics=tuple(cast(list[str], data["success_metrics"])),
            gpu_count=cast(int, data["gpu_count"]),
            gpu_preference=cast(str, data["gpu_preference"]),
            max_runtime_minutes=cast(int, data["max_runtime_minutes"]),
            status=status,
        )
    except (TypeError, ValueError) as error:
        raise StatusCommentError("validated request is malformed") from error


def parse_status_comment(comment: str) -> ValidatedStatus:
    """
    Parse and integrity-check one exact v1 machine status comment.

    :param comment: The complete machine-maintained comment body.

    :returns: The canonical validated status.

    :raises StatusCommentError: If the marker, schema, canonical encoding,
        timestamp, request, or digest is invalid.
    """
    if type(comment) is not str:
        raise StatusCommentError("validated status marker is missing")
    if len(comment) > MAX_STATUS_COMMENT_CHARS:
        raise StatusCommentError(
            f"validated status comment exceeds {MAX_STATUS_COMMENT_CHARS} characters"
        )
    marker_count = len(_STATUS_MARKER_LINE.findall(comment))
    if marker_count == 0:
        raise StatusCommentError("validated status marker is missing")
    if marker_count != 1:
        raise StatusCommentError("validated status marker must appear exactly once")
    prefix = STATUS_MARKER + "\n"
    if not comment.startswith(prefix):
        raise StatusCommentError("validated status comment must contain only marker and payload")

    encoded = comment[len(prefix) :]
    try:
        payload = json.loads(encoded, parse_int=_bounded_json_integer)
    except _OversizedIntegerToken as error:
        raise StatusCommentError(
            f"validated status contains an integer longer than {MAX_INTEGER_TOKEN_CHARS} characters"
        ) from error
    except (ValueError, RecursionError) as error:
        raise StatusCommentError("validated status payload is not valid JSON") from error
    if type(payload) is not dict:
        raise StatusCommentError("validated status payload fields are invalid")
    if encoded != json.dumps(payload, sort_keys=True, separators=(",", ":")):
        raise StatusCommentError("validated status payload must use canonical JSON")

    data = cast(dict[str, object], payload)
    if set(data) != _STATUS_FIELDS:
        raise StatusCommentError("validated status payload fields are invalid")

    timestamp_text = data["validated_at"]
    if type(timestamp_text) is not str or _TIMESTAMP.fullmatch(timestamp_text) is None:
        raise StatusCommentError("validation timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        validated_at = datetime.strptime(timestamp_text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise StatusCommentError("validation timestamp must use YYYY-MM-DDTHH:MM:SSZ") from error

    digest = data["request_digest"]
    if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
        raise StatusCommentError(
            "validated request digest must be 64 lowercase hexadecimal characters"
        )

    request = _request_from_payload(data["request"])
    recomputed = hashlib.sha256(request.canonical_json().encode()).hexdigest()
    if digest != recomputed:
        raise StatusCommentError("validated request digest does not match canonical request")

    return ValidatedStatus(
        request=request,
        request_digest=digest,
        validated_at=validated_at,
    )


def validated_status_for_request(comment: str, request: JobRequest) -> ValidatedStatus:
    """
    Require one status comment to match the currently parsed Issue request.

    :param comment: The complete machine-maintained status comment.
    :param request: The current Issue body parsed as a request.

    :returns: The matching validated status.

    :raises StatusCommentError: If status data is malformed, tampered, or stale.
    """
    status = parse_status_comment(comment)
    if (
        status.request_digest != request.digest
        or status.request.canonical_json() != request.canonical_json()
    ):
        raise StatusCommentError("validated status is stale for the current request")
    return status
