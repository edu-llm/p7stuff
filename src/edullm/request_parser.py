"""
Parse deterministic eduLLM GitHub Issue-form requests.
"""

import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import cast

from edullm.models import JobRequest

ISSUE_HEADINGS = (
    "Purpose",
    "Study",
    "Condition",
    "Comparison",
    "Commit SHA",
    "Entrypoint profile",
    "Script path",
    "Launcher",
    "Arguments JSON",
    "Data manifest",
    "Data manifest SHA-256",
    "Data classification",
    "Seed",
    "W&B project",
    "Success signal",
    "Success metrics",
    "GPU count",
    "GPU preference",
    "Maximum runtime minutes",
)

# Match GitHub's Issue body ceiling and keep nested parsing substantially smaller.
MAX_ISSUE_BODY_CHARS = 65_536
MAX_ARGUMENTS_JSON_CHARS = 4_096
# Covers the reviewed 32-bit seed maximum before any decimal conversion.
MAX_INTEGER_TOKEN_CHARS = 10

_HEADING = re.compile(r"^### (?P<name>[^\r\n]+)\r?$", re.MULTILINE)
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")


class IssueParseError(ValueError):
    """Actionable errors found while parsing an Issue-form request."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _shape_errors(names: list[str]) -> list[str]:
    counts = Counter(names)
    errors = []
    for heading in ISSUE_HEADINGS:
        if counts[heading] == 0:
            errors.append(f"missing heading: {heading}")
    for heading in ISSUE_HEADINGS:
        if counts[heading] > 1:
            errors.append(f"duplicate heading: {heading}")

    expected = set(ISSUE_HEADINGS)
    errors.extend(
        f"unexpected heading at index {index}"
        for index, name in enumerate(names)
        if name not in expected
    )

    if not errors and names != list(ISSUE_HEADINGS):
        errors.append("headings must appear in Issue-form order")
    return errors


def fields_from_markdown(body: str) -> dict[str, str]:
    """
    Extract exact Issue-form fields from Markdown.

    :param body: The GitHub Issue body.

    :returns: Field values keyed by their exact headings.

    :raises IssueParseError: If headings or field values are malformed.
    """
    if type(body) is not str:
        raise IssueParseError(("Issue body must be text",))
    if len(body) > MAX_ISSUE_BODY_CHARS:
        raise IssueParseError((f"Issue body exceeds {MAX_ISSUE_BODY_CHARS} characters",))

    matches = list(_HEADING.finditer(body))
    names = [match.group("name") for match in matches]
    errors = _shape_errors(names)
    if not errors and matches and body[: matches[0].start()].strip():
        errors.append("unexpected content before first heading")
    if errors:
        raise IssueParseError(tuple(errors))

    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        fields[match.group("name")] = body[start:end].strip()

    empty = [f"{heading} must not be empty" for heading in ISSUE_HEADINGS if not fields[heading]]
    if empty:
        raise IssueParseError(tuple(empty))
    return fields


def issue_body_from_fields(fields: Mapping[str, str]) -> str:
    """Render the exact Issue-form Markdown accepted by :func:`parse_issue`."""
    if not isinstance(fields, Mapping):
        raise IssueParseError(("Issue fields must be a mapping",))
    names = list(fields)
    errors = _shape_errors(names)
    for heading in ISSUE_HEADINGS:
        if heading in fields and type(fields[heading]) is not str:
            errors.append(f"{heading} must be text")
    if errors:
        raise IssueParseError(tuple(errors))
    typed = cast(Mapping[str, str], fields)
    return "\n\n".join(f"### {heading}\n{typed[heading]}" for heading in ISSUE_HEADINGS)


def _parse_arguments(value: str, errors: list[str]) -> tuple[str, ...]:
    if len(value) > MAX_ARGUMENTS_JSON_CHARS:
        errors.append(f"Arguments JSON exceeds {MAX_ARGUMENTS_JSON_CHARS} characters")
        return ()
    try:
        parsed = json.loads(value, parse_int=_bounded_json_integer)
    except (ValueError, RecursionError):
        errors.append("Arguments JSON must be valid JSON")
        return ()
    if type(parsed) is not list:
        errors.append("Arguments JSON must be an array of strings")
        return ()

    arguments = cast(list[object], parsed)
    for index, argument in enumerate(arguments):
        if type(argument) is not str:
            errors.append(f"Arguments JSON[{index}] must be a string")
    if errors:
        return ()
    return tuple(cast(list[str], arguments))


def _bounded_json_integer(value: str) -> int:
    if len(value) > MAX_INTEGER_TOKEN_CHARS:
        raise ValueError("JSON integer token is too long")
    return int(value)


def _parse_integer(value: str, heading: str, errors: list[str]) -> int:
    if len(value) > MAX_INTEGER_TOKEN_CHARS:
        errors.append(f"{heading} integer exceeds {MAX_INTEGER_TOKEN_CHARS} characters")
        return 0
    if _INTEGER.fullmatch(value) is None:
        errors.append(f"{heading} must be an integer")
        return 0
    try:
        return int(value)
    except ValueError:
        errors.append(f"{heading} must be an integer")
        return 0


def parse_issue(body: str, *, issue_number: int, requester: str) -> JobRequest:
    """
    Parse one exact Issue-form body into an immutable request.

    :param body: The GitHub Issue body.
    :param issue_number: The trusted GitHub Issue number.
    :param requester: The trusted GitHub Issue author login.

    :returns: The parsed immutable job request.

    :raises IssueParseError: If trusted metadata or Issue-form input is malformed.
    """
    metadata_errors = []
    if type(issue_number) is not int or issue_number < 1:
        metadata_errors.append("issue number must be a positive integer")
    if type(requester) is not str or not requester.strip():
        metadata_errors.append("requester must not be empty")
    if metadata_errors:
        raise IssueParseError(tuple(metadata_errors))

    fields = fields_from_markdown(body)
    errors: list[str] = []
    arguments = _parse_arguments(fields["Arguments JSON"], errors)
    seed = _parse_integer(fields["Seed"], "Seed", errors)
    gpu_count = _parse_integer(fields["GPU count"], "GPU count", errors)
    max_runtime_minutes = _parse_integer(
        fields["Maximum runtime minutes"],
        "Maximum runtime minutes",
        errors,
    )
    if errors:
        raise IssueParseError(tuple(errors))

    return JobRequest(
        issue_number=issue_number,
        requester=requester.strip(),
        purpose=fields["Purpose"],
        study=fields["Study"],
        condition=fields["Condition"],
        comparison=fields["Comparison"],
        commit_sha=fields["Commit SHA"],
        entrypoint_profile=fields["Entrypoint profile"],
        script_path=fields["Script path"],
        launcher=fields["Launcher"].lower(),
        argv=arguments,
        data_manifest=fields["Data manifest"],
        data_manifest_sha256=fields["Data manifest SHA-256"],
        data_classification=fields["Data classification"],
        seed=seed,
        wandb_project=fields["W&B project"],
        success_signal=fields["Success signal"],
        success_metrics=tuple(
            metric.strip() for metric in fields["Success metrics"].split(",") if metric.strip()
        ),
        gpu_count=gpu_count,
        gpu_preference=fields["GPU preference"].lower(),
        max_runtime_minutes=max_runtime_minutes,
    )
