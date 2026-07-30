from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn, cast

from edullm.policy import load_policy
from edullm.request_parser import IssueParseError, issue_body_from_fields, parse_issue
from edullm.validation import validate_request


class SubmissionInputError(ValueError):
    """A sanitized request-input error safe to show to the submitter."""


class _SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        raise SubmissionInputError("invalid command arguments")


def _read_document(input_path: Path) -> object:
    try:
        text = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise SubmissionInputError("request input could not be read") from None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise SubmissionInputError("request input is not valid JSON") from None


def validate_submission(
    input_path: Path,
    *,
    requester: str,
    policy_path: Path,
    entrypoints_path: Path,
) -> str:
    """
    Render and validate a request with the authoritative parser and policy.

    :param input_path: A private JSON file containing Issue-form fields.
    :param requester: The public GitHub login that will author the Issue.
    :param policy_path: The trusted queue policy path.
    :param entrypoints_path: The trusted entrypoint policy path.

    :returns: The exact validated Issue Markdown.
    """
    document = _read_document(input_path)
    if type(document) is not dict:
        raise ValueError("request JSON must be an object")
    fields = cast(dict[str, object], document)
    if any(type(key) is not str or type(value) is not str for key, value in fields.items()):
        raise ValueError("request JSON fields and values must be strings")

    body = issue_body_from_fields(cast(dict[str, str], fields))
    request = parse_issue(body, issue_number=1, requester=requester)
    try:
        policy = load_policy(policy_path, entrypoints_path)
    except (OSError, UnicodeError, ValueError):
        raise ValueError("trusted policy could not be loaded") from None
    errors = validate_request(request, policy)
    if errors:
        raise ValueError("\n".join(errors))
    return body


def main() -> int:
    """Validate private request JSON and print only canonical Issue Markdown."""
    parser = _SanitizedArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--requester", required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/edullm/policy.yaml"),
    )
    parser.add_argument(
        "--entrypoints",
        type=Path,
        default=Path("config/edullm/entrypoints.yaml"),
    )
    try:
        arguments = parser.parse_args()
        body = validate_submission(
            arguments.input_json,
            requester=arguments.requester,
            policy_path=arguments.policy,
            entrypoints_path=arguments.entrypoints,
        )
    except (IssueParseError, SubmissionInputError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        sys.stdout.write(body + "\n")
        sys.stdout.flush()
    except (OSError, UnicodeError):
        print("validated request could not be written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
