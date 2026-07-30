from pathlib import Path

import pytest

from edullm.request_parser import (
    IssueParseError,
    fields_from_markdown,
    issue_body_from_fields,
    parse_issue,
)

FIXTURE = Path("src/test/edullm/fixtures/valid_issue.md")


def _valid_body() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _replace_field(body: str, heading: str, value: str) -> str:
    prefix, rest = body.split(f"### {heading}\n", 1)
    _, separator, suffix = rest.partition("\n\n### ")
    if separator:
        return f"{prefix}### {heading}\n{value}\n\n### {suffix}"
    return f"{prefix}### {heading}\n{value}"


def test_issue_body_renderer_round_trips_the_authoritative_parser():
    original = FIXTURE.read_text(encoding="utf-8")
    fields = fields_from_markdown(original)
    rendered = issue_body_from_fields(fields)

    assert fields_from_markdown(rendered) == fields
    assert (
        parse_issue(rendered, issue_number=42, requester="student").canonical_json()
        == parse_issue(original, issue_number=42, requester="student").canonical_json()
    )


def test_issue_body_renderer_rejects_missing_extra_and_non_string_fields():
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(IssueParseError, match="missing heading: Purpose"):
        issue_body_from_fields({key: value for key, value in fields.items() if key != "Purpose"})
    with pytest.raises(IssueParseError, match="unexpected heading at index"):
        issue_body_from_fields({**fields, "Shell command": "sbatch anything"})
    with pytest.raises(IssueParseError, match="GPU count must be text"):
        issue_body_from_fields({**fields, "GPU count": 1})  # type: ignore[dict-item]


def test_parse_valid_issue():
    request = parse_issue(_valid_body(), issue_number=42, requester="student")

    assert request.issue_number == 42
    assert request.requester == "student"
    assert request.study == "skill-dag-v1"
    assert request.launcher == "python"
    assert request.argv == ("train_single", "skilldag-natural", "local", "--seed=0")
    assert request.success_metrics == ("train/loss",)
    assert request.gpu_count == 1
    assert len(request.digest) == 64


def test_parse_normalizes_dropdown_case_and_comma_separated_metrics():
    body = _replace_field(_valid_body(), "Launcher", "Python")
    body = _replace_field(body, "GPU preference", "L40S")
    body = _replace_field(body, "Success metrics", "train/loss, eval/perplexity")

    request = parse_issue(body, issue_number=42, requester="student")

    assert request.launcher == "python"
    assert request.gpu_preference == "l40s"
    assert request.success_metrics == ("train/loss", "eval/perplexity")


def test_parse_reports_missing_duplicate_and_unexpected_headings_deterministically():
    body = _valid_body().replace("### Purpose\nSkill-DAG smoke\n\n", "", 1)
    body += "\n\n### Study\nduplicate\n\n### Unexpected\nvalue\n"

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == (
        "missing heading: Purpose",
        "duplicate heading: Study",
        "unexpected heading at index 19",
    )


def test_parse_rejects_headings_outside_issue_form_order():
    body = _valid_body()
    purpose = "### Purpose\nSkill-DAG smoke\n\n"
    study = "### Study\nskill-dag-v1\n\n"
    body = body.replace(purpose + study, study + purpose, 1)

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("headings must appear in Issue-form order",)


def test_parse_rejects_unparsed_text_before_the_first_heading():
    body = "untracked preamble\n\n" + _valid_body()

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("unexpected content before first heading",)


def test_parse_rejects_malformed_heading_syntax_with_actionable_error():
    body = _valid_body().replace("### Purpose", "## Purpose", 1)

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("missing heading: Purpose",)


@pytest.mark.parametrize("body", [None, ["not", "markdown"]])
def test_parse_rejects_non_text_issue_bodies(body):
    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("Issue body must be text",)


def test_parse_rejects_oversized_issue_body_before_field_parsing():
    body = _valid_body() + (" " * 65_536)

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("Issue body exceeds 65536 characters",)


def test_parse_rejects_an_empty_required_field():
    body = _replace_field(_valid_body(), "Purpose", "   ")

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("Purpose must not be empty",)


def test_parse_rejects_malformed_arguments_json():
    body = _replace_field(_valid_body(), "Arguments JSON", '["train_single"')

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("Arguments JSON must be valid JSON",)


def test_parse_rejects_oversized_arguments_json_before_json_loading():
    body = _replace_field(_valid_body(), "Arguments JSON", f"[{'9' * 5000}]")

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == ("Arguments JSON exceeds 4096 characters",)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ('"train_single --seed=0"', "Arguments JSON must be an array of strings"),
        (
            '["train_single", 7]',
            "Arguments JSON[1] must be a string",
        ),
        (
            '{"command": "train_single"}',
            "Arguments JSON must be an array of strings",
        ),
    ],
)
def test_parse_requires_arguments_to_be_a_json_array_of_strings(arguments, message):
    body = _replace_field(_valid_body(), "Arguments JSON", arguments)

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == (message,)


@pytest.mark.parametrize(
    ("heading", "value", "message"),
    [
        ("Seed", "zero", "Seed must be an integer"),
        ("Seed", "1.5", "Seed must be an integer"),
        ("GPU count", "one", "GPU count must be an integer"),
        (
            "Maximum runtime minutes",
            "30 minutes",
            "Maximum runtime minutes must be an integer",
        ),
    ],
)
def test_parse_reports_typed_integer_errors(heading, value, message):
    body = _replace_field(_valid_body(), heading, value)

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == (message,)


@pytest.mark.parametrize("heading", ["Seed", "GPU count", "Maximum runtime minutes"])
def test_parse_rejects_oversized_integer_tokens_without_calling_int(heading):
    body = _replace_field(_valid_body(), heading, "9" * 5000)

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    assert raised.value.errors == (f"{heading} integer exceeds 10 characters",)


@pytest.mark.parametrize(
    ("issue_number", "requester", "message"),
    [
        (0, "student", "issue number must be a positive integer"),
        (True, "student", "issue number must be a positive integer"),
        (1.5, "student", "issue number must be a positive integer"),
        (42, "", "requester must not be empty"),
        (42, "  ", "requester must not be empty"),
        (42, None, "requester must not be empty"),
    ],
)
def test_parse_rejects_invalid_trusted_metadata(issue_number, requester, message):
    with pytest.raises(IssueParseError) as raised:
        parse_issue(_valid_body(), issue_number=issue_number, requester=requester)

    assert raised.value.errors == (message,)


def test_parse_error_order_is_stable_across_calls():
    body = _valid_body().replace("### Purpose\nSkill-DAG smoke\n\n", "", 1)
    body += "\n\n### Zeta\nvalue\n\n### Alpha\nvalue\n"

    observed = []
    for _ in range(3):
        with pytest.raises(IssueParseError) as raised:
            parse_issue(body, issue_number=42, requester="student")
        observed.append(raised.value.errors)

    assert (
        observed
        == [
            (
                "missing heading: Purpose",
                "unexpected heading at index 18",
                "unexpected heading at index 19",
            )
        ]
        * 3
    )


def test_parse_errors_never_echo_secret_bearing_unknown_heading():
    secret = "ghp_DO_NOT_ECHO_THIS_SECRET"
    body = _valid_body() + f"\n\n### {secret}\nvalue\n"

    with pytest.raises(IssueParseError) as raised:
        parse_issue(body, issue_number=42, requester="student")

    rendered = str(raised.value)
    assert raised.value.errors == ("unexpected heading at index 19",)
    assert secret not in rendered
