from __future__ import annotations

from collections.abc import Mapping

import pytest
import requests

from edullm.notifications import (
    SlackNotificationError,
    SlackNotifier,
    SlackValidationError,
)

WEBHOOK = "https://hooks.slack.com/services/T12345678/B12345678/AbCdEf0123456789"


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        text="ok",
        headers: Mapping[str, str] | None = None,
    ):
        self.status_code = status_code
        self.text = text
        self.headers = dict(headers or {})


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def test_slack_payload_keeps_untrusted_title_in_plain_text_and_escapes_fallback():
    session = FakeSession()
    notifier = SlackNotifier(WEBHOOK, session=session)
    title = "<@U99999999> & click <https://evil.test|here> " + "x" * 200

    notifier.assignment(
        issue=42,
        title=title,
        operator_slack_id="U12345678",
        kind="assignment",
    )

    url, payload, timeout = session.calls[0]
    assert url == WEBHOOK
    assert timeout == (5, 10)
    assert payload["blocks"][0]["text"] == {
        "type": "mrkdwn",
        "text": "eduLLM job #42 assigned to <@U12345678>",
    }
    assert payload["blocks"][1]["text"]["type"] == "plain_text"
    assert payload["blocks"][1]["text"]["text"] == title[:120]
    assert "<@U12345678>" not in payload["text"]
    assert "<@U99999999>" not in payload["text"]
    assert "&lt;@U99999999&gt;" in payload["text"]
    assert "&amp;" in payload["text"]
    assert len(payload["blocks"][1]["text"]["text"]) == 120


def test_reassignment_payload_labels_the_trusted_notification_kind():
    session = FakeSession()

    SlackNotifier(WEBHOOK, session=session).assignment(
        issue=42,
        title="Safe title",
        operator_slack_id="W12345678",
        kind="reassignment",
    )

    payload = session.calls[0][1]
    assert payload["blocks"][0]["text"]["text"] == ("eduLLM job #42 reassigned to <@W12345678>")


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled", "preempted"])
def test_terminal_payload_uses_only_trusted_bounded_fields(state):
    session = FakeSession()

    SlackNotifier(WEBHOOK, session=session).terminal(
        issue=42,
        operator_slack_id="U12345678",
        state=state,
    )

    payload = session.calls[0][1]
    assert payload == {
        "text": f"eduLLM job #42 {state}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"eduLLM job #42 {state}; operator <@U12345678>",
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "issue,user_id,state",
    [
        (0, "U12345678", "completed"),
        (True, "U12345678", "completed"),
        (42, "U123", "completed"),
        (42, "U12345678", "running"),
        (42, "U12345678", "COMPLETED"),
    ],
)
def test_terminal_notification_inputs_are_strict(issue, user_id, state):
    session = FakeSession()

    with pytest.raises(SlackValidationError):
        SlackNotifier(WEBHOOK, session=session).terminal(
            issue=issue,
            operator_slack_id=user_id,
            state=state,
        )

    assert session.calls == []


@pytest.mark.parametrize(
    "webhook",
    [
        "",
        "http://hooks.slack.com/services/T/B/C",
        "https://evil.test/services/T12345678/B12345678/secret",
        "https://hooks.slack.com.evil.test/services/T12345678/B12345678/secret",
        "https://user@hooks.slack.com/services/T12345678/B12345678/secret",
        "https://hooks.slack.com/services/T12345678/B12345678",
        "https://hooks.slack.com/services/T12345678/B12345678/secret?leak=yes",
        7,
    ],
)
def test_webhook_validation_is_strict_and_never_echoes_secret(webhook):
    with pytest.raises(SlackValidationError) as raised:
        SlackNotifier(webhook)

    if str(webhook):
        assert str(webhook) not in str(raised.value)


@pytest.mark.parametrize(
    "issue,user_id,kind",
    [
        (0, "U12345678", "assignment"),
        (True, "U12345678", "assignment"),
        (42, "U123", "assignment"),
        (42, "U1234567<@U99999999>", "assignment"),
        (42, "U12345678", "reminder"),
    ],
)
def test_notification_inputs_are_validated_before_http(issue, user_id, kind):
    session = FakeSession()
    notifier = SlackNotifier(WEBHOOK, session=session)

    with pytest.raises(SlackValidationError):
        notifier.assignment(
            issue=issue,
            title="title",
            operator_slack_id=user_id,
            kind=kind,
        )

    assert session.calls == []


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(status_code=400, text="invalid_payload secret body"),
        FakeResponse(status_code=500, text="server secret body"),
        FakeResponse(status_code=429, text="rate limited", headers={"Retry-After": "30"}),
    ],
)
def test_non_success_http_is_sanitized_retryable_and_does_not_echo_body(response):
    notifier = SlackNotifier(WEBHOOK, session=FakeSession(response=response))

    with pytest.raises(SlackNotificationError) as raised:
        notifier.assignment(
            issue=42,
            title="title",
            operator_slack_id="U12345678",
            kind="assignment",
        )

    assert raised.value.ambiguous is False
    assert "secret body" not in str(raised.value)
    assert WEBHOOK not in repr(raised.value)
    if response.status_code == 429:
        assert raised.value.retry_after_seconds == 30


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("timeout leaked secret"),
        requests.ConnectionError("connection leaked secret"),
    ],
)
def test_timeout_or_network_error_is_sanitized_and_ambiguous(error):
    notifier = SlackNotifier(WEBHOOK, session=FakeSession(error=error))

    with pytest.raises(SlackNotificationError) as raised:
        notifier.assignment(
            issue=42,
            title="title",
            operator_slack_id="U12345678",
            kind="assignment",
        )

    assert raised.value.ambiguous is True
    assert "leaked secret" not in str(raised.value)
    assert WEBHOOK not in repr(raised.value)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(status_code=True),
        FakeResponse(status_code=200, text="not-ok"),
        FakeResponse(status_code=200, text=None),
        FakeResponse(status_code=429, headers={"Retry-After": "secret"}),
    ],
)
def test_malformed_slack_response_fails_safely_without_claiming_delivery(response):
    notifier = SlackNotifier(WEBHOOK, session=FakeSession(response=response))

    with pytest.raises(SlackNotificationError) as raised:
        notifier.assignment(
            issue=42,
            title="title",
            operator_slack_id="U12345678",
            kind="assignment",
        )

    assert "secret" not in str(raised.value)


def test_notifier_repr_never_contains_webhook():
    notifier = SlackNotifier(WEBHOOK, session=FakeSession())

    assert WEBHOOK not in repr(notifier)
    assert "hooks.slack.com" not in repr(notifier)
