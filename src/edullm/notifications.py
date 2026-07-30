"""
Sanitized Slack incoming-webhook notifications for eduLLM assignment.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

import requests

_DEFAULT_TIMEOUT = (5, 10)
_SLACK_ID = re.compile(r"[UW][A-Z0-9]{8,20}\Z")
_WEBHOOK_PATH = re.compile(r"/services/[A-Za-z0-9]+/[A-Za-z0-9]+/[A-Za-z0-9]+\Z")
_MAX_TITLE_CHARS = 120
_KINDS = frozenset({"assignment", "reassignment"})
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "preempted"})


class SlackValidationError(ValueError):
    """Raised when a Slack notification input is malformed."""


class SlackNotificationError(RuntimeError):
    """A sanitized Slack delivery failure with ambiguity metadata."""

    def __init__(
        self,
        message: str,
        *,
        ambiguous: bool,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous
        self.retry_after_seconds = retry_after_seconds


class SlackNotifier:
    """Send bounded assignment messages through one protected incoming webhook."""

    def __init__(
        self,
        webhook: str,
        *,
        session: Any | None = None,
        timeout: tuple[float, float] = _DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize a notifier without retaining a printable webhook.

        :param webhook: A protected Slack incoming-webhook URL.
        :param session: An injectable requests-compatible HTTP boundary.
        :param timeout: Finite connect and read timeouts.

        :raises SlackValidationError: If the URL or timeout is malformed.
        """
        self._webhook = _validate_webhook(webhook)
        self._session = requests.Session() if session is None else session
        self._timeout = _validate_timeout(timeout)

    def __repr__(self) -> str:
        """Return a representation that cannot expose the webhook."""
        return f"{type(self).__name__}(webhook=<redacted>)"

    def assignment(
        self,
        *,
        issue: int,
        title: str,
        operator_slack_id: str,
        kind: str,
    ) -> None:
        """
        Send one assignment or reassignment message.

        The operator mention is validated and kept in a trusted mrkdwn block.
        The Issue title is bounded in a plain-text block and HTML-escaped in the
        notification fallback.

        :param issue: The positive GitHub Issue number.
        :param title: The current untrusted Issue title.
        :param operator_slack_id: The protected Slack user ID.
        :param kind: ``assignment`` or ``reassignment``.

        :raises SlackValidationError: If an input is malformed.
        :raises SlackNotificationError: If delivery fails or is ambiguous.
        """
        if type(issue) is not int or issue <= 0:
            raise SlackValidationError("Slack Issue number must be a positive integer")
        if type(title) is not str or any(
            ord(character) < 32 and character not in {"\t"} or ord(character) == 127
            for character in title
        ):
            raise SlackValidationError("Slack Issue title is invalid")
        if type(operator_slack_id) is not str or _SLACK_ID.fullmatch(operator_slack_id) is None:
            raise SlackValidationError("Slack operator user ID is invalid")
        if type(kind) is not str or kind not in _KINDS:
            raise SlackValidationError("Slack notification kind is invalid")

        verb = "assigned" if kind == "assignment" else "reassigned"
        safe_title = title[:_MAX_TITLE_CHARS]
        trusted = f"eduLLM job #{issue} {verb} to <@{operator_slack_id}>"
        payload = {
            "text": (f"eduLLM job #{issue} {kind}: " f"{html.escape(safe_title, quote=False)}"),
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": trusted},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": safe_title,
                        "emoji": False,
                    },
                },
            ],
        }
        self._send(payload)

    def terminal(
        self,
        *,
        issue: int,
        operator_slack_id: str,
        state: str,
    ) -> None:
        """Send one bounded terminal event using only protected and canonical fields."""
        if type(issue) is not int or issue <= 0:
            raise SlackValidationError("Slack Issue number must be a positive integer")
        if type(operator_slack_id) is not str or _SLACK_ID.fullmatch(operator_slack_id) is None:
            raise SlackValidationError("Slack operator user ID is invalid")
        if type(state) is not str or state not in _TERMINAL_STATES:
            raise SlackValidationError("Slack terminal state is invalid")
        payload = {
            "text": f"eduLLM job #{issue} {state}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"eduLLM job #{issue} {state}; operator <@{operator_slack_id}>",
                    },
                }
            ],
        }
        self._send(payload)

    def _send(self, payload: Mapping[str, object]) -> None:
        """Deliver one already-validated payload with sanitized outcome semantics."""
        try:
            response = self._session.post(
                self._webhook,
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException:
            raise SlackNotificationError(
                "Slack notification outcome is unknown",
                ambiguous=True,
            ) from None

        status = getattr(response, "status_code", None)
        if type(status) is not int:
            raise SlackNotificationError(
                "Slack notification response was malformed",
                ambiguous=True,
            )
        if status == 429:
            raise SlackNotificationError(
                "Slack notification was rate limited",
                ambiguous=False,
                retry_after_seconds=_retry_after(response),
            )
        if not 200 <= status < 300:
            raise SlackNotificationError(
                "Slack notification failed",
                ambiguous=False,
            )
        text = getattr(response, "text", None)
        if type(text) is not str or text != "ok":
            raise SlackNotificationError(
                "Slack notification outcome is unknown",
                ambiguous=True,
            )


def _validate_webhook(value: object) -> str:
    if type(value) is not str:
        raise SlackValidationError("Slack webhook is invalid")
    parsed = urlsplit(cast(str, value))
    if (
        parsed.scheme != "https"
        or parsed.hostname != "hooks.slack.com"
        or parsed.netloc != "hooks.slack.com"
        or parsed.query
        or parsed.fragment
        or _WEBHOOK_PATH.fullmatch(parsed.path) is None
    ):
        raise SlackValidationError("Slack webhook is invalid")
    return cast(str, value)


def _validate_timeout(value: object) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise SlackValidationError("Slack timeout is invalid")
    timeout = cast(tuple[object, object], value)
    if any(
        type(item) not in {int, float}
        or not math.isfinite(cast(float, item))
        or cast(float, item) <= 0
        for item in timeout
    ):
        raise SlackValidationError("Slack timeout is invalid")
    return cast(tuple[float, float], timeout)


def _retry_after(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if not hasattr(headers, "get"):
        return None
    value = cast(Any, headers).get("Retry-After")
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        return None
    parsed = int(value)
    return parsed if 0 < parsed <= 86_400 else None
