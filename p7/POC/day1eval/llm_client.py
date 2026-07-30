"""Minimal OpenAI-compatible client for the TrueFoundry PromptLens gateway.

Used by the LLM-as-a-judge scorer. Handles the two gateway quirks called out in
the usage guide:
  - OpenAI-family models want ``max_completion_tokens`` (not ``max_tokens``).
  - 429 / 5xx responses must be retried (exponential backoff + Retry-After).

The API key is read from the environment (default var ``PROMPTLENS_API_KEY``);
it is never logged or persisted.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import requests

DEFAULT_GATEWAY_URL = "https://tfy.promptlens.trilogy.com/v1/chat/completions"


class LLMClientError(RuntimeError):
    """Raised for non-retryable failures or after retries are exhausted."""


def _is_openai_model(model: str) -> bool:
    # e.g. "openai-group/gpt-5.6-sol" -> True ; "claude-group/..." -> False
    return model.lower().startswith("openai")


def _sleep_backoff(attempt: int, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            time.sleep(min(float(retry_after), 60.0))
            return
        except ValueError:
            pass
    time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    *,
    api_key: str | None = None,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    api_key_env: str = "PROMPTLENS_API_KEY",
    max_tokens: int = 4000,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    max_retries: int = 5,
    timeout: int = 300,
) -> str:
    """Send a chat-completions request and return the assistant message text.

    Retries on connection errors, 429, and 5xx. Raises ``LLMClientError`` on
    auth/4xx errors or once retries are exhausted.
    """
    api_key = api_key or os.environ.get(api_key_env)
    if not api_key:
        raise LLMClientError(
            f"{api_key_env} not set. `export {api_key_env}=...` or add it to .env."
        )

    payload: dict[str, Any] = {"model": model, "messages": messages}
    # Token-cap parameter name differs between OpenAI and other model families.
    if _is_openai_model(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(gateway_url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:  # connection/timeout — retry
            last_err = exc
            _sleep_backoff(attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = LLMClientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            _sleep_backoff(attempt, resp.headers.get("Retry-After"))
            continue

        if resp.status_code >= 400:  # non-retryable client error (e.g. 401/400)
            raise LLMClientError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
            return data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            # Malformed body — retry a couple of times, then give up.
            last_err = LLMClientError(f"Bad response body: {resp.text[:300]}")
            _sleep_backoff(attempt)
            continue

    raise LLMClientError(f"Failed after {max_retries} retries: {last_err}")
