"""Provider options, retries, and structured JSON responses."""

import json
import re
import sys
import time
from typing import Any

from .config import MemoryConfig


def _chat_completion_with_retry(
    create: Any, *, retry_log: bool = False, **kwargs: Any
) -> Any:
    """Retry transient provider failures without discarding agent state."""
    attempt = 0
    while True:
        try:
            return create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            temporary_quota = status == 403 and (
                "insufficient_user_quota" in str(exc)
                or "用户额度不足" in str(exc)
            )
            transient = (
                isinstance(exc, (ConnectionError, TimeoutError))
                or type(exc).__name__ in {
                    "APIConnectionError",
                    "APITimeoutError",
                }
                or status == 429
                or temporary_quota
                or isinstance(status, int) and 500 <= status < 600
            )
            if not transient:
                raise
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(headers.get("retry-after", 0))
            except (TypeError, ValueError):
                delay = 0
            retry_delay = delay or (
                30 if temporary_quota else min(2 ** attempt, 120)
            )
            if retry_log:
                print(
                    f"V11 provider retry attempt={attempt + 1} "
                    f"status={status} delay={retry_delay} "
                    f"error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(retry_delay)
            attempt += 1


def _provider_options(config: MemoryConfig) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if config.reasoning_effort:
        options["reasoning_effort"] = config.reasoning_effort
    if config.thinking:
        options["extra_body"] = {"thinking": {"type": config.thinking}}
    return options


def _json_response(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    usage_logger: Any | None,
    config: MemoryConfig,
) -> dict[str, Any]:
    if not any("json" in message["content"].lower() for message in messages):
        messages = [
            {"role": "system", "content": "Return a valid JSON object."},
            *messages,
        ]
    response = _chat_completion_with_retry(
        client.chat.completions.create,
        retry_log=config.retry_log,
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0.0,
        response_format={"type": "json_object"},
        **_provider_options(config),
    )
    if usage_logger is not None:
        usage_logger(response)
    content = response.choices[0].message.content or ""
    content = re.sub(
        r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE
    )
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("verification model did not return JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("verification response must be a JSON object")
    return value
