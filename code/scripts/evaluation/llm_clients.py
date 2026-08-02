"""LLM clients for the unified evaluation protocol.

- Answerer: deepseek-v4-flash via Aliyun (same model used for memory
  construction/retrieval across all systems), temperature=0.
- Judge: gpt-4o-mini via OpenRouter, temperature=0 — same judge model as the
  literature's LoCoMo numbers. Protocol decision 2026-07-03: gpt-5.5 as judge
  is systematically stricter on paraphrase answers (LightMem s0: 18/58 WRONG
  flipped to CORRECT under 4o-mini), which the full-context calibration cannot
  detect because full-context answers stay close to the source wording. Judge
  cost is negligible (~$0.1 for all 1540 LoCoMo questions).
  Fallback is NOT silent — judge failures raise after retries so runs never
  mix judge models.
"""

import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from urllib.parse import urlparse

import httpx as _httpx
from openai import OpenAI

ALIYUN_KEY = os.environ.get("ALIYUN_KEY", "not-set")
ALIYUN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

# Three configurable roles. Every stage (memory building, answering, judging)
# reads model + base_url + key from env vars, so switching e.g. the builder
# to gpt-4o-mini is just:
#   BUILDER_MODEL=gpt-4o-mini BUILDER_BASE=<openai-compatible-url> BUILDER_KEY=<key>
# Defaults: builder/answerer = deepseek-v4-flash via Aliyun; judge =
# gpt-4o-mini via OpenRouter (key from OPENROUTER_API_KEY or JUDGE_KEY).

BUILDER_MODEL = os.environ.get("BUILDER_MODEL", "deepseek-v4-flash")
BUILDER_BASE = os.environ.get("BUILDER_BASE", ALIYUN_BASE)
BUILDER_KEY = os.environ.get("BUILDER_KEY", ALIYUN_KEY)

ANSWERER_MODEL = os.environ.get("ANSWERER_MODEL", "deepseek-v4-flash")
ANSWERER_BASE = os.environ.get("ANSWERER_BASE", ALIYUN_BASE)
ANSWERER_KEY = os.environ.get("ANSWERER_KEY", ALIYUN_KEY)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")
JUDGE_BASE = os.environ.get("JUDGE_BASE", "https://openrouter.ai/api/v1")
JUDGE_KEY = os.environ.get("JUDGE_KEY",
                           os.environ.get("OPENROUTER_API_KEY", "not-set"))

_builder_client = None
_answerer_client = None
_judge_client = None
_attempt_observer: ContextVar = ContextVar("evaluation_attempt_observer", default=None)


class LLMCallError(RuntimeError):
    """A failed chat call with a complete record of its request attempts."""

    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage


def _should_trust_proxy(base_url):
    """Use the system proxy externally, never for localhost or Aliyun."""
    host = (urlparse(base_url).hostname or "").lower()
    return not (
        host in {"localhost", "127.0.0.1", "::1"}
        or host.endswith(".aliyuncs.com")
    )


def _http_client(base_url):
    # localhost proxying caused 502s; OpenRouter/OpenAI direct access is region
    # blocked on this machine and therefore must retain the system SOCKS proxy.
    return _httpx.Client(
        trust_env=_should_trust_proxy(base_url), timeout=180
    )


def _client(api_key, base_url):
    """Create a client whose SDK performs exactly one HTTP request per call.

    Retry accounting belongs to :func:`chat`; SDK-internal retries would make
    the recorded request-attempt count smaller than the physical HTTP count.
    """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=_http_client(base_url),
        max_retries=0,
    )


@contextmanager
def observe_attempts(callback):
    """Install a per-context durable-attempt observer for formal evaluators."""
    token = _attempt_observer.set(callback)
    try:
        yield
    finally:
        _attempt_observer.reset(token)


def emit_attempt_event(event):
    """Emit an immutable event after an HTTP attempt or judge parse decision."""
    callback = _attempt_observer.get()
    if callback is not None:
        callback(deepcopy(event))


def builder_client():
    global _builder_client
    if _builder_client is None:
        _builder_client = _client(BUILDER_KEY, BUILDER_BASE)
    return _builder_client


def answerer_client():
    global _answerer_client
    if _answerer_client is None:
        _answerer_client = _client(ANSWERER_KEY, ANSWERER_BASE)
    return _answerer_client


def judge_client():
    global _judge_client
    if _judge_client is None:
        _judge_client = _client(JUDGE_KEY, JUDGE_BASE)
    return _judge_client


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _token_count(usage, name):
    """Return a non-negative token count, or None when the provider omitted it."""
    value = getattr(usage, name, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _summarize_attempts(attempts, accepted=None):
    """Build usage without inventing IDs or token counts for failed requests."""
    accepted = accepted or {}
    return {
        "prompt_tokens": sum(
            item["prompt_tokens"] or 0 for item in attempts
        ),
        "completion_tokens": sum(
            item["completion_tokens"] or 0 for item in attempts
        ),
        "request_attempts": len(attempts),
        "physical_http_attempts": len(attempts),
        "logical_client_calls": 1,
        "failed_request_attempts": sum(
            item["status"] != "accepted" for item in attempts
        ),
        "unknown_token_attempts": sum(
            item["prompt_tokens"] is None
            or item["completion_tokens"] is None
            for item in attempts
        ),
        "request_attempt_details": attempts,
        "requested_model": accepted.get("requested_model"),
        "response_model": accepted.get("response_model"),
        "response_id": accepted.get("response_id"),
        "finish_reason": accepted.get("finish_reason"),
        "refusal": accepted.get("refusal"),
        "choice_count": accepted.get("choice_count"),
    }


def chat(client, model, messages, max_tokens=512, retries=3, retry_wait=3):
    last_err = None
    attempts = []
    for attempt in range(retries):
        resp = None
        attempt_record = {
            "request_attempt": attempt + 1,
            "status": "error",
            "failure_type": None,
            "error_type": None,
            "error_message": None,
            "requested_model": model,
            "response_model": None,
            "response_id": None,
            "finish_reason": None,
            "refusal": None,
            "choice_count": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0)
            usage = getattr(resp, "usage", None)
            attempt_record.update({
                "response_model": getattr(resp, "model", None),
                "response_id": getattr(resp, "id", None),
                "prompt_tokens": _token_count(usage, "prompt_tokens"),
                "completion_tokens": _token_count(usage, "completion_tokens"),
            })
            choices = getattr(resp, "choices", None)
            try:
                choice_count = len(choices)
            except TypeError as exc:
                raise ValueError("LLM response choices are missing") from exc
            attempt_record["choice_count"] = choice_count
            if choice_count != 1:
                raise ValueError(
                    f"LLM response has {choice_count} choices; expected exactly one"
                )
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            attempt_record["finish_reason"] = finish_reason
            if finish_reason != "stop":
                raise ValueError(
                    f"LLM response finish_reason is {finish_reason!r}; expected 'stop'"
                )
            message = getattr(choice, "message", None)
            if message is None:
                raise ValueError("LLM response choice has no message")
            refusal = getattr(message, "refusal", None)
            attempt_record["refusal"] = refusal or None
            if refusal not in (None, ""):
                raise ValueError("LLM response contains a refusal")
            content = getattr(message, "content", None) or ""
            attempt_record["status"] = "accepted"
            attempts.append(attempt_record)
            emit_attempt_event({
                "event": "physical_http_attempt",
                "attempt": attempt_record,
            })
            return _strip_think(content), _summarize_attempts(
                attempts, attempt_record
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            attempt_record.update({
                "status": "rejected" if resp is not None else "error",
                "failure_type": (
                    "invalid_response" if resp is not None else "transport_error"
                ),
                "error_type": type(e).__name__,
                "error_message": str(e)[:500],
            })
            attempts.append(attempt_record)
            emit_attempt_event({
                "event": "physical_http_attempt",
                "attempt": attempt_record,
            })
            if attempt + 1 < retries:
                time.sleep(retry_wait * (attempt + 1))
    raise LLMCallError(
        f"LLM call failed after {retries} retries: {last_err}",
        _summarize_attempts(attempts),
    )


def builder_chat(messages, max_tokens=4000):
    return chat(builder_client(), BUILDER_MODEL, messages, max_tokens)


def answerer_chat(messages, max_tokens=512):
    return chat(answerer_client(), ANSWERER_MODEL, messages, max_tokens)


def judge_chat(messages, max_tokens=300):
    retries = int(os.environ.get("JUDGE_HTTP_RETRIES", "3"))
    if retries < 1:
        raise ValueError("JUDGE_HTTP_RETRIES must be positive")
    return chat(
        judge_client(), JUDGE_MODEL, messages, max_tokens, retries=retries
    )
