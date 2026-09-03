"""OpenAI Chat Completions-compatible proxy routing to ChatGPT codex responses endpoint.

Translates between OpenAI Chat Completions API format and the Responses API format
used by chatgpt.com/backend-api/codex/responses.

Supports: chat completions with function calling / tool use.
Usage: python3 chatgpt_proxy.py  # starts on port 8199
"""
import json
import time
import uuid
import requests
import re
import os
import threading
import hashlib
import copy
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
MAX_CONCURRENCY = max(1, int(os.environ.get("CHATGPT_PROXY_CONCURRENCY", "4")))
_UPSTREAM_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT = {}
MAX_ATTEMPTS = max(1, int(os.environ.get("CHATGPT_PROXY_MAX_ATTEMPTS", "2")))
CONNECT_TIMEOUT_S = max(
    1.0, float(os.environ.get("CHATGPT_PROXY_CONNECT_TIMEOUT_S", "20"))
)
READ_TIMEOUT_S = max(
    1.0, float(os.environ.get("CHATGPT_PROXY_READ_TIMEOUT_S", "150"))
)
QUEUE_TIMEOUT_S = max(
    0.0, float(os.environ.get("CHATGPT_PROXY_QUEUE_TIMEOUT_S", "1"))
)
STREAM_DEADLINE_S = max(
    READ_TIMEOUT_S,
    float(os.environ.get("CHATGPT_PROXY_STREAM_DEADLINE_S", "840")),
)
OUTPUT_IDLE_TIMEOUT_S = max(
    1.0, float(os.environ.get("CHATGPT_PROXY_OUTPUT_IDLE_TIMEOUT_S", "300"))
)
REQUEST_LOG = os.environ.get("CHATGPT_PROXY_LOG")
PROTOCOL_LOG = os.environ.get("CHATGPT_PROXY_PROTOCOL_LOG")
_LOG_LOCK = threading.Lock()
REQUESTED_REASONING_EFFORT = os.environ.get(
    "CHATGPT_REASONING_EFFORT", "none"
).strip().lower()
FORCE_MODEL = os.environ.get("CHATGPT_PROXY_FORCE_MODEL", "").strip()
PORT = int(os.environ.get("CHATGPT_PROXY_PORT", "8199"))
if not 1 <= PORT <= 65535:
    raise ValueError("CHATGPT_PROXY_PORT must be between 1 and 65535")
if REQUESTED_REASONING_EFFORT not in {"none", "low", "medium", "high", "xhigh"}:
    raise ValueError(
        "CHATGPT_REASONING_EFFORT must be one of "
        "none, low, medium, high, or xhigh"
    )


def get_access_token():
    """Read the token fresh each call — codex CLI refreshes auth.json in the
    background, and a stale in-memory token causes upstream 401/502."""
    auth = json.load(open(Path.home() / ".codex" / "auth.json"))
    return auth["tokens"]["access_token"]


def get_account_id():
    """Read the ChatGPT account/workspace identifier without caching it."""
    auth = json.load(open(Path.home() / ".codex" / "auth.json"))
    account_id = (auth.get("tokens") or {}).get("account_id")
    if not account_id:
        raise RuntimeError("ChatGPT account_id is missing from Codex auth")
    return account_id


def _write_request_log(record):
    if not REQUEST_LOG:
        return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with _LOG_LOCK:
        Path(REQUEST_LOG).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(Path(REQUEST_LOG).expanduser(), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_protocol_log(record):
    if not PROTOCOL_LOG:
        return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with _LOG_LOCK:
        Path(PROTOCOL_LOG).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(Path(PROTOCOL_LOG).expanduser(), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def _upstream_slot():
    """Bound admission so abandoned SDK retries cannot form a hidden queue."""
    acquired = _UPSTREAM_SLOTS.acquire(timeout=QUEUE_TIMEOUT_S)
    try:
        yield acquired
    finally:
        if acquired:
            _UPSTREAM_SLOTS.release()


def _call_responses_api_upstream(input_msgs, model="gpt-5.5", tools=None,
                                 max_output_tokens=None, instructions=None,
                                 allow_empty_output=False):
    """Call the codex responses endpoint and parse the streaming response."""
    payload = {
        "model": model,
        "input": input_msgs,
        "reasoning": {"effort": REQUESTED_REASONING_EFFORT},
        "store": False,
        "stream": True,
    }
    if instructions:
        payload["instructions"] = instructions

    if tools:
        # Convert from Chat Completions tool format to Responses API format
        resp_tools = []
        for t in tools:
            # Claude Agent SDK sends Anthropic-style tool definitions to an
            # ANTHROPIC_BASE_URL, while direct OpenAI-compatible clients send
            # {type: function, function: {...}}.  Preserve both contracts.
            wrapped = t.get("function")
            fn = wrapped if isinstance(wrapped, dict) else t
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("tool definition has no non-empty name")
            resp_tools.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", fn.get("input_schema", {})),
            })
        payload["tools"] = resp_tools
    # The ChatGPT subscription endpoint rejects max_output_tokens.  The
    # compatible client still sends max_tokens, so record that it was ignored
    # without first issuing a guaranteed-to-fail upstream request.
    ignored_client_parameters = (
        ["max_output_tokens"] if max_output_tokens is not None else []
    )

    # Retry only explicit HTTP statuses that state the request failed.  A
    # transport timeout or incomplete stream has unknown completion state and
    # is not retried here.
    retryable = {429, 500, 502, 503, 504, 520}
    last_error = "unknown upstream failure"
    started = time.monotonic()
    unsupported_parameters = []
    attempts_made = 0
    last_actual_reasoning_effort = None
    last_reasoning_tokens = None
    last_reasoning_tokens_present = False
    with _upstream_slot() as acquired:
        if not acquired:
            last_error = (
                "proxy upstream capacity is busy; request was not queued "
                f"after {QUEUE_TIMEOUT_S:g}s"
            )
            _write_request_log({
                "status": "rejected_busy",
                "requested_model": model,
                "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
                "attempts": 0,
                "latency_s": round(time.monotonic() - started, 3),
                "error": last_error,
            })
            return {"error": last_error, "proxy_busy": True}
        for attempt in range(MAX_ATTEMPTS):
            attempts_made = attempt + 1
            r = None
            http_status = None
            request_id = None
            try:
                r = requests.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {get_access_token()}",
                        "ChatGPT-Account-Id": get_account_id(),
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    json=payload,
                    timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
                    stream=True,
                )
                http_status = r.status_code
                request_id = (r.headers.get("x-request-id")
                              or r.headers.get("request-id"))
                if r.status_code != 200:
                    last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                    if "usage_limit_reached" in r.text:
                        break
                    if r.status_code not in retryable:
                        break
                else:
                    result = parse_responses_stream(
                        r,
                        allow_empty_output=allow_empty_output,
                        deadline=started + STREAM_DEADLINE_S,
                    )
                    if "error" not in result:
                        reasoning_tokens = (
                            (result.get("usage") or {})
                            .get("completion_tokens_details", {})
                            .get("reasoning_tokens", 0)
                            or 0
                        )
                        reasoning_tokens_present = result.get(
                            "reasoning_tokens_present", False
                        )
                        actual_reasoning_effort = result.get(
                            "actual_reasoning_effort"
                        )
                        last_actual_reasoning_effort = actual_reasoning_effort
                        last_reasoning_tokens = reasoning_tokens
                        last_reasoning_tokens_present = reasoning_tokens_present
                        if (
                            REQUESTED_REASONING_EFFORT == "none"
                            and actual_reasoning_effort not in (None, "none")
                        ):
                            last_error = (
                                "upstream violated reasoning.effort=none: "
                                "reported actual effort "
                                f"{actual_reasoning_effort!r}"
                            )
                            break
                        if (
                            REQUESTED_REASONING_EFFORT == "none"
                            and actual_reasoning_effort is None
                            and not reasoning_tokens_present
                        ):
                            last_error = (
                                "upstream returned no evidence that "
                                "reasoning.effort=none was honored"
                            )
                            break
                        if (
                            REQUESTED_REASONING_EFFORT == "none"
                            and reasoning_tokens != 0
                        ):
                            last_error = (
                                "upstream violated reasoning.effort=none: "
                                f"reported {reasoning_tokens} reasoning tokens"
                            )
                            break
                        result["attempts"] = attempt + 1
                        result["http_request_id"] = request_id
                        result["unsupported_parameters"] = unsupported_parameters
                        result["ignored_client_parameters"] = (
                            ignored_client_parameters
                        )
                        result["requested_reasoning_effort"] = (
                            REQUESTED_REASONING_EFFORT
                        )
                        _write_request_log({
                            "status": "success",
                            "requested_model": model,
                            "actual_model": result.get("actual_model"),
                            "requested_reasoning_effort": (
                                REQUESTED_REASONING_EFFORT
                            ),
                            "actual_reasoning_effort": actual_reasoning_effort,
                            "response_id": result.get("response_id"),
                            "http_request_id": request_id,
                            "attempts": attempt + 1,
                            "unsupported_parameters": unsupported_parameters,
                            "ignored_client_parameters": (
                                ignored_client_parameters
                            ),
                            "latency_s": round(time.monotonic() - started, 3),
                            "usage": result.get("usage"),
                            "empty_output": bool(result.get("empty_output")),
                            "empty_output_accepted": bool(
                                result.get("empty_output")
                                and allow_empty_output
                            ),
                        })
                        return result
                    last_error = result["error"]
                    # A stream may have been accepted and computed upstream.
                    # Retrying an incomplete stream can duplicate that work.
                    break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # Transport failures have unknown upstream completion state.
                # Leave recovery to the per-question checkpoint instead of
                # submitting the same operation again here.
                break
            finally:
                if r is not None:
                    r.close()
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))

    _write_request_log({
        "status": "error", "requested_model": model,
        "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
        "attempts": attempts_made,
        "unsupported_parameters": unsupported_parameters,
        "ignored_client_parameters": ignored_client_parameters,
        "latency_s": round(time.monotonic() - started, 3),
        "error": last_error, "http_status": http_status,
        "http_request_id": request_id,
        "actual_reasoning_effort": last_actual_reasoning_effort,
        "reasoning_tokens": last_reasoning_tokens,
        "reasoning_tokens_present": last_reasoning_tokens_present,
    })
    return {"error": f"upstream failed after retries: {last_error}"}


def call_responses_api(input_msgs, model="gpt-5.5", tools=None,
                       max_output_tokens=None, instructions=None,
                       allow_empty_output=False):
    """Coalesce identical concurrent SDK retries onto one upstream request."""
    fingerprint_payload = {
        "input": input_msgs,
        "model": model,
        "tools": tools,
        "max_output_tokens": max_output_tokens,
        "instructions": instructions,
        "allow_empty_output": allow_empty_output,
        "reasoning_effort": REQUESTED_REASONING_EFFORT,
    }
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    started = time.monotonic()
    with _INFLIGHT_LOCK:
        entry = _INFLIGHT.get(fingerprint)
        follower = entry is not None
        if entry is None:
            entry = {"event": threading.Event(), "result": None}
            _INFLIGHT[fingerprint] = entry

    if follower:
        completed = entry["event"].wait(STREAM_DEADLINE_S + 5)
        if not completed:
            error = "timed out waiting for identical in-flight request"
            _write_request_log({
                "status": "coalesced_error",
                "requested_model": model,
                "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
                "attempts": 0,
                "latency_s": round(time.monotonic() - started, 3),
                "request_fingerprint": fingerprint,
                "error": error,
            })
            return {"error": error}
        result = copy.deepcopy(entry["result"])
        _write_request_log({
            "status": (
                "coalesced_error" if "error" in result else "coalesced_success"
            ),
            "requested_model": model,
            "actual_model": result.get("actual_model"),
            "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
            "actual_reasoning_effort": result.get("actual_reasoning_effort"),
            "response_id": result.get("response_id"),
            "attempts": 0,
            "latency_s": round(time.monotonic() - started, 3),
            "request_fingerprint": fingerprint,
            "coalesced": True,
            "error": result.get("error"),
        })
        return result

    try:
        result = _call_responses_api_upstream(
            input_msgs,
            model,
            tools,
            max_output_tokens=max_output_tokens,
            instructions=instructions,
            allow_empty_output=allow_empty_output,
        )
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"proxy leader exception: {type(exc).__name__}: {exc}"}
    finally:
        with _INFLIGHT_LOCK:
            entry["result"] = copy.deepcopy(result)
            entry["event"].set()
            _INFLIGHT.pop(fingerprint, None)
    return result


def parse_responses_stream(r, *, allow_empty_output=False, deadline=None):
    """Parse one Codex Responses SSE stream and reject partial responses."""

    text_parts = []
    tool_calls = []
    function_calls = {}
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed = False
    stream_error = None
    response_id = None
    actual_model = None
    actual_reasoning_effort = None
    event_type_counts = {}
    terminal_event = None
    deadline_expired = threading.Event()
    deadline_timer = None
    output_idle_expired = threading.Event()
    output_idle_timer = None

    def abort_at_deadline():
        deadline_expired.set()
        # ``requests`` read timeouts are idle-byte timeouts. A peer that sends
        # heartbeat bytes without completing an SSE line can therefore keep
        # ``iter_lines`` blocked forever. Shut down the underlying socket from
        # an independent watchdog so the wall-clock deadline does not depend
        # on the iterator yielding control.
        try:
            sock = r.raw._fp.fp.raw._sock
            sock.shutdown(socket.SHUT_RDWR)
        except (AttributeError, OSError):
            pass

    def abort_on_output_idle():
        output_idle_expired.set()
        try:
            sock = r.raw._fp.fp.raw._sock
            sock.shutdown(socket.SHUT_RDWR)
        except (AttributeError, OSError):
            pass
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass

    def arm_output_idle_watchdog():
        nonlocal output_idle_timer
        if output_idle_timer is not None:
            output_idle_timer.cancel()
        output_idle_timer = threading.Timer(
            OUTPUT_IDLE_TIMEOUT_S, abort_on_output_idle
        )
        output_idle_timer.daemon = True
        output_idle_timer.start()
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass

    if deadline is not None:
        remaining = max(0.0, deadline - time.monotonic())
        deadline_timer = threading.Timer(remaining, abort_at_deadline)
        deadline_timer.daemon = True
        deadline_timer.start()
    arm_output_idle_watchdog()

    def call_key(chunk, item=None):
        item = item or {}
        key = (chunk.get("item_id") or item.get("id")
               or chunk.get("output_index"))
        if key is None and len(function_calls) == 1:
            return next(iter(function_calls))
        return str(key) if key is not None else None

    try:
        for line in r.iter_lines():
            if deadline_expired.is_set() or (
                deadline is not None and time.monotonic() >= deadline
            ):
                raise requests.ReadTimeout(
                    "upstream stream exceeded absolute deadline of "
                    f"{STREAM_DEADLINE_S:g}s"
                )
            if not line:
                continue
            decoded = line.decode()
            if decoded == "data: [DONE]":
                break
            if not decoded.startswith("data: "):
                continue

            try:
                chunk = json.loads(decoded[6:])
            except json.JSONDecodeError:
                continue

            event_type = chunk.get("type", "")
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            terminal_event = {
                "type": event_type,
                "response_status": (chunk.get("response") or {}).get("status"),
                "error_type": (chunk.get("error") or {}).get("type")
                if isinstance(chunk.get("error"), dict) else None,
                "error_code": (chunk.get("error") or {}).get("code")
                if isinstance(chunk.get("error"), dict) else None,
            }

            if event_type in {
                "response.output_text.delta",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.completed",
            }:
                arm_output_idle_watchdog()

            if event_type in ("error", "response.failed", "response.incomplete"):
                err = (chunk.get("error") or chunk.get("response", {}).get("error")
                       or chunk.get("response", {}).get("incomplete_details")
                       or chunk)
                stream_error = json.dumps(err, ensure_ascii=False)[:500]
                continue

        # Real token usage arrives on the completed event
            if event_type == "response.completed":
                completed = True
                response = chunk.get("response") or {}
                response_id = response.get("id")
                actual_model = response.get("model")
                actual_reasoning_effort = (
                    (response.get("reasoning") or {}).get("effort")
                )
                u = response.get("usage") or {}
                output_token_details = u.get("output_tokens_details") or {}
                usage_info = {
                "prompt_tokens": u.get("input_tokens", 0) or 0,
                "completion_tokens": u.get("output_tokens", 0) or 0,
                "total_tokens": u.get("total_tokens", 0) or 0,
                "prompt_tokens_details": {
                    "cached_tokens": (u.get("input_tokens_details") or {}).get(
                        "cached_tokens", 0) or 0,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": output_token_details.get(
                        "reasoning_tokens", 0
                    ) or 0,
                },
            }
                reasoning_tokens_present = "reasoning_tokens" in output_token_details
                break

        # Text output
            if event_type == "response.output_text.delta":
                text_parts.append(chunk.get("delta", ""))

        # Function call started
            elif event_type == "response.output_item.added":
                item = chunk.get("item", {})
                if item.get("type") == "function_call":
                    key = call_key(chunk, item) or f"anonymous-{len(function_calls)}"
                    function_calls[key] = {
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": item.get("name", ""),
                    "arguments": "",
                    }

        # Function call arguments streaming
            elif event_type == "response.function_call_arguments.delta":
                key = call_key(chunk)
                if key in function_calls:
                    function_calls[key]["arguments"] += chunk.get("delta", "")

        # Function call complete
            elif event_type == "response.function_call_arguments.done":
                key = call_key(chunk)
                current_fc = function_calls.pop(key, None)
                if current_fc:
                    current_fc["arguments"] = chunk.get("arguments", current_fc["arguments"])
                    tool_calls.append({
                    "id": current_fc["id"],
                    "type": "function",
                    "function": {
                        "name": current_fc["name"],
                        "arguments": current_fc["arguments"],
                    }
                    })
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        if output_idle_timer is not None:
            output_idle_timer.cancel()

    if deadline_expired.is_set():
        raise requests.ReadTimeout(
            "upstream stream exceeded absolute deadline of "
            f"{STREAM_DEADLINE_S:g}s"
        )
    if output_idle_expired.is_set():
        raise requests.ReadTimeout(
            "upstream stream produced no content or tool-call progress for "
            f"{OUTPUT_IDLE_TIMEOUT_S:g}s"
        )

    text = "".join(text_parts).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    _write_protocol_log({
        "kind": "stream_summary",
        "completed": completed,
        "event_type_counts": event_type_counts,
        "terminal_event": terminal_event,
        "stream_error_present": bool(stream_error),
        "text_present": bool(text),
        "tool_call_count": len(tool_calls),
    })

    if stream_error:
        return {
            "error": f"upstream stream failed: {stream_error}; "
            f"events={event_type_counts}"
        }
    if not completed:
        return {
            "error": "upstream stream ended before response.completed; "
            f"events={event_type_counts}; terminal={terminal_event}"
        }
    empty_output = not text and not tool_calls
    if empty_output and not allow_empty_output:
        return {"error": "upstream completed with empty output"}

    return {"text": text or None, "tool_calls": tool_calls or None,
            "usage": usage_info, "response_id": response_id,
            "actual_model": actual_model,
            "actual_reasoning_effort": actual_reasoning_effort,
            "reasoning_tokens_present": reasoning_tokens_present,
            "upstream_event_type_counts": event_type_counts,
            "empty_output": empty_output}


def _has_anthropic_tool_result(messages):
    """Return whether an Anthropic message history contains tool side effects.

    An empty terminal turn is meaningful only after the model has requested a
    tool and the client has returned its result.  Keeping this check at the
    protocol boundary means a genuinely empty first response still fails
    closed for both Anthropic and Chat Completions callers.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "tool_result"
            for part in content
        ):
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    def _send_payload(self, status, payload, content_type="application/json"):
        """Write a response without turning a departed client into a traceback."""
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
            return False

    def do_GET(self):
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return
        try:
            get_access_token()
            auth_ok = True
        except Exception:  # noqa: BLE001
            auth_ok = False
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        body = json.dumps({
            "status": "ok" if auth_ok else "degraded",
            "auth_readable": auth_ok,
            "max_concurrency": MAX_CONCURRENCY,
            "max_attempts": MAX_ATTEMPTS,
            "connect_timeout_s": CONNECT_TIMEOUT_S,
            "read_timeout_s": READ_TIMEOUT_S,
            "queue_timeout_s": QUEUE_TIMEOUT_S,
            "stream_deadline_s": STREAM_DEADLINE_S,
            "output_idle_timeout_s": OUTPUT_IDLE_TIMEOUT_S,
            "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
            "code_sha256": code_hash,
            "request_log": REQUEST_LOG,
            "port": PORT,
        }).encode()
        self._send_payload(200 if auth_ok else 503, body)

    def do_POST(self):
        try:
            self._do_post()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            try:
                self._send_payload(
                    500,
                    json.dumps({"error": f"proxy exception: {e}"}).encode(),
                )
            except Exception:  # noqa: BLE001
                pass

    def _do_post(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        model = FORCE_MODEL or body.get("model", "gpt-5.5")
        tools = body.get("tools", [])
        request_path = self.path.split("?", 1)[0].rstrip("/")
        anthropic_messages = request_path.endswith("/messages")

        # Anthropic Messages carries the system prompt in a top-level field,
        # not in ``messages``.  Preserve it as Responses API instructions;
        # dropping this field silently removes the memory format contract.
        instructions = body.get("system")
        if isinstance(instructions, list):
            instructions = "\n".join(
                str(part.get("text", ""))
                for part in instructions
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif instructions is not None:
            instructions = str(instructions)

        # Convert Chat Completions messages to Responses API input
        input_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            # Normalize Chat Completions content-part lists (e.g. MIRIX sends
            # [{"type": "text", "text": ...}]) to a plain string; the Responses
            # API rejects part type "text" (wants input_text/output_text).
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    kind = part.get("type")
                    if kind in ("text", "input_text", "output_text"):
                        text_parts.append(part.get("text", ""))
                    elif kind == "tool_use":
                        input_msgs.append({
                            "type": "function_call",
                            "call_id": part.get("id", ""),
                            "name": part.get("name", ""),
                            "arguments": json.dumps(
                                part.get("input", {}), ensure_ascii=False
                            ),
                        })
                    elif kind == "tool_result":
                        output = part.get("content", "")
                        if isinstance(output, list):
                            output = "\n".join(
                                str(item.get("text", ""))
                                for item in output if isinstance(item, dict)
                            )
                        input_msgs.append({
                            "type": "function_call_output",
                            "call_id": part.get("tool_use_id", ""),
                            "output": str(output),
                        })
                content = "\n".join(text_parts)

            # Handle tool results
            if role == "tool":
                input_msgs.append({
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id", ""),
                    "output": content,
                })
                continue

            # Handle assistant messages with tool calls
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    input_msgs.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })
                continue

            if content:
                input_msgs.append({"role": role, "content": content})

        max_output_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        result = call_responses_api(
            input_msgs,
            model,
            tools or None,
            max_output_tokens=max_output_tokens,
            instructions=instructions if anthropic_messages else None,
            allow_empty_output=(
                anthropic_messages and _has_anthropic_tool_result(messages)
            ),
        )

        if "error" in result:
            status = 503 if result.get("proxy_busy") else 500
            self._send_payload(
                status, json.dumps({"error": result["error"]}).encode()
            )
            return

        if anthropic_messages:
            usage = result.get("usage") or {}
            prompt_details = usage.get("prompt_tokens_details") or {}
            response_id = result.get("response_id") or f"msg_{uuid.uuid4().hex[:16]}"
            events = []
            emitted_event_types = []

            def emit(event, data):
                emitted_event_types.append(event)
                events.append(
                    f"event: {event}\ndata: "
                    + json.dumps(data, ensure_ascii=False)
                    + "\n\n"
                )

            emit("message_start", {
                "type": "message_start",
                "message": {
                    "id": response_id, "type": "message", "role": "assistant",
                    "content": [], "model": result.get("actual_model") or model,
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {
                        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": int(
                            prompt_details.get("cached_tokens", 0) or 0
                        ),
                    },
                },
            })
            block_index = 0
            if result.get("text"):
                emit("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                emit("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "text_delta", "text": result["text"]},
                })
                emit("content_block_stop", {
                    "type": "content_block_stop", "index": block_index,
                })
                block_index += 1
            elif result.get("empty_output"):
                # Claude Code may finish a tool-driven task without a prose
                # summary.  Preserve that valid Anthropic end_turn shape as an
                # explicit empty text block instead of turning it into a 500.
                emit("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                emit("content_block_stop", {
                    "type": "content_block_stop", "index": block_index,
                })
                block_index += 1
            for call in result.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments", "{}")
                emit("content_block_start", {
                    "type": "content_block_start", "index": block_index,
                    "content_block": {
                        "type": "tool_use", "id": call.get("id", ""),
                        "name": function.get("name", ""), "input": {},
                    },
                })
                emit("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments},
                })
                emit("content_block_stop", {
                    "type": "content_block_stop", "index": block_index,
                })
                block_index += 1
            stop_reason = "tool_use" if result.get("tool_calls") else "end_turn"
            emit("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0)
                },
            })
            emit("message_stop", {"type": "message_stop"})
            payload = "".join(events).encode()
            sent = self._send_payload(200, payload, "text/event-stream")
            _write_protocol_log({
                "response_id": response_id,
                "sent": sent,
                "upstream_event_type_counts": result.get(
                    "upstream_event_type_counts", {}
                ),
                "anthropic_event_types": emitted_event_types,
                "stop_reason": stop_reason,
                "tool_call_count": len(result.get("tool_calls") or []),
                "empty_output": bool(result.get("empty_output")),
            })
            return

        response = {
            "id": result.get("response_id") or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.get("actual_model") or model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                    "tool_calls": result["tool_calls"],
                },
                "finish_reason": "tool_calls" if result["tool_calls"] else "stop",
            }],
            "usage": result.get("usage") or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "proxy_meta": {
                "attempts": result.get("attempts"),
                "http_request_id": result.get("http_request_id"),
                "unsupported_parameters": result.get("unsupported_parameters", []),
                "ignored_client_parameters": result.get(
                    "ignored_client_parameters", []
                ),
                "requested_reasoning_effort": result.get(
                    "requested_reasoning_effort"
                ),
                "actual_reasoning_effort": result.get(
                    "actual_reasoning_effort"
                ),
            },
        }

        self._send_payload(200, json.dumps(response).encode())

    def log_message(self, format, *args):
        pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    print(f"ChatGPT proxy on http://localhost:{PORT} "
          f"(subscription-based, supports tools, max_upstream={MAX_CONCURRENCY})")
    ThreadedHTTPServer(("localhost", PORT), Handler).serve_forever()
