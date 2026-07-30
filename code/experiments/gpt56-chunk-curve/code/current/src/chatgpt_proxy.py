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
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
MAX_CONCURRENCY = max(1, int(os.environ.get("CHATGPT_PROXY_CONCURRENCY", "4")))
_UPSTREAM_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)
MAX_ATTEMPTS = max(1, int(os.environ.get("CHATGPT_PROXY_MAX_ATTEMPTS", "2")))
CONNECT_TIMEOUT_S = max(
    1.0, float(os.environ.get("CHATGPT_PROXY_CONNECT_TIMEOUT_S", "20"))
)
READ_TIMEOUT_S = max(
    1.0, float(os.environ.get("CHATGPT_PROXY_READ_TIMEOUT_S", "150"))
)
REQUEST_LOG = os.environ.get("CHATGPT_PROXY_LOG")
_LOG_LOCK = threading.Lock()
REQUESTED_REASONING_EFFORT = os.environ.get(
    "CHATGPT_REASONING_EFFORT", "none"
).strip().lower()
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


def _write_request_log(record):
    if not REQUEST_LOG:
        return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with _LOG_LOCK:
        Path(REQUEST_LOG).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(Path(REQUEST_LOG).expanduser(), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_responses_api(input_msgs, model="gpt-5.5", tools=None,
                       max_output_tokens=None):
    """Call the codex responses endpoint and parse the streaming response."""
    payload = {
        "model": model,
        "input": input_msgs,
        "reasoning": {"effort": REQUESTED_REASONING_EFFORT},
        "store": False,
        "stream": True,
    }

    if tools:
        # Convert from Chat Completions tool format to Responses API format
        resp_tools = []
        for t in tools:
            fn = t.get("function", {})
            resp_tools.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        payload["tools"] = resp_tools
    # The ChatGPT subscription endpoint rejects max_output_tokens.  The
    # compatible client still sends max_tokens, so record that it was ignored
    # without first issuing a guaranteed-to-fail upstream request.
    ignored_client_parameters = (
        ["max_output_tokens"] if max_output_tokens is not None else []
    )

    # Retry explicit HTTP statuses that state the request failed, plus a
    # prematurely closed chunked stream or failed TLS handshake whose output
    # has not been applied. Other transport failures retain the conservative
    # single-attempt policy.
    retryable = {429, 500, 502, 503, 504, 520}
    last_error = "unknown upstream failure"
    started = time.monotonic()
    unsupported_parameters = []
    attempts_made = 0
    last_actual_reasoning_effort = None
    last_reasoning_tokens = None
    last_reasoning_tokens_present = False
    with _UPSTREAM_SLOTS:
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
                        "Content-Type": "application/json",
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
                    result = parse_responses_stream(r)
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
                        })
                        return result
                    last_error = result["error"]
                    # An explicit provider-side server failure has no usable
                    # output and is safe to retry. Other stream failures may
                    # represent accepted work, so retain the fail-closed path.
                    if result.get("stream_error_code") != "server_error":
                        break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # These failures have no usable result and the caller has not
                # applied any output, so retrying cannot duplicate local state.
                if not isinstance(
                    exc,
                    (
                        requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.SSLError,
                    ),
                ):
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


def parse_responses_stream(r):
    """Parse one Codex Responses SSE stream and reject partial responses."""

    text_parts = []
    tool_calls = []
    function_calls = {}
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed = False
    stream_error = None
    stream_error_code = None
    response_id = None
    actual_model = None
    actual_reasoning_effort = None
    completed_output = []

    def call_key(chunk, item=None):
        item = item or {}
        key = (chunk.get("item_id") or item.get("id")
               or chunk.get("output_index"))
        if key is None and len(function_calls) == 1:
            return next(iter(function_calls))
        return str(key) if key is not None else None

    for line in r.iter_lines():
        if not line:
            continue
        decoded = line.decode()
        if not decoded.startswith("data: ") or decoded == "data: [DONE]":
            continue

        try:
            chunk = json.loads(decoded[6:])
        except json.JSONDecodeError:
            continue

        event_type = chunk.get("type", "")

        if event_type in ("error", "response.failed", "response.incomplete"):
            err = (chunk.get("error") or chunk.get("response", {}).get("error")
                   or chunk.get("response", {}).get("incomplete_details")
                   or chunk)
            if isinstance(err, dict):
                stream_error_code = err.get("code")
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
            completed_output = response.get("output") or []
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

    text = "".join(text_parts).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    if not text and not tool_calls:
        for item in completed_output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if (
                        isinstance(content, dict)
                        and content.get("type") == "output_text"
                    ):
                        text_parts.append(content.get("text", ""))
            elif item.get("type") == "output_text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "function_call":
                tool_calls.append(
                    {
                        "id": item.get(
                            "call_id", f"call_{uuid.uuid4().hex[:8]}"
                        ),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }
                )
        text = "".join(text_parts).strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    if stream_error:
        return {
            "error": f"upstream stream failed: {stream_error}",
            "stream_error_code": stream_error_code,
        }
    if not completed:
        return {"error": "upstream stream ended before response.completed"}
    if not text and not tool_calls:
        return {"error": "upstream completed with empty output"}

    return {"text": text or None, "tool_calls": tool_calls or None,
            "usage": usage_info, "response_id": response_id,
            "actual_model": actual_model,
            "actual_reasoning_effort": actual_reasoning_effort,
            "reasoning_tokens_present": reasoning_tokens_present}


class Handler(BaseHTTPRequestHandler):
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
            "requested_reasoning_effort": REQUESTED_REASONING_EFFORT,
            "code_sha256": code_hash,
            "request_log": REQUEST_LOG,
        }).encode()
        self.send_response(200 if auth_ok else 503)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            self._do_post()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"proxy exception: {e}"}).encode())
            except Exception:  # noqa: BLE001
                pass

    def _do_post(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        model = body.get("model", "gpt-5.5")
        tools = body.get("tools", [])

        # Convert Chat Completions messages to Responses API input
        input_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            # Normalize Chat Completions content-part lists (e.g. MIRIX sends
            # [{"type": "text", "text": ...}]) to a plain string; the Responses
            # API rejects part type "text" (wants input_text/output_text).
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text")
                )

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
        result = call_responses_api(input_msgs, model, tools or None,
                                    max_output_tokens=max_output_tokens)

        if "error" in result:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": result["error"]}).encode())
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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    port = int(os.environ.get("CHATGPT_PROXY_PORT", "8199"))
    print(f"ChatGPT proxy on http://localhost:{port} "
          f"(subscription-based, supports tools, max_upstream={MAX_CONCURRENCY})")
    ThreadedHTTPServer(("localhost", port), Handler).serve_forever()
