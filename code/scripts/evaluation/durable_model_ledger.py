"""Durable, request-linked accounting for experimental model calls.

This module does not create a model client.  It wraps one existing
``chat.completions.create`` method, writes the exact model-visible request before
the call, and persists the response and proxy linkage before returning it to
the caller.  A process failure between those points leaves an orphan start that
resume code must reject instead of silently repeating.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


LEDGER_SCHEMA = "durable-model-ledger/v1"
REQUEST_SCHEMA = "durable-model-request/v1"
RESPONSE_SCHEMA = "durable-model-response/v1"
ZERO_HASH = "0" * 64


class DurableLedgerError(RuntimeError):
    """Raised when durable accounting cannot be proven."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> str:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


def _chain_hash(record_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(record_without_hash).encode("utf-8"))


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise DurableLedgerError(f"ledger is not a regular file: {path}")
    if path.stat(follow_symlinks=False).st_nlink != 1:
        raise DurableLedgerError(f"ledger must not be hardlinked: {path}")
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise DurableLedgerError(
                    f"ledger has an incomplete final line at {line_number}"
                )
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DurableLedgerError(
                    f"invalid ledger JSON at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise DurableLedgerError(
                    f"ledger line {line_number} is not an object"
                )
            records.append(value)
    previous = ZERO_HASH
    for index, record in enumerate(records, start=1):
        if record.get("schema") != LEDGER_SCHEMA:
            raise DurableLedgerError(f"ledger schema mismatch at line {index}")
        if record.get("sequence") != index:
            raise DurableLedgerError(f"ledger sequence mismatch at line {index}")
        if record.get("previous_event_sha256") != previous:
            raise DurableLedgerError(f"ledger chain mismatch at line {index}")
        content = dict(record)
        recorded_hash = content.pop("event_sha256", None)
        expected_hash = _chain_hash(content)
        if recorded_hash != expected_hash:
            raise DurableLedgerError(f"ledger event hash mismatch at line {index}")
        previous = expected_hash
    return records


class HashChainLedger:
    """Thread-safe append-only JSONL ledger with fsync on every event."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        self.records = read_ledger(path)
        if self.records and any(record.get("run_id") != run_id
                                for record in self.records):
            raise DurableLedgerError("ledger contains another run_id")
        self._previous = (
            self.records[-1]["event_sha256"] if self.records else ZERO_HASH
        )

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = {
                "schema": LEDGER_SCHEMA,
                "run_id": self.run_id,
                "sequence": len(self.records) + 1,
                "timestamp": utc_now(),
                "event": event,
                "previous_event_sha256": self._previous,
                **fields,
            }
            record["event_sha256"] = _chain_hash(record)
            payload = (canonical_json(record) + "\n").encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            try:
                file_stat = os.fstat(descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise DurableLedgerError("ledger is not a regular file")
                if file_stat.st_nlink != 1:
                    raise DurableLedgerError("ledger file is hardlinked")
                written = 0
                while written < len(payload):
                    count = os.write(descriptor, payload[written:])
                    if count <= 0:
                        raise DurableLedgerError("short write to durable ledger")
                    written += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.records.append(record)
            self._previous = record["event_sha256"]
            return record


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return {"python_type": type(value).__name__, "repr": repr(value)}


def response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif isinstance(response, Mapping):
        value = dict(response)
    else:
        value = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": _jsonable(getattr(response, "usage", None)),
        }
    if not isinstance(value, dict):
        raise DurableLedgerError("model response cannot be serialized as an object")
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, Mapping):
        for key, item in extra.items():
            value.setdefault(str(key), _jsonable(item))
    return _jsonable(value)


def normalize_usage(value: Any) -> dict[str, int | None]:
    usage = _jsonable(value)
    if not isinstance(usage, dict):
        usage = {}

    def integer(name: str) -> int | None:
        raw = usage.get(name)
        if raw is None:
            return None
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    return {
        "prompt_tokens": integer("prompt_tokens"),
        "completion_tokens": integer("completion_tokens"),
        "total_tokens": integer("total_tokens"),
    }


def proxy_prefix(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise DurableLedgerError(f"exclusive proxy log is unavailable: {path}")
    payload = _stable_proxy_payload(path)
    return {
        "path": str(path.resolve()),
        "byte_offset": len(payload),
        "prefix_sha256": sha256_bytes(payload),
    }


def _stable_proxy_payload(path: Path) -> bytes:
    """Read an append-only proxy log at a complete-line boundary.

    Model maintenance can issue calls concurrently.  The proxy fsyncs each
    event before returning its HTTP response, but another proxy thread may be
    appending while this process takes a cutoff.  Short bounded retries avoid
    accepting a partial JSON line without weakening the fail-closed policy.
    """
    if path.is_symlink() or not path.is_file():
        raise DurableLedgerError(f"exclusive proxy log is unavailable: {path}")
    if path.stat(follow_symlinks=False).st_nlink != 1:
        raise DurableLedgerError("exclusive proxy log is hardlinked")
    for attempt in range(21):
        payload = path.read_bytes()
        if not payload or payload.endswith(b"\n"):
            return payload
        if attempt < 20:
            time.sleep(0.005)
    raise DurableLedgerError("exclusive proxy log has an incomplete final line")


def read_proxy_events(path: Path, *, logical_call_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    payload = _stable_proxy_payload(path)
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DurableLedgerError(
                f"invalid exclusive proxy JSON at line {line_number}"
            ) from exc
        if isinstance(value, dict) and value.get("logical_call_id") == logical_call_id:
            events.append(value)
    return events


def proxy_evidence(
    path: Path | None,
    *,
    logical_call_id: str,
    formal: bool,
) -> dict[str, Any]:
    if path is None:
        if formal:
            raise DurableLedgerError("formal model calls require an exclusive proxy log")
        return {
            "mode": "synthetic",
            "events": [],
            "client_http_attempts": 0,
            "upstream_http_attempts": 0,
            "unsupported_parameters": [],
            "log_prefix": None,
        }
    events = read_proxy_events(path, logical_call_id=logical_call_id)
    if formal and not events:
        raise DurableLedgerError(
            f"exclusive proxy has no event for logical call {logical_call_id}"
        )
    client_attempts = sum(int(event.get("client_http_attempts", 0) or 0)
                          for event in events)
    upstream_values = [event.get("upstream_http_attempts") for event in events]
    if formal and any(value is None for value in upstream_values):
        raise DurableLedgerError("proxy event lacks physical upstream attempt count")
    upstream_attempts = sum(int(value or 0) for value in upstream_values)
    unsupported = sorted({
        str(parameter)
        for event in events
        for parameter in event.get("unsupported_parameters", [])
    })
    return {
        "mode": "exclusive_proxy" if path is not None else "synthetic",
        "events": events,
        "client_http_attempts": client_attempts,
        "upstream_http_attempts": upstream_attempts,
        "unsupported_parameters": unsupported,
        "log_prefix": proxy_prefix(path),
    }


class DurableModelObserver:
    """Wrap one completion resource and persist every call attempt."""

    def __init__(
        self,
        *,
        ledger: HashChainLedger,
        artifact_root: Path,
        token_counter: Any,
        expected_model: str,
        proxy_log: Path | None,
        formal: bool,
    ) -> None:
        self.ledger = ledger
        self.artifact_root = artifact_root
        self.token_counter = token_counter
        self.expected_model = expected_model
        self.proxy_log = proxy_log
        self.formal = formal
        self._lock = threading.Lock()
        self._operation_id: str | None = None
        self._operation_ordinals: dict[str, int] = {}
        self._resource: Any = None
        self._create: Any = None

    @contextmanager
    def operation(self, operation_id: str) -> Iterator[None]:
        with self._lock:
            if self._operation_id is not None:
                raise DurableLedgerError("nested model-ledger operations are forbidden")
            self._operation_id = operation_id
        try:
            yield
        finally:
            with self._lock:
                self._operation_id = None

    def _next_call_id(self) -> tuple[str, str]:
        with self._lock:
            operation_id = self._operation_id
            if operation_id is None:
                raise DurableLedgerError("model call occurred outside a ledger operation")
            ordinal = self._operation_ordinals.get(operation_id, 0) + 1
            self._operation_ordinals[operation_id] = ordinal
        logical_call_id = f"{self.ledger.run_id}:{operation_id}:call-{ordinal:04d}"
        filename = hashlib.sha256(logical_call_id.encode()).hexdigest()[:32]
        return logical_call_id, filename

    def install(self, completion_resource: Any) -> None:
        if self._resource is not None:
            raise DurableLedgerError("model observer is already installed")
        original = getattr(completion_resource, "create", None)
        if not callable(original):
            raise DurableLedgerError("completion resource has no callable create")
        self._resource = completion_resource
        self._create = original
        completion_resource.create = self._observed_create

    def restore(self) -> None:
        if self._resource is not None:
            self._resource.create = self._create
            self._resource = None
            self._create = None

    def _observed_create(self, *args: Any, **kwargs: Any) -> Any:
        logical_call_id, filename = self._next_call_id()
        requested_model = str(kwargs.get("model", ""))
        if requested_model != self.expected_model:
            raise DurableLedgerError(
                f"unexpected requested model {requested_model!r}"
            )
        original_headers = dict(kwargs.get("extra_headers") or {})
        original_headers["X-Controlled-Logical-Call-ID"] = logical_call_id
        kwargs["extra_headers"] = original_headers
        visible_payload = {
            key: _jsonable(kwargs.get(key))
            for key in ("messages", "tools", "response_format")
            if kwargs.get(key) is not None
        }
        visible_text = canonical_json(visible_payload)
        request_payload = {
            "schema": REQUEST_SCHEMA,
            "logical_call_id": logical_call_id,
            "operation_id": self._operation_id,
            "requested_model": requested_model,
            "model_visible_payload": visible_payload,
            "model_visible_sha256": sha256_bytes(visible_text.encode("utf-8")),
            "local_visible_tokens": self.token_counter.count(visible_text),
            "tokenizer": self.token_counter.identity,
            "request_options": {
                key: _jsonable(value)
                for key, value in kwargs.items()
                if key not in {"messages", "tools", "response_format", "extra_headers"}
            },
            "transport_headers": {
                "X-Controlled-Logical-Call-ID": logical_call_id
            },
        }
        calls_dir = self.artifact_root / "calls"
        request_path = calls_dir / f"{filename}.request.json"
        response_path = calls_dir / f"{filename}.response.json"
        if request_path.exists() or response_path.exists():
            raise DurableLedgerError("model call artifact path already exists")
        request_sha = atomic_json(request_path, request_payload)
        start_prefix = proxy_prefix(self.proxy_log)
        started = time.monotonic()
        self.ledger.append(
            "model_call_started",
            operation_id=self._operation_id,
            logical_call_id=logical_call_id,
            request_path=str(request_path.relative_to(self.artifact_root)),
            request_sha256=request_sha,
            requested_model=requested_model,
            local_visible_tokens=request_payload["local_visible_tokens"],
            tokenizer=request_payload["tokenizer"],
            proxy_log_start=start_prefix,
        )
        try:
            response = self._create(*args, **kwargs)
            serialized = response_dict(response)
            response_payload = {
                "schema": RESPONSE_SCHEMA,
                "logical_call_id": logical_call_id,
                "response": serialized,
            }
            response_sha = atomic_json(response_path, response_payload)
            evidence = proxy_evidence(
                self.proxy_log, logical_call_id=logical_call_id, formal=self.formal
            )
            response_id = str(serialized.get("id", "") or "")
            response_model = str(serialized.get("model", "") or "")
            usage = normalize_usage(serialized.get("usage"))
            if self.formal:
                if response_model != self.expected_model or not response_id:
                    raise DurableLedgerError("formal response identity is incomplete")
                successful = [
                    event for event in evidence["events"]
                    if event.get("status") == "success"
                ]
                if len(successful) != 1:
                    raise DurableLedgerError(
                        "logical call must have exactly one successful proxy event"
                    )
                proxy_event = successful[0]
                if (proxy_event.get("response_id") != response_id
                        or proxy_event.get("actual_model") != response_model):
                    raise DurableLedgerError("response/proxy identity mismatch")
                if normalize_usage(proxy_event.get("usage")) != usage:
                    raise DurableLedgerError("response/proxy usage mismatch")
            self.ledger.append(
                "model_call_finished",
                operation_id=self._operation_id,
                logical_call_id=logical_call_id,
                response_path=str(response_path.relative_to(self.artifact_root)),
                response_sha256=response_sha,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                latency_s=round(time.monotonic() - started, 6),
                proxy_evidence=evidence,
            )
            return response
        except BaseException as exc:
            evidence_error = None
            try:
                evidence = proxy_evidence(
                    self.proxy_log,
                    logical_call_id=logical_call_id,
                    formal=False,
                )
            except Exception as proxy_exc:  # noqa: BLE001
                evidence = None
                evidence_error = f"{type(proxy_exc).__name__}: {proxy_exc}"
            self.ledger.append(
                "model_call_failed",
                operation_id=self._operation_id,
                logical_call_id=logical_call_id,
                response_path=(
                    str(response_path.relative_to(self.artifact_root))
                    if response_path.exists() else None
                ),
                response_sha256=(
                    sha256_file(response_path) if response_path.exists() else None
                ),
                latency_s=round(time.monotonic() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
                proxy_evidence=evidence,
                proxy_evidence_error=evidence_error,
            )
            raise


def ledger_state(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return terminal state and reject orphan model or operation starts."""
    operations: dict[str, str] = {}
    calls: dict[str, str] = {}
    committed: dict[str, dict[str, Any]] = {}
    failed_operations: list[str] = []
    for record in records:
        event = record.get("event")
        operation_id = record.get("operation_id")
        logical_call_id = record.get("logical_call_id")
        if event == "operation_started":
            if operation_id in operations:
                raise DurableLedgerError(f"duplicate operation start: {operation_id}")
            operations[str(operation_id)] = "started"
        elif event == "operation_committed":
            if operations.get(str(operation_id)) != "started":
                raise DurableLedgerError(f"operation commit without start: {operation_id}")
            operations[str(operation_id)] = "committed"
            committed[str(operation_id)] = record
        elif event == "operation_failed":
            if operations.get(str(operation_id)) != "started":
                raise DurableLedgerError(f"operation failure without start: {operation_id}")
            operations[str(operation_id)] = "failed"
            failed_operations.append(str(operation_id))
        elif event == "model_call_started":
            if operations.get(str(operation_id)) != "started":
                raise DurableLedgerError("model call start outside active operation")
            if logical_call_id in calls:
                raise DurableLedgerError(f"duplicate model call: {logical_call_id}")
            calls[str(logical_call_id)] = "started"
        elif event in {"model_call_finished", "model_call_failed"}:
            if calls.get(str(logical_call_id)) != "started":
                raise DurableLedgerError("model call terminal without start")
            calls[str(logical_call_id)] = str(event)
    orphan_operations = [key for key, value in operations.items() if value == "started"]
    orphan_calls = [key for key, value in calls.items() if value == "started"]
    if orphan_operations or orphan_calls:
        raise DurableLedgerError(
            f"unresolved ledger state: operations={orphan_operations}, "
            f"model_calls={orphan_calls}"
        )
    if failed_operations:
        raise DurableLedgerError(
            f"failed operations require a new output root: {failed_operations}"
        )
    return {
        "committed_operations": committed,
        "model_call_count": len(calls),
        "successful_model_calls": sum(value == "model_call_finished"
                                      for value in calls.values()),
        "failed_model_calls": sum(value == "model_call_failed"
                                  for value in calls.values()),
    }
