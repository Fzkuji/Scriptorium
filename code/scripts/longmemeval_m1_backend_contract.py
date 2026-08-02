#!/usr/bin/env python3
"""Execution contract shared by the LongMemEval-S Mem0/Graphiti runner and auditor.

The module contains only deterministic validation, durable accounting, and
model-observer code.  Importing it cannot create a backend, contact a model, or
write an artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import longmemeval_m1_contract as base  # noqa: E402
import run_longmemeval_m1_baselines as base_runner  # noqa: E402
from scripts.evaluation import durable_model_ledger as durable  # noqa: E402


BACKEND_RUN_SCHEMA = "longmemeval-m1-backend-run-v1"
BACKEND_SOURCE_TRACE_SCHEMA = "longmemeval-m1-backend-source-trace-v1"
BACKEND_RETRIEVAL_SCHEMA = "longmemeval-m1-backend-retrieval-v1"
BACKEND_PROXY_SLICE_SCHEMA = "longmemeval-m1-backend-proxy-slice-v1"
BACKEND_AUDIT_SCHEMA = "longmemeval-m1-backend-audit-v1"
BACKEND_EXECUTION_VERSION = "1"
RETRIEVAL_LIMIT = 100
EXPECTED_ACTUAL_MODEL = base.EXPECTED_MODEL
ACCEPTED_ACTUAL_MODEL_REGEX = r"^gpt-5\.5$"


class BackendContractError(RuntimeError):
    """Raised when a backend plan or execution artifact is not admissible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendContractError(message)


def content_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return base.canonical_hash(payload)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "scripts/longmemeval_m1_backend_contract.py": Path(__file__).resolve(),
        "scripts/run_longmemeval_m1_backends.py": (
            root / "scripts/run_longmemeval_m1_backends.py"
        ),
        "scripts/audit_longmemeval_m1_backends.py": (
            root / "scripts/audit_longmemeval_m1_backends.py"
        ),
        "scripts/longmemeval_m1_contract.py": Path(base.__file__).resolve(),
        "scripts/run_longmemeval_m1_baselines.py": (
            root / "scripts/run_longmemeval_m1_baselines.py"
        ),
        "scripts/audit_longmemeval_m1_baselines.py": (
            root / "scripts/audit_longmemeval_m1_baselines.py"
        ),
        "scripts/controlled_r301_model_proxy.py": (
            root / "scripts/controlled_r301_model_proxy.py"
        ),
        "src/evaluation/durable_model_ledger.py": Path(durable.__file__).resolve(),
        "src/evaluation/visible_token_budget.py": (
            root / "src/evaluation/visible_token_budget.py"
        ),
        "src/evaluation/visible_token_audit.py": (
            root / "src/evaluation/visible_token_audit.py"
        ),
        "src/evaluation/prompts.py": root / "src/evaluation/prompts.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    require(not missing, f"backend execution source files are missing: {missing}")
    return {name: base.sha256_file(path) for name, path in paths.items()}


def validate_plan(
    *,
    plan_path: Path,
    dataset_path: Path,
    preregistration_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate a frozen backend plan against the pinned dataset and preregistration."""

    plan_path = plan_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    try:
        base.ensure_distinct_paths(
            {
                "plan": plan_path,
                "dataset": dataset_path,
                "preregistration": preregistration_path,
            }
        )
        data = base.validate_dataset(dataset_path)
        preregistration = base_runner.validate_preregistration(
            preregistration_path=preregistration_path,
            dataset_path=dataset_path,
        )
        value = base.read_json(plan_path)
    except base.ContractError as exc:
        raise BackendContractError(str(exc)) from exc
    require(isinstance(value, dict), "backend plan is not an object")
    plan = dict(value)
    method = plan.get("method")
    scope = plan.get("scope")
    require(method in base.PLANNED_BACKENDS, "backend plan method is invalid")
    require(scope in {"smoke", "formal"}, "backend plan scope is invalid")
    require(
        plan.get("schema_version") == base.BACKEND_PLAN_SCHEMA_VERSION
        and plan.get("status") == "planned_not_executed"
        and plan.get("benchmark") == "LongMemEval-S"
        and plan.get("model_calls") == 0
        and plan.get("network_calls") == 0,
        "backend plan execution status differs",
    )
    require(
        plan.get("plan_content_sha256") == content_hash(plan, "plan_content_sha256"),
        "backend plan content hash differs",
    )
    require(
        plan.get("configuration") == base.method_configuration(str(method)),
        "backend plan configuration differs",
    )
    require(
        plan.get("dependencies") == base.strict_dependency_snapshot(),
        "backend plan dependency snapshot differs",
    )
    require(
        plan.get("dataset_sha256")
        == base.sha256_file(dataset_path)
        == base.EXPECTED_DATASET_SHA256
        and plan.get("preregistration_sha256") == base.sha256_file(preregistration_path)
        and plan.get("preregistration_content_sha256")
        == preregistration["preregistration_content_sha256"],
        "backend plan frozen inputs differ",
    )
    output_root_value = plan.get("output_root")
    require(
        isinstance(output_root_value, str) and output_root_value,
        "backend plan output root is missing",
    )
    output_root = Path(output_root_value)
    try:
        base.reject_symlink_components(output_root)
    except base.ContractError as exc:
        raise BackendContractError(str(exc)) from exc
    items = plan.get("items")
    require(isinstance(items, list), "backend plan items are invalid")
    if scope == "formal":
        expected_indices = list(range(base.EXPECTED_ITEMS))
    else:
        require(len(items) == 1, "smoke plan must contain exactly one item")
        smoke_index = items[0].get("dataset_index") if items else None
        require(
            isinstance(smoke_index, int)
            and not isinstance(smoke_index, bool)
            and 0 <= smoke_index < base.EXPECTED_ITEMS,
            "smoke plan dataset index is invalid",
        )
        expected_indices = [smoke_index]
    require(
        [item.get("dataset_index") for item in items] == expected_indices,
        "backend plan item ordering differs",
    )
    workspace_identities: set[str] = set()
    for record, index in zip(items, expected_indices):
        require(isinstance(record, dict), f"backend plan item {index} is invalid")
        source = data[index]
        require(
            record.get("question_id") == source["question_id"]
            and record.get("history_content_sha256")
            == base.history_content_hash(source)
            and record.get("history_owner_sha256") == base.history_owner_hash(source)
            and record.get("history_session_ids") == source["haystack_session_ids"],
            f"backend plan item {index} history binding differs",
        )
        expected_workspace = base.backend_workspace_descriptor(
            method=str(method),
            output_root=output_root,
            item_index=index,
            question_id=str(source["question_id"]),
        )
        require(
            record.get("workspace") == expected_workspace,
            f"backend plan item {index} workspace differs",
        )
        workspace_identity = base.path_identity(Path(expected_workspace["workspace"]))
        require(
            workspace_identity not in workspace_identities,
            "backend plan reuses an item workspace",
        )
        workspace_identities.add(workspace_identity)
        require(
            record.get("checkpoint") == "checkpoint.json"
            and record.get("attempt_ledger") == "attempts.jsonl"
            and record.get("visible_token_budget") == base.VISIBLE_BUDGET_TOKENS,
            f"backend plan item {index} execution interface differs",
        )
    require(plan.get("item_count") == len(items), "backend plan item count differs")
    return plan, data, preregistration


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return {"python_type": type(value).__name__, "repr": repr(value)}


class AsyncDurableModelObserver:
    """Async equivalent of ``DurableModelObserver`` for Graphiti's AsyncOpenAI client."""

    def __init__(
        self,
        *,
        ledger: durable.HashChainLedger,
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
        self._ordinals: dict[str, int] = {}
        self._resource: Any = None
        self._create: Any = None

    @contextmanager
    def operation(self, operation_id: str) -> Iterator[None]:
        with self._lock:
            if self._operation_id is not None:
                raise durable.DurableLedgerError(
                    "nested model-ledger operations are forbidden"
                )
            self._operation_id = operation_id
        try:
            yield
        finally:
            with self._lock:
                self._operation_id = None

    def install(self, completion_resource: Any) -> None:
        if self._resource is not None:
            raise durable.DurableLedgerError("model observer is already installed")
        original = getattr(completion_resource, "create", None)
        if not callable(original):
            raise durable.DurableLedgerError("completion resource has no create method")
        self._resource = completion_resource
        self._create = original
        completion_resource.create = self._observed_create

    def restore(self) -> None:
        if self._resource is not None:
            self._resource.create = self._create
            self._resource = None
            self._create = None

    def _next_call(self) -> tuple[str, str, str]:
        with self._lock:
            operation_id = self._operation_id
            if operation_id is None:
                raise durable.DurableLedgerError(
                    "model call occurred outside a ledger operation"
                )
            ordinal = self._ordinals.get(operation_id, 0) + 1
            self._ordinals[operation_id] = ordinal
        logical_call_id = f"{self.ledger.run_id}:{operation_id}:call-{ordinal:04d}"
        filename = hashlib.sha256(logical_call_id.encode()).hexdigest()[:32]
        return operation_id, logical_call_id, filename

    async def _observed_create(self, *args: Any, **kwargs: Any) -> Any:
        operation_id, logical_call_id, filename = self._next_call()
        requested_model = str(kwargs.get("model", ""))
        if requested_model != self.expected_model:
            raise durable.DurableLedgerError(
                f"unexpected requested model {requested_model!r}"
            )
        headers = dict(kwargs.get("extra_headers") or {})
        headers["X-Controlled-Logical-Call-ID"] = logical_call_id
        kwargs["extra_headers"] = headers
        visible_payload = {
            key: jsonable(kwargs.get(key))
            for key in ("messages", "tools", "response_format")
            if kwargs.get(key) is not None
        }
        visible_text = durable.canonical_json(visible_payload)
        request_payload = {
            "schema": durable.REQUEST_SCHEMA,
            "logical_call_id": logical_call_id,
            "operation_id": operation_id,
            "requested_model": requested_model,
            "model_visible_payload": visible_payload,
            "model_visible_sha256": durable.sha256_bytes(visible_text.encode()),
            "local_visible_tokens": self.token_counter.count(visible_text),
            "tokenizer": self.token_counter.identity,
            "request_options": {
                key: jsonable(value)
                for key, value in kwargs.items()
                if key not in {"messages", "tools", "response_format", "extra_headers"}
            },
            "transport_headers": {"X-Controlled-Logical-Call-ID": logical_call_id},
        }
        calls_dir = self.artifact_root / "calls"
        request_path = calls_dir / f"{filename}.request.json"
        response_path = calls_dir / f"{filename}.response.json"
        if request_path.exists() or response_path.exists():
            raise durable.DurableLedgerError("model call artifact already exists")
        request_sha = durable.atomic_json(request_path, request_payload)
        start_prefix = durable.proxy_prefix(self.proxy_log)
        self.ledger.append(
            "model_call_started",
            operation_id=operation_id,
            logical_call_id=logical_call_id,
            request_path=str(request_path.relative_to(self.artifact_root)),
            request_sha256=request_sha,
            requested_model=requested_model,
            local_visible_tokens=request_payload["local_visible_tokens"],
            tokenizer=request_payload["tokenizer"],
            proxy_log_start=start_prefix,
        )
        started = time.monotonic()
        try:
            response = await self._create(*args, **kwargs)
            serialized = durable.response_dict(response)
            response_sha = durable.atomic_json(
                response_path,
                {
                    "schema": durable.RESPONSE_SCHEMA,
                    "logical_call_id": logical_call_id,
                    "response": serialized,
                },
            )
            evidence = durable.proxy_evidence(
                self.proxy_log,
                logical_call_id=logical_call_id,
                formal=self.formal,
            )
            response_id = str(serialized.get("id", "") or "")
            response_model = str(serialized.get("model", "") or "")
            usage = durable.normalize_usage(serialized.get("usage"))
            if self.formal:
                successful = [
                    event
                    for event in evidence["events"]
                    if event.get("status") == "success"
                ]
                if (
                    response_model != self.expected_model
                    or not response_id
                    or len(successful) != 1
                ):
                    raise durable.DurableLedgerError(
                        "formal response identity is incomplete"
                    )
                proxy_event = successful[0]
                if (
                    proxy_event.get("requested_model") != self.expected_model
                    or proxy_event.get("actual_model") != response_model
                    or proxy_event.get("response_id") != response_id
                    or durable.normalize_usage(proxy_event.get("usage")) != usage
                ):
                    raise durable.DurableLedgerError(
                        "response and proxy identity differ"
                    )
            self.ledger.append(
                "model_call_finished",
                operation_id=operation_id,
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
                evidence = durable.proxy_evidence(
                    self.proxy_log,
                    logical_call_id=logical_call_id,
                    formal=False,
                )
            except Exception as proxy_exc:  # noqa: BLE001
                evidence = None
                evidence_error = f"{type(proxy_exc).__name__}: {proxy_exc}"
            self.ledger.append(
                "model_call_failed",
                operation_id=operation_id,
                logical_call_id=logical_call_id,
                response_path=(
                    str(response_path.relative_to(self.artifact_root))
                    if response_path.exists()
                    else None
                ),
                response_sha256=(
                    durable.sha256_file(response_path)
                    if response_path.exists()
                    else None
                ),
                latency_s=round(time.monotonic() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
                proxy_evidence=evidence,
                proxy_evidence_error=evidence_error,
            )
            raise


def model_ledger_summary(
    *,
    ledger_path: Path,
    artifact_root: Path,
    formal: bool,
) -> dict[str, Any]:
    records = durable.read_ledger(ledger_path)
    try:
        state = durable.ledger_state(records)
    except durable.DurableLedgerError as exc:
        raise BackendContractError(str(exc)) from exc
    require(state["failed_model_calls"] == 0, "model ledger has a failed call")
    for operation_id, record in state["committed_operations"].items():
        relative = record.get("artifact_path")
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"operation {operation_id} artifact path is invalid",
        )
        artifact = artifact_root / relative
        require(
            artifact.is_file()
            and not artifact.is_symlink()
            and artifact.stat(follow_symlinks=False).st_nlink == 1
            and base.sha256_file(artifact) == record.get("artifact_sha256"),
            f"operation {operation_id} artifact hash differs",
        )
    finished = [
        record for record in records if record.get("event") == "model_call_finished"
    ]
    logical_ids: list[str] = []
    upstream_attempts = 0
    for record in finished:
        logical_call_id = record.get("logical_call_id")
        require(
            isinstance(logical_call_id, str) and logical_call_id,
            "model ledger logical call ID is invalid",
        )
        logical_ids.append(logical_call_id)
        require(
            (not formal) or record.get("response_model") == EXPECTED_ACTUAL_MODEL,
            "model ledger actual model differs",
        )
        relative = record.get("response_path")
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "model ledger response path is invalid",
        )
        path = artifact_root / relative
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat(follow_symlinks=False).st_nlink == 1
            and base.sha256_file(path) == record.get("response_sha256"),
            "model ledger response hash differs",
        )
        response_artifact = base.read_json(path)
        response = (
            response_artifact.get("response")
            if isinstance(response_artifact, Mapping)
            else None
        )
        require(
            isinstance(response_artifact, Mapping)
            and response_artifact.get("schema") == durable.RESPONSE_SCHEMA
            and response_artifact.get("logical_call_id") == logical_call_id
            and isinstance(response, Mapping)
            and response.get("id") == record.get("response_id")
            and response.get("model") == record.get("response_model")
            and durable.normalize_usage(response.get("usage")) == record.get("usage"),
            "model response artifact identity differs",
        )
        evidence = record.get("proxy_evidence")
        require(isinstance(evidence, dict), "model ledger proxy evidence is missing")
        if formal:
            successful = [
                event
                for event in evidence.get("events", [])
                if isinstance(event, dict) and event.get("status") == "success"
            ]
            require(
                evidence.get("mode") == "exclusive_proxy" and len(successful) == 1,
                "formal model call lacks exclusive proxy evidence",
            )
            proxy_event = successful[0]
            proxy_meta = response.get("exclusive_proxy_meta")
            require(
                proxy_event.get("logical_call_id") == logical_call_id
                and proxy_event.get("requested_model") == EXPECTED_ACTUAL_MODEL
                and proxy_event.get("actual_model") == record.get("response_model")
                and proxy_event.get("response_id") == record.get("response_id")
                and durable.normalize_usage(proxy_event.get("usage"))
                == record.get("usage")
                and isinstance(proxy_meta, Mapping)
                and proxy_meta.get("event_id") == proxy_event.get("event_id")
                and proxy_meta.get("logical_call_id") == logical_call_id
                and proxy_meta.get("request_sha256")
                == proxy_event.get("request_sha256")
                and proxy_meta.get("upstream_http_attempts")
                == proxy_event.get("upstream_http_attempts"),
                "model response and proxy identity differ",
            )
        upstream = evidence.get("upstream_http_attempts", 0)
        require(
            isinstance(upstream, int)
            and not isinstance(upstream, bool)
            and upstream >= 0,
            "proxy upstream attempt count is invalid",
        )
        upstream_attempts += upstream
    starts = [
        record for record in records if record.get("event") == "model_call_started"
    ]
    for record in starts:
        relative = record.get("request_path")
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "model request path is invalid",
        )
        path = artifact_root / relative
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat(follow_symlinks=False).st_nlink == 1
            and base.sha256_file(path) == record.get("request_sha256"),
            "model request artifact hash differs",
        )
        request_artifact = base.read_json(path)
        require(
            isinstance(request_artifact, Mapping)
            and request_artifact.get("schema") == durable.REQUEST_SCHEMA
            and request_artifact.get("logical_call_id") == record.get("logical_call_id")
            and request_artifact.get("requested_model") == record.get("requested_model")
            and (
                (not formal) or record.get("requested_model") == EXPECTED_ACTUAL_MODEL
            ),
            "requested builder model differs",
        )
    require(len(starts) == len(finished), "model ledger call count differs")
    if formal:
        require(finished, "formal backend build made no model calls")
        require(upstream_attempts > 0, "formal backend build made no upstream attempts")
    return {
        "record_count": len(records),
        "model_calls": len(finished),
        "network_calls": upstream_attempts,
        "logical_call_ids": logical_ids,
        "ledger_sha256": base.sha256_file(ledger_path),
        "state": state,
    }


def proxy_slice(
    *,
    path: Path | None,
    start: Mapping[str, Any] | None,
    logical_call_ids: Sequence[str],
    formal: bool,
) -> dict[str, Any]:
    if path is None:
        require(not formal, "formal backend execution requires an exclusive proxy log")
        return {
            "schema_version": BACKEND_PROXY_SLICE_SCHEMA,
            "mode": "synthetic",
            "start": None,
            "end": None,
            "event_count": 0,
            "logical_call_ids": list(logical_call_ids),
            "slice_sha256": base.sha256_bytes(b""),
        }
    require(start is not None, "proxy start prefix is missing")
    end = durable.proxy_prefix(path)
    require(end is not None, "proxy end prefix is missing")
    resolved = str(path.resolve())
    require(
        start.get("path") == resolved and end.get("path") == resolved,
        "proxy prefix path differs",
    )
    payload = path.read_bytes()
    start_offset = start.get("byte_offset")
    end_offset = end.get("byte_offset")
    require(
        isinstance(start_offset, int)
        and isinstance(end_offset, int)
        and 0 <= start_offset <= end_offset <= len(payload),
        "proxy prefix offsets are invalid",
    )
    require(
        base.sha256_bytes(payload[:start_offset]) == start.get("prefix_sha256")
        and base.sha256_bytes(payload[:end_offset]) == end.get("prefix_sha256"),
        "proxy prefix hash differs",
    )
    raw_slice = payload[start_offset:end_offset]
    require(not raw_slice or raw_slice.endswith(b"\n"), "proxy slice is incomplete")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_slice.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendContractError(
                f"proxy slice line {line_number} is invalid"
            ) from exc
        require(isinstance(value, dict), "proxy slice event is not an object")
        events.append(value)
    observed_ids = [str(event.get("logical_call_id", "")) for event in events]
    expected_ids = list(logical_call_ids)
    require(
        len(expected_ids) == len(set(expected_ids))
        and set(observed_ids) == set(expected_ids),
        "exclusive proxy slice contains missing or unrelated calls",
    )
    if formal:
        require(
            all(
                event.get("requested_model") == EXPECTED_ACTUAL_MODEL
                and (
                    event.get("status") != "success"
                    or event.get("actual_model") == EXPECTED_ACTUAL_MODEL
                )
                for event in events
            ),
            "exclusive proxy slice contains a wrong-model call",
        )
        require(
            all(
                len(
                    [
                        event
                        for event in events
                        if event.get("logical_call_id") == logical_call_id
                        and event.get("status") == "success"
                    ]
                )
                == 1
                for logical_call_id in expected_ids
            ),
            "exclusive proxy slice must contain one success per logical call",
        )
    return {
        "schema_version": BACKEND_PROXY_SLICE_SCHEMA,
        "mode": "exclusive_proxy",
        "path": resolved,
        "start": dict(start),
        "end": dict(end),
        "event_count": len(events),
        "logical_call_ids": expected_ids,
        "event_ids": [event.get("event_id") for event in events],
        "slice_sha256": base.sha256_bytes(raw_slice),
    }


def validate_proxy_slice(value: Mapping[str, Any], *, formal: bool) -> None:
    require(
        value.get("schema_version") == BACKEND_PROXY_SLICE_SCHEMA,
        "proxy slice schema differs",
    )
    if value.get("mode") == "synthetic":
        require(not formal, "formal execution recorded a synthetic proxy slice")
        return
    path_value = value.get("path")
    require(isinstance(path_value, str) and path_value, "proxy slice path is invalid")
    path = Path(path_value)
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat(follow_symlinks=False).st_nlink == 1,
        "exclusive proxy log is unavailable",
    )
    start = value.get("start")
    end = value.get("end")
    require(
        isinstance(start, Mapping) and isinstance(end, Mapping),
        "proxy slice prefixes are missing",
    )
    resolved = str(path.resolve())
    require(
        start.get("path") == resolved
        and end.get("path") == resolved
        and path_value == resolved,
        "proxy slice path differs",
    )
    payload = path.read_bytes()
    start_offset = start.get("byte_offset")
    end_offset = end.get("byte_offset")
    require(
        isinstance(start_offset, int)
        and isinstance(end_offset, int)
        and 0 <= start_offset <= end_offset <= len(payload),
        "proxy slice offsets are invalid",
    )
    require(
        base.sha256_bytes(payload[:start_offset]) == start.get("prefix_sha256")
        and base.sha256_bytes(payload[:end_offset]) == end.get("prefix_sha256"),
        "proxy slice prefix hash differs",
    )
    raw_slice = payload[start_offset:end_offset]
    require(not raw_slice or raw_slice.endswith(b"\n"), "proxy slice is incomplete")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_slice.splitlines(), start=1):
        try:
            parsed = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendContractError(
                f"proxy slice line {line_number} is invalid"
            ) from exc
        require(isinstance(parsed, dict), "proxy slice event is not an object")
        events.append(parsed)
    logical_call_ids = value.get("logical_call_ids")
    require(
        isinstance(logical_call_ids, list), "proxy logical call inventory is invalid"
    )
    observed_ids = [str(event.get("logical_call_id", "")) for event in events]
    require(
        len(logical_call_ids) == len(set(logical_call_ids))
        and set(observed_ids) == set(logical_call_ids),
        "exclusive proxy slice contains missing or unrelated calls",
    )
    if formal:
        require(
            all(
                event.get("requested_model") == EXPECTED_ACTUAL_MODEL
                and (
                    event.get("status") != "success"
                    or event.get("actual_model") == EXPECTED_ACTUAL_MODEL
                )
                for event in events
            ),
            "exclusive proxy slice contains a wrong-model call",
        )
        require(
            all(
                len(
                    [
                        event
                        for event in events
                        if event.get("logical_call_id") == logical_call_id
                        and event.get("status") == "success"
                    ]
                )
                == 1
                for logical_call_id in logical_call_ids
            ),
            "exclusive proxy slice must contain one success per logical call",
        )
    observed = {
        "schema_version": BACKEND_PROXY_SLICE_SCHEMA,
        "mode": "exclusive_proxy",
        "path": resolved,
        "start": dict(start),
        "end": dict(end),
        "event_count": len(events),
        "logical_call_ids": logical_call_ids,
        "event_ids": [event.get("event_id") for event in events],
        "slice_sha256": base.sha256_bytes(raw_slice),
    }
    require(observed == dict(value), "proxy slice evidence differs from the live log")


def validate_smoke_audit(
    *,
    path: Path,
    method: str,
    formal_workspace_paths: Sequence[Path],
) -> dict[str, Any]:
    value = base.read_json(path.expanduser().resolve())
    require(isinstance(value, dict), "smoke audit is not an object")
    report = dict(value)
    require(
        report.get("schema_version") == BACKEND_AUDIT_SCHEMA
        and report.get("status") == "passed"
        and report.get("scope") == "smoke"
        and report.get("method") == method
        and report.get("items") == 1
        and isinstance(report.get("model_calls"), int)
        and report.get("model_calls", 0) > 0
        and isinstance(report.get("network_calls"), int)
        and report.get("network_calls", 0) > 0,
        "formal launch requires one passed real-model smoke audit",
    )
    smoke_workspaces = report.get("workspace_paths")
    require(
        isinstance(smoke_workspaces, list) and len(smoke_workspaces) == 1,
        "smoke audit workspace inventory is invalid",
    )
    smoke_identity = base.path_identity(Path(smoke_workspaces[0]))
    require(
        all(
            base.path_identity(path) != smoke_identity
            for path in formal_workspace_paths
        ),
        "smoke and formal plans share a backend workspace",
    )
    return report


__all__ = [
    "ACCEPTED_ACTUAL_MODEL_REGEX",
    "AsyncDurableModelObserver",
    "BACKEND_AUDIT_SCHEMA",
    "BACKEND_EXECUTION_VERSION",
    "BACKEND_PROXY_SLICE_SCHEMA",
    "BACKEND_RETRIEVAL_SCHEMA",
    "BACKEND_RUN_SCHEMA",
    "BACKEND_SOURCE_TRACE_SCHEMA",
    "BackendContractError",
    "EXPECTED_ACTUAL_MODEL",
    "RETRIEVAL_LIMIT",
    "content_hash",
    "model_ledger_summary",
    "proxy_slice",
    "require",
    "source_hashes",
    "validate_plan",
    "validate_proxy_slice",
    "validate_smoke_audit",
]
