#!/usr/bin/env python3
"""Independent artifact auditor shared by R116 and R203."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import controlled_locomo_answer_contract as answer_contract  # noqa: E402
from src.evaluation.durable_model_ledger import (  # noqa: E402
    ledger_state,
    read_ledger,
)
from src.evaluation.visible_token_audit import audit_visible_token_trace  # noqa: E402
from src.evaluation.visible_token_budget import (  # noqa: E402
    TokenCounter,
    snapshot_memory_path,
)


SCHEMA_VERSION = "readonly-nativemem-control-v1"
EXPECTED_MODEL = "gpt-5.5"
EXPECTED_BUDGET = 20_000
DEFAULT_ANSWER_PROMPT_KIND = "locomo-lme-short-answer-v1"
DEFAULT_ANSWER_PROMPT_SHA256 = answer_contract.sha256_bytes(
    answer_contract.ANSWER_PROMPT.encode("utf-8")
)
STAGE_EVIDENCE_SCHEMA = "r203-stage-evidence-v1"
STAGE_PROVENANCE_SCHEMA = "r203-stage-provenance-v1"
SOURCE_ENABLED = {
    "dual_source": True,
    "topic_source": True,
    "timeline_source": True,
    "dual_no_source": False,
}
RETRIEVAL_PROMPT = """You retrieve evidence from a read-only NativeMem tree.
Condition: {condition}
Question: {question}

The following inventory has already passed the experiment's visible-token
gate. Use only the provided read-only tools. Inspect all evidence needed for
the question, then stop calling tools. Do not answer the question in this
stage.

{inventory}
"""


class ReadOnlyAuditError(RuntimeError):
    pass


def _strict_json(path: Path) -> Any:
    return answer_contract.read_json(path)


def _inside(root: Path, name: Any, *, label: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ReadOnlyAuditError(f"{label} path is invalid")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReadOnlyAuditError(f"{label} path escapes its artifact root")
    path = root / relative
    answer_contract.reject_symlink_components(path)
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ReadOnlyAuditError(f"{label} path escapes its artifact root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ReadOnlyAuditError(f"{label} is not a regular file")
    return resolved


def _trace(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise ReadOnlyAuditError("visible-token trace has an incomplete line")
    records = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReadOnlyAuditError(
                f"visible-token trace line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise ReadOnlyAuditError("visible-token trace record is not an object")
        records.append(value)
    return records


def _expand_sources(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    def add(session: int, start: int, end: int) -> None:
        if end < start or end - start > 500:
            raise ReadOnlyAuditError("invalid source range in delivered text")
        for turn in range(start, end + 1):
            value = f"D{session}:{turn}"
            if value not in seen:
                seen.add(value)
                values.append(value)

    for match in re.finditer(r"D(\d+):(\d+)(?:-(\d+))?", text):
        add(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or match.group(2)),
        )
    for bracket in re.findall(r"\[([^\]]*D\d+:[^\]]*)\]", text):
        current_session: int | None = None
        for part in re.split(r"\s*,\s*", bracket):
            exact = re.fullmatch(r"D(\d+):(\d+)(?:-(\d+))?", part)
            compact = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
            if exact is not None:
                current_session = int(exact.group(1))
                add(
                    current_session,
                    int(exact.group(2)),
                    int(exact.group(3) or exact.group(2)),
                )
            elif compact is not None and current_session is not None:
                add(
                    current_session,
                    int(compact.group(1)),
                    int(compact.group(2) or compact.group(1)),
                )
    return values


def expand_source_specs(values: Sequence[Any]) -> list[str]:
    """Independently expand exact, range, and compact-list source specs."""

    expanded: list[str] = []
    seen: set[str] = set()
    for raw in values:
        current_session: int | None = None
        for part in re.split(r"\s*,\s*", str(raw).strip().strip("[]")):
            exact = re.fullmatch(r"D(\d+):(\d+)(?:-(\d+))?", part)
            compact = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
            if exact is not None:
                current_session = int(exact.group(1))
                start = int(exact.group(2))
                end = int(exact.group(3) or start)
            elif compact is not None and current_session is not None:
                start = int(compact.group(1))
                end = int(compact.group(2) or start)
            else:
                raise ReadOnlyAuditError(f"invalid source specification: {part!r}")
            if end < start or end - start > 500:
                raise ReadOnlyAuditError(f"invalid source range: {part!r}")
            for turn in range(start, end + 1):
                source_id = f"D{current_session}:{turn}"
                if source_id not in seen:
                    seen.add(source_id)
                    expanded.append(source_id)
    return expanded


def audit_current_source_hashes(
    recorded: Any,
    *,
    expected_paths: Sequence[str],
    frozen_hashes: Mapping[str, str] | None = None,
) -> None:
    """Bind a run manifest to the exact current audited implementation."""

    if not isinstance(recorded, dict) or set(recorded) != set(expected_paths):
        raise ReadOnlyAuditError("source-hash inventory differs")
    for relative in expected_paths:
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise ReadOnlyAuditError("source-hash path escapes the repository") from exc
        if not path.is_file() or path.is_symlink():
            raise ReadOnlyAuditError(f"source-hash path is invalid: {relative}")
        live = answer_contract.sha256_file(path)
        if recorded.get(relative) != live:
            raise ReadOnlyAuditError(f"source hash differs: {relative}")
        if frozen_hashes is not None and relative in frozen_hashes:
            if live != frozen_hashes[relative]:
                raise ReadOnlyAuditError(f"frozen source hash differs: {relative}")


def _audit_prefix(prefix: Any) -> None:
    if prefix is None:
        return
    if not isinstance(prefix, dict):
        raise ReadOnlyAuditError("proxy prefix is invalid")
    path = Path(str(prefix.get("path", "")))
    if path.is_symlink() or not path.is_file():
        raise ReadOnlyAuditError("proxy prefix path is unavailable")
    offset = prefix.get("byte_offset")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ReadOnlyAuditError("proxy prefix byte offset is invalid")
    payload = path.read_bytes()
    if offset > len(payload):
        raise ReadOnlyAuditError("proxy prefix exceeds the current log")
    if (
        answer_contract.sha256_bytes(payload[:offset])
        != prefix.get("prefix_sha256")
    ):
        raise ReadOnlyAuditError("proxy prefix hash differs")


def _normalized_tool_call(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadOnlyAuditError("retrieval tool-call serialization differs")
    function = value.get("function")
    if isinstance(function, Mapping):
        call_id = value.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        rendered = value.get("repr")
        match = (
            re.fullmatch(
                r"namespace\(id=(.+?), function=namespace\(name=(.+?), "
                r"arguments=(.+)\)\)",
                rendered,
            )
            if value.get("python_type") == "SimpleNamespace"
            and isinstance(rendered, str)
            else None
        )
        if match is None:
            raise ReadOnlyAuditError("retrieval tool-call serialization differs")
        try:
            call_id, name, arguments = (
                ast.literal_eval(match.group(index)) for index in (1, 2, 3)
            )
        except (SyntaxError, ValueError) as exc:
            raise ReadOnlyAuditError(
                "retrieval tool-call serialization differs"
            ) from exc
    if not all(isinstance(item, str) and item for item in (call_id, name, arguments)):
        raise ReadOnlyAuditError("retrieval tool-call serialization differs")
    return {
        "id": call_id,
        "function": {"name": name, "arguments": arguments},
    }


def _normalized_message_chain(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ReadOnlyAuditError("retrieval message chain reconstruction differs")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ReadOnlyAuditError("retrieval message chain reconstruction differs")
        role = message.get("role")
        current = {"role": role, "content": message.get("content")}
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                raise ReadOnlyAuditError(
                    "retrieval message chain reconstruction differs"
                )
            current["tool_calls"] = [
                _normalized_tool_call(tool_call) for tool_call in tool_calls
            ]
        elif role == "tool":
            current["tool_call_id"] = message.get("tool_call_id")
        elif role != "user":
            raise ReadOnlyAuditError("retrieval message chain reconstruction differs")
        normalized.append(current)
    return normalized


def _expected_retrieval_tools(*, source_enabled: bool) -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_memory_files",
                "description": "List accessible markdown memory files.",
                "parameters": {
                    "type": "object",
                    "properties": {"prefix": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_memory_file",
                "description": "Read one accessible markdown memory file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": (
                    "Literal case-insensitive search over accessible files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    if source_enabled:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "resolve_sources",
                    "description": (
                        "Resolve LoCoMo Dn:m identifiers to original turns."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["source_ids"],
                    },
                },
            }
        )
    return tools


def _audit_model_ledger(
    artifact_dir: Path,
    ledger_path: Path,
    *,
    expected_calls: int,
    expected_question: str,
    condition: str,
    trace_records: Sequence[Mapping[str, Any]],
    formal: bool,
) -> dict[str, Any]:
    records = read_ledger(ledger_path)
    state = ledger_state(records)
    if state["model_call_count"] != expected_calls:
        raise ReadOnlyAuditError("retrieval model-call count differs")
    starts = {
        record["logical_call_id"]: record
        for record in records
        if record.get("event") == "model_call_started"
    }
    finishes = {
        record["logical_call_id"]: record
        for record in records
        if record.get("event") == "model_call_finished"
    }
    if set(starts) != set(finishes):
        raise ReadOnlyAuditError("retrieval call lifecycle differs")
    inventory_records = [
        record
        for record in trace_records
        if record.get("record_type") == "delivery"
        and record.get("event_id") == "observation-0000"
    ]
    if len(inventory_records) != 1 or not isinstance(
        inventory_records[0].get("delivered"), Mapping
    ):
        raise ReadOnlyAuditError("retrieval inventory reconstruction differs")
    expected_prompt = RETRIEVAL_PROMPT.format(
        condition=condition,
        question=expected_question,
        inventory=str(inventory_records[0]["delivered"].get("text", "")),
    )
    expected_messages: list[dict[str, Any]] = [
        {"role": "user", "content": expected_prompt}
    ]
    deliveries_by_call_id: dict[str, Mapping[str, Any]] = {}
    for record in trace_records:
        if record.get("record_type") != "delivery":
            continue
        metadata = record.get("metadata")
        tool_call_id = (
            metadata.get("tool_call_id")
            if isinstance(metadata, Mapping)
            else None
        )
        if tool_call_id is None:
            continue
        if not isinstance(tool_call_id, str) or tool_call_id in deliveries_by_call_id:
            raise ReadOnlyAuditError("retrieval delivery linkage differs")
        deliveries_by_call_id[tool_call_id] = record
    consumed_deliveries: set[str] = set()
    ordered_starts = sorted(starts.items(), key=lambda item: item[1]["sequence"])
    for call_index, (logical_call_id, start) in enumerate(ordered_starts):
        request_path = _inside(
            artifact_dir,
            start.get("request_path"),
            label="retrieval request",
        )
        if answer_contract.sha256_file(request_path) != start.get("request_sha256"):
            raise ReadOnlyAuditError("retrieval request hash differs")
        request = _strict_json(request_path)
        visible_payload = (
            request.get("model_visible_payload")
            if isinstance(request, dict)
            else None
        )
        messages = (
            visible_payload.get("messages")
            if isinstance(visible_payload, Mapping)
            else None
        )
        if not isinstance(visible_payload, Mapping) or visible_payload.get(
            "tools"
        ) != _expected_retrieval_tools(source_enabled=SOURCE_ENABLED[condition]):
            raise ReadOnlyAuditError(
                "retrieval tool definition reconstruction differs"
            )
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(messages[0], Mapping)
            or messages[0].get("role") != "user"
            or messages[0].get("content") != expected_prompt
        ):
            raise ReadOnlyAuditError("retrieval prompt reconstruction differs")
        if _normalized_message_chain(messages) != _normalized_message_chain(
            expected_messages
        ):
            raise ReadOnlyAuditError(
                "retrieval message chain reconstruction differs"
            )
        if (
            not isinstance(request, dict)
            or request.get("logical_call_id") != logical_call_id
            or request.get("requested_model") != EXPECTED_MODEL
            or request.get("transport_headers", {}).get(
                "X-Controlled-Logical-Call-ID"
            )
            != logical_call_id
            or request.get("local_visible_tokens") != start.get("local_visible_tokens")
        ):
            raise ReadOnlyAuditError("retrieval request linkage differs")
        finish = finishes[logical_call_id]
        response_path = _inside(
            artifact_dir,
            finish.get("response_path"),
            label="retrieval response",
        )
        if answer_contract.sha256_file(response_path) != finish.get("response_sha256"):
            raise ReadOnlyAuditError("retrieval response hash differs")
        response = _strict_json(response_path)
        response_body = response.get("response") if isinstance(response, dict) else None
        if (
            not isinstance(response_body, dict)
            or response.get("logical_call_id") != logical_call_id
            or response_body.get("id") != finish.get("response_id")
            or response_body.get("model") != finish.get("response_model")
            or finish.get("response_model") != EXPECTED_MODEL
        ):
            raise ReadOnlyAuditError("retrieval response identity differs")
        choices = response_body.get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], Mapping)
            else None
        )
        if not isinstance(message, Mapping):
            raise ReadOnlyAuditError("retrieval response message differs")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list):
                raise ReadOnlyAuditError("retrieval response tool calls differ")
            expected_messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    raise ReadOnlyAuditError("retrieval response tool call differs")
                tool_call_id = tool_call.get("id")
                delivery = deliveries_by_call_id.get(str(tool_call_id))
                delivered = (
                    delivery.get("delivered")
                    if isinstance(delivery, Mapping)
                    else None
                )
                if (
                    not isinstance(tool_call_id, str)
                    or tool_call_id in consumed_deliveries
                    or delivery is None
                    or (
                        delivered is not None
                        and not isinstance(delivered, Mapping)
                    )
                ):
                    raise ReadOnlyAuditError("retrieval delivery linkage differs")
                consumed_deliveries.add(tool_call_id)
                expected_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            str(delivered.get("text", ""))
                            if isinstance(delivered, Mapping)
                            else ""
                        ),
                    }
                )
        elif call_index + 1 < len(ordered_starts):
            raise ReadOnlyAuditError("retrieval continued after a terminal response")
        latency = finish.get("latency_s")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or latency < 0
        ):
            raise ReadOnlyAuditError("retrieval model-call latency is invalid")
        evidence = finish.get("proxy_evidence")
        if not isinstance(evidence, dict):
            raise ReadOnlyAuditError("retrieval proxy evidence is missing")
        if formal and evidence.get("mode") != "exclusive_proxy":
            raise ReadOnlyAuditError("formal retrieval lacks exclusive proxy evidence")
        _audit_prefix(start.get("proxy_log_start"))
        _audit_prefix(evidence.get("log_prefix"))
    if consumed_deliveries != set(deliveries_by_call_id):
        raise ReadOnlyAuditError("retrieval delivery linkage differs")
    return state


def _diagnostics(
    deliveries: Sequence[Mapping[str, Any]],
    access_log: Sequence[Mapping[str, Any]],
    gold_source_ids: Sequence[str],
    source_recall_eligible: bool,
) -> dict[str, Any]:
    retrieved: list[str] = []
    seen: set[str] = set()
    for record in deliveries:
        delivered = record.get("delivered")
        if isinstance(delivered, dict):
            for source_id in _expand_sources(str(delivered.get("text", ""))):
                if source_id not in seen:
                    seen.add(source_id)
                    retrieved.append(source_id)
    gold = list(dict.fromkeys(str(value) for value in gold_source_ids))
    hits = [value for value in gold if value in seen]
    first_relevant = None
    for access in access_log:
        if access.get("path") and set(access.get("source_ids", [])) & set(gold):
            first_relevant = access["path"]
            break
    return {
        "retrieved_source_ids": retrieved,
        "gold_source_ids": gold,
        "mapped_source_hits": hits,
        "mapped_source_recall": (
            len(hits) / len(gold)
            if source_recall_eligible and gold
            else None
        ),
        "source_recall_eligible": bool(source_recall_eligible),
        "first_relevant_file": first_relevant,
        "read_calls": sum(access.get("path") is not None for access in access_log),
        "tool_calls": len(access_log),
        "tool_latency_s": round(
            sum(float(access.get("latency_s", 0.0)) for access in access_log), 6
        ),
    }


def build_turn_index(conversation: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if match is None or not isinstance(value, list):
            continue
        date = str(conversation.get(f"session_{match.group(1)}_date_time", ""))
        for turn in value:
            if not isinstance(turn, Mapping):
                raise ReadOnlyAuditError("conversation turn is invalid")
            source_id = str(turn.get("dia_id", ""))
            if re.fullmatch(r"D\d+:\d+", source_id) is None or source_id in index:
                raise ReadOnlyAuditError("conversation source identifier is invalid")
            index[source_id] = {
                "date": date,
                "speaker": str(turn.get("speaker", "")),
                "text": str(turn.get("text", "")),
            }
    if not index:
        raise ReadOnlyAuditError("conversation turn index is empty")
    return index


def _memory_entries(root: Path, views: Sequence[str]) -> list[dict[str, Any]]:
    snapshot_memory_path(root)
    entries: list[dict[str, Any]] = []
    for view in views:
        view_root = root / view
        if not view_root.exists():
            continue
        if view_root.is_symlink() or not view_root.is_dir():
            raise ReadOnlyAuditError("stage-evidence view root is invalid")
        for path in sorted(view_root.rglob("*.md")):
            answer_contract.reject_symlink_components(path)
            if path.is_symlink() or not path.is_file():
                raise ReadOnlyAuditError("stage-evidence memory entry is invalid")
            text = path.read_text(encoding="utf-8")
            payload = {
                "path": path.relative_to(root).as_posix(),
                "content_sha256": answer_contract.sha256_bytes(
                    text.encode("utf-8")
                ),
                "source_ids": _expand_sources(text),
            }
            payload["trace_id"] = (
                f"memory-entry:{answer_contract.canonical_hash(payload)}"
            )
            entries.append(payload)
    return entries


def _trace_id_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str)
        and re.fullmatch(r"[a-z][a-z0-9_-]*:[^\s]+", item) is not None
        for item in value
    ):
        raise ReadOnlyAuditError(f"{label} trace IDs are invalid")
    return list(value)


def _stage_count(hits: Sequence[str], expected: Sequence[str]) -> dict[str, Any]:
    unique = [value for value in expected if value in set(hits)]
    return {
        "hit_source_ids": unique,
        "hit_count": len(unique),
        "expected_count": len(expected),
        "ratio": len(unique) / len(expected) if expected else None,
    }


def _rebuild_stage_evidence(
    *,
    provenance: Mapping[str, Any],
    memory_root: Path,
    memory_before: Mapping[str, Any],
    memory_after: Mapping[str, Any],
    condition: str,
    question_id: str,
    gold_source_ids: Sequence[str],
    source_recall_eligible: bool,
    turn_index: Mapping[str, Mapping[str, str]],
    trace_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold = list(dict.fromkeys(str(value) for value in gold_source_ids))
    mapping = provenance.get("mapping")
    canonical = provenance.get("canonical_entries_before_maintenance")
    maintenance = provenance.get("maintenance")
    if (
        provenance.get("schema_version") != STAGE_PROVENANCE_SCHEMA
        or provenance.get("question_id") != question_id
        or not isinstance(mapping, Mapping)
        or not isinstance(canonical, Mapping)
        or not isinstance(maintenance, Mapping)
    ):
        raise ReadOnlyAuditError("stage provenance identity differs")
    mapping_payload = {
        "question_id": question_id,
        "mapping_complete": mapping.get("mapping_complete"),
        "gold_source_ids": mapping.get("gold_source_ids"),
    }
    if (
        not isinstance(mapping.get("mapping_complete"), bool)
        or mapping.get("mapping_complete") is not bool(source_recall_eligible)
        or mapping.get("gold_source_ids") != gold
        or not all(re.fullmatch(r"D\d+:\d+", value) for value in gold)
        or mapping.get("record_sha256")
        != answer_contract.canonical_hash(mapping_payload)
    ):
        raise ReadOnlyAuditError("stage mapping provenance differs")
    mapping_trace_ids = _trace_id_list(mapping.get("trace_ids"), "mapping")
    if mapping_trace_ids != [
        f"gold-mapping:{question_id}:{mapping['record_sha256']}"
    ]:
        raise ReadOnlyAuditError("mapping trace ID differs")
    canonical_trace_ids = _trace_id_list(canonical.get("trace_ids"), "canonical")
    maintenance_trace_ids = _trace_id_list(
        maintenance.get("trace_ids"), "maintenance"
    )
    entries = canonical.get("entries")
    if not isinstance(entries, list):
        raise ReadOnlyAuditError("canonical stage entries are invalid")
    seen_paths: set[str] = set()
    expected_entry_traces: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReadOnlyAuditError("canonical stage entry differs")
        entry_path = Path(str(entry.get("path", "")))
        if (
            not isinstance(entry.get("path"), str)
            or not entry["path"].startswith("topics/")
            or not entry["path"].endswith(".md")
            or entry_path.is_absolute()
            or ".." in entry_path.parts
            or entry["path"] in seen_paths
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("content_sha256", "")))
            is None
            or not isinstance(entry.get("source_ids"), list)
            or not all(isinstance(item, str) for item in entry["source_ids"])
            or len(set(entry["source_ids"])) != len(entry["source_ids"])
            or not all(
                re.fullmatch(r"D\d+:\d+", item) for item in entry["source_ids"]
            )
        ):
            raise ReadOnlyAuditError("canonical stage entry differs")
        seen_paths.add(entry["path"])
        payload = {
            "path": entry["path"],
            "content_sha256": entry["content_sha256"],
            "source_ids": entry["source_ids"],
        }
        if entry.get("trace_id") != (
            f"memory-entry:{answer_contract.canonical_hash(payload)}"
        ):
            raise ReadOnlyAuditError("canonical stage entry trace differs")
        expected_entry_traces.append(str(entry["trace_id"]))
    canonical_sha = canonical.get("memory_sha256")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(canonical_sha)) is None
        or canonical_trace_ids
        != [f"canonical-memory:{canonical_sha}", *expected_entry_traces]
    ):
        raise ReadOnlyAuditError("canonical stage trace inventory differs")
    if (
        not isinstance(maintenance.get("operation_count"), int)
        or isinstance(maintenance.get("operation_count"), bool)
        or maintenance["operation_count"] < 0
        or maintenance.get("input_memory_sha256") != canonical_sha
        or maintenance.get("output_memory_sha256") != memory_before.get("sha256")
        or memory_before.get("sha256") != memory_after.get("sha256")
    ):
        raise ReadOnlyAuditError("maintenance stage binding differs")

    trace_ids: dict[str, list[str]] = {
        "mapping": mapping_trace_ids,
        "canonical_entries": [],
        "maintenance": maintenance_trace_ids,
        "paths": [],
        "retrieval": [],
        "source_resolution": [],
    }
    rebuilt: dict[str, Any] = {
        "schema_version": STAGE_EVIDENCE_SCHEMA,
        "question_id": question_id,
        "condition_id": condition,
        "gold_source_ids": gold,
        "gold_source_mapping_complete": mapping["mapping_complete"],
        "resolver_enabled": SOURCE_ENABLED[condition],
        "all_gold_required": True,
        "trace_ids": trace_ids,
        "stage_counts": {},
    }
    if not mapping["mapping_complete"]:
        rebuilt["evidence_status"] = "excluded_incomplete_mapping"
        return rebuilt
    if not gold:
        raise ReadOnlyAuditError("complete stage mapping has no gold sources")
    missing_turns = [value for value in gold if value not in turn_index]
    if missing_turns:
        raise ReadOnlyAuditError("mapped gold sources are absent from turn index")

    canonical_hits: list[str] = []
    for entry in entries:
        hits = [value for value in gold if value in entry["source_ids"]]
        if hits:
            canonical_hits.extend(hits)
            trace_ids["canonical_entries"].append(str(entry["trace_id"]))
    final_canonical = _memory_entries(memory_root, ("topics",))
    survived_hits = [
        value
        for value in gold
        if any(value in entry["source_ids"] for entry in final_canonical)
    ]
    views = {
        "dual_source": ("topics", "timeline"),
        "topic_source": ("topics",),
        "timeline_source": ("timeline",),
        "dual_no_source": ("topics", "timeline"),
    }[condition]
    accessible = _memory_entries(memory_root, views)
    path_hits: list[str] = []
    for entry in accessible:
        hits = [value for value in gold if value in entry["source_ids"]]
        if hits:
            path_hits.extend(hits)
            trace_ids["paths"].append(entry["trace_id"])

    reached_hits: list[str] = []
    resolved_hits: list[str] = []
    for record in trace_records:
        if record.get("record_type") != "delivery":
            continue
        delivered = record.get("delivered")
        if not isinstance(delivered, Mapping):
            continue
        text = str(delivered.get("text", ""))
        source_ids = _expand_sources(text)
        event_trace = (
            f"delivery:{record.get('event_id')}:{delivered.get('sha256')}"
        )
        if record.get("kind") == "source_resolution":
            for value in gold:
                turn = turn_index.get(value)
                if turn is None:
                    raise ReadOnlyAuditError("gold source is absent from turn index")
                expected = (
                    f"[{value}] ({turn['date']}) {turn['speaker']}: {turn['text']}"
                )
                if value in source_ids and expected in text:
                    resolved_hits.append(value)
                    if event_trace not in trace_ids["source_resolution"]:
                        trace_ids["source_resolution"].append(event_trace)
        else:
            for value in gold:
                if value in source_ids:
                    reached_hits.append(value)
                    if event_trace not in trace_ids["retrieval"]:
                        trace_ids["retrieval"].append(event_trace)

    counts = {
        "gold_source_in_canonical_entries": _stage_count(canonical_hits, gold),
        "gold_source_survived_maintenance": _stage_count(survived_hits, gold),
        "gold_source_path_valid": _stage_count(path_hits, gold),
        "retrieval_reached_gold_source": _stage_count(reached_hits, gold),
        "source_resolution_returned_gold_content": _stage_count(
            resolved_hits, gold
        ),
    }
    rebuilt.update(
        {
            key: count["hit_count"] == count["expected_count"]
            for key, count in counts.items()
        }
    )
    rebuilt["stage_counts"] = counts
    rebuilt["evidence_status"] = "complete"
    return rebuilt


def audit_question(
    *,
    artifact_dir: Path,
    memory_root: Path,
    method: str,
    condition: str,
    question_id: str,
    question: str,
    gold_source_ids: Sequence[str],
    source_recall_eligible: bool,
    formal: bool,
    stage_provenance: Mapping[str, Any] | None = None,
    turn_index: Mapping[str, Mapping[str, str]] | None = None,
    answer_prompt_template: str = answer_contract.ANSWER_PROMPT,
    answer_prompt_kind: str = DEFAULT_ANSWER_PROMPT_KIND,
    answer_prompt_template_sha256: str = DEFAULT_ANSWER_PROMPT_SHA256,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    result_path = artifact_dir / "result.json"
    result = _strict_json(result_path)
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != "complete"
        or result.get("method") != method
        or result.get("condition") != condition
        or result.get("question_id") != question_id
        or result.get("question_sha256")
        != answer_contract.sha256_bytes(question.encode("utf-8"))
    ):
        raise ReadOnlyAuditError("question result identity differs")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReadOnlyAuditError("question artifacts are missing")
    paths: dict[str, Path] = {}
    for key in (
        "visible_token_trace",
        "visible_token_manifest",
        "retrieval_model_ledger",
        "answer_ledger",
    ):
        paths[key] = _inside(artifact_dir, artifacts.get(key), label=key)
        if (
            answer_contract.sha256_file(paths[key])
            != artifacts.get(f"{key}_sha256")
        ):
            raise ReadOnlyAuditError(f"{key} hash differs")
    stage_recorded = result.get("stage_evidence")
    if stage_provenance is None:
        if stage_recorded is not None or "stage_evidence" in artifacts:
            raise ReadOnlyAuditError("unexpected stage evidence is attached")
    else:
        if not isinstance(stage_recorded, dict) or turn_index is None:
            raise ReadOnlyAuditError("required stage evidence is absent")
        paths["stage_evidence"] = _inside(
            artifact_dir, artifacts.get("stage_evidence"), label="stage_evidence"
        )
        if (
            answer_contract.sha256_file(paths["stage_evidence"])
            != artifacts.get("stage_evidence_sha256")
        ):
            raise ReadOnlyAuditError("stage-evidence hash differs")
    answer_contract.ensure_distinct_paths({"result": result_path, **paths})
    gate_audit = audit_visible_token_trace(
        paths["visible_token_trace"],
        manifest_path=paths["visible_token_manifest"],
        require_complete=True,
    )
    if gate_audit.get("audit_status") != "pass":
        raise ReadOnlyAuditError("visible-token audit failed")
    trace = _trace(paths["visible_token_trace"])
    header, finalizer = trace[0], trace[-1]
    deliveries = trace[1:-1]
    if (
        header.get("config", {}).get("configured_budget_tokens") != EXPECTED_BUDGET
        or header.get("metadata", {}).get("condition") != condition
        or header.get("metadata", {}).get("question_id") != question_id
        or finalizer.get("memory_changed") is not False
        or len(deliveries) < 1
        or deliveries[0].get("metadata", {}).get("tool_name")
        != "memory_inventory"
    ):
        raise ReadOnlyAuditError("visible-token linkage differs")
    for ordinal, delivery in enumerate(deliveries):
        if (
            delivery.get("record_type") != "delivery"
            or delivery.get("event_id") != f"observation-{ordinal:04d}"
            or delivery.get("metadata", {}).get("condition") != condition
            or (
                delivery.get("kind") == "source_resolution"
                and not SOURCE_ENABLED[condition]
            )
        ):
            raise ReadOnlyAuditError("delivery order or condition differs")
    source_total = gate_audit.get("cumulative_source_resolution_tokens")
    if condition == "dual_no_source" and source_total != 0:
        raise ReadOnlyAuditError("dual-no-source has source-resolution tokens")
    delivered_text = "".join(
        str(record["delivered"]["text"])
        for record in deliveries
        if isinstance(record.get("delivered"), dict)
    )
    if (
        answer_contract.sha256_bytes(answer_prompt_template.encode("utf-8"))
        != answer_prompt_template_sha256
    ):
        raise ReadOnlyAuditError("declared answer prompt template hash differs")
    reconstructed_prompt = answer_prompt_template.format(
        memories=delivered_text,
        question=question,
    )
    prompt = result.get("prompt")
    tokenizer = TokenCounter.from_identity(header["config"]["tokenizer"])
    if (
        not isinstance(prompt, dict)
        or prompt.get("kind") != answer_prompt_kind
        or prompt.get("template_sha256") != answer_prompt_template_sha256
        or prompt.get("sha256")
        != answer_contract.sha256_bytes(reconstructed_prompt.encode("utf-8"))
        or prompt.get("local_tokens") != tokenizer.count(reconstructed_prompt)
        or prompt["local_tokens"]
        + prompt.get("answer_completion_reservation_tokens", -1)
        > prompt.get("model_context_limit_tokens", -1)
    ):
        raise ReadOnlyAuditError("fixed answer prompt reconstruction differs")
    budget = result.get("budget")
    if (
        not isinstance(budget, dict)
        or budget.get("configured_tokens") != EXPECTED_BUDGET
        or budget.get("visible_tokens")
        != gate_audit.get("cumulative_visible_tokens")
        or budget.get("source_resolution_tokens") != source_total
        or budget.get("tokenizer") != header["config"]["tokenizer"]
    ):
        raise ReadOnlyAuditError("budget summary differs")
    retrieval = result.get("retrieval")
    if not isinstance(retrieval, dict) or retrieval.get("model") != EXPECTED_MODEL:
        raise ReadOnlyAuditError("retrieval identity differs")
    model_state = _audit_model_ledger(
        artifact_dir,
        paths["retrieval_model_ledger"],
        expected_calls=retrieval.get("model_calls", -1),
        expected_question=question,
        condition=condition,
        trace_records=trace,
        formal=formal,
    )
    if retrieval.get("ledger_state") != model_state:
        raise ReadOnlyAuditError("retrieval ledger state differs")
    committed = model_state.get("committed_operations", {})
    if len(committed) != 1:
        raise ReadOnlyAuditError("retrieval operation count differs")
    committed_record = next(iter(committed.values()))
    retrieval_latency = retrieval.get("latency_s")
    if (
        committed_record.get("question_id") != question_id
        or committed_record.get("condition") != condition
        or committed_record.get("model_calls") != retrieval.get("model_calls")
        or committed_record.get("access_log") != retrieval.get("access_log")
        or committed_record.get("retrieval_latency_s") != retrieval_latency
        or not isinstance(retrieval_latency, (int, float))
        or isinstance(retrieval_latency, bool)
        or retrieval_latency < 0
    ):
        raise ReadOnlyAuditError("retrieval operation linkage differs")
    answer = result.get("answer")
    if (
        not isinstance(answer, dict)
        or answer.get("requested_model") != EXPECTED_MODEL
        or answer.get("response_model") != EXPECTED_MODEL
        or not isinstance(answer.get("response_id"), str)
        or not answer["response_id"]
        or answer.get("raw_output_sha256")
        != answer_contract.sha256_bytes(str(answer.get("raw_output", "")).encode())
        or answer.get("text")
        != answer_contract.extract_answer(str(answer.get("raw_output", "")))
    ):
        raise ReadOnlyAuditError("fixed answer identity differs")
    answer_records = answer_contract.audit_ledger(
        paths["answer_ledger"],
        expected_run_id=f"{result['run_id']}:{question_id}:{condition}:answer",
    )
    if (
        not answer_records
        or answer_records[0]["event"] != "answer_started"
        or answer_records[-1]["event"] != "answer_completed"
    ):
        raise ReadOnlyAuditError("answer ledger lifecycle differs")
    evidence = answer.get("proxy_evidence")
    if not isinstance(evidence, dict):
        raise ReadOnlyAuditError("answer proxy evidence is missing")
    if formal and evidence.get("mode") != "exclusive_proxy":
        raise ReadOnlyAuditError("formal answer lacks exclusive proxy evidence")
    _audit_prefix(evidence.get("log_prefix"))
    events = evidence.get("events", [])
    if (
        not isinstance(events, list)
        or [event.get("event_id") for event in events]
        != answer.get("exclusive_proxy_event_ids")
        or sum(int(event.get("client_http_attempts", 0)) for event in events)
        != answer.get("client_http_attempts")
        or sum(int(event.get("upstream_http_attempts", 0)) for event in events)
        != answer.get("upstream_http_attempts")
    ):
        raise ReadOnlyAuditError("answer proxy attempt evidence differs")
    memory = result.get("memory")
    live = snapshot_memory_path(memory_root).descriptor
    if (
        not isinstance(memory, dict)
        or memory.get("unchanged") is not True
        or memory.get("before", {}).get("sha256") != live["sha256"]
        or memory.get("after", {}).get("sha256") != live["sha256"]
    ):
        raise ReadOnlyAuditError("read-only memory hash differs")
    expected_diagnostics = _diagnostics(
        deliveries,
        retrieval.get("access_log", []),
        gold_source_ids,
        source_recall_eligible,
    )
    if result.get("diagnostics") != expected_diagnostics:
        raise ReadOnlyAuditError("source/navigation diagnostics differ")
    if stage_provenance is not None:
        stage_file = _strict_json(paths["stage_evidence"])
        if stage_file != stage_recorded:
            raise ReadOnlyAuditError("stage-evidence artifact differs")
        rebuilt_stage = _rebuild_stage_evidence(
            provenance=stage_provenance,
            memory_root=memory_root,
            memory_before=memory["before"],
            memory_after=memory["after"],
            condition=condition,
            question_id=question_id,
            gold_source_ids=gold_source_ids,
            source_recall_eligible=source_recall_eligible,
            turn_index=turn_index,
            trace_records=trace,
        )
        if stage_recorded != rebuilt_stage:
            raise ReadOnlyAuditError("stage evidence does not reconstruct")
    report = {
        "status": "passed",
        "question_id": question_id,
        "condition": condition,
        "visible_tokens": budget["visible_tokens"],
        "source_resolution_tokens": budget["source_resolution_tokens"],
        "retrieval_model_calls": retrieval["model_calls"],
        "retrieval_latency_s": retrieval_latency,
        "tool_latency_s": expected_diagnostics["tool_latency_s"],
        "answer_client_http_attempts": answer["client_http_attempts"],
        "memory_sha256": live["sha256"],
        "mapped_source_recall": expected_diagnostics["mapped_source_recall"],
        "first_relevant_file": expected_diagnostics["first_relevant_file"],
    }
    if stage_recorded is not None:
        report["stage_evidence"] = stage_recorded
    return report


__all__ = [
    "ReadOnlyAuditError",
    "audit_current_source_hashes",
    "audit_question",
    "build_turn_index",
    "expand_source_specs",
]
