#!/usr/bin/env python3
"""Shared read-only retrieval boundary for R116 and R203.

The module never imports or modifies the active NativeMem runners.  It exposes
only deterministic read/list/search/source-resolution tools over an immutable
memory tree.  Every model-visible memory payload crosses one
``VisibleTokenBudgetGate`` before it can enter either retrieval messages or the
fixed answer prompt.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import controlled_locomo_answer_contract as answer_contract  # noqa: E402
from src.evaluation.durable_model_ledger import (  # noqa: E402
    DurableModelObserver,
    HashChainLedger,
    ledger_state,
    proxy_evidence,
)
from src.evaluation.visible_token_budget import (  # noqa: E402
    DeliveryResult,
    TokenCounter,
    VisibleTokenBudgetGate,
    snapshot_memory_path,
)


SCHEMA_VERSION = "readonly-nativemem-control-v1"
STAGE_EVIDENCE_SCHEMA = "r203-stage-evidence-v1"
STAGE_PROVENANCE_SCHEMA = "r203-stage-provenance-v1"
EXPECTED_MODEL = "gpt-5.5"
EXPECTED_BUDGET = 20_000
DEFAULT_ANSWER_PROMPT_KIND = "locomo-lme-short-answer-v1"
DEFAULT_ANSWER_PROMPT_SHA256 = answer_contract.sha256_bytes(
    answer_contract.ANSWER_PROMPT.encode("utf-8")
)
CONDITIONS = (
    "dual_source",
    "topic_source",
    "timeline_source",
    "dual_no_source",
)
CONDITION_VIEWS = {
    "dual_source": ("topics", "timeline"),
    "topic_source": ("topics",),
    "timeline_source": ("timeline",),
    "dual_no_source": ("topics", "timeline"),
}
CONDITION_SOURCE_ENABLED = {
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


class ReadOnlyControlError(RuntimeError):
    """Raised when an R116/R203 integrity requirement fails."""


class ReadOnlyToolInputError(ReadOnlyControlError):
    """Raised when a model-selected read-only tool input is invalid."""


def canonical_hash(value: Any) -> str:
    return answer_contract.canonical_hash(value)


def sha256_file(path: Path) -> str:
    return answer_contract.sha256_file(path)


def assert_safe_tree(root: Path) -> None:
    answer_contract.reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise ReadOnlyControlError(f"memory root is not a regular directory: {root}")
    seen_inodes: dict[tuple[int, int], str] = {}
    seen_names: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        normalized = relative.casefold()
        previous = seen_names.get(normalized)
        if previous is not None and previous != relative:
            raise ReadOnlyControlError(
                f"case-colliding memory paths: {previous} / {relative}"
            )
        seen_names[normalized] = relative
        if path.is_symlink():
            raise ReadOnlyControlError(f"memory tree contains a symlink: {relative}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            if stat.st_nlink != 1:
                raise ReadOnlyControlError(f"memory file is hardlinked: {relative}")
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen_inodes:
                raise ReadOnlyControlError(
                    f"memory files share an inode: {seen_inodes[inode]} / {relative}"
                )
            seen_inodes[inode] = relative
        elif not path.is_dir():
            raise ReadOnlyControlError(
                f"memory tree contains a special node: {relative}"
            )


def memory_descriptor(root: Path) -> dict[str, Any]:
    assert_safe_tree(root)
    return snapshot_memory_path(root).descriptor


def copy_memory_tree(source: Path, destination: Path) -> dict[str, Any]:
    """Create one byte-identical no-clobber condition copy."""

    answer_contract.reject_symlink_components(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    source_descriptor = memory_descriptor(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    copied_descriptor = memory_descriptor(destination)
    comparable_source = {
        key: value
        for key, value in source_descriptor.items()
        if key != "path"
    }
    comparable_copy = {
        key: value
        for key, value in copied_descriptor.items()
        if key != "path"
    }
    if comparable_copy != comparable_source:
        raise ReadOnlyControlError("condition copy differs from source memory")
    return copied_descriptor


def build_turn_index(conversation: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if match is None or not isinstance(value, list):
            continue
        date = str(conversation.get(f"session_{match.group(1)}_date_time", ""))
        for turn in value:
            if not isinstance(turn, Mapping):
                raise ReadOnlyControlError("conversation turn is not an object")
            dia_id = str(turn.get("dia_id", ""))
            if re.fullmatch(r"D\d+:\d+", dia_id) is None or dia_id in index:
                raise ReadOnlyControlError(f"invalid or duplicate dia_id: {dia_id!r}")
            index[dia_id] = {
                "date": date,
                "speaker": str(turn.get("speaker", "")),
                "text": str(turn.get("text", "")),
            }
    if not index:
        raise ReadOnlyControlError("conversation has no indexed turns")
    return index


def expand_source_specs(values: Sequence[Any]) -> list[str]:
    """Expand exact, range, and compact-list LoCoMo source identifiers."""

    expanded: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw).strip().strip("[]")
        current_session: int | None = None
        for part in re.split(r"\s*,\s*", text):
            match = re.fullmatch(r"D(\d+):(\d+)(?:-(\d+))?", part)
            compact = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
            if match:
                current_session = int(match.group(1))
                start = int(match.group(2))
                end = int(match.group(3) or start)
            elif compact and current_session is not None:
                start = int(compact.group(1))
                end = int(compact.group(2) or start)
            else:
                for exact in re.findall(r"D\d+:\d+", part):
                    if exact not in seen:
                        seen.add(exact)
                        expanded.append(exact)
                continue
            if end < start or end - start > 500:
                raise ReadOnlyControlError(f"invalid source range: {part!r}")
            for turn in range(start, end + 1):
                dia_id = f"D{current_session}:{turn}"
                if dia_id not in seen:
                    seen.add(dia_id)
                    expanded.append(dia_id)
    return expanded


def source_ids_in_text(text: str) -> list[str]:
    values: list[str] = []
    values.extend(re.findall(r"D\d+:\d+(?:-\d+)?", text))
    for bracket in re.findall(r"\[([^\]]*D\d+:[^\]]*)\]", text):
        values.append(bracket)
    return expand_source_specs(values)


def resolve_sources(
    source_ids: Sequence[Any], turn_index: Mapping[str, Mapping[str, str]]
) -> tuple[str, list[str]]:
    expanded = expand_source_specs(source_ids)
    valid = [dia_id for dia_id in expanded if dia_id in turn_index]
    blocks = []
    for dia_id in valid:
        turn = turn_index[dia_id]
        blocks.append(
            f"[{dia_id}] ({turn['date']}) {turn['speaker']}: {turn['text']}"
        )
    return "\n".join(blocks), valid


def _indexed_memory_entries(
    root: Path, *, views: Sequence[str]
) -> list[dict[str, Any]]:
    """Index source identifiers in a safe frozen memory tree."""

    assert_safe_tree(root)
    entries: list[dict[str, Any]] = []
    for view in views:
        view_root = root / view
        if not view_root.exists():
            continue
        if view_root.is_symlink() or not view_root.is_dir():
            raise ReadOnlyControlError(f"invalid evidence view root: {view}")
        for path in sorted(view_root.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                raise ReadOnlyControlError("stage-evidence entry is not a regular file")
            content = path.read_text(encoding="utf-8")
            relative = path.relative_to(root).as_posix()
            payload = {
                "path": relative,
                "content_sha256": answer_contract.sha256_bytes(
                    content.encode("utf-8")
                ),
                "source_ids": source_ids_in_text(content),
            }
            payload["trace_id"] = f"memory-entry:{canonical_hash(payload)}"
            entries.append(payload)
    return entries


def build_zero_maintenance_stage_provenance(
    *,
    memory_root: Path,
    question_id: str,
    gold_source_ids: Sequence[str],
    mapping_complete: bool,
) -> dict[str, Any]:
    """Describe a frozen input tree before and after zero local maintenance.

    For formal R203, the input is the independently audited final NativeMem
    tree at the R203 boundary.  The resulting zero-operation record applies
    only to R203 read-only execution and does not claim NativeMem builder
    pre-maintenance provenance.  Callers must bind that scope explicitly.
    """

    gold = list(dict.fromkeys(str(value) for value in gold_source_ids))
    if mapping_complete and not gold:
        raise ReadOnlyControlError("complete source mapping has no gold identifiers")
    descriptor = memory_descriptor(memory_root)
    mapping_payload = {
        "question_id": question_id,
        "mapping_complete": bool(mapping_complete),
        "gold_source_ids": gold,
    }
    mapping_hash = canonical_hash(mapping_payload)
    entries = _indexed_memory_entries(memory_root, views=("topics",))
    return {
        "schema_version": STAGE_PROVENANCE_SCHEMA,
        "question_id": question_id,
        "mapping": {
            **mapping_payload,
            "record_sha256": mapping_hash,
            "trace_ids": [f"gold-mapping:{question_id}:{mapping_hash}"],
        },
        "canonical_entries_before_maintenance": {
            "memory_sha256": descriptor["sha256"],
            "entries": entries,
            "trace_ids": [
                f"canonical-memory:{descriptor['sha256']}",
                *(entry["trace_id"] for entry in entries),
            ],
        },
        "maintenance": {
            "operation_count": 0,
            "input_memory_sha256": descriptor["sha256"],
            "output_memory_sha256": descriptor["sha256"],
            "trace_ids": [f"maintenance-zero:{descriptor['sha256']}"],
        },
    }


def _validate_trace_ids(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str)
        and re.fullmatch(r"[a-z][a-z0-9_-]*:[^\s]+", item) is not None
        for item in value
    ):
        raise ReadOnlyControlError(f"{label} trace IDs are invalid")
    return list(value)


def _validate_stage_provenance(
    provenance: Mapping[str, Any],
    *,
    question_id: str,
    gold_source_ids: Sequence[str],
    source_recall_eligible: bool,
) -> None:
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
        raise ReadOnlyControlError("stage provenance identity is invalid")
    mapping_payload = {
        "question_id": question_id,
        "mapping_complete": bool(mapping.get("mapping_complete")),
        "gold_source_ids": mapping.get("gold_source_ids"),
    }
    if (
        not isinstance(mapping.get("mapping_complete"), bool)
        or mapping.get("mapping_complete") is not bool(source_recall_eligible)
        or mapping.get("gold_source_ids") != gold
        or not all(re.fullmatch(r"D\d+:\d+", value) for value in gold)
        or mapping.get("record_sha256") != canonical_hash(mapping_payload)
    ):
        raise ReadOnlyControlError("stage provenance mapping differs")
    if _validate_trace_ids(mapping.get("trace_ids"), label="mapping") != [
        f"gold-mapping:{question_id}:{mapping['record_sha256']}"
    ]:
        raise ReadOnlyControlError("mapping trace ID differs")
    entries = canonical.get("entries")
    if not isinstance(entries, list):
        raise ReadOnlyControlError("canonical stage entries are invalid")
    seen_paths: set[str] = set()
    expected_entry_traces: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReadOnlyControlError("canonical stage entry is invalid")
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
            raise ReadOnlyControlError("canonical stage entry is invalid")
        seen_paths.add(entry["path"])
        payload = {
            "path": entry["path"],
            "content_sha256": entry["content_sha256"],
            "source_ids": entry["source_ids"],
        }
        if entry.get("trace_id") != f"memory-entry:{canonical_hash(payload)}":
            raise ReadOnlyControlError("canonical stage trace ID differs")
        expected_entry_traces.append(entry["trace_id"])
    canonical_sha = canonical.get("memory_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(canonical_sha)) is None:
        raise ReadOnlyControlError("canonical memory hash is invalid")
    if _validate_trace_ids(canonical.get("trace_ids"), label="canonical") != [
        f"canonical-memory:{canonical_sha}",
        *expected_entry_traces,
    ]:
        raise ReadOnlyControlError("canonical trace inventory differs")
    _validate_trace_ids(maintenance.get("trace_ids"), label="maintenance")
    if (
        not isinstance(maintenance.get("operation_count"), int)
        or isinstance(maintenance.get("operation_count"), bool)
        or maintenance["operation_count"] < 0
        or maintenance.get("input_memory_sha256") != canonical_sha
        or re.fullmatch(
            r"[0-9a-f]{64}", str(maintenance.get("input_memory_sha256", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(maintenance.get("output_memory_sha256", ""))
        )
        is None
    ):
        raise ReadOnlyControlError("maintenance stage provenance is invalid")


def _stage_count(hits: Sequence[str], expected: Sequence[str]) -> dict[str, Any]:
    unique_hits = [value for value in expected if value in set(hits)]
    count = len(unique_hits)
    total = len(expected)
    return {
        "hit_source_ids": unique_hits,
        "hit_count": count,
        "expected_count": total,
        "ratio": count / total if total else None,
    }


def build_stage_evidence(
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
    """Derive all M4 source-stage booleans from concrete R203 evidence."""

    _validate_stage_provenance(
        provenance,
        question_id=question_id,
        gold_source_ids=gold_source_ids,
        source_recall_eligible=source_recall_eligible,
    )
    if condition not in CONDITIONS:
        raise ReadOnlyControlError("stage evidence has an unknown condition")
    gold = list(dict.fromkeys(str(value) for value in gold_source_ids))
    mapping = provenance["mapping"]
    mapping_complete = mapping["mapping_complete"]
    trace_ids: dict[str, list[str]] = {
        "mapping": list(mapping["trace_ids"]),
        "canonical_entries": [],
        "maintenance": list(provenance["maintenance"]["trace_ids"]),
        "paths": [],
        "retrieval": [],
        "source_resolution": [],
    }
    result: dict[str, Any] = {
        "schema_version": STAGE_EVIDENCE_SCHEMA,
        "question_id": question_id,
        "condition_id": condition,
        "gold_source_ids": gold,
        "gold_source_mapping_complete": mapping_complete,
        "resolver_enabled": CONDITION_SOURCE_ENABLED[condition],
        "all_gold_required": True,
        "trace_ids": trace_ids,
        "stage_counts": {},
    }
    if not mapping_complete:
        result["evidence_status"] = "excluded_incomplete_mapping"
        return result
    if not gold:
        raise ReadOnlyControlError("complete stage evidence has no gold sources")
    missing_turns = [value for value in gold if value not in turn_index]
    if missing_turns:
        raise ReadOnlyControlError(
            f"mapped gold sources are absent from the turn index: {missing_turns}"
        )

    canonical_entries = provenance["canonical_entries_before_maintenance"][
        "entries"
    ]
    canonical_hits: list[str] = []
    for entry in canonical_entries:
        entry_hits = [value for value in gold if value in entry["source_ids"]]
        if entry_hits:
            canonical_hits.extend(entry_hits)
            trace_ids["canonical_entries"].append(entry["trace_id"])

    final_canonical = _indexed_memory_entries(memory_root, views=("topics",))
    survived_hits = [
        value
        for value in gold
        if any(value in entry["source_ids"] for entry in final_canonical)
    ]
    accessible_entries = _indexed_memory_entries(
        memory_root, views=CONDITION_VIEWS[condition]
    )
    path_hits: list[str] = []
    for entry in accessible_entries:
        entry_hits = [value for value in gold if value in entry["source_ids"]]
        if entry_hits:
            path_hits.extend(entry_hits)
            trace_ids["paths"].append(entry["trace_id"])

    retrieval_hits: list[str] = []
    resolution_hits: list[str] = []
    for record in trace_records:
        if record.get("record_type") != "delivery":
            continue
        delivered = record.get("delivered")
        if not isinstance(delivered, Mapping):
            continue
        text = str(delivered.get("text", ""))
        delivered_ids = source_ids_in_text(text)
        event_trace = (
            f"delivery:{record.get('event_id')}:{delivered.get('sha256')}"
        )
        if record.get("kind") == "source_resolution":
            for value in gold:
                turn = turn_index.get(value)
                if turn is None:
                    raise ReadOnlyControlError(
                        f"gold source is absent from turn index: {value}"
                    )
                expected_block = (
                    f"[{value}] ({turn['date']}) {turn['speaker']}: {turn['text']}"
                )
                if value in delivered_ids and expected_block in text:
                    resolution_hits.append(value)
                    if event_trace not in trace_ids["source_resolution"]:
                        trace_ids["source_resolution"].append(event_trace)
        else:
            for value in gold:
                if value in delivered_ids:
                    retrieval_hits.append(value)
                    if event_trace not in trace_ids["retrieval"]:
                        trace_ids["retrieval"].append(event_trace)

    counts = {
        "gold_source_in_canonical_entries": _stage_count(canonical_hits, gold),
        "gold_source_survived_maintenance": _stage_count(survived_hits, gold),
        "gold_source_path_valid": _stage_count(path_hits, gold),
        "retrieval_reached_gold_source": _stage_count(retrieval_hits, gold),
        "source_resolution_returned_gold_content": _stage_count(
            resolution_hits, gold
        ),
    }
    maintenance = provenance["maintenance"]
    if (
        maintenance["output_memory_sha256"] != memory_before.get("sha256")
        or memory_before.get("sha256") != memory_after.get("sha256")
    ):
        raise ReadOnlyControlError("maintenance provenance does not bind live memory")
    result.update(
        {
            key: value["hit_count"] == value["expected_count"]
            for key, value in counts.items()
        }
    )
    result["stage_counts"] = counts
    result["evidence_status"] = "complete"
    return result


def _tool_definitions(source_enabled: bool) -> list[dict[str, Any]]:
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
                "description": "Literal case-insensitive search over accessible files.",
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
                    "description": "Resolve LoCoMo Dn:m identifiers to original turns.",
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


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    raw_text: str
    source_ids: tuple[str, ...]
    path: str | None
    latency_s: float


class ReadOnlyMemoryStore:
    def __init__(
        self,
        *,
        root: Path,
        condition: str,
        turn_index: Mapping[str, Mapping[str, str]],
    ) -> None:
        if condition not in CONDITIONS:
            raise ReadOnlyControlError(f"unknown condition: {condition}")
        answer_contract.reject_symlink_components(root)
        if root.is_symlink():
            raise ReadOnlyControlError("memory root must not be a symlink")
        self.root = root.resolve()
        self.condition = condition
        self.views = CONDITION_VIEWS[condition]
        self.source_enabled = CONDITION_SOURCE_ENABLED[condition]
        self.turn_index = turn_index
        self.before = memory_descriptor(self.root)
        self.access_log: list[dict[str, Any]] = []

    def _files(self) -> list[Path]:
        files: list[Path] = []
        for view in self.views:
            view_root = self.root / view
            if not view_root.exists():
                continue
            if view_root.is_symlink() or not view_root.is_dir():
                raise ReadOnlyControlError(f"invalid view root: {view}")
            files.extend(
                sorted(
                    path
                    for path in view_root.rglob("*.md")
                    if path.is_file() and not path.is_symlink()
                )
            )
        return sorted(set(files), key=lambda path: path.relative_to(self.root).as_posix())

    def inventory_text(self) -> str:
        lines = [f"accessible_views={','.join(self.views)}"]
        lines.extend(path.relative_to(self.root).as_posix() for path in self._files())
        return "\n".join(lines)

    def _safe_file(self, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ReadOnlyToolInputError("read path must be a non-empty string")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReadOnlyToolInputError("read path escapes the memory root")
        if not any(relative.parts and relative.parts[0] == view for view in self.views):
            raise ReadOnlyToolInputError("read path is outside the condition views")
        candidate = self.root / relative
        answer_contract.reject_symlink_components(candidate)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ReadOnlyToolInputError("read path escapes the memory root") from exc
        if resolved.is_symlink() or not resolved.is_file() or resolved.suffix != ".md":
            raise ReadOnlyToolInputError(
                "read path is not an accessible markdown file"
            )
        return resolved

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolObservation:
        started = time.monotonic()
        path_value: str | None = None
        source_ids: list[str] = []
        if tool_name == "list_memory_files":
            prefix = str(arguments.get("prefix", "")).strip()
            paths = [path.relative_to(self.root).as_posix() for path in self._files()]
            if prefix:
                paths = [path for path in paths if path.startswith(prefix)]
            raw = "\n".join(paths)
        elif tool_name == "read_memory_file":
            path = self._safe_file(arguments.get("path"))
            path_value = path.relative_to(self.root).as_posix()
            content = path.read_text(encoding="utf-8")
            source_ids = source_ids_in_text(content)
            raw = f"path={path_value}\n{content}"
        elif tool_name == "search_memory":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ReadOnlyControlError("search query must be non-empty")
            limit = arguments.get("max_results", 20)
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
                raise ReadOnlyControlError("search max_results must be in 1..100")
            matches: list[str] = []
            lowered = query.casefold()
            for path in self._files():
                relative = path.relative_to(self.root).as_posix()
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if lowered in line.casefold():
                        matches.append(f"{relative}:{line_number}:{line}")
                        if len(matches) == limit:
                            break
                if len(matches) == limit:
                    break
            raw = "\n".join(matches)
            source_ids = source_ids_in_text(raw)
        elif tool_name == "resolve_sources":
            if not self.source_enabled:
                raise ReadOnlyControlError("source resolution is disabled")
            values = arguments.get("source_ids", [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ReadOnlyControlError("source_ids must be a list")
            raw, source_ids = resolve_sources(values, self.turn_index)
        else:
            raise ReadOnlyControlError(f"unknown read-only tool: {tool_name}")
        observation = ToolObservation(
            tool_name=tool_name,
            raw_text=raw,
            source_ids=tuple(source_ids),
            path=path_value,
            latency_s=round(time.monotonic() - started, 6),
        )
        self.access_log.append(
            {
                "tool_name": observation.tool_name,
                "path": observation.path,
                "source_ids": list(observation.source_ids),
                "raw_sha256": answer_contract.sha256_bytes(raw.encode("utf-8")),
                "latency_s": observation.latency_s,
            }
        )
        return observation

    def execute_for_model(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> ToolObservation:
        started = time.monotonic()
        try:
            return self.execute(tool_name, arguments)
        except ReadOnlyToolInputError as exc:
            raw = json.dumps(
                {
                    "error": {
                        "code": "inaccessible_memory_path",
                        "message": str(exc),
                    },
                    "ok": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            observation = ToolObservation(
                tool_name=tool_name,
                raw_text=raw,
                source_ids=(),
                path=None,
                latency_s=round(time.monotonic() - started, 6),
            )
            self.access_log.append(
                {
                    "tool_name": observation.tool_name,
                    "path": observation.path,
                    "source_ids": [],
                    "raw_sha256": answer_contract.sha256_bytes(
                        raw.encode("utf-8")
                    ),
                    "latency_s": observation.latency_s,
                }
            )
            return observation

    def assert_unchanged(self) -> dict[str, Any]:
        after = memory_descriptor(self.root)
        if after["sha256"] != self.before["sha256"] or after != {
            **self.before,
            "path": after["path"],
        }:
            raise ReadOnlyControlError("read-only retrieval changed the memory tree")
        return after


@dataclass(frozen=True)
class RetrievalOutcome:
    deliveries: tuple[DeliveryResult, ...]
    access_log: tuple[dict[str, Any], ...]
    model_calls: int
    latency_s: float


def _message_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _tool_call_parts(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    call_id = str(_message_value(tool_call, "id", ""))
    function = _message_value(tool_call, "function")
    name = str(_message_value(function, "name", ""))
    raw_arguments = _message_value(function, "arguments", "{}")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError as exc:
        raise ReadOnlyControlError("retrieval tool arguments are invalid JSON") from exc
    if not call_id or not name or not isinstance(arguments, dict):
        raise ReadOnlyControlError("retrieval tool call is incomplete")
    return call_id, name, arguments


def run_retrieval(
    *,
    question_id: str,
    question: str,
    condition: str,
    store: ReadOnlyMemoryStore,
    completion_resource: Any,
    observer: DurableModelObserver,
    gate: VisibleTokenBudgetGate,
    max_rounds: int = 12,
) -> RetrievalOutcome:
    """Run one read-only retrieval with gate-filtered observations only."""

    if not 1 <= max_rounds <= 24:
        raise ReadOnlyControlError("retrieval max_rounds must be in 1..24")
    started = time.monotonic()
    deliveries: list[DeliveryResult] = []
    inventory_raw = f"[inventory]\n{store.inventory_text()}"
    inventory = gate.deliver_tool_result(
        event_id="observation-0000",
        raw_text=inventory_raw,
        tool_name="memory_inventory",
        tool_call_id=None,
        metadata={"condition": condition, "ordinal": 0},
    )
    deliveries.append(inventory)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": RETRIEVAL_PROMPT.format(
                condition=condition,
                question=question,
                inventory=inventory.delivered_text or "",
            ),
        }
    ]
    tools = _tool_definitions(store.source_enabled)
    calls = 0
    observer.install(completion_resource)
    try:
        with observer.operation(f"retrieval:{question_id}:{condition}"):
            for _round in range(max_rounds):
                response = completion_resource.create(
                    model=EXPECTED_MODEL,
                    messages=messages,
                    tools=tools,
                    max_tokens=1200,
                    temperature=0.0,
                    extra_headers={"X-Controlled-Question-ID": question_id},
                )
                calls += 1
                choices = _message_value(response, "choices", [])
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ReadOnlyControlError("retrieval response must have one choice")
                message = _message_value(choices[0], "message")
                tool_calls = _message_value(message, "tool_calls", None)
                content = _message_value(message, "content", None)
                if not tool_calls:
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                for tool_call in tool_calls:
                    call_id, name, arguments = _tool_call_parts(tool_call)
                    observation = store.execute_for_model(name, arguments)
                    ordinal = len(deliveries)
                    rendered = (
                        f"\n[{name}:{call_id}]\n{observation.raw_text}"
                    )
                    if name == "resolve_sources":
                        delivery = gate.deliver_source_resolution(
                            event_id=f"observation-{ordinal:04d}",
                            raw_text=rendered,
                            source_ids=list(observation.source_ids),
                            metadata={
                                "condition": condition,
                                "ordinal": ordinal,
                                "tool_call_id": call_id,
                            },
                        )
                    else:
                        delivery = gate.deliver_tool_result(
                            event_id=f"observation-{ordinal:04d}",
                            raw_text=rendered,
                            tool_name=name,
                            tool_call_id=call_id,
                            metadata={
                                "condition": condition,
                                "ordinal": ordinal,
                                "path": observation.path,
                                "observed_source_ids": list(observation.source_ids),
                            },
                        )
                    deliveries.append(delivery)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": delivery.delivered_text or "",
                        }
                    )
                if gate.exhausted:
                    break
    finally:
        observer.restore()
    store.assert_unchanged()
    return RetrievalOutcome(
        deliveries=tuple(deliveries),
        access_log=tuple(store.access_log),
        model_calls=calls,
        latency_s=round(time.monotonic() - started, 6),
    )


def diagnostic_metrics(
    *,
    deliveries: Sequence[DeliveryResult],
    access_log: Sequence[Mapping[str, Any]],
    gold_source_ids: Sequence[str],
    source_recall_eligible: bool,
) -> dict[str, Any]:
    delivered_sources: list[str] = []
    for delivery in deliveries:
        if delivery.delivered_text:
            delivered_sources.extend(source_ids_in_text(delivery.delivered_text))
    retrieved = list(dict.fromkeys(delivered_sources))
    gold = list(dict.fromkeys(str(value) for value in gold_source_ids))
    hits = [value for value in gold if value in set(retrieved)]
    first_relevant_file = None
    for access in access_log:
        if access.get("path") and set(access.get("source_ids", [])) & set(gold):
            first_relevant_file = access["path"]
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
        "first_relevant_file": first_relevant_file,
        "read_calls": sum(access.get("path") is not None for access in access_log),
        "tool_calls": len(access_log),
        "tool_latency_s": round(
            sum(float(access.get("latency_s", 0.0)) for access in access_log), 6
        ),
    }


def execute_question(
    *,
    artifact_dir: Path,
    run_id: str,
    method: str,
    condition: str,
    memory_root: Path,
    turn_index: Mapping[str, Mapping[str, str]],
    question_id: str,
    question: str,
    gold_source_ids: Sequence[str],
    source_recall_eligible: bool,
    completion_resource: Any,
    answer_client: Any,
    tokenizer: TokenCounter,
    formal: bool,
    proxy_log: Path | None,
    budget_tokens: int = EXPECTED_BUDGET,
    max_rounds: int = 12,
    model_context_limit_tokens: int = 100_000,
    answer_completion_reservation_tokens: int = 512,
    stage_provenance: Mapping[str, Any] | None = None,
    answer_prompt_builder: Callable[..., str] = (
        answer_contract.assemble_answer_prompt
    ),
    answer_prompt_kind: str = DEFAULT_ANSWER_PROMPT_KIND,
    answer_prompt_template_sha256: str = DEFAULT_ANSWER_PROMPT_SHA256,
    attempt_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one gated retrieval and fixed-answer call in a new artifact dir."""

    if budget_tokens != EXPECTED_BUDGET and formal:
        raise ReadOnlyControlError("formal R116/R203 budget must be exactly 20000")
    if method == "r203_readonly_views" and stage_provenance is None:
        raise ReadOnlyControlError("R203 requires complete stage-evidence provenance")
    if (
        not answer_prompt_kind
        or re.fullmatch(r"[0-9a-f]{64}", answer_prompt_template_sha256) is None
    ):
        raise ReadOnlyControlError("answer prompt identity is invalid")
    if stage_provenance is not None:
        _validate_stage_provenance(
            stage_provenance,
            question_id=question_id,
            gold_source_ids=gold_source_ids,
            source_recall_eligible=source_recall_eligible,
        )
    answer_contract.reject_symlink_components(artifact_dir)
    if artifact_dir.exists() or artifact_dir.is_symlink():
        raise FileExistsError(artifact_dir)
    artifact_dir.mkdir(parents=True)
    attempt_manifest_path = artifact_dir / "attempt_manifest.json"
    if attempt_manifest is not None:
        if attempt_manifest.get("execution_run_id") != run_id:
            raise ReadOnlyControlError("attempt manifest run identity differs")
        answer_contract.atomic_json_no_clobber(
            attempt_manifest_path, dict(attempt_manifest)
        )
    before = snapshot_memory_path(memory_root)
    gate_trace = artifact_dir / "visible_tokens.jsonl"
    gate_manifest_path = artifact_dir / "visible_tokens.manifest.json"
    retrieval_ledger_path = artifact_dir / "retrieval_model_ledger.jsonl"
    answer_ledger_path = artifact_dir / "answer_ledger.jsonl"
    retrieval_ledger = HashChainLedger(retrieval_ledger_path, run_id=run_id)
    retrieval_operation = f"retrieval:{question_id}:{condition}"
    retrieval_ledger.append(
        "operation_started",
        operation_id=retrieval_operation,
        question_id=question_id,
        condition=condition,
        memory_before_sha256=before.sha256,
    )
    gate = VisibleTokenBudgetGate(
        trace_path=gate_trace,
        manifest_path=gate_manifest_path,
        run_id=f"{run_id}:{question_id}:{condition}",
        configured_budget_tokens=budget_tokens,
        tokenizer=tokenizer,
        memory_before=before,
        overflow_policy="truncate",
        metadata={
            "method": method,
            "condition": condition,
            "question_id": question_id,
            "prompt_boundary": "DeliveryResult.delivered_text_only",
        },
    )
    store = ReadOnlyMemoryStore(
        root=memory_root,
        condition=condition,
        turn_index=turn_index,
    )
    observer = create_observer(
        ledger=retrieval_ledger,
        artifact_root=artifact_dir,
        tokenizer=tokenizer,
        proxy_log=proxy_log,
        formal=formal,
    )
    committed = False
    try:
        retrieval = run_retrieval(
            question_id=question_id,
            question=question,
            condition=condition,
            store=store,
            completion_resource=completion_resource,
            observer=observer,
            gate=gate,
            max_rounds=max_rounds,
        )
        after_retrieval = snapshot_memory_path(memory_root)
        if after_retrieval.sha256 != before.sha256:
            raise ReadOnlyControlError("retrieval changed the memory tree")
        retrieval_ledger.append(
            "operation_committed",
            operation_id=retrieval_operation,
            question_id=question_id,
            condition=condition,
            model_calls=retrieval.model_calls,
            access_log=list(retrieval.access_log),
            retrieval_latency_s=retrieval.latency_s,
            memory_after_sha256=after_retrieval.sha256,
        )
        committed = True
        model_state = completed_model_ledger(retrieval_ledger_path)
        prompt = answer_prompt_builder(
            question=question,
            deliveries=retrieval.deliveries,
        )
        prompt_sha256 = answer_contract.sha256_bytes(prompt.encode("utf-8"))
        prompt_tokens = tokenizer.count(prompt)
        if (
            prompt_tokens + answer_completion_reservation_tokens
            > model_context_limit_tokens
        ):
            raise ReadOnlyControlError("answer prompt exceeds the declared context limit")
        answer_call_id = f"{run_id}:{question_id}:{condition}:answer"
        with answer_contract.DurableLedger(
            answer_ledger_path, run_id=answer_call_id
        ) as answer_ledger:
            answer_ledger.append(
                "answer_started",
                {
                    "question_id": question_id,
                    "condition": condition,
                    "prompt_sha256": prompt_sha256,
                },
            )
            answer_call = answer_client.complete(
                prompt=prompt,
                question_id=question_id,
                logical_call_id=answer_call_id,
                ledger=answer_ledger,
            )
            answer_ledger.append(
                "answer_completed",
                {
                    "response_id": answer_call.response_id,
                    "response_model": answer_call.response_model,
                    "client_http_attempts": answer_call.client_http_attempts,
                    "upstream_http_attempts": answer_call.upstream_http_attempts,
                },
            )
        if answer_call.response_model != EXPECTED_MODEL:
            raise ReadOnlyControlError("fixed answerer returned a different model")
        answer_proxy_evidence = proxy_evidence(
            proxy_log,
            logical_call_id=answer_call_id,
            formal=formal,
        )
        provider_total = answer_call.usage.get("total_tokens")
        if (
            not isinstance(provider_total, int)
            or isinstance(provider_total, bool)
            or provider_total < 0
            or provider_total > model_context_limit_tokens
        ):
            raise ReadOnlyControlError("answer usage exceeds the declared context limit")
        final_memory = snapshot_memory_path(memory_root)
        if final_memory.sha256 != before.sha256:
            raise ReadOnlyControlError("answering changed the memory tree")
        gate_manifest = gate.finalize(
            memory_after=final_memory,
            actual_model_usage={
                "retrieval": model_state,
                "answer": {
                    "requested_model": EXPECTED_MODEL,
                    "response_model": answer_call.response_model,
                    "response_id": answer_call.response_id,
                    "usage": answer_call.usage,
                    "logical_calls": answer_call.logical_calls,
                    "client_http_attempts": answer_call.client_http_attempts,
                    "upstream_http_attempts": answer_call.upstream_http_attempts,
                    "exclusive_proxy_event_ids": list(answer_call.proxy_event_ids),
                    "unsupported_parameters": list(
                        answer_call.unsupported_parameters
                    ),
                    "proxy_evidence": answer_proxy_evidence,
                },
            },
            metadata={
                "question_id": question_id,
                "condition": condition,
                "prompt_sha256": prompt_sha256,
            },
        )
        stage_evidence: dict[str, Any] | None = None
        stage_evidence_path = artifact_dir / "stage_evidence.json"
        if stage_provenance is not None:
            trace_records = [
                json.loads(line)
                for line in gate_trace.read_text(encoding="utf-8").splitlines()
            ]
            stage_evidence = build_stage_evidence(
                provenance=stage_provenance,
                memory_root=memory_root,
                memory_before=before.descriptor,
                memory_after=final_memory.descriptor,
                condition=condition,
                question_id=question_id,
                gold_source_ids=gold_source_ids,
                source_recall_eligible=source_recall_eligible,
                turn_index=turn_index,
                trace_records=trace_records,
            )
            answer_contract.atomic_json_no_clobber(
                stage_evidence_path, stage_evidence
            )
        diagnostics = diagnostic_metrics(
            deliveries=retrieval.deliveries,
            access_log=retrieval.access_log,
            gold_source_ids=gold_source_ids,
            source_recall_eligible=source_recall_eligible,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "run_id": run_id,
            "method": method,
            "condition": condition,
            "question_id": question_id,
            "question_sha256": answer_contract.sha256_bytes(
                question.encode("utf-8")
            ),
            "budget": {
                "policy": "hard_cap",
                "configured_tokens": budget_tokens,
                "visible_tokens": gate_manifest["summary"][
                    "cumulative_visible_tokens"
                ],
                "source_resolution_tokens": gate_manifest["summary"][
                    "cumulative_source_resolution_tokens"
                ],
                "tokenizer": tokenizer.identity,
            },
            "prompt": {
                "kind": answer_prompt_kind,
                "template_sha256": answer_prompt_template_sha256,
                "sha256": prompt_sha256,
                "local_tokens": prompt_tokens,
                "model_context_limit_tokens": model_context_limit_tokens,
                "answer_completion_reservation_tokens": (
                    answer_completion_reservation_tokens
                ),
            },
            "retrieval": {
                "model": EXPECTED_MODEL,
                "model_calls": retrieval.model_calls,
                "latency_s": retrieval.latency_s,
                "access_log": list(retrieval.access_log),
                "ledger_state": model_state,
            },
            "answer": {
                "text": answer_contract.extract_answer(answer_call.raw_output),
                "raw_output": answer_call.raw_output,
                "raw_output_sha256": answer_contract.sha256_bytes(
                    answer_call.raw_output.encode("utf-8")
                ),
                "requested_model": EXPECTED_MODEL,
                "response_model": answer_call.response_model,
                "response_id": answer_call.response_id,
                "usage": answer_call.usage,
                "logical_calls": answer_call.logical_calls,
                "client_http_attempts": answer_call.client_http_attempts,
                "upstream_http_attempts": answer_call.upstream_http_attempts,
                "exclusive_proxy_event_ids": list(answer_call.proxy_event_ids),
                "request_sha256s": list(answer_call.request_sha256s),
                "response_sha256s": list(answer_call.response_sha256s),
                "unsupported_parameters": list(answer_call.unsupported_parameters),
                "proxy_evidence": answer_proxy_evidence,
            },
            "diagnostics": diagnostics,
            "stage_evidence": stage_evidence,
            "memory": {
                "before": before.descriptor,
                "after": final_memory.descriptor,
                "unchanged": before.sha256 == final_memory.sha256,
            },
            "artifacts": {
                "visible_token_trace": gate_trace.name,
                "visible_token_trace_sha256": sha256_file(gate_trace),
                "visible_token_manifest": gate_manifest_path.name,
                "visible_token_manifest_sha256": sha256_file(gate_manifest_path),
                "retrieval_model_ledger": retrieval_ledger_path.name,
                "retrieval_model_ledger_sha256": sha256_file(
                    retrieval_ledger_path
                ),
                "answer_ledger": answer_ledger_path.name,
                "answer_ledger_sha256": sha256_file(answer_ledger_path),
            },
        }
        if stage_evidence is not None:
            result["artifacts"].update(
                {
                    "stage_evidence": stage_evidence_path.name,
                    "stage_evidence_sha256": sha256_file(stage_evidence_path),
                }
            )
        if attempt_manifest is not None:
            result["artifacts"].update(
                {
                    "attempt_manifest": attempt_manifest_path.name,
                    "attempt_manifest_sha256": sha256_file(
                        attempt_manifest_path
                    ),
                }
            )
        result_path = artifact_dir / "result.json"
        answer_contract.atomic_json_no_clobber(result_path, result)
        return result
    except Exception:
        if not committed:
            retrieval_ledger.append(
                "operation_failed",
                operation_id=retrieval_operation,
                question_id=question_id,
                condition=condition,
            )
        gate.close_incomplete()
        raise


class ScriptedCompletionResource:
    """OpenAI-like deterministic completion resource for no-network sanity."""

    def __init__(self, scripts: Sequence[Sequence[Mapping[str, Any]]]):
        self.scripts = [list(item) for item in scripts]
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        if self.calls >= len(self.scripts):
            raise ReadOnlyControlError("scripted completion sequence exhausted")
        tool_specs = self.scripts[self.calls]
        self.calls += 1
        tool_calls = []
        for index, spec in enumerate(tool_specs):
            tool_calls.append(
                SimpleNamespace(
                    id=f"fake-call-{self.calls:02d}-{index:02d}",
                    function=SimpleNamespace(
                        name=spec["name"],
                        arguments=json.dumps(spec.get("arguments", {})),
                    ),
                )
            )
        message = SimpleNamespace(
            content="retrieval complete" if not tool_calls else None,
            tool_calls=tool_calls or None,
        )
        response_id = f"fake-retrieval-{self.calls:04d}"
        usage = {
            "prompt_tokens": len(json.dumps(kwargs, default=str)),
            "completion_tokens": 1,
            "total_tokens": len(json.dumps(kwargs, default=str)) + 1,
        }
        payload = {
            "id": response_id,
            "model": EXPECTED_MODEL,
            "choices": [
                {
                    "message": {
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ]
                        or None,
                    }
                }
            ],
            "usage": usage,
        }
        return SimpleNamespace(
            id=response_id,
            model=EXPECTED_MODEL,
            usage=usage,
            choices=[SimpleNamespace(message=message)],
            model_dump=lambda mode="json": payload,
        )


def create_observer(
    *,
    ledger: HashChainLedger,
    artifact_root: Path,
    tokenizer: TokenCounter,
    proxy_log: Path | None,
    formal: bool,
) -> DurableModelObserver:
    return DurableModelObserver(
        ledger=ledger,
        artifact_root=artifact_root,
        token_counter=tokenizer,
        expected_model=EXPECTED_MODEL,
        proxy_log=proxy_log,
        formal=formal,
    )


def completed_model_ledger(path: Path) -> dict[str, Any]:
    from src.evaluation.durable_model_ledger import read_ledger

    return ledger_state(read_ledger(path))


__all__ = [
    "CONDITIONS",
    "CONDITION_SOURCE_ENABLED",
    "CONDITION_VIEWS",
    "EXPECTED_BUDGET",
    "EXPECTED_MODEL",
    "ReadOnlyControlError",
    "ReadOnlyMemoryStore",
    "RetrievalOutcome",
    "SCHEMA_VERSION",
    "STAGE_EVIDENCE_SCHEMA",
    "STAGE_PROVENANCE_SCHEMA",
    "ScriptedCompletionResource",
    "ToolObservation",
    "build_turn_index",
    "build_stage_evidence",
    "build_zero_maintenance_stage_provenance",
    "canonical_hash",
    "completed_model_ledger",
    "copy_memory_tree",
    "create_observer",
    "diagnostic_metrics",
    "execute_question",
    "expand_source_specs",
    "memory_descriptor",
    "resolve_sources",
    "run_retrieval",
    "sha256_file",
    "source_ids_in_text",
]
