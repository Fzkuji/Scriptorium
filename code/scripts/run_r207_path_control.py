#!/usr/bin/env python3
"""Capture a frozen v8.8 entry bank and materialize R207 path controls.

The capture hook is installed around the existing adapter at runtime.  It
observes every ``distill_events`` return and verifies that the same event list
is handed to ``write_events`` before allowing the ordinary build to continue.
No frozen source file is modified.

Formal model traffic requires ``--allow-model-requests``.  The
``--synthetic-sanity`` mode exercises capture and materialization without any
model request.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from src import openai_gpt55_flex_gateway_evidence as flex_evidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
EVIDENCE_QUESTIONS = (
    ROOT / "results" / "paper-experiments-20260714" / "evidence-mapping"
    / "v1" / "evidence_mapping.v1.questions.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results" / "r207-path-control-gpt55-20260714"
EXPECTED_DATA_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
EXPECTED_EVIDENCE_SHA256 = (
    "3291df579bc20d0678f99b3361ac6d287beee583e4d3dff93b996f4522e56318"
)

BANK_SCHEMA = "nativemem.canonical-entry-bank.v1"
PLACEMENT_SCHEMA = "nativemem.path-placement.v1"
ORIGINAL_PLACEMENT_SCHEMA = "nativemem.original-path-placement.v1"
CAPTURE_SCHEMA = "nativemem.distill-write-capture.v1"
SAMPLE_SCHEMA = "nativemem.r207-sample.v1"
RUN_SCHEMA = "nativemem.r207-run.v1"
TRACE_SCHEMA = "nativemem.r207-question-trace.v1"

CONDITIONS = ("model_directed", "deterministic_permutation")
FROZEN_SOURCE_HASHES = {
    "src/nativemem.py": (
        "305de2a10dae40631fd3e376be576cd37afcca37fe11aa739e13ed60efb46d82"
    ),
    "src/v8_memory.py": (
        "7847825c7819c90270950dbbba8aa52fa3d93118240f25d6300c121e90b22c8b"
    ),
    "src/adapters/run_nativemem.py": (
        "b5e9388660f6665d47149763437bd9044f7945eb4fdc460a9da1d68a88ba7031"
    ),
    "src/chatgpt_proxy.py": (
        "a0fba59861f99e5bb9134e4359297b26b9f45240be57906cc2fdfea64aef0321"
    ),
}
OBSERVED_BENCHMARK_RUNNERS = (
    "scripts/run_v88_gpt55_locomo.py",
    "scripts/run_v88_gpt55_longmemeval.py",
    "scripts/run_v88_gpt55_beam.py",
)
CORE_SOURCES = (
    *FROZEN_SOURCE_HASHES,
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "scripts/gpt55_run_proxy.py",
    "scripts/controlled_gpt55_run_proxy.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "scripts/run_r207_path_control.py",
    "scripts/audit_r207_path_control.py",
)

FINAL_ENV = {
    "NATIVEMEM_PROMPT": "v8",
    "NATIVEMEM_STORE_MODE": "oneshot",
    "NATIVEMEM_V8_SINGLE": "1",
    "NATIVEMEM_CHUNK_TURNS": "6",
    "NATIVEMEM_V8_SEGMENT": "fixed",
    "NATIVEMEM_V8_TIDY": "on",
    "NATIVEMEM_V8_SECTIONS": "on",
    "NATIVEMEM_V8_ARTICLE": "off",
    "NATIVEMEM_V8_TIDY_COMBINED": "off",
    "NATIVEMEM_V8_VERIFY": "on",
    "NATIVEMEM_V8_MERGE_LINES": "on",
    "NATIVEMEM_V8_MAX_TOPICS": "30",
    "NATIVEMEM_V8_MAP": "dir",
    "NATIVEMEM_V8_MAP_INLINE": "8",
    "NATIVEMEM_V8_MAX_ROUNDS": "12",
    "NATIVEMEM_V8_MAX_TOKENS": "1200",
    "NATIVEMEM_V8_READ_CONTEXT": "1",
    "NATIVEMEM_V8_REWRITE_MIN": "8",
    "NATIVEMEM_V8_MAX_DEPTH": "4",
    "NATIVEMEM_TOPK": "20",
    "NATIVEMEM_TRUST_PROXY": "0",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def value_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = b"".join(canonical_bytes(record) + b"\n" for record in records)
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


def parse_samples(spec: str) -> list[int]:
    values: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(x) for x in part.split("-", 1))
            values.update(range(min(left, right), max(left, right) + 1))
        else:
            values.add(int(part))
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("samples must be non-negative")
    return sorted(values)


def source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in CORE_SOURCES:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required source is missing or a symlink: {relative}")
        hashes[relative] = file_sha256(path)
        expected = FROZEN_SOURCE_HASHES.get(relative)
        if expected is not None and hashes[relative] != expected:
            raise RuntimeError(
                f"frozen source hash mismatch for {relative}: "
                f"expected {expected}, got {hashes[relative]}"
            )
    return hashes


def environment_observations() -> dict[str, str]:
    return {
        relative: file_sha256(ROOT / relative)
        for relative in OBSERVED_BENCHMARK_RUNNERS
        if (ROOT / relative).is_file() and not (ROOT / relative).is_symlink()
    }


def natural_dia_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"D(\d+):(\d+)", value)
    if match:
        return int(match.group(1)), int(match.group(2)), ""
    return sys.maxsize, sys.maxsize, value


def sorted_dia_ids(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)},
                  key=natural_dia_key)


def session_from_dia_ids(values: Iterable[Any]) -> int:
    sessions: set[int] = set()
    for value in values:
        match = re.fullmatch(r"D(\d+):\d+", str(value))
        if not match:
            raise RuntimeError(f"invalid source identifier: {value!r}")
        sessions.add(int(match.group(1)))
    if len(sessions) != 1:
        raise RuntimeError(
            f"one extraction chunk must belong to one session, got {sessions}"
        )
    return next(iter(sessions))


def entry_identity_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": entry["sample"],
        "session": entry["session"],
        "chunk": entry["chunk"],
        "ordinal": entry["ordinal"],
        "when": entry["when"],
        "summary": entry["summary"],
        "summary_inline": entry["summary_inline"],
        "dia_ids": entry["dia_ids"],
    }


def stable_entry_id(entry: dict[str, Any]) -> str:
    suffix = value_sha256(entry_identity_payload(entry))[:16]
    return (
        f"s{entry['sample']:02d}-d{entry['session']:03d}-"
        f"c{entry['chunk']:03d}-e{entry['ordinal']:03d}-{suffix}"
    )


def collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def assert_safe_tree(root: Path) -> None:
    seen_paths: dict[str, str] = {}
    seen_inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        key = collision_key(relative)
        previous = seen_paths.get(key)
        if previous is not None and previous != relative:
            raise RuntimeError(f"Unicode/case path collision: {previous} / {relative}")
        seen_paths[key] = relative
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden: {relative}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            if stat.st_nlink != 1:
                raise RuntimeError(f"hardlink is forbidden: {relative}")
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen_inodes:
                raise RuntimeError(
                    f"shared inode: {seen_inodes[inode]} / {relative}"
                )
            seen_inodes[inode] = relative
        elif not path.is_dir():
            raise RuntimeError(f"special filesystem node is forbidden: {relative}")


def tree_descriptor(root: Path) -> dict[str, Any]:
    assert_safe_tree(root)
    files = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append({
            "path": relative,
            "size": path.stat(follow_symlinks=False).st_size,
            "sha256": file_sha256(path),
        })
    return {
        "file_count": len(files),
        "tree_sha256": value_sha256(files),
    }


class CaptureRecorder:
    """Convert captured v8 events into a topic-free canonical entry bank."""

    def __init__(
        self,
        sample: int,
        sanitize_topic: Callable[[str], str],
        expected_dia_ids: set[str],
    ) -> None:
        self.sample = sample
        self.sanitize_topic = sanitize_topic
        self.expected_dia_ids = expected_dia_ids
        self.entries: list[dict[str, Any]] = []
        self.original_placements: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self._chunks: defaultdict[int, int] = defaultdict(int)
        self._pending: dict[int, int] = {}
        self._returned_payloads: dict[int, str] = {}

    def record_distill(
        self,
        events: Any,
        dia_ids: Iterable[Any],
        obs_date: Any,
    ) -> int:
        chunk_sources = sorted_dia_ids(dia_ids)
        session = session_from_dia_ids(chunk_sources)
        self._chunks[session] += 1
        chunk = self._chunks[session]
        if not isinstance(events, list):
            raise RuntimeError("distill_events must return a list")
        call_index = len(self.calls) + 1
        event_ids: list[str] = []
        for ordinal, raw_event in enumerate(events, start=1):
            if not isinstance(raw_event, dict):
                raise RuntimeError("every distilled event must be an object")
            dia_values = sorted_dia_ids(raw_event.get("dia_ids", []))
            if not dia_values:
                raise RuntimeError("captured event has no source identifier")
            if not set(dia_values).issubset(self.expected_dia_ids):
                raise RuntimeError("captured event references a source outside the sample")
            if not set(dia_values).issubset(set(chunk_sources)):
                raise RuntimeError("captured event references a source outside its chunk")
            entry: dict[str, Any] = {
                "sample": self.sample,
                "session": session,
                "chunk": chunk,
                "ordinal": ordinal,
                "when": str(raw_event.get("when", obs_date)),
                "summary": str(raw_event.get("summary", "")),
                "summary_inline": str(raw_event.get("summary_inline", "")),
                "dia_ids": dia_values,
            }
            if not entry["summary"]:
                raise RuntimeError("captured event has an empty summary")
            entry["entry_id"] = stable_entry_id(entry)
            if any(existing["entry_id"] == entry["entry_id"]
                   for existing in self.entries):
                raise RuntimeError(f"duplicate stable entry_id: {entry['entry_id']}")
            raw_topic = str(raw_event.get("topic", "misc"))
            topic_path = self.sanitize_topic(raw_topic)
            if not topic_path:
                raise RuntimeError("sanitized topic path is empty")
            self.entries.append(entry)
            self.original_placements.append({
                "entry_id": entry["entry_id"],
                "original_topic": raw_topic,
                "topic_path": topic_path,
            })
            event_ids.append(entry["entry_id"])
        payload_hash = value_sha256(copy.deepcopy(events))
        self.calls.append({
            "call_index": call_index,
            "sample": self.sample,
            "session": session,
            "chunk": chunk,
            "observation_date": str(obs_date),
            "chunk_dia_ids": chunk_sources,
            "event_count": len(events),
            "entry_ids": event_ids,
            "returned_events_sha256": payload_hash,
            "write_observed": False,
        })
        if events:
            if id(events) in self._pending:
                raise RuntimeError(
                    "distill_events reused an event-list object before its write"
                )
            self._pending[id(events)] = call_index
            self._returned_payloads[call_index] = payload_hash
        return call_index

    def observe_write(self, events: Any) -> None:
        call_index = self._pending.pop(id(events), None)
        if call_index is None:
            raise RuntimeError(
                "write_events did not receive an observed distill_events return"
            )
        actual = value_sha256(copy.deepcopy(events))
        if actual != self._returned_payloads.pop(call_index):
            raise RuntimeError("events changed between distill_events and write_events")
        record = self.calls[call_index - 1]
        if record["write_observed"]:
            raise RuntimeError("one distill_events return was written more than once")
        record["write_observed"] = True

    def assert_complete(self) -> None:
        if self._pending or self._returned_payloads:
            raise RuntimeError("at least one non-empty distill return was never written")
        for call in self.calls:
            expected = call["event_count"] > 0
            if call["write_observed"] != expected:
                raise RuntimeError("distill/write capture pairing is incomplete")


class AdapterCaptureHook(AbstractContextManager["AdapterCaptureHook"]):
    """Temporarily observe an adapter without changing the returned events."""

    def __init__(self, adapter: Any, recorder: CaptureRecorder) -> None:
        self.adapter = adapter
        self.recorder = recorder

    def __enter__(self) -> "AdapterCaptureHook":
        self._distill = self.adapter.v8_memory.distill_events
        self._distill_alias = getattr(self.adapter, "distill_events", None)
        self._write = self.adapter.write_events

        def observed_distill(
            turns: Any,
            obs_date: Any,
            dia_ids: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            events = self._distill(
                turns, obs_date, dia_ids, *args, **kwargs
            )
            self.recorder.record_distill(events, dia_ids, obs_date)
            return events

        def observed_write(memory_dir: Any, events: Any) -> Any:
            self.recorder.observe_write(events)
            return self._write(memory_dir, events)

        self.adapter.v8_memory.distill_events = observed_distill
        self.adapter.distill_events = observed_distill
        self.adapter.write_events = observed_write

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.adapter.v8_memory.distill_events = self._distill
        if self._distill_alias is not None:
            self.adapter.distill_events = self._distill_alias
        else:
            delattr(self.adapter, "distill_events")
        self.adapter.write_events = self._write
        if exc_type is None:
            self.recorder.assert_complete()
        return None


def bank_payload(sample: int, entries: list[dict[str, Any]]) -> dict[str, Any]:
    source_identity = [
        {"entry_id": entry["entry_id"], "dia_ids": entry["dia_ids"]}
        for entry in entries
    ]
    return {
        "schema": BANK_SCHEMA,
        "sample": sample,
        "entry_count": len(entries),
        "entries_sha256": value_sha256(entries),
        "source_identity_sha256": value_sha256(source_identity),
        "entries": entries,
    }


def placement_payload(
    condition: str,
    bank: dict[str, Any],
    placements: list[dict[str, str]],
    permutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topic_counts = Counter(item["topic_path"] for item in placements)
    metrics = {
        "entry_count": len(placements),
        "topic_count": len(topic_counts),
        "per_topic_counts": dict(sorted(topic_counts.items())),
        "maximum_depth": max(
            (len(item["topic_path"].split("/")) for item in placements),
            default=0,
        ),
        "path_string_total_bytes": sum(
            len(item["topic_path"].encode("utf-8")) for item in placements
        ),
    }
    payload: dict[str, Any] = {
        "schema": PLACEMENT_SCHEMA,
        "condition": condition,
        "sample": bank["sample"],
        "bank_entries_sha256": bank["entries_sha256"],
        "placements_sha256": value_sha256(placements),
        "metrics": metrics,
        "placements": placements,
    }
    if permutation is not None:
        payload["permutation"] = permutation
    return payload


def deterministic_permutation(
    model_placements: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    ordered = sorted(model_placements, key=lambda item: item["entry_id"])
    count = len(ordered)
    if count <= 1:
        offset = 0
        fixed = count
    else:
        paths = [item["topic_path"] for item in ordered]
        candidates = []
        for candidate in range(1, count):
            candidate_fixed = sum(
                paths[index] == paths[(index + candidate) % count]
                for index in range(count)
            )
            candidates.append((candidate_fixed, candidate))
        fixed, offset = min(candidates)
    paths = [item["topic_path"] for item in ordered]
    permuted = [
        {
            "entry_id": item["entry_id"],
            "topic_path": paths[(index + offset) % count] if count else "",
        }
        for index, item in enumerate(ordered)
    ]
    metadata = {
        "algorithm": "minimum-fixed-point-cyclic-offset-v1",
        "ordering": "entry_id ascending",
        "offset": offset,
        "fixed_points": fixed,
        "fixed_point_rate": (fixed / count) if count else 0.0,
        "minimum_cyclic_fixed_points": fixed,
        "candidate_offsets": max(0, count - 1),
    }
    return permuted, metadata


def assert_placement_match(
    left: dict[str, Any], right: dict[str, Any]
) -> None:
    left_metrics = left["metrics"]
    right_metrics = right["metrics"]
    for field in (
        "entry_count",
        "topic_count",
        "per_topic_counts",
        "maximum_depth",
        "path_string_total_bytes",
    ):
        if left_metrics[field] != right_metrics[field]:
            raise RuntimeError(f"path control mismatch for {field}")


def materialize_condition(
    target: Path,
    bank: dict[str, Any],
    placement: dict[str, Any],
    write_events: Callable[[str, list[dict[str, Any]]], Any],
    dedup_topic_files: Callable[[str], Any],
) -> dict[str, Any]:
    if target.exists():
        raise RuntimeError(f"refusing stale condition directory: {target}")
    target.mkdir(parents=True)
    by_id = {entry["entry_id"]: entry for entry in bank["entries"]}
    for item in placement["placements"]:
        entry = by_id[item["entry_id"]]
        event = {
            "when": entry["when"],
            "summary": entry["summary"],
            "summary_inline": entry["summary_inline"],
            "dia_ids": entry["dia_ids"],
            "topic": item["topic_path"],
        }
        write_events(str(target), [event])
    removed = int(dedup_topic_files(str(target)) or 0)
    descriptor = tree_descriptor(target)
    descriptor["exact_duplicate_lines_removed"] = removed
    return descriptor


def conversation_payload(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = sample.get("conversation")
    if isinstance(conversation, dict):
        return conversation
    return sample


def sample_expected_dia_ids(conv: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    session = 1
    while f"session_{session}" in conv:
        for turn in conv[f"session_{session}"]:
            if isinstance(turn, dict) and turn.get("dia_id"):
                values.add(str(turn["dia_id"]))
        session += 1
    return values


def load_formal_question_mappings() -> dict[int, list[dict[str, Any]]]:
    if (not EVIDENCE_QUESTIONS.is_file() or EVIDENCE_QUESTIONS.is_symlink()
            or file_sha256(EVIDENCE_QUESTIONS) != EXPECTED_EVIDENCE_SHA256):
        raise RuntimeError("frozen LoCoMo evidence mapping hash mismatch")
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    with EVIDENCE_QUESTIONS.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid evidence mapping line {line_number}"
                ) from exc
            if (record.get("benchmark") != "LoCoMo"
                    or record.get("qa_scoring_eligible") is not True):
                continue
            question_id = str(record.get("question_id", ""))
            if not question_id or question_id in seen:
                raise RuntimeError("evidence mapping question IDs are invalid")
            seen.add(question_id)
            grouped[int(record["sample_index"])].append(record)
    if len(seen) != 1540 or set(grouped) != set(range(10)):
        raise RuntimeError("formal evidence mapping inventory differs")
    for records in grouped.values():
        records.sort(key=lambda record: int(record["question_index"]))
    return dict(grouped)


def source_locations(memory_dir: Path) -> dict[str, list[str]]:
    output: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(memory_dir.rglob("*.md")):
        relative = path.relative_to(memory_dir).as_posix()
        values = set(re.findall(
            r"(?<![\w])D\d+:\d+(?!\w)", path.read_text(encoding="utf-8")
        ))
        for source_id in sorted(values):
            output[source_id].append(relative)
    return {key: value for key, value in sorted(output.items())}


def stable_trace_id(question_id: str, condition: str) -> str:
    return f"r207:{question_id}:{condition}"


def trace_stage_evidence(
    mapping: dict[str, Any],
    *,
    condition: str,
    canonical_sources: set[str],
    maintained_sources: set[str],
    condition_locations: dict[str, list[str]],
) -> dict[str, Any]:
    source_ids = [str(value) for value in mapping["normalized_source_ids"]]
    eligible = bool(mapping["source_recall_eligible"])
    present = sorted(set(source_ids) & canonical_sources)
    missing = sorted(set(source_ids) - canonical_sources)
    mapping_stage = {
        "status": "observed",
        "value": eligible,
        "artifact": "frozen_evidence_mapping",
        "question_id": mapping["question_id"],
        "source_ids": source_ids,
        "source_exclusion_reasons": mapping[
            "source_recall_exclusion_reasons"
        ],
    }
    if not eligible:
        canonical = {
            "status": "not_applicable",
            "value": None,
            "reason": "source_recall_excluded",
            "present_source_ids": [],
            "missing_source_ids": [],
        }
    else:
        canonical = {
            "status": "observed",
            "value": not missing,
            "artifact": "canonical_entry_bank.json",
            "present_source_ids": present,
            "missing_source_ids": missing,
        }
    if canonical["value"] is True:
        maintenance_missing = sorted(set(source_ids) - maintained_sources)
        maintenance = {
            "status": "observed",
            "value": not maintenance_missing,
            "artifact": "captured_final_build",
            "present_source_ids": sorted(
                set(source_ids) & maintained_sources
            ),
            "missing_source_ids": maintenance_missing,
        }
        locations = {
            source_id: condition_locations.get(source_id, [])
            for source_id in source_ids
        }
        path_stage = {
            "status": "observed",
            "value": all(locations[source_id] for source_id in source_ids),
            "artifact": f"conditions/{condition}",
            "source_paths": locations,
        }
    else:
        maintenance = {
            "status": "not_applicable",
            "value": None,
            "reason": "canonical_source_stage_not_complete",
            "present_source_ids": [],
            "missing_source_ids": [],
        }
        path_stage = {
            "status": "not_applicable",
            "value": None,
            "reason": "canonical_source_stage_not_complete",
            "source_paths": {},
        }
    unavailable = {
        "status": "not_observed",
        "value": None,
        "reason": "retrieval_results_not_attached",
    }
    stages = {
        "mapping_complete": mapping_stage,
        "canonical_entry_source_exists": canonical,
        "maintenance_survival": maintenance,
        "path_validity": path_stage,
        "retrieval_reach": dict(unavailable),
        "source_resolution": dict(unavailable),
    }
    question_id = str(mapping["question_id"])
    trace_ids = {
        "mapping": [f"gold-mapping:{question_id}:{value_sha256({
            'source_ids': source_ids,
            'mapping_complete': eligible,
        })}"],
        "canonical_entries": (
            [f"canonical-sources:{value_sha256({
                'question_id': question_id,
                'present_source_ids': canonical['present_source_ids'],
            })}"] if canonical["status"] == "observed" else []
        ),
        "maintenance": (
            [f"maintenance-sources:{value_sha256({
                'question_id': question_id,
                'present_source_ids': maintenance['present_source_ids'],
                'missing_source_ids': maintenance['missing_source_ids'],
            })}"] if maintenance["status"] == "observed" else []
        ),
        "paths": (
            [
                f"condition-path:{value_sha256({
                    'condition': condition,
                    'source_id': source_id,
                    'path': path,
                })}"
                for source_id, paths in path_stage["source_paths"].items()
                for path in paths
            ] if path_stage["status"] == "observed" else []
        ),
        "retrieval": [],
        "source_resolution": [],
    }
    stages["trace_ids"] = trace_ids
    stages["m4_fields"] = {
        "gold_source_mapping_complete": eligible,
        "gold_source_in_canonical_entries": canonical["value"],
        "gold_source_survived_maintenance": maintenance["value"],
        "gold_source_path_valid": path_stage["value"],
        "retrieval_reached_gold_source": None,
        "source_resolution_returned_gold_content": None,
    }
    stages["evidence_status"] = (
        "partial_pre_retrieval" if eligible else "excluded_incomplete_mapping"
    )
    return stages


def question_trace_records(
    *,
    sample: int,
    sample_id: str,
    mappings: list[dict[str, Any]],
    bank: dict[str, Any],
    build_dir: Path,
    conditions_dir: Path,
) -> list[dict[str, Any]]:
    canonical_sources = {
        str(source_id)
        for entry in bank["entries"]
        for source_id in entry["dia_ids"]
    }
    maintained_sources = set(source_locations(build_dir))
    locations = {
        condition: source_locations(conditions_dir / condition)
        for condition in CONDITIONS
    }
    output = []
    for mapping in mappings:
        if (int(mapping["sample_index"]) != sample
                or mapping["sample_id"] != sample_id):
            raise RuntimeError("question mapping sample identity mismatch")
        for condition in CONDITIONS:
            question_id = str(mapping["question_id"])
            output.append({
                "schema": TRACE_SCHEMA,
                "trace_id": stable_trace_id(question_id, condition),
                "sample": sample,
                "sample_id": sample_id,
                "question_index": int(mapping["question_index"]),
                "question_id": question_id,
                "condition": condition,
                "normalized_source_ids": [
                    str(value) for value in mapping["normalized_source_ids"]
                ],
                "source_recall_eligible": bool(
                    mapping["source_recall_eligible"]
                ),
                "stage_evidence": trace_stage_evidence(
                    mapping,
                    condition=condition,
                    canonical_sources=canonical_sources,
                    maintained_sources=maintained_sources,
                    condition_locations=locations[condition],
                ),
            })
    trace_ids = [record["trace_id"] for record in output]
    if len(trace_ids) != len(set(trace_ids)):
        raise RuntimeError("question trace IDs are not unique")
    return output


def model_cost_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    starts = [item for item in values if item.get("event") == "model_call_started"]
    terminals = [item for item in values
                 if item.get("event") in {"model_call_finished", "model_call_failed"}]
    finished = [item for item in terminals
                if item.get("event") == "model_call_finished"]
    return {
        "logical_model_calls": len(starts),
        "successful_model_calls": len(finished),
        "failed_model_calls": len(terminals) - len(finished),
        "client_http_attempts": sum(
            int(item.get("proxy_evidence", {}).get("client_http_attempts") or 0)
            for item in terminals
        ),
        "physical_upstream_attempts": sum(
            int(item.get("proxy_evidence", {}).get("upstream_http_attempts") or 0)
            for item in terminals
        ),
        "local_visible_tokens": sum(
            int(item.get("local_visible_tokens") or 0) for item in starts
        ),
        "provider_tokens": {
            field: sum(int(item.get("usage", {}).get(field) or 0)
                       for item in finished)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "model_latency_s": round(sum(
            float(item.get("latency_s") or 0.0) for item in terminals
        ), 6),
        "unsupported_parameters": sorted({
            str(parameter)
            for item in terminals
            for parameter in item.get("proxy_evidence", {}).get(
                "unsupported_parameters", []
            )
        }),
    }


def model_call_payload(
    sample: int, ledger: Any, target: Path
) -> dict[str, Any]:
    starts = [record for record in ledger.records
              if record.get("event") == "model_call_started"]
    terminals = {
        str(record["logical_call_id"]): record
        for record in ledger.records
        if record.get("event") in {"model_call_finished", "model_call_failed"}
    }
    calls = []
    for start in starts:
        logical_id = str(start["logical_call_id"])
        terminal = terminals.get(logical_id)
        if terminal is None:
            raise RuntimeError(f"orphan model call: {logical_id}")
        calls.append({
            "logical_call_id": logical_id,
            "operation_id": start["operation_id"],
            "status": (
                "success" if terminal["event"] == "model_call_finished"
                else "failed"
            ),
            "requested_model": start["requested_model"],
            "request_path": start["request_path"],
            "request_sha256": start["request_sha256"],
            "local_visible_tokens": start["local_visible_tokens"],
            "tokenizer": start["tokenizer"],
            "response_path": terminal.get("response_path"),
            "response_sha256": terminal.get("response_sha256"),
            "response_model": terminal.get("response_model"),
            "response_id": terminal.get("response_id"),
            "usage": terminal.get("usage"),
            "latency_s": terminal.get("latency_s"),
            "client_http_attempts": terminal.get("proxy_evidence", {}).get(
                "client_http_attempts", 0
            ),
            "physical_upstream_attempts": terminal.get(
                "proxy_evidence", {}
            ).get("upstream_http_attempts", 0),
            "unsupported_parameters": terminal.get(
                "proxy_evidence", {}
            ).get("unsupported_parameters", []),
        })
    cost = model_cost_summary(ledger.records)
    return {
        "schema": "nativemem.r207-model-calls.v2",
        "sample": sample,
        "ledger_path": "operations.jsonl",
        "ledger_sha256": file_sha256(target / "operations.jsonl"),
        "ledger_event_count": len(ledger.records),
        "call_count": len(calls),
        "successful_call_count": cost["successful_model_calls"],
        "failed_call_count": cost["failed_model_calls"],
        "calls_sha256": value_sha256(calls),
        "cost": cost,
        "call_artifact_tree": tree_descriptor(target / "calls"),
        "calls": calls,
    }


def sample_artifact(
    sample: int,
    sample_id: str,
    conv: dict[str, Any],
    question_mappings: list[dict[str, Any]],
    target: Path,
    adapter: Any,
    run_fingerprint: str,
    source_before: dict[str, str],
    model_config: dict[str, Any],
    token_counter: Any,
    proxy_log: Path | None,
    formal: bool,
) -> dict[str, Any]:
    if target.exists():
        raise RuntimeError(f"refusing stale sample staging directory: {target}")
    target.mkdir(parents=True)
    build_dir = target / "captured_final_build"
    (target / "calls").mkdir()
    expected_sources = sample_expected_dia_ids(conv)
    recorder = CaptureRecorder(
        sample, adapter.v8_memory._sanitize_topic, expected_sources
    )
    from src.evaluation.durable_model_ledger import (
        DurableModelObserver,
        HashChainLedger,
        ledger_state,
    )

    operation_id = f"sample-{sample:02d}/capture-build"
    ledger = HashChainLedger(
        target / "operations.jsonl",
        run_id=f"{run_fingerprint}:sample-{sample}",
    )
    empty_tree = {
        "file_count": 0,
        "tree_sha256": value_sha256([]),
    }
    operation_metadata = {
        "sample": sample,
        "selected_input_sha256": value_sha256(conv),
        "capture_point": "distill_events return before write_events",
        "method": "NativeMem-v8.8+calendar",
    }
    operation_input = value_sha256({
        "kind": "capture_final_build", "metadata": operation_metadata,
    })
    ledger.append(
        "operation_started",
        operation_id=operation_id,
        kind="capture_final_build",
        operation_input_sha256=operation_input,
        metadata=operation_metadata,
        continuing_tree_before=empty_tree,
    )
    observer = None
    if formal:
        client = getattr(adapter.v8_memory, "client", None)
        resource = getattr(getattr(client, "chat", None), "completions", None)
        if not callable(getattr(resource, "create", None)):
            raise RuntimeError("formal adapter has no completion resource")
        observer = DurableModelObserver(
            ledger=ledger,
            artifact_root=target,
            token_counter=token_counter,
            expected_model=str(model_config["requested_model"]),
            proxy_log=proxy_log,
            formal=True,
        )
        observer.install(resource)
    started = time.monotonic()
    try:
        with AdapterCaptureHook(adapter, recorder):
            if observer is None:
                build_result = adapter.build_memory(conv, str(build_dir))
            else:
                with observer.operation(operation_id):
                    build_result = adapter.build_memory(conv, str(build_dir))
        after_tree = tree_descriptor(build_dir)
        operation_records = [
            record for record in ledger.records
            if record.get("operation_id") == operation_id
            and record.get("event", "").startswith("model_call_")
        ]
        ledger.append(
            "operation_committed",
            operation_id=operation_id,
            kind="capture_final_build",
            operation_input_sha256=operation_input,
            continuing_tree_before=empty_tree,
            continuing_tree_after=after_tree,
            latency_s=round(time.monotonic() - started, 6),
            cost=model_cost_summary(operation_records),
            result={
                "build_result": (
                    list(build_result) if isinstance(build_result, tuple)
                    else build_result
                ),
                "distill_call_count": len(recorder.calls),
                "entry_count": len(recorder.entries),
            },
        )
    except BaseException as exc:
        ledger.append(
            "operation_failed",
            operation_id=operation_id,
            kind="capture_final_build",
            operation_input_sha256=operation_input,
            continuing_tree_after=(
                tree_descriptor(build_dir) if build_dir.is_dir() else empty_tree
            ),
            latency_s=round(time.monotonic() - started, 6),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if observer is not None:
            observer.restore()
    ledger_state(ledger.records)
    if source_hashes() != source_before:
        raise RuntimeError("core source changed during the active build")
    if not recorder.calls or not recorder.entries:
        raise RuntimeError("capture build produced no distill calls or no entries")
    if (isinstance(build_result, tuple) and len(build_result) >= 2
            and int(build_result[1]) != len(recorder.entries)):
        raise RuntimeError(
            "adapter event count differs from the captured canonical entry count"
        )
    model_calls_payload = model_call_payload(sample, ledger, target)
    if formal and not model_calls_payload["call_count"]:
        raise RuntimeError("formal capture build observed no model calls")
    if not formal and model_calls_payload["call_count"]:
        raise RuntimeError("synthetic sanity unexpectedly made a model call")

    bank = bank_payload(sample, recorder.entries)
    model_placements = [
        {"entry_id": item["entry_id"], "topic_path": item["topic_path"]}
        for item in recorder.original_placements
    ]
    model_payload = placement_payload("model_directed", bank, model_placements)
    permuted, permutation_meta = deterministic_permutation(model_placements)
    permutation_payload = placement_payload(
        "deterministic_permutation", bank, permuted, permutation_meta
    )
    assert_placement_match(model_payload, permutation_payload)

    original_payload = {
        "schema": ORIGINAL_PLACEMENT_SCHEMA,
        "sample": sample,
        "bank_entries_sha256": bank["entries_sha256"],
        "placements_sha256": value_sha256(recorder.original_placements),
        "placements": recorder.original_placements,
    }
    capture_payload = {
        "schema": CAPTURE_SCHEMA,
        "sample": sample,
        "capture_exactness": (
            "Events are copied immediately after final v8.8 distill_events "
            "returns, before write_events, and checked against the same list "
            "passed to write_events. "
            "Extraction still sees final-method original topics and known_topics; "
            "topic removal applies only to downstream placement experiments."
        ),
        "call_count": len(recorder.calls),
        "calls_sha256": value_sha256(recorder.calls),
        "calls": recorder.calls,
    }
    atomic_json(target / "canonical_entry_bank.json", bank)
    atomic_json(target / "original_placement.json", original_payload)
    atomic_json(target / "capture_calls.json", capture_payload)
    atomic_json(target / "model_calls.json", model_calls_payload)
    placement_dir = target / "placements"
    atomic_json(placement_dir / "model_directed.json", model_payload)
    atomic_json(
        placement_dir / "deterministic_permutation.json", permutation_payload
    )

    conditions_dir = target / "conditions"
    condition_trees = {
        "model_directed": materialize_condition(
            conditions_dir / "model_directed",
            bank,
            model_payload,
            adapter.v8_memory.write_events,
            adapter.v8_memory.dedup_topic_files,
        ),
        "deterministic_permutation": materialize_condition(
            conditions_dir / "deterministic_permutation",
            bank,
            permutation_payload,
            adapter.v8_memory.write_events,
            adapter.v8_memory.dedup_topic_files,
        ),
    }
    traces = question_trace_records(
        sample=sample,
        sample_id=sample_id,
        mappings=question_mappings,
        bank=bank,
        build_dir=build_dir,
        conditions_dir=conditions_dir,
    )
    atomic_jsonl(target / "question_traces.jsonl", traces)
    build_tree = tree_descriptor(build_dir)
    artifacts = {
        "canonical_entry_bank.json": file_sha256(
            target / "canonical_entry_bank.json"
        ),
        "original_placement.json": file_sha256(target / "original_placement.json"),
        "capture_calls.json": file_sha256(target / "capture_calls.json"),
        "model_calls.json": file_sha256(target / "model_calls.json"),
        "operations.jsonl": file_sha256(target / "operations.jsonl"),
        "question_traces.jsonl": file_sha256(
            target / "question_traces.jsonl"
        ),
        "placements/model_directed.json": file_sha256(
            placement_dir / "model_directed.json"
        ),
        "placements/deterministic_permutation.json": file_sha256(
            placement_dir / "deterministic_permutation.json"
        ),
    }
    sample_manifest = {
        "schema": SAMPLE_SCHEMA,
        "status": "complete",
        "sample": sample,
        "sample_id": sample_id,
        "run_fingerprint": run_fingerprint,
        "completed_at": utc_now(),
        "selected_input_sha256": value_sha256(conv),
        "expected_source_count": len(expected_sources),
        "core_source_hashes": source_before,
        "model_config": model_config,
        "build_result": list(build_result) if isinstance(build_result, tuple)
        else build_result,
        "canonical_bank": {
            "entry_count": bank["entry_count"],
            "entries_sha256": bank["entries_sha256"],
            "source_identity_sha256": bank["source_identity_sha256"],
        },
        "path_match": {
            "fields": [
                "entry_count",
                "topic_count",
                "per_topic_counts",
                "maximum_depth",
                "path_string_total_bytes",
            ],
            "residual_path_string_bytes": 0,
            "permutation": permutation_meta,
        },
        "maintenance_policy": {
            "captured_final_build": "final v8.8 maintenance remains enabled",
            "path_only_conditions": (
                "model maintenance disabled; deterministic write_events, "
                "timeline generation, and exact deduplication only"
            ),
        },
        "artifact_sha256": artifacts,
        "tree_hashes": {
            "captured_final_build": build_tree,
            "conditions": condition_trees,
        },
        "model_ledger": {
            "schema": "durable-model-ledger/v1",
            "ledger_sha256": file_sha256(target / "operations.jsonl"),
            "ledger_event_count": len(ledger.records),
            "cost": model_calls_payload["cost"],
            "call_artifact_tree": model_calls_payload["call_artifact_tree"],
        },
        "question_traces": {
            "schema": TRACE_SCHEMA,
            "trace_count": len(traces),
            "trace_ids_sha256": value_sha256([
                record["trace_id"] for record in traces
            ]),
            "artifact_sha256": file_sha256(target / "question_traces.jsonl"),
            "unobserved_stages": ["retrieval_reach", "source_resolution"],
        },
    }
    atomic_json(target / "sample_manifest.json", sample_manifest)
    assert_safe_tree(target)
    return sample_manifest


def configure_environment(
    args: argparse.Namespace, builder_base: str
) -> dict[str, Any]:
    os.environ.update(FINAL_ENV)
    os.environ["NATIVEMEM_V8_CONCURRENCY"] = str(args.request_concurrency)
    os.environ["BUILDER_MODEL"] = args.model
    os.environ["BUILDER_BASE"] = builder_base
    os.environ["BUILDER_KEY"] = args.api_key
    os.environ["ALIYUN_KEY"] = args.api_key
    os.environ.pop("MODEL", None)
    os.environ.pop("NATIVEMEM_V9_PIPELINE", None)
    os.environ.pop("NATIVEMEM_V9_SCRIBE_MODE", None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    return {
        "method": "NativeMem-v8.8+calendar",
        "calendar_context": "deterministic calendar_strip from final v8.8 source",
        "requested_model": args.model,
        "transport": (
            "synthetic_no_transport" if args.synthetic_sanity
            else "managed_exclusive_proxy"
        ),
        "upstream_base_url": (
            None
            if args.synthetic_sanity
            else args.gateway_contract["origin"]
        ),
        "api_key_recorded": False,
        "api_key_configured": bool(args.api_key),
        "request_concurrency": args.request_concurrency,
        "environment": {
            **FINAL_ENV,
            "NATIVEMEM_V8_CONCURRENCY": str(args.request_concurrency),
        },
        "forbidden_environment_absent": [
            "MODEL", "NATIVEMEM_V9_PIPELINE", "NATIVEMEM_V9_SCRIBE_MODE"
        ],
    }


def load_adapter() -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("src.adapters.run_nativemem")


class ManagedProxy:
    def __init__(
        self, output: Path, gateway_contract: dict[str, Any], run_id: str
    ) -> None:
        self.output = output
        self.gateway_contract = gateway_contract
        self.upstream = str(gateway_contract["origin"])
        self.run_id = run_id
        self.process: subprocess.Popen[Any] | None = None
        self.log: Path | None = None
        self.ready: Path | None = None
        self.base_url: str | None = None
        self.provider_window: dict[str, Any] | None = None
        self.closed = False

    def start(self) -> None:
        self.provider_window = flex_evidence.capture_start(
            Path(self.gateway_contract["result_root"])
        )
        proxy_dir = self.output / "proxy"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        launch = len(list(proxy_dir.glob("launch-*.ready.json"))) + 1
        self.log = proxy_dir / f"launch-{launch:03d}.jsonl"
        self.ready = proxy_dir / f"launch-{launch:03d}.ready.json"
        if self.log.exists() or self.ready.exists():
            raise RuntimeError("proxy launch artifacts already exist")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "scripts/controlled_gpt55_run_proxy.py",
                "--port", "0",
                "--upstream", self.upstream,
                "--log", str(self.log),
                "--ready", str(self.ready),
                "--run-id", f"{self.run_id}:launch-{launch:03d}",
            ],
            cwd=ROOT,
        )
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("exclusive proxy exited before readiness")
                if self.ready.is_file():
                    value = json.loads(self.ready.read_text(encoding="utf-8"))
                    self.base_url = str(value["base_url"])
                    return
                time.sleep(0.05)
            raise RuntimeError("exclusive proxy readiness timed out")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.closed:
            return
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.ready is not None and self.provider_window is not None:
            closed = flex_evidence.capture_end(self.provider_window)
            ready = json.loads(self.ready.read_text(encoding="utf-8"))
            ready["gateway_contract"] = self.gateway_contract
            ready["provider_window"] = closed
            atomic_json(self.ready, ready)
        self.closed = True


def synthetic_adapter(real_v8: Any) -> tuple[Any, dict[str, Any]]:
    calls = [
        {
            "turns": [("A", "A adopted a dog"), ("B", "B moved home")],
            "obs_date": "2023-05-07",
            "dia_ids": ["D1:1", "D1:2"],
            "events": [
                {
                    "when": "2023-05-06",
                    "summary": "A adopted a dog",
                    "summary_inline": "A adopted a dog [D1:1]",
                    "dia_ids": ["D1:1"],
                    "topic": "A/pets",
                },
                {
                    "when": "2023-05-07",
                    "summary": "B moved home",
                    "summary_inline": "B moved home [D1:2]",
                    "dia_ids": ["D1:2"],
                    "topic": "B/home",
                },
            ],
        },
        {
            "turns": [("A", "A visited a park")],
            "obs_date": "2023-05-08",
            "dia_ids": ["D2:1"],
            "events": [
                {
                    "when": "2023-05-08",
                    "summary": "A visited a park",
                    "summary_inline": "A visited a park [D2:1]",
                    "dia_ids": ["D2:1"],
                    "topic": "A/pets",
                }
            ],
        },
    ]
    queue = [copy.deepcopy(item["events"]) for item in calls]

    def distill_events(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        del args, kwargs
        return queue.pop(0)

    fake_v8 = SimpleNamespace(
        distill_events=distill_events,
        write_events=real_v8.write_events,
        dedup_topic_files=real_v8.dedup_topic_files,
        _sanitize_topic=real_v8._sanitize_topic,
    )
    adapter = SimpleNamespace(v8_memory=fake_v8)
    adapter.distill_events = distill_events
    adapter.write_events = real_v8.write_events

    def build_memory(conv: dict[str, Any], memory_dir: str) -> tuple[float, int]:
        del conv
        count = 0
        for item in calls:
            events = adapter.v8_memory.distill_events(
                item["turns"], item["obs_date"], item["dia_ids"],
                known_topics=[], recent=[]
            )
            adapter.write_events(memory_dir, events)
            count += len(events)
        return 0.0, count

    adapter.build_memory = build_memory
    conv = {
        "session_1": [
            {"dia_id": "D1:1", "speaker": "A", "text": "A adopted a dog"},
            {"dia_id": "D1:2", "speaker": "B", "text": "B moved home"},
        ],
        "session_1_date_time": "2023-05-07",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "A", "text": "A visited a park"}
        ],
        "session_2_date_time": "2023-05-08",
        "qa": [],
    }
    return adapter, conv


def completed_sample_record_for_resume(
    sample_dir: Path,
    *,
    sample: int,
    sample_id: str,
    run_fingerprint: str,
    source_before: dict[str, str],
    model_config: dict[str, Any],
    selected_input_sha256: str,
) -> dict[str, Any]:
    from src.evaluation.durable_model_ledger import read_ledger, ledger_state

    expected_direct = {
        "canonical_entry_bank.json", "original_placement.json",
        "capture_calls.json", "model_calls.json", "operations.jsonl", "calls",
        "question_traces.jsonl", "placements", "conditions", "captured_final_build",
        "sample_manifest.json",
    }
    if (not sample_dir.is_dir() or sample_dir.is_symlink()
            or {path.name for path in sample_dir.iterdir()} != expected_direct):
        raise RuntimeError(
            f"sample {sample} cannot be recovered; use a new output root"
        )
    assert_safe_tree(sample_dir)
    manifest_path = sample_dir / "sample_manifest.json"
    try:
        sample_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"sample {sample} has an invalid manifest; use a new output root"
        ) from exc
    if (
        sample_manifest.get("schema") != SAMPLE_SCHEMA
        or sample_manifest.get("status") != "complete"
        or sample_manifest.get("sample") != sample
        or sample_manifest.get("sample_id") != sample_id
        or sample_manifest.get("run_fingerprint") != run_fingerprint
        or sample_manifest.get("selected_input_sha256")
        != selected_input_sha256
        or sample_manifest.get("core_source_hashes") != source_before
        or sample_manifest.get("model_config") != model_config
    ):
        raise RuntimeError(
            f"sample {sample} provenance differs; use a new output root"
        )
    ledger_path = sample_dir / "operations.jsonl"
    records = read_ledger(ledger_path)
    if not records:
        raise RuntimeError(
            f"sample {sample} has no durable operation; use a new output root"
        )
    ledger_state(records)
    run_id = f"{run_fingerprint}:sample-{sample}"
    if any(record.get("run_id") != run_id for record in records):
        raise RuntimeError(
            f"sample {sample} ledger run ID differs; use a new output root"
        )
    model_ledger = sample_manifest.get("model_ledger")
    if (
        not isinstance(model_ledger, dict)
        or model_ledger.get("schema") != "durable-model-ledger/v1"
        or model_ledger.get("ledger_sha256") != file_sha256(ledger_path)
        or model_ledger.get("ledger_event_count") != len(records)
        or model_ledger.get("call_artifact_tree")
        != tree_descriptor(sample_dir / "calls")
    ):
        raise RuntimeError(
            f"sample {sample} ledger summary differs; use a new output root"
        )
    artifact_hashes = sample_manifest.get("artifact_sha256")
    expected_hash_paths = {
        "canonical_entry_bank.json", "original_placement.json",
        "capture_calls.json", "model_calls.json", "operations.jsonl",
        "question_traces.jsonl",
        "placements/model_directed.json",
        "placements/deterministic_permutation.json",
    }
    if (not isinstance(artifact_hashes, dict)
            or set(artifact_hashes) != expected_hash_paths
            or any(
                not (sample_dir / relative).is_file()
                or (sample_dir / relative).is_symlink()
                or file_sha256(sample_dir / relative) != expected
                for relative, expected in artifact_hashes.items()
            )):
        raise RuntimeError(
            f"sample {sample} artifact hashes differ; use a new output root"
        )
    canonical = sample_manifest.get("canonical_bank")
    if not isinstance(canonical, dict) or not isinstance(
        canonical.get("entry_count"), int
    ):
        raise RuntimeError(
            f"sample {sample} bank summary is invalid; use a new output root"
        )
    return {
        "status": "complete",
        "completed_at": sample_manifest["completed_at"],
        "sample_manifest_sha256": file_sha256(manifest_path),
        "entry_count": canonical["entry_count"],
    }


def existing_root_is_resumable(
    output: Path,
    run_fingerprint: str,
    requested_samples: list[int],
    *,
    expected_config: dict[str, Any],
    source_before: dict[str, str],
    selected_input_hashes: dict[int, str],
    sample_ids: list[str],
) -> dict[str, Any] | None:
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("output directory is not a regular directory")
    assert_safe_tree(output)
    allowed = {"run_manifest.json", "samples", "proxy"}
    unexpected = sorted(path.name for path in output.iterdir()
                        if path.name not in allowed)
    if unexpected:
        raise RuntimeError(f"stale or unknown root artifacts: {unexpected}")
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("existing output lacks a regular run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_SCHEMA:
        raise RuntimeError("existing output has a different schema")
    if manifest.get("status") == "failed":
        raise RuntimeError("failed runs require a new output root")
    if manifest.get("status") not in {"running", "complete"}:
        raise RuntimeError("existing output has a non-resumable status")
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("existing output has a different run fingerprint")
    if manifest.get("requested_samples") != requested_samples:
        raise RuntimeError("existing output has a different sample set")
    if (manifest.get("config") != expected_config
            or manifest.get("core_source_hashes") != source_before):
        raise RuntimeError("existing output provenance differs")
    registry = manifest.get("samples")
    if not isinstance(registry, dict) or not set(registry).issubset(
        {str(sample) for sample in requested_samples}
    ):
        raise RuntimeError("existing output has an invalid sample registry")
    synthetic = manifest.get("config", {}).get("data", {}).get("mode") == (
        "synthetic_sanity"
    )
    if synthetic and (output / "proxy").exists():
        raise RuntimeError("synthetic output contains proxy artifacts")
    samples_dir = output / "samples"
    if not samples_dir.is_dir() or samples_dir.is_symlink():
        raise RuntimeError("existing output lacks a regular samples directory")
    final_dirs: dict[int, Path] = {}
    staging_dirs: dict[int, Path] = {}
    for path in samples_dir.iterdir():
        final_match = re.fullmatch(r"sample-(\d+)", path.name)
        staging_match = re.fullmatch(r"\.sample-(\d+)\.staging-\d+", path.name)
        if final_match:
            sample = int(final_match.group(1))
            if sample in final_dirs:
                raise RuntimeError("duplicate final sample directory")
            final_dirs[sample] = path
        elif staging_match:
            sample = int(staging_match.group(1))
            if sample in staging_dirs:
                raise RuntimeError("duplicate staging sample directory")
            staging_dirs[sample] = path
        else:
            raise RuntimeError(f"stale sample artifact is present: {path.name}")
    allowed_samples = set(requested_samples)
    if (set(final_dirs) | set(staging_dirs)) - allowed_samples:
        raise RuntimeError("sample artifact is outside the requested set")

    from src.evaluation.durable_model_ledger import read_ledger, ledger_state

    for sample, staging in sorted(staging_dirs.items()):
        if sample in final_dirs:
            raise RuntimeError(
                f"sample {sample} has final and staging artifacts; "
                "use a new output root"
            )
        ledger_path = staging / "operations.jsonl"
        try:
            records = read_ledger(ledger_path)
            if not records:
                raise RuntimeError("staging ledger is empty")
            ledger_state(records)
        except Exception as exc:
            raise RuntimeError(
                f"sample {sample} staging state cannot be resumed without "
                "repeating work; use a new output root: {exc}"
            ) from exc
        record = completed_sample_record_for_resume(
            staging,
            sample=sample,
            sample_id=sample_ids[sample],
            run_fingerprint=run_fingerprint,
            source_before=source_before,
            model_config=expected_config["model_config"],
            selected_input_sha256=selected_input_hashes[sample],
        )
        final = samples_dir / f"sample-{sample}"
        os.replace(staging, final)
        directory = os.open(samples_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        final_dirs[sample] = final
        manifest["samples"][str(sample)] = record
        atomic_json(output / "run_manifest.json", manifest)

    for sample, sample_dir in sorted(final_dirs.items()):
        recovered = completed_sample_record_for_resume(
            sample_dir,
            sample=sample,
            sample_id=sample_ids[sample],
            run_fingerprint=run_fingerprint,
            source_before=source_before,
            model_config=expected_config["model_config"],
            selected_input_sha256=selected_input_hashes[sample],
        )
        registered = manifest["samples"].get(str(sample))
        if registered is not None and registered.get("status") == "complete":
            if registered != recovered:
                raise RuntimeError(f"completed sample {sample} was modified")
        else:
            manifest["samples"][str(sample)] = recovered
            atomic_json(output / "run_manifest.json", manifest)

    for sample_text, record in manifest["samples"].items():
        if record.get("status") == "complete" and int(sample_text) not in final_dirs:
            raise RuntimeError(f"completed sample {sample_text} is missing")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=parse_samples, default=parse_samples("0-9"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--api-key", default=os.environ.get("BUILDER_KEY", "x"))
    parser.add_argument("--request-concurrency", type=int, default=10)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.request_concurrency < 1:
        parser.error("--request-concurrency must be positive")
    if os.environ.get("NATIVEMEM_V9_PIPELINE") == "two_tier":
        parser.error(
            "NATIVEMEM_V9_PIPELINE=two_tier is forbidden for the final v8.8 gate"
        )
    if not args.synthetic_sanity and not args.allow_model_requests:
        parser.error("formal builds require the explicit --allow-model-requests gate")
    if args.synthetic_sanity:
        if args.allow_model_requests:
            parser.error("synthetic R207 forbids --allow-model-requests")
        if args.gateway_root is not None:
            parser.error("synthetic R207 forbids --gateway-root")
        args.samples = [0]
        args.model = "synthetic-no-model"
        args.gateway_root = None
    else:
        if args.samples != list(range(10)):
            parser.error("formal R207 requires all samples 0-9")
        if args.model != "gpt-5.5":
            parser.error("formal R207 requires --model gpt-5.5")
        if args.request_concurrency != 10:
            parser.error("formal R207 requires --request-concurrency 10")
        if args.gateway_root is None:
            parser.error("formal R207 requires --gateway-root")

    raw_output = args.output_dir.expanduser()
    absolute_output = raw_output if raw_output.is_absolute() else ROOT / raw_output
    current = Path(absolute_output.anchor)
    for part in absolute_output.parts[1:]:
        current /= part
        if current.is_symlink():
            parser.error(f"output path contains symlink component: {current}")
    output = absolute_output.resolve()
    data_path = args.data.expanduser().resolve()
    if output == ROOT or ROOT not in output.parents:
        parser.error("--output-dir must be a descendant of the repository root")
    lock_path = ROOT / "results" / ".r207-active-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another R207 capture build is active") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(json.dumps({"pid": os.getpid(), "output": str(output),
                                  "started_at": utc_now()}))
    lock_handle.flush()
    os.fsync(lock_handle.fileno())

    lock_cleaned = False

    def cleanup_lock() -> None:
        nonlocal lock_cleaned
        if lock_cleaned:
            return
        lock_cleaned = True
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        try:
            lock_handle.close()
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)

    atexit.register(cleanup_lock)

    args.gateway_contract = (
        None
        if args.synthetic_sanity
        else flex_evidence.active_contract(args.gateway_root.expanduser().resolve())
    )
    source_before = source_hashes()
    observations = environment_observations()
    model_config = configure_environment(
        args, "none" if args.synthetic_sanity else "managed-exclusive-proxy"
    )
    if args.synthetic_sanity:
        real_adapter = load_adapter()
        adapter, synthetic_conv = synthetic_adapter(real_adapter.v8_memory)
        dataset: list[dict[str, Any]] = [synthetic_conv]
        conversations = [synthetic_conv]
        sample_ids = ["synthetic-sample-0"]
        question_mappings = {0: []}
        dataset_sha = value_sha256(dataset)
        data_descriptor = {"mode": "synthetic_sanity", "sha256": dataset_sha}
        mapping_descriptor = {
            "mode": "synthetic_no_questions",
            "sha256": value_sha256([]),
            "record_count": 0,
        }
    else:
        if (not data_path.is_file() or data_path.is_symlink()
                or data_path != DEFAULT_DATA.resolve()):
            raise SystemExit("formal R207 requires the frozen LoCoMo dataset")
        dataset = json.loads(data_path.read_text(encoding="utf-8"))
        dataset_sha = file_sha256(data_path)
        if dataset_sha != EXPECTED_DATA_SHA256:
            raise SystemExit("formal LoCoMo dataset hash mismatch")
        data_descriptor = {
            "mode": "locomo",
            "path": data_path.relative_to(ROOT).as_posix(),
            "sha256": dataset_sha,
        }
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("conversation"), dict)
            and isinstance(item.get("sample_id"), str)
            for item in dataset
        ):
            raise SystemExit("formal LoCoMo samples are malformed")
        conversations = [conversation_payload(item) for item in dataset]
        sample_ids = [str(item["sample_id"]) for item in dataset]
        question_mappings = load_formal_question_mappings()
        mapping_descriptor = {
            "mode": "frozen_evidence_mapping",
            "path": EVIDENCE_QUESTIONS.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_EVIDENCE_SHA256,
            "record_count": 1540,
        }
        adapter = None
    if not isinstance(dataset, list):
        raise SystemExit("dataset root must be a list")
    if any(sample >= len(dataset) for sample in args.samples):
        raise SystemExit("sample index is outside the dataset")

    gateway_contract = args.gateway_contract
    run_config = {
        "samples": args.samples,
        "data": data_descriptor,
        "question_mapping": mapping_descriptor,
        "model_config": model_config,
        "capture_point": "distill_events return before write_events",
        "canonical_bank_topics_removed": True,
        "condition_maintenance": "off",
        "materialization": "write_events + dedup_topic_files",
        "gateway_contract": gateway_contract,
    }
    run_fingerprint = value_sha256({
        "schema": RUN_SCHEMA,
        "config": run_config,
        "core_source_hashes": source_before,
    })
    selected_input_hashes = {
        sample: value_sha256(conversations[sample]) for sample in args.samples
    }
    existing = existing_root_is_resumable(
        output,
        run_fingerprint,
        args.samples,
        expected_config=run_config,
        source_before=source_before,
        selected_input_hashes=selected_input_hashes,
        sample_ids=sample_ids,
    ) if output.exists() else None
    if existing is not None and not args.resume:
        raise SystemExit("output exists; pass --resume only for the same fingerprint")
    if existing is None:
        output.mkdir(parents=True)
        (output / "samples").mkdir()
        manifest: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "status": "running",
            "created_at": utc_now(),
            "requested_samples": args.samples,
            "run_fingerprint": run_fingerprint,
            "config": run_config,
            "core_source_hashes": source_before,
            "environment_observations": observations,
            "active_build_exclusivity": {
                "mechanism": "non-blocking POSIX advisory lock",
                "lock_path": lock_path.relative_to(ROOT).as_posix(),
                "one_active_runner_process": True,
            },
            "samples": {},
        }
        atomic_json(output / "run_manifest.json", manifest)
    else:
        manifest = existing
        if all(
            manifest["samples"].get(str(sample), {}).get("status") == "complete"
            for sample in args.samples
        ):
            manifest["status"] = "complete"
            manifest.setdefault("completed_at", utc_now())
            atomic_json(output / "run_manifest.json", manifest)
            assert_safe_tree(output)
            cleanup_lock()
            print("all requested samples are already complete", flush=True)
            return 0
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
        atomic_json(output / "run_manifest.json", manifest)

    from src.evaluation.visible_token_budget import TokenCounter

    proxy = None
    provider_lock = None
    if args.synthetic_sanity:
        token_counter = TokenCounter.utf8_bytes(
            requested_model=args.model, fallback_reason="synthetic_sanity"
        )
    else:
        try:
            version = importlib.metadata.version("tiktoken")
        except importlib.metadata.PackageNotFoundError as exc:
            raise SystemExit("formal R207 requires tiktoken==0.12.0") from exc
        if version != "0.12.0":
            raise SystemExit("formal R207 requires tiktoken==0.12.0")
        provider_lock = flex_evidence.acquire_consumer_lock(
            Path(gateway_contract["result_root"])
        )
        if flex_evidence.active_contract(
            Path(gateway_contract["result_root"])
        ) != gateway_contract:
            provider_lock.close()
            raise RuntimeError("Flex gateway contract changed before execution")
        proxy = ManagedProxy(output, gateway_contract, run_fingerprint)
        try:
            proxy.start()
        except BaseException:
            provider_lock.close()
            raise
        atexit.register(proxy.stop)
        configure_environment(args, str(proxy.base_url))
        adapter = load_adapter()
        token_counter = TokenCounter.resolve(
            requested_model=args.model,
            fallback_encoding="o200k_base",
            allow_byte_fallback=False,
        )

    try:
        for sample in args.samples:
            key = str(sample)
            if manifest["samples"].get(key, {}).get("status") == "complete":
                print(f"sample {sample}: already complete", flush=True)
                continue
            staging = output / "samples" / f".sample-{sample}.staging-{os.getpid()}"
            final = output / "samples" / f"sample-{sample}"
            if staging.exists() or final.exists():
                raise RuntimeError(f"refusing stale sample artifact for {sample}")
            manifest["samples"][key] = {
                "status": "running", "started_at": utc_now()
            }
            atomic_json(output / "run_manifest.json", manifest)
            sample_manifest = sample_artifact(
                sample,
                sample_ids[sample],
                conversations[sample],
                question_mappings[sample],
                staging,
                adapter,
                run_fingerprint,
                source_before,
                model_config,
                token_counter,
                proxy.log if proxy else None,
                formal=not args.synthetic_sanity,
            )
            os.replace(staging, final)
            directory = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            manifest["samples"][key] = {
                "status": "complete",
                "completed_at": sample_manifest["completed_at"],
                "sample_manifest_sha256": file_sha256(
                    final / "sample_manifest.json"
                ),
                "entry_count": sample_manifest["canonical_bank"]["entry_count"],
            }
            atomic_json(output / "run_manifest.json", manifest)
            print(
                f"sample {sample}: complete, "
                f"entries={sample_manifest['canonical_bank']['entry_count']}",
                flush=True,
            )
        if source_hashes() != source_before:
            raise RuntimeError("core source changed before run publication")
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        atomic_json(output / "run_manifest.json", manifest)
        assert_safe_tree(output)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        atomic_json(output / "run_manifest.json", manifest)
        raise
    finally:
        try:
            if proxy is not None:
                proxy.stop()
        finally:
            if provider_lock is not None:
                provider_lock.close()
            cleanup_lock()


if __name__ == "__main__":
    raise SystemExit(main())
