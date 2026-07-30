#!/usr/bin/env python3
"""Independently audit R207 canonical-bank and path-control artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src import openai_gpt55_flex_gateway_evidence as flex_evidence


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_QUESTIONS = (
    ROOT / "results" / "paper-experiments-20260714" / "evidence-mapping"
    / "v1" / "evidence_mapping.v1.questions.jsonl"
)
BANK_SCHEMA = "nativemem.canonical-entry-bank.v1"
PLACEMENT_SCHEMA = "nativemem.path-placement.v1"
ORIGINAL_PLACEMENT_SCHEMA = "nativemem.original-path-placement.v1"
CAPTURE_SCHEMA = "nativemem.distill-write-capture.v1"
SAMPLE_SCHEMA = "nativemem.r207-sample.v1"
RUN_SCHEMA = "nativemem.r207-run.v1"
LEDGER_SCHEMA = "durable-model-ledger/v1"
REQUEST_SCHEMA = "durable-model-request/v1"
RESPONSE_SCHEMA = "durable-model-response/v1"
ZERO_HASH = "0" * 64
TRACE_SCHEMA = "nativemem.r207-question-trace.v1"
CONDITIONS = ("model_directed", "deterministic_permutation")
EXPECTED_DATA_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
EXPECTED_EVIDENCE_SHA256 = (
    "3291df579bc20d0678f99b3361ac6d287beee583e4d3dff93b996f4522e56318"
)
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


class AuditFailure(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


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


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(),
            f"missing or non-regular JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"invalid JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(),
            f"missing or non-regular JSONL file: {path}")
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), f"incomplete JSONL file: {path}")
    records = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditFailure(
                f"invalid JSONL {path}:{line_number}: {exc}"
            ) from exc
        require(isinstance(value, dict),
                f"non-object JSONL record: {path}:{line_number}")
        records.append(value)
    return records


def collision_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def assert_safe_tree(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(),
            f"artifact root is not a regular directory: {root}")
    paths: dict[str, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        key = collision_key(relative)
        require(key not in paths or paths[key] == relative,
                f"Unicode/case path collision: {paths.get(key)} / {relative}")
        paths[key] = relative
        require(not path.is_symlink(), f"symlink is forbidden: {relative}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            require(stat.st_nlink == 1, f"hardlink is forbidden: {relative}")
            inode = (stat.st_dev, stat.st_ino)
            require(inode not in inodes,
                    f"shared inode: {inodes.get(inode)} / {relative}")
            inodes[inode] = relative
        else:
            require(path.is_dir(),
                    f"special filesystem node is forbidden: {relative}")


def tree_descriptor(root: Path) -> dict[str, Any]:
    assert_safe_tree(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat(follow_symlinks=False).st_size,
            "sha256": file_sha256(path),
        })
    return {"file_count": len(files), "tree_sha256": value_sha256(files)}


def safe_artifact_path(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label}: invalid path")
    candidate = Path(relative)
    require(not candidate.is_absolute() and ".." not in candidate.parts,
            f"{label}: unsafe relative path")
    resolved = root / candidate
    require(resolved.is_file() and not resolved.is_symlink(),
            f"{label}: file does not exist")
    return resolved


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(actual == expected,
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")


def natural_dia_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"D(\d+):(\d+)", value)
    if match:
        return int(match.group(1)), int(match.group(2)), ""
    return sys.maxsize, sys.maxsize, value


def sorted_dia_ids(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)},
                  key=natural_dia_key)


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


def independent_sanitize_topic(topic: str, max_depth: int) -> str:
    def clean(segment: str) -> str:
        return re.sub(r"[^\w]+", "-", segment).strip("-")

    segments = [clean(segment) for segment in topic.split("/")]
    segments = [segment for segment in segments if segment and segment != ".."]
    if not segments:
        return "misc"
    if len(segments) > max_depth:
        segments = (
            segments[:max_depth - 1]
            + ["-".join(segments[max_depth - 1:])]
        )
    return "/".join(segments)


def validate_topic_path(topic_path: str, label: str) -> None:
    require(isinstance(topic_path, str) and bool(topic_path),
            f"{label}: empty topic path")
    require(not topic_path.startswith("/") and "\\" not in topic_path,
            f"{label}: absolute or platform-dependent path")
    parts = topic_path.split("/")
    require(all(part not in ("", ".", "..") for part in parts),
            f"{label}: unsafe path component")


def placement_metrics(placements: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(item["topic_path"] for item in placements)
    return {
        "entry_count": len(placements),
        "topic_count": len(counts),
        "per_topic_counts": dict(sorted(counts.items())),
        "maximum_depth": max(
            (len(item["topic_path"].split("/")) for item in placements),
            default=0,
        ),
        "path_string_total_bytes": sum(
            len(item["topic_path"].encode("utf-8")) for item in placements
        ),
    }


def expected_permutation(
    model_placements: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    ordered = sorted(model_placements, key=lambda item: item["entry_id"])
    count = len(ordered)
    if count <= 1:
        fixed, offset = count, 0
    else:
        paths = [item["topic_path"] for item in ordered]
        candidates = []
        for candidate in range(1, count):
            fixed_for_candidate = sum(
                paths[index] == paths[(index + candidate) % count]
                for index in range(count)
            )
            candidates.append((fixed_for_candidate, candidate))
        fixed, offset = min(candidates)
    paths = [item["topic_path"] for item in ordered]
    placements = [
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
        "fixed_point_rate": fixed / count if count else 0.0,
        "minimum_cyclic_fixed_points": fixed,
        "candidate_offsets": max(0, count - 1),
    }
    return placements, metadata


def dataset_source_ids(conv: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    session = 1
    while f"session_{session}" in conv:
        turns = conv[f"session_{session}"]
        require(isinstance(turns, list), f"session_{session} is not a list")
        for turn in turns:
            if isinstance(turn, dict) and turn.get("dia_id"):
                identifiers.add(str(turn["dia_id"]))
        session += 1
    return identifiers


def conversation_payload(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = sample.get("conversation")
    require(isinstance(conversation, dict), "formal sample lacks conversation")
    return conversation


def load_formal_question_mappings() -> dict[int, list[dict[str, Any]]]:
    require(EVIDENCE_QUESTIONS.is_file() and not EVIDENCE_QUESTIONS.is_symlink()
            and file_sha256(EVIDENCE_QUESTIONS) == EXPECTED_EVIDENCE_SHA256,
            "frozen evidence mapping hash mismatch")
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in load_jsonl(EVIDENCE_QUESTIONS):
        if (record.get("benchmark") != "LoCoMo"
                or record.get("qa_scoring_eligible") is not True):
            continue
        question_id = str(record.get("question_id", ""))
        require(question_id and question_id not in seen,
                "evidence mapping question ID is invalid")
        seen.add(question_id)
        grouped[int(record["sample_index"])].append(record)
    require(len(seen) == 1540 and set(grouped) == set(range(10)),
            "formal evidence mapping inventory differs")
    for records in grouped.values():
        records.sort(key=lambda record: int(record["question_index"]))
    return dict(grouped)


def trace_source_locations(memory_dir: Path) -> dict[str, list[str]]:
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


def expected_trace_stage_evidence(
    mapping: dict[str, Any],
    *,
    condition: str,
    canonical_sources: set[str],
    maintained_sources: set[str],
    condition_locations: dict[str, list[str]],
) -> dict[str, Any]:
    source_values = [str(value) for value in mapping["normalized_source_ids"]]
    eligible = bool(mapping["source_recall_eligible"])
    present = sorted(set(source_values) & canonical_sources)
    missing = sorted(set(source_values) - canonical_sources)
    mapping_stage = {
        "status": "observed",
        "value": eligible,
        "artifact": "frozen_evidence_mapping",
        "question_id": mapping["question_id"],
        "source_ids": source_values,
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
        maintenance_missing = sorted(set(source_values) - maintained_sources)
        maintenance = {
            "status": "observed",
            "value": not maintenance_missing,
            "artifact": "captured_final_build",
            "present_source_ids": sorted(
                set(source_values) & maintained_sources
            ),
            "missing_source_ids": maintenance_missing,
        }
        locations = {
            source_id: condition_locations.get(source_id, [])
            for source_id in source_values
        }
        path_stage = {
            "status": "observed",
            "value": all(locations[source_id] for source_id in source_values),
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
            'source_ids': source_values,
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


def expected_trace_records(
    *,
    sample: int,
    sample_id: str,
    mappings: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    sample_dir: Path,
) -> list[dict[str, Any]]:
    canonical_sources = {
        str(source_id) for entry in entries for source_id in entry["dia_ids"]
    }
    maintained_sources = set(trace_source_locations(
        sample_dir / "captured_final_build"
    ))
    locations = {
        condition: trace_source_locations(sample_dir / "conditions" / condition)
        for condition in CONDITIONS
    }
    output = []
    for mapping in mappings:
        require(int(mapping["sample_index"]) == sample
                and mapping["sample_id"] == sample_id,
                "question mapping sample identity mismatch")
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
                "stage_evidence": expected_trace_stage_evidence(
                    mapping,
                    condition=condition,
                    canonical_sources=canonical_sources,
                    maintained_sources=maintained_sources,
                    condition_locations=locations[condition],
                ),
            })
    require(len({record["trace_id"] for record in output}) == len(output),
            "question trace IDs are not unique")
    return output


def audit_bank(
    bank: dict[str, Any], sample: int, expected_sources: set[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    exact_keys(bank, {
        "schema", "sample", "entry_count", "entries_sha256",
        "source_identity_sha256", "entries",
    }, "canonical bank")
    require(bank["schema"] == BANK_SCHEMA, "wrong canonical bank schema")
    require(bank["sample"] == sample, "canonical bank sample mismatch")
    entries = bank["entries"]
    require(isinstance(entries, list), "canonical bank entries are not a list")
    require(bank["entry_count"] == len(entries), "bank entry count mismatch")
    require(bank["entries_sha256"] == value_sha256(entries),
            "bank entry digest mismatch")
    entry_ids: set[str] = set()
    grouped_ordinals: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"bank entry {index} is not an object")
        exact_keys(entry, {
            "entry_id", "sample", "session", "chunk", "ordinal", "when",
            "summary", "summary_inline", "dia_ids",
        }, f"bank entry {index}")
        require(not ({"topic", "topic_path", "original_topic"} & set(entry)),
                f"bank entry {index} contains placement data")
        require(entry["sample"] == sample, f"bank entry {index} sample mismatch")
        for numeric in ("session", "chunk", "ordinal"):
            require(isinstance(entry[numeric], int) and entry[numeric] >= 1,
                    f"bank entry {index} invalid {numeric}")
        require(isinstance(entry["when"], str), f"bank entry {index} invalid when")
        require(isinstance(entry["summary"], str) and bool(entry["summary"]),
                f"bank entry {index} empty summary")
        require(isinstance(entry["summary_inline"], str),
                f"bank entry {index} invalid summary_inline")
        dia_ids = entry["dia_ids"]
        require(isinstance(dia_ids, list) and bool(dia_ids),
                f"bank entry {index} has no source")
        require(dia_ids == sorted_dia_ids(dia_ids),
                f"bank entry {index} sources are not sorted and unique")
        require(set(dia_ids).issubset(expected_sources),
                f"bank entry {index} source is absent from the dataset")
        sessions = {
            int(match.group(1))
            for dia_id in dia_ids
            if (match := re.fullmatch(r"D(\d+):\d+", dia_id))
        }
        require(len(sessions) == 1 and entry["session"] in sessions,
                f"bank entry {index} session/source mismatch")
        require(entry["entry_id"] == stable_entry_id(entry),
                f"bank entry {index} stable ID mismatch")
        require(entry["entry_id"] not in entry_ids,
                f"duplicate entry ID {entry['entry_id']}")
        entry_ids.add(entry["entry_id"])
        grouped_ordinals[(entry["session"], entry["chunk"])].append(
            entry["ordinal"]
        )
    for group, ordinals in grouped_ordinals.items():
        require(ordinals == list(range(1, len(ordinals) + 1)),
                f"non-contiguous entry ordinals in {group}")
    source_identity = [
        {"entry_id": entry["entry_id"], "dia_ids": entry["dia_ids"]}
        for entry in entries
    ]
    require(bank["source_identity_sha256"] == value_sha256(source_identity),
            "bank source identity digest mismatch")
    return entries, entry_ids


def audit_capture(
    capture: dict[str, Any], sample: int, entries: list[dict[str, Any]]
) -> None:
    exact_keys(capture, {
        "schema", "sample", "capture_exactness", "call_count",
        "calls_sha256", "calls",
    }, "capture log")
    require(capture["schema"] == CAPTURE_SCHEMA, "wrong capture schema")
    require(capture["sample"] == sample, "capture sample mismatch")
    require("before write_events" in capture["capture_exactness"],
            "capture exactness boundary is missing")
    calls = capture["calls"]
    require(isinstance(calls, list), "capture calls are not a list")
    require(capture["call_count"] == len(calls), "capture call count mismatch")
    require(capture["calls_sha256"] == value_sha256(calls),
            "capture call digest mismatch")
    flattened: list[str] = []
    entry_by_id = {entry["entry_id"]: entry for entry in entries}
    previous_chunk: defaultdict[int, int] = defaultdict(int)
    for index, call in enumerate(calls, start=1):
        exact_keys(call, {
            "call_index", "sample", "session", "chunk", "observation_date",
            "chunk_dia_ids", "event_count", "entry_ids",
            "returned_events_sha256", "write_observed",
        }, f"capture call {index}")
        require(call["call_index"] == index, "capture call index mismatch")
        require(call["sample"] == sample, "capture call sample mismatch")
        require(call["chunk"] == previous_chunk[call["session"]] + 1,
                "capture chunk order is not contiguous")
        previous_chunk[call["session"]] = call["chunk"]
        require(call["chunk_dia_ids"] == sorted_dia_ids(call["chunk_dia_ids"]),
                "capture chunk sources are not sorted and unique")
        require(call["event_count"] == len(call["entry_ids"]),
                "capture event count mismatch")
        require(call["write_observed"] is (call["event_count"] > 0),
                "capture write pairing mismatch")
        for ordinal, entry_id in enumerate(call["entry_ids"], start=1):
            require(entry_id in entry_by_id, "capture refers to unknown entry")
            entry = entry_by_id[entry_id]
            require(
                (entry["session"], entry["chunk"], entry["ordinal"])
                == (call["session"], call["chunk"], ordinal),
                "capture entry position mismatch",
            )
            require(set(entry["dia_ids"]).issubset(set(call["chunk_dia_ids"])),
                    "capture entry source is outside its chunk")
        flattened.extend(call["entry_ids"])
    require(flattened == [entry["entry_id"] for entry in entries],
            "capture order does not reproduce the canonical bank")


def audit_original_placement(
    payload: dict[str, Any], sample: int, bank_hash: str,
    entry_ids: set[str], max_depth: int,
) -> list[dict[str, Any]]:
    exact_keys(payload, {
        "schema", "sample", "bank_entries_sha256", "placements_sha256",
        "placements",
    }, "original placement")
    require(payload["schema"] == ORIGINAL_PLACEMENT_SCHEMA,
            "wrong original placement schema")
    require(payload["sample"] == sample, "original placement sample mismatch")
    require(payload["bank_entries_sha256"] == bank_hash,
            "original placement bank mismatch")
    placements = payload["placements"]
    require(isinstance(placements, list), "original placements are not a list")
    require(payload["placements_sha256"] == value_sha256(placements),
            "original placement digest mismatch")
    seen: set[str] = set()
    for index, item in enumerate(placements):
        exact_keys(item, {"entry_id", "original_topic", "topic_path"},
                   f"original placement {index}")
        require(item["entry_id"] in entry_ids,
                f"original placement {index} has unknown entry")
        require(item["entry_id"] not in seen,
                f"original placement {index} duplicates an entry")
        seen.add(item["entry_id"])
        validate_topic_path(item["topic_path"], f"original placement {index}")
        expected = independent_sanitize_topic(item["original_topic"], max_depth)
        require(item["topic_path"] == expected,
                f"original placement {index} sanitization mismatch")
    require(seen == entry_ids, "original placement is not a bank bijection")
    return placements


def audit_placement(
    payload: dict[str, Any], condition: str, sample: int,
    bank_hash: str, entry_ids: set[str],
) -> list[dict[str, str]]:
    expected_keys = {
        "schema", "condition", "sample", "bank_entries_sha256",
        "placements_sha256", "metrics", "placements",
    }
    if condition == "deterministic_permutation":
        expected_keys.add("permutation")
    exact_keys(payload, expected_keys, f"{condition} placement")
    require(payload["schema"] == PLACEMENT_SCHEMA,
            f"{condition}: wrong placement schema")
    require(payload["condition"] == condition,
            f"{condition}: condition mismatch")
    require(payload["sample"] == sample, f"{condition}: sample mismatch")
    require(payload["bank_entries_sha256"] == bank_hash,
            f"{condition}: bank mismatch")
    placements = payload["placements"]
    require(isinstance(placements, list), f"{condition}: placements not a list")
    require(payload["placements_sha256"] == value_sha256(placements),
            f"{condition}: placement digest mismatch")
    seen: set[str] = set()
    distinct_paths: dict[str, str] = {}
    for index, item in enumerate(placements):
        require(isinstance(item, dict), f"{condition}[{index}] is not an object")
        exact_keys(item, {"entry_id", "topic_path"}, f"{condition}[{index}]")
        require(item["entry_id"] in entry_ids,
                f"{condition}[{index}] unknown entry")
        require(item["entry_id"] not in seen,
                f"{condition}[{index}] duplicate entry")
        seen.add(item["entry_id"])
        validate_topic_path(item["topic_path"], f"{condition}[{index}]")
        key = collision_key(item["topic_path"])
        previous = distinct_paths.get(key)
        require(previous is None or previous == item["topic_path"],
                f"{condition}: colliding topic paths {previous} / "
                f"{item['topic_path']}")
        distinct_paths[key] = item["topic_path"]
    require(seen == entry_ids, f"{condition}: placement is not a bank bijection")
    require(payload["metrics"] == placement_metrics(placements),
            f"{condition}: placement metrics mismatch")
    return placements


def reconstruct_condition(
    entries: list[dict[str, Any]], placements: list[dict[str, str]],
    v8: Any,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="r207-audit-") as raw_tmp:
        target = Path(raw_tmp) / "condition"
        target.mkdir()
        by_id = {entry["entry_id"]: entry for entry in entries}
        for placement in placements:
            entry = by_id[placement["entry_id"]]
            event = {
                "when": entry["when"],
                "summary": entry["summary"],
                "summary_inline": entry["summary_inline"],
                "dia_ids": entry["dia_ids"],
                "topic": placement["topic_path"],
            }
            v8.write_events(str(target), [event])
        removed = int(v8.dedup_topic_files(str(target)) or 0)
        descriptor = tree_descriptor(target)
        descriptor["exact_duplicate_lines_removed"] = removed
        return descriptor


def read_ledger_independent(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    previous = ZERO_HASH
    for index, record in enumerate(records, start=1):
        require(record.get("schema") == LEDGER_SCHEMA,
                f"ledger schema mismatch at line {index}")
        require(record.get("sequence") == index,
                f"ledger sequence mismatch at line {index}")
        require(record.get("previous_event_sha256") == previous,
                f"ledger previous hash mismatch at line {index}")
        content = dict(record)
        recorded = content.pop("event_sha256", None)
        expected = value_sha256(content)
        require(recorded == expected, f"ledger hash mismatch at line {index}")
        previous = expected
    return records


def normalize_usage(value: Any) -> dict[str, int | None]:
    value = value if isinstance(value, dict) else {}
    output: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = value.get(name)
        try:
            parsed = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            parsed = None
        output[name] = parsed if parsed is None or parsed >= 0 else None
    return output


def cost_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    starts = [item for item in values if item.get("event") == "model_call_started"]
    terminals = [
        item for item in values
        if item.get("event") in {"model_call_finished", "model_call_failed"}
    ]
    finished = [
        item for item in terminals if item.get("event") == "model_call_finished"
    ]
    return {
        "logical_model_calls": len(starts),
        "successful_model_calls": len(finished),
        "failed_model_calls": len(terminals) - len(finished),
        "client_http_attempts": sum(
            int(item.get("proxy_evidence", {}).get("client_http_attempts") or 0)
            for item in terminals
            if isinstance(item.get("proxy_evidence"), dict)
        ),
        "physical_upstream_attempts": sum(
            int(item.get("proxy_evidence", {}).get("upstream_http_attempts") or 0)
            for item in terminals
            if isinstance(item.get("proxy_evidence"), dict)
        ),
        "local_visible_tokens": sum(
            int(item.get("local_visible_tokens") or 0) for item in starts
        ),
        "provider_tokens": {
            name: sum(
                int(item.get("usage", {}).get(name) or 0)
                for item in finished
                if isinstance(item.get("usage"), dict)
            )
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "model_latency_s": round(sum(
            float(item.get("latency_s") or 0.0) for item in terminals
        ), 6),
        "unsupported_parameters": sorted({
            str(parameter)
            for item in terminals
            if isinstance(item.get("proxy_evidence"), dict)
            for parameter in item["proxy_evidence"].get(
                "unsupported_parameters", []
            )
        }),
    }


def token_count(text: str, identity: dict[str, Any], mode: str) -> int:
    require(identity.get("provider_exact") is False,
            "local tokenizer claims provider-exact counts")
    if mode == "synthetic_sanity":
        require(identity.get("implementation") == "builtin_utf8_bytes"
                and identity.get("encoding_name") == "utf8_bytes_v1",
                "synthetic tokenizer identity mismatch")
        return len(text.encode("utf-8"))
    require(identity.get("implementation") == "tiktoken",
            "formal tokenizer is not tiktoken")
    require(identity.get("implementation_version") == "0.12.0",
            "formal tiktoken version is not frozen")
    require(importlib.metadata.version("tiktoken") == "0.12.0",
            "auditor tiktoken version is not 0.12.0")
    import tiktoken  # type: ignore[import-not-found]

    encoding = tiktoken.get_encoding(str(identity.get("encoding_name")))
    return len(encoding.encode(text))


def validate_request_artifact(
    sample_dir: Path,
    start: dict[str, Any],
    *,
    mode: str,
    expected_model: str,
) -> Path:
    logical_id = str(start.get("logical_call_id", ""))
    require(logical_id, "model-call start lacks logical_call_id")
    expected_name = hashlib.sha256(logical_id.encode()).hexdigest()[:32]
    path = safe_artifact_path(sample_dir, start.get("request_path"), logical_id)
    require(path.name == f"{expected_name}.request.json"
            and path.parent == sample_dir / "calls",
            f"unexpected request path for {logical_id}")
    require(file_sha256(path) == start.get("request_sha256"),
            f"request hash mismatch for {logical_id}")
    request = load_json(path)
    require(request.get("schema") == REQUEST_SCHEMA,
            f"request schema mismatch for {logical_id}")
    require(request.get("logical_call_id") == logical_id
            and request.get("operation_id") == start.get("operation_id"),
            f"request linkage mismatch for {logical_id}")
    require(request.get("requested_model") == expected_model
            and start.get("requested_model") == expected_model,
            f"requested model mismatch for {logical_id}")
    visible = request.get("model_visible_payload")
    require(isinstance(visible, dict), f"invalid visible payload for {logical_id}")
    visible_text = canonical_bytes(visible).decode("utf-8")
    require(request.get("model_visible_sha256") == hashlib.sha256(
        visible_text.encode("utf-8")
    ).hexdigest(), f"visible-payload hash mismatch for {logical_id}")
    identity = request.get("tokenizer")
    require(isinstance(identity, dict), f"tokenizer missing for {logical_id}")
    count = token_count(visible_text, identity, mode)
    require(request.get("local_visible_tokens") == count
            and start.get("local_visible_tokens") == count
            and start.get("tokenizer") == identity,
            f"local visible-token mismatch for {logical_id}")
    require(request.get("transport_headers") == {
        "X-Controlled-Logical-Call-ID": logical_id
    }, f"transport linkage header mismatch for {logical_id}")
    options = request.get("request_options")
    require(isinstance(options, dict) and options.get("model") == expected_model,
            f"request options mismatch for {logical_id}")
    return path


def validate_response_artifact(
    sample_dir: Path,
    terminal: dict[str, Any],
    *,
    expected_model: str,
    formal: bool,
) -> Path | None:
    logical_id = str(terminal["logical_call_id"])
    relative = terminal.get("response_path")
    if relative is None:
        require(terminal.get("event") == "model_call_failed",
                f"successful call lacks response for {logical_id}")
        require(terminal.get("response_sha256") is None,
                f"response hash without response for {logical_id}")
        return None
    path = safe_artifact_path(sample_dir, relative, logical_id)
    expected_name = hashlib.sha256(logical_id.encode()).hexdigest()[:32]
    require(path.name == f"{expected_name}.response.json"
            and path.parent == sample_dir / "calls",
            f"unexpected response path for {logical_id}")
    require(file_sha256(path) == terminal.get("response_sha256"),
            f"response hash mismatch for {logical_id}")
    payload = load_json(path)
    require(payload.get("schema") == RESPONSE_SCHEMA
            and payload.get("logical_call_id") == logical_id,
            f"response linkage mismatch for {logical_id}")
    response = payload.get("response")
    require(isinstance(response, dict), f"invalid response for {logical_id}")
    if terminal.get("event") == "model_call_finished":
        require(str(response.get("id") or "") == terminal.get("response_id"),
                f"response ID mismatch for {logical_id}")
        require(str(response.get("model") or "") == expected_model
                and terminal.get("response_model") == expected_model,
                f"response model mismatch for {logical_id}")
        require(normalize_usage(response.get("usage")) == terminal.get("usage"),
                f"provider usage mismatch for {logical_id}")
        if formal:
            meta = response.get("exclusive_proxy_meta")
            require(isinstance(meta, dict)
                    and meta.get("logical_call_id") == logical_id
                    and isinstance(meta.get("event_id"), str)
                    and meta.get("client_http_attempts") == 1
                    and isinstance(meta.get("upstream_http_attempts"), int)
                    and meta["upstream_http_attempts"] >= 1
                    and isinstance(meta.get("unsupported_parameters"), list)
                    and re.fullmatch(
                        r"[0-9a-f]{64}", str(meta.get("request_sha256", ""))
                    ), f"exclusive response metadata mismatch for {logical_id}")
    return path


def audit_ledger(
    sample_dir: Path,
    *,
    sample: int,
    run_fingerprint: str,
    mode: str,
    expected_model: str,
    selected_input_sha256: str,
    build_result: list[Any],
    distill_call_count: int,
    entry_count: int,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    records = read_ledger_independent(sample_dir / "operations.jsonl")
    require(records, f"sample {sample}: operation ledger is empty")
    run_id = f"{run_fingerprint}:sample-{sample}"
    require(all(record.get("run_id") == run_id for record in records),
            f"sample {sample}: ledger run_id mismatch")
    operation_id = f"sample-{sample:02d}/capture-build"
    metadata = {
        "sample": sample,
        "selected_input_sha256": selected_input_sha256,
        "capture_point": "distill_events return before write_events",
        "method": "NativeMem-v8.8+calendar",
    }
    operation_input = value_sha256({
        "kind": "capture_final_build", "metadata": metadata,
    })
    empty_tree = {"file_count": 0, "tree_sha256": value_sha256([])}
    active: dict[str, Any] | None = None
    commit: dict[str, Any] | None = None
    starts: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    for record in records:
        event = record.get("event")
        if event == "operation_started":
            require(active is None and commit is None,
                    f"sample {sample}: duplicate or nested operation")
            require(record.get("operation_id") == operation_id
                    and record.get("kind") == "capture_final_build"
                    and record.get("metadata") == metadata
                    and record.get("operation_input_sha256") == operation_input
                    and record.get("continuing_tree_before") == empty_tree,
                    f"sample {sample}: operation start contract mismatch")
            active = record
        elif event == "model_call_started":
            require(active is not None and commit is None
                    and record.get("operation_id") == operation_id,
                    f"sample {sample}: model call outside active operation")
            logical_id = record.get("logical_call_id")
            require(isinstance(logical_id, str) and logical_id not in starts,
                    f"sample {sample}: duplicate model-call start")
            starts[logical_id] = record
        elif event in {"model_call_finished", "model_call_failed"}:
            logical_id = record.get("logical_call_id")
            require(isinstance(logical_id, str) and logical_id in starts
                    and logical_id not in terminals
                    and active is not None
                    and record.get("operation_id") == operation_id,
                    f"sample {sample}: orphan model-call terminal")
            require(isinstance(record.get("latency_s"), (int, float))
                    and record["latency_s"] >= 0,
                    f"sample {sample}: invalid model latency")
            terminals[logical_id] = record
        elif event == "operation_committed":
            require(active is not None and commit is None
                    and record.get("operation_id") == operation_id,
                    f"sample {sample}: operation commit without start")
            require(record.get("kind") == active.get("kind")
                    and record.get("operation_input_sha256") == operation_input
                    and record.get("continuing_tree_before") == empty_tree,
                    f"sample {sample}: operation commit drift")
            require(set(starts) == set(terminals),
                    f"sample {sample}: operation committed with orphan calls")
            scoped = [
                item for item in records
                if active["sequence"] < item["sequence"] < record["sequence"]
            ]
            require(record.get("cost") == cost_summary(scoped),
                    f"sample {sample}: operation cost mismatch")
            require(isinstance(record.get("latency_s"), (int, float))
                    and record["latency_s"] >= 0,
                    f"sample {sample}: invalid operation latency")
            commit = record
            active = None
        elif event == "operation_failed":
            raise AuditFailure(
                f"sample {sample}: completed artifact has failed operation"
            )
        else:
            raise AuditFailure(f"sample {sample}: unknown ledger event {event}")
    require(active is None and commit is not None,
            f"sample {sample}: ledger ends with an orphan operation")
    require(set(starts) == set(terminals),
            f"sample {sample}: ledger has orphan model calls")
    require(commit.get("continuing_tree_after") == tree_descriptor(
        sample_dir / "captured_final_build"
    ), f"sample {sample}: committed build tree mismatch")
    require(commit.get("result") == {
        "build_result": build_result,
        "distill_call_count": distill_call_count,
        "entry_count": entry_count,
    }, f"sample {sample}: operation result mismatch")

    request_paths: set[Path] = set()
    response_paths: set[Path] = set()
    formal = mode == "locomo"
    for logical_id, start in starts.items():
        request_paths.add(validate_request_artifact(
            sample_dir, start, mode=mode, expected_model=expected_model
        ))
        terminal = terminals[logical_id]
        response = validate_response_artifact(
            sample_dir, terminal, expected_model=expected_model, formal=formal
        )
        if response is not None:
            response_paths.add(response)
    calls_dir = sample_dir / "calls"
    require(calls_dir.is_dir() and not calls_dir.is_symlink(),
            f"sample {sample}: calls directory is missing")
    actual_call_files = {path for path in calls_dir.iterdir() if path.is_file()}
    require(actual_call_files == request_paths | response_paths,
            f"sample {sample}: unlinked or missing call artifacts")
    require(all(path.is_file() for path in calls_dir.iterdir()),
            f"sample {sample}: calls directory has non-file artifacts")
    if formal:
        require(starts and all(
            terminal.get("event") == "model_call_finished"
            for terminal in terminals.values()
        ), f"sample {sample}: formal build lacks successful durable calls")
    else:
        require(not starts,
                f"sample {sample}: synthetic build contains model calls")
    return records, starts, terminals


def audit_model_calls(
    sample_dir: Path,
    *,
    sample: int,
    records: list[dict[str, Any]],
    starts: dict[str, dict[str, Any]],
    terminals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = load_json(sample_dir / "model_calls.json")
    exact_keys(payload, {
        "schema", "sample", "ledger_path", "ledger_sha256",
        "ledger_event_count", "call_count", "successful_call_count",
        "failed_call_count", "calls_sha256", "cost", "call_artifact_tree",
        "calls",
    }, "model calls")
    require(payload["schema"] == "nativemem.r207-model-calls.v2",
            f"sample {sample}: wrong model calls schema")
    require(payload["sample"] == sample
            and payload["ledger_path"] == "operations.jsonl"
            and payload["ledger_sha256"]
            == file_sha256(sample_dir / "operations.jsonl")
            and payload["ledger_event_count"] == len(records),
            f"sample {sample}: model-call ledger identity mismatch")
    expected_calls = []
    for logical_id, start in starts.items():
        terminal = terminals[logical_id]
        evidence = terminal.get("proxy_evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        expected_calls.append({
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
            "client_http_attempts": evidence.get("client_http_attempts", 0),
            "physical_upstream_attempts": evidence.get(
                "upstream_http_attempts", 0
            ),
            "unsupported_parameters": evidence.get(
                "unsupported_parameters", []
            ),
        })
    cost = cost_summary(records)
    require(payload["calls"] == expected_calls
            and payload["call_count"] == len(expected_calls)
            and payload["successful_call_count"]
            == cost["successful_model_calls"]
            and payload["failed_call_count"] == cost["failed_model_calls"]
            and payload["calls_sha256"] == value_sha256(expected_calls)
            and payload["cost"] == cost
            and payload["call_artifact_tree"]
            == tree_descriptor(sample_dir / "calls"),
            f"sample {sample}: model-call summary does not reconcile")
    return payload


def parse_proxy_log(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(),
            f"proxy log is not a regular file: {path}")
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), f"incomplete proxy log: {path}")
    output = []
    offset = 0
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditFailure(f"invalid proxy log {path}:{line_number}") from exc
        require(isinstance(event, dict), f"non-object proxy event: {path}")
        end = offset + len(line)
        output.append({"event": event, "start": offset, "end": end})
        offset = end
    return output


def validate_prefix(prefix: Any, log: Path, *, label: str) -> int:
    require(isinstance(prefix, dict), f"{label}: missing proxy prefix")
    require(Path(str(prefix.get("path"))).resolve() == log.resolve(),
            f"{label}: proxy log path mismatch")
    offset = prefix.get("byte_offset")
    require(isinstance(offset, int) and 0 <= offset <= log.stat().st_size,
            f"{label}: invalid proxy byte offset")
    raw = log.read_bytes()
    require(prefix.get("prefix_sha256")
            == hashlib.sha256(raw[:offset]).hexdigest(),
            f"{label}: proxy prefix hash mismatch")
    return offset


def audit_proxy(
    root: Path,
    *,
    mode: str,
    starts_by_sample: list[dict[str, dict[str, Any]]],
    terminals_by_sample: list[dict[str, dict[str, Any]]],
) -> dict[str, int]:
    if mode == "synthetic_sanity":
        require(not (root / "proxy").exists(),
                "synthetic artifact contains proxy files")
        return {
            "events": 0,
            "client_http_attempts": 0,
            "upstream_http_attempts": 0,
        }
    proxy_dir = root / "proxy"
    require(proxy_dir.is_dir() and not proxy_dir.is_symlink(),
            "formal artifact lacks proxy directory")
    logs = sorted(proxy_dir.glob("launch-*.jsonl"))
    ready = sorted(proxy_dir.glob("launch-*.ready.json"))
    require(logs and len(logs) == len(ready),
            "proxy launch files are incomplete")
    require({path.name for path in proxy_dir.iterdir()} == {
        *(path.name for path in logs), *(path.name for path in ready),
    }, "proxy directory has unknown artifacts")
    require({path.stem for path in logs} == {
        path.name.removesuffix(".ready.json") for path in ready
    }, "proxy launch log/ready mismatch")
    fingerprint = load_json(root / "run_manifest.json")["run_fingerprint"]
    gateway_contract = load_json(root / "run_manifest.json").get("config", {}).get(
        "gateway_contract"
    )
    try:
        validated_contract = flex_evidence.validate_recorded_contract(
            gateway_contract
        )
    except flex_evidence.EvidenceError as exc:
        raise AuditFailure(f"Flex gateway contract differs: {exc}") from exc
    parsed_by_path: dict[Path, list[dict[str, Any]]] = {}
    all_events: list[tuple[Path, dict[str, Any]]] = []
    flex_reports: list[dict[str, Any]] = []
    for log, ready_path in zip(logs, ready, strict=True):
        ready_value = load_json(ready_path)
        require(Path(str(ready_value.get("log"))).resolve() == log.resolve(),
                f"proxy ready/log mismatch: {log.name}")
        launch_name = log.stem
        require(ready_value.get("run_id") == f"{fingerprint}:{launch_name}"
                and ready_value.get("base_wrapper_sha256")
                == file_sha256(ROOT / "scripts/gpt55_run_proxy.py")
                and ready_value.get("controlled_wrapper_sha256")
                == file_sha256(ROOT / "scripts/controlled_gpt55_run_proxy.py"),
                f"proxy ready provenance mismatch: {log.name}")
        parsed = parse_proxy_log(log)
        try:
            flex_reports.append(
                flex_evidence.audit_window(
                    ready_value.get("provider_window"),
                    consumer_records=[item["event"] for item in parsed],
                )
            )
        except flex_evidence.EvidenceError as exc:
            raise AuditFailure(f"Flex provider window differs: {exc}") from exc
        require(
            ready_value.get("gateway_contract") == validated_contract,
            f"proxy gateway contract mismatch: {log.name}",
        )
        parsed_by_path[log.resolve()] = parsed
        all_events.extend((log.resolve(), item) for item in parsed)

    starts = {
        key: value for sample_calls in starts_by_sample
        for key, value in sample_calls.items()
    }
    terminals = {
        key: value for sample_calls in terminals_by_sample
        for key, value in sample_calls.items()
    }
    require(set(starts) == set(terminals), "formal proxy call-set mismatch")
    assigned: set[tuple[Path, int]] = set()
    for logical_id, start in starts.items():
        terminal = terminals[logical_id]
        evidence = terminal.get("proxy_evidence")
        require(isinstance(evidence, dict)
                and evidence.get("mode") == "exclusive_proxy",
                f"formal call lacks exclusive evidence: {logical_id}")
        start_prefix = start.get("proxy_log_start")
        end_prefix = evidence.get("log_prefix")
        require(isinstance(start_prefix, dict) and isinstance(end_prefix, dict),
                f"proxy cutoffs missing: {logical_id}")
        log = Path(str(start_prefix.get("path"))).resolve()
        require(log in parsed_by_path,
                f"call references unknown proxy log: {logical_id}")
        start_offset = validate_prefix(start_prefix, log, label=logical_id)
        end_offset = validate_prefix(end_prefix, log, label=logical_id)
        require(start_offset <= end_offset,
                f"proxy cutoffs reversed: {logical_id}")
        linked_items = [
            item for item in parsed_by_path[log]
            if item["event"].get("logical_call_id") == logical_id
        ]
        require(linked_items, f"proxy has no event for {logical_id}")
        require(all(
            item["start"] >= start_offset and item["end"] <= end_offset
            for item in linked_items
        ), f"proxy event falls outside durable cutoffs: {logical_id}")
        linked_events = [item["event"] for item in linked_items]
        require(evidence.get("events") == linked_events,
                f"embedded proxy evidence differs from log: {logical_id}")
        for item in linked_items:
            key = (log, item["start"])
            require(key not in assigned,
                    f"proxy event assigned twice: {logical_id}")
            assigned.add(key)
        require(all(
            event.get("requested_model") == "gpt-5.5"
            and event.get("client_http_attempts") == 1
            and isinstance(event.get("upstream_http_attempts"), int)
            and event["upstream_http_attempts"] >= 0
            and isinstance(event.get("unsupported_parameters"), list)
            and re.fullmatch(
                r"[0-9a-f]{64}", str(event.get("request_sha256", ""))
            )
            and re.fullmatch(
                r"[0-9a-f]{64}", str(event.get("response_sha256", ""))
            )
            and isinstance(event.get("gateway_request_id"), str)
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(event.get("gateway_request_sha256", "")),
            )
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(event.get("provider_request_sha256", "")),
            )
            for event in linked_events
        ), f"proxy attempt metadata is incomplete: {logical_id}")
        require(evidence.get("client_http_attempts") == len(linked_events),
                f"client attempt count mismatch: {logical_id}")
        require(evidence.get("upstream_http_attempts") == sum(
            event["upstream_http_attempts"] for event in linked_events
        ), f"upstream attempt count mismatch: {logical_id}")
        unsupported = sorted({
            str(value) for event in linked_events
            for value in event["unsupported_parameters"]
        })
        require(evidence.get("unsupported_parameters") == unsupported,
                f"unsupported-parameter mismatch: {logical_id}")
        successful = [
            event for event in linked_events if event.get("status") == "success"
        ]
        require(terminal.get("event") == "model_call_finished"
                and len(successful) == 1,
                f"completed call lacks one proxy success: {logical_id}")
        success = successful[0]
        require(success.get("response_id") == terminal.get("response_id")
                and success.get("actual_model")
                == terminal.get("response_model")
                and normalize_usage(success.get("usage"))
                == terminal.get("usage"),
                f"proxy/response identity mismatch: {logical_id}")
        sample_match = re.match(r"sample-(\d+)/", str(start.get("operation_id", "")))
        require(sample_match is not None,
                f"cannot locate response artifact: {logical_id}")
        response_path = safe_artifact_path(
            root / "samples" / f"sample-{int(sample_match.group(1))}",
            terminal.get("response_path"), logical_id,
        )
        response = load_json(response_path)["response"]
        meta = response.get("exclusive_proxy_meta")
        require(isinstance(meta, dict)
                and meta.get("event_id") == success.get("event_id")
                and meta.get("run_id") == success.get("run_id")
                and meta.get("logical_call_id") == logical_id
                and meta.get("request_sha256") == success.get("request_sha256")
                and meta.get("client_http_attempts") == 1
                and meta.get("upstream_http_attempts")
                == success.get("upstream_http_attempts")
                and meta.get("unsupported_parameters")
                == success.get("unsupported_parameters"),
                f"response/proxy metadata mismatch: {logical_id}")
    require(len(assigned) == len(all_events), "unassigned proxy events exist")
    return {
        "events": len(all_events),
        "client_http_attempts": sum(
            int(item["event"]["client_http_attempts"]) for _, item in all_events
        ),
        "upstream_http_attempts": sum(
            int(item["event"]["upstream_http_attempts"])
            for _, item in all_events
        ),
        "flex_gateway_windows": flex_reports,
    }


def audit_sample(
    sample_dir: Path,
    sample: int,
    sample_id: str,
    root_manifest: dict[str, Any],
    root_sample_record: dict[str, Any],
    conv: dict[str, Any],
    question_mappings: list[dict[str, Any]],
    v8: Any,
) -> dict[str, Any]:
    expected_direct = {
        "canonical_entry_bank.json", "original_placement.json",
        "capture_calls.json", "model_calls.json", "operations.jsonl", "calls",
        "question_traces.jsonl",
        "placements", "conditions", "captured_final_build",
        "sample_manifest.json",
    }
    actual_direct = {path.name for path in sample_dir.iterdir()}
    require(actual_direct == expected_direct,
            f"sample {sample}: stale/missing direct artifacts "
            f"{sorted(actual_direct ^ expected_direct)}")
    sample_manifest_path = sample_dir / "sample_manifest.json"
    require(file_sha256(sample_manifest_path)
            == root_sample_record["sample_manifest_sha256"],
            f"sample {sample}: sample manifest hash mismatch")
    manifest = load_json(sample_manifest_path)
    require(manifest.get("schema") == SAMPLE_SCHEMA,
            f"sample {sample}: wrong sample manifest schema")
    require(manifest.get("status") == "complete",
            f"sample {sample}: sample manifest is not complete")
    require(manifest.get("sample") == sample,
            f"sample {sample}: sample manifest ID mismatch")
    require(manifest.get("sample_id") == sample_id,
            f"sample {sample}: sample identity mismatch")
    require(manifest.get("run_fingerprint") == root_manifest["run_fingerprint"],
            f"sample {sample}: run fingerprint mismatch")
    require(manifest.get("selected_input_sha256") == value_sha256(conv),
            f"sample {sample}: selected input hash mismatch")
    require(manifest.get("core_source_hashes")
            == root_manifest["core_source_hashes"],
            f"sample {sample}: source hashes differ from root")
    require(manifest.get("model_config") == root_manifest["config"]["model_config"],
            f"sample {sample}: model config differs from root")
    expected_sources = dataset_source_ids(conv)
    require(manifest.get("expected_source_count") == len(expected_sources),
            f"sample {sample}: expected source count mismatch")

    artifact_hashes = manifest.get("artifact_sha256")
    require(isinstance(artifact_hashes, dict),
            f"sample {sample}: missing artifact hashes")
    expected_hash_paths = {
        "canonical_entry_bank.json", "original_placement.json",
        "capture_calls.json", "model_calls.json", "operations.jsonl",
        "question_traces.jsonl",
        "placements/model_directed.json",
        "placements/deterministic_permutation.json",
    }
    require(set(artifact_hashes) == expected_hash_paths,
            f"sample {sample}: artifact hash inventory mismatch")
    for relative, expected_hash in artifact_hashes.items():
        artifact_path = sample_dir / relative
        require(artifact_path.is_file() and not artifact_path.is_symlink(),
                f"sample {sample}: artifact is missing: {relative}")
        require(file_sha256(artifact_path) == expected_hash,
                f"sample {sample}: artifact changed: {relative}")

    bank = load_json(sample_dir / "canonical_entry_bank.json")
    entries, entry_ids = audit_bank(bank, sample, expected_sources)
    build_result = manifest.get("build_result")
    require(isinstance(build_result, list) and len(build_result) >= 2
            and int(build_result[1]) == len(entries),
            f"sample {sample}: adapter build event count differs from bank")
    require(manifest["canonical_bank"] == {
        "entry_count": bank["entry_count"],
        "entries_sha256": bank["entries_sha256"],
        "source_identity_sha256": bank["source_identity_sha256"],
    }, f"sample {sample}: manifest bank identity mismatch")
    require(root_sample_record["entry_count"] == len(entries),
            f"sample {sample}: root entry count mismatch")

    capture = load_json(sample_dir / "capture_calls.json")
    audit_capture(capture, sample, entries)
    mode = root_manifest["config"]["data"]["mode"]
    expected_model = root_manifest["config"]["model_config"]["requested_model"]
    records, starts, terminals = audit_ledger(
        sample_dir,
        sample=sample,
        run_fingerprint=root_manifest["run_fingerprint"],
        mode=mode,
        expected_model=expected_model,
        selected_input_sha256=value_sha256(conv),
        build_result=build_result,
        distill_call_count=capture["call_count"],
        entry_count=len(entries),
    )
    model_calls = audit_model_calls(
        sample_dir,
        sample=sample,
        records=records,
        starts=starts,
        terminals=terminals,
    )
    require(manifest.get("model_ledger") == {
        "schema": LEDGER_SCHEMA,
        "ledger_sha256": file_sha256(sample_dir / "operations.jsonl"),
        "ledger_event_count": len(records),
        "cost": model_calls["cost"],
        "call_artifact_tree": model_calls["call_artifact_tree"],
    }, f"sample {sample}: manifest model ledger does not reconcile")

    max_depth = int(root_manifest["config"]["model_config"]["environment"]
                    ["NATIVEMEM_V8_MAX_DEPTH"])
    original = audit_original_placement(
        load_json(sample_dir / "original_placement.json"), sample,
        bank["entries_sha256"], entry_ids, max_depth,
    )
    placement_dir = sample_dir / "placements"
    require({path.name for path in placement_dir.iterdir()} == {
        "model_directed.json", "deterministic_permutation.json"
    }, f"sample {sample}: placement inventory mismatch")
    payloads = {
        condition: load_json(placement_dir / f"{condition}.json")
        for condition in CONDITIONS
    }
    placements = {
        condition: audit_placement(
            payloads[condition], condition, sample, bank["entries_sha256"],
            entry_ids,
        )
        for condition in CONDITIONS
    }
    expected_model = [
        {"entry_id": item["entry_id"], "topic_path": item["topic_path"]}
        for item in original
    ]
    require(placements["model_directed"] == expected_model,
            f"sample {sample}: model placement differs from original sidecar")
    expected_control, expected_meta = expected_permutation(expected_model)
    require(placements["deterministic_permutation"] == expected_control,
            f"sample {sample}: deterministic permutation differs")
    require(payloads["deterministic_permutation"]["permutation"] == expected_meta,
            f"sample {sample}: permutation metadata differs")
    require(Counter(item["topic_path"] for item in expected_model)
            == Counter(item["topic_path"] for item in expected_control),
            f"sample {sample}: path multiset differs")
    require(payloads["model_directed"]["metrics"]
            == payloads["deterministic_permutation"]["metrics"],
            f"sample {sample}: matched path metrics differ")
    match = manifest.get("path_match", {})
    require(match.get("residual_path_string_bytes") == 0,
            f"sample {sample}: path byte residual is not zero")
    require(match.get("permutation") == expected_meta,
            f"sample {sample}: manifest permutation differs")

    condition_dir = sample_dir / "conditions"
    require({path.name for path in condition_dir.iterdir()} == set(CONDITIONS),
            f"sample {sample}: condition inventory mismatch")
    manifest_trees = manifest.get("tree_hashes", {})
    for condition in CONDITIONS:
        actual_tree = tree_descriptor(condition_dir / condition)
        expected_tree = dict(manifest_trees["conditions"][condition])
        removed = expected_tree.pop("exact_duplicate_lines_removed")
        require(actual_tree == expected_tree,
                f"sample {sample}: {condition} tree hash mismatch")
        reconstructed = reconstruct_condition(entries, placements[condition], v8)
        require(reconstructed == manifest_trees["conditions"][condition],
                f"sample {sample}: {condition} tree does not reconstruct")
        require(reconstructed["exact_duplicate_lines_removed"] == removed,
                f"sample {sample}: {condition} dedup count differs")
        for path in (condition_dir / condition).rglob("*"):
            if path.is_file():
                require(path.suffix == ".md",
                        f"sample {sample}: non-Markdown condition artifact")
    require(tree_descriptor(sample_dir / "captured_final_build")
            == manifest_trees["captured_final_build"],
            f"sample {sample}: captured final build tree mismatch")
    traces = load_jsonl(sample_dir / "question_traces.jsonl")
    expected_traces = expected_trace_records(
        sample=sample,
        sample_id=sample_id,
        mappings=question_mappings,
        entries=entries,
        sample_dir=sample_dir,
    )
    require(traces == expected_traces,
            f"sample {sample}: question trace evidence differs")
    require(manifest.get("question_traces") == {
        "schema": TRACE_SCHEMA,
        "trace_count": len(traces),
        "trace_ids_sha256": value_sha256([
            record["trace_id"] for record in traces
        ]),
        "artifact_sha256": file_sha256(
            sample_dir / "question_traces.jsonl"
        ),
        "unobserved_stages": ["retrieval_reach", "source_resolution"],
    }, f"sample {sample}: question trace summary differs")
    return {
        "sample": sample,
        "entries": len(entries),
        "distill_calls": capture["call_count"],
        "model_calls": model_calls["call_count"],
        "cost": model_calls["cost"],
        "ledger_events": len(records),
        "question_traces": len(traces),
        "starts": starts,
        "terminals": terminals,
        "fixed_point_rate": expected_meta["fixed_point_rate"],
        "condition_tree_sha256": {
            condition: manifest_trees["conditions"][condition]["tree_sha256"]
            for condition in CONDITIONS
        },
    }


def audit(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    require(artifact_dir != ROOT and ROOT in artifact_dir.parents,
            "artifact directory must be under the repository root")
    assert_safe_tree(artifact_dir)
    manifest = load_json(artifact_dir / "run_manifest.json")
    require(manifest.get("schema") == RUN_SCHEMA, "wrong run schema")
    require(manifest.get("status") == "complete", "run is not complete")
    requested = manifest.get("requested_samples")
    require(isinstance(requested, list) and requested
            and requested == sorted(set(requested)),
            "requested sample list is invalid")
    config = manifest.get("config")
    require(isinstance(config, dict), "run config is missing")
    data = config.get("data")
    require(isinstance(data, dict), "data descriptor is missing")
    mode = data.get("mode")
    require(mode in {"synthetic_sanity", "locomo"}, "unsupported data mode")
    if mode == "synthetic_sanity":
        require(requested == [0], "synthetic run must contain sample zero only")
        expected_root = {"run_manifest.json", "samples"}
    else:
        require(requested == list(range(10)),
                "formal R207 must contain all samples 0-9")
        expected_root = {"run_manifest.json", "samples", "proxy"}
    require({path.name for path in artifact_dir.iterdir()} == expected_root,
            "root has stale, missing, or unknown artifacts")

    model_config = config.get("model_config")
    require(isinstance(model_config, dict), "model config is missing")
    concurrency = model_config.get("request_concurrency")
    require(isinstance(concurrency, int) and concurrency >= 1,
            "request concurrency is invalid")
    if mode == "locomo":
        require(concurrency == 10, "formal request concurrency is not ten")
        try:
            gateway_contract = flex_evidence.validate_recorded_contract(
                config.get("gateway_contract")
            )
        except flex_evidence.EvidenceError as exc:
            raise AuditFailure(f"Flex gateway contract differs: {exc}") from exc
    else:
        require(config.get("gateway_contract") is None,
                "synthetic gateway contract must be absent")
        gateway_contract = None
    requested_model = (
        "synthetic-no-model" if mode == "synthetic_sanity" else "gpt-5.5"
    )
    upstream = model_config.get("upstream_base_url")
    require((mode == "synthetic_sanity" and upstream is None)
            or (mode == "locomo" and upstream == gateway_contract["origin"]),
            "upstream URL contract differs")
    expected_model_config = {
        "method": "NativeMem-v8.8+calendar",
        "calendar_context": (
            "deterministic calendar_strip from final v8.8 source"
        ),
        "requested_model": requested_model,
        "transport": (
            "synthetic_no_transport" if mode == "synthetic_sanity"
            else "managed_exclusive_proxy"
        ),
        "upstream_base_url": upstream,
        "api_key_recorded": False,
        "api_key_configured": model_config.get("api_key_configured"),
        "request_concurrency": concurrency,
        "environment": {
            **FINAL_ENV,
            "NATIVEMEM_V8_CONCURRENCY": str(concurrency),
        },
        "forbidden_environment_absent": [
            "MODEL", "NATIVEMEM_V9_PIPELINE", "NATIVEMEM_V9_SCRIBE_MODE"
        ],
    }
    require(isinstance(model_config.get("api_key_configured"), bool)
            and model_config == expected_model_config,
            "model config drift")
    question_mapping = config.get("question_mapping")
    if mode == "synthetic_sanity":
        expected_mapping = {
            "mode": "synthetic_no_questions",
            "sha256": value_sha256([]),
            "record_count": 0,
        }
    else:
        expected_mapping = {
            "mode": "frozen_evidence_mapping",
            "path": EVIDENCE_QUESTIONS.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_EVIDENCE_SHA256,
            "record_count": 1540,
        }
    require(question_mapping == expected_mapping,
            "question mapping descriptor differs")
    expected_config = {
        "samples": requested,
        "data": data,
        "question_mapping": expected_mapping,
        "model_config": model_config,
        "capture_point": "distill_events return before write_events",
        "canonical_bank_topics_removed": True,
        "condition_maintenance": "off",
        "materialization": "write_events + dedup_topic_files",
        "gateway_contract": gateway_contract,
    }
    require(config == expected_config, "run config drift")

    sources = manifest.get("core_source_hashes")
    require(isinstance(sources, dict) and set(sources) == set(CORE_SOURCES),
            "core source inventory differs")
    for relative, expected in sources.items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(),
                f"core source missing or symlinked: {relative}")
        require(file_sha256(path) == expected,
                f"core source changed: {relative}")
        frozen_expected = FROZEN_SOURCE_HASHES.get(relative)
        require(frozen_expected is None or expected == frozen_expected,
                f"recorded frozen hash is not the preregistered hash: {relative}")
    expected_fingerprint = value_sha256({
        "schema": RUN_SCHEMA,
        "config": manifest.get("config"),
        "core_source_hashes": sources,
    })
    require(manifest.get("run_fingerprint") == expected_fingerprint,
            "run fingerprint mismatch")
    observations = manifest.get("environment_observations")
    require(isinstance(observations, dict)
            and set(observations).issubset(set(OBSERVED_BENCHMARK_RUNNERS))
            and all(re.fullmatch(r"[0-9a-f]{64}", str(value))
                    for value in observations.values()),
            "environment observation inventory is invalid")

    if mode == "synthetic_sanity":
        dataset = [
            {
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
        ]
        require(data == {
            "mode": "synthetic_sanity", "sha256": value_sha256(dataset),
        },
                "synthetic dataset identity mismatch")
        conversations = dataset
        sample_ids = ["synthetic-sample-0"]
        question_mappings = {0: []}
    else:
        expected_relative = "benchmarks/locomo/data/locomo10.json"
        require(data == {
            "mode": "locomo",
            "path": expected_relative,
            "sha256": EXPECTED_DATA_SHA256,
        }, "formal dataset descriptor mismatch")
        data_path = ROOT / expected_relative
        require(data_path.is_file() and not data_path.is_symlink()
                and file_sha256(data_path) == EXPECTED_DATA_SHA256,
                "frozen LoCoMo dataset hash mismatch")
        dataset = load_json(data_path)
        require(all(
            isinstance(item, dict)
            and isinstance(item.get("sample_id"), str)
            for item in dataset
        ), "formal dataset sample identity is invalid")
        conversations = [conversation_payload(item) for item in dataset]
        sample_ids = [str(item["sample_id"]) for item in dataset]
        question_mappings = load_formal_question_mappings()
    require(isinstance(dataset, list) and len(dataset) > max(requested),
            "dataset root or sample coverage is invalid")

    environment = model_config["environment"]
    os.environ["NATIVEMEM_V8_MAX_DEPTH"] = environment["NATIVEMEM_V8_MAX_DEPTH"]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    v8 = importlib.import_module("src.v8_memory")

    root_samples = manifest.get("samples")
    require(isinstance(root_samples, dict)
            and set(root_samples) == {str(sample) for sample in requested},
            "root sample registry differs from requested samples")
    sample_root = artifact_dir / "samples"
    require(sample_root.is_dir() and not sample_root.is_symlink(),
            "sample root is unsafe")
    require({path.name for path in sample_root.iterdir()}
            == {f"sample-{sample}" for sample in requested},
            "sample directory inventory differs")
    reports = []
    starts_by_sample = []
    terminals_by_sample = []
    for sample in requested:
        record = root_samples[str(sample)]
        require(record.get("status") == "complete",
                f"sample {sample} is not complete in the root manifest")
        require(sample < len(dataset), f"sample {sample} is outside the dataset")
        sample_report = audit_sample(
            sample_root / f"sample-{sample}", sample, sample_ids[sample],
            manifest, record, conversations[sample], question_mappings[sample],
            v8,
        )
        starts_by_sample.append(sample_report.pop("starts"))
        terminals_by_sample.append(sample_report.pop("terminals"))
        reports.append(sample_report)
    proxy_report = audit_proxy(
        artifact_dir,
        mode=mode,
        starts_by_sample=starts_by_sample,
        terminals_by_sample=terminals_by_sample,
    )
    return {
        "schema": "nativemem.r207-audit.v1",
        "status": "pass",
        "artifact_dir": str(artifact_dir),
        "run_fingerprint": manifest["run_fingerprint"],
        "sample_count": len(reports),
        "entry_count": sum(report["entries"] for report in reports),
        "question_trace_count": sum(
            report["question_traces"] for report in reports
        ),
        "model_call_count": sum(report["model_calls"] for report in reports),
        "failed_model_calls": sum(
            report["cost"]["failed_model_calls"] for report in reports
        ),
        "proxy": proxy_report,
        "samples": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.artifact_dir)
    except AuditFailure as exc:
        failure = {
            "schema": "nativemem.r207-audit.v1",
            "status": "fail",
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True))
        return 2
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.report:
        report_path = args.report.expanduser().resolve()
        require(ROOT in report_path.parents,
                "audit report path must be under the repository")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
