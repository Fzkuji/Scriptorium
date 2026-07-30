#!/usr/bin/env python3
"""Independently audit R403 chronological-growth artifacts.

The auditor intentionally does not import the runner.  It recomputes frozen
labels, operation scheduling, hash chains, model-call linkage, checkpoint
inventories, and source reachability from preserved inputs and artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import stat
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src import openai_gpt55_flex_gateway_evidence as flex_evidence


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
EVIDENCE_QUESTIONS = (
    ROOT / "results" / "paper-experiments-20260714" / "evidence-mapping"
    / "v1" / "evidence_mapping.v1.questions.jsonl"
)
RUN_SCHEMA = "nativemem.r403-growth-run/v1"
SAMPLE_SCHEMA = "nativemem.r403-growth-sample/v1"
LABEL_SCHEMA = "nativemem.r403-frozen-labels/v1"
CHECKPOINT_SCHEMA = "nativemem.r403-checkpoint/v1"
INVENTORY_SCHEMA = "nativemem.r403-question-inventory/v2"
BINDING_SCHEMA = "nativemem.r403-task-score-binding/v1"
LEDGER_SCHEMA = "durable-model-ledger/v1"
REQUEST_SCHEMA = "durable-model-request/v1"
RESPONSE_SCHEMA = "durable-model-response/v1"
ZERO_HASH = "0" * 64
CHECKPOINTS = (10, 25, 50, 100)

EXPECTED_INPUT_HASHES = {
    "benchmarks/locomo/data/locomo10.json": (
        "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    ),
    "results/paper-experiments-20260714/evidence-mapping/v1/"
    "evidence_mapping.v1.manifest.json": (
        "64fa89b05bc2ad075ad046fb514b1cded164cba211f5b1d246719ac67916b735"
    ),
    "results/paper-experiments-20260714/evidence-mapping/v1/"
    "evidence_mapping.v1.questions.jsonl": (
        "3291df579bc20d0678f99b3361ac6d287beee583e4d3dff93b996f4522e56318"
    ),
    "results/paper-experiments-20260714/evidence-mapping/v1/"
    "evidence_mapping.v1.audit.json": (
        "cb5800a3af5b9e9afb14e25ed036c34a107ce8727ff1b6d5c28641462ddb84d8"
    ),
}
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
SOURCE_FILES = (
    *FROZEN_SOURCE_HASHES,
    "src/evaluation/durable_model_ledger.py",
    "src/evaluation/visible_token_budget.py",
    "scripts/gpt55_run_proxy.py",
    "scripts/controlled_gpt55_run_proxy.py",
    "src/openai_gpt55_flex_gateway.py",
    "src/openai_gpt55_flex_gateway_evidence.py",
    "scripts/run_r403_incremental_growth.py",
    "scripts/audit_r403_incremental_growth.py",
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
    "NATIVEMEM_V8_CONCURRENCY": "10",
}
UPDATE_PATTERNS = (
    r"\bnow\b", r"\bcurrently\b", r"\blatest\b", r"most recent",
    r"more recently", r"\bchanged\b", r"\bno longer\b", r"\binstead\b",
    r"over time", r"new .* change",
)


class AuditFailure(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"invalid JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing JSONL: {path}")
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), f"incomplete JSONL: {path}")
    records = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditFailure(f"invalid JSONL {path}:{line_number}") from exc
        require(isinstance(value, dict), f"non-object JSONL {path}:{line_number}")
        records.append(value)
    return records


def collision_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def assert_safe_tree(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"unsafe root: {root}")
    paths: dict[str, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        key = collision_key(relative)
        require(key not in paths or paths[key] == relative,
                f"path collision: {paths.get(key)} / {relative}")
        paths[key] = relative
        require(not path.is_symlink(), f"symlink is forbidden: {relative}")
        info = path.stat(follow_symlinks=False)
        if stat.S_ISREG(info.st_mode):
            require(info.st_nlink == 1, f"hardlink is forbidden: {relative}")
            inode = (info.st_dev, info.st_ino)
            require(inode not in inodes,
                    f"shared inode: {inodes.get(inode)} / {relative}")
            inodes[inode] = relative
        else:
            require(stat.S_ISDIR(info.st_mode),
                    f"special node is forbidden: {relative}")


def tree_descriptor(
    root: Path, *, exclude_relative: frozenset[str] = frozenset()
) -> dict[str, Any]:
    assert_safe_tree(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude_relative:
            continue
        files.append({
            "path": relative,
            "size": path.stat(follow_symlinks=False).st_size,
            "sha256": file_sha256(path),
        })
    return {
        "file_count": len(files),
        "byte_count": sum(item["size"] for item in files),
        "tree_sha256": value_sha256(files),
    }


def safe_artifact_path(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label}: invalid path")
    candidate = Path(relative)
    require(not candidate.is_absolute() and ".." not in candidate.parts,
            f"{label}: unsafe relative path")
    resolved = root / candidate
    require(resolved.is_file() and not resolved.is_symlink(),
            f"{label}: file does not exist")
    return resolved


def session_count(conv: dict[str, Any]) -> int:
    count = 0
    while f"session_{count + 1}" in conv:
        count += 1
    return count


def checkpoint_boundaries(total: int) -> dict[str, int]:
    require(total > 0, "conversation has no sessions")
    return {str(percent): math.ceil(total * percent / 100)
            for percent in CHECKPOINTS}


def normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def dia_session(value: str) -> int:
    match = re.fullmatch(r"D(\d+):\d+", value)
    require(match is not None, f"invalid dia_id: {value}")
    return int(match.group(1))


def normalize_date(value: Any) -> str:
    """Independent equivalent of the frozen adapter's LoCoMo date normalizer."""
    text = str(value or "").strip()
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    # LoCoMo also uses English day-month-year strings.
    months = {
        name: index for index, name in enumerate(
            ("january", "february", "march", "april", "may", "june",
             "july", "august", "september", "october", "november",
             "december"),
            start=1,
        )
    }
    lowered = text.casefold().replace(",", " ")
    match = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", lowered)
    if match and match.group(2) in months:
        return f"{int(match.group(3)):04d}-{months[match.group(2)]:02d}-{int(match.group(1)):02d}"
    match = re.search(r"([a-z]+)\s+(\d{1,2})\s+(\d{4})", lowered)
    if match and match.group(1) in months:
        return f"{int(match.group(3)):04d}-{months[match.group(1)]:02d}-{int(match.group(2)):02d}"
    return text


def synthetic_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions = {
        f"session_{index}": [{
            "speaker": "A", "dia_id": f"D{index}:1", "text": f"fact {index}",
        }]
        for index in range(1, 11)
    }
    dates = {f"session_{index}_date_time": f"2023-05-{index:02d}"
             for index in range(1, 11)}
    sample = {
        "sample_id": "synthetic-growth",
        "conversation": {**sessions, **dates},
        "qa": [
            {"question": "What was fact one?", "answer": "fact 1",
             "evidence": ["D1:1"], "category": 1},
            {"question": "What changed over time?", "answer": "fact 5",
             "evidence": ["D1:1", "D5:1"], "category": 1},
            {"question": "Which final fact appeared?", "answer": "fact 10",
             "evidence": ["D10:1"], "category": 2},
            {"question": "Unmapped question?", "answer": "unknown",
             "evidence": [], "category": 3},
        ],
    }
    evidence = []
    for index, qa in enumerate(sample["qa"]):
        ids = list(qa["evidence"])
        eligible = bool(ids)
        evidence.append({
            "benchmark": "LoCoMo", "qa_scoring_eligible": True,
            "source_recall_eligible": eligible,
            "source_recall_exclusion_reasons": (
                [] if eligible else ["gold_evidence_empty"]
            ),
            "normalized_source_ids": ids, "sample_index": 0,
            "sample_id": sample["sample_id"], "question_index": index,
            "question_id": f"locomo:synthetic:q{index:03d}",
        })
    return [sample], evidence


def load_formal_evidence() -> list[dict[str, Any]]:
    records = [record for record in load_jsonl(EVIDENCE_QUESTIONS)
               if record.get("benchmark") == "LoCoMo"
               and record.get("qa_scoring_eligible") is True]
    require(len(records) == 1540, "formal primary evidence count is not 1540")
    require(sum(not record.get("source_recall_eligible") for record in records) == 7,
            "formal source-exclusion count is not seven")
    return records


def freeze_labels_independent(
    dataset: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    *,
    dataset_sha256: str,
    source_mapping_sha256: str,
) -> dict[str, Any]:
    evidence = {(int(item["sample_index"]), int(item["question_index"])): item
                for item in evidence_records}
    records = []
    for sample_index, sample in enumerate(dataset):
        conv = sample["conversation"]
        boundaries = checkpoint_boundaries(session_count(conv))
        turns: dict[str, dict[str, Any]] = {}
        for session in range(1, session_count(conv) + 1):
            for turn in conv[f"session_{session}"]:
                turns[str(turn["dia_id"])] = {
                    "session": session, "text": str(turn.get("text", "")),
                }
        for question_index, qa in enumerate(sample["qa"]):
            if int(qa.get("category", 0)) == 5:
                continue
            mapping = evidence.get((sample_index, question_index))
            require(mapping is not None, "primary question lacks evidence mapping")
            require(mapping.get("sample_id") == sample["sample_id"],
                    "evidence mapping sample identity mismatch")
            source_ids = [str(value) for value in mapping["normalized_source_ids"]]
            source_sessions = sorted({dia_session(value) for value in source_ids})
            source_eligible = bool(mapping["source_recall_eligible"])
            require(not source_eligible
                    or bool(source_ids) and set(source_ids).issubset(turns),
                    "source-eligible mapping has absent raw anchors")
            eligibility = {
                key: bool(source_eligible and source_ids
                          and max(source_sessions) <= boundary)
                for key, boundary in boundaries.items()
            }
            first_checkpoint = next(
                (int(key) for key in map(str, CHECKPOINTS) if eligibility[key]),
                None,
            )
            answer_norm = normalize_text(qa.get("answer", ""))
            answer_history = [
                {"dia_id": dia_id, "session": turn["session"],
                 "match": "normalized_substring"}
                for dia_id, turn in turns.items()
                if len(answer_norm) >= 2
                and answer_norm in normalize_text(turn["text"])
            ]
            question_text = str(qa["question"])
            update_cues = [pattern for pattern in UPDATE_PATTERNS
                           if re.search(pattern, question_text.casefold())]
            update_label = bool(
                source_eligible and len(source_sessions) >= 2 and update_cues
            )
            previous = None
            old_fact: dict[str, bool] = {}
            for percent in CHECKPOINTS:
                key = str(percent)
                old_fact[key] = bool(
                    previous is not None and eligibility[str(previous)]
                )
                previous = percent
            records.append({
                "sample_index": sample_index, "sample_id": sample["sample_id"],
                "question_index": question_index,
                "question_id": mapping["question_id"],
                "category": int(qa["category"]),
                "question_sha256": value_sha256(str(qa["question"])),
                "gold_answer_sha256": value_sha256(qa.get("answer", "")),
                "source_recall_eligible": source_eligible,
                "source_exclusion_reasons": mapping[
                    "source_recall_exclusion_reasons"
                ],
                "normalized_source_ids": source_ids,
                "source_sessions": source_sessions,
                "eligibility_by_checkpoint": eligibility,
                "first_eligible_checkpoint": first_checkpoint,
                "old_fact_by_checkpoint": old_fact,
                "update_label": update_label,
                "update_first_checkpoint": first_checkpoint if update_label else None,
                "update_cues": update_cues,
                "answer_history": answer_history,
                "label_policy": {
                    "old_fact": (
                        "all mapped final-gold anchors were already present at "
                        "the immediately preceding checkpoint"
                    ),
                    "update": (
                        "complete multi-session gold anchors plus an explicit "
                        "pre-registered update cue in the raw question"
                    ),
                    "answer_history": (
                        "normalized gold-answer substring in raw turns"
                    ),
                },
            })
    question_ids = [record["question_id"] for record in records]
    require(len(question_ids) == len(set(question_ids)),
            "frozen labels contain duplicate question IDs")
    return {
        "schema": LABEL_SCHEMA,
        "checkpoint_unit": "chronological session prefix",
        "checkpoint_boundary_rule": "ceil(total_sessions * percent / 100)",
        "checkpoints": list(CHECKPOINTS),
        "source_mapping_sha256": source_mapping_sha256,
        "dataset_sha256": dataset_sha256,
        "record_count": len(records),
        "source_exclusion_count": sum(
            not record["source_recall_eligible"] for record in records
        ),
        "update_label_count": sum(record["update_label"] for record in records),
        "records_sha256": value_sha256(records),
        "records": records,
    }


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
            name: sum(int(item.get("usage", {}).get(name) or 0)
                      for item in finished)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
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


def source_ids(memory_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in memory_dir.rglob("*.md"):
        values.update(re.findall(r"(?<![\w])D\d+:\d+(?!\w)",
                                 path.read_text(encoding="utf-8")))
    return values


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


def canonical_sources_through(
    ledger_records: list[dict[str, Any]], session_boundary: int
) -> set[str]:
    sources: set[str] = set()
    for record in ledger_records:
        if (record.get("event") != "operation_committed"
                or record.get("kind") != "fixed6_extract_write"):
            continue
        match = re.search(
            r"/session-(\d+)/segment-", str(record.get("operation_id", ""))
        )
        require(match is not None, "extraction operation ID is invalid")
        if int(match.group(1)) <= session_boundary:
            referenced = record.get("result", {}).get("referenced_dia_ids")
            require(isinstance(referenced, list)
                    and all(isinstance(value, str) for value in referenced),
                    "extraction operation source result is invalid")
            sources.update(referenced)
    return sources


def stable_trace_id(question_id: str, checkpoint: int) -> str:
    return f"r403:{question_id}:checkpoint-{checkpoint:03d}"


def expected_stage_evidence(
    label: dict[str, Any],
    *,
    checkpoint: int,
    canonical_sources: set[str],
    stored_sources: set[str],
    locations_by_source: dict[str, list[str]],
) -> dict[str, Any]:
    source_values = list(label["normalized_source_ids"])
    eligible = bool(label["source_recall_eligible"])
    present = sorted(set(source_values) & canonical_sources)
    missing = sorted(set(source_values) - canonical_sources)
    mapping = {
        "status": "observed",
        "value": eligible,
        "artifact": "frozen_labels.json",
        "question_id": label["question_id"],
        "source_ids": source_values,
        "source_exclusion_reasons": label["source_exclusion_reasons"],
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
            "artifact": "operations.jsonl",
            "present_source_ids": present,
            "missing_source_ids": missing,
        }
    if canonical["value"] is True:
        missing_after = sorted(set(source_values) - stored_sources)
        maintenance = {
            "status": "observed",
            "value": not missing_after,
            "artifact": f"checkpoints/checkpoint-{checkpoint:03d}/memory",
            "present_source_ids": sorted(set(source_values) & stored_sources),
            "missing_source_ids": missing_after,
        }
    else:
        maintenance = {
            "status": "not_applicable",
            "value": None,
            "reason": "canonical_source_stage_not_complete",
            "present_source_ids": [],
            "missing_source_ids": [],
        }
    if maintenance["value"] is True:
        locations = {
            source_id: locations_by_source.get(source_id, [])
            for source_id in source_values
        }
        path_stage = {
            "status": "observed",
            "value": all(locations[source_id] for source_id in source_values),
            "artifact": f"checkpoints/checkpoint-{checkpoint:03d}/memory",
            "source_paths": locations,
        }
    else:
        path_stage = {
            "status": "not_applicable",
            "value": None,
            "reason": "maintenance_survival_stage_not_complete",
            "source_paths": {},
        }
    unavailable = {
        "status": "not_observed",
        "value": None,
        "reason": "task_results_not_attached",
    }
    stages = {
        "mapping_complete": mapping,
        "canonical_entry_source_exists": canonical,
        "maintenance_survival": maintenance,
        "path_validity": path_stage,
        "retrieval_reach": dict(unavailable),
        "source_resolution": dict(unavailable),
    }
    mapping_trace = f"gold-mapping:{label['question_id']}:{value_sha256({
        'source_ids': source_values,
        'mapping_complete': eligible,
    })}"
    trace_ids = {
        "mapping": [mapping_trace],
        "canonical_entries": (
            [f"canonical-sources:{value_sha256({
                'checkpoint': checkpoint,
                'question_id': label['question_id'],
                'present_source_ids': canonical['present_source_ids'],
            })}"] if canonical["status"] == "observed" else []
        ),
        "maintenance": (
            [f"maintenance-sources:{value_sha256({
                'checkpoint': checkpoint,
                'question_id': label['question_id'],
                'present_source_ids': maintenance['present_source_ids'],
                'missing_source_ids': maintenance['missing_source_ids'],
            })}"] if maintenance["status"] == "observed" else []
        ),
        "paths": (
            [
                f"memory-path:{value_sha256({
                    'source_id': source_id, 'path': path,
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


def independent_chunks(session: Any, size: int = 6) -> list[tuple[list[Any], list[str]]]:
    require(isinstance(session, list), "session is not a list")
    turns = [item for item in session
             if isinstance(item, dict) and str(item.get("text", "")).strip()]
    output = []
    for start in range(0, len(turns), size):
        group = turns[start:start + size]
        output.append((
            [(item.get("speaker", "user"), item.get("text", ""))
             for item in group],
            [str(item["dia_id"]) for item in group if "dia_id" in item],
        ))
    return output


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
        if expected_model == "gpt-5.5":
            meta = response.get("exclusive_proxy_meta")
            require(isinstance(meta, dict)
                    and meta.get("logical_call_id") == logical_id
                    and isinstance(meta.get("event_id"), str)
                    and meta.get("client_http_attempts") == 1
                    and isinstance(meta.get("upstream_http_attempts"), int)
                    and meta["upstream_http_attempts"] >= 1
                    and isinstance(meta.get("unsupported_parameters"), list)
                    and re.fullmatch(r"[0-9a-f]{64}",
                                     str(meta.get("request_sha256", ""))),
                    f"exclusive response metadata mismatch for {logical_id}")
    return path


def audit_ledger(
    sample_dir: Path,
    *,
    run_id: str,
    mode: str,
    expected_model: str,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]],
    dict[str, dict[str, Any]], dict[str, dict[str, Any]],
]:
    records = read_ledger_independent(sample_dir / "operations.jsonl")
    require(records, "operation ledger is empty")
    require(all(record.get("run_id") == run_id for record in records),
            "ledger run_id mismatch")
    active: dict[str, Any] | None = None
    operations = []
    starts: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    previous_tree: dict[str, Any] | None = None
    for record in records:
        event = record.get("event")
        if event == "operation_started":
            require(active is None, "nested or unterminated operation")
            operation_id = record.get("operation_id")
            require(isinstance(operation_id, str) and operation_id,
                    "operation start lacks ID")
            metadata = record.get("metadata")
            require(isinstance(metadata, dict), f"metadata missing: {operation_id}")
            require(record.get("operation_input_sha256") == value_sha256({
                "kind": record.get("kind"), "metadata": metadata,
            }), f"operation input hash mismatch: {operation_id}")
            if previous_tree is None:
                require(record.get("continuing_tree_before") == {
                    "file_count": 0, "byte_count": 0,
                    "tree_sha256": value_sha256([]),
                }, "first operation did not start from empty memory")
            else:
                require(record.get("continuing_tree_before") == previous_tree,
                        f"operation tree chain mismatch: {operation_id}")
            active = record
        elif event == "model_call_started":
            require(active is not None
                    and record.get("operation_id") == active.get("operation_id"),
                    "model call occurred outside active operation")
            logical_id = record.get("logical_call_id")
            require(isinstance(logical_id, str) and logical_id not in starts,
                    f"duplicate or invalid model-call ID: {logical_id}")
            starts[logical_id] = record
        elif event in {"model_call_finished", "model_call_failed"}:
            logical_id = record.get("logical_call_id")
            require(isinstance(logical_id, str) and logical_id in starts
                    and logical_id not in terminals,
                    f"orphan or duplicate model-call terminal: {logical_id}")
            require(active is not None
                    and record.get("operation_id") == active.get("operation_id")
                    == starts[logical_id].get("operation_id"),
                    f"model-call operation mismatch: {logical_id}")
            require(isinstance(record.get("latency_s"), (int, float))
                    and record["latency_s"] >= 0,
                    f"invalid model latency: {logical_id}")
            terminals[logical_id] = record
        elif event == "operation_committed":
            require(active is not None
                    and record.get("operation_id") == active.get("operation_id"),
                    "operation commit without matching start")
            operation_id = str(record["operation_id"])
            require(record.get("kind") == active.get("kind")
                    and record.get("operation_input_sha256")
                    == active.get("operation_input_sha256"),
                    f"operation commit drift: {operation_id}")
            require(record.get("continuing_tree_before")
                    == active.get("continuing_tree_before"),
                    f"operation before-tree drift: {operation_id}")
            unresolved = [logical_id for logical_id, start in starts.items()
                          if start.get("operation_id") == operation_id
                          and logical_id not in terminals]
            require(not unresolved,
                    f"operation committed with unresolved calls: {unresolved}")
            scoped = [item for item in records
                      if active["sequence"] < item["sequence"] < record["sequence"]]
            require(record.get("cost") == cost_summary(scoped),
                    f"operation cost mismatch: {operation_id}")
            require(isinstance(record.get("latency_s"), (int, float))
                    and record["latency_s"] >= 0,
                    f"invalid operation latency: {operation_id}")
            previous_tree = record.get("continuing_tree_after")
            require(isinstance(previous_tree, dict),
                    f"operation after-tree missing: {operation_id}")
            operations.append({"start": active, "commit": record})
            active = None
        elif event == "operation_failed":
            raise AuditFailure(f"completed artifact has failed operation: {record}")
        else:
            raise AuditFailure(f"unknown ledger event: {event}")
    require(active is None, "ledger ends with an orphan operation")
    require(set(starts) == set(terminals), "ledger has orphan model calls")

    request_paths: set[Path] = set()
    response_paths: set[Path] = set()
    for logical_id, start in starts.items():
        request_paths.add(validate_request_artifact(
            sample_dir, start, mode=mode, expected_model=expected_model
        ))
        terminal = terminals[logical_id]
        response = validate_response_artifact(
            sample_dir, terminal, expected_model=expected_model
        )
        if response is not None:
            response_paths.add(response)
        if mode == "synthetic_sanity":
            require(start.get("proxy_log_start") is None,
                    f"synthetic call has proxy start: {logical_id}")
            require(terminal.get("proxy_evidence") == {
                "mode": "synthetic", "events": [],
                "client_http_attempts": 0, "upstream_http_attempts": 0,
                "unsupported_parameters": [], "log_prefix": None,
            }, f"synthetic proxy evidence mismatch: {logical_id}")
    calls_dir = sample_dir / "calls"
    require(calls_dir.is_dir(), "calls directory is missing")
    actual_call_files = {path for path in calls_dir.iterdir() if path.is_file()}
    require(actual_call_files == request_paths | response_paths,
            "unlinked or missing call artifacts")
    return records, operations, starts, terminals


def parse_proxy_log(path: Path) -> list[dict[str, Any]]:
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
    require(prefix.get("prefix_sha256") == hashlib.sha256(raw[:offset]).hexdigest(),
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
        require(not (root / "proxy").exists(), "synthetic artifact contains proxy files")
        return {"events": 0, "client_http_attempts": 0,
                "upstream_http_attempts": 0}
    proxy_dir = root / "proxy"
    require(proxy_dir.is_dir(), "formal artifact lacks proxy directory")
    logs = sorted(proxy_dir.glob("launch-*.jsonl"))
    ready = sorted(proxy_dir.glob("launch-*.ready.json"))
    require(logs and len(logs) == len(ready), "proxy launch files are incomplete")
    require({path.stem for path in logs}
            == {path.name.removesuffix(".ready.json") for path in ready},
            "proxy launch log/ready mismatch")
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
        run_fingerprint = load_json(root / "run_manifest.json")["run_fingerprint"]
        require(ready_value.get("run_id") == f"{run_fingerprint}:{launch_name}"
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

    starts = {key: value for sample in starts_by_sample for key, value in sample.items()}
    terminals = {
        key: value for sample in terminals_by_sample for key, value in sample.items()
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
        require(log in parsed_by_path, f"call references unknown proxy log: {logical_id}")
        start_offset = validate_prefix(start_prefix, log, label=logical_id)
        end_offset = validate_prefix(end_prefix, log, label=logical_id)
        require(start_offset <= end_offset, f"proxy cutoffs reversed: {logical_id}")
        linked_items = [item for item in parsed_by_path[log]
                        if item["event"].get("logical_call_id") == logical_id]
        require(linked_items, f"proxy has no event for {logical_id}")
        require(all(item["start"] >= start_offset and item["end"] <= end_offset
                    for item in linked_items),
                f"proxy event falls outside durable cutoffs: {logical_id}")
        linked_events = [item["event"] for item in linked_items]
        require(evidence.get("events") == linked_events,
                f"embedded proxy evidence differs from log: {logical_id}")
        for item in linked_items:
            key = (log, item["start"])
            require(key not in assigned, f"proxy event assigned twice: {logical_id}")
            assigned.add(key)
        require(all(event.get("requested_model") == "gpt-5.5"
                    and event.get("client_http_attempts") == 1
                    and isinstance(event.get("upstream_http_attempts"), int)
                    and event["upstream_http_attempts"] >= 0
                    and isinstance(event.get("unsupported_parameters"), list)
                    and re.fullmatch(r"[0-9a-f]{64}",
                                     str(event.get("request_sha256", "")))
                    and re.fullmatch(r"[0-9a-f]{64}",
                                     str(event.get("response_sha256", "")))
                    and isinstance(event.get("gateway_request_id"), str)
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(event.get("gateway_request_sha256", "")),
                    )
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(event.get("provider_request_sha256", "")),
                    )
                    for event in linked_events),
                f"proxy attempt metadata is incomplete: {logical_id}")
        require(evidence.get("client_http_attempts") == len(linked_events),
                f"client attempt count mismatch: {logical_id}")
        require(evidence.get("upstream_http_attempts") == sum(
            event["upstream_http_attempts"] for event in linked_events
        ), f"upstream attempt count mismatch: {logical_id}")
        unsupported = sorted({str(value) for event in linked_events
                              for value in event["unsupported_parameters"]})
        require(evidence.get("unsupported_parameters") == unsupported,
                f"unsupported-parameter mismatch: {logical_id}")
        successful = [event for event in linked_events
                      if event.get("status") == "success"]
        if terminal.get("event") == "model_call_finished":
            require(len(successful) == 1,
                    f"finished call lacks one proxy success: {logical_id}")
            success = successful[0]
            require(success.get("response_id") == terminal.get("response_id")
                    and success.get("actual_model") == terminal.get("response_model")
                    and normalize_usage(success.get("usage")) == terminal.get("usage"),
                    f"proxy/response identity mismatch: {logical_id}")
            sample_match = re.match(
                r"sample-(\d+)/", str(start.get("operation_id", ""))
            )
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
        else:
            require(not successful,
                    f"failed call has a successful proxy event: {logical_id}")
    require(len(assigned) == len(all_events), "unassigned proxy events exist")
    return {
        "events": len(all_events),
        "client_http_attempts": sum(
            int(item["event"]["client_http_attempts"]) for _, item in all_events
        ),
        "upstream_http_attempts": sum(
            int(item["event"]["upstream_http_attempts"]) for _, item in all_events
        ),
        "flex_gateway_windows": flex_reports,
    }


def expected_inventory(
    sample: dict[str, Any],
    sample_index: int,
    labels: list[dict[str, Any]],
    *,
    percent: int,
    stored_sources: set[str],
    canonical_sources: set[str],
    locations_by_source: dict[str, list[str]],
) -> list[dict[str, Any]]:
    label_by_q = {record["question_index"]: record for record in labels}
    output = []
    for question_index, qa in enumerate(sample["qa"]):
        if int(qa.get("category", 0)) == 5:
            continue
        label = label_by_q[question_index]
        key = str(percent)
        growth = bool(label["eligibility_by_checkpoint"][key])
        output.append({
            "schema": INVENTORY_SCHEMA,
            "trace_id": stable_trace_id(label["question_id"], percent),
            "sample_index": sample_index,
            "sample_id": sample["sample_id"],
            "checkpoint": percent,
            "question_index": question_index,
            "question_id": label["question_id"],
            "question": str(qa["question"]),
            "gold": qa.get("answer", ""),
            "category": int(qa["category"]),
            "normalized_source_ids": label["normalized_source_ids"],
            "growth_eligible": growth,
            "source_cohort_eligible": growth,
            "source_reachable": (
                set(label["normalized_source_ids"]).issubset(stored_sources)
                if growth else None
            ),
            "old_fact_cohort": bool(label["old_fact_by_checkpoint"][key]),
            "update_cohort": bool(
                label["update_label"]
                and label["update_first_checkpoint"] == percent
            ),
            "main_qa_100_cohort": percent == 100,
            "source_exclusion_reasons": label["source_exclusion_reasons"],
            "stage_evidence": expected_stage_evidence(
                label,
                checkpoint=percent,
                canonical_sources=canonical_sources,
                stored_sources=stored_sources,
                locations_by_source=locations_by_source,
            ),
        })
    return output


def audit_checkpoint(
    *,
    root: Path,
    sample_dir: Path,
    sample: dict[str, Any],
    sample_index: int,
    labels: list[dict[str, Any]],
    percent: int,
    operation: dict[str, Any],
    ledger_records: list[dict[str, Any]],
    run_fingerprint: str,
    source_mapping_sha256: str,
) -> dict[str, Any]:
    checkpoint_root = sample_dir / "checkpoints" / f"checkpoint-{percent:03d}"
    require(checkpoint_root.is_dir(), f"checkpoint missing: {percent}")
    require({path.name for path in checkpoint_root.iterdir()}
            == {"memory", "question_inventory.jsonl",
                "task_score_binding.json", "checkpoint_manifest.json"},
            f"checkpoint has stale artifacts: {percent}")
    memory = checkpoint_root / "memory"
    memory_tree = tree_descriptor(memory)
    stored = source_ids(memory)
    locations = source_locations(memory)
    canonical_sources = canonical_sources_through(
        ledger_records, operation["start"]["metadata"]["session_boundary"]
    )
    inventory_path = checkpoint_root / "question_inventory.jsonl"
    inventory = load_jsonl(inventory_path)
    expected = expected_inventory(
        sample,
        sample_index,
        labels,
        percent=percent,
        stored_sources=stored,
        canonical_sources=canonical_sources,
        locations_by_source=locations,
    )
    require(inventory == expected, f"question inventory mismatch: {percent}")
    inventory_sha = file_sha256(inventory_path)
    cohort_ids = {
        "growth": [item["question_id"] for item in inventory
                   if item["growth_eligible"]],
        "old_fact": [item["question_id"] for item in inventory
                     if item["old_fact_cohort"]],
        "update": [item["question_id"] for item in inventory
                   if item["update_cohort"]],
        "main_qa_100": [item["question_id"] for item in inventory
                        if item["main_qa_100_cohort"]],
    }
    binding_path = checkpoint_root / "task_score_binding.json"
    expected_binding = {
        "schema": BINDING_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "sample_index": sample_index,
        "checkpoint": percent,
        "memory_tree_sha256": memory_tree["tree_sha256"],
        "question_inventory_sha256": inventory_sha,
        "frozen_labels_sha256": file_sha256(root / "frozen_labels.json"),
        "source_mapping_sha256": source_mapping_sha256,
        "cohort_question_ids_sha256": {
            name: value_sha256(ids) for name, ids in cohort_ids.items()
        },
        "expected_external_result_schema": "nativemem.r403-task-results/v1",
        "task_results_status": "not_attached",
    }
    require(load_json(binding_path) == expected_binding,
            f"task-score binding mismatch: {percent}")
    binding_sha = file_sha256(binding_path)
    reachability = [item for item in inventory if item["source_cohort_eligible"]]
    counts = {
        "primary_questions": len(inventory),
        "growth_eligible": len(cohort_ids["growth"]),
        "source_excluded": sum(
            bool(item["source_exclusion_reasons"]) for item in inventory
        ),
        "old_fact": len(cohort_ids["old_fact"]),
        "update": len(cohort_ids["update"]),
        "main_qa_100": len(cohort_ids["main_qa_100"]),
        "source_reachable": sum(
            item["source_reachable"] is True for item in reachability
        ),
        "source_unreachable": sum(
            item["source_reachable"] is False for item in reachability
        ),
    }
    commit = operation["commit"]
    model_records = [item for item in ledger_records
                     if item["sequence"] < commit["sequence"]
                     and str(item.get("event", "")).startswith("model_call_")]
    operation_id = str(commit["operation_id"])
    checkpoint_model = [item for item in model_records
                        if item.get("operation_id") == operation_id]
    continuing_model = [item for item in model_records
                        if "/checkpoint-" not in str(item.get("operation_id", ""))]
    expected_cost = {
        "checkpoint_finalization_model": cost_summary(checkpoint_model),
        "cumulative_continuing_build_model": cost_summary(continuing_model),
        "cumulative_all_model": cost_summary(model_records),
    }
    manifest_path = checkpoint_root / "checkpoint_manifest.json"
    manifest = load_json(manifest_path)
    expected_keys = {
        "schema", "run_fingerprint", "sample_index", "sample_id", "checkpoint",
        "session_boundary", "total_sessions", "actual_session_fraction",
        "continuing_state_finalized", "checkpoint_copy_finalized", "memory_tree",
        "stored_source_id_count", "question_inventory_sha256",
        "task_score_binding_sha256", "cost", "inventory_counts", "task_scores",
    }
    require(set(manifest) == expected_keys,
            f"checkpoint manifest keys mismatch: {percent}")
    boundary = checkpoint_boundaries(session_count(sample["conversation"]))[
        str(percent)
    ]
    require(manifest == {
        "schema": CHECKPOINT_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "sample_index": sample_index,
        "sample_id": sample["sample_id"],
        "checkpoint": percent,
        "session_boundary": boundary,
        "total_sessions": session_count(sample["conversation"]),
        "actual_session_fraction": boundary / session_count(sample["conversation"]),
        "continuing_state_finalized": False,
        "checkpoint_copy_finalized": True,
        "memory_tree": memory_tree,
        "stored_source_id_count": len(stored),
        "question_inventory_sha256": inventory_sha,
        "task_score_binding_sha256": binding_sha,
        "cost": expected_cost,
        "inventory_counts": counts,
        "task_scores": {
            "status": "not_attached", "task_accuracy": None,
            "old_fact_retention": None, "update_accuracy": None,
        },
    }, f"checkpoint manifest mismatch: {percent}")
    checkpoint_tree = tree_descriptor(checkpoint_root)
    expected_result = {
        "checkpoint": percent,
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_tree": checkpoint_tree,
        "memory_tree": memory_tree,
        "inventory_counts": counts,
    }
    require(commit.get("result") == expected_result,
            f"checkpoint ledger result mismatch: {percent}")
    require(commit.get("continuing_tree_before")
            == commit.get("continuing_tree_after"),
            f"checkpoint mutated continuing state: {percent}")
    if percent == 100:
        require(counts["main_qa_100"] == len(inventory),
                "100% main QA denominator omits primary questions")
        require(counts["source_excluded"] == sum(
            not item["source_recall_eligible"] for item in labels
        ), "100% source exclusions are not retained in main QA")
    else:
        require(counts["main_qa_100"] == 0,
                f"pre-100 checkpoint entered main QA: {percent}")
    return expected_result


def audit_schedule(
    *,
    root: Path,
    sample_dir: Path,
    sample: dict[str, Any],
    sample_index: int,
    labels: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    ledger_records: list[dict[str, Any]],
    run_fingerprint: str,
    source_mapping_sha256: str,
) -> dict[str, Any]:
    conv = sample["conversation"]
    total = session_count(conv)
    boundaries = checkpoint_boundaries(total)
    checkpoints_at: dict[int, list[int]] = defaultdict(list)
    for percent in CHECKPOINTS:
        checkpoints_at[boundaries[str(percent)]].append(percent)
    by_id = {item["start"]["operation_id"]: item for item in operations}
    require(len(by_id) == len(operations), "duplicate operation IDs")
    expected_order: list[str] = []
    checkpoint_results: dict[str, Any] = {}
    dataset_dia_ids = {
        str(turn["dia_id"])
        for session in range(1, total + 1)
        for turn in conv[f"session_{session}"]
        if isinstance(turn, dict) and turn.get("dia_id")
    }
    for session_index in range(1, total + 1):
        recent: list[str] = []
        touched: set[str] = set()
        chunks = independent_chunks(conv[f"session_{session_index}"], 6)
        known_topics: str | None = None
        for chunk_index, (turns, dia_ids) in enumerate(chunks, start=1):
            operation_id = (
                f"sample-{sample_index:02d}/session-{session_index:03d}/"
                f"segment-{chunk_index:03d}"
            )
            expected_order.append(operation_id)
            require(operation_id in by_id, f"missing segment operation: {operation_id}")
            operation = by_id[operation_id]
            start = operation["start"]
            commit = operation["commit"]
            metadata = start["metadata"]
            require(start.get("kind") == "fixed6_extract_write",
                    f"segment kind mismatch: {operation_id}")
            require(metadata.get("sample_index") == sample_index
                    and metadata.get("session") == session_index
                    and metadata.get("segment") == chunk_index
                    and metadata.get("observation_date") == normalize_date(
                        conv.get(f"session_{session_index}_date_time", "")
                    )
                    and metadata.get("dia_ids") == dia_ids
                    and metadata.get("turns_sha256") == value_sha256(turns)
                    and metadata.get("recent_sha256") == value_sha256(recent[-20:])
                    and metadata.get("known_topics_sha256")
                    == value_sha256(metadata.get("known_topics")),
                    f"segment metadata mismatch: {operation_id}")
            if known_topics is None:
                known_topics = metadata["known_topics"]
            require(metadata.get("known_topics") == known_topics,
                    f"known-topic snapshot changed within session: {operation_id}")
            result = commit.get("result")
            require(isinstance(result, dict)
                    and isinstance(result.get("summaries"), list)
                    and isinstance(result.get("touched_topics"), list)
                    and isinstance(result.get("referenced_dia_ids"), list)
                    and isinstance(result.get("event_count"), int)
                    and result["event_count"] >= 0
                    and set(result["referenced_dia_ids"]).issubset(dataset_dia_ids),
                    f"segment result is invalid: {operation_id}")
            if result["event_count"] == 0:
                require(result == {
                    "event_count": 0,
                    "events_sha256": value_sha256([]),
                    "summaries": [],
                    "touched_topics": [],
                    "referenced_dia_ids": [],
                }, f"empty segment result mismatch: {operation_id}")
            else:
                require(bool(set(result["referenced_dia_ids"]).intersection(dia_ids)),
                        f"segment result lacks an in-segment source: {operation_id}")
            recent.extend(str(value) for value in result["summaries"])
            touched.update(str(value) for value in result["touched_topics"])

        dedup_id = (
            f"sample-{sample_index:02d}/session-{session_index:03d}/exact-dedup"
        )
        expected_order.append(dedup_id)
        require(dedup_id in by_id, f"missing dedup operation: {dedup_id}")
        dedup = by_id[dedup_id]
        require(dedup["start"].get("kind") == "session_exact_dedup"
                and dedup["start"].get("metadata") == {
                    "sample_index": sample_index, "session": session_index,
                }, f"dedup metadata mismatch: {dedup_id}")
        dedup_result = dedup["commit"].get("result")
        require(isinstance(dedup_result, dict)
                and isinstance(dedup_result.get("removed_lines"), int)
                and isinstance(dedup_result.get("topic_count_after"), int)
                and dedup_result["topic_count_after"] >= 0,
                f"dedup result mismatch: {dedup_id}")
        if dedup_result["topic_count_after"] > 30:
            consolidate_id = (
                f"sample-{sample_index:02d}/session-{session_index:03d}/"
                "topic-consolidation"
            )
            expected_order.append(consolidate_id)
            require(consolidate_id in by_id,
                    f"missing consolidation operation: {consolidate_id}")
            consolidate = by_id[consolidate_id]
            require(consolidate["start"].get("kind")
                    == "session_topic_consolidation"
                    and consolidate["start"].get("metadata") == {
                        "sample_index": sample_index, "session": session_index,
                        "trigger_topic_count": dedup_result["topic_count_after"],
                        "threshold": 30,
                    }, f"consolidation metadata mismatch: {consolidate_id}")
            result = consolidate["commit"].get("result")
            require(isinstance(result, dict)
                    and result.get("topic_count_before")
                    == dedup_result["topic_count_after"]
                    and isinstance(result.get("topic_count_after"), int)
                    and isinstance(result.get("merged"), int),
                    f"consolidation result mismatch: {consolidate_id}")
        if touched:
            tidy_id = (
                f"sample-{sample_index:02d}/session-{session_index:03d}/session-tidy"
            )
            expected_order.append(tidy_id)
            require(tidy_id in by_id, f"missing tidy operation: {tidy_id}")
            tidy = by_id[tidy_id]
            require(tidy["start"].get("kind") == "session_model_maintenance"
                    and tidy["start"].get("metadata") == {
                        "sample_index": sample_index, "session": session_index,
                        "touched_topics": sorted(touched), "sections": True,
                        "article": False, "final": False,
                    }, f"tidy metadata mismatch: {tidy_id}")
        for percent in checkpoints_at.get(session_index, []):
            checkpoint_id = f"sample-{sample_index:02d}/checkpoint-{percent:03d}"
            expected_order.append(checkpoint_id)
            require(checkpoint_id in by_id,
                    f"missing checkpoint operation: {checkpoint_id}")
            checkpoint = by_id[checkpoint_id]
            require(checkpoint["start"].get("kind") == "finalized_checkpoint_copy"
                    and checkpoint["start"].get("metadata") == {
                        "sample_index": sample_index, "checkpoint": percent,
                        "session_boundary": session_index,
                        "continuing_state_finalized": False,
                        "checkpoint_copy_finalized": True,
                    }, f"checkpoint metadata mismatch: {checkpoint_id}")
            checkpoint_results[str(percent)] = audit_checkpoint(
                root=root, sample_dir=sample_dir, sample=sample,
                sample_index=sample_index, labels=labels, percent=percent,
                operation=checkpoint, ledger_records=ledger_records,
                run_fingerprint=run_fingerprint,
                source_mapping_sha256=source_mapping_sha256,
            )
    actual_order = [item["start"]["operation_id"] for item in operations]
    require(actual_order == expected_order, "operation schedule is not exact")
    final_tree = tree_descriptor(sample_dir / "continuing_memory")
    require(operations[-1]["commit"].get("continuing_tree_after") == final_tree,
            "final continuing tree differs from ledger")
    return {
        "total_sessions": total,
        "checkpoint_boundaries": boundaries,
        "continuing_tree": final_tree,
        "checkpoints": checkpoint_results,
    }


def audit_sample(
    *,
    root: Path,
    sample_index: int,
    sample: dict[str, Any],
    labels: list[dict[str, Any]],
    run_fingerprint: str,
    mode: str,
    model: str,
    source_mapping_sha256: str,
) -> dict[str, Any]:
    sample_dir = root / "samples" / f"sample-{sample_index}"
    require(sample_dir.is_dir(), f"sample directory missing: {sample_index}")
    expected_names = {
        "continuing_memory", "checkpoints", "calls", "operations.jsonl",
        "sample_manifest.json",
    }
    require({path.name for path in sample_dir.iterdir()} == expected_names,
            f"sample has stale artifacts: {sample_index}")
    ledger_records, operations, starts, terminals = audit_ledger(
        sample_dir,
        run_id=f"{run_fingerprint}:sample-{sample_index}",
        mode=mode,
        expected_model=model,
    )
    require(starts, f"sample has no actual model calls: {sample_index}")
    expected_result = audit_schedule(
        root=root, sample_dir=sample_dir, sample=sample,
        sample_index=sample_index, labels=labels, operations=operations,
        ledger_records=ledger_records, run_fingerprint=run_fingerprint,
        source_mapping_sha256=source_mapping_sha256,
    )
    manifest_path = sample_dir / "sample_manifest.json"
    manifest = load_json(manifest_path)
    require(set(manifest) == {
        "schema", "status", "sample_index", "sample_id", "run_fingerprint",
        "completed_at", "result", "ledger_sha256", "ledger_event_count",
        "artifact_tree_before_manifest",
    }, f"sample manifest keys mismatch: {sample_index}")
    require(manifest.get("schema") == SAMPLE_SCHEMA
            and manifest.get("status") == "complete"
            and manifest.get("sample_index") == sample_index
            and manifest.get("sample_id") == sample["sample_id"]
            and manifest.get("run_fingerprint") == run_fingerprint
            and manifest.get("result") == expected_result
            and manifest.get("ledger_sha256")
            == file_sha256(sample_dir / "operations.jsonl")
            and manifest.get("ledger_event_count") == len(ledger_records)
            and manifest.get("artifact_tree_before_manifest")
            == tree_descriptor(
                sample_dir,
                exclude_relative=frozenset({"sample_manifest.json"}),
            ), f"sample manifest mismatch: {sample_index}")
    return {
        "sample_manifest_sha256": file_sha256(manifest_path),
        "model_call_count": len(starts),
        "successful_model_calls": sum(
            item.get("event") == "model_call_finished"
            for item in terminals.values()
        ),
        "failed_model_calls": sum(
            item.get("event") == "model_call_failed"
            for item in terminals.values()
        ),
        "operation_count": len(operations),
        "starts": starts,
        "terminals": terminals,
        "result": expected_result,
    }


def audit(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    assert_safe_tree(root)
    manifest = load_json(root / "run_manifest.json")
    require(manifest.get("schema") == RUN_SCHEMA, "run schema mismatch")
    require(manifest.get("status") == "complete", "run is not complete")
    config = manifest.get("config")
    require(isinstance(config, dict), "run config is missing")
    mode = config.get("mode")
    require(mode in {"synthetic_sanity", "formal"}, "invalid run mode")
    samples = [0] if mode == "synthetic_sanity" else list(range(10))
    model = "synthetic-no-model" if mode == "synthetic_sanity" else "gpt-5.5"
    if mode == "formal":
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
    expected_config = {
        "mode": mode,
        "samples": samples,
        "method": "NativeMem-v8.8+calendar",
        "model": model,
        "checkpoint_unit": "chronological session prefix",
        "checkpoint_boundary_rule": "ceil(total_sessions * percent / 100)",
        "checkpoints": list(CHECKPOINTS),
        "continuing_state_finalized": False,
        "checkpoint_copies_finalized": True,
        "environment": FINAL_ENV,
        "upstream_base_url": (
            None if mode == "synthetic_sanity" else gateway_contract["origin"]
        ),
        "gateway_contract": gateway_contract,
        "tokenizer": (
            "utf8_bytes_v1" if mode == "synthetic_sanity"
            else "tiktoken-0.12.0:o200k_base"
        ),
    }
    require(config == expected_config, "run config drift")
    require(set(manifest.get("input_hashes", {})) == set(EXPECTED_INPUT_HASHES),
            "input hash inventory mismatch")
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        current = file_sha256(ROOT / relative)
        require(current == expected
                and manifest["input_hashes"].get(relative) == expected,
                f"frozen input hash mismatch: {relative}")
    source_manifest = manifest.get("source_hashes")
    require(isinstance(source_manifest, dict)
            and set(source_manifest) == set(SOURCE_FILES),
            "source hash inventory mismatch")
    for relative in SOURCE_FILES:
        current = file_sha256(ROOT / relative)
        require(source_manifest.get(relative) == current,
                f"source changed since run: {relative}")
        if relative in FROZEN_SOURCE_HASHES:
            require(current == FROZEN_SOURCE_HASHES[relative],
                    f"frozen method source mismatch: {relative}")
    observations = manifest.get("environment_observations")
    require(isinstance(observations, dict)
            and set(observations).issubset(set(OBSERVED_BENCHMARK_RUNNERS))
            and all(re.fullmatch(r"[0-9a-f]{64}", str(value))
                    for value in observations.values()),
            "environment observation inventory is invalid")

    if mode == "synthetic_sanity":
        dataset, evidence = synthetic_inputs()
        dataset_sha = value_sha256(dataset)
        mapping_sha = value_sha256(evidence)
        require(not (root / "proxy").exists(), "synthetic root has proxy artifacts")
        expected_root_names = {"run_manifest.json", "frozen_labels.json", "samples"}
    else:
        dataset = load_json(DATA)
        require(isinstance(dataset, list) and len(dataset) == 10,
                "formal LoCoMo dataset is invalid")
        evidence = load_formal_evidence()
        dataset_sha = file_sha256(DATA)
        mapping_sha = file_sha256(EVIDENCE_QUESTIONS)
        expected_root_names = {
            "run_manifest.json", "frozen_labels.json", "samples", "proxy",
        }
    require({path.name for path in root.iterdir()} == expected_root_names,
            "run root has stale artifacts")
    expected_labels = freeze_labels_independent(
        dataset, evidence, dataset_sha256=dataset_sha,
        source_mapping_sha256=mapping_sha,
    )
    labels_path = root / "frozen_labels.json"
    labels = load_json(labels_path)
    require(labels == expected_labels, "frozen labels differ from raw inputs")
    require(manifest.get("frozen_labels_sha256") == file_sha256(labels_path),
            "frozen-label file hash mismatch")
    expected_fingerprint = value_sha256({
        "schema": RUN_SCHEMA,
        "config": config,
        "source_hashes": source_manifest,
        "input_hashes": manifest["input_hashes"],
        "labels_records_sha256": labels["records_sha256"],
    })
    require(manifest.get("run_fingerprint") == expected_fingerprint,
            "run fingerprint mismatch")
    run_fingerprint = expected_fingerprint
    sample_root = root / "samples"
    require(sample_root.is_dir(), "samples root is missing")
    require({path.name for path in sample_root.iterdir()}
            == {f"sample-{index}" for index in samples},
            "sample directory inventory mismatch")
    manifest_samples = manifest.get("samples")
    require(isinstance(manifest_samples, dict)
            and set(manifest_samples) == {str(index) for index in samples},
            "run sample manifest inventory mismatch")

    sample_reports = []
    starts_by_sample = []
    terminals_by_sample = []
    for sample_index in samples:
        sample_labels = [record for record in labels["records"]
                         if record["sample_index"] == sample_index]
        report = audit_sample(
            root=root, sample_index=sample_index, sample=dataset[sample_index],
            labels=sample_labels, run_fingerprint=run_fingerprint, mode=mode,
            model=model, source_mapping_sha256=mapping_sha,
        )
        linked = manifest_samples[str(sample_index)]
        require(linked.get("status") == "complete"
                and linked.get("sample_manifest_sha256")
                == report["sample_manifest_sha256"],
                f"run/sample manifest linkage mismatch: {sample_index}")
        sample_reports.append(report)
        starts_by_sample.append(report["starts"])
        terminals_by_sample.append(report["terminals"])
    proxy_report = audit_proxy(
        root, mode=mode, starts_by_sample=starts_by_sample,
        terminals_by_sample=terminals_by_sample,
    )
    return {
        "schema": "nativemem.r403-growth-audit/v1",
        "status": "pass",
        "mode": mode,
        "run_fingerprint": run_fingerprint,
        "sample_count": len(samples),
        "frozen_label_count": labels["record_count"],
        "source_exclusion_count": labels["source_exclusion_count"],
        "model_call_count": sum(item["model_call_count"] for item in sample_reports),
        "successful_model_calls": sum(
            item["successful_model_calls"] for item in sample_reports
        ),
        "failed_model_calls": sum(
            item["failed_model_calls"] for item in sample_reports
        ),
        "operation_count": sum(item["operation_count"] for item in sample_reports),
        "proxy": proxy_report,
        "checkpoints": list(CHECKPOINTS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.artifact)
    except AuditFailure as exc:
        print(json.dumps({"status": "fail", "error": str(exc)},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        require(args.artifact.expanduser().resolve() not in report_path.parents,
                "audit report must be outside the audited artifact")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
