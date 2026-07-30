#!/usr/bin/env python3
"""Run true chronological NativeMem growth with durable per-call accounting.

The continuing state is created once per LoCoMo conversation.  Intermediate
checkpoints are finalized copies; the continuing state is never finalized or
rebuilt.  Formal model traffic requires an explicit gate and an exclusive
per-launch forwarding proxy.  ``--synthetic-sanity`` uses a fake completion
resource and makes no network request.
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
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from src import openai_gpt55_flex_gateway_evidence as flex_evidence


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
EVIDENCE_DIR = (
    ROOT / "results" / "paper-experiments-20260714" / "evidence-mapping" / "v1"
)
EVIDENCE_MANIFEST = EVIDENCE_DIR / "evidence_mapping.v1.manifest.json"
EVIDENCE_QUESTIONS = EVIDENCE_DIR / "evidence_mapping.v1.questions.jsonl"
EVIDENCE_AUDIT = EVIDENCE_DIR / "evidence_mapping.v1.audit.json"
DEFAULT_OUTPUT = (
    ROOT / "results" / "paper-experiments-20260714"
    / "r403-incremental-growth-gpt55"
)

RUN_SCHEMA = "nativemem.r403-growth-run/v1"
SAMPLE_SCHEMA = "nativemem.r403-growth-sample/v1"
LABEL_SCHEMA = "nativemem.r403-frozen-labels/v1"
CHECKPOINT_SCHEMA = "nativemem.r403-checkpoint/v1"
INVENTORY_SCHEMA = "nativemem.r403-question-inventory/v2"
BINDING_SCHEMA = "nativemem.r403-task-score-binding/v1"
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
    r"\bnow\b",
    r"\bcurrently\b",
    r"\blatest\b",
    r"most recent",
    r"more recently",
    r"\bchanged\b",
    r"\bno longer\b",
    r"\binstead\b",
    r"over time",
    r"new .* change",
)


class GrowthError(RuntimeError):
    pass


class SafeStop(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_json(path: Path, value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    payload = b"".join(canonical_bytes(record) + b"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def parse_samples(spec: str) -> list[int]:
    output: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(item) for item in part.split("-", 1))
            output.update(range(min(left, right), max(left, right) + 1))
        else:
            output.add(int(part))
    values = sorted(output)
    if not values or values[0] < 0 or values[-1] > 9:
        raise argparse.ArgumentTypeError("samples must be in 0..9")
    return values


def collision_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def assert_safe_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise GrowthError(f"unsafe directory: {root}")
    paths: dict[str, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        key = collision_key(relative)
        if key in paths and paths[key] != relative:
            raise GrowthError(f"path collision: {paths[key]} / {relative}")
        paths[key] = relative
        if path.is_symlink():
            raise GrowthError(f"symlink is forbidden: {relative}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            if stat.st_nlink != 1:
                raise GrowthError(f"hardlink is forbidden: {relative}")
            inode = (stat.st_dev, stat.st_ino)
            if inode in inodes:
                raise GrowthError(f"shared inode: {inodes[inode]} / {relative}")
            inodes[inode] = relative
        elif not path.is_dir():
            raise GrowthError(f"special filesystem node: {relative}")


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


def source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise GrowthError(f"required source is missing or symlinked: {relative}")
        output[relative] = file_sha256(path)
        expected = FROZEN_SOURCE_HASHES.get(relative)
        if expected and output[relative] != expected:
            raise GrowthError(f"frozen source hash mismatch: {relative}")
    return output


def environment_observations() -> dict[str, str]:
    output = {}
    for relative in OBSERVED_BENCHMARK_RUNNERS:
        path = ROOT / relative
        if path.is_file() and not path.is_symlink():
            output[relative] = file_sha256(path)
    return output


def input_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise GrowthError(f"required input is missing or symlinked: {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise GrowthError(f"frozen input hash mismatch: {relative}")
        output[relative] = actual
    return output


def session_count(conv: dict[str, Any]) -> int:
    count = 0
    while f"session_{count + 1}" in conv:
        count += 1
    return count


def checkpoint_boundaries(total_sessions: int) -> dict[str, int]:
    if total_sessions < 1:
        raise GrowthError("conversation has no sessions")
    return {
        str(percent): math.ceil(total_sessions * percent / 100)
        for percent in CHECKPOINTS
    }


def normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def dia_session(dia_id: str) -> int:
    match = re.fullmatch(r"D(\d+):\d+", dia_id)
    if not match:
        raise GrowthError(f"invalid dia_id: {dia_id}")
    return int(match.group(1))


def load_evidence_records() -> list[dict[str, Any]]:
    records = []
    with EVIDENCE_QUESTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if (value.get("benchmark") == "LoCoMo"
                    and value.get("qa_scoring_eligible") is True):
                records.append(value)
    if len(records) != 1540:
        raise GrowthError(f"expected 1540 primary evidence records, got {len(records)}")
    excluded = [record for record in records
                if not record.get("source_recall_eligible")]
    if len(excluded) != 7:
        raise GrowthError(f"expected seven primary source exclusions, got {len(excluded)}")
    return records


def freeze_labels(
    dataset: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    *,
    dataset_sha256: str,
    source_mapping_sha256: str,
    expected_record_count: int,
    expected_source_exclusion_count: int,
) -> dict[str, Any]:
    evidence = {
        (int(record["sample_index"]), int(record["question_index"])): record
        for record in evidence_records
    }
    records = []
    for sample_index, sample in enumerate(dataset):
        conv = sample["conversation"]
        boundaries = checkpoint_boundaries(session_count(conv))
        turns: dict[str, dict[str, Any]] = {}
        for session in range(1, session_count(conv) + 1):
            for turn in conv[f"session_{session}"]:
                turns[str(turn["dia_id"])] = {
                    "session": session,
                    "text": str(turn.get("text", "")),
                }
        for question_index, qa in enumerate(sample["qa"]):
            if int(qa.get("category", 0)) == 5:
                continue
            mapping = evidence.get((sample_index, question_index))
            if mapping is None:
                raise GrowthError("primary question lacks frozen evidence mapping")
            if mapping.get("sample_id") != sample["sample_id"]:
                raise GrowthError("evidence mapping sample identity mismatch")
            source_ids = [str(value) for value in mapping["normalized_source_ids"]]
            source_sessions = sorted({dia_session(value) for value in source_ids})
            source_eligible = bool(mapping["source_recall_eligible"])
            if source_eligible and (
                not source_ids or not set(source_ids).issubset(turns)
            ):
                raise GrowthError(
                    "source-eligible mapping has absent raw conversation anchors"
                )
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
                {
                    "dia_id": dia_id,
                    "session": turn["session"],
                    "match": "normalized_substring",
                }
                for dia_id, turn in turns.items()
                if len(answer_norm) >= 2 and answer_norm in normalize_text(turn["text"])
            ]
            question_text = str(qa["question"])
            update_cues = [
                pattern for pattern in UPDATE_PATTERNS
                if re.search(pattern, question_text.casefold())
            ]
            update_label = bool(
                source_eligible and len(source_sessions) >= 2 and update_cues
            )
            previous = None
            old_fact_by_checkpoint: dict[str, bool] = {}
            for percent in CHECKPOINTS:
                key = str(percent)
                old_fact_by_checkpoint[key] = bool(
                    previous is not None and eligibility[str(previous)]
                )
                previous = percent
            records.append({
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
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
                "old_fact_by_checkpoint": old_fact_by_checkpoint,
                "update_label": update_label,
                "update_first_checkpoint": (
                    first_checkpoint if update_label else None
                ),
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
                    "answer_history": "normalized gold-answer substring in raw turns",
                },
            })
    if len(records) != expected_record_count:
        raise GrowthError(
            "frozen label inventory count mismatch: "
            f"expected {expected_record_count}, got {len(records)}"
        )
    question_ids = [record["question_id"] for record in records]
    if len(question_ids) != len(set(question_ids)):
        raise GrowthError("frozen labels contain duplicate question IDs")
    source_exclusion_count = sum(
        not record["source_recall_eligible"] for record in records
    )
    if source_exclusion_count != expected_source_exclusion_count:
        raise GrowthError(
            "frozen source-exclusion count mismatch: "
            f"expected {expected_source_exclusion_count}, "
            f"got {source_exclusion_count}"
        )
    return {
        "schema": LABEL_SCHEMA,
        "checkpoint_unit": "chronological session prefix",
        "checkpoint_boundary_rule": "ceil(total_sessions * percent / 100)",
        "checkpoints": list(CHECKPOINTS),
        "source_mapping_sha256": source_mapping_sha256,
        "dataset_sha256": dataset_sha256,
        "record_count": len(records),
        "source_exclusion_count": source_exclusion_count,
        "update_label_count": sum(record["update_label"] for record in records),
        "records_sha256": value_sha256(records),
        "records": records,
    }


def cost_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    starts = [record for record in records
              if record.get("event") == "model_call_started"]
    terminals = [record for record in records
                 if record.get("event") in {"model_call_finished", "model_call_failed"}]
    finished = [record for record in terminals
                if record.get("event") == "model_call_finished"]
    provider_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    provider = {
        field: sum(int(record.get("usage", {}).get(field) or 0)
                   for record in finished)
        for field in provider_fields
    }
    return {
        "logical_model_calls": len(starts),
        "successful_model_calls": len(finished),
        "failed_model_calls": len(terminals) - len(finished),
        "client_http_attempts": sum(
            int(record.get("proxy_evidence", {}).get("client_http_attempts") or 0)
            for record in terminals
        ),
        "physical_upstream_attempts": sum(
            int(record.get("proxy_evidence", {}).get("upstream_http_attempts") or 0)
            for record in terminals
        ),
        "local_visible_tokens": sum(
            int(record.get("local_visible_tokens") or 0) for record in starts
        ),
        "provider_tokens": provider,
        "model_latency_s": round(sum(
            float(record.get("latency_s") or 0.0) for record in terminals
        ), 6),
        "unsupported_parameters": sorted({
            str(parameter)
            for record in terminals
            for parameter in record.get("proxy_evidence", {}).get(
                "unsupported_parameters", []
            )
        }),
    }


def stable_trace_id(question_id: str, checkpoint: int) -> str:
    return f"r403:{question_id}:checkpoint-{checkpoint:03d}"


def question_stage_evidence(
    label: dict[str, Any],
    *,
    checkpoint: int,
    canonical_sources: set[str],
    stored_sources: set[str],
    source_locations: dict[str, list[str]],
) -> dict[str, Any]:
    source_ids = list(label["normalized_source_ids"])
    eligible = bool(label["source_recall_eligible"])
    present = sorted(set(source_ids) & canonical_sources)
    missing = sorted(set(source_ids) - canonical_sources)
    mapping = {
        "status": "observed",
        "value": eligible,
        "artifact": "frozen_labels.json",
        "question_id": label["question_id"],
        "source_ids": source_ids,
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
        missing_after_maintenance = sorted(set(source_ids) - stored_sources)
        maintenance = {
            "status": "observed",
            "value": not missing_after_maintenance,
            "artifact": f"checkpoints/checkpoint-{checkpoint:03d}/memory",
            "present_source_ids": sorted(set(source_ids) & stored_sources),
            "missing_source_ids": missing_after_maintenance,
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
            source_id: source_locations.get(source_id, [])
            for source_id in source_ids
        }
        path_value = all(locations[source_id] for source_id in source_ids)
        path_stage = {
            "status": "observed",
            "value": path_value,
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
        'source_ids': source_ids,
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


class IncrementalEngine:
    def __init__(
        self,
        *,
        adapter: Any,
        sample: dict[str, Any],
        sample_index: int,
        sample_dir: Path,
        labels: list[dict[str, Any]],
        ledger: Any,
        observer: Any,
        run_fingerprint: str,
        source_mapping_sha256: str,
        stop_after_operations: int | None = None,
        stop_requested: threading.Event | None = None,
    ) -> None:
        self.adapter = adapter
        self.sample = sample
        self.sample_index = sample_index
        self.sample_dir = sample_dir
        self.labels = labels
        self.ledger = ledger
        self.observer = observer
        self.run_fingerprint = run_fingerprint
        self.source_mapping_sha256 = source_mapping_sha256
        self.continuing = sample_dir / "continuing_memory"
        self.continuing.mkdir(parents=True, exist_ok=True)
        from src.evaluation.durable_model_ledger import ledger_state

        state = ledger_state(ledger.records)
        self.committed = state["committed_operations"]
        self.operation_starts = {
            str(record["operation_id"]): record
            for record in ledger.records
            if record.get("event") == "operation_started"
        }
        self.seen_uncommitted = False
        self.completed_this_launch = 0
        self.stop_after_operations = stop_after_operations
        self.stop_requested = stop_requested
        if self.committed:
            last = max(self.committed.values(), key=lambda record: record["sequence"])
            if tree_descriptor(self.continuing) != last["continuing_tree_after"]:
                raise GrowthError("continuing tree differs from the durable ledger")
        elif any(self.continuing.iterdir()):
            raise GrowthError("uncommitted continuing state is not empty")

    def run_operation(
        self,
        operation_id: str,
        kind: str,
        metadata: dict[str, Any],
        action: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        expected_input = value_sha256({"kind": kind, "metadata": metadata})
        existing = self.committed.get(operation_id)
        if existing is not None:
            if self.seen_uncommitted:
                raise GrowthError("ledger commits are not a contiguous schedule prefix")
            if (existing.get("kind") != kind
                    or existing.get("operation_input_sha256") != expected_input):
                raise GrowthError(f"operation drift on resume: {operation_id}")
            if kind == "finalized_checkpoint_copy":
                checkpoint = int(metadata["checkpoint"])
                checkpoint_root = (
                    self.sample_dir / "checkpoints"
                    / f"checkpoint-{checkpoint:03d}"
                )
                if (
                    not checkpoint_root.is_dir()
                    or existing.get("result", {}).get("checkpoint_tree")
                    != tree_descriptor(checkpoint_root)
                ):
                    raise GrowthError(
                        f"committed checkpoint artifact changed: {operation_id}"
                    )
            return copy.deepcopy(existing["result"])
        self.seen_uncommitted = True
        before = tree_descriptor(self.continuing)
        start_index = len(self.ledger.records)
        self.ledger.append(
            "operation_started",
            operation_id=operation_id,
            kind=kind,
            operation_input_sha256=expected_input,
            metadata=metadata,
            continuing_tree_before=before,
        )
        started = time.monotonic()
        try:
            with self.observer.operation(operation_id):
                result = action()
            after = tree_descriptor(self.continuing)
            operation_records = self.ledger.records[start_index + 1:]
            commit = self.ledger.append(
                "operation_committed",
                operation_id=operation_id,
                kind=kind,
                operation_input_sha256=expected_input,
                continuing_tree_before=before,
                continuing_tree_after=after,
                latency_s=round(time.monotonic() - started, 6),
                cost=cost_summary(operation_records),
                result=result,
            )
            self.committed[operation_id] = commit
            self.completed_this_launch += 1
            if (self.stop_after_operations is not None
                    and self.completed_this_launch >= self.stop_after_operations):
                raise SafeStop("requested synthetic stop after committed operation")
            if self.stop_requested is not None and self.stop_requested.is_set():
                raise SafeStop("safe stop requested after committed operation")
            return copy.deepcopy(result)
        except SafeStop:
            raise
        except BaseException as exc:
            self.ledger.append(
                "operation_failed",
                operation_id=operation_id,
                kind=kind,
                operation_input_sha256=expected_input,
                latency_s=round(time.monotonic() - started, 6),
                continuing_tree_after=tree_descriptor(self.continuing),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _source_ids(self, memory_dir: Path) -> set[str]:
        values: set[str] = set()
        for path in memory_dir.rglob("*.md"):
            values.update(self.adapter.v8_memory.dia_ids_in(
                path.read_text(encoding="utf-8")
            ))
        return values

    def _source_locations(self, memory_dir: Path) -> dict[str, list[str]]:
        output: defaultdict[str, list[str]] = defaultdict(list)
        for path in sorted(memory_dir.rglob("*.md")):
            relative = path.relative_to(memory_dir).as_posix()
            for source_id in sorted(self.adapter.v8_memory.dia_ids_in(
                path.read_text(encoding="utf-8")
            )):
                output[str(source_id)].append(relative)
        return {key: value for key, value in sorted(output.items())}

    def _canonical_sources_through(self, session_boundary: int) -> set[str]:
        sources: set[str] = set()
        for record in self.ledger.records:
            if (record.get("event") != "operation_committed"
                    or record.get("kind") != "fixed6_extract_write"):
                continue
            match = re.search(
                r"/session-(\d+)/segment-", str(record.get("operation_id", ""))
            )
            if match and int(match.group(1)) <= session_boundary:
                sources.update(str(value) for value in record.get(
                    "result", {}
                ).get("referenced_dia_ids", []))
        return sources

    def _checkpoint_action(
        self, percent: int, session_boundary: int
    ) -> dict[str, Any]:
        checkpoint_root = self.sample_dir / "checkpoints" / f"checkpoint-{percent:03d}"
        if checkpoint_root.exists():
            raise GrowthError(f"uncommitted checkpoint already exists: {checkpoint_root}")
        staging = checkpoint_root.with_name(
            f".{checkpoint_root.name}.staging-{os.getpid()}"
        )
        if staging.exists():
            raise GrowthError(f"stale checkpoint staging exists: {staging}")
        memory_copy = staging / "memory"
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.continuing, memory_copy)
        all_topics = set(self.adapter.v8_memory._topics_dir_files(str(memory_copy)))
        if all_topics:
            self.adapter._v8_tidy(
                str(memory_copy), all_topics, True, False, final=True
            )
        memory_tree = tree_descriptor(memory_copy)
        stored_sources = self._source_ids(memory_copy)
        source_locations = self._source_locations(memory_copy)
        canonical_sources = self._canonical_sources_through(session_boundary)
        label_by_q = {record["question_index"]: record for record in self.labels}
        inventory = []
        for question_index, qa in enumerate(self.sample["qa"]):
            if int(qa.get("category", 0)) == 5:
                continue
            label = label_by_q[question_index]
            key = str(percent)
            growth_eligible = bool(label["eligibility_by_checkpoint"][key])
            source_reachable = (
                set(label["normalized_source_ids"]).issubset(stored_sources)
                if growth_eligible else None
            )
            inventory.append({
                "schema": INVENTORY_SCHEMA,
                "trace_id": stable_trace_id(label["question_id"], percent),
                "sample_index": self.sample_index,
                "sample_id": self.sample["sample_id"],
                "checkpoint": percent,
                "question_index": question_index,
                "question_id": label["question_id"],
                "question": str(qa["question"]),
                "gold": qa.get("answer", ""),
                "category": int(qa["category"]),
                "normalized_source_ids": label["normalized_source_ids"],
                "growth_eligible": growth_eligible,
                "source_cohort_eligible": growth_eligible,
                "source_reachable": source_reachable,
                "old_fact_cohort": bool(label["old_fact_by_checkpoint"][key]),
                "update_cohort": bool(
                    label["update_label"]
                    and label["update_first_checkpoint"] == percent
                ),
                "main_qa_100_cohort": percent == 100,
                "source_exclusion_reasons": label["source_exclusion_reasons"],
                "stage_evidence": question_stage_evidence(
                    label,
                    checkpoint=percent,
                    canonical_sources=canonical_sources,
                    stored_sources=stored_sources,
                    source_locations=source_locations,
                ),
            })
        inventory_path = staging / "question_inventory.jsonl"
        inventory_sha = atomic_jsonl(inventory_path, inventory)
        cohort_ids = {
            "growth": [record["question_id"] for record in inventory
                       if record["growth_eligible"]],
            "old_fact": [record["question_id"] for record in inventory
                         if record["old_fact_cohort"]],
            "update": [record["question_id"] for record in inventory
                       if record["update_cohort"]],
            "main_qa_100": [record["question_id"] for record in inventory
                            if record["main_qa_100_cohort"]],
        }
        binding = {
            "schema": BINDING_SCHEMA,
            "run_fingerprint": self.run_fingerprint,
            "sample_index": self.sample_index,
            "checkpoint": percent,
            "memory_tree_sha256": memory_tree["tree_sha256"],
            "question_inventory_sha256": inventory_sha,
            "frozen_labels_sha256": file_sha256(
                self.sample_dir.parent.parent / "frozen_labels.json"
            ),
            "source_mapping_sha256": self.source_mapping_sha256,
            "cohort_question_ids_sha256": {
                name: value_sha256(ids) for name, ids in cohort_ids.items()
            },
            "expected_external_result_schema": (
                "nativemem.r403-task-results/v1"
            ),
            "task_results_status": "not_attached",
        }
        binding_sha = atomic_json(staging / "task_score_binding.json", binding)
        reachability = [record for record in inventory
                        if record["source_cohort_eligible"]]
        current_operation_id = (
            f"sample-{self.sample_index:02d}/checkpoint-{percent:03d}"
        )
        model_records = [
            record for record in self.ledger.records
            if record.get("event", "").startswith("model_call_")
        ]
        checkpoint_model_records = [
            record for record in model_records
            if record.get("operation_id") == current_operation_id
        ]
        continuing_model_records = [
            record for record in model_records
            if "/checkpoint-" not in str(record.get("operation_id", ""))
        ]
        checkpoint_manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "run_fingerprint": self.run_fingerprint,
            "sample_index": self.sample_index,
            "sample_id": self.sample["sample_id"],
            "checkpoint": percent,
            "session_boundary": session_boundary,
            "total_sessions": session_count(self.sample["conversation"]),
            "actual_session_fraction": (
                session_boundary / session_count(self.sample["conversation"])
            ),
            "continuing_state_finalized": False,
            "checkpoint_copy_finalized": True,
            "memory_tree": memory_tree,
            "stored_source_id_count": len(stored_sources),
            "question_inventory_sha256": inventory_sha,
            "task_score_binding_sha256": binding_sha,
            "cost": {
                "checkpoint_finalization_model": cost_summary(
                    checkpoint_model_records
                ),
                "cumulative_continuing_build_model": cost_summary(
                    continuing_model_records
                ),
                "cumulative_all_model": cost_summary(model_records),
            },
            "inventory_counts": {
                "primary_questions": len(inventory),
                "growth_eligible": len(cohort_ids["growth"]),
                "source_excluded": sum(
                    bool(record["source_exclusion_reasons"])
                    for record in inventory
                ),
                "old_fact": len(cohort_ids["old_fact"]),
                "update": len(cohort_ids["update"]),
                "main_qa_100": len(cohort_ids["main_qa_100"]),
                "source_reachable": sum(
                    record["source_reachable"] is True for record in reachability
                ),
                "source_unreachable": sum(
                    record["source_reachable"] is False for record in reachability
                ),
            },
            "task_scores": {
                "status": "not_attached",
                "task_accuracy": None,
                "old_fact_retention": None,
                "update_accuracy": None,
            },
        }
        manifest_sha = atomic_json(staging / "checkpoint_manifest.json",
                                   checkpoint_manifest)
        assert_safe_tree(staging)
        os.replace(staging, checkpoint_root)
        directory = os.open(checkpoint_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {
            "checkpoint": percent,
            "checkpoint_manifest_sha256": manifest_sha,
            "checkpoint_tree": tree_descriptor(checkpoint_root),
            "memory_tree": memory_tree,
            "inventory_counts": checkpoint_manifest["inventory_counts"],
        }

    def run(self) -> dict[str, Any]:
        conv = self.sample["conversation"]
        total_sessions = session_count(conv)
        boundaries = checkpoint_boundaries(total_sessions)
        checkpoints_at: dict[int, list[int]] = {}
        for percent in CHECKPOINTS:
            checkpoints_at.setdefault(boundaries[str(percent)], []).append(percent)
        for session_index in range(1, total_sessions + 1):
            known_view: Any = None
            recent: list[str] = []
            touched: set[str] = set()
            chunks = self.adapter.split_into_chunks_structured(
                conv[f"session_{session_index}"], 6
            )
            for chunk_index, (turns, dia_ids) in enumerate(chunks, start=1):
                operation_id = (
                    f"sample-{self.sample_index:02d}/session-{session_index:03d}/"
                    f"segment-{chunk_index:03d}"
                )
                existing = self.committed.get(operation_id)
                if known_view is None:
                    if existing is not None:
                        known_view = copy.deepcopy(
                            self.operation_starts[operation_id]["metadata"][
                                "known_topics"
                            ]
                        )
                    else:
                        known_view = self.adapter._format_topics_snapshot(
                            self.adapter._topics_snapshot(str(self.continuing))
                        )
                metadata = {
                    "sample_index": self.sample_index,
                    "session": session_index,
                    "segment": chunk_index,
                    "observation_date": self.adapter.normalize_date(
                        conv.get(f"session_{session_index}_date_time", "")
                    ),
                    "dia_ids": [str(value) for value in dia_ids],
                    "turns_sha256": value_sha256(turns),
                    "known_topics": known_view,
                    "known_topics_sha256": value_sha256(known_view),
                    "recent_sha256": value_sha256(recent[-20:]),
                }

                def segment_action(
                    turns=turns,
                    dia_ids=dia_ids,
                    metadata=metadata,
                ) -> dict[str, Any]:
                    events = self.adapter.v8_memory.distill_events(
                        turns,
                        metadata["observation_date"],
                        dia_ids,
                        known_topics=known_view,
                        recent=recent[-20:],
                    )
                    if not events:
                        return {
                            "event_count": 0,
                            "events_sha256": value_sha256([]),
                            "summaries": [],
                            "touched_topics": [],
                            "referenced_dia_ids": [],
                        }
                    expected = {str(value) for value in dia_ids}
                    referenced = {
                        str(value)
                        for event in events
                        for value in event.get("dia_ids", [])
                    }
                    if not referenced.intersection(expected):
                        raise GrowthError("segment events contain no source from the segment")
                    self.adapter.write_events(str(self.continuing), events)
                    summaries = [str(event["summary"]) for event in events]
                    topics = sorted({
                        self.adapter.v8_memory._sanitize_topic(
                            str(event.get("topic", "misc"))
                        )
                        for event in events
                    })
                    return {
                        "event_count": len(events),
                        "events_sha256": value_sha256(events),
                        "summaries": summaries,
                        "touched_topics": topics,
                        "referenced_dia_ids": sorted(referenced),
                    }

                result = self.run_operation(
                    operation_id, "fixed6_extract_write", metadata, segment_action
                )
                recent.extend(result["summaries"])
                touched.update(result["touched_topics"])

            dedup_id = (
                f"sample-{self.sample_index:02d}/session-{session_index:03d}/"
                "exact-dedup"
            )
            dedup_result = self.run_operation(
                dedup_id,
                "session_exact_dedup",
                {"sample_index": self.sample_index, "session": session_index},
                lambda: self._dedup_action(),
            )
            topic_count = int(dedup_result["topic_count_after"])
            if topic_count > 30:
                consolidate_id = (
                    f"sample-{self.sample_index:02d}/session-{session_index:03d}/"
                    "topic-consolidation"
                )
                self.run_operation(
                    consolidate_id,
                    "session_topic_consolidation",
                    {
                        "sample_index": self.sample_index,
                        "session": session_index,
                        "trigger_topic_count": topic_count,
                        "threshold": 30,
                    },
                    lambda: self._consolidation_action(),
                )
            if touched:
                tidy_id = (
                    f"sample-{self.sample_index:02d}/session-{session_index:03d}/"
                    "session-tidy"
                )
                self.run_operation(
                    tidy_id,
                    "session_model_maintenance",
                    {
                        "sample_index": self.sample_index,
                        "session": session_index,
                        "touched_topics": sorted(touched),
                        "sections": True,
                        "article": False,
                        "final": False,
                    },
                    lambda: (
                        self.adapter._v8_tidy(
                            str(self.continuing), touched, True, False, final=False
                        )
                        or {"maintained_topics": sorted(touched)}
                    ),
                )
            for percent in checkpoints_at.get(session_index, []):
                checkpoint_id = (
                    f"sample-{self.sample_index:02d}/checkpoint-{percent:03d}"
                )
                self.run_operation(
                    checkpoint_id,
                    "finalized_checkpoint_copy",
                    {
                        "sample_index": self.sample_index,
                        "checkpoint": percent,
                        "session_boundary": session_index,
                        "continuing_state_finalized": False,
                        "checkpoint_copy_finalized": True,
                    },
                    lambda percent=percent, session_index=session_index: (
                        self._checkpoint_action(percent, session_index)
                    ),
                )
        checkpoint_results = {
            str(percent): self.committed[
                f"sample-{self.sample_index:02d}/checkpoint-{percent:03d}"
            ]["result"]
            for percent in CHECKPOINTS
        }
        return {
            "total_sessions": total_sessions,
            "checkpoint_boundaries": boundaries,
            "continuing_tree": tree_descriptor(self.continuing),
            "checkpoints": checkpoint_results,
        }

    def _dedup_action(self) -> dict[str, Any]:
        removed = int(
            self.adapter.v8_memory.dedup_topic_files(str(self.continuing)) or 0
        )
        topic_count = len(
            self.adapter.v8_memory._topics_dir_files(str(self.continuing))
        )
        return {"removed_lines": removed, "topic_count_after": topic_count}

    def _consolidation_action(self) -> dict[str, Any]:
        before = len(
            self.adapter.v8_memory._topics_dir_files(str(self.continuing))
        )
        merged = int(
            self.adapter.v8_memory.consolidate_topic_files(
                str(self.continuing)
            ) or 0
        )
        after = len(
            self.adapter.v8_memory._topics_dir_files(str(self.continuing))
        )
        return {
            "merged": merged,
            "topic_count_before": before,
            "topic_count_after": after,
        }


class FakeResponse:
    def __init__(self, call_index: int, logical_call_id: str) -> None:
        suffix = hashlib.sha256(logical_call_id.encode()).hexdigest()[:16]
        self.id = f"synthetic-response-{suffix}"
        self.model = "synthetic-no-model"
        self.usage = SimpleNamespace(
            prompt_tokens=10 + call_index,
            completion_tokens=2,
            total_tokens=12 + call_index,
        )

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        return {
            "id": self.id,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }


def synthetic_adapter(real_adapter: Any) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    calls = {"count": 0}

    def create(*args: Any, **kwargs: Any) -> FakeResponse:
        del args
        calls["count"] += 1
        logical_call_id = str(
            (kwargs.get("extra_headers") or {}).get(
                "X-Controlled-Logical-Call-ID", "missing-logical-call-id"
            )
        )
        return FakeResponse(calls["count"], logical_call_id)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    v8 = SimpleNamespace(
        client=client,
        write_events=real_adapter.v8_memory.write_events,
        dedup_topic_files=real_adapter.v8_memory.dedup_topic_files,
        _sanitize_topic=real_adapter.v8_memory._sanitize_topic,
        _topics_dir_files=real_adapter.v8_memory._topics_dir_files,
        dia_ids_in=real_adapter.v8_memory.dia_ids_in,
    )

    def distill_events(
        turns: list[tuple[str, str]], obs_date: str, dia_ids: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        client.chat.completions.create(
            model="synthetic-no-model",
            messages=[{"role": "user", "content": json.dumps(turns)}],
            max_tokens=100,
        )
        return [{
            "when": obs_date,
            "summary": f"fact from {dia_ids[0]}",
            "summary_inline": f"fact from [{dia_ids[0]}]",
            "dia_ids": list(dia_ids),
            "topic": f"person/session-{dia_session(dia_ids[0])}",
        }]

    v8.distill_events = distill_events
    v8.consolidate_topic_files = lambda memory_dir: 0

    def tidy(memory_dir: str, touched: set[str], sections: bool,
             article: bool, final: bool = False) -> None:
        del memory_dir, sections, article
        for topic in sorted(touched):
            client.chat.completions.create(
                model="synthetic-no-model",
                messages=[{"role": "user", "content": f"tidy:{topic}:final={final}"}],
                max_tokens=100,
            )

    adapter = SimpleNamespace(
        v8_memory=v8,
        write_events=real_adapter.v8_memory.write_events,
        split_into_chunks_structured=real_adapter.split_into_chunks_structured,
        normalize_date=real_adapter.normalize_date,
        _topics_snapshot=real_adapter._topics_snapshot,
        _format_topics_snapshot=real_adapter._format_topics_snapshot,
        _v8_tidy=tidy,
    )
    sessions = {
        f"session_{index}": [{
            "speaker": "A",
            "dia_id": f"D{index}:1",
            "text": f"fact {index}",
        }]
        for index in range(1, 11)
    }
    dates = {
        f"session_{index}_date_time": f"2023-05-{index:02d}"
        for index in range(1, 11)
    }
    sample = {
        "sample_id": "synthetic-growth",
        "conversation": {**sessions, **dates},
        "qa": [
            {
                "question": "What was fact one?",
                "answer": "fact 1",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": "What changed over time?",
                "answer": "fact 5",
                "evidence": ["D1:1", "D5:1"],
                "category": 1,
            },
            {
                "question": "Which final fact appeared?",
                "answer": "fact 10",
                "evidence": ["D10:1"],
                "category": 2,
            },
            {
                "question": "Unmapped question?",
                "answer": "unknown",
                "evidence": [],
                "category": 3,
            },
        ],
    }
    evidence = []
    for index, qa in enumerate(sample["qa"]):
        ids = list(qa["evidence"])
        eligible = bool(ids)
        evidence.append({
            "benchmark": "LoCoMo",
            "qa_scoring_eligible": True,
            "source_recall_eligible": eligible,
            "source_recall_exclusion_reasons": [] if eligible else ["gold_evidence_empty"],
            "normalized_source_ids": ids,
            "sample_index": 0,
            "sample_id": sample["sample_id"],
            "question_index": index,
            "question_id": f"locomo:synthetic:q{index:03d}",
        })
    return adapter, [sample], evidence


def configure_environment(model: str, base_url: str, api_key: str) -> None:
    os.environ.update(FINAL_ENV)
    os.environ["BUILDER_MODEL"] = model
    os.environ["BUILDER_BASE"] = base_url
    os.environ["BUILDER_KEY"] = api_key
    os.environ["ALIYUN_KEY"] = api_key
    os.environ.pop("MODEL", None)
    os.environ.pop("NATIVEMEM_V9_PIPELINE", None)
    os.environ.pop("NATIVEMEM_V9_SCRIBE_MODE", None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"


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
        launches = sorted(proxy_dir.glob("launch-*.ready.json"))
        launch = len(launches) + 1
        self.log = proxy_dir / f"launch-{launch:03d}.jsonl"
        self.ready = proxy_dir / f"launch-{launch:03d}.ready.json"
        if self.log.exists() or self.ready.exists():
            raise GrowthError("proxy launch artifacts already exist")
        command = [
            sys.executable,
            "scripts/controlled_gpt55_run_proxy.py",
            "--port", "0",
            "--upstream", self.upstream,
            "--log", str(self.log),
            "--ready", str(self.ready),
            "--run-id", f"{self.run_id}:launch-{launch:03d}",
        ]
        self.process = subprocess.Popen(command, cwd=ROOT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise GrowthError("exclusive proxy exited before readiness")
            if self.ready.is_file():
                value = json.loads(self.ready.read_text(encoding="utf-8"))
                self.base_url = str(value["base_url"])
                return
            time.sleep(0.05)
        raise GrowthError("exclusive proxy readiness timed out")

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=parse_samples, default=parse_samples("0-9"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--api-key", default="x")
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-operations", type=int)
    args = parser.parse_args()
    if os.environ.get("NATIVEMEM_V9_PIPELINE") == "two_tier":
        parser.error("v9 two_tier is forbidden for R403")
    if not args.synthetic_sanity and not args.allow_model_requests:
        parser.error("formal execution requires --allow-model-requests")
    if args.stop_after_operations is not None and not args.synthetic_sanity:
        parser.error("--stop-after-operations is synthetic-only")
    if args.synthetic_sanity:
        if args.allow_model_requests:
            parser.error("synthetic R403 forbids --allow-model-requests")
        if args.gateway_root is not None:
            parser.error("synthetic R403 forbids --gateway-root")
        args.samples = [0]
        args.model = "synthetic-no-model"
    else:
        if args.samples != list(range(10)):
            parser.error("formal R403 requires all samples 0-9")
        if args.model != "gpt-5.5":
            parser.error("formal R403 requires --model gpt-5.5")
        if args.gateway_root is None:
            parser.error("formal R403 requires --gateway-root")

    raw_output = args.output_dir.expanduser()
    absolute_output = raw_output if raw_output.is_absolute() else ROOT / raw_output
    current = Path(absolute_output.anchor)
    for part in absolute_output.parts[1:]:
        current /= part
        if current.is_symlink():
            parser.error(f"output path contains symlink component: {current}")
    output = absolute_output.resolve()
    if output == ROOT or ROOT not in output.parents:
        parser.error("output must be under the repository")
    lock_path = ROOT / "results" / ".r403-active-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another R403 runner is active") from exc

    lock_cleaned = False

    def cleanup_lock() -> None:
        nonlocal lock_cleaned
        if lock_cleaned:
            return
        lock_cleaned = True
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        try:
            lock.close()
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)

    atexit.register(cleanup_lock)

    gateway_contract = (
        None
        if args.synthetic_sanity
        else flex_evidence.active_contract(args.gateway_root.expanduser().resolve())
    )
    sources = source_hashes()
    observations = environment_observations()
    inputs = input_hashes()
    if args.synthetic_sanity:
        configure_environment(args.model, "synthetic", args.api_key)
        real_adapter = load_adapter()
        adapter, dataset, evidence_records = synthetic_adapter(real_adapter)
        dataset_sha256 = value_sha256(dataset)
        source_mapping_sha256 = value_sha256(evidence_records)
        expected_record_count = 4
        expected_source_exclusion_count = 1
    else:
        dataset = json.loads(DATA.read_text(encoding="utf-8"))
        evidence_records = load_evidence_records()
        adapter = None
        dataset_sha256 = file_sha256(DATA)
        source_mapping_sha256 = file_sha256(EVIDENCE_QUESTIONS)
        expected_record_count = 1540
        expected_source_exclusion_count = 7
    labels = freeze_labels(
        dataset,
        evidence_records,
        dataset_sha256=dataset_sha256,
        source_mapping_sha256=source_mapping_sha256,
        expected_record_count=expected_record_count,
        expected_source_exclusion_count=expected_source_exclusion_count,
    )
    run_config = {
        "mode": "synthetic_sanity" if args.synthetic_sanity else "formal",
        "samples": args.samples,
        "method": "NativeMem-v8.8+calendar",
        "model": args.model,
        "checkpoint_unit": "chronological session prefix",
        "checkpoint_boundary_rule": "ceil(total_sessions * percent / 100)",
        "checkpoints": list(CHECKPOINTS),
        "continuing_state_finalized": False,
        "checkpoint_copies_finalized": True,
        "environment": FINAL_ENV,
        "upstream_base_url": (
            None if args.synthetic_sanity else gateway_contract["origin"]
        ),
        "gateway_contract": gateway_contract,
        "tokenizer": (
            "utf8_bytes_v1" if args.synthetic_sanity else "tiktoken-0.12.0:o200k_base"
        ),
    }
    run_fingerprint = value_sha256({
        "schema": RUN_SCHEMA,
        "config": run_config,
        "source_hashes": sources,
        "input_hashes": inputs,
        "labels_records_sha256": labels["records_sha256"],
    })
    if output.exists() and not args.resume:
        raise SystemExit("output exists; use --resume only for the same fingerprint")
    if not output.exists():
        output.mkdir(parents=True)
        (output / "samples").mkdir()
        labels_sha = atomic_json(output / "frozen_labels.json", labels)
        manifest: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "status": "running",
            "created_at": utc_now(),
            "run_fingerprint": run_fingerprint,
            "config": run_config,
            "source_hashes": sources,
            "environment_observations": observations,
            "input_hashes": inputs,
            "frozen_labels_sha256": labels_sha,
            "samples": {},
        }
        atomic_json(output / "run_manifest.json", manifest)
    else:
        if output.is_symlink() or not output.is_dir():
            raise SystemExit("existing output is unsafe")
        assert_safe_tree(output)
        manifest = json.loads((output / "run_manifest.json").read_text())
        if manifest.get("schema") != RUN_SCHEMA:
            raise SystemExit("resume schema mismatch")
        if manifest.get("status") == "failed":
            raise SystemExit("failed runs require a new output root")
        if manifest.get("run_fingerprint") != run_fingerprint:
            raise SystemExit("resume fingerprint mismatch")
        if (
            manifest.get("config") != run_config
            or manifest.get("source_hashes") != sources
            or manifest.get("input_hashes") != inputs
        ):
            raise SystemExit("resume manifest provenance mismatch")
        if file_sha256(output / "frozen_labels.json") != manifest[
            "frozen_labels_sha256"
        ]:
            raise SystemExit("frozen labels changed")
        if json.loads(
            (output / "frozen_labels.json").read_text(encoding="utf-8")
        ) != labels:
            raise SystemExit("frozen labels differ from raw inputs")
        allowed = {"run_manifest.json", "frozen_labels.json", "samples", "proxy"}
        if any(path.name not in allowed for path in output.iterdir()):
            raise SystemExit("existing output has stale root artifacts")
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
        atomic_json(output / "run_manifest.json", manifest)

    proxy = None
    provider_lock = None
    if not args.synthetic_sanity:
        try:
            version = importlib.metadata.version("tiktoken")
        except importlib.metadata.PackageNotFoundError as exc:
            raise SystemExit("formal R403 requires tiktoken==0.12.0") from exc
        if version != "0.12.0":
            raise SystemExit("formal R403 requires tiktoken==0.12.0")
        provider_lock = flex_evidence.acquire_consumer_lock(
            Path(gateway_contract["result_root"])
        )
        if flex_evidence.active_contract(
            Path(gateway_contract["result_root"])
        ) != gateway_contract:
            provider_lock.close()
            raise GrowthError("Flex gateway contract changed before execution")
        proxy = ManagedProxy(output, gateway_contract, run_fingerprint)
        try:
            proxy.start()
        except BaseException:
            provider_lock.close()
            raise
        atexit.register(proxy.stop)
        configure_environment(args.model, str(proxy.base_url), args.api_key)
        adapter = load_adapter()

    from src.evaluation.durable_model_ledger import (
        DurableModelObserver,
        HashChainLedger,
        ledger_state,
    )
    from src.evaluation.visible_token_budget import TokenCounter

    token_counter = (
        TokenCounter.utf8_bytes(requested_model=args.model,
                                fallback_reason="synthetic_sanity")
        if args.synthetic_sanity else
        TokenCounter.resolve(requested_model=args.model,
                             fallback_encoding="o200k_base",
                             allow_byte_fallback=False)
    )
    stop_requested = threading.Event()

    def signal_handler(signum: int, frame: Any) -> None:
        del signum, frame
        stop_requested.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        label_records = labels["records"]
        for sample_index in args.samples:
            if stop_requested.is_set():
                raise SafeStop("safe stop requested")
            key = str(sample_index)
            if manifest["samples"].get(key, {}).get("status") == "complete":
                continue
            sample_dir = output / "samples" / f"sample-{sample_index}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            ledger = HashChainLedger(
                sample_dir / "operations.jsonl",
                run_id=f"{run_fingerprint}:sample-{sample_index}",
            )
            prior_sample_manifest = sample_dir / "sample_manifest.json"
            if prior_sample_manifest.is_file():
                recovered = json.loads(
                    prior_sample_manifest.read_text(encoding="utf-8")
                )
                ledger_path = sample_dir / "operations.jsonl"
                if (
                    recovered.get("schema") != SAMPLE_SCHEMA
                    or recovered.get("status") != "complete"
                    or recovered.get("sample_index") != sample_index
                    or recovered.get("sample_id")
                    != dataset[sample_index]["sample_id"]
                    or recovered.get("run_fingerprint") != run_fingerprint
                    or recovered.get("ledger_sha256") != file_sha256(ledger_path)
                    or recovered.get("ledger_event_count") != len(ledger.records)
                    or recovered.get("artifact_tree_before_manifest")
                    != tree_descriptor(
                        sample_dir,
                        exclude_relative=frozenset({"sample_manifest.json"}),
                    )
                ):
                    raise GrowthError("unlinked sample manifest is invalid")
                ledger_state(ledger.records)
                manifest_sha = file_sha256(prior_sample_manifest)
                manifest["samples"][key] = {
                    "status": "complete",
                    "completed_at": recovered["completed_at"],
                    "sample_manifest_sha256": manifest_sha,
                }
                atomic_json(output / "run_manifest.json", manifest)
                continue
            observer = DurableModelObserver(
                ledger=ledger,
                artifact_root=sample_dir,
                token_counter=token_counter,
                expected_model=args.model,
                proxy_log=proxy.log if proxy else None,
                formal=not args.synthetic_sanity,
            )
            observer.install(adapter.v8_memory.client.chat.completions)
            manifest["samples"][key] = {
                "status": "running", "started_at": utc_now()
            }
            atomic_json(output / "run_manifest.json", manifest)
            try:
                engine = IncrementalEngine(
                    adapter=adapter,
                    sample=dataset[sample_index],
                    sample_index=sample_index,
                    sample_dir=sample_dir,
                    labels=[record for record in label_records
                            if record["sample_index"] == sample_index],
                    ledger=ledger,
                    observer=observer,
                    run_fingerprint=run_fingerprint,
                    source_mapping_sha256=labels["source_mapping_sha256"],
                    stop_after_operations=args.stop_after_operations,
                    stop_requested=stop_requested,
                )
                result = engine.run()
            finally:
                observer.restore()
            sample_manifest = {
                "schema": SAMPLE_SCHEMA,
                "status": "complete",
                "sample_index": sample_index,
                "sample_id": dataset[sample_index]["sample_id"],
                "run_fingerprint": run_fingerprint,
                "completed_at": utc_now(),
                "result": result,
                "ledger_sha256": file_sha256(sample_dir / "operations.jsonl"),
                "ledger_event_count": len(ledger.records),
                "artifact_tree_before_manifest": tree_descriptor(sample_dir),
            }
            manifest_sha = atomic_json(sample_dir / "sample_manifest.json",
                                       sample_manifest)
            manifest["samples"][key] = {
                "status": "complete",
                "completed_at": sample_manifest["completed_at"],
                "sample_manifest_sha256": manifest_sha,
            }
            atomic_json(output / "run_manifest.json", manifest)
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        atomic_json(output / "run_manifest.json", manifest)
        assert_safe_tree(output)
        return 0
    except SafeStop as exc:
        manifest["status"] = "partial_safe_stop"
        manifest["stopped_at"] = utc_now()
        manifest["stop_reason"] = str(exc)
        atomic_json(output / "run_manifest.json", manifest)
        return 3
    except BaseException as exc:
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
