#!/usr/bin/env python3
"""Shared strict contract for LoCoMo retrieval-only baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CATEGORIES = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
EXPECTED_QUESTIONS = 1986
EXPECTED_PRIMARY = 1540
EXPECTED_ADVERSARIAL = 446
REQUIRED_BUILD_FIELDS = {
    "build_time_s",
    "num_memories",
    "build_calls",
    "build_tokens_in",
    "build_tokens_out",
    "build_llm_time_s",
    "notes",
    "builder",
}
REQUIRED_RETRIEVAL_FIELDS = {
    "latency_s",
    "k",
    "requested_k",
    "returned_count",
    "calls",
    "tokens_in",
    "tokens_out",
}


class ContractError(RuntimeError):
    """Raised when an artifact violates the formal LoCoMo contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON from {path}: {exc}") from exc


def expected_gold(qa: dict[str, Any]) -> str:
    if "answer" in qa:
        return str(qa["answer"])
    if "adversarial_answer" in qa:
        return str(qa["adversarial_answer"])
    raise ContractError("dataset question lacks both answer fields")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or len(payload) != 10:
        raise ContractError("LoCoMo dataset must contain exactly ten conversations")
    categories: Counter[int] = Counter()
    evidence_count = 0
    for sample, item in enumerate(payload):
        if not isinstance(item, dict) or not isinstance(item.get("qa"), list):
            raise ContractError(f"dataset sample {sample} has no QA list")
        for qa in item["qa"]:
            if not isinstance(qa, dict):
                raise ContractError(f"dataset sample {sample} contains invalid QA")
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise ContractError(f"dataset sample {sample} has an empty question")
            try:
                category = int(qa["category"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError(
                    f"dataset sample {sample} has an invalid category"
                ) from exc
            expected_gold(qa)
            evidence = qa.get("evidence")
            if not isinstance(evidence, list) or not all(
                isinstance(value, str) and value.strip() for value in evidence
            ):
                raise ContractError(f"dataset sample {sample} has invalid evidence")
            evidence_count += 1
            categories[category] += 1
    if evidence_count != EXPECTED_QUESTIONS:
        raise ContractError(
            f"LoCoMo dataset has {evidence_count} questions, expected {EXPECTED_QUESTIONS}"
        )
    if dict(sorted(categories.items())) != EXPECTED_CATEGORIES:
        raise ContractError(
            f"LoCoMo category inventory differs: {dict(sorted(categories.items()))}"
        )
    return payload


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _nonnegative_number(value: object) -> bool:
    return _is_number(value) and float(value) >= 0


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_LIST_FAILURES = re.compile(
    r"\b(?:failed_sessions|failed|failures)\s*=\s*\[([^\]]*)\]", re.IGNORECASE
)
_COUNT_FAILURES = re.compile(
    r"\b(?:failed_turns|pages_failed|llm_errors|cascade_failed|"
    r"failed_retryable|failed_permanent)\s*=\s*(-?\d+)",
    re.IGNORECASE,
)
_BUILD_ERROR = re.compile(r"\bbuild_error\s*=\s*([^;,]+)", re.IGNORECASE)
_LOG_FAILURES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"traceback \(most recent call last\)",
        r"\busing fallback\b",
        r"\bfalling back\b",
        r"\bper-session fallback\b",
        r"\bbuild error\b",
        r"\bsearch error\b",
        r"\bretrieve error\b",
        r"\brecall error\b",
        r"\bget_all(?:_memories)? failed\b",
        r"\bfinal flush failed\b",
        r"\bfact count failed\b",
        r"\bprobe failed\b",
        r"\bfailed after retr(?:y|ies)\b",
        r"\bsession\s+\S+\s+failed\b",
        r"\bturn failed\b",
        r"\bserver died\b",
        r"\bserver not healthy\b",
        r"\bdrain timeout\b",
        r"\[error\]",
    )
)


def validate_failure_markers(build: dict[str, Any], log_text: str) -> None:
    notes = str(build.get("notes", ""))
    if not notes.strip():
        raise ContractError("_build_stats.notes is empty")
    if re.search(r"\bfallback\b", notes, re.IGNORECASE):
        raise ContractError("_build_stats records a fallback")
    for match in _LIST_FAILURES.finditer(notes):
        if match.group(1).strip():
            raise ContractError(f"_build_stats records {match.group(0)}")
    for match in _COUNT_FAILURES.finditer(notes):
        if int(match.group(1)) != 0:
            raise ContractError(f"_build_stats records {match.group(0)}")
    build_error = _BUILD_ERROR.search(notes)
    if build_error and build_error.group(1).strip().lower() not in {"none", "null", ""}:
        raise ContractError(f"_build_stats records {build_error.group(0)}")
    for key in (
        "failed_sessions",
        "failed_turns",
        "failed_pages",
        "pages_failed",
        "build_error",
        "error",
        "fallback",
    ):
        if key in build and build[key] not in (None, False, 0, "", [], {}):
            raise ContractError(f"_build_stats.{key} reports failure: {build[key]!r}")
    for pattern in _LOG_FAILURES:
        match = pattern.search(log_text)
        if match:
            raise ContractError(
                f"attempt log contains failure marker: {match.group(0)!r}"
            )


def normalize_sample_records(
    rows: Any,
    dataset_item: dict[str, Any],
    sample: int,
    *,
    builder_model: str,
    builder_base_url: str,
    usage_tracking: str,
) -> list[dict[str, Any]]:
    """Inject dataset evidence and explicit retrieval/build provenance."""
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ContractError(f"sample {sample} output is not a non-empty record list")
    expected = dataset_item.get("qa")
    if not isinstance(expected, list) or len(rows) != len(expected) + 1:
        raise ContractError(f"sample {sample} output count differs from dataset")
    normalized = json.loads(json.dumps(rows, ensure_ascii=False))
    normalized[0]["builder"] = {
        "requested_model": builder_model,
        "base_url": builder_base_url,
        "usage_tracking": usage_tracking,
    }
    for index, (record, source) in enumerate(zip(normalized[1:], expected)):
        if not isinstance(record, dict):
            raise ContractError(f"s{sample}_q{index} is not an object")
        existing_evidence = record.get("evidence")
        if existing_evidence is not None and existing_evidence != source["evidence"]:
            raise ContractError(f"s{sample}_q{index} evidence differs from dataset")
        record["evidence"] = list(source["evidence"])
        retrieval = record.get("retrieval")
        memories = record.get("memories")
        if not isinstance(retrieval, dict) or not isinstance(memories, list):
            raise ContractError(f"s{sample}_q{index} lacks retrieval records")
        legacy_k = retrieval.get("k")
        if "requested_k" not in retrieval:
            retrieval["requested_k"] = legacy_k
        retrieval["returned_count"] = len(memories)
    return normalized


def validate_sample_records(
    rows: Any,
    dataset_item: dict[str, Any],
    sample: int,
    *,
    log_text: str = "",
    expected_builder: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"sample {sample} output is not a non-empty list")
    if not isinstance(rows[0], dict) or rows[0].get("question_id") != "_build_stats":
        raise ContractError(f"sample {sample} lacks leading _build_stats")
    if any(
        isinstance(record, dict) and record.get("question_id") == "_build_stats"
        for record in rows[1:]
    ):
        raise ContractError(f"sample {sample} contains multiple _build_stats")
    build = rows[0]
    missing_build = sorted(REQUIRED_BUILD_FIELDS - set(build))
    if missing_build:
        raise ContractError(f"sample {sample} _build_stats misses fields: {missing_build}")
    if not _nonnegative_number(build["build_time_s"]):
        raise ContractError(f"sample {sample} has invalid build_time_s")
    if not isinstance(build["num_memories"], int) or isinstance(
        build["num_memories"], bool
    ) or build["num_memories"] <= 0:
        raise ContractError(f"sample {sample} has no built memories")
    for key in ("build_calls", "build_tokens_in", "build_tokens_out"):
        if not _nonnegative_integer(build[key]):
            raise ContractError(f"sample {sample} has invalid {key}")
    if not _nonnegative_number(build["build_llm_time_s"]):
        raise ContractError(f"sample {sample} has invalid build_llm_time_s")
    if not isinstance(build["builder"], dict):
        raise ContractError(f"sample {sample} has invalid builder provenance")
    if expected_builder is not None and build["builder"] != expected_builder:
        raise ContractError(f"sample {sample} builder provenance differs")
    validate_failure_markers(build, log_text)

    expected = dataset_item.get("qa")
    if not isinstance(expected, list):
        raise ContractError(f"dataset sample {sample} has no QA list")
    questions = rows[1:]
    if len(questions) != len(expected):
        raise ContractError(
            f"sample {sample} has {len(questions)} questions; expected {len(expected)}"
        )
    for index, (record, source) in enumerate(zip(questions, expected)):
        question_id = f"s{sample}_q{index}"
        if not isinstance(record, dict):
            raise ContractError(f"{question_id} is not an object")
        if record.get("question_id") != question_id:
            raise ContractError(f"{question_id} has a mismatched question_id")
        if record.get("question") != source.get("question"):
            raise ContractError(f"{question_id} question differs from dataset")
        if record.get("gold") != expected_gold(source):
            raise ContractError(f"{question_id} gold differs from dataset")
        if record.get("category") != source.get("category"):
            raise ContractError(f"{question_id} category differs from dataset")
        if record.get("evidence") != source.get("evidence"):
            raise ContractError(f"{question_id} evidence differs from dataset")
        if "answer" in record:
            raise ContractError(f"{question_id} contains an answerer output")
        memories = record.get("memories")
        if not isinstance(memories, list) or not memories:
            raise ContractError(f"{question_id} has empty memories")
        for memory_index, memory in enumerate(memories):
            if not isinstance(memory, dict):
                raise ContractError(f"{question_id} memory {memory_index} is not an object")
            text = memory.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ContractError(f"{question_id} memory {memory_index} has empty text")
            lowered = text.strip().lower()
            if lowered.startswith(("[error]", "[fallback]")) or lowered in {
                "no memory",
                "no memories",
                "no relevant memory",
                "no relevant memory found.",
            }:
                raise ContractError(
                    f"{question_id} memory {memory_index} is a failure placeholder"
                )
            if "date" not in memory or not (
                memory["date"] is None or isinstance(memory["date"], str)
            ):
                raise ContractError(f"{question_id} memory {memory_index} has invalid date")
        retrieval = record.get("retrieval")
        if not isinstance(retrieval, dict):
            raise ContractError(f"{question_id} lacks retrieval metadata")
        missing = sorted(REQUIRED_RETRIEVAL_FIELDS - set(retrieval))
        if missing:
            raise ContractError(f"{question_id} retrieval misses fields: {missing}")
        if not _nonnegative_number(retrieval["latency_s"]):
            raise ContractError(f"{question_id} has invalid retrieval latency")
        for key in ("k", "requested_k"):
            if not isinstance(retrieval[key], int) or isinstance(retrieval[key], bool) or retrieval[key] <= 0:
                raise ContractError(f"{question_id} has invalid retrieval {key}")
        if retrieval["returned_count"] != len(memories):
            raise ContractError(f"{question_id} returned_count differs")
        for key in ("calls", "tokens_in", "tokens_out"):
            if not _nonnegative_integer(retrieval[key]):
                raise ContractError(f"{question_id} has invalid retrieval {key}")
        for key in ("error", "build_error", "fallback"):
            if retrieval.get(key) not in (None, False, 0, "", [], {}):
                raise ContractError(f"{question_id} retrieval reports {key}")
    return {
        "questions": len(questions),
        "evidence_records": len(questions),
        "num_memories": build["num_memories"],
        "build_calls": build["build_calls"],
        "retrieval_calls": sum(record["retrieval"]["calls"] for record in questions),
        "returned_memories": sum(len(record["memories"]) for record in questions),
    }
