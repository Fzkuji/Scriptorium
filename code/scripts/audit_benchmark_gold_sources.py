#!/usr/bin/env python3
"""Independently recompute and audit frozen benchmark gold-source mappings."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "freeze_benchmark_gold_sources.py"
DEFAULT_LOCOMO = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
DEFAULT_LME = (
    ROOT / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
)
DEFAULT_BEAM_CACHE = ROOT / "benchmarks" / "beam" / "hf_cache"
DEFAULT_BEAM_RUNNER = ROOT / "scripts" / "run_v88_gpt55_beam.py"
DEFAULT_ARTIFACT_DIR = (
    ROOT / "results" / "paper-experiments-20260714" / "evidence-mapping" / "v1"
)

SCHEMA_VERSION = "benchmark-gold-source-mapping/v1"
ARTIFACT_STEM = "evidence_mapping.v1"
MANIFEST_NAME = f"{ARTIFACT_STEM}.manifest.json"
QUESTIONS_NAME = f"{ARTIFACT_STEM}.questions.jsonl"
CHECKSUM_NAME = f"{ARTIFACT_STEM}.manifest.sha256"
AUDIT_NAME = f"{ARTIFACT_STEM}.audit.json"
AUDIT_CHECKSUM_NAME = f"{ARTIFACT_STEM}.audit.sha256"
LOCK_NAME = ".evidence_mapping.lock"

BEAM_DATASET = "Mohammadta/BEAM"
BEAM_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
BEAM_REVISION_PARTS = (
    "Mohammadta___beam",
    "default",
    "0.0.0",
    BEAM_REVISION,
)
BEAM_SCOPE = {"100K": tuple(range(10)), "1M": tuple(range(35))}
EXPECTED_BEAM_ROWS = {"100K": 20, "1M": 35}
EXPECTED_CATEGORIES = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}

NORMALIZATION_RULES = {
    "locomo": [
        {
            "id": "identity",
            "rule": "Keep canonical D<session>:<turn> anchors unchanged.",
        },
        {
            "id": "split_compound_anchor",
            "rule": (
                "Split a field consisting only of two or more canonical "
                "anchors separated by whitespace, semicolon, or comma."
            ),
        },
        {
            "id": "strip_leading_zero",
            "rule": (
                "Canonicalize numeric components with int(), e.g. D30:05 to D30:5."
            ),
        },
        {
            "id": "frozen_typo_D_colon_session_colon_turn",
            "rule": "Normalize the exact released value D:11:26 to D11:26.",
        },
        {
            "id": "unsupported_anchor_syntax",
            "rule": "Do not infer an anchor from any other malformed value.",
        },
    ],
    "longmemeval_s": [
        {
            "id": "session_id_identity",
            "rule": "Stringify answer_session_ids without turn-level expansion.",
        }
    ],
    "beam": [
        {
            "id": "recursive_source_chat_ids",
            "rule": (
                "Recursively visit dict values in sorted-key order and "
                "list values in released order, then stringify integer IDs."
            ),
        },
        {
            "id": "runner_source_id_map",
            "rule": (
                "Expand each raw source ID through the pinned runner's "
                "conversation_to_native source_id_map and preserve all Dn:m occurrences."
            ),
        },
    ],
}


class AuditError(RuntimeError):
    """Raised when an artifact differs from an independent recomputation."""


def absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def no_symlink(path: Path, *, must_exist: bool = False) -> Path:
    checked = absolute_path(path)
    current = Path(checked.anchor)
    for part in checked.parts[1:]:
        current /= part
        if current.is_symlink():
            raise AuditError(f"symlink paths are not allowed: {current}")
    if must_exist and not checked.exists():
        raise AuditError(f"required path does not exist: {checked}")
    return checked


def regular_file(path: Path) -> Path:
    checked = no_symlink(path, must_exist=True)
    if not checked.is_file():
        raise AuditError(f"not a regular file: {checked}")
    return checked


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    checked = regular_file(path)
    digest = hashlib.sha256()
    with checked.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    checked = regular_file(path)
    try:
        with checked.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read valid JSON from {checked}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    checked = regular_file(path)
    records: list[dict[str, Any]] = []
    try:
        with checked.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise AuditError(f"blank JSONL line at {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditError(f"JSONL line {line_number} is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read valid JSONL from {checked}: {exc}") from exc
    return records


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n"
    ).encode("utf-8")


def source_path(path: Path) -> str:
    checked = absolute_path(path)
    try:
        return str(checked.relative_to(ROOT))
    except ValueError:
        return str(checked)


def git_source(path: Path) -> dict[str, Any]:
    checked = regular_file(path)
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=checked.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=git_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"cannot obtain git revision for {checked}") from exc
    return {
        "kind": "git_checkout_file",
        "path": source_path(checked),
        "sha256": sha256_file(checked),
        "git_root": source_path(Path(git_root)),
        "git_revision": revision,
    }


@contextmanager
def artifact_lock(artifact_dir: Path) -> Iterator[Path]:
    checked = no_symlink(artifact_dir, must_exist=True)
    if not checked.is_dir():
        raise AuditError(f"artifact path is not a directory: {checked}")
    lock_path = regular_file(checked / LOCK_NAME)
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AuditError(f"artifact directory is locked: {checked}") from exc
        yield checked
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_no_collision(path: Path, payload: bytes) -> None:
    target = no_symlink(path)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise AuditError(f"refusing to overwrite different audit file: {target}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def normalize_locomo(raw: str) -> tuple[list[str], str]:
    text = raw.strip()
    if text == "D:11:26":
        return ["D11:26"], "frozen_typo_D_colon_session_colon_turn"
    compound = re.fullmatch(
        r"\s*D[0-9]+:[0-9]+"
        r"(?:(?:\s*;\s*|\s*,\s*|\s+)D[0-9]+:[0-9]+)+\s*",
        text,
    )
    if compound:
        return [
            f"D{int(session)}:{int(turn)}"
            for session, turn in re.findall(r"D([0-9]+):([0-9]+)", text)
        ], "split_compound_anchor"
    match = re.fullmatch(r"D([0-9]+):([0-9]+)", text)
    if not match:
        return [], "unsupported_anchor_syntax"
    session, turn = match.groups()
    if int(session) < 1 or int(turn) < 1:
        return [], "nonpositive_anchor_component"
    normalized = f"D{int(session)}:{int(turn)}"
    return [normalized], "identity" if normalized == text else "strip_leading_zero"


def recompute_locomo(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list) or len(data) != 10:
        raise AuditError("LoCoMo source inventory differs from ten conversations")
    records: list[dict[str, Any]] = []
    categories: Counter[int] = Counter()
    excluded: list[dict[str, Any]] = []
    normalization_counts: Counter[str] = Counter()
    anchor_counts: list[int] = []
    for sample_index, sample in enumerate(data):
        if not isinstance(sample, dict) or not isinstance(sample.get("qa"), list):
            raise AuditError(f"LoCoMo sample {sample_index} is invalid")
        sample_id = str(sample.get("sample_id", "")).strip()
        conversation = sample.get("conversation")
        if not sample_id or not isinstance(conversation, dict):
            raise AuditError(f"LoCoMo sample {sample_index} lacks identity/history")
        anchor_list = [
            turn.get("dia_id")
            for key, turns in conversation.items()
            if key.startswith("session_") and isinstance(turns, list)
            for turn in turns
            if isinstance(turn, dict)
        ]
        if not anchor_list or len(anchor_list) != len(set(anchor_list)) or any(
            not isinstance(value, str)
            or not re.fullmatch(r"D[1-9][0-9]*:[1-9][0-9]*", value)
            for value in anchor_list
        ):
            raise AuditError(f"LoCoMo sample {sample_index} has invalid anchors")
        anchors = set(anchor_list)
        anchor_counts.append(len(anchors))
        for question_index, question in enumerate(sample["qa"]):
            if not isinstance(question, dict):
                raise AuditError("LoCoMo QA entry is not an object")
            evidence = question.get("evidence")
            text = question.get("question")
            if not isinstance(evidence, list) or not all(
                isinstance(value, str) and value.strip() for value in evidence
            ):
                if evidence != []:
                    raise AuditError("LoCoMo evidence is not a string list")
            if not isinstance(text, str) or not text.strip():
                raise AuditError("LoCoMo question is empty")
            category = int(question["category"])
            categories[category] += 1
            normalized: list[str] = []
            normalization: list[dict[str, Any]] = []
            malformed: list[str] = []
            for raw in evidence:
                values, reason = normalize_locomo(raw)
                normalization_counts[reason] += 1
                normalization.append(
                    {"raw": raw, "normalized": values, "reason": reason}
                )
                normalized.extend(values)
                if not values:
                    malformed.append(raw)
            missing = [value for value in normalized if value not in anchors]
            reasons: list[str] = []
            if not evidence:
                reasons.append("gold_evidence_empty")
            reasons.extend(f"malformed_gold_anchor:{value}" for value in malformed)
            reasons.extend(f"gold_anchor_not_found:{value}" for value in missing)
            primary = category in {1, 2, 3, 4}
            question_id = f"locomo:{sample_id}:q{question_index:03d}"
            record = {
                "schema_version": 1,
                "benchmark": "LoCoMo",
                "question_id": question_id,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "question_index": question_index,
                "question": text,
                "category": category,
                "gold_source_granularity": "turn",
                "raw_gold_source_ids": evidence,
                "normalization": normalization,
                "normalized_source_ids": normalized,
                "gold_source_ids": [value for value in normalized if value in anchors],
                "missing_source_ids": missing,
                "qa_scoring_eligible": primary,
                "source_recall_eligible": primary and not reasons,
                "source_recall_exclusion_reasons": (
                    reasons
                    if primary
                    else ["category_5_outside_primary_source_recall_scope"]
                ),
            }
            records.append(record)
            if primary and reasons:
                excluded.append(
                    {
                        "question_id": question_id,
                        "raw_gold_source_ids": evidence,
                        "normalized_source_ids": normalized,
                        "missing_source_ids": missing,
                        "reasons": reasons,
                    }
                )
    actual_categories = dict(sorted(categories.items()))
    mapped = sum(record["source_recall_eligible"] for record in records)
    actual = (
        len(records),
        sum(categories[value] for value in (1, 2, 3, 4)),
        categories[5],
        mapped,
        len(excluded),
        actual_categories,
    )
    expected = (1986, 1540, 446, 1533, 7, EXPECTED_CATEGORIES)
    if actual != expected:
        raise AuditError(f"LoCoMo independent count mismatch: {actual}")
    summary = {
        "questions": 1986,
        "primary_qa_questions": 1540,
        "qa_scoring_question_denominator": sum(
            record["qa_scoring_eligible"] for record in records
        ),
        "category_5_questions": 446,
        "source_recall_question_denominator": 1533,
        "primary_source_recall_exclusions": 7,
        "categories": actual_categories,
        "gold_source_granularity": "turn",
        "qa_scoring_note": (
            "All 1,540 category 1--4 questions remain in QA scoring; the seven "
            "listed questions are excluded only from source-recall scoring."
        ),
        "sample_anchor_counts": anchor_counts,
        "normalization_reason_counts": dict(sorted(normalization_counts.items())),
        "source_recall_exclusions": excluded,
    }
    return records, summary


def recompute_lme(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise AuditError("LongMemEval-S source is not a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_haystack_items = 0
    duplicate_haystack_occurrences = 0
    for dataset_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise AuditError(f"LongMemEval-S item {dataset_index} is invalid")
        dataset_question_id = str(item.get("question_id", "")).strip()
        question = item.get("question")
        answer_ids = item.get("answer_session_ids")
        haystack_ids = item.get("haystack_session_ids")
        if (
            not dataset_question_id
            or dataset_question_id in seen
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise AuditError(f"LongMemEval-S item {dataset_index} has bad identity")
        seen.add(dataset_question_id)
        if not isinstance(answer_ids, list) or not answer_ids:
            raise AuditError(f"LongMemEval-S item {dataset_index} has no gold sessions")
        if not isinstance(haystack_ids, list) or not haystack_ids:
            raise AuditError(f"LongMemEval-S item {dataset_index} has no history IDs")
        gold = [str(value).strip() for value in answer_ids]
        haystack = [str(value).strip() for value in haystack_ids]
        if any(not value for value in gold + haystack) or any(
            value not in haystack for value in gold
        ):
            raise AuditError(f"LongMemEval-S item {dataset_index} has invalid mapping")
        duplicate_count = len(haystack) - len(set(haystack))
        duplicate_haystack_items += int(duplicate_count > 0)
        duplicate_haystack_occurrences += duplicate_count
        records.append(
            {
                "schema_version": 1,
                "benchmark": "LongMemEval-S",
                "question_id": f"longmemeval-s:{dataset_question_id}",
                "dataset_question_id": dataset_question_id,
                "dataset_index": dataset_index,
                "question": question,
                "question_type": item.get("question_type"),
                "gold_source_granularity": "session",
                "mapping_level": "session",
                "turn_level_gold_mapping": False,
                "raw_gold_source_ids": answer_ids,
                "normalization": [
                    {
                        "raw": raw,
                        "normalized": normalized,
                        "reason": (
                            "identity" if raw == normalized else "stringify_session_id"
                        ),
                    }
                    for raw, normalized in zip(answer_ids, gold)
                ],
                "normalized_source_ids": gold,
                "gold_source_ids": gold,
                "missing_source_ids": [],
                "qa_scoring_eligible": True,
                "source_recall_eligible": True,
                "source_recall_exclusion_reasons": [],
            }
        )
    if len(records) != 500:
        raise AuditError(f"LongMemEval-S independent count is {len(records)}, not 500")
    summary = {
        "questions": 500,
        "qa_scoring_questions": 500,
        "source_recall_question_denominator": 500,
        "source_recall_exclusions": [],
        "gold_source_granularity": "session",
        "mapping_level": "session",
        "turn_level_gold_mapping": False,
        "items_with_duplicate_haystack_session_ids": duplicate_haystack_items,
        "duplicate_haystack_session_id_occurrences": (
            duplicate_haystack_occurrences
        ),
        "granularity_note": (
            "answer_session_ids identify evidence sessions. This artifact does "
            "not define or claim turn-level LongMemEval-S gold sources."
        ),
    }
    return records, summary


def import_beam_runner(path: Path) -> Any:
    checked = regular_file(path)
    spec = importlib.util.spec_from_file_location("audited_beam_runner", checked)
    if spec is None or spec.loader is None:
        raise AuditError(f"cannot import BEAM runner: {checked}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.DEFAULT_DATASET_REVISION != BEAM_REVISION:
        raise AuditError("BEAM runner's pinned revision differs")
    return module


def flatten_beam(value: Any) -> list[tuple[Any, str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[tuple[Any, str, str]] = []
        for key in sorted(value.keys()):
            result += flatten_beam(value[key])
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            result += flatten_beam(child)
        return result
    if isinstance(value, bool):
        raise AuditError("BEAM gold source contains a boolean")
    if isinstance(value, int):
        return [(value, str(value), "integer_to_string")]
    if isinstance(value, str) and value.strip():
        stripped = value.strip()
        return [
            (value, stripped, "identity" if value == stripped else "strip_whitespace")
        ]
    raise AuditError(f"BEAM gold source scalar is invalid: {value!r}")


def recompute_beam(
    cache_dir: Path, runner_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from datasets import Dataset

    runner = import_beam_runner(runner_path)
    revision_dir = no_symlink(
        absolute_path(cache_dir).joinpath(*BEAM_REVISION_PARTS), must_exist=True
    )
    info_path = regular_file(revision_dir / "dataset_info.json")
    info = read_json(info_path)
    split_info = info.get("splits", {}) if isinstance(info, dict) else {}
    records: list[dict[str, Any]] = []
    empty_types: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    cache_files: dict[str, dict[str, Any]] = {
        "dataset_info.json": {
            "path": source_path(info_path),
            "sha256": sha256_file(info_path),
        }
    }
    for split, indices in BEAM_SCOPE.items():
        arrow_path = regular_file(revision_dir / f"beam-{split}.arrow")
        cache_files[arrow_path.name] = {
            "path": source_path(arrow_path),
            "sha256": sha256_file(arrow_path),
        }
        expected_rows = EXPECTED_BEAM_ROWS[split]
        if split_info.get(split, {}).get("num_examples") != expected_rows:
            raise AuditError(f"BEAM dataset_info differs for {split}")
        dataset = Dataset.from_file(str(arrow_path))
        if len(dataset) != expected_rows:
            raise AuditError(f"BEAM Arrow rows differ for {split}")
        for conversation_index in indices:
            row = dataset[conversation_index]
            native, stats = runner.conversation_to_native(row)
            source_map = stats.get("source_id_map")
            if not isinstance(source_map, dict):
                raise AuditError("BEAM source_id_map is absent")
            anchor_set = {
                turn["dia_id"]
                for key, turns in native.items()
                if key.startswith("session_") and isinstance(turns, list)
                for turn in turns
            }
            if any(
                not isinstance(key, str)
                or not isinstance(values, list)
                or not values
                or any(value not in anchor_set for value in values)
                for key, values in source_map.items()
            ):
                raise AuditError("BEAM source_id_map does not resolve to chat anchors")
            questions = runner.extract_questions(row)
            if len(questions) != 20:
                raise AuditError("BEAM conversation does not contain 20 questions")
            conversation_id = str(row.get("conversation_id", "")).strip()
            if not conversation_id:
                raise AuditError("BEAM conversation_id is empty")
            for question_index, question in enumerate(questions):
                question_type = str(question.get("question_type", "")).strip()
                raw_sources = question.get("source_chat_ids")
                flattened = flatten_beam(raw_sources)
                normalized = [entry[1] for entry in flattened]
                normalization: list[dict[str, Any]] = []
                mapped: list[str] = []
                missing: list[str] = []
                for raw, normalized_id, reason in flattened:
                    mapped_ids = source_map.get(normalized_id, [])
                    normalization.append(
                        {
                            "raw": raw,
                            "normalized": normalized_id,
                            "reason": reason,
                            "mapped_source_ids": mapped_ids,
                        }
                    )
                    if mapped_ids:
                        mapped.extend(mapped_ids)
                    else:
                        missing.append(normalized_id)
                mapped = list(dict.fromkeys(mapped))
                missing = list(dict.fromkeys(missing))
                if missing:
                    raise AuditError(
                        f"BEAM independently found unmapped source IDs: {missing}"
                    )
                eligible = bool(normalized)
                if not eligible:
                    empty_types[question_type] += 1
                split_counts[split] += 1
                question_id = (
                    f"beam:{split}:c{conversation_index:03d}:"
                    f"q{question_index:02d}:{question_type}"
                )
                records.append(
                    {
                        "schema_version": 1,
                        "benchmark": "BEAM",
                        "question_id": question_id,
                        "chat_size": split,
                        "conversation_index": conversation_index,
                        "conversation_id": conversation_id,
                        "question_index": question_index,
                        "question": question.get("question_text"),
                        "question_type": question_type,
                        "gold_source_granularity": "turn",
                        "raw_gold_source_ids": raw_sources,
                        "normalization": normalization,
                        "normalized_source_ids": normalized,
                        "gold_source_ids": mapped,
                        "missing_source_ids": [],
                        "qa_scoring_eligible": True,
                        "source_recall_eligible": eligible,
                        "source_recall_exclusion_reasons": (
                            []
                            if eligible
                            else [f"no_gold_source_ids:{question_type}"]
                        ),
                        "source_recall_empty_class": (
                            None
                            if eligible
                            else (
                                "abstention"
                                if question_type == "abstention"
                                else "other"
                            )
                        ),
                    }
                )
    mapped_count = sum(record["source_recall_eligible"] for record in records)
    empty_count = len(records) - mapped_count
    actual = (
        len(records),
        mapped_count,
        empty_count,
        empty_types["abstention"],
        empty_count - empty_types["abstention"],
        dict(sorted(split_counts.items())),
    )
    expected = (900, 804, 96, 90, 6, {"100K": 200, "1M": 700})
    if actual != expected:
        raise AuditError(f"BEAM independent count mismatch: {actual}")
    exclusions = [
        {
            "question_id": record["question_id"],
            "question_type": record["question_type"],
            "reasons": record["source_recall_exclusion_reasons"],
        }
        for record in records
        if not record["source_recall_eligible"]
    ]
    summary = {
        "questions": 900,
        "qa_scoring_questions": 900,
        "source_recall_question_denominator": 804,
        "questions_without_gold_sources": 96,
        "empty_abstention_questions": 90,
        "empty_other_questions": 6,
        "split_question_counts": dict(sorted(split_counts.items())),
        "formal_scope": {
            "100K": {"conversation_indices": list(BEAM_SCOPE["100K"])},
            "1M": {"conversation_indices": list(BEAM_SCOPE["1M"])},
        },
        "gold_source_granularity": "turn",
        "empty_question_types": dict(sorted(empty_types.items())),
        "source_recall_exclusions": exclusions,
    }
    runner_path = regular_file(runner_path)
    source = {
        "kind": "pinned_huggingface_cache",
        "dataset": BEAM_DATASET,
        "config": "default",
        "revision": BEAM_REVISION,
        "cache_revision_dir": source_path(revision_dir),
        "files": cache_files,
        "runner_source_id_map": {
            "path": source_path(runner_path),
            "sha256": sha256_file(runner_path),
            "function": "conversation_to_native",
            "rule": (
                "Non-empty normalized chat turns receive batch-local Dn:m IDs; "
                "each raw turn id maps to every corresponding Dn:m occurrence."
            ),
        },
    }
    return records, summary, source


def expected_bundle(
    *,
    locomo_dataset: Path,
    lme_dataset: Path,
    beam_cache_dir: Path,
    beam_runner: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    locomo_records, locomo_counts = recompute_locomo(locomo_dataset)
    lme_records, lme_counts = recompute_lme(lme_dataset)
    beam_records, beam_counts, beam_source = recompute_beam(
        beam_cache_dir, beam_runner
    )
    records = [*locomo_records, *lme_records, *beam_records]
    identifiers = [record["question_id"] for record in records]
    if len(records) != 3386 or len(identifiers) != len(set(identifiers)):
        raise AuditError("independent combined inventory is not 3,386 unique questions")
    question_bytes = canonical_jsonl(records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_version": 1,
        "description": (
            "Frozen question-level gold-source mappings and source-recall "
            "denominators. This artifact is data-only and makes no model calls."
        ),
        "questions_file": QUESTIONS_NAME,
        "questions_sha256": sha256_bytes(question_bytes),
        "question_count": len(records),
        "normalization_rules": NORMALIZATION_RULES,
        "benchmarks": {
            "LoCoMo": {
                "source": git_source(locomo_dataset),
                "counts": locomo_counts,
            },
            "LongMemEval-S": {
                "source": git_source(lme_dataset),
                "counts": lme_counts,
            },
            "BEAM": {"source": beam_source, "counts": beam_counts},
        },
        "tool_sources": {
            "generator": {
                "path": source_path(GENERATOR),
                "sha256": sha256_file(GENERATOR),
            },
            "auditor": {
                "path": source_path(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    return records, manifest


def verify_checksum(path: Path, expected_hash: str) -> None:
    text = regular_file(path).read_text(encoding="ascii")
    expected = f"{expected_hash}  {MANIFEST_NAME}\n"
    if text != expected:
        raise AuditError(f"manifest checksum file differs: {path}")


def audit(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    locomo_dataset: Path = DEFAULT_LOCOMO,
    lme_dataset: Path = DEFAULT_LME,
    beam_cache_dir: Path = DEFAULT_BEAM_CACHE,
    beam_runner: Path = DEFAULT_BEAM_RUNNER,
) -> dict[str, Any]:
    with artifact_lock(artifact_dir) as checked_dir:
        manifest_path = regular_file(checked_dir / MANIFEST_NAME)
        questions_path = regular_file(checked_dir / QUESTIONS_NAME)
        checksum_path = regular_file(checked_dir / CHECKSUM_NAME)
        manifest = read_json(manifest_path)
        records = read_jsonl(questions_path)
        raw_questions = questions_path.read_bytes()
        if raw_questions != canonical_jsonl(records):
            raise AuditError("question JSONL is not in canonical byte format")
        if manifest.get("questions_sha256") != sha256_bytes(raw_questions):
            raise AuditError("manifest question hash does not match JSONL bytes")
        raw_manifest = manifest_path.read_bytes()
        if raw_manifest != canonical_json(manifest):
            raise AuditError("manifest JSON is not in canonical byte format")
        verify_checksum(checksum_path, sha256_bytes(raw_manifest))

        expected_records, expected_manifest = expected_bundle(
            locomo_dataset=locomo_dataset,
            lme_dataset=lme_dataset,
            beam_cache_dir=beam_cache_dir,
            beam_runner=beam_runner,
        )
        # JSON object keys are strings on disk (notably LoCoMo category keys).
        expected_manifest = json.loads(canonical_json(expected_manifest))
        if records != expected_records:
            for index, (actual, expected) in enumerate(
                zip(records, expected_records, strict=False)
            ):
                if actual != expected:
                    raise AuditError(
                        f"question record {index} differs: "
                        f"{actual.get('question_id')} != {expected.get('question_id')}"
                    )
            raise AuditError(
                f"question record count differs: {len(records)} != "
                f"{len(expected_records)}"
            )
        if manifest != expected_manifest:
            differing = sorted(
                key
                for key in set(manifest) | set(expected_manifest)
                if manifest.get(key) != expected_manifest.get(key)
            )
            raise AuditError(f"manifest differs in top-level fields: {differing}")

        report = {
            "schema_version": "benchmark-gold-source-mapping-audit/v1",
            "status": "passed",
            "manifest_file": MANIFEST_NAME,
            "manifest_sha256": sha256_bytes(raw_manifest),
            "questions_file": QUESTIONS_NAME,
            "questions_sha256": sha256_bytes(raw_questions),
            "question_count": len(records),
            "source_recall_question_denominators": {
                "LoCoMo": 1533,
                "LongMemEval-S": 500,
                "BEAM": 804,
            },
            "qa_scoring_question_counts": {
                "LoCoMo_primary": 1540,
                "LongMemEval-S": 500,
                "BEAM": 900,
            },
            "auditor_sha256": sha256_file(Path(__file__)),
        }
        report_payload = canonical_json(report)
        report_checksum = (
            f"{sha256_bytes(report_payload)}  {AUDIT_NAME}\n"
        ).encode("ascii")
        report_path = checked_dir / AUDIT_NAME
        report_checksum_path = checked_dir / AUDIT_CHECKSUM_NAME
        for path, payload in (
            (report_path, report_payload),
            (report_checksum_path, report_checksum),
        ):
            if path.exists() and (
                path.is_symlink() or not path.is_file() or path.read_bytes() != payload
            ):
                raise AuditError(f"different audit artifact already exists: {path}")
        atomic_no_collision(report_path, report_payload)
        atomic_no_collision(report_checksum_path, report_checksum)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit benchmark gold-source mapping artifacts"
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--locomo-dataset", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--longmemeval-dataset", type=Path, default=DEFAULT_LME)
    parser.add_argument("--beam-cache-dir", type=Path, default=DEFAULT_BEAM_CACHE)
    parser.add_argument("--beam-runner", type=Path, default=DEFAULT_BEAM_RUNNER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit(
        artifact_dir=args.artifact_dir,
        locomo_dataset=args.locomo_dataset,
        lme_dataset=args.longmemeval_dataset,
        beam_cache_dir=args.beam_cache_dir,
        beam_runner=args.beam_runner,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
