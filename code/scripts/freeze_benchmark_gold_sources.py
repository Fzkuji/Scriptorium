#!/usr/bin/env python3
"""Freeze benchmark gold-source mappings without making model calls.

The artifact produced here defines the question-level denominators used by
source-recall experiments.  It covers the full local LoCoMo and LongMemEval-S
datasets and the formal BEAM scope (100K conversations 0--9 and all 1M
conversations).  Existing artifacts are never overwritten with different
bytes.
"""

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
DEFAULT_LOCOMO = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
DEFAULT_LME = (
    ROOT / "benchmarks" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
)
DEFAULT_BEAM_CACHE = ROOT / "benchmarks" / "beam" / "hf_cache"
DEFAULT_BEAM_RUNNER = ROOT / "scripts" / "run_v88_gpt55_beam.py"
DEFAULT_OUTPUT = (
    ROOT / "results" / "paper-experiments-20260714" / "evidence-mapping" / "v1"
)
AUDITOR = ROOT / "scripts" / "audit_benchmark_gold_sources.py"

SCHEMA_VERSION = "benchmark-gold-source-mapping/v1"
ARTIFACT_STEM = "evidence_mapping.v1"
MANIFEST_NAME = f"{ARTIFACT_STEM}.manifest.json"
QUESTIONS_NAME = f"{ARTIFACT_STEM}.questions.jsonl"
CHECKSUM_NAME = f"{ARTIFACT_STEM}.manifest.sha256"
LOCK_NAME = ".evidence_mapping.lock"

BEAM_DATASET = "Mohammadta/BEAM"
BEAM_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
BEAM_CACHE_COMPONENTS = (
    "Mohammadta___beam",
    "default",
    "0.0.0",
    BEAM_REVISION,
)
BEAM_SCOPE = {"100K": tuple(range(10)), "1M": tuple(range(35))}
EXPECTED_BEAM_ROWS = {"100K": 20, "1M": 35}
EXPECTED_CATEGORIES = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}

_CANONICAL_ANCHOR = re.compile(r"^D([0-9]+):([0-9]+)$")
_COMPOUND_ANCHORS = re.compile(
    r"^\s*D[0-9]+:[0-9]+"
    r"(?:(?:\s*;\s*|\s*,\s*|\s+)D[0-9]+:[0-9]+)+\s*$"
)
_ANCHOR_WITHIN_COMPOUND = re.compile(r"D([0-9]+):([0-9]+)")


class FreezeError(RuntimeError):
    """Raised when source data or an output path violates the frozen contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def ensure_no_symlink(path: Path, *, require_exists: bool = False) -> Path:
    """Reject symlinks in every existing component, including broken links."""
    absolute = absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise FreezeError(f"symlink paths are not allowed: {current}")
    if require_exists and not absolute.exists():
        raise FreezeError(f"required path does not exist: {absolute}")
    return absolute


def require_regular_file(path: Path) -> Path:
    checked = ensure_no_symlink(path, require_exists=True)
    if not checked.is_file():
        raise FreezeError(f"required regular file is missing: {checked}")
    return checked


def sha256_file(path: Path) -> str:
    checked = require_regular_file(path)
    digest = hashlib.sha256()
    with checked.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    checked = require_regular_file(path)
    try:
        with checked.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot read valid JSON from {checked}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def collision_safe_atomic_write(path: Path, payload: bytes) -> str:
    target = ensure_no_symlink(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink(target.parent, require_exists=True)
    if target.exists():
        if not target.is_file():
            raise FreezeError(f"artifact path is not a regular file: {target}")
        existing = target.read_bytes()
        if existing != payload:
            raise FreezeError(f"refusing to overwrite different artifact: {target}")
        return "unchanged"

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
    return "written"


def preflight_collisions(payloads: dict[Path, bytes]) -> None:
    identities = [str(absolute_without_resolving(path)).casefold() for path in payloads]
    if len(set(identities)) != len(identities):
        raise FreezeError("artifact output paths collide")
    for path, payload in payloads.items():
        checked = ensure_no_symlink(path)
        if checked.exists():
            if not checked.is_file():
                raise FreezeError(f"artifact path is not a regular file: {checked}")
            if checked.read_bytes() != payload:
                raise FreezeError(
                    f"refusing to overwrite different artifact: {checked}"
                )


@contextmanager
def exclusive_output_lock(output_dir: Path) -> Iterator[Path]:
    checked = ensure_no_symlink(output_dir)
    checked.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink(checked, require_exists=True)
    lock_path = checked / LOCK_NAME
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise FreezeError(f"cannot open output lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FreezeError(f"output directory is locked: {checked}") from exc
        yield checked
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def git_revision(path: Path) -> tuple[str, str]:
    checked = ensure_no_symlink(path, require_exists=True)
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=checked.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeError(f"cannot determine dataset revision for {checked}") from exc
    return root, revision


def source_path(path: Path) -> str:
    checked = absolute_without_resolving(path)
    try:
        return str(checked.relative_to(ROOT))
    except ValueError:
        return str(checked)


def normalize_anchor(raw: str) -> tuple[list[str], str]:
    """Apply only the explicitly frozen LoCoMo evidence corrections."""
    text = raw.strip()
    if text == "D:11:26":
        return ["D11:26"], "frozen_typo_D_colon_session_colon_turn"
    if _COMPOUND_ANCHORS.fullmatch(text):
        values = [
            f"D{int(session)}:{int(turn)}"
            for session, turn in _ANCHOR_WITHIN_COMPOUND.findall(text)
        ]
        return values, "split_compound_anchor"
    match = _CANONICAL_ANCHOR.fullmatch(text)
    if not match:
        return [], "unsupported_anchor_syntax"
    session, turn = match.groups()
    if int(session) <= 0 or int(turn) <= 0:
        return [], "nonpositive_anchor_component"
    normalized = f"D{int(session)}:{int(turn)}"
    if normalized != text:
        return [normalized], "strip_leading_zero"
    return [normalized], "identity"


def _conversation_anchors(sample: dict[str, Any], sample_index: int) -> set[str]:
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        raise FreezeError(f"LoCoMo sample {sample_index} lacks a conversation")
    anchors: list[str] = []
    for key, value in conversation.items():
        if not key.startswith("session_") or not isinstance(value, list):
            continue
        for turn in value:
            if not isinstance(turn, dict):
                raise FreezeError(
                    f"LoCoMo sample {sample_index} has a non-object turn"
                )
            anchor = turn.get("dia_id")
            if not isinstance(anchor, str) or not re.fullmatch(
                r"D[1-9][0-9]*:[1-9][0-9]*", anchor
            ):
                raise FreezeError(
                    f"LoCoMo sample {sample_index} has invalid dia_id {anchor!r}"
                )
            anchors.append(anchor)
    if not anchors or len(anchors) != len(set(anchors)):
        raise FreezeError(
            f"LoCoMo sample {sample_index} has no anchors or duplicate dia_ids"
        )
    return set(anchors)


def build_locomo_records(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset = read_json(dataset_path)
    if not isinstance(dataset, list) or len(dataset) != 10:
        raise FreezeError("LoCoMo must contain exactly ten conversations")
    records: list[dict[str, Any]] = []
    categories: Counter[int] = Counter()
    exclusions: list[dict[str, Any]] = []
    normalization_reasons: Counter[str] = Counter()
    sample_anchor_counts: list[int] = []

    for sample_index, sample in enumerate(dataset):
        if not isinstance(sample, dict) or not isinstance(sample.get("qa"), list):
            raise FreezeError(f"LoCoMo sample {sample_index} lacks a QA list")
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            raise FreezeError(f"LoCoMo sample {sample_index} lacks sample_id")
        anchors = _conversation_anchors(sample, sample_index)
        sample_anchor_counts.append(len(anchors))
        for question_index, question in enumerate(sample["qa"]):
            if not isinstance(question, dict):
                raise FreezeError("LoCoMo QA entry is not an object")
            text = question.get("question")
            evidence = question.get("evidence")
            if not isinstance(text, str) or not text.strip():
                raise FreezeError("LoCoMo QA entry has an empty question")
            if not isinstance(evidence, list) or not all(
                isinstance(value, str) and value.strip() for value in evidence
            ):
                if evidence != []:
                    raise FreezeError("LoCoMo evidence must be a list of strings")
            try:
                category = int(question["category"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FreezeError("LoCoMo QA entry has invalid category") from exc
            categories[category] += 1

            normalized: list[str] = []
            normalization: list[dict[str, Any]] = []
            malformed: list[str] = []
            for raw in evidence:
                values, reason = normalize_anchor(raw)
                normalization_reasons[reason] += 1
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
            eligible = primary and not reasons
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
                "source_recall_eligible": eligible,
                "source_recall_exclusion_reasons": (
                    reasons
                    if primary
                    else ["category_5_outside_primary_source_recall_scope"]
                ),
            }
            records.append(record)
            if primary and reasons:
                exclusions.append(
                    {
                        "question_id": question_id,
                        "raw_gold_source_ids": evidence,
                        "normalized_source_ids": normalized,
                        "missing_source_ids": missing,
                        "reasons": reasons,
                    }
                )

    actual_categories = dict(sorted(categories.items()))
    primary_total = sum(categories[value] for value in (1, 2, 3, 4))
    mapped_primary = sum(record["source_recall_eligible"] for record in records)
    qa_denominator = sum(record["qa_scoring_eligible"] for record in records)
    expected = {
        "questions": 1986,
        "primary_qa_questions": 1540,
        "qa_scoring_question_denominator": 1540,
        "category_5_questions": 446,
        "source_recall_question_denominator": 1533,
        "primary_source_recall_exclusions": 7,
        "categories": EXPECTED_CATEGORIES,
    }
    actual = {
        "questions": len(records),
        "primary_qa_questions": primary_total,
        "qa_scoring_question_denominator": qa_denominator,
        "category_5_questions": categories[5],
        "source_recall_question_denominator": mapped_primary,
        "primary_source_recall_exclusions": len(exclusions),
        "categories": actual_categories,
    }
    if actual != expected:
        raise FreezeError(
            f"LoCoMo frozen count mismatch: expected {expected}, got {actual}"
        )
    git_root, revision = git_revision(dataset_path)
    source = {
        "kind": "git_checkout_file",
        "path": source_path(dataset_path),
        "sha256": sha256_file(dataset_path),
        "git_root": source_path(Path(git_root)),
        "git_revision": revision,
    }
    summary = {
        **actual,
        "gold_source_granularity": "turn",
        "qa_scoring_note": (
            "All 1,540 category 1--4 questions remain in QA scoring; the seven "
            "listed questions are excluded only from source-recall scoring."
        ),
        "sample_anchor_counts": sample_anchor_counts,
        "normalization_reason_counts": dict(sorted(normalization_reasons.items())),
        "source_recall_exclusions": exclusions,
    }
    return records, summary, source


def build_lme_records(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset = read_json(dataset_path)
    if not isinstance(dataset, list):
        raise FreezeError("LongMemEval-S dataset must be a JSON list")
    records: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    duplicate_haystack_items = 0
    duplicate_haystack_occurrences = 0
    for dataset_index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise FreezeError(f"LongMemEval-S item {dataset_index} is not an object")
        question_id = str(item.get("question_id", "")).strip()
        question = item.get("question")
        answer_ids = item.get("answer_session_ids")
        haystack_ids = item.get("haystack_session_ids")
        if not question_id or question_id in seen_question_ids:
            raise FreezeError(
                f"LongMemEval-S item {dataset_index} has invalid/duplicate question_id"
            )
        seen_question_ids.add(question_id)
        if not isinstance(question, str) or not question.strip():
            raise FreezeError(f"LongMemEval-S item {dataset_index} has empty question")
        if not isinstance(answer_ids, list) or not answer_ids:
            raise FreezeError(
                f"LongMemEval-S item {dataset_index} has empty answer_session_ids"
            )
        if not isinstance(haystack_ids, list) or not haystack_ids:
            raise FreezeError(
                f"LongMemEval-S item {dataset_index} has empty haystack_session_ids"
            )
        normalized_answers = [str(value).strip() for value in answer_ids]
        normalized_haystack = [str(value).strip() for value in haystack_ids]
        if any(not value for value in normalized_answers + normalized_haystack):
            raise FreezeError(
                f"LongMemEval-S item {dataset_index} has empty session ID"
            )
        duplicate_count = len(normalized_haystack) - len(set(normalized_haystack))
        duplicate_haystack_items += int(duplicate_count > 0)
        duplicate_haystack_occurrences += duplicate_count
        missing = [
            value for value in normalized_answers if value not in normalized_haystack
        ]
        if missing:
            raise FreezeError(
                f"LongMemEval-S item {dataset_index} gold sessions missing: {missing}"
            )
        records.append(
            {
                "schema_version": 1,
                "benchmark": "LongMemEval-S",
                "question_id": f"longmemeval-s:{question_id}",
                "dataset_question_id": question_id,
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
                    for raw, normalized in zip(answer_ids, normalized_answers)
                ],
                "normalized_source_ids": normalized_answers,
                "gold_source_ids": normalized_answers,
                "missing_source_ids": [],
                "qa_scoring_eligible": True,
                "source_recall_eligible": True,
                "source_recall_exclusion_reasons": [],
            }
        )
    if len(records) != 500:
        raise FreezeError(
            f"LongMemEval-S frozen count mismatch: expected 500, got {len(records)}"
        )
    git_root, revision = git_revision(dataset_path)
    source = {
        "kind": "git_checkout_file",
        "path": source_path(dataset_path),
        "sha256": sha256_file(dataset_path),
        "git_root": source_path(Path(git_root)),
        "git_revision": revision,
    }
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
    return records, summary, source


def load_beam_runner(path: Path) -> Any:
    checked = require_regular_file(path)
    spec = importlib.util.spec_from_file_location("frozen_beam_runner", checked)
    if spec is None or spec.loader is None:
        raise FreezeError(f"cannot import BEAM runner: {checked}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.DEFAULT_DATASET_REVISION != BEAM_REVISION:
        raise FreezeError(
            "BEAM runner revision differs from the evidence-mapping revision"
        )
    return module


def flatten_beam_source_ids(value: Any) -> list[tuple[Any, str, str]]:
    """Recursively flatten dict/list source_chat_ids in deterministic order."""
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: list[tuple[Any, str, str]] = []
        for key in sorted(value):
            flattened.extend(flatten_beam_source_ids(value[key]))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(flatten_beam_source_ids(item))
        return flattened
    if isinstance(value, bool):
        raise FreezeError("BEAM source_chat_ids contains a boolean")
    if isinstance(value, int):
        return [(value, str(value), "integer_to_string")]
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        reason = "identity" if normalized == value else "strip_whitespace"
        return [(value, normalized, reason)]
    raise FreezeError(f"unsupported BEAM source_chat_ids scalar: {value!r}")


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def beam_cache_revision_dir(cache_dir: Path) -> Path:
    return ensure_no_symlink(
        absolute_without_resolving(cache_dir).joinpath(*BEAM_CACHE_COMPONENTS),
        require_exists=True,
    )


def build_beam_records(
    cache_dir: Path, runner_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from datasets import Dataset

    runner = load_beam_runner(runner_path)
    revision_dir = beam_cache_revision_dir(cache_dir)
    info_path = require_regular_file(revision_dir / "dataset_info.json")
    info = read_json(info_path)
    split_info = info.get("splits", {}) if isinstance(info, dict) else {}
    records: list[dict[str, Any]] = []
    empty_by_type: Counter[str] = Counter()
    split_question_counts: Counter[str] = Counter()
    cache_files: dict[str, dict[str, Any]] = {
        "dataset_info.json": {
            "path": source_path(info_path),
            "sha256": sha256_file(info_path),
        }
    }

    for split, indices in BEAM_SCOPE.items():
        arrow_path = require_regular_file(revision_dir / f"beam-{split}.arrow")
        cache_files[arrow_path.name] = {
            "path": source_path(arrow_path),
            "sha256": sha256_file(arrow_path),
        }
        expected_rows = EXPECTED_BEAM_ROWS[split]
        if split_info.get(split, {}).get("num_examples") != expected_rows:
            raise FreezeError(
                f"BEAM dataset_info row count mismatch for {split}: "
                f"{split_info.get(split)}"
            )
        dataset = Dataset.from_file(str(arrow_path))
        if len(dataset) != expected_rows:
            raise FreezeError(
                f"BEAM cached row count mismatch for {split}: {len(dataset)}"
            )
        for conversation_index in indices:
            item = dataset[conversation_index]
            native_conversation, input_stats = runner.conversation_to_native(item)
            source_id_map = input_stats.get("source_id_map")
            if not isinstance(source_id_map, dict):
                raise FreezeError("BEAM runner did not return source_id_map")
            native_anchors = {
                turn["dia_id"]
                for key, turns in native_conversation.items()
                if key.startswith("session_") and isinstance(turns, list)
                for turn in turns
            }
            for raw_source_id, mapped_ids in source_id_map.items():
                if not isinstance(raw_source_id, str) or not isinstance(
                    mapped_ids, list
                ):
                    raise FreezeError("BEAM source_id_map has invalid shape")
                if not mapped_ids or any(value not in native_anchors for value in mapped_ids):
                    raise FreezeError(
                        f"BEAM source_id_map references a missing anchor: {raw_source_id}"
                    )
            questions = runner.extract_questions(item)
            if len(questions) != 20:
                raise FreezeError(
                    f"BEAM {split}/{conversation_index} has {len(questions)} questions"
                )
            conversation_id = str(item.get("conversation_id", "")).strip()
            if not conversation_id:
                raise FreezeError(
                    f"BEAM {split}/{conversation_index} lacks conversation_id"
                )
            for question_index, question in enumerate(questions):
                question_type = str(question.get("question_type", "")).strip()
                raw_source_ids = question.get("source_chat_ids")
                flattened = flatten_beam_source_ids(raw_source_ids)
                normalized = [value for _, value, _ in flattened]
                normalization: list[dict[str, Any]] = []
                mapped: list[str] = []
                missing: list[str] = []
                for raw, value, reason in flattened:
                    mapped_values = source_id_map.get(value, [])
                    normalization.append(
                        {
                            "raw": raw,
                            "normalized": value,
                            "reason": reason,
                            "mapped_source_ids": mapped_values,
                        }
                    )
                    if mapped_values:
                        mapped.extend(mapped_values)
                    else:
                        missing.append(value)
                mapped = unique_in_order(mapped)
                missing = unique_in_order(missing)
                if missing:
                    raise FreezeError(
                        f"BEAM {split}/{conversation_index}/q{question_index} "
                        f"contains unmapped source IDs: {missing}"
                    )
                eligible = bool(normalized)
                if not eligible:
                    empty_by_type[question_type] += 1
                split_question_counts[split] += 1
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
                        "raw_gold_source_ids": raw_source_ids,
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
    abstention_empty = empty_by_type["abstention"]
    other_empty = empty_count - abstention_empty
    actual = {
        "questions": len(records),
        "qa_scoring_questions": len(records),
        "source_recall_question_denominator": mapped_count,
        "questions_without_gold_sources": empty_count,
        "empty_abstention_questions": abstention_empty,
        "empty_other_questions": other_empty,
        "split_question_counts": dict(sorted(split_question_counts.items())),
    }
    expected = {
        "questions": 900,
        "qa_scoring_questions": 900,
        "source_recall_question_denominator": 804,
        "questions_without_gold_sources": 96,
        "empty_abstention_questions": 90,
        "empty_other_questions": 6,
        "split_question_counts": {"100K": 200, "1M": 700},
    }
    if actual != expected:
        raise FreezeError(
            f"BEAM frozen count mismatch: expected {expected}, got {actual}"
        )
    runner_path = require_regular_file(runner_path)
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
        **actual,
        "formal_scope": {
            "100K": {"conversation_indices": list(BEAM_SCOPE["100K"])},
            "1M": {"conversation_indices": list(BEAM_SCOPE["1M"])},
        },
        "gold_source_granularity": "turn",
        "empty_question_types": dict(sorted(empty_by_type.items())),
        "source_recall_exclusions": exclusions,
    }
    return records, summary, source


def build_bundle(
    *,
    locomo_dataset: Path = DEFAULT_LOCOMO,
    lme_dataset: Path = DEFAULT_LME,
    beam_cache_dir: Path = DEFAULT_BEAM_CACHE,
    beam_runner: Path = DEFAULT_BEAM_RUNNER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    locomo_records, locomo_summary, locomo_source = build_locomo_records(
        locomo_dataset
    )
    lme_records, lme_summary, lme_source = build_lme_records(lme_dataset)
    beam_records, beam_summary, beam_source = build_beam_records(
        beam_cache_dir, beam_runner
    )
    records = [*locomo_records, *lme_records, *beam_records]
    question_ids = [record["question_id"] for record in records]
    if len(question_ids) != 3386 or len(question_ids) != len(set(question_ids)):
        raise FreezeError("combined artifact must contain 3,386 unique questions")
    questions_payload = canonical_jsonl_bytes(records)
    tool_sources = {
        "generator": {
            "path": source_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "auditor": {
            "path": source_path(AUDITOR),
            "sha256": sha256_file(AUDITOR),
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_version": 1,
        "description": (
            "Frozen question-level gold-source mappings and source-recall "
            "denominators. This artifact is data-only and makes no model calls."
        ),
        "questions_file": QUESTIONS_NAME,
        "questions_sha256": sha256_bytes(questions_payload),
        "question_count": len(records),
        "normalization_rules": {
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
                    "rule": "Canonicalize numeric components with int(), e.g. D30:05 to D30:5.",
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
        },
        "benchmarks": {
            "LoCoMo": {"source": locomo_source, "counts": locomo_summary},
            "LongMemEval-S": {"source": lme_source, "counts": lme_summary},
            "BEAM": {"source": beam_source, "counts": beam_summary},
        },
        "tool_sources": tool_sources,
    }
    return records, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze LoCoMo, LongMemEval-S, and BEAM gold-source mappings"
    )
    parser.add_argument("--locomo-dataset", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--longmemeval-dataset", type=Path, default=DEFAULT_LME)
    parser.add_argument("--beam-cache-dir", type=Path, default=DEFAULT_BEAM_CACHE)
    parser.add_argument("--beam-runner", type=Path, default=DEFAULT_BEAM_RUNNER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with exclusive_output_lock(args.output_dir) as output_dir:
        records, manifest = build_bundle(
            locomo_dataset=args.locomo_dataset,
            lme_dataset=args.longmemeval_dataset,
            beam_cache_dir=args.beam_cache_dir,
            beam_runner=args.beam_runner,
        )
        questions_payload = canonical_jsonl_bytes(records)
        manifest_payload = canonical_json_bytes(manifest)
        checksum_payload = (
            f"{sha256_bytes(manifest_payload)}  {MANIFEST_NAME}\n"
        ).encode("ascii")
        payloads = {
            output_dir / QUESTIONS_NAME: questions_payload,
            output_dir / MANIFEST_NAME: manifest_payload,
            output_dir / CHECKSUM_NAME: checksum_payload,
        }
        input_paths = (
            args.locomo_dataset,
            args.longmemeval_dataset,
            args.beam_runner,
        )
        input_identities = {
            str(absolute_without_resolving(path)).casefold() for path in input_paths
        }
        if input_identities & {
            str(absolute_without_resolving(path)).casefold() for path in payloads
        }:
            raise FreezeError("an artifact output path collides with an input path")
        preflight_collisions(payloads)
        statuses = {
            path.name: collision_safe_atomic_write(path, payload)
            for path, payload in payloads.items()
        }
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "questions": len(records),
                "questions_sha256": manifest["questions_sha256"],
                "files": statuses,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreezeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
