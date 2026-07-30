#!/usr/bin/env python3
"""Build a deterministic blinded 100-question LoCoMo annotation packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.m4_reliability import ReliabilityError  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_SEED = "locomo-human-validation-v1"
QUESTION_PATTERN = re.compile(r"^s([0-9]+)_q[0-9]+$")
VERDICTS = ("correct", "incorrect", "uncertain")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _hash_order(seed: str, *parts: object) -> str:
    return hashlib.sha256("|".join([seed, *(str(part) for part in parts)]).encode()).hexdigest()


def allocate_categories(counts: dict[int, int], total: int = 10) -> dict[int, int]:
    positive = {category: count for category, count in counts.items() if count > 0}
    if not positive or sum(positive.values()) < total or len(positive) > total:
        raise ReliabilityError("category inventory cannot support requested packet quota")
    allocation = {category: 1 for category in positive}
    remaining = total - len(positive)
    denominator = sum(positive.values())
    quotas = {
        category: remaining * count / denominator for category, count in positive.items()
    }
    for category, quota in quotas.items():
        addition = min(positive[category] - 1, int(quota))
        allocation[category] += addition
    still_needed = total - sum(allocation.values())
    order = sorted(
        positive,
        key=lambda category: (-(quotas[category] - int(quotas[category])), category),
    )
    while still_needed:
        eligible = [
            category for category in order if allocation[category] < positive[category]
        ]
        if not eligible:
            raise ReliabilityError("category allocation exhausted unexpectedly")
        for category in eligible:
            if still_needed == 0:
                break
            allocation[category] += 1
            still_needed -= 1
    return allocation


def validate_source(payload: object) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ReliabilityError("scored source is not an object")
    meta = payload.get("meta")
    results = payload.get("results")
    if not isinstance(meta, dict) or not isinstance(results, list):
        raise ReliabilityError("scored source lacks meta/results")
    if meta.get("status") != "complete" or meta.get("benchmark") != "locomo":
        raise ReliabilityError("human packet requires a complete LoCoMo score artifact")
    if meta.get("judge_profile") != "protocol-primary-openrouter-gpt4o-mini":
        raise ReliabilityError("human packet requires the frozen primary judge profile")
    questions = [record for record in results if record.get("question_id") != "_build_stats"]
    if len(questions) != 1540:
        raise ReliabilityError("human packet source must contain exactly 1,540 questions")
    seen: set[str] = set()
    for record in questions:
        question_id = record.get("question_id")
        match = QUESTION_PATTERN.fullmatch(str(question_id))
        if not match or question_id in seen:
            raise ReliabilityError("human packet source has invalid question identities")
        if record.get("category") not in {1, 2, 3, 4}:
            raise ReliabilityError(f"{question_id} has invalid primary category")
        if record.get("judge_score") not in {0, 1}:
            raise ReliabilityError(f"{question_id} has invalid judge_score")
        for key in ("question", "gold", "answer"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ReliabilityError(f"{question_id} has invalid {key}")
        seen.add(question_id)
    return meta, questions


def select_questions(
    questions: list[dict[str, Any]], *, seed: str = DEFAULT_SEED
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in questions:
        match = QUESTION_PATTERN.fullmatch(record["question_id"])
        assert match is not None
        grouped[int(match.group(1))][int(record["category"])].append(record)
    if sorted(grouped) != list(range(10)):
        raise ReliabilityError("LoCoMo human packet requires samples 0 through 9")
    selected: list[dict[str, Any]] = []
    for sample in range(10):
        categories = grouped[sample]
        allocation = allocate_categories(
            {category: len(records) for category, records in categories.items()}
        )
        for category, quota in sorted(allocation.items()):
            ordered = sorted(
                categories[category],
                key=lambda record: _hash_order(
                    seed, sample, category, record["question_id"]
                ),
            )
            selected.extend(ordered[:quota])
    if len(selected) != 100 or len({item["question_id"] for item in selected}) != 100:
        raise ReliabilityError("human packet selection is not exactly 100 unique questions")
    return sorted(
        selected,
        key=lambda record: _hash_order(seed, "packet-order", record["question_id"]),
    )


def build_payloads(
    source: Path, *, seed: str = DEFAULT_SEED
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    meta, questions = validate_source(read_json(source))
    selected = select_questions(questions, seed=seed)
    packet_items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for record in selected:
        packet_id = "H" + hashlib.sha256(
            f"{seed}|packet-id|{record['question_id']}".encode()
        ).hexdigest()[:15].upper()
        if packet_id in identifiers:
            raise ReliabilityError("human packet identifier collision")
        identifiers.add(packet_id)
        packet_items.append(
            {
                "packet_id": packet_id,
                "question": record["question"],
                "reference_answer": record["gold"],
                "candidate_answer": record["answer"],
                "instruction": (
                    "Label whether the candidate answer is semantically correct for "
                    "the question relative to the reference answer. Use uncertain only "
                    "when correctness cannot be decided from these fields."
                ),
                "allowed_verdicts": list(VERDICTS),
            }
        )
        sample = int(QUESTION_PATTERN.fullmatch(record["question_id"]).group(1))
        key_items.append(
            {
                "packet_id": packet_id,
                "question_id": record["question_id"],
                "sample": sample,
                "category": record["category"],
                "primary_judge_score": record["judge_score"],
            }
        )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_external_labels",
        "packet_id": "locomo-human-validation-100-v1",
        "blinding": [
            "method identity omitted",
            "source record identifiers omitted",
            "automated judge labels omitted",
            "sample and category strata omitted",
        ],
        "selection": {
            "seed": seed,
            "rule": (
                "ten questions per conversation; at least one per present category; "
                "remaining quota by largest-remainder proportional allocation; "
                "SHA-256 order within strata"
            ),
            "questions": 100,
        },
        "annotation_schema": {
            "verdict": list(VERDICTS),
            "confidence": "integer 1 (lowest) through 5 (highest)",
            "rationale": "optional short text",
        },
        "items": packet_items,
    }
    private_key = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet["packet_id"],
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "judge_profile": meta["judge_profile"],
        },
        "items": key_items,
    }
    templates = [
        {"packet_id": item["packet_id"], "verdict": "", "confidence": "", "rationale": ""}
        for item in packet_items
    ]
    return packet, private_key, templates


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=["packet_id", "verdict", "confidence", "rationale"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_packet(source: Path, output_dir: Path, *, seed: str) -> dict[str, Any]:
    if output_dir.exists():
        raise ReliabilityError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    packet, private_key, templates = build_payloads(source, seed=seed)
    files = {
        "public/packet.json": json_bytes(packet),
        "private/private_key.json": json_bytes(private_key),
        "public/annotator_A.csv": csv_bytes(templates),
        "public/annotator_B.csv": csv_bytes(templates),
    }
    for name, payload in files.items():
        atomic_bytes(output_dir / name, payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_sha256": sha256_file(source),
        "seed": seed,
        "questions": 100,
        "files": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(files.items())
        },
        "external_labels_required": 2,
        "distribution_rule": (
            "Give annotators only public/packet.json and their own blank CSV; "
            "do not distribute private/private_key.json."
        ),
    }
    atomic_bytes(output_dir / "manifest.json", json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = write_packet(
        args.source.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        seed=args.seed,
    )
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
