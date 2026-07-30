#!/usr/bin/env python3
"""Analyze build-only v10 write-interval runs without model requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
_DIA_RE = re.compile(r"D\d+:\d+(?:[-,]\d+)*")
_EVENT_LINE_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]")
_DATE_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s*")
_REF_RE = re.compile(r"\s*(?:·\s*)?(?:\[D\d+:[^\]]+\]\s*)+$")


class AnalysisError(RuntimeError):
    pass


def expand_dia_ids(text: str) -> set[str]:
    expanded: set[str] = set()
    for token in _DIA_RE.findall(text or ""):
        conversation, rhs = token.split(":", 1)
        for part in rhs.split(","):
            if "-" in part:
                start, end = (int(value) for value in part.split("-", 1))
                lo, hi = sorted((start, end))
                numbers = range(lo, hi + 1)
            else:
                numbers = [int(part)]
            expanded.update(f"{conversation}:{number}" for number in numbers)
    return expanded


def _load_build_record(path: Path) -> dict:
    value = json.loads(path.read_text())
    if isinstance(value, list):
        if len(value) != 1:
            raise AnalysisError(f"{path} must contain one build-only record")
        value = value[0]
    if not isinstance(value, dict) or value.get("question_id") != "_build_stats":
        raise AnalysisError(f"{path} is not a v10 build record")
    if value.get("method") != "NativeMem-v10":
        raise AnalysisError(f"{path} method is not NativeMem-v10")
    return value


def _normalize_event(line: str) -> str:
    text = _DATE_PREFIX_RE.sub("", line.strip())
    text = _REF_RE.sub("", text)
    return " ".join(text.lower().split())


def _evidence_in_session_scope(evidence: set[str], max_sessions: int) -> bool:
    if not evidence:
        return False
    session_ids = []
    for dia_id in evidence:
        match = re.match(r"D(\d+):", dia_id)
        if match is None:
            return False
        session_ids.append(int(match.group(1)))
    return all(session_id <= max_sessions for session_id in session_ids)


def analyze_build(
    path: Path,
    dataset: list[dict],
    max_sessions_override: int | None = None,
) -> dict:
    record = _load_build_record(path)
    sample = record.get("sample")
    if not isinstance(sample, int) or not 0 <= sample < len(dataset):
        raise AnalysisError(f"{path} has invalid sample {sample!r}")
    memory_dir = Path(str(record.get("memory_dir", ""))).expanduser()
    if not memory_dir.is_absolute():
        memory_dir = (path.parent / memory_dir).resolve()
    topics_dir = memory_dir / "topics"
    if not topics_dir.is_dir():
        raise AnalysisError(f"topics directory is missing: {topics_dir}")

    topic_files = sorted(topics_dir.rglob("*.md"))
    memory_ids: set[str] = set()
    event_lines: list[str] = []
    memory_bytes = 0
    for topic_file in topic_files:
        text = topic_file.read_text(errors="replace")
        memory_bytes += topic_file.stat().st_size
        memory_ids.update(expand_dia_ids(text))
        event_lines.extend(
            line.strip()
            for line in text.splitlines()
            if _EVENT_LINE_RE.match(line.strip())
        )

    qas = [
        qa
        for qa in dataset[sample].get("qa", [])
        if int(qa.get("category", 0)) in {1, 2, 3, 4}
    ]
    evidence_sets = [set(map(str, qa.get("evidence", []))) for qa in qas]
    evidence_sets = [evidence for evidence in evidence_sets if evidence]
    max_sessions = (
        max_sessions_override
        if max_sessions_override is not None
        else record.get("max_sessions")
    )
    if max_sessions is None:
        scoped_evidence_sets = evidence_sets
    else:
        max_sessions = int(max_sessions)
        scoped_evidence_sets = [
            evidence
            for evidence in evidence_sets
            if _evidence_in_session_scope(evidence, max_sessions)
        ]
    gold_ids = set().union(*evidence_sets) if evidence_sets else set()
    scoped_gold_ids = (
        set().union(*scoped_evidence_sets) if scoped_evidence_sets else set()
    )
    any_hits = sum(bool(evidence & memory_ids) for evidence in evidence_sets)
    all_hits = sum(evidence <= memory_ids for evidence in evidence_sets)
    scoped_any_hits = sum(
        bool(evidence & memory_ids) for evidence in scoped_evidence_sets
    )
    scoped_all_hits = sum(
        evidence <= memory_ids for evidence in scoped_evidence_sets
    )
    normalized = [_normalize_event(line) for line in event_lines]
    unique_normalized = {line for line in normalized if line}

    config = record.get("v10_config", {})
    return {
        "build_record": str(path.resolve()),
        "sample": sample,
        "max_sessions": max_sessions,
        "builder_model": record.get("builder_model"),
        "write_turns": config.get("write_turns"),
        "context_mode": config.get("context_mode"),
        "context_items": config.get("context_items"),
        "summary_max_words": config.get("summary_max_words"),
        "tidy_every_sessions": config.get("tidy_every_sessions"),
        "session_tidy_passes": config.get("session_tidy_passes"),
        "final_tidy_passes": config.get("final_tidy_passes"),
        "questions_with_evidence": len(evidence_sets),
        "evidence_question_any_recall": (
            any_hits / len(evidence_sets) if evidence_sets else None
        ),
        "evidence_question_all_recall": (
            all_hits / len(evidence_sets) if evidence_sets else None
        ),
        "unique_gold_evidence_ids": len(gold_ids),
        "unique_gold_evidence_id_recall": (
            len(gold_ids & memory_ids) / len(gold_ids) if gold_ids else None
        ),
        "questions_with_evidence_in_build_scope": len(scoped_evidence_sets),
        "evidence_question_any_recall_in_build_scope": (
            scoped_any_hits / len(scoped_evidence_sets)
            if scoped_evidence_sets
            else None
        ),
        "evidence_question_all_recall_in_build_scope": (
            scoped_all_hits / len(scoped_evidence_sets)
            if scoped_evidence_sets
            else None
        ),
        "unique_gold_evidence_ids_in_build_scope": len(scoped_gold_ids),
        "unique_gold_evidence_id_recall_in_build_scope": (
            len(scoped_gold_ids & memory_ids) / len(scoped_gold_ids)
            if scoped_gold_ids
            else None
        ),
        "event_lines": len(event_lines),
        "unique_event_lines_normalized": len(unique_normalized),
        "exact_normalized_duplicate_fraction": (
            1 - len(unique_normalized) / len(normalized) if normalized else None
        ),
        "topic_files": len(topic_files),
        "topic_memory_bytes": memory_bytes,
        "build_calls": record.get("build_calls"),
        "build_tokens_in": record.get("build_tokens_in"),
        "build_tokens_out": record.get("build_tokens_out"),
        "build_time_s": record.get("build_time_s"),
        "build_phase_usage": record.get("build_phase_usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_records", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="override the built session count for scoped evidence metrics",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    rows = [
        analyze_build(path, dataset, max_sessions_override=args.max_sessions)
        for path in args.build_records
    ]
    output = {
        "schema_version": 1,
        "study": "NativeMem-v10-write-interval",
        "dataset": str(args.dataset.resolve()),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} analyzed builds -> {args.output}")


if __name__ == "__main__":
    main()
