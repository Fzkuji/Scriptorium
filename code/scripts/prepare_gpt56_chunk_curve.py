#!/usr/bin/env python3
"""Freeze the GPT-5.6 three-benchmark chunk-size screening matrix.

This command is deliberately plan-only.  It reads local datasets, validates
their frozen hashes and selected unit identities, and writes a manifest plus a
JSONL run matrix.  It has no execution mode and never sends a model request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "gpt56-chunk-curve"

LOCOMO_DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
LONGMEMEVAL_DATA = (
    ROOT / "benchmarks" / "longmemeval" / "data" /
    "longmemeval_s_cleaned.json"
)
BEAM_100K = (
    ROOT / "benchmarks" / "beam" / "hf_cache" / "Mohammadta___beam" /
    "default" / "0.0.0" / "3205395e897e7318c7b094ef4e6047b9b82dbb03" /
    "beam-100K.arrow"
)
LOCOMO_EVALUATOR = ROOT / "scripts" / "eval_full.py"
LONGMEMEVAL_EVALUATOR = (
    ROOT / "benchmarks" / "longmemeval" / "src" / "evaluation" /
    "evaluate_qa.py"
)
BEAM_EVALUATOR = ROOT / "scripts" / "evaluate_v88_gpt55_beam.py"
EXECUTION_RUNNER = ROOT / "scripts" / "run_gpt56_chunk_curve.py"
NATIVEMEM_RUNTIME = ROOT / "src" / "nativemem.py"
SUBSCRIPTION_PROXY = ROOT / "src" / "chatgpt_proxy.py"
LONGMEMEVAL_CONVERTER = ROOT / "scripts" / "run_v88_gpt55_longmemeval.py"
BEAM_CONVERTER = ROOT / "scripts" / "run_v88_gpt55_beam.py"

EXPECTED_HASHES = {
    LOCOMO_DATA: "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    LONGMEMEVAL_DATA: "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    BEAM_100K: "7b78964d628b866242dcb5df03bb6e3140d60cc8b366c7c87cbf43cc708007d0",
    LOCOMO_EVALUATOR: "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd",
}

WINDOWS: tuple[int | str, ...] = (4, 6, 8, 12, 16, 20, 24, 32, "session")
MODELS = (
    {
        "tier": "sol",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "none",
        "api_reference_price_usd_per_million_tokens": {
            "input": 5.0,
            "output": 30.0,
        },
    },
    {
        "tier": "terra",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "api_reference_price_usd_per_million_tokens": {
            "input": 2.5,
            "output": 15.0,
        },
    },
    {
        "tier": "luna",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "api_reference_price_usd_per_million_tokens": {
            "input": 1.0,
            "output": 6.0,
        },
    },
)

LOCOMO_SELECTIONS = (
    {"sample_index": 5, "sample_id": "conv-44"},
    {"sample_index": 7, "sample_id": "conv-48"},
)
LONGMEMEVAL_SELECTIONS = (
    {"dataset_index": 97, "question_id": "2318644b"},
    {"dataset_index": 248, "question_id": "gpt4_6dc9b45b"},
)
BEAM_SELECTIONS = (
    {"split": "100K", "conversation_index": 0, "conversation_id": "1"},
    {"split": "100K", "conversation_index": 1, "conversation_id": "2"},
)

FIXED_ENV = {
    "NATIVEMEM_PROMPT": "v10",
    "NATIVEMEM_STORE_MODE": "oneshot",
    "NATIVEMEM_V10_CONTEXT_MODE": "none",
    "NATIVEMEM_V10_CONTEXT_ITEMS": "0",
    "NATIVEMEM_V10_SUMMARY_MAX_WORDS": "180",
    "NATIVEMEM_V10_TIDY_EVERY_SESSIONS": "1",
    "NATIVEMEM_V10_SESSION_TIDY_PASSES": "1",
    "NATIVEMEM_V10_FINAL_TIDY_PASSES": "1",
    "NATIVEMEM_V8_SINGLE": "1",
    "NATIVEMEM_V8_SEGMENT": "fixed",
    "NATIVEMEM_V8_SECTIONS": "on",
    "NATIVEMEM_V8_ARTICLE": "off",
    "NATIVEMEM_V8_VERIFY": "on",
    "NATIVEMEM_V8_MERGE_LINES": "on",
    "NATIVEMEM_V8_TIDY_COMBINED": "off",
    "NATIVEMEM_V8_MAX_TOPICS": "30",
    "NATIVEMEM_TOPK": "20",
    "NATIVEMEM_V8_MAX_ROUNDS": "12",
    "NATIVEMEM_V8_MAX_TOKENS": "1200",
    "NATIVEMEM_V8_READ_CONTEXT": "1",
    "NATIVEMEM_V8_MAP": "dir",
    "NATIVEMEM_V8_MAP_INLINE": "8",
    "NATIVEMEM_V8_MAX_DEPTH": "4",
    "NATIVEMEM_V8_REWRITE_MIN": "8",
    "NATIVEMEM_V8_SEGMENT_MAX": "10",
}


class PreparationError(RuntimeError):
    """The local experiment inputs do not match the frozen protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise PreparationError(f"required file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PreparationError(
                f"frozen hash mismatch for {path}: expected {expected}, got {actual}"
            )
        observed[str(path.relative_to(ROOT))] = actual
    return observed


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def locomo_units() -> list[dict[str, Any]]:
    dataset = read_json(LOCOMO_DATA)
    units: list[dict[str, Any]] = []
    for selected in LOCOMO_SELECTIONS:
        index = selected["sample_index"]
        item = dataset[index]
        if item.get("sample_id") != selected["sample_id"]:
            raise PreparationError(f"LoCoMo identity mismatch at index {index}")
        conversation = item["conversation"]
        sessions = [
            conversation[f"session_{number}"]
            for number in range(1, 10000)
            if f"session_{number}" in conversation
        ]
        qa_count = sum(
            1 for qa in item.get("qa", []) if qa.get("category") in (1, 2, 3, 4)
        )
        units.append({
            "unit_id": selected["sample_id"],
            "selection": dict(selected),
            "statistics": {
                "sessions": len(sessions),
                "messages": sum(len(session) for session in sessions),
                "max_session_messages": max(map(len, sessions)),
                "cat1_4_questions": qa_count,
            },
        })
    return units


def longmemeval_units() -> list[dict[str, Any]]:
    dataset = read_json(LONGMEMEVAL_DATA)
    units: list[dict[str, Any]] = []
    for selected in LONGMEMEVAL_SELECTIONS:
        index = selected["dataset_index"]
        item = dataset[index]
        if item.get("question_id") != selected["question_id"]:
            raise PreparationError(
                f"LongMemEval identity mismatch at index {index}"
            )
        sessions = item["haystack_sessions"]
        units.append({
            "unit_id": selected["question_id"],
            "selection": dict(selected),
            "statistics": {
                "sessions": len(sessions),
                "messages": sum(len(session) for session in sessions),
                "max_session_messages": max(map(len, sessions)),
                "questions": 1,
                "answer_sessions": len(item.get("answer_session_ids", [])),
                "question_type": item.get("question_type"),
                "question": item.get("question"),
            },
        })
    return units


def beam_units() -> list[dict[str, Any]]:
    try:
        from datasets import Dataset
        from scripts.run_v88_gpt55_beam import (
            conversation_to_native,
            extract_questions,
        )
    except ImportError as exc:
        raise PreparationError(
            "local BEAM preparation requires the installed datasets package"
        ) from exc

    dataset = Dataset.from_file(str(BEAM_100K))
    units: list[dict[str, Any]] = []
    for selected in BEAM_SELECTIONS:
        index = selected["conversation_index"]
        item = dataset[index]
        if str(item.get("conversation_id")) != selected["conversation_id"]:
            raise PreparationError(f"BEAM identity mismatch at index {index}")
        conversation, metadata = conversation_to_native(item)
        sessions = [
            conversation[f"session_{number}"]
            for number in range(1, 10000)
            if f"session_{number}" in conversation
        ]
        source_ids = metadata["source_id_map"]
        units.append({
            "unit_id": f"100K-conv-{selected['conversation_id']}",
            "selection": dict(selected),
            "statistics": {
                "sessions": len(sessions),
                "messages": sum(len(session) for session in sessions),
                "session_messages": [len(session) for session in sessions],
                "max_session_messages": max(map(len, sessions)),
                "questions": len(extract_questions(item)),
                "source_references": sum(len(value) for value in source_ids.values()),
            },
        })
    return units


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        EXECUTION_RUNNER,
        ROOT / "src" / "v10_memory.py",
        ROOT / "src" / "v8_memory.py",
        NATIVEMEM_RUNTIME,
        ROOT / "src" / "adapters" / "run_nativemem.py",
        SUBSCRIPTION_PROXY,
        LONGMEMEVAL_CONVERTER,
        BEAM_CONVERTER,
        LONGMEMEVAL_EVALUATOR,
        BEAM_EVALUATOR,
    )
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in paths
    }


def benchmark_specs() -> list[dict[str, Any]]:
    return [
        {
            "benchmark": "locomo",
            "dataset": str(LOCOMO_DATA.relative_to(ROOT)),
            "units": locomo_units(),
            "evaluation": {
                "script": str(LOCOMO_EVALUATOR.relative_to(ROOT)),
                "sha256": EXPECTED_HASHES[LOCOMO_EVALUATOR],
                "judge": "openai/gpt-4o-mini",
                "scope": "categories 1-4; exact locked repository evaluator",
            },
            "claim_scope": "screening subset; not the full LoCoMo score",
        },
        {
            "benchmark": "longmemeval-s",
            "dataset": str(LONGMEMEVAL_DATA.relative_to(ROOT)),
            "units": longmemeval_units(),
            "evaluation": {
                "script": str(LONGMEMEVAL_EVALUATOR.relative_to(ROOT)),
                "sha256": sha256_file(LONGMEMEVAL_EVALUATOR),
                "judge": "gpt-4o-mini",
                "scope": "official LongMemEval QA evaluator",
            },
            "claim_scope": (
                "two histories and two questions; integration and chunk-curve "
                "screening only, not a LongMemEval accuracy estimate"
            ),
        },
        {
            "benchmark": "beam-100k",
            "dataset": str(BEAM_100K.relative_to(ROOT)),
            "units": beam_units(),
            "evaluation": {
                "script": str(BEAM_EVALUATOR.relative_to(ROOT)),
                "sha256": sha256_file(BEAM_EVALUATOR),
                "profile": "primary",
                "judge": "openai/gpt-4o-mini",
                "scope": "existing project primary BEAM protocol",
            },
            "claim_scope": "two-conversation screening subset; not the full BEAM score",
        },
    ]


def build_manifest() -> dict[str, Any]:
    observed_hashes = validate_hashes()
    benchmarks = benchmark_specs()
    unit_count = sum(len(spec["units"]) for spec in benchmarks)
    planned_builds = unit_count * len(MODELS) * len(WINDOWS)
    return {
        "schema_version": 2,
        "study_id": "gpt56-chunk-curve-3bench-6unit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "model_requests_authorized": False,
        "plan_only_generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "research_question": (
            "How does the number of new messages per Writer call affect memory "
            "quality and construction cost across three GPT-5.6 capability tiers?"
        ),
        "independent_variable": {
            "name": "NATIVEMEM_V10_WRITE_TURNS",
            "values": list(WINDOWS),
            "unit": "individual user or assistant messages",
        },
        "models": [dict(model) for model in MODELS],
        "pricing_snapshot": {
            "date": "2026-07-18",
            "purpose": "API reference only; subscription quota is reported separately",
            "source": "https://developers.openai.com/api/docs/models",
        },
        "fixed_method_environment": dict(FIXED_ENV),
        "reasoning_policy": "reasoning_effort=none for all three tiers",
        "benchmarks": benchmarks,
        "planned_builds": planned_builds,
        "metrics": {
            "quality": [
                "benchmark QA score using the benchmark-specific frozen evaluator",
                "first-pass evidence recall",
                "post-verify evidence recall",
                "first-pass versus post-verify delta",
            ],
            "cost": [
                "Writer, Verify, and Tidy calls separately",
                "input, cached-input, output, and reasoning tokens separately",
                "wall time and provider-reported cost",
                "cost per 100 messages and per 1000 source tokens",
            ],
            "efficiency": [
                "quality per dollar",
                "quality per 100 Writer calls",
                "maximum supported window under the frozen quality threshold",
            ],
        },
        "output_root": "results/formal/gpt56-chunk-curve",
        "frozen_input_hashes": observed_hashes,
        "source_hashes": source_hashes(),
        "execution_note": (
            "No model executor is invoked by this generator.  Execution requires "
            "a separately reviewed runner and explicit model-request authorization."
        ),
    }


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def build_matrix(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark in manifest["benchmarks"]:
        benchmark_name = benchmark["benchmark"]
        for unit in benchmark["units"]:
            for model in manifest["models"]:
                for window in manifest["independent_variable"]["values"]:
                    window_slug = str(window)
                    run_id = "-".join((
                        "gpt56",
                        slug(benchmark_name),
                        slug(unit["unit_id"]),
                        model["tier"],
                        f"w{window_slug}",
                    ))
                    env = dict(manifest["fixed_method_environment"])
                    env["NATIVEMEM_V10_WRITE_TURNS"] = str(window)
                    rows.append({
                        "schema_version": 1,
                        "run_id": run_id,
                        "status": "planned",
                        "model_requests_authorized": False,
                        "benchmark": benchmark_name,
                        "unit_id": unit["unit_id"],
                        "selection": unit["selection"],
                        "model": model["model"],
                        "tier": model["tier"],
                        "reasoning_effort": model["reasoning_effort"],
                        "write_turns": window,
                        "environment": env,
                        "output_dir": str(Path(manifest["output_root"]) /
                                          benchmark_name / unit["unit_id"] /
                                          model["tier"] / f"w{window_slug}"),
                    })
    if len(rows) != manifest["planned_builds"]:
        raise PreparationError("generated matrix size does not match manifest")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise PreparationError("generated matrix contains duplicate run IDs")
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_plan(output_dir: Path, manifest: dict[str, Any],
               matrix: list[dict[str, Any]]) -> None:
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    matrix_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in matrix
    )
    atomic_text(output_dir / "experiment_manifest.json", manifest_text)
    atomic_text(output_dir / "run_matrix.jsonl", matrix_text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="plan artifact directory; no model output is written here",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest()
    matrix = build_matrix(manifest)
    write_plan(args.output_dir.resolve(), manifest, matrix)
    print(
        f"planned {len(matrix)} build configurations in {args.output_dir.resolve()}"
    )
    print("model requests sent: 0 (this command has no execution mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
