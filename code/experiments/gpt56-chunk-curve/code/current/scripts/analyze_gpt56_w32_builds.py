#!/usr/bin/env python3
"""Compute audited build-cost and source-reference diagnostics for GPT-5.6 W32."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import gpt56_chunk_curve_qa_contract as contract
from scripts import readonly_nativemem_control as readonly_control
from scripts import run_gpt56_chunk_curve as build_runner
from scripts import controlled_locomo_answer_contract as answer_contract


_SOURCE_SPEC_RE = re.compile(r"D\d+:\d+(?:[-,]\d+)*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = build_runner.DEFAULT_MATRIX
DEFAULT_BUILD_ROOT = (
    ROOT
    / "results"
    / "formal"
    / "gpt56-chunk-curve-subscription-20260719-w32-only"
)
DEFAULT_EXECUTION_MANIFEST = DEFAULT_BUILD_ROOT / "execution_manifest.json"
DEFAULT_OUTPUT = DEFAULT_BUILD_ROOT / "analysis" / "build-quality-cost-v2.json"

EXPECTED_RUN_IDS = (
    "gpt56-beam-100k-100K-conv-1-terra-w32",
    "gpt56-beam-100k-100K-conv-2-terra-w32",
    "gpt56-longmemeval-s-2318644b-terra-w32",
    "gpt56-longmemeval-s-gpt4-6dc9b45b-terra-w32",
    "gpt56-locomo-conv-44-terra-w32",
    "gpt56-locomo-conv-48-terra-w32",
    "gpt56-beam-100k-100K-conv-1-sol-w32",
    "gpt56-beam-100k-100K-conv-2-sol-w32",
    "gpt56-longmemeval-s-2318644b-sol-w32",
    "gpt56-longmemeval-s-gpt4-6dc9b45b-sol-w32",
    "gpt56-locomo-conv-44-sol-w32",
    "gpt56-locomo-conv-48-sol-w32",
)


class BuildAnalysisError(RuntimeError):
    """A build or validated source unit cannot support the analysis."""


def validate_exact_run_ids(run_ids: object) -> tuple[str, ...]:
    """Return the canonical current campaign order after exact-set validation."""

    if not isinstance(run_ids, (list, tuple)):
        raise BuildAnalysisError("analysis requires the exact 12-run scope")
    normalized = tuple(str(value) for value in run_ids)
    if len(normalized) != len(set(normalized)) or set(normalized) != set(
        EXPECTED_RUN_IDS
    ):
        raise BuildAnalysisError("analysis requires the exact 12-run scope")
    return EXPECTED_RUN_IDS


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the exact 12 completed GPT-5.6 Terra/Sol W32 builds "
            "without model requests."
        )
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        default=DEFAULT_EXECUTION_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", dest="run_ids", action="append")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


def sha256_file(path: Path) -> str:
    return build_runner.sha256_file(path)


def memory_sha256(memory_dir: Path) -> str:
    return str(readonly_control.memory_descriptor(memory_dir)["sha256"])


def write_report_no_clobber(path: Path, report: Mapping[str, Any]) -> None:
    expected = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise BuildAnalysisError(f"refusing to clobber analysis report: {path}")
        return
    try:
        answer_contract.atomic_json_no_clobber(path, report)
    except FileExistsError as exc:
        raise BuildAnalysisError(f"refusing to clobber analysis report: {path}") from exc


def _referenced_source_ids(memory_dir: Path) -> set[str]:
    specs: list[str] = []
    for path in sorted(memory_dir.rglob("*.md")):
        specs.extend(_SOURCE_SPEC_RE.findall(path.read_text(encoding="utf-8")))
    return set(readonly_control.expand_source_specs(specs))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _is_memory_entry(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def analyze_view(view_dir: Path, *, source_ids: set[str]) -> dict[str, Any]:
    """Measure one Markdown view using the frozen canonical source parser."""

    try:
        readonly_control.assert_safe_tree(view_dir)
    except readonly_control.ReadOnlyControlError as exc:
        raise BuildAnalysisError(str(exc)) from exc
    files = sorted(view_dir.rglob("*.md"))
    byte_count = 0
    entries = 0
    nonempty_lines = 0
    grounded_entries = 0
    grounded_lines = 0
    expanded_mentions = 0
    valid_mentions = 0
    invalid_mentions = 0
    valid_ids: set[str] = set()
    invalid_ids: set[str] = set()
    for path in files:
        raw = path.read_bytes()
        byte_count += len(raw)
        text = raw.decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            nonempty_lines += 1
            is_entry = _is_memory_entry(line)
            if is_entry:
                entries += 1
            references = readonly_control.source_ids_in_text(line)
            valid = [value for value in references if value in source_ids]
            invalid = [value for value in references if value not in source_ids]
            expanded_mentions += len(references)
            valid_mentions += len(valid)
            invalid_mentions += len(invalid)
            valid_ids.update(valid)
            invalid_ids.update(invalid)
            if valid:
                grounded_lines += 1
                if is_entry:
                    grounded_entries += 1
    return {
        "size": {
            "bytes": byte_count,
            "files": len(files),
            "entries": entries,
            "nonempty_lines": nonempty_lines,
        },
        "references": {
            "expanded_mentions": expanded_mentions,
            "valid_mentions": valid_mentions,
            "invalid_mentions": invalid_mentions,
            "unique_valid_ids": len(valid_ids),
            "unique_invalid_id_count": len(invalid_ids),
            "unique_invalid_ids": sorted(invalid_ids),
            "reference_precision": _ratio(valid_mentions, expanded_mentions),
            "valid_ids": sorted(valid_ids),
        },
        "grounding": {
            "grounded_entries": grounded_entries,
            "entry_groundedness": _ratio(grounded_entries, entries),
            "grounded_nonempty_lines": grounded_lines,
            "line_groundedness": _ratio(grounded_lines, nonempty_lines),
        },
    }


def gold_coverage(
    *,
    benchmark: str,
    questions: object,
    valid_references: set[str],
    answer_session_numbers: set[int] | None = None,
) -> dict[str, Any]:
    """Compute only the evidence coverage level supplied by each benchmark."""

    if not isinstance(questions, list):
        raise BuildAnalysisError("validated questions are not a list")
    if benchmark == "beam-100k":
        return {
            "availability": "not_available",
            "level": None,
            "eligible_questions": None,
            "question_any_coverage": None,
            "question_all_coverage": None,
            "unique_gold_targets": None,
            "unique_gold_target_coverage": None,
            "reason": "BEAM does not provide turn-level gold evidence IDs.",
        }
    if benchmark == "locomo":
        targets = [
            set(readonly_control.expand_source_specs(question["evidence"]))
            for question in questions
            if isinstance(question, Mapping)
            and isinstance(question.get("evidence"), list)
            and question["evidence"]
        ]
        level = "turn"
        observed: set[str] = set(valid_references)
    elif benchmark == "longmemeval-s":
        if (
            answer_session_numbers is None
            or not answer_session_numbers
            or len(questions) != 1
            or not isinstance(questions[0], Mapping)
            or not isinstance(questions[0].get("answer_session_ids"), list)
        ):
            raise BuildAnalysisError(
                "LongMemEval answer sessions were not strictly mapped"
            )
        targets = [{str(number) for number in answer_session_numbers}]
        observed = {
            match.group(1)
            for value in valid_references
            if (match := re.fullmatch(r"D(\d+):\d+", value)) is not None
        }
        level = "session"
    else:
        raise BuildAnalysisError(f"unsupported benchmark: {benchmark}")
    union = set().union(*targets) if targets else set()
    any_hits = sum(bool(target & observed) for target in targets)
    all_hits = sum(target <= observed for target in targets)
    return {
        "availability": "available",
        "level": level,
        "eligible_questions": len(targets),
        "question_any_coverage": _ratio(any_hits, len(targets)),
        "question_all_coverage": _ratio(all_hits, len(targets)),
        "unique_gold_targets": len(union),
        "unique_gold_target_coverage": _ratio(len(union & observed), len(union)),
    }


def longmemeval_answer_session_numbers(
    binding: contract.RunBinding,
    validated_unit: Mapping[str, Any],
) -> set[int]:
    """Map frozen LongMemEval session IDs to the generated Dn session index."""

    if binding.row.get("benchmark") != "longmemeval-s":
        raise BuildAnalysisError("session mapping is only defined for LongMemEval")
    raw = validated_unit.get("raw_dataset")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
        raise BuildAnalysisError("validated raw dataset binding is absent")
    dataset = build_runner.read_json(Path(raw["path"]))
    selection = binding.row.get("selection")
    if not isinstance(selection, Mapping):
        raise BuildAnalysisError("LongMemEval selection is absent")
    index = selection.get("dataset_index")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not isinstance(dataset, list)
        or not 0 <= index < len(dataset)
        or not isinstance(dataset[index], Mapping)
    ):
        raise BuildAnalysisError("LongMemEval dataset index is invalid")
    item = dataset[index]
    if item.get("question_id") != selection.get("question_id"):
        raise BuildAnalysisError("LongMemEval question identity changed")
    session_ids = item.get("haystack_session_ids")
    answer_ids = item.get("answer_session_ids")
    if (
        not isinstance(session_ids, list)
        or not session_ids
        or len(session_ids) != len(set(map(str, session_ids)))
        or not isinstance(answer_ids, list)
        or not answer_ids
        or len(answer_ids) != len(set(map(str, answer_ids)))
    ):
        raise BuildAnalysisError("LongMemEval session IDs are invalid")
    normalized_sessions = [str(value) for value in session_ids]
    normalized_answers = [str(value) for value in answer_ids]
    if not set(normalized_answers) <= set(normalized_sessions):
        raise BuildAnalysisError("LongMemEval answer sessions are outside history")
    questions = validated_unit.get("questions")
    if (
        not isinstance(questions, list)
        or len(questions) != 1
        or not isinstance(questions[0], Mapping)
        or list(map(str, questions[0].get("answer_session_ids", [])))
        != normalized_answers
    ):
        raise BuildAnalysisError("LongMemEval answer session IDs differ")
    positions = {value: number for number, value in enumerate(normalized_sessions, 1)}
    return {positions[value] for value in normalized_answers}


def build_report(
    *,
    rows: list[dict[str, Any]],
    matrix_path: Path,
    build_root: Path,
    execution_manifest_path: Path,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    required_run_hashes = {
        "matrix_row",
        "build",
        "success",
        "unit",
        "raw_dataset",
        "conversation",
        "questions",
        "memory",
    }
    for row in rows:
        identity = row.get("identity")
        hashes = row.get("source_hashes")
        if not isinstance(identity, Mapping) or not isinstance(hashes, Mapping):
            raise BuildAnalysisError("run identity or source hashes are absent")
        run_id = str(identity.get("run_id"))
        if run_id in by_id:
            raise BuildAnalysisError("analysis contains duplicate run IDs")
        if not required_run_hashes <= set(hashes):
            raise BuildAnalysisError("run source hash binding is incomplete")
        by_id[run_id] = row
    run_ids = validate_exact_run_ids(tuple(by_id))
    ordered_rows = [by_id[run_id] for run_id in run_ids]
    manifest = _validate_execution_manifest(
        path=execution_manifest_path,
        matrix_path=matrix_path,
        build_root=build_root,
    )
    raw_bindings = {
        (
            str(row["source_bindings"]["raw_dataset"]),
            str(row["source_hashes"]["raw_dataset"]),
        )
        for row in ordered_rows
    }
    return {
        "schema_version": 2,
        "status": "complete",
        "study": "gpt56-w32-screening-build-analysis",
        "build_root": str(build_root.expanduser().resolve()),
        "run_count": len(ordered_rows),
        "run_ids": list(run_ids),
        "runs": ordered_rows,
        "cost_summary": summarize_cost(ordered_rows),
        "raw_dataset_bindings": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(raw_bindings)
        ],
        "execution_manifest": manifest,
        "source_hashes": {
            "matrix": sha256_file(matrix_path),
            "analyzer": sha256_file(Path(__file__).resolve()),
            "qa_contract": sha256_file(Path(contract.__file__).resolve()),
            "canonical_source_parser": sha256_file(
                Path(readonly_control.__file__).resolve()
            ),
            "build_runner": sha256_file(Path(build_runner.__file__).resolve()),
            "atomic_report_contract": sha256_file(
                Path(answer_contract.__file__).resolve()
            ),
            "execution_manifest": sha256_file(execution_manifest_path),
        },
        "metric_definitions": {
            "source_bytes": "UTF-8 turn texts joined by one newline",
            "abstract_bytes": "sum of Markdown file bytes in topics and timeline",
            "entry": "nonempty Markdown line that is not a heading",
            "reference_mention": (
                "expanded canonical source IDs returned per nonempty line by "
                "readonly_nativemem_control.source_ids_in_text"
            ),
            "reference_precision": (
                "valid expanded mentions divided by all expanded mentions"
            ),
            "grounded_entry_or_line": (
                "entry or nonempty line containing at least one valid source ID"
            ),
            "cross_view_duplication": (
                "overlap of unique valid source IDs in topics and timeline; "
                "not semantic text duplication"
            ),
            "longmemeval_gold": (
                "session-level eligibility mapped from answer_session_ids through "
                "the frozen haystack_session_ids order"
            ),
            "beam_gold": "not available; reported as null rather than zero",
            "usd": "not available for subscription execution",
        },
        "external_build_audit_dependency": {
            "status": "required_separate_artifact",
            "performed_by_this_analyzer": False,
            "requirement": (
                "strict proxy-response, retry, provider, and request-ledger audit "
                "must be supplied by the separate campaign build auditor"
            ),
        },
        "claim_boundary": {
            "scope": "exact 12 completed Terra/Sol GPT-5.6 W32 screening builds",
            "source_reference_metric": (
                "syntactic validity and coverage of stable source-turn IDs"
            ),
            "gold_metric": (
                "LoCoMo turn-level evidence; LongMemEval session-level eligibility; "
                "BEAM unavailable"
            ),
            "not_established": [
                "semantic correctness of a memory entry",
                "full-benchmark accuracy",
                "causal effect of writer window size",
                "strict proxy-request integrity without the external build audit",
            ],
        },
    }


def _validate_execution_manifest(
    *, path: Path, matrix_path: Path, build_root: Path
) -> dict[str, Any]:
    try:
        value = build_runner.read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildAnalysisError("execution manifest is unreadable") from exc
    if not isinstance(value, Mapping):
        raise BuildAnalysisError("execution manifest is not an object")
    selected = value.get("selected_run_ids")
    if not isinstance(selected, list) or tuple(map(str, selected)) != EXPECTED_RUN_IDS:
        raise BuildAnalysisError("execution manifest differs from exact 12-run scope")
    if path.expanduser().resolve().parent != build_root.expanduser().resolve():
        raise BuildAnalysisError("execution manifest is outside the build root")
    if value.get("matrix") != str(matrix_path.expanduser().resolve()):
        raise BuildAnalysisError("execution manifest matrix path differs")
    if value.get("matrix_sha256") != sha256_file(matrix_path):
        raise BuildAnalysisError("execution manifest matrix hash differs")
    provider = value.get("provider")
    if (
        value.get("reasoning_effort") != "none"
        or not isinstance(provider, Mapping)
        or provider.get("name") != "subscription-proxy"
    ):
        raise BuildAnalysisError("execution manifest provider contract differs")
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": sha256_file(path),
        "selected_run_ids": list(EXPECTED_RUN_IDS),
        "matrix_sha256": sha256_file(matrix_path),
        "provider": "subscription-proxy",
        "reasoning_effort": "none",
    }


def _gold_reference_metrics(
    questions: object,
    valid_references: set[str],
) -> dict[str, Any]:
    if not isinstance(questions, list):
        raise BuildAnalysisError("validated questions are not a list")
    evidence_sets = [
        {str(value) for value in question.get("evidence", [])}
        for question in questions
        if isinstance(question, Mapping)
        and isinstance(question.get("evidence"), list)
        and question.get("evidence")
    ]
    gold_ids = set().union(*evidence_sets) if evidence_sets else set()
    any_hits = sum(bool(evidence & valid_references) for evidence in evidence_sets)
    all_hits = sum(evidence <= valid_references for evidence in evidence_sets)
    return {
        "gold_evidence_questions": len(evidence_sets),
        "gold_question_any_reference_recall": _ratio(any_hits, len(evidence_sets)),
        "gold_question_all_reference_recall": _ratio(all_hits, len(evidence_sets)),
        "unique_gold_evidence_ids": len(gold_ids),
        "unique_gold_evidence_id_recall": _ratio(
            len(gold_ids & valid_references), len(gold_ids)
        ),
    }


def _source_text_bytes(conversation: Mapping[str, Any]) -> bytes:
    texts: list[str] = []
    number = 1
    while f"session_{number}" in conversation:
        turns = conversation[f"session_{number}"]
        if not isinstance(turns, list):
            raise BuildAnalysisError("conversation session is not a list")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise BuildAnalysisError("conversation turn is not an object")
            texts.append(str(turn.get("text", "")))
        number += 1
    return "\n".join(texts).encode("utf-8")


def _integer_usage(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BuildAnalysisError(f"invalid usage field: {field}")
    return value


def _normalized_usage(value: object, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise BuildAnalysisError(f"{context} usage is absent")
    normalized = {
        field: _integer_usage(value.get(field, 0), field=f"{context}.{field}")
        for field in (
            "calls",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        )
    }
    if normalized["cached_input_tokens"] > normalized["input_tokens"]:
        raise BuildAnalysisError(f"{context} cached input exceeds input tokens")
    normalized["uncached_input_tokens"] = (
        normalized["input_tokens"] - normalized["cached_input_tokens"]
    )
    return normalized


def run_cost(build: Mapping[str, Any]) -> dict[str, Any]:
    usage = _normalized_usage(build.get("usage"), context="build")
    wall_time = build.get("wall_time_s")
    if (
        not isinstance(wall_time, (int, float))
        or isinstance(wall_time, bool)
        or wall_time < 0
    ):
        raise BuildAnalysisError("build wall time is invalid")
    phases = build.get("phase_usage")
    if not isinstance(phases, Mapping):
        raise BuildAnalysisError("build phase usage is absent")
    normalized_phases: dict[str, dict[str, Any]] = {}
    for phase in sorted(map(str, phases)):
        phase_usage = _normalized_usage(phases[phase], context=f"phase {phase}")
        normalized_phases[phase] = {
            **phase_usage,
            "fraction_of_run_calls": _ratio(
                phase_usage["calls"], usage["calls"]
            ),
            "fraction_of_run_input_tokens": _ratio(
                phase_usage["input_tokens"], usage["input_tokens"]
            ),
            "fraction_of_run_cached_input_tokens": _ratio(
                phase_usage["cached_input_tokens"],
                usage["cached_input_tokens"],
            ),
            "fraction_of_run_uncached_input_tokens": _ratio(
                phase_usage["uncached_input_tokens"],
                usage["uncached_input_tokens"],
            ),
            "fraction_of_run_output_tokens": _ratio(
                phase_usage["output_tokens"], usage["output_tokens"]
            ),
        }
    return {
        **usage,
        "wall_time_s": float(wall_time),
        "phase_usage": normalized_phases,
        "usd": {
            "availability": "not_available",
            "value": None,
            "currency": "USD",
            "reason": "subscription execution has no per-token billed price",
        },
    }


_COST_FIELDS = (
    "calls",
    "input_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "wall_time_s",
)


def _aggregate_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise BuildAnalysisError("cannot summarize an empty run group")
    totals: dict[str, int | float] = {field: 0 for field in _COST_FIELDS}
    phase_totals: dict[str, dict[str, int]] = {}
    for row in rows:
        cost = row.get("cost")
        if not isinstance(cost, Mapping):
            raise BuildAnalysisError("run cost is absent")
        for field in _COST_FIELDS:
            value = cost.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                raise BuildAnalysisError(f"invalid aggregated cost field: {field}")
            totals[field] += value
        phases = cost.get("phase_usage")
        if not isinstance(phases, Mapping):
            raise BuildAnalysisError("run phase usage is absent")
        for phase, value in phases.items():
            if not isinstance(value, Mapping):
                raise BuildAnalysisError("phase usage is not an object")
            target = phase_totals.setdefault(
                str(phase),
                {
                    field: 0
                    for field in _COST_FIELDS
                    if field != "wall_time_s"
                },
            )
            for field in target:
                observed = value.get(field)
                if (
                    not isinstance(observed, int)
                    or isinstance(observed, bool)
                    or observed < 0
                ):
                    raise BuildAnalysisError(
                        f"invalid aggregated phase field: {phase}.{field}"
                    )
                target[field] += observed
    run_count = len(rows)
    means = {field: value / run_count for field, value in totals.items()}
    phases_report: dict[str, dict[str, Any]] = {}
    for phase in sorted(phase_totals):
        value = phase_totals[phase]
        phases_report[phase] = {
            **value,
            "fraction_of_group_calls": _ratio(
                value["calls"], int(totals["calls"])
            ),
            "fraction_of_group_input_tokens": _ratio(
                value["input_tokens"], int(totals["input_tokens"])
            ),
            "fraction_of_group_cached_input_tokens": _ratio(
                value["cached_input_tokens"],
                int(totals["cached_input_tokens"]),
            ),
            "fraction_of_group_uncached_input_tokens": _ratio(
                value["uncached_input_tokens"],
                int(totals["uncached_input_tokens"]),
            ),
            "fraction_of_group_output_tokens": _ratio(
                value["output_tokens"], int(totals["output_tokens"])
            ),
        }
    return {
        "run_count": run_count,
        "totals": totals,
        "mean_per_run": means,
        "phase_usage": phases_report,
        "usd": {
            "availability": "not_available",
            "value": None,
            "currency": "USD",
            "reason": "subscription execution has no per-token billed price",
        },
    }


def summarize_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate physical build usage without assigning subscription USD cost."""

    by_tier: dict[str, list[dict[str, Any]]] = {}
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        identity = row.get("identity")
        if not isinstance(identity, Mapping):
            raise BuildAnalysisError("run identity is absent")
        by_tier.setdefault(str(identity.get("tier")), []).append(row)
        by_benchmark.setdefault(str(identity.get("benchmark")), []).append(row)
    return {
        "overall": _aggregate_cost(rows),
        "by_tier": {
            key: _aggregate_cost(by_tier[key]) for key in sorted(by_tier)
        },
        "by_benchmark": {
            key: _aggregate_cost(by_benchmark[key])
            for key in sorted(by_benchmark)
        },
    }


def _validate_run_artifacts(
    binding: contract.RunBinding,
    validated_unit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    build_path = binding.run_dir / "build.json"
    success_path = binding.run_dir / "memory" / "_SUCCESS.json"
    unit_path = binding.run_dir / "unit.json"
    try:
        build = build_runner.read_json(build_path)
        success = build_runner.read_json(success_path)
        unit = build_runner.read_json(unit_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildAnalysisError("build artifacts are unreadable") from exc
    if not all(isinstance(value, dict) for value in (build, success, unit)):
        raise BuildAnalysisError("build artifacts are not JSON objects")
    row = binding.row
    expected = {
        "run_id": binding.run_id,
        "benchmark": row.get("benchmark"),
        "unit_id": row.get("unit_id"),
        "tier": row.get("tier"),
        "builder_model": row.get("model"),
        "write_turns": 32,
        "reasoning_effort": "none",
    }
    if build.get("status") != "complete" or any(
        build.get(key) != value for key, value in expected.items()
    ):
        raise BuildAnalysisError("build identity differs from the exact W32 row")
    if any(
        success.get(key) != value
        for key, value in {
            "run_id": binding.run_id,
            "builder_model": row.get("model"),
            "write_turns": 32,
        }.items()
    ):
        raise BuildAnalysisError("success marker differs from the exact W32 row")
    for key in ("run_id", "benchmark", "unit_id", "selection"):
        if unit.get(key) != row.get(key):
            raise BuildAnalysisError(f"unit binding differs at {key}")
    if validated_unit.get("unit") != unit:
        raise BuildAnalysisError("validated unit differs from unit.json")
    for key in ("messages", "source_tokens", "source_tokenizer"):
        if unit.get(key) != build.get(key):
            raise BuildAnalysisError(f"unit and build differ at {key}")
    return build, success, unit


def analyze_run(
    binding: contract.RunBinding,
    validated_unit: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze one strictly completed W32 build without model requests."""

    build, _success, unit = _validate_run_artifacts(binding, validated_unit)
    build_path = binding.run_dir / "build.json"
    success_path = binding.run_dir / "memory" / "_SUCCESS.json"
    unit_path = binding.run_dir / "unit.json"
    conversation = validated_unit.get("conversation")
    if not isinstance(conversation, Mapping):
        raise BuildAnalysisError("validated conversation is absent")
    source_ids = set(readonly_control.build_turn_index(conversation))
    if len(source_ids) != build["messages"]:
        raise BuildAnalysisError("source turn count differs from build messages")
    memory_dir = binding.run_dir / "memory"
    views = {
        name: analyze_view(memory_dir / name, source_ids=source_ids)
        for name in ("topics", "timeline")
    }
    questions = validated_unit.get("questions")
    answer_sessions = (
        longmemeval_answer_session_numbers(binding, validated_unit)
        if build["benchmark"] == "longmemeval-s"
        else None
    )
    for view in views.values():
        view["gold_coverage"] = gold_coverage(
            benchmark=str(build["benchmark"]),
            questions=questions,
            valid_references=set(view["references"]["valid_ids"]),
            answer_session_numbers=answer_sessions,
        )
    topic_valid = set(views["topics"]["references"]["valid_ids"])
    timeline_valid = set(views["timeline"]["references"]["valid_ids"])
    combined_valid = topic_valid | timeline_valid
    invalid_ids = sorted(
        set(views["topics"]["references"]["unique_invalid_ids"])
        | set(views["timeline"]["references"]["unique_invalid_ids"])
    )
    expanded_mentions = sum(
        int(view["references"]["expanded_mentions"]) for view in views.values()
    )
    valid_mentions = sum(
        int(view["references"]["valid_mentions"]) for view in views.values()
    )
    invalid_mentions = sum(
        int(view["references"]["invalid_mentions"]) for view in views.values()
    )
    shared = topic_valid & timeline_valid
    union = topic_valid | timeline_valid
    smaller = min(len(topic_valid), len(timeline_valid))
    source_bytes = _source_text_bytes(conversation)
    messages = int(build["messages"])
    source_tokens = int(build["source_tokens"])
    abstract = {
        field: sum(int(view["size"][field]) for view in views.values())
        for field in ("bytes", "files", "entries", "nonempty_lines")
    }
    raw_dataset = validated_unit.get("raw_dataset")
    if not isinstance(raw_dataset, Mapping):
        raise BuildAnalysisError("validated raw dataset binding is absent")
    raw_path = raw_dataset.get("path")
    raw_sha = raw_dataset.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(raw_sha, str):
        raise BuildAnalysisError("validated raw dataset binding is invalid")
    if sha256_file(Path(raw_path)) != raw_sha:
        raise BuildAnalysisError("raw dataset hash changed after validation")
    row = {
        "identity": {
            "run_id": binding.run_id,
            "benchmark": build["benchmark"],
            "unit_id": build["unit_id"],
            "tier": build["tier"],
            "builder_model": build["builder_model"],
            "write_turns": build["write_turns"],
        },
        "source": {
            "turns": messages,
            "bytes": len(source_bytes),
            "tokens": source_tokens,
            "tokenizer": build["source_tokenizer"],
            "unique_source_ids": len(source_ids),
            "byte_definition": "UTF-8 turn texts joined by one newline",
        },
        "views": views,
        "combined_references": {
            "expanded_mentions": expanded_mentions,
            "valid_mentions": valid_mentions,
            "invalid_mentions": invalid_mentions,
            "unique_valid_ids": len(combined_valid),
            "unique_invalid_ids": invalid_ids,
            "reference_precision": _ratio(valid_mentions, expanded_mentions),
            "unique_source_id_coverage": _ratio(len(combined_valid), len(source_ids)),
            "gold_coverage": gold_coverage(
                benchmark=str(build["benchmark"]),
                questions=questions,
                valid_references=combined_valid,
                answer_session_numbers=answer_sessions,
            ),
        },
        "cross_view_reference_overlap": {
            "shared_unique_valid_ids": len(shared),
            "shared_fraction_of_union": _ratio(len(shared), len(union)),
            "shared_fraction_of_smaller_view": _ratio(len(shared), smaller),
        },
        "memory_size": {
            "source": {
                "bytes": len(source_bytes),
                "tokens": source_tokens,
                "turns": messages,
            },
            "abstract": abstract,
            "by_view": {
                name: dict(view["size"]) for name, view in views.items()
            },
            "abstract_to_source_byte_ratio": _ratio(
                int(abstract["bytes"]), len(source_bytes)
            ),
        },
        "cost": run_cost(build),
        "normalizations": {
            "calls_per_100_messages": _ratio(
                int(build["usage"]["calls"]) * 100, messages
            ),
            "input_tokens_per_1000_source_tokens": _ratio(
                int(build["usage"]["input_tokens"]) * 1000, source_tokens
            ),
            "output_tokens_per_1000_source_tokens": _ratio(
                int(build["usage"]["output_tokens"]) * 1000, source_tokens
            ),
        },
        "distill": build.get("distill"),
        "source_bindings": {
            "run_dir": str(binding.run_dir.expanduser().resolve()),
            "raw_dataset": str(Path(raw_path).expanduser().resolve()),
        },
        "source_hashes": {
            "matrix_row": answer_contract.canonical_hash(binding.row),
            "build": sha256_file(build_path),
            "success": sha256_file(success_path),
            "unit": sha256_file(unit_path),
            "raw_dataset": raw_sha,
            "conversation": answer_contract.canonical_hash(conversation),
            "questions": answer_contract.canonical_hash(questions),
            "memory": memory_sha256(memory_dir),
        },
        "validation": {
            "local_contract_validation": "passed",
            "build_unit_success_binding": "passed",
            "raw_dataset_reconstruction": "passed",
            "proxy_request_audit": "not_performed",
            "proxy_request_audit_dependency": (
                "requires the separate strict campaign build audit"
            ),
        },
    }
    return row


def analyze_campaign(
    *,
    matrix_path: Path,
    build_root: Path,
    execution_manifest_path: Path,
    output_path: Path,
    run_ids: Sequence[str] = EXPECTED_RUN_IDS,
) -> dict[str, Any]:
    """Validate, analyze, and persist the exact current W32 campaign."""

    canonical_run_ids = validate_exact_run_ids(tuple(run_ids))
    _validate_execution_manifest(
        path=execution_manifest_path,
        matrix_path=matrix_path,
        build_root=build_root,
    )
    try:
        bindings = contract.select_completed_runs(
            matrix_path=matrix_path,
            build_root=build_root,
            run_ids=canonical_run_ids,
        )
        validated_units = [
            contract.validate_unit_binding(binding) for binding in bindings
        ]
    except contract.QAContractError as exc:
        raise BuildAnalysisError(str(exc)) from exc
    rows = [
        analyze_run(binding, validated)
        for binding, validated in zip(bindings, validated_units, strict=True)
    ]
    report = build_report(
        rows=rows,
        matrix_path=matrix_path,
        build_root=build_root,
        execution_manifest_path=execution_manifest_path,
    )
    write_report_no_clobber(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        run_ids = validate_exact_run_ids(
            args.run_ids if args.run_ids is not None else EXPECTED_RUN_IDS
        )
        report = analyze_campaign(
            matrix_path=args.matrix,
            build_root=args.build_root,
            execution_manifest_path=args.execution_manifest,
            output_path=args.output,
            run_ids=run_ids,
        )
    except (BuildAnalysisError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    totals = report["cost_summary"]["overall"]["totals"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "output": str(args.output.expanduser().resolve()),
                "calls": totals["calls"],
                "input_tokens": totals["input_tokens"],
                "cached_input_tokens": totals["cached_input_tokens"],
                "uncached_input_tokens": totals["uncached_input_tokens"],
                "output_tokens": totals["output_tokens"],
                "wall_time_s": totals["wall_time_s"],
                "usd": None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_BUILD_ROOT",
    "DEFAULT_EXECUTION_MANIFEST",
    "DEFAULT_MATRIX",
    "DEFAULT_OUTPUT",
    "EXPECTED_RUN_IDS",
    "BuildAnalysisError",
    "analyze_campaign",
    "analyze_run",
    "analyze_view",
    "build_report",
    "gold_coverage",
    "longmemeval_answer_session_numbers",
    "main",
    "memory_sha256",
    "parse_args",
    "run_cost",
    "sha256_file",
    "summarize_cost",
    "validate_exact_run_ids",
    "write_report_no_clobber",
]


if __name__ == "__main__":
    raise SystemExit(main())
