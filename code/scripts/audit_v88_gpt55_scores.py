#!/usr/bin/env python3
"""Strictly audit completed LoCoMo/LongMemEval-S judge outputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import score_v88_gpt55_benchmarks as scoring  # noqa: E402


class ScoreAuditError(RuntimeError):
    """Raised when a completed score artifact is not internally reproducible."""


def _expect_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ScoreAuditError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _audit_scores_unlocked(
    *,
    benchmark: str,
    source_input: Path,
    source_audit: Path,
    evaluation: Path,
    profile_name: str,
    proxy_log: Path | None = None,
) -> dict[str, Any]:
    source_input = source_input.expanduser().resolve()
    source_audit = source_audit.expanduser().resolve()
    evaluation = evaluation.expanduser().resolve()
    try:
        provenance = scoring.validate_upstream_audit(
            benchmark, source_input, source_audit
        )
    except scoring.ScoringError as exc:
        raise ScoreAuditError(str(exc)) from exc
    expected_proxy_log = Path(provenance["proxy_log_path"])
    if proxy_log is not None:
        proxy_log = proxy_log.expanduser().resolve()
    if profile_name == "primary" and proxy_log is not None:
        raise ScoreAuditError("--proxy-log is valid only for the secondary judge")
    source_records = scoring._load_record_list(source_input)
    selected = scoring.select_official_records(source_records, benchmark)
    evaluation_payload = evaluation.read_bytes()
    evaluation_sha256 = hashlib.sha256(evaluation_payload).hexdigest()
    try:
        value = json.loads(evaluation_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreAuditError("evaluation output is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ScoreAuditError("evaluation output is not a JSON object")
    meta = value.get("meta")
    records = value.get("results")
    if not isinstance(meta, dict) or not isinstance(records, list):
        raise ScoreAuditError("evaluation output must contain meta and results")

    profile = scoring.judge_profile(benchmark, profile_name)
    transport_contract = meta.get("transport_contract")
    gateway_binding = meta.get("openrouter_gateway")
    flex_gateway_binding = meta.get("openai_flex_gateway")
    if profile_name == "primary":
        if transport_contract == "marked_openrouter_gateway":
            if not isinstance(gateway_binding, dict):
                raise ScoreAuditError("primary gateway binding is absent")
            try:
                scoring.openrouter_gateway_evidence.validate_binding(
                    gateway_binding,
                    result_root=Path(
                        str(gateway_binding.get("result_root", ""))
                    ),
                    base_url=str(gateway_binding.get("base_url", "")),
                )
            except scoring.openrouter_gateway_evidence.GatewayEvidenceError as exc:
                raise ScoreAuditError(str(exc)) from exc
            profile["base_url"] = str(gateway_binding["base_url"])
        elif transport_contract == "injected_test_judge":
            if gateway_binding is not None:
                raise ScoreAuditError("injected test judge claims gateway evidence")
        else:
            raise ScoreAuditError("primary transport contract is not recognized")
    elif transport_contract == "openai_gpt55_flex_gateway":
        if gateway_binding is not None or not isinstance(flex_gateway_binding, dict):
            raise ScoreAuditError("secondary Flex gateway binding differs")
        if proxy_log is not None:
            raise ScoreAuditError("secondary Flex audit does not accept --proxy-log")
        try:
            scoring.flex_evidence.validate_recorded_contract(flex_gateway_binding)
        except scoring.flex_evidence.EvidenceError as exc:
            raise ScoreAuditError(str(exc)) from exc
        profile["base_url"] = str(flex_gateway_binding["base_url"])
    elif transport_contract == "secondary_local_proxy":
        if gateway_binding is not None or flex_gateway_binding is not None:
            raise ScoreAuditError("injected secondary claims gateway evidence")
        if proxy_log is None:
            proxy_log = expected_proxy_log
        if proxy_log != expected_proxy_log:
            raise ScoreAuditError(
                "secondary judge proxy log differs from the upstream shared log"
            )
    else:
        raise ScoreAuditError("secondary transport contract differs")
    current_hashes = scoring.code_hashes()
    base_run_key = scoring.sha256_json({
        "benchmark": benchmark,
        "profile": profile_name,
        "source_input_sha256": scoring.sha256_file(source_input),
        "selected_input_sha256": scoring.sha256_json(selected),
        "source_manifest_sha256": provenance["manifest_sha256"],
        "code_hashes": current_hashes,
        "transport_contract": transport_contract,
        "openrouter_gateway": scoring.gateway_start_binding(gateway_binding),
        "openai_flex_gateway": scoring.flex_gateway_start_binding(
            flex_gateway_binding
        ),
        "flex_evidence_root": meta.get("flex_evidence_root"),
    })
    generation_id = meta.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ScoreAuditError("evaluation has no durable generation id")
    expected_run_key = scoring.sha256_json({
        "base_run_key": base_run_key,
        "generation_id": generation_id,
    })
    ledger_path = scoring.attempt_ledger_path(evaluation)
    required_meta = {
        "schema_version": 3,
        "status": "complete",
        "benchmark": benchmark,
        "scope": scoring.SCOPES[benchmark],
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "judge_profile": profile["id"],
        "judge_provider": profile["provider"],
        "judge_requested_model": profile["requested_model"],
        "judge_base_url": profile["base_url"],
        "comparison_status": profile["comparison_status"],
        "comparable_to_published_primary": profile[
            "comparable_to_published_primary"
        ],
        "comparison_note": profile["note"],
        "metrics": scoring.metrics_for_benchmark(benchmark),
        "source_input": str(source_input),
        "source_input_sha256": scoring.sha256_file(source_input),
        "selected_input_sha256": scoring.sha256_json(selected),
        "source_dataset_sha256": scoring.dataset_sha256(benchmark),
        "source_audit": provenance["audit_path"],
        "source_audit_sha256": provenance["audit_sha256"],
        "source_run_dir": provenance["run_dir"],
        "source_manifest": provenance["manifest_path"],
        "source_manifest_sha256": provenance["manifest_sha256"],
        "upstream_proxy_evidence": provenance["upstream_proxy_evidence"],
        "judge_proxy_log": str(proxy_log) if proxy_log else None,
        "code_hashes": current_hashes,
        "run_key": expected_run_key,
        "generation_id": generation_id,
        "attempt_ledger": scoring.ledger_report(ledger_path, expected_run_key),
        "transport_contract": transport_contract,
        "openrouter_gateway": gateway_binding,
        "openai_flex_gateway": flex_gateway_binding,
        "flex_evidence_root": meta.get("flex_evidence_root"),
    }
    for key, expected in required_meta.items():
        _expect_equal(f"meta.{key}", meta.get(key), expected)
    if "last_error" in meta:
        raise ScoreAuditError("completed evaluation retains last_error metadata")
    if not meta.get("created_at") or not meta.get("finished_at"):
        raise ScoreAuditError("evaluation timestamps are incomplete")
    if len(records) != len(selected):
        raise ScoreAuditError("evaluation record count differs from selected input")

    expected_with_lexical = copy.deepcopy(selected)
    scoring.add_lexical_metrics(expected_with_lexical)
    if benchmark == "locomo-cat5":
        scoring.add_cat5_refusal_diagnostics(expected_with_lexical)
    scored = []
    requested_models: list[str] = []
    response_models: list[str] = []
    response_ids: list[str] = []
    finish_reasons: list[str] = []
    refusals: list[object] = []
    choice_counts: list[int] = []
    prompt_tokens = 0
    completion_tokens = 0
    parse_attempts = 0
    request_attempts = 0
    failed_request_attempts = 0
    unknown_token_attempts = 0
    request_response_ids: list[str] = []
    for index, (source, expected_lexical, record) in enumerate(
        zip(selected, expected_with_lexical, records)
    ):
        if not isinstance(record, dict):
            raise ScoreAuditError(f"evaluation record {index} is not an object")
        if scoring._base_record(record) != source:
            raise ScoreAuditError(f"evaluation source fields differ at index {index}")
        if record.get("question_id") == "_build_stats":
            unexpected = scoring.DERIVED_QUESTION_KEYS.intersection(record)
            if unexpected:
                raise ScoreAuditError(
                    f"build record {index} contains scoring fields: {sorted(unexpected)}"
                )
            continue
        question_id = str(record.get("question_id"))
        _expect_equal(
            f"{question_id}.lexical",
            record.get("lexical"), expected_lexical.get("lexical"),
        )
        if benchmark == "locomo-cat5":
            _expect_equal(
                f"{question_id}.lexical_refusal",
                record.get("lexical_refusal"),
                expected_lexical.get("lexical_refusal"),
            )
        try:
            scoring.validate_judge_result(
                record.get("judge_score"), record.get("judge_raw"),
                record.get("judge_usage"), benchmark, profile_name, question_id,
            )
        except scoring.ScoringError as exc:
            raise ScoreAuditError(str(exc)) from exc
        usage = scoring.normalize_judge_usage(record["judge_usage"])
        scored.append(record)
        requested_models.extend(usage["requested_models"])
        response_models.extend(usage["response_models"])
        response_ids.extend(usage["response_ids"])
        finish_reasons.extend(usage["finish_reasons"])
        refusals.extend(usage["refusals"])
        choice_counts.extend(usage["choice_counts"])
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        parse_attempts += usage["parse_attempts"]
        request_attempts += usage["request_attempts"]
        failed_request_attempts += usage["failed_request_attempts"]
        unknown_token_attempts += usage["unknown_token_attempts"]
        request_response_ids.extend(
            detail["response_id"]
            for detail in usage["request_attempt_details"]
            if detail["response_id"] is not None
        )

    expected_questions = {
        "locomo": 1540,
        "locomo-cat5": 446,
        "longmemeval": 500,
    }[benchmark]
    expected_builds = 10 if benchmark in scoring.LOCOMO_BENCHMARKS else 500
    _expect_equal("scored question count", len(scored), expected_questions)
    _expect_equal(
        "build record count",
        sum(record.get("question_id") == "_build_stats" for record in records),
        expected_builds,
    )
    question_ids = [str(record["question_id"]) for record in scored]
    if len(set(question_ids)) != expected_questions:
        raise ScoreAuditError("evaluation question ids are duplicated")
    if len(set(response_ids)) != len(response_ids):
        raise ScoreAuditError("judge response ids are duplicated")
    if len(set(request_response_ids)) != len(request_response_ids):
        raise ScoreAuditError("judge request-attempt response ids are duplicated")

    correct = sum(record["judge_score"] for record in scored)
    _expect_equal("meta.question_count", meta.get("question_count"), expected_questions)
    _expect_equal("meta.completed_questions", meta.get("completed_questions"), expected_questions)
    _expect_equal("meta.correct_questions", meta.get("correct_questions"), correct)
    aggregate = scoring.recompute_aggregate(records, benchmark)
    _expect_equal("meta.aggregate", meta.get("aggregate"), aggregate)

    if transport_contract == "openai_gpt55_flex_gateway":
        proxy_evidence = None
        try:
            gateway_evidence = scoring.validate_secondary_flex_evidence(
                records, Path(str(meta.get("flex_evidence_root", "")))
            )
        except scoring.ScoringError as exc:
            raise ScoreAuditError(str(exc)) from exc
    elif profile_name == "secondary":
        try:
            proxy_evidence = scoring.validate_secondary_proxy_evidence(
                records,
                proxy_log,
                frozen=meta.get("judge_proxy_evidence"),
            )
        except scoring.ScoringError as exc:
            raise ScoreAuditError(str(exc)) from exc
        gateway_evidence = None
    else:
        proxy_evidence = None
    _expect_equal(
        "meta.judge_proxy_evidence",
        meta.get("judge_proxy_evidence"),
        proxy_evidence,
    )
    if transport_contract == "marked_openrouter_gateway":
        try:
            gateway_evidence = (
                scoring.openrouter_gateway_evidence.verify_response_ids(
                    gateway_binding, response_ids
                )
            )
        except scoring.openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise ScoreAuditError(str(exc)) from exc
    elif transport_contract == "injected_test_judge":
        gateway_evidence = {"status": "synthetic_injected_judge_only"}
    elif transport_contract != "openai_gpt55_flex_gateway":
        gateway_evidence = None
    _expect_equal(
        "meta.judge_gateway_evidence",
        meta.get("judge_gateway_evidence"),
        gateway_evidence,
    )

    ledger_events = scoring.read_attempt_ledger(ledger_path)
    try:
        scoring.reconcile_attempt_ledger(
            ledger_events,
            expected_run_key,
            benchmark=benchmark,
            profile_name=profile_name,
        )
    except scoring.ScoringError as exc:
        raise ScoreAuditError(str(exc)) from exc
    result_events: dict[str, dict[str, Any]] = {}
    failure_ids: set[str] = set()
    for event in ledger_events:
        if event.get("run_key") != expected_run_key:
            continue
        if event.get("event") == "judge_failure":
            failure_ids.add(event["event_id"])
        elif event.get("event") == "judge_result":
            question_id = event.get("question_id")
            if not isinstance(question_id, str):
                raise ScoreAuditError("attempt ledger result lacks question id")
            result_events[question_id] = event
    if set(result_events) != set(question_ids):
        raise ScoreAuditError("attempt ledger result inventory differs from scores")
    consumed_failure_ids: set[str] = set()
    by_question = {str(record["question_id"]): record for record in scored}
    for question_id, event in result_events.items():
        record = by_question[question_id]
        _expect_equal(f"{question_id}.ledger score", event.get("score"), record["judge_score"])
        _expect_equal(f"{question_id}.ledger raw", event.get("raw"), record["judge_raw"])
        _expect_equal(
            f"{question_id}.ledger usage",
            scoring.normalize_judge_usage(event.get("usage")),
            scoring.normalize_judge_usage(record["judge_usage"]),
        )
        consumed = event.get("consumed_failure_event_ids", [])
        if not isinstance(consumed, list) or any(
            not isinstance(value, str) for value in consumed
        ):
            raise ScoreAuditError("attempt ledger has invalid failure references")
        consumed_failure_ids.update(consumed)
    if consumed_failure_ids != failure_ids:
        raise ScoreAuditError(
            "attempt ledger failure inventory is not exactly consumed by results"
        )
    physical_events = sum(
        event.get("run_key") == expected_run_key
        and event.get("event") == "physical_http_attempt"
        for event in ledger_events
    )
    logical_events = sum(
        event.get("run_key") == expected_run_key
        and event.get("event") == "logical_judge_call"
        for event in ledger_events
    )
    _expect_equal("ledger physical attempts", physical_events, request_attempts)
    _expect_equal(
        "ledger logical calls",
        logical_events,
        sum(
            scoring.normalize_judge_usage(record["judge_usage"])[
                "logical_judge_calls"
            ]
            for record in scored
        ),
    )

    if benchmark == "locomo":
        _expect_equal("aggregate.n_main", aggregate.get("n_main"), 1540)
        _expect_equal("aggregate.n_adversarial", aggregate.get("n_adversarial"), 0)
    elif benchmark == "locomo-cat5":
        _expect_equal("aggregate.n", aggregate.get("n"), 446)
        _expect_equal("aggregate.n_abstention", aggregate.get("n_abstention"), 444)
        _expect_equal(
            "aggregate.n_explicit_answer", aggregate.get("n_explicit_answer"), 2
        )
        _expect_equal(
            "aggregate.excluded_from_cat1_4",
            aggregate.get("excluded_from_cat1_4"),
            True,
        )
        _expect_equal(
            "aggregate semantic accuracy",
            aggregate.get("semantic_judge_accuracy"),
            correct / 446,
        )
        abstention_records = [
            record for record in scored if record.get("cat5_abstention") is True
        ]
        explicit_answer_records = [
            record for record in scored if record.get("cat5_abstention") is False
        ]
        _expect_equal(
            "aggregate abstention semantic accuracy",
            aggregate.get("semantic_abstention_judge_accuracy"),
            sum(record["judge_score"] for record in abstention_records) / 444,
        )
        _expect_equal(
            "aggregate explicit-answer accuracy",
            aggregate.get("explicit_answer_judge_accuracy"),
            sum(record["judge_score"] for record in explicit_answer_records) / 2,
        )
    else:
        _expect_equal("aggregate.n", aggregate.get("n"), 500)
        _expect_equal("aggregate.n_abstention", aggregate.get("n_abstention"), 30)
        expected_type_counts = {
            "multi-session": 133,
            "temporal-reasoning": 133,
            "knowledge-update": 78,
            "single-session-user": 70,
            "single-session-assistant": 56,
            "single-session-preference": 30,
        }
        actual_type_counts = {
            question_type: block.get("n")
            for question_type, block in aggregate.get("by_type", {}).items()
        }
        _expect_equal(
            "aggregate LongMemEval type counts",
            actual_type_counts,
            expected_type_counts,
        )
        _expect_equal("aggregate.overall_acc", aggregate.get("overall_acc"), correct / 500)

    report = {
        "status": "passed",
        "evaluation": str(evaluation),
        "benchmark": benchmark,
        "scope": meta.get("scope"),
        "method": meta["method"],
        "backbone": meta["backbone"],
        "judge_profile": meta["judge_profile"],
        "judge_provider": meta["judge_provider"],
        "judge_requested_model": meta["judge_requested_model"],
        "comparison_status": meta["comparison_status"],
        "comparable_to_published_primary": meta[
            "comparable_to_published_primary"
        ],
        "comparison_note": meta["comparison_note"],
        "source_provenance": {
            "run_dir": provenance["run_dir"],
            "audit": provenance["audit_path"],
            "manifest": provenance["manifest_path"],
            "upstream_proxy_evidence": provenance["upstream_proxy_evidence"],
        },
        "questions": expected_questions,
        "correct": correct,
        "accuracy": correct / expected_questions,
        "aggregate": aggregate,
        "judge_proxy_evidence": proxy_evidence,
        "judge_gateway_evidence": gateway_evidence,
        "attempt_ledger": scoring.ledger_report(ledger_path, expected_run_key),
        "judge_usage": {
            "parse_attempts": parse_attempts,
            "logical_judge_calls": sum(
                scoring.normalize_judge_usage(record["judge_usage"])[
                    "logical_judge_calls"
                ] for record in scored
            ),
            "request_attempts": request_attempts,
            "physical_http_attempts": sum(
                scoring.normalize_judge_usage(record["judge_usage"])[
                    "physical_http_attempts"
                ] for record in scored
            ),
            "failed_request_attempts": failed_request_attempts,
            "unknown_token_attempts": unknown_token_attempts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "requested_models": _counter_dict(requested_models),
            "response_models": _counter_dict(response_models),
            "unique_response_ids": len(set(response_ids)),
            "unique_request_response_ids": len(set(request_response_ids)),
            "finish_reasons": _counter_dict(finish_reasons),
            "refusal_attempts": sum(value is not None for value in refusals),
            "choice_counts": dict(sorted(Counter(choice_counts).items())),
        },
        "hashes": {
            "source_input_sha256": scoring.sha256_file(source_input),
            "selected_input_sha256": scoring.sha256_json(selected),
            "evaluation_sha256": evaluation_sha256,
            "source_dataset_sha256": scoring.dataset_sha256(benchmark),
            "source_audit_sha256": provenance["audit_sha256"],
            "source_manifest_sha256": provenance["manifest_sha256"],
            "code": current_hashes,
            "auditor": scoring.sha256_file(Path(__file__).resolve()),
        },
    }
    return report


def audit_scores(
    *,
    benchmark: str,
    source_input: Path,
    source_audit: Path,
    evaluation: Path,
    profile_name: str,
    proxy_log: Path | None = None,
) -> dict[str, Any]:
    """Audit one immutable evaluation snapshot under the scorer's lock."""
    evaluation = evaluation.expanduser().resolve()
    with scoring.exclusive_output_lock(evaluation):
        return _audit_scores_unlocked(
            benchmark=benchmark,
            source_input=source_input,
            source_audit=source_audit,
            evaluation=evaluation,
            profile_name=profile_name,
            proxy_log=proxy_log,
        )


def _recorded_proxy_path(source_audit: Path) -> Path | None:
    """Resolve the recorded shared log before any audit output is removed."""
    try:
        report = scoring.read_json(source_audit.expanduser().resolve())
    except (OSError, json.JSONDecodeError):
        return None
    value = report.get("proxy_window", {}).get("path") if isinstance(report, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=sorted(scoring.SCOPES))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--judge", required=True, choices=sorted(scoring.JUDGE_PROFILES))
    parser.add_argument(
        "--proxy-log", type=Path,
        help="shared GPT-5.5 proxy JSONL; secondary only (defaults to source audit)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evaluation = args.evaluation.expanduser().resolve()
    source_input = args.input.expanduser().resolve()
    source_audit = args.source_audit.expanduser().resolve()
    proxy_log = args.proxy_log.expanduser().resolve() if args.proxy_log else None
    output = (
        args.output.expanduser().resolve()
        if args.output else evaluation.with_suffix(".audit.json")
    )
    try:
        recorded_proxy = _recorded_proxy_path(source_audit)
        scoring.ensure_distinct_paths({
            "audit_output": output,
            "evaluation": evaluation,
            "evaluation_lock": Path(f"{evaluation}.lock"),
            "source_input": source_input,
            "source_audit": source_audit,
            "source_manifest": source_input.parent / "run_manifest.json",
            "proxy_log": proxy_log or recorded_proxy,
            "audit_registry": scoring.score_audit_registry_path(evaluation),
            "attempt_ledger": scoring.attempt_ledger_path(evaluation),
        })
        with scoring.exclusive_output_lock(evaluation):
            scoring.register_score_audit(evaluation, output)
            output.unlink(missing_ok=True)
            try:
                report = _audit_scores_unlocked(
                    benchmark=args.benchmark,
                    source_input=source_input,
                    source_audit=source_audit,
                    evaluation=evaluation,
                    profile_name=args.judge,
                    proxy_log=proxy_log,
                )
                scoring.atomic_json(output, report)
            except Exception:
                output.unlink(missing_ok=True)
                raise
    except (
        ScoreAuditError, scoring.ScoringError, OSError, json.JSONDecodeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
