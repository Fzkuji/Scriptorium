#!/usr/bin/env python3
"""Freeze the formal shared-answerer and R004 protocol after input audits pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from baselines.locomo_baselines import audit_gpt55_locomo_baselines as baseline_auditor  # noqa: E402
from baselines.controlled_locomo import controlled_locomo_answer_contract as contract  # noqa: E402


DEFAULT_INPUT_ROOT = ROOT / "results/gpt55-locomo-baselines-20260714"


def _input_binding(method: str, run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    try:
        report = baseline_auditor.audit(run_dir)
    except (
        baseline_auditor.BaselineAuditError,
        baseline_auditor.runner.BaselineRunError,
        OSError,
    ) as exc:
        raise contract.ControlledAnswerError(
            f"{method} baseline input is unavailable or invalid: {exc}"
        ) from exc
    if report.get("status") != "passed":
        raise contract.ControlledAnswerError(f"{method} baseline audit did not pass")
    if report.get("method") != method:
        raise contract.ControlledAnswerError(
            f"{method} input contains method {report.get('method')!r}"
        )
    scope = report.get("scope")
    if not isinstance(scope, dict) or scope.get("kind") != "formal":
        raise contract.ControlledAnswerError(f"{method} input is not formal")
    if (
        scope.get("questions") != contract.EXPECTED_QUESTIONS
        or scope.get("primary_cat1_4") != contract.EXPECTED_PRIMARY
        or scope.get("adversarial_cat5") != contract.EXPECTED_ADVERSARIAL
    ):
        raise contract.ControlledAnswerError(f"{method} inventory differs")
    paths = {
        "run_manifest": run_dir / "run_manifest.json",
        "inputs_manifest": run_dir / "inputs_manifest.json",
        "questions": run_dir / "questions.json",
    }
    contract.ensure_distinct_paths(paths)
    manifest = contract.read_json(paths["run_manifest"])
    if not isinstance(manifest, dict):
        raise contract.ControlledAnswerError(f"{method} run manifest is invalid")
    return {
        "run_dir": str(run_dir),
        "method": method,
        "run_fingerprint": manifest.get("fingerprint"),
        "run_manifest_sha256": contract.sha256_file(paths["run_manifest"]),
        "inputs_manifest_sha256": contract.sha256_file(paths["inputs_manifest"]),
        "questions_sha256": contract.sha256_file(paths["questions"]),
        "dataset_sha256": report.get("dataset_sha256"),
        "inventory": {
            "questions": contract.EXPECTED_QUESTIONS,
            "primary_cat1_4": contract.EXPECTED_PRIMARY,
            "adversarial_cat5": contract.EXPECTED_ADVERSARIAL,
            "categories": {
                str(key): value for key, value in contract.EXPECTED_CATEGORIES.items()
            },
        },
        "baseline_audit_contract": {
            "source_hashes_match": report.get("source_hashes_match"),
            "task_scope": report.get("task_scope"),
            "evaluation": report.get("evaluation"),
        },
    }


def create_protocol(
    *,
    input_dirs: dict[str, Path],
    model_context_limit_tokens: int,
    answer_max_tokens_requested: int,
) -> dict[str, Any]:
    if model_context_limit_tokens <= 0:
        raise contract.ControlledAnswerError("model context limit must be positive")
    if answer_max_tokens_requested <= 0:
        raise contract.ControlledAnswerError("answer max tokens must be positive")
    tokenizer = contract.formal_token_counter()
    bindings = {
        method: _input_binding(method, input_dirs[method])
        for method in contract.FORMAL_METHODS
    }
    dataset_hashes = {binding["dataset_sha256"] for binding in bindings.values()}
    if len(dataset_hashes) != 1:
        raise contract.ControlledAnswerError("formal rows use different datasets")
    rows: list[dict[str, Any]] = []
    for method in contract.FORMAL_METHODS:
        full_context = method == "full_context"
        rows.append(
            {
                "method": method,
                "public_name": "Graphiti OSS" if method == "zep" else method,
                "input": bindings[method],
                "budget_policy": (
                    contract.FULL_CONTEXT_POLICY
                    if full_context
                    else contract.HARD_CAP_POLICY
                ),
                "budget_tokens": "unbounded" if full_context else contract.EXPECTED_HARD_BUDGET,
                "matched_cap_claim_allowed": not full_context,
                "full_context_name_requires_no_truncation": full_context,
                "model_context_limit_tokens": model_context_limit_tokens,
                "answer_completion_reservation_tokens": answer_max_tokens_requested,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": contract.PREREG_SCHEMA_VERSION,
        "status": "frozen",
        "frozen_at": contract.utc_now(),
        "benchmark": "LoCoMo",
        "formal_methods": list(contract.FORMAL_METHODS),
        "rows": rows,
        "answerer": {
            "model": contract.EXPECTED_MODEL,
            "temperature": 0,
            "prompt_template_sha256": contract.sha256_bytes(
                contract.ANSWER_PROMPT.encode("utf-8")
            ),
            "prompt_source_hashes": contract.prompt_source_hashes(ROOT),
            "answer_max_tokens_requested": answer_max_tokens_requested,
            "provider_hard_output_cap_claimed": False,
            "unsupported_parameter_and_actual_usage_must_be_recorded": True,
            "shared_across_all_rows": True,
        },
        "tokenizer": tokenizer.identity,
        "visible_token_gate": {
            "hard_cap_tokens": contract.EXPECTED_HARD_BUDGET,
            "hard_cap_methods": ["bm25", "mem0", "zep", "nativemem"],
            "full_context_policy": contract.FULL_CONTEXT_POLICY,
            "overflow_policy": "truncate",
            "rendering": "format_memories_equivalent_individual_events_v1",
            "event_order": "stable_chronological_if_any_date_else_input_order",
            "prompt_may_use": "DeliveryResult.delivered_text_only",
            "source_resolution_tokens_for_baselines": 0,
            "one_trace_and_manifest_per_question": True,
            "provider_exact": False,
        },
        "inventory": {
            "questions": contract.EXPECTED_QUESTIONS,
            "primary_cat1_4": contract.EXPECTED_PRIMARY,
            "adversarial_cat5": contract.EXPECTED_ADVERSARIAL,
            "categories": {
                str(key): value for key, value in contract.EXPECTED_CATEGORIES.items()
            },
        },
        "scoring_labels": {
            "source": "raw LoCoMo dataset, reconstructed by question_id",
            "categories_1_to_4": "raw_dataset.answer",
            "category_5": (
                "raw_dataset.answer when explicitly present; otherwise "
                "Not mentioned in the conversation"
            ),
            "adversarial_answer_role": "distractor_not_gold",
            "category_5_reported_separately": True,
            "baseline_input_gold_is_not_authoritative": True,
        },
        "future_integrations": {
            "nativemem": {
                "budget_policy": contract.HARD_CAP_POLICY,
                "budget_tokens": contract.EXPECTED_HARD_BUDGET,
                "status": "requires_canonical_questions_adapter",
            },
            "R203": {
                "budget_policy": contract.HARD_CAP_POLICY,
                "budget_tokens": contract.EXPECTED_HARD_BUDGET,
                "status": "requires_condition_runner_adapter",
            },
            "R301": {
                "budget_policy": contract.HARD_CAP_POLICY,
                "budget_tokens": contract.EXPECTED_HARD_BUDGET,
                "status": "requires_normalized_evidence_bundle_adapter",
            },
        },
        "claims": {
            "legacy_6000_token_claim_retained": False,
            "hard_cap_unit": "local tiktoken o200k_base tokens",
            "full_context_is_matched_cap": False,
        },
    }
    payload["protocol_content_sha256"] = contract.protocol_content_hash(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for method in contract.FORMAL_METHODS:
        parser.add_argument(
            f"--{method.replace('_', '-')}-input-dir",
            dest=f"{method}_input_dir",
            type=Path,
            default=DEFAULT_INPUT_ROOT / method,
        )
    parser.add_argument("--model-context-limit-tokens", type=int, required=True)
    parser.add_argument("--answer-max-tokens-requested", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dirs = {
        method: getattr(args, f"{method}_input_dir")
        for method in contract.FORMAL_METHODS
    }
    paths = {"output": args.output, **{f"input_{k}": v for k, v in input_dirs.items()}}
    contract.ensure_distinct_paths(paths)
    output = args.output.expanduser().absolute()
    for method, input_dir in input_dirs.items():
        source = input_dir.expanduser().absolute()
        try:
            output.relative_to(source)
        except ValueError:
            pass
        else:
            raise contract.ControlledAnswerError(
                f"protocol output must not be inside {method} input"
            )
    payload = create_protocol(
        input_dirs=input_dirs,
        model_context_limit_tokens=args.model_context_limit_tokens,
        answer_max_tokens_requested=args.answer_max_tokens_requested,
    )
    contract.atomic_json_no_clobber(output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": "frozen",
                "output": str(output.resolve()),
                "file_sha256": contract.sha256_file(output.resolve()),
                "protocol_content_sha256": payload["protocol_content_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except contract.ControlledAnswerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
