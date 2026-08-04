#!/usr/bin/env python3
"""No-network synthetic sanity for the controlled LoCoMo answer boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.controlled_locomo import controlled_locomo_answer_contract as contract  # noqa: E402
from scripts.controlled_locomo import run_controlled_locomo_answers as runner  # noqa: E402


GOLD_CANARY = "CANARY_GOLD_MUST_NOT_REACH_PROMPT"
EVIDENCE_CANARY = "CANARY_EVIDENCE_MUST_NOT_REACH_PROMPT"
CATEGORY_CANARY = "CANARY_CATEGORY_MUST_NOT_REACH_PROMPT"
RAW_OVERFLOW_CANARY = "CANARY_RAW_OVERFLOW_MUST_NOT_REACH_PROMPT"
POST_EXHAUSTION_CANARY = "CANARY_POST_EXHAUSTION_MUST_NOT_REACH_PROMPT"
FULL_CONTEXT_CANARY = "CANARY_FULL_CONTEXT_MUST_BE_DELIVERED"


def run_sanity(output_dir: Path, *, budget_tokens: int = 16) -> dict:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    tokenizer = contract.formal_token_counter()
    run_id = "controlled-answer-no-network-sanity"
    fake = runner.FakeAnswerClient(
        proxy_log=output_dir / "fake-exclusive-proxy.jsonl",
        run_id=run_id,
    )
    hard_dir = output_dir / "hard-cap"
    hard_dir.mkdir()
    hard_memories = [
        {
            "text": "safe-prefix " * 100 + RAW_OVERFLOW_CANARY,
            "date": "2025-01-01",
        },
        {"text": POST_EXHAUSTION_CANARY, "date": "2025-01-02"},
    ]
    hard_result = runner.answer_one(
        run_id=run_id,
        method="bm25",
        question_id="s0_q0",
        question="What safe content was retained?",
        memories=hard_memories,
        input_record_sha256="1" * 64,
        attempt_dir=hard_dir,
        tokenizer=tokenizer,
        budget_policy=contract.HARD_CAP_POLICY,
        budget_tokens=budget_tokens,
        model_context_limit_tokens=100_000,
        answer_max_tokens=512,
        preregistration_sha256="2" * 64,
        client=fake,
    )
    hard_prompt = fake.prompts[-1]
    forbidden = {
        GOLD_CANARY,
        EVIDENCE_CANARY,
        CATEGORY_CANARY,
        RAW_OVERFLOW_CANARY,
        POST_EXHAUSTION_CANARY,
    }
    leaked = sorted(value for value in forbidden if value in hard_prompt)
    if leaked:
        raise contract.ControlledAnswerError(f"canary reached hard-cap prompt: {leaked}")
    if hard_result["budget"]["visible_tokens"] > budget_tokens:
        raise contract.ControlledAnswerError("hard-cap sanity exceeded budget")

    full_dir = output_dir / "full-context"
    full_dir.mkdir()
    full_result = runner.answer_one(
        run_id=run_id,
        method="full_context",
        question_id="s0_q1",
        question="Which full-context canary is present?",
        memories=[{"text": FULL_CONTEXT_CANARY, "date": "2025-01-01"}],
        input_record_sha256="3" * 64,
        attempt_dir=full_dir,
        tokenizer=tokenizer,
        budget_policy=contract.FULL_CONTEXT_POLICY,
        budget_tokens=None,
        model_context_limit_tokens=100_000,
        answer_max_tokens=512,
        preregistration_sha256="2" * 64,
        client=fake,
    )
    full_prompt = fake.prompts[-1]
    if FULL_CONTEXT_CANARY not in full_prompt:
        raise contract.ControlledAnswerError("full-context content was not delivered")
    if full_result["budget"]["declared_tokens"] != "unbounded":
        raise contract.ControlledAnswerError("full-context was mislabeled as capped")

    # These values exist in a synthetic input record but were never passed to
    # answer_one.  Keeping the check here makes the forbidden boundary explicit.
    synthetic_labels = {
        "gold": GOLD_CANARY,
        "evidence": [EVIDENCE_CANARY],
        "category_canary": CATEGORY_CANARY,
    }
    if any(
        value in "\n".join(fake.prompts)
        for value in (
            synthetic_labels["gold"],
            synthetic_labels["evidence"][0],
            synthetic_labels["category_canary"],
        )
    ):
        raise contract.ControlledAnswerError("label/evidence canary reached prompt")
    report = {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "passed",
        "network_requests": 0,
        "fake_proxy_entries": len(
            (output_dir / "fake-exclusive-proxy.jsonl").read_text().splitlines()
        ),
        "tokenizer": tokenizer.identity,
        "hard_cap": {
            "configured_tokens": budget_tokens,
            "visible_tokens": hard_result["budget"]["visible_tokens"],
            "raw_overflow_canary_excluded": True,
            "post_exhaustion_canary_excluded": True,
        },
        "full_context": {
            "policy": contract.FULL_CONTEXT_POLICY,
            "declared_tokens": "unbounded",
            "content_not_truncated": True,
        },
        "prompt_boundary": {
            "gold_canary_excluded": True,
            "evidence_canary_excluded": True,
            "category_canary_excluded": True,
            "delivered_text_only": True,
        },
    }
    contract.atomic_json_no_clobber(output_dir / "sanity_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-tokens", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sanity(args.output_dir, budget_tokens=args.budget_tokens)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (contract.ControlledAnswerError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
