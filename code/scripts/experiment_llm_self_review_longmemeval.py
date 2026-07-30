#!/usr/bin/env python3
"""Run generic LLM-directed retrieval variants on frozen LongMemEval memories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reanswer_longmemeval_existing_memory as reanswer  # noqa: E402
from scripts import run_v88_gpt55_longmemeval as lme  # noqa: E402


RETRIEVAL_ERROR_INDICES = (83, 109, 141, 153, 166, 299, 326, 389, 460)
CONTROL_INDICES = (7, 159, 193, 246, 373, 449)
PILOT_INDICES = frozenset(RETRIEVAL_ERROR_INDICES + CONTROL_INDICES)

REFLECTIVE_PROMPT = """You are answering one LongMemEval question by reading a NativeMem library.
The bash tool is read-only and its working directory is the library root.

The library has two complementary views:
- topics/: topic-oriented memory files with inline [Dn:m] source anchors.
- timeline/YYYY/MM/DD.md: chronological events with [Dn:m] anchors.
Use read_original when exact wording or nearby turns are needed. Never infer an
answer only from a file name.

Library structure:
{structure}

Current Date: {question_date}
Question: {question}

Search adaptively using the files and tools available. Before finalizing, form a
provisional answer and consider what missing or conflicting memory could change
it. If a plausible gap remains, continue searching with alternate wording,
related entities, or other relevant files. You decide which searches are useful
and when the evidence is sufficient. If the history genuinely lacks the
requested information, say so without inventing a fact.

Return the answer in the form appropriate for the question. Output exactly one
<answer>...</answer> block after tool use and do not expose reasoning.
"""

REVIEW_PROMPT = """Treat the previous answer as a draft. Reconsider whether any
missing or conflicting memory could change it. You may use the existing tools
again and decide what to inspect. When satisfied, output exactly one revised
<answer>...</answer> block and do not expose reasoning."""

COUNTERCHECK_ADDITION = """
Before finalizing, consider whether a missing or conflicting record could change
the current answer. If so, perform the most useful additional search you can
identify. Do not discard direct evidence merely because another search returns
no result.
"""

CONSERVATIVE_REVIEW_PROMPT = """Treat the previous answer as a supported draft.
Consider whether a missing or conflicting record could change it and use the
existing tools again if useful. Change the draft only when newly inspected
memory provides direct evidence for the change. A search returning no result is
not evidence against the draft. Then output exactly one final
<answer>...</answer> block and do not expose reasoning."""

VARIANTS = {
    "reflective-prompt": {
        "prompt_template": REFLECTIVE_PROMPT,
        "review_prompt": None,
    },
    "same-model-review": {
        "prompt_template": REFLECTIVE_PROMPT,
        "review_prompt": REVIEW_PROMPT,
    },
    "countercheck-prompt": {
        "prompt_template": lme.LME_SINGLE_PROMPT + COUNTERCHECK_ADDITION,
        "review_prompt": None,
    },
    "conservative-review": {
        "prompt_template": lme.LME_SINGLE_PROMPT,
        "review_prompt": CONSERVATIVE_REVIEW_PROMPT,
    },
}


def install_variant(name: str, *, pilot_only: bool = True) -> None:
    variant = VARIANTS[name]
    original_collect = lme.collect_and_answer_longmemeval
    original_sources = reanswer.source_records

    def collect(
        backend: Any,
        item: dict[str, Any],
        memory_dir: Path,
        turn_index: dict[str, Any],
    ) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
        return original_collect(
            backend,
            item,
            memory_dir,
            turn_index,
            prompt_template=variant["prompt_template"],
            review_prompt=variant["review_prompt"],
        )

    def pilot_sources(path: Path) -> list[dict[str, Any]]:
        return [
            record
            for record in original_sources(path)
            if int(record["dataset_index"]) in PILOT_INDICES
        ]

    lme.collect_and_answer_longmemeval = collect
    if pilot_only:
        reanswer.source_records = pilot_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--all-records", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    install_variant(args.variant, pilot_only=not args.all_records)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        return reanswer.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
