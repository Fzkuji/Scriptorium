#!/usr/bin/env python3
"""Run the official LongMemEval judge through an OpenAI-compatible gateway."""

from pathlib import Path


OFFICIAL_EVALUATOR = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/longmemeval/src/evaluation/evaluate_qa.py"
)
REPLACEMENTS = (
    (
        "openai_api_base = None",
        "openai_api_base = os.getenv('OPENAI_BASE_URL')",
    ),
    (
        "metric_model, metric_model_source = model_zoo[metric_model_short]",
        "metric_model, metric_model_source = model_zoo[metric_model_short]\n"
        "    metric_model = os.getenv('OPENAI_MODEL', metric_model)",
    ),
)


def patched_source(source: str) -> str:
    for original, replacement in REPLACEMENTS:
        if source.count(original) != 1:
            raise RuntimeError(f"official evaluator source differs: {original}")
        source = source.replace(original, replacement)
    return source


if __name__ == "__main__":
    source = patched_source(OFFICIAL_EVALUATOR.read_text())
    exec(
        compile(source, str(OFFICIAL_EVALUATOR), "exec"),
        {"__name__": "__main__", "__file__": str(OFFICIAL_EVALUATOR)},
    )
