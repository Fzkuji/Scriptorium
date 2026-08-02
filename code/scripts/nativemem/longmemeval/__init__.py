"""LongMemEval NativeMem runner components."""

from scripts.nativemem.common import atomic_json, read_json, stop_on_signal
from .execution import (
    answer_one,
    collect_answer,
    lme,
    run_pending,
)
from .results import load_completed_results, source_records, write_results


def main() -> int:
    from .cli import main as run

    return run()


__all__ = [
    "answer_one",
    "atomic_json",
    "collect_answer",
    "lme",
    "load_completed_results",
    "main",
    "read_json",
    "run_pending",
    "source_records",
    "stop_on_signal",
    "write_results",
]
