#!/usr/bin/env python3
"""Re-answer frozen LongMemEval items from existing NativeMem libraries."""

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.nativemem.longmemeval import (  # noqa: E402
    answer_one,
    atomic_json,
    collect_answer,
    lme,
    load_completed_results,
    main,
    read_json,
    run_pending,
    source_records,
    stop_on_signal,
    write_results,
)

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


if __name__ == "__main__":
    raise SystemExit(main())
