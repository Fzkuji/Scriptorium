#!/usr/bin/env python3
"""Build and evaluate one complete LoCoMo conversation with NativeMem V11."""

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.v11_locomo import (
    load_sample,
    main,
    parse_args,
    sample_inventory,
    summarize_usage,
)

__all__ = [
    "load_sample",
    "main",
    "parse_args",
    "sample_inventory",
    "summarize_usage",
]


if __name__ == "__main__":
    raise SystemExit(main())
