#!/usr/bin/env python3
"""Run paired NativeMem retrieval ablations on frozen benchmark memories."""

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.nativemem.ablation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
