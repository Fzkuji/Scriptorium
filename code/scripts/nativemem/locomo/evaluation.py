"""Invocation of the immutable LoCoMo evaluator."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.nativemem.common import sha256_file


CODE_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = CODE_ROOT / "scripts" / "evaluation" / "eval_full.py"
EVALUATOR_SHA256 = "17ef2179cd2781880649eed4a7d62988c069123b5044f007c194cdab8e63f88b"


def verify_evaluator() -> None:
    if sha256_file(EVALUATOR) != EVALUATOR_SHA256:
        raise RuntimeError("locked LoCoMo evaluator hash mismatch")


def run_evaluator(args: argparse.Namespace, output_dir: Path) -> None:
    verify_evaluator()
    log_path = output_dir / "eval.log"
    command = [
        sys.executable,
        str(EVALUATOR),
        str(output_dir),
        args.model,
        args.base_url,
        args.api_key,
        args.judge_api_key,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=CODE_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"locked evaluator failed; see {log_path}")
