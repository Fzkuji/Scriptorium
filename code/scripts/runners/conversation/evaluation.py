"""Invocation of the evaluator a benchmark is scored by.

LoCoMo is scored by the locked evaluator, whose hash is checked before every
run so a reported LoCoMo number always came from the same code. Benchmarks
added since are scored by the unified evaluator, which takes the benchmark by
name and records its own hash into the result file it writes.

Both judge with the same model on the same endpoint, so a score from one is
comparable with a score from the other.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scripts.runners.common import sha256_file


CODE_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = CODE_ROOT / "scripts" / "evaluation" / "eval_full.py"
# Re-locked after the package rename src -> memory. The only change
# inside the evaluator was its import line; the scoring is untouched.
EVALUATOR_SHA256 = "17a47459994dfaf00074598dd0a7090b8bacb8a377d69ad482d04dd0b4bdf0b0"
RESULT_NAME = "eval_full.json"


def verify_evaluator(benchmark: str = "locomo") -> None:
    """Fail before spending anything if the LoCoMo evaluator has drifted."""
    if benchmark != "locomo":
        return
    if sha256_file(EVALUATOR) != EVALUATOR_SHA256:
        raise RuntimeError("locked LoCoMo evaluator hash mismatch")


def _run(command: list[str], log_path: Path, env=None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=CODE_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"evaluator failed; see {log_path}")


def run_evaluator(
    args: argparse.Namespace, output_dir: Path, questions_path: Path
) -> None:
    verify_evaluator(args.benchmark)
    log_path = output_dir / "eval.log"
    if args.benchmark == "locomo":
        _run(
            [
                sys.executable,
                str(EVALUATOR),
                str(output_dir),
                args.model,
                args.base_url,
                args.api_key,
                args.judge_api_key,
            ],
            log_path,
        )
        return
    _run(
        [
            sys.executable,
            "-m",
            "scripts.evaluation.evaluate",
            "--benchmark", args.benchmark,
            "--input", str(questions_path),
            "--output", str(output_dir / RESULT_NAME),
            "--label", output_dir.name,
            # Answers already exist; lexical overlap says little about the
            # long answers these benchmarks ask for.
            "--metrics", "judge",
        ],
        log_path,
        env={
            **os.environ,
            # Only reached if a question somehow has no answer yet; the
            # judge keeps its default model and endpoint so its verdicts stay
            # comparable with the locked LoCoMo evaluator's.
            "ANSWERER_MODEL": args.model,
            "ANSWERER_BASE": args.base_url,
            "ANSWERER_KEY": args.api_key,
            "JUDGE_KEY": args.judge_api_key,
        },
    )
