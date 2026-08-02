"""Command-line configuration for one complete LoCoMo conversation."""

import argparse
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = CODE_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate one LoCoMo conversation with NativeMem."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample-id", default="conv-50")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--judge-api-key", required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--api-format", choices=("openai", "anthropic"), default="openai"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--input-usd-per-million", type=float, required=True)
    parser.add_argument("--output-usd-per-million", type=float, required=True)
    parser.add_argument("--session-batch", type=int, default=5)
    parser.add_argument("--writer-calibration", type=Path)
    parser.add_argument("--local-reorg-every-sessions", type=int, default=5)
    parser.add_argument("--verify-every-sessions", type=int, default=5)
    parser.add_argument(
        "--verify-writes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--final-manage", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--recent-limit", type=int, default=50)
    parser.add_argument("--core-max-tokens", type=int, default=2_000)
    parser.add_argument("--agent-max-rounds", type=int, default=12)
    parser.add_argument("--manager-max-rounds", type=int, default=8)
    parser.add_argument("--thinking", choices=("enabled", "disabled"))
    parser.add_argument("--retry-log", action="store_true")
    parser.add_argument("--retrieval-max-rounds", type=int, default=8)
    parser.add_argument("--retrieval-max-tool-calls", type=int, default=5)
    parser.add_argument("--memory-visible-tokens", type=int, default=10_000)
    parser.add_argument("--answer-max-tokens", type=int, default=1_200)
    parser.add_argument(
        "--verify-sources", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evaluate", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.input_usd_per_million < 0 or args.output_usd_per_million < 0:
        parser.error("model prices must be non-negative")
    return args
