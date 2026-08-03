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
    parser.add_argument("--claude-cli")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument("--input-usd-per-million", type=float, required=True)
    parser.add_argument("--output-usd-per-million", type=float, required=True)
    parser.add_argument("--session-batch", type=int, default=5)
    parser.add_argument("--writer-calibration", type=Path)
    parser.add_argument("--writer-input-token-cap", type=int)
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
    parser.add_argument(
        "--verify-sources", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evaluate", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    if args.max_budget_usd is not None and args.max_budget_usd <= 0:
        parser.error("--max-budget-usd must be positive")
    if args.writer_input_token_cap is not None and args.writer_input_token_cap < 1:
        parser.error("--writer-input-token-cap must be positive")
    if args.input_usd_per_million < 0 or args.output_usd_per_million < 0:
        parser.error("model prices must be non-negative")
    return args
