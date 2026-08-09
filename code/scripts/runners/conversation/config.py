"""Command-line configuration for one complete conversation."""

import argparse
from pathlib import Path

from ..common import run_config

CODE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = CODE_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"

# Config values naming a file resolve against the config file's directory.
_PATH_KEYS = ("data", "output_dir", "writer_calibration", "claude_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate one LoCoMo conversation. Pass --config to "
            "supply defaults from a JSON file; explicit flags override it."
        )
    )
    run_config.add_config_flag(parser)
    parser.add_argument(
        "--benchmark",
        choices=("locomo", "beam"),
        default="locomo",
        help=(
            "which questions the sample carries, and therefore which "
            "evaluator scores it"
        ),
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
    # Cache reads are billed far below fresh input; omit to price them
    # at the input rate, which overstates cost on cache-heavy runs.
    parser.add_argument("--cache-read-usd-per-million", type=float)
    parser.add_argument("--cache-write-usd-per-million", type=float)
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
    parser.add_argument("--core-max-tokens", type=int, default=3_000)
    parser.add_argument("--core-repair-target-tokens", type=int, default=2_700)
    parser.add_argument("--core-repair-max-checks", type=int, default=8)
    parser.add_argument("--core-repair-max-trajectories", type=int, default=2)
    parser.add_argument("--core-repair-stagnation-limit", type=int, default=2)
    parser.add_argument(
        "--shell-backend",
        choices=("auto", "posix-bash", "native"),
        default="auto",
        help=(
            "Writer command backend: use posix-bash for WSL experiments "
            "whose memory workspace is on Linux ext4"
        ),
    )
    parser.add_argument(
        "--verify-sources", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evaluate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--build-only", action="store_true")
    try:
        run_config.apply(parser, argv, path_keys=_PATH_KEYS)
    except run_config.ConfigError as exc:
        parser.error(str(exc))
    args = parser.parse_args(argv)
    # A config file sets defaults, which argparse never validates.
    if args.benchmark not in ("locomo", "beam"):
        parser.error(f"--benchmark must be locomo or beam, got {args.benchmark!r}")
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
