"""CLI for re-answering frozen LongMemEval items."""

import argparse
import signal
from pathlib import Path
from typing import Any

from scripts.nativemem.common import run_config, stop_on_signal
from src import retrieval

from .execution import answer_one, lme, run_pending
from .results import load_completed_results, source_records, write_results

# Config values naming a file resolve against the config file's directory.
_PATH_KEYS = ("analysis", "data", "output_dir", "claude_cli")


def main() -> int:
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser(
        description=(
            "Re-answer frozen LongMemEval items. Pass --config to supply "
            "defaults from a JSON file; explicit flags override it."
        )
    )
    run_config.add_config_flag(parser)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=lme.DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gateway-base-url", required=True)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--claude-cli")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument(
        "--verify-sources", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--condition",
        choices=("native", *retrieval.CONDITION_VIEWS),
        default="native",
    )
    try:
        run_config.apply(parser, None, path_keys=_PATH_KEYS)
    except run_config.ConfigError as exc:
        parser.error(str(exc))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    if args.max_budget_usd is not None and args.max_budget_usd <= 0:
        parser.error("--max-budget-usd must be positive")

    dataset = lme.load_dataset(
        args.data.resolve(), lme.EXPECTED_LONGMEMEVAL_SIZE
    )
    lme.validate_longmemeval_s(dataset)
    sources = source_records(args.analysis.resolve())
    output_dir = args.output_dir.resolve()
    backend = retrieval.create_runtime(
        args.gateway_base_url,
        model=args.model,
        api_key=args.api_key,
        cli_path=args.claude_cli,
    )
    query_config = retrieval.QueryConfig(
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        verify_sources=args.verify_sources,
    )
    completed = load_completed_results(output_dir, dataset, sources)
    write_results(output_dir, completed)
    pending = [
        source
        for source in sources
        if int(source["dataset_index"]) not in completed
    ]
    print(
        f"resumed={len(completed)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    def work(source: dict[str, Any]) -> dict[str, Any]:
        return answer_one(
            backend,
            dataset,
            source,
            output_dir,
            args.condition,
            model=args.model,
            query_config=query_config,
        )

    def save_completion(record: dict[str, Any]) -> None:
        index = int(record["dataset_index"])
        completed[index] = record
        write_results(output_dir, completed)
        print(
            f"complete={len(completed)}/{len(sources)} item={index} "
            f"answer={record['answer']!r}",
            flush=True,
        )

    _, interrupted = run_pending(
        pending,
        workers=args.workers,
        work=work,
        on_complete=save_completion,
    )
    if interrupted:
        print(
            f"interrupted; saved={len(completed)}/{len(sources)}", flush=True
        )
        return 130
    if len(completed) != len(sources):
        raise RuntimeError(
            f"only completed {len(completed)}/{len(sources)} items"
        )
    return 0
