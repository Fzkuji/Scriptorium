"""CLI for re-answering frozen LongMemEval items."""

import argparse
import signal
from pathlib import Path
from typing import Any

from scripts.nativemem.common import stop_on_signal
from src import retrieval

from .execution import answer_one, install_trace_hooks, lme, run_pending
from .results import load_completed_results, source_records, write_results


def main() -> int:
    signal.signal(signal.SIGTERM, stop_on_signal)
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=lme.DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gateway-base-url", required=True)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--api-format", choices=("openai", "anthropic"), default="openai"
    )
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--visible-token-limit", type=int, default=10_000)
    parser.add_argument(
        "--verify-sources", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--condition",
        choices=("native", *retrieval.CONDITION_VIEWS),
        default="native",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_rounds < 0:
        parser.error("--max-rounds must be non-negative")
    if args.max_tool_calls < 0:
        parser.error("--max-tool-calls must be non-negative")
    if args.visible_token_limit < 0:
        parser.error("--visible-token-limit must be non-negative")

    dataset = lme.load_dataset(
        args.data.resolve(), lme.EXPECTED_LONGMEMEVAL_SIZE
    )
    lme.validate_longmemeval_s(dataset)
    sources = source_records(args.analysis.resolve())
    output_dir = args.output_dir.resolve()
    backend = retrieval.create_runtime(
        args.gateway_base_url,
        api_format=args.api_format,
        model=args.model,
        api_key=args.api_key,
    )
    query_config = retrieval.QueryConfig(
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
        visible_token_limit=args.visible_token_limit,
        max_output_tokens=args.max_tokens,
        verify_sources=args.verify_sources,
    )
    trace_state = install_trace_hooks(backend)
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
            trace_state,
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
