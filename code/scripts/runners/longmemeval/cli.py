"""CLI for re-answering frozen LongMemEval items."""

import argparse
import signal
from pathlib import Path
from typing import Any

from scripts.runners.common import run_config, stop_on_signal
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
    parser.add_argument(
        "--search-tools", choices=("split", "fused"), default="split"
    )
    parser.add_argument(
        "--evidence-ledger", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--evidence-ledger-max-entries", type=int, default=24)
    parser.add_argument(
        "--reasoning-ledger", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--retrieval-plan", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--pipeline", action=argparse.BooleanOptionalAction, default=False,
        help="Use the deterministic P0 retrieval pipeline (default: disabled).",
    )
    parser.add_argument(
        "--pipeline-version", choices=("p0-v2", "p0-v3", "p0b", "p0b-r1", "m3"), default="p0-v2",
    )
    parser.add_argument(
        "--pipeline-evidence-packet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Present unchanged Pipeline candidates as a deterministic P1 packet.",
    )
    parser.add_argument(
        "--pipeline-evidence-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the deterministic P2 sufficiency gate to the P1 packet.",
    )
    parser.add_argument(
        "--pipeline-supplement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow one bounded deterministic P3 supplement on a code-detected gap.",
    )
    parser.add_argument(
        "--reflective-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use H-Lite-U general first recall followed by adaptive "
            "evidence-sufficiency reflection and ordinary retrieval tools."
        ),
    )
    parser.add_argument(
        "--adaptive-workspace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use A0-R mutable workspace while preserving free agent retrieval.",
    )
    parser.add_argument(
        "--claim-evidence-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use A0-C mutable answer-level claim/evidence state while preserving "
            "the original agent-controlled retrieval tools and view selection."
        ),
    )
    parser.add_argument(
        "--claim-evidence-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the C4 event-driven claim/evidence loop while A0 retains "
            "retrieval and stop control."
        ),
    )
    parser.add_argument(
        "--claim-evidence-loop-initial-batch-size",
        type=int,
        default=2,
        help="Novel evidence batches required for the first organizer checkpoint.",
    )
    parser.add_argument(
        "--claim-evidence-loop-initial-organizer-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the first evidence-scale organizer checkpoint. Disable for "
            "the C5-A final-only organizer ablation."
        ),
    )
    parser.add_argument(
        "--claim-evidence-loop-event-guidance-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable C5-B semantic-event guidance and one nonbinding reminder "
            "after three pending novel evidence batches."
        ),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--condition",
        choices=("native", *retrieval.CONDITION_VIEWS),
        default="native",
    )
    parser.add_argument(
        "--memory-components",
        help=(
            "Comma-separated retrieval-time component mask drawn from: "
            + ",".join(retrieval.MEMORY_COMPONENTS)
            + ". Reuses frozen full memory and requires --condition native."
        ),
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
        search_tools=args.search_tools,
        memory_components=args.memory_components,
        evidence_ledger_enabled=args.evidence_ledger,
        evidence_ledger_max_entries=args.evidence_ledger_max_entries,
        reasoning_ledger_enabled=args.reasoning_ledger,
        retrieval_plan_enabled=args.retrieval_plan,
        pipeline_enabled=args.pipeline,
        pipeline_version=args.pipeline_version,
        pipeline_evidence_packet_enabled=args.pipeline_evidence_packet,
        pipeline_evidence_gate_enabled=args.pipeline_evidence_gate,
        pipeline_supplement_enabled=args.pipeline_supplement,
        reflective_retrieval_enabled=args.reflective_retrieval,
        adaptive_workspace_enabled=args.adaptive_workspace,
        claim_evidence_state_enabled=args.claim_evidence_state,
        claim_evidence_loop_enabled=args.claim_evidence_loop,
        claim_evidence_loop_initial_batch_size=(
            args.claim_evidence_loop_initial_batch_size
        ),
        claim_evidence_loop_initial_organizer_enabled=(
            args.claim_evidence_loop_initial_organizer_enabled
        ),
        claim_evidence_loop_event_guidance_enabled=(
            args.claim_evidence_loop_event_guidance_enabled
        ),
    )
    if query_config.memory_components is not None and args.condition != "native":
        parser.error("--memory-components requires --condition native")
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
