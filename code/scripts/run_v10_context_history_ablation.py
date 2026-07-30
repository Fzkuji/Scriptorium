#!/usr/bin/env python3
"""Run the frozen NativeMem-v10 context-history build ablation.

The default invocation is a no-request dry run. Pass ``--execute`` only after
inspecting the generated manifest. Runs are sequential and resumable at the
completed-build level; a partially written memory directory is never reused.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "src" / "adapters" / "run_nativemem.py"
DATASET = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
EVALUATOR = ROOT / "scripts" / "eval_full.py"
EVALUATOR_SHA256 = "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"


@dataclass(frozen=True)
class ContextVariant:
    name: str
    display_name: str
    context_mode: str
    context_items: int
    summary_max_words: int
    budget_unit: str
    budget: int | None


VARIANTS = {
    item.name: item
    for item in (
        ContextVariant(
            "none", "无前序历史（仅看当前 6 条消息）",
            "none", 0, 180, "none", None,
        ),
        ContextVariant(
            "raw20", "前序原始消息（最近 20 条）",
            "raw", 20, 180, "raw_turns", 20,
        ),
        ContextVariant(
            "raw50", "前序原始消息（最近 50 条）",
            "raw", 50, 180, "raw_turns", 50,
        ),
        ContextVariant(
            "events20", "前序已提炼 Event（最近 20 条）",
            "events", 20, 180, "events", 20,
        ),
        ContextVariant(
            "events50", "前序已提炼 Event（最近 50 条）",
            "events", 50, 180, "events", 50,
        ),
        ContextVariant(
            "summary100", "前序滚动聊天摘要（最多 100 words）",
            "summary", 0, 100, "summary_words", 100,
        ),
        ContextVariant(
            "summary250", "前序滚动聊天摘要（最多 250 words）",
            "summary", 0, 250, "summary_words", 250,
        ),
    )
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    base_url: str
    key_env: str | None
    fixed_key: str | None
    trust_proxy: bool
    reasoning: str | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frozen_env(variant: ContextVariant) -> dict[str, str]:
    """Return all algorithm variables that must be identical across models."""

    return {
        "NATIVEMEM_PROMPT": "v10",
        "NATIVEMEM_V10_WRITE_TURNS": "6",
        "NATIVEMEM_V10_CONTEXT_MODE": variant.context_mode,
        "NATIVEMEM_V10_CONTEXT_ITEMS": str(variant.context_items),
        "NATIVEMEM_V10_SUMMARY_MAX_WORDS": str(variant.summary_max_words),
        "NATIVEMEM_V10_TIDY_EVERY_SESSIONS": "1",
        "NATIVEMEM_V10_SESSION_TIDY_PASSES": "1",
        "NATIVEMEM_V10_FINAL_TIDY_PASSES": "1",
        "NATIVEMEM_V8_VERIFY": "on",
        "NATIVEMEM_V8_SECTIONS": "on",
        "NATIVEMEM_V8_ARTICLE": "off",
        "NATIVEMEM_V8_CONCURRENCY": "1",
        "NATIVEMEM_FAIL_FAST": "1",
        "NATIVEMEM_OPENAI_MAX_RETRIES": "0",
        "NATIVEMEM_HTTP_TIMEOUT": "240",
        "PYTHONUNBUFFERED": "1",
    }


def model_specs(args: argparse.Namespace) -> dict[str, ModelSpec]:
    return {
        "gpt55": ModelSpec(
            name="gpt55",
            model=args.gpt_model,
            base_url=args.gpt_base,
            key_env=None,
            fixed_key="subscription-proxy",
            trust_proxy=False,
            reasoning="none",
        ),
        "minimax": ModelSpec(
            name="minimax",
            model=args.minimax_model,
            base_url=args.minimax_base,
            key_env=args.minimax_key_env,
            fixed_key=None,
            trust_proxy=args.minimax_trust_proxy,
            reasoning=None,
        ),
        "deepseek_flash": ModelSpec(
            name="deepseek_flash",
            model=args.deepseek_model,
            base_url=args.deepseek_base,
            key_env=args.deepseek_key_env,
            fixed_key=None,
            trust_proxy=args.deepseek_trust_proxy,
            reasoning="none",
        ),
    }


def _read_json_no_proxy(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def preflight(specs: list[ModelSpec]) -> dict:
    result: dict[str, object] = {}
    for spec in specs:
        if spec.name == "gpt55":
            health_url = spec.base_url.removesuffix("/v1") + "/healthz"
            health = _read_json_no_proxy(health_url)
            if health.get("status") != "ok" or not health.get("auth_readable"):
                raise RuntimeError(f"GPT subscription proxy is not healthy: {health}")
            if health.get("requested_reasoning_effort") != "none":
                raise RuntimeError("GPT subscription proxy is not thinking=off")
            result[spec.name] = {
                "health_url": health_url,
                "status": health.get("status"),
                "requested_reasoning_effort": health.get(
                    "requested_reasoning_effort"
                ),
                "proxy_code_sha256": health.get("code_sha256"),
            }
        else:
            key = os.environ.get(spec.key_env or "", "")
            if not key:
                raise RuntimeError(
                    f"{spec.key_env} is required to execute {spec.name} runs"
                )
            result[spec.name] = {
                "base_url": spec.base_url,
                "credential_present": True,
            }
    return result


def build_command(
    *, output: Path, sample: int, max_sessions: int | None
) -> list[str]:
    command = [
        sys.executable,
        str(ADAPTER),
        "--sample",
        str(sample),
        "--output",
        str(output),
        "--build-only",
    ]
    if max_sessions is not None:
        command.extend(["--max-sessions", str(max_sessions)])
    return command


def _valid_completed_build(path: Path, expected_model: str) -> bool:
    if not path.is_file():
        return False
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(records, list)
        and len(records) == 1
        and records[0].get("question_id") == "_build_stats"
        and records[0].get("method") == "NativeMem-v10"
        and records[0].get("builder_model") == expected_model
    )


def _scope(args: argparse.Namespace) -> tuple[list[int], int | None]:
    if args.samples:
        samples = args.samples
    elif args.scope == "smoke":
        samples = [1]
    else:
        samples = [0, 1]
    if args.max_sessions is not None:
        max_sessions = None if args.max_sessions == 0 else args.max_sessions
    else:
        max_sessions = 1 if args.scope == "smoke" else None
    return samples, max_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--samples", type=int, nargs="+")
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="override scope; 0 means all sessions",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("gpt55", "minimax", "deepseek_flash"),
        default=["gpt55", "minimax"],
    )
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--gpt-model", default="gpt-5.5")
    parser.add_argument("--gpt-base", default="http://127.0.0.1:8199/v1")
    parser.add_argument("--minimax-model", default="minimax/minimax-m2.7")
    parser.add_argument("--minimax-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--minimax-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--minimax-trust-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deepseek-model", default="deepseek/deepseek-v4-flash"
    )
    parser.add_argument(
        "--deepseek-base", default="https://openrouter.ai/api/v1"
    )
    parser.add_argument("--deepseek-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--deepseek-trust-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    samples, max_sessions = _scope(args)
    specs_by_name = model_specs(args)
    specs = [specs_by_name[name] for name in args.models]
    variants = [VARIANTS[name] for name in args.variants]

    evaluator_hash = sha256(EVALUATOR)
    if evaluator_hash != EVALUATOR_SHA256:
        raise RuntimeError(
            f"locked evaluator hash mismatch: {evaluator_hash} != {EVALUATOR_SHA256}"
        )

    expected_runs = [
        {
            "model": spec.name,
            "builder_model": spec.model,
            "variant": variant.name,
            "sample": sample,
            "max_sessions": max_sessions,
        }
        for spec in specs
        for variant in variants
        for sample in samples
    ]
    manifest = {
        "schema_version": 1,
        "study": "NativeMem-v10-context-history",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "scope": args.scope,
        "samples": samples,
        "max_sessions": max_sessions,
        "sequential": True,
        "source_hashes": {
            "dataset": sha256(DATASET),
            "adapter": sha256(ADAPTER),
            "v10_memory": sha256(ROOT / "src" / "v10_memory.py"),
            "evaluator": evaluator_hash,
        },
        "locked_evaluator": str(EVALUATOR),
        "variants": [asdict(variant) for variant in variants],
        "models": [
            {
                "name": spec.name,
                "model": spec.model,
                "base_url": spec.base_url,
                "credential_env": spec.key_env,
                "reasoning": spec.reasoning,
                "trust_proxy": spec.trust_proxy,
            }
            for spec in specs
        ],
        "fixed_env": {
            key: value
            for key, value in frozen_env(VARIANTS["none"]).items()
            if key
            not in {
                "NATIVEMEM_V10_CONTEXT_MODE",
                "NATIVEMEM_V10_CONTEXT_ITEMS",
                "NATIVEMEM_V10_SUMMARY_MAX_WORDS",
            }
        },
        "expected_runs": expected_runs,
    }
    atomic_json(results_dir / "experiment_manifest.json", manifest)

    if not args.execute:
        atomic_json(
            results_dir / "status.json",
            {
                "phase": "dry_run",
                "completed": 0,
                "total": len(expected_runs),
                "failed": [],
            },
        )
        print(
            f"dry-run manifest: {len(expected_runs)} runs -> "
            f"{results_dir / 'experiment_manifest.json'}"
        )
        return

    manifest["preflight"] = preflight(specs)
    manifest["execute"] = True
    atomic_json(results_dir / "experiment_manifest.json", manifest)

    completed = 0
    failed: list[dict[str, object]] = []
    total = len(expected_runs)
    for spec in specs:
        for variant in variants:
            for sample in samples:
                run_dir = (
                    results_dir
                    / "runs"
                    / spec.name
                    / variant.name
                    / f"sample{sample}-max{max_sessions or 'all'}"
                )
                output = run_dir / "build.json"
                log_path = run_dir / "run.log"
                status_path = run_dir / "status.json"
                if _valid_completed_build(output, spec.model):
                    completed += 1
                    atomic_json(
                        status_path,
                        {"phase": "complete", "resumed": True, "output": str(output)},
                    )
                    continue
                memory_dir = run_dir / f"memory_sample{sample}"
                if memory_dir.exists():
                    reason = (
                        "partial memory directory exists without a valid build; "
                        "refusing to append to it"
                    )
                    failed.append(
                        {
                            "model": spec.name,
                            "variant": variant.name,
                            "sample": sample,
                            "reason": reason,
                        }
                    )
                    atomic_json(status_path, {"phase": "failed", "reason": reason})
                    if not args.continue_on_error:
                        atomic_json(
                            results_dir / "status.json",
                            {
                                "phase": "failed",
                                "completed": completed,
                                "total": total,
                                "failed": failed,
                            },
                        )
                        raise RuntimeError(reason)
                    continue

                env = os.environ.copy()
                env.update(frozen_env(variant))
                env.update(
                    {
                        "BUILDER_BASE": spec.base_url,
                        "BUILDER_MODEL": spec.model,
                        "BUILDER_KEY": (
                            spec.fixed_key
                            if spec.fixed_key is not None
                            else os.environ.get(spec.key_env or "", "")
                        ),
                        "NATIVEMEM_TRUST_PROXY": "1" if spec.trust_proxy else "0",
                    }
                )
                command = build_command(
                    output=output, sample=sample, max_sessions=max_sessions
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    status_path,
                    {
                        "phase": "running",
                        "model": spec.name,
                        "builder_model": spec.model,
                        "variant": asdict(variant),
                        "sample": sample,
                        "max_sessions": max_sessions,
                        "command": command,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                atomic_json(
                    results_dir / "status.json",
                    {
                        "phase": "running",
                        "current": {
                            "model": spec.name,
                            "variant": variant.name,
                            "sample": sample,
                        },
                        "completed": completed,
                        "total": total,
                        "failed": failed,
                    },
                )
                print(
                    f"[{completed + 1}/{total}] {spec.name} "
                    f"{variant.name} sample={sample} max_sessions={max_sessions}"
                )
                with log_path.open("a", encoding="utf-8") as log_handle:
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode != 0 or not _valid_completed_build(
                    output, spec.model
                ):
                    reason = f"adapter exit={process.returncode}; see {log_path}"
                    failed.append(
                        {
                            "model": spec.name,
                            "variant": variant.name,
                            "sample": sample,
                            "reason": reason,
                        }
                    )
                    atomic_json(status_path, {"phase": "failed", "reason": reason})
                    if not args.continue_on_error:
                        atomic_json(
                            results_dir / "status.json",
                            {
                                "phase": "failed",
                                "completed": completed,
                                "total": total,
                                "failed": failed,
                            },
                        )
                        raise RuntimeError(reason)
                    continue
                completed += 1
                atomic_json(
                    status_path,
                    {
                        "phase": "complete",
                        "output": str(output),
                        "log": str(log_path),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

    atomic_json(
        results_dir / "status.json",
        {
            "phase": "complete" if not failed else "complete_with_failures",
            "completed": completed,
            "total": total,
            "failed": failed,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"completed={completed}/{total}, failed={len(failed)}")


if __name__ == "__main__":
    main()
