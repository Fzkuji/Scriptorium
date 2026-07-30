#!/usr/bin/env python3
"""Execute the frozen GPT-5.6 Writer-window build matrix.

The default invocation is read-only and prints the selected runs.  Real model
requests require both ``--execute`` and ``--allow-model-requests``.  Every run
uses a separate worker process so model/environment state cannot leak between
configurations.  This stage builds memory only; fixed-model answering and the
benchmark-specific judges are separate stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "experiments" / "gpt56-chunk-curve" / "run_matrix.jsonl"
DEFAULT_RESULTS = ROOT / "results" / "formal" / "gpt56-chunk-curve"
LOCOMO_DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
LONGMEMEVAL_DATA = (
    ROOT / "benchmarks" / "longmemeval" / "data" /
    "longmemeval_s_cleaned.json"
)
BEAM_ARROW = (
    ROOT / "benchmarks" / "beam" / "hf_cache" / "Mohammadta___beam" /
    "default" / "0.0.0" / "3205395e897e7318c7b094ef4e6047b9b82dbb03" /
    "beam-100K.arrow"
)
LOCOMO_EVALUATOR = ROOT / "scripts" / "eval_full.py"
LOCKED_LOCOMO_EVALUATOR_SHA256 = (
    "f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd"
)
FRONTIER_BASE_URL = "https://api.frontier-intelligence.tech/v1"
PROVIDER_CHOICES = ("subscription-proxy", "frontier")

MODEL_ORDER = {"luna": 0, "terra": 1, "sol": 2}
BENCHMARK_ORDER = {"beam-100k": 0, "longmemeval-s": 1, "locomo": 2}
WINDOW_ORDER = {
    "session": 0, "32": 1, "24": 2, "20": 3, "16": 4,
    "12": 5, "8": 6, "6": 7, "4": 8,
}


class RunError(RuntimeError):
    """A frozen build cannot be executed or validated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_hashes(matrix_path: Path) -> dict[str, str]:
    """Hash every local source that can change a Writer-window build."""
    paths = {
        "matrix": matrix_path,
        "runner": Path(__file__).resolve(),
        "adapter": ROOT / "src" / "adapters" / "run_nativemem.py",
        "runtime": ROOT / "src" / "nativemem.py",
        "v10_memory": ROOT / "src" / "v10_memory.py",
        "v8_memory": ROOT / "src" / "v8_memory.py",
        "longmemeval_converter": ROOT / "scripts" / "run_v88_gpt55_longmemeval.py",
        "beam_converter": ROOT / "scripts" / "run_v88_gpt55_beam.py",
        "locomo_evaluator": LOCOMO_EVALUATOR,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_matrix(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if path.resolve() == DEFAULT_MATRIX.resolve() and len(rows) != 162:
        raise RunError(f"expected frozen 162-row matrix, found {len(rows)}")
    if not rows:
        raise RunError("matrix is empty")
    run_ids = [row.get("run_id") for row in rows]
    if None in run_ids or len(set(run_ids)) != len(run_ids):
        raise RunError("matrix contains missing or duplicate run IDs")
    if any(row.get("reasoning_effort") != "none" for row in rows):
        raise RunError("every matrix row must freeze reasoning_effort=none")
    return rows


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    wanted_windows = {str(value) for value in args.windows} if args.windows else None
    wanted_ids = set(args.run_ids or [])
    for row in rows:
        if args.models and row["tier"] not in args.models:
            continue
        if args.benchmarks and row["benchmark"] not in args.benchmarks:
            continue
        if wanted_windows and str(row["write_turns"]) not in wanted_windows:
            continue
        if wanted_ids and row["run_id"] not in wanted_ids:
            continue
        selected.append(row)
    selected.sort(key=lambda row: (
        MODEL_ORDER[row["tier"]],
        WINDOW_ORDER[str(row["write_turns"])],
        BENCHMARK_ORDER[row["benchmark"]],
        row["unit_id"],
    ))
    if args.max_runs is not None:
        selected = selected[:args.max_runs]
    return selected


def direct_health(base_url: str) -> dict[str, Any]:
    health_url = base_url.removesuffix("/v1") + "/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(health_url, timeout=10) as response:
        health = json.loads(response.read().decode("utf-8"))
    if health.get("status") != "ok" or not health.get("auth_readable"):
        raise RunError(f"subscription proxy is not healthy: {health}")
    if health.get("requested_reasoning_effort") != "none":
        raise RunError("subscription proxy is not frozen at reasoning=none")
    return health


def canonical_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def frontier_models(base_url: str, api_key: str) -> set[str]:
    """Read model IDs without exposing the provider credential."""
    import httpx

    response = httpx.get(
        canonical_base_url(base_url) + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
        follow_redirects=False,
        trust_env=True,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RunError("frontier /models response has no data list")
    return {
        str(row["id"])
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def provider_preflight(
    provider: str,
    base_url: str,
    required_models: set[str],
) -> dict[str, Any]:
    """Validate authentication, model availability, and reasoning policy."""
    if provider == "subscription-proxy":
        health = direct_health(base_url)
        return {
            "name": provider,
            "base_url": canonical_base_url(base_url),
            "auth_source": "codex-auth-via-local-proxy",
            "requested_reasoning_effort": "none",
            "code_sha256": health.get("code_sha256"),
        }
    if provider != "frontier":
        raise RunError(f"unsupported provider: {provider}")
    if canonical_base_url(base_url) != FRONTIER_BASE_URL:
        raise RunError(
            "frontier provider requires the frozen endpoint " + FRONTIER_BASE_URL
        )
    api_key = os.environ.get("FRONTIER_API_KEY")
    if not api_key:
        raise RunError("FRONTIER_API_KEY is not set")
    available = frontier_models(base_url, api_key)
    missing = sorted(required_models - available)
    if missing:
        raise RunError("frontier provider is missing models: " + ", ".join(missing))
    return {
        "name": provider,
        "base_url": FRONTIER_BASE_URL,
        "auth_source": "FRONTIER_API_KEY",
        "requested_reasoning_effort": "none",
        "available_required_models": sorted(required_models),
    }


def runtime_environment(
    row: dict[str, Any],
    base_url: str,
    provider: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return process environment and a credential-free manifest copy."""
    secret = (
        "subscription-proxy"
        if provider == "subscription-proxy"
        else os.environ.get("FRONTIER_API_KEY", "")
    )
    if not secret:
        raise RunError("frontier credential is unavailable")
    recorded = dict(row["environment"])
    recorded.update({
        "BUILDER_BASE": canonical_base_url(base_url),
        "BUILDER_MODEL": row["model"],
        "BUILDER_KEY_SOURCE": (
            "local-proxy-placeholder"
            if provider == "subscription-proxy"
            else "FRONTIER_API_KEY"
        ),
        "NATIVEMEM_TRUST_PROXY": "0" if provider == "subscription-proxy" else "1",
        "NATIVEMEM_V8_CONCURRENCY": "1",
        "NATIVEMEM_FAIL_FAST": "1",
        "NATIVEMEM_OPENAI_MAX_RETRIES": (
            "6" if provider == "frontier" else "2"
        ),
        "NATIVEMEM_HTTP_TIMEOUT": os.environ.get(
            "NATIVEMEM_HTTP_TIMEOUT", "240"
        ),
        "NATIVEMEM_REASONING_EFFORT": "none",
    })
    process_env = dict(recorded)
    process_env["BUILDER_KEY"] = secret
    return process_env, recorded


def run_dir(results_dir: Path, row: dict[str, Any]) -> Path:
    return results_dir / Path(row["output_dir"]).relative_to(
        "results/formal/gpt56-chunk-curve"
    )


def valid_complete(path: Path, row: dict[str, Any]) -> bool:
    build_path = path / "build.json"
    marker = path / "memory" / "_SUCCESS.json"
    if not build_path.is_file() or not marker.is_file():
        return False
    try:
        build = read_json(build_path)
        success = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        build.get("status") == "complete"
        and build.get("run_id") == row["run_id"]
        and build.get("builder_model") == row["model"]
        and str(build.get("write_turns")) == str(row["write_turns"])
        and int(build.get("session_group_size", 1))
        == int(row.get("session_group_size", 1))
        and success.get("run_id") == row["run_id"]
    )


def source_message_count(conv: dict[str, Any]) -> int:
    count = 0
    number = 1
    while f"session_{number}" in conv:
        count += len(conv[f"session_{number}"])
        number += 1
    return count


def source_token_count(conv: dict[str, Any]) -> tuple[int, str]:
    texts = []
    number = 1
    while f"session_{number}" in conv:
        texts.extend(
            str(turn.get("text", ""))
            for turn in conv[f"session_{number}"]
            if isinstance(turn, dict)
        )
        number += 1
    joined = "\n".join(texts)
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(joined)), "o200k_base"
    except Exception:  # noqa: BLE001
        return max(1, len(joined) // 4), "chars_div_4_fallback"


def load_unit(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selection = row["selection"]
    benchmark = row["benchmark"]
    if benchmark == "locomo":
        item = read_json(LOCOMO_DATA)[selection["sample_index"]]
        if item["sample_id"] != selection["sample_id"]:
            raise RunError("LoCoMo unit identity changed")
        questions = [
            {
                "question_id": f"q{index}",
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []),
            }
            for index, qa in enumerate(item["qa"])
            if qa.get("category") in (1, 2, 3, 4)
        ]
        return item["conversation"], questions, {"sample_id": item["sample_id"]}

    if benchmark == "longmemeval-s":
        from scripts.run_v88_gpt55_longmemeval import longmemeval_to_locomo

        index = selection["dataset_index"]
        item = read_json(LONGMEMEVAL_DATA)[index]
        if item["question_id"] != selection["question_id"]:
            raise RunError("LongMemEval unit identity changed")
        question = {
            "question_id": item["question_id"],
            "question": item["question"],
            "question_date": item["question_date"],
            "gold": item["answer"],
            "question_type": item["question_type"],
            "answer_session_ids": item["answer_session_ids"],
        }
        return longmemeval_to_locomo(item, index), [question], {
            "dataset_index": index,
        }

    if benchmark == "beam-100k":
        from scripts.gpt56_chunk_curve_qa_contract import load_arrow_row
        from scripts.run_v88_gpt55_beam import conversation_to_native, extract_questions

        index = selection["conversation_index"]
        item = load_arrow_row(BEAM_ARROW, index)
        if str(item["conversation_id"]) != selection["conversation_id"]:
            raise RunError("BEAM unit identity changed")
        conv, metadata = conversation_to_native(item)
        questions = extract_questions(item)
        for question_index, question in enumerate(questions):
            question["question_id"] = f"{item['conversation_id']}-q{question_index}"
        return conv, questions, {
            "conversation_id": str(item["conversation_id"]),
            "source_id_count": len(metadata["source_id_map"]),
        }

    raise RunError(f"unsupported benchmark: {benchmark}")


def summarize_trace(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if path.is_file() else []
    first_refs = {
        dia_id
        for record in records
        for event in record.get("first_pass_events", [])
        for dia_id in event.get("dia_ids", [])
    }
    post_refs = {
        dia_id
        for record in records
        for event in record.get("post_verify_events", [])
        for dia_id in event.get("dia_ids", [])
    }
    return {
        "writer_chunks": len(records),
        "chunks_triggering_verify": sum(
            bool(record.get("missing_coverage_points")) for record in records
        ),
        "first_pass_events": sum(
            len(record.get("first_pass_events", [])) for record in records
        ),
        "verify_added_events": sum(
            len(record.get("verify_events", [])) for record in records
        ),
        "post_verify_events": sum(
            len(record.get("post_verify_events", [])) for record in records
        ),
        "first_pass_unique_dia_ids": len(first_refs),
        "post_verify_unique_dia_ids": len(post_refs),
    }


def phase_usage(call_log: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    phases: dict[str, dict[str, int]] = {}
    for entry in call_log:
        phase = str(entry.get("phase", "unknown"))
        totals = phases.setdefault(phase, {
            "calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        })
        totals["calls"] += 1
        totals["input_tokens"] += int(entry.get("prompt_tokens", 0) or 0)
        totals["cached_input_tokens"] += int(entry.get("cached_tokens", 0) or 0)
        totals["output_tokens"] += int(entry.get("completion_tokens", 0) or 0)
        totals["reasoning_tokens"] += int(entry.get("reasoning_tokens", 0) or 0)
    return phases


def execute_worker(
    row: dict[str, Any],
    results_dir: Path,
    base_url: str,
    matrix_path: Path,
    provider: str,
) -> int:
    if sha256_file(LOCOMO_EVALUATOR) != LOCKED_LOCOMO_EVALUATOR_SHA256:
        raise RunError("locked LoCoMo evaluator hash mismatch")
    provider_contract = provider_preflight(provider, base_url, {row["model"]})
    target = run_dir(results_dir, row)
    status_path = target / "status.json"
    if valid_complete(target, row):
        atomic_json(status_path, {
            "phase": "complete", "resumed": True, "run_id": row["run_id"],
        })
        return 0
    allow_reasoning_tokens = (
        row.get("environment", {}).get("NATIVEMEM_ALLOW_REASONING_TOKENS") == "1"
    )
    failed_call_log = target / "call_log.failed.json"
    failed_status = read_json(status_path) if status_path.is_file() else {}
    failed_memory_value = failed_status.get("failed_memory")
    failed_memory = Path(str(failed_memory_value)) if failed_memory_value else None
    if (
        allow_reasoning_tokens
        and str(failed_status.get("error", "")).startswith(
            "RunError: reasoning tokens must be zero"
        )
        and failed_memory is not None
        and failed_memory.is_dir()
        and failed_call_log.is_file()
        and not (target / "memory").exists()
    ):
        call_log = read_json(failed_call_log)
        actual_models = {
            str(entry.get("response_model") or "") for entry in call_log
        }
        reasoning_tokens = sum(
            int(entry.get("reasoning_tokens", 0) or 0) for entry in call_log
        )
        if actual_models != {row["model"]} or reasoning_tokens <= 0:
            raise RunError("reasoning-failure recovery evidence is invalid")
        conv, _, _ = load_unit(row)
        source_tokens, tokenizer = source_token_count(conv)
        phases = phase_usage(call_log)
        totals = {
            name: sum(values[name] for values in phases.values())
            for name in (
                "calls", "input_tokens", "cached_input_tokens",
                "output_tokens", "reasoning_tokens",
            )
        }
        success = {
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
            "session_group_size": row.get("session_group_size", 1),
            "completed_at": utc_now(),
            "recovered_from_reasoning_token_policy": True,
        }
        atomic_json(failed_memory / "_SUCCESS.json", success)
        os.replace(failed_memory, target / "memory")
        build_record = {
            "schema_version": 1,
            "status": "complete",
            "run_id": row["run_id"],
            "benchmark": row["benchmark"],
            "unit_id": row["unit_id"],
            "builder_model": row["model"],
            "tier": row["tier"],
            "reasoning_effort": row["reasoning_effort"],
            "reasoning_tokens_allowed": True,
            "write_turns": row["write_turns"],
            "session_group_size": row.get("session_group_size", 1),
            "environment": row["environment"],
            "messages": source_message_count(conv),
            "source_tokens": source_tokens,
            "source_tokenizer": tokenizer,
            "event_count": None,
            "reported_build_time_s": None,
            "wall_time_s": None,
            "tracker_totals": None,
            "usage": totals,
            "phase_usage": phases,
            "distill": summarize_trace(target / "distill_trace.jsonl"),
            "response_ids": [entry.get("response_id") for entry in call_log],
            "actual_models": sorted(actual_models),
            "provider": provider_contract,
            "source_hashes": runtime_source_hashes(matrix_path),
            "memory_dir": str((target / "memory").resolve()),
            "recovered_from_reasoning_token_policy": True,
            "finished_at": utc_now(),
        }
        atomic_json(target / "call_log.json", call_log)
        atomic_json(target / "build.json", build_record)
        atomic_json(status_path, {
            "phase": "complete",
            "run_id": row["run_id"],
            "build": str((target / "build.json").resolve()),
            "recovered_from_reasoning_token_policy": True,
            "finished_at": utc_now(),
        })
        return 0
    if (target / "memory").exists():
        raise RunError("memory directory exists without a valid completion marker")
    unresolved_staging = sorted(target.glob("memory.staging.*"))
    if unresolved_staging:
        raise RunError(
            "unresolved staging directory exists: "
            + ", ".join(str(path) for path in unresolved_staging)
        )

    target.mkdir(parents=True, exist_ok=True)
    trace_path = target / "distill_trace.jsonl"
    staging = target / f"memory.staging.{os.getpid()}"
    if staging.exists():
        raise RunError(f"staging directory already exists: {staging}")

    process_env, recorded_env = runtime_environment(row, base_url, provider)
    process_env["NATIVEMEM_DISTILL_TRACE"] = str(trace_path)
    recorded_env["NATIVEMEM_DISTILL_TRACE"] = str(trace_path)
    os.environ.update(process_env)

    conv, questions, unit_meta = load_unit(row)
    source_tokens, tokenizer = source_token_count(conv)
    atomic_json(target / "unit.json", {
        "run_id": row["run_id"],
        "benchmark": row["benchmark"],
        "unit_id": row["unit_id"],
        "selection": row["selection"],
        "metadata": unit_meta,
        "messages": source_message_count(conv),
        "source_tokens": source_tokens,
        "source_tokenizer": tokenizer,
        "questions": questions,
    })
    atomic_json(status_path, {
        "phase": "running",
        "run_id": row["run_id"],
        "builder_model": row["model"],
        "write_turns": row["write_turns"],
        "started_at": utc_now(),
    })

    from src.adapters import run_nativemem as native
    import src.nativemem as runtime

    runtime.CALL_LOG.clear()
    runtime.TOTAL_CALLS = 0
    runtime.TOTAL_TOKENS = 0
    native.tracker.reset("build")
    started = time.monotonic()
    call_log: list[dict[str, Any]] = []
    try:
        build_time, event_count = native.build_memory(conv, str(staging))
        tracked = native.tracker.snapshot("build")
        call_log = list(runtime.CALL_LOG)
        actual_models = {
            str(entry.get("response_model") or "") for entry in call_log
        }
        if actual_models != {row["model"]}:
            raise RunError(
                f"response model mismatch: requested {row['model']}, got {actual_models}"
            )
        reasoning_tokens = sum(
            int(entry.get("reasoning_tokens", 0) or 0) for entry in call_log
        )
        if reasoning_tokens != 0 and not allow_reasoning_tokens:
            raise RunError(f"reasoning tokens must be zero, got {reasoning_tokens}")
        success = {
            "run_id": row["run_id"],
            "builder_model": row["model"],
            "write_turns": row["write_turns"],
            "session_group_size": row.get("session_group_size", 1),
            "completed_at": utc_now(),
        }
        atomic_json(staging / "_SUCCESS.json", success)
        os.replace(staging, target / "memory")
    except Exception as exc:
        failed_staging = target / f"memory.failed.{int(time.time())}"
        if staging.exists():
            os.replace(staging, failed_staging)
        if call_log:
            atomic_json(target / "call_log.failed.json", call_log)
        atomic_json(status_path, {
            "phase": "failed",
            "run_id": row["run_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "failed_memory": str(failed_staging) if failed_staging.exists() else None,
            "finished_at": utc_now(),
        })
        raise

    phases = phase_usage(call_log)
    totals = {
        name: sum(values[name] for values in phases.values())
        for name in (
            "calls", "input_tokens", "cached_input_tokens",
            "output_tokens", "reasoning_tokens",
        )
    }
    build_record = {
        "schema_version": 1,
        "status": "complete",
        "run_id": row["run_id"],
        "benchmark": row["benchmark"],
        "unit_id": row["unit_id"],
        "builder_model": row["model"],
        "tier": row["tier"],
        "reasoning_effort": row["reasoning_effort"],
        "reasoning_tokens_allowed": allow_reasoning_tokens,
        "write_turns": row["write_turns"],
        "session_group_size": row.get("session_group_size", 1),
        "environment": recorded_env,
        "messages": source_message_count(conv),
        "source_tokens": source_tokens,
        "source_tokenizer": tokenizer,
        "event_count": event_count,
        "reported_build_time_s": round(float(build_time), 3),
        "wall_time_s": round(time.monotonic() - started, 3),
        "tracker_totals": tracked,
        "usage": totals,
        "phase_usage": phases,
        "distill": summarize_trace(trace_path),
        "response_ids": [entry.get("response_id") for entry in call_log],
        "actual_models": sorted(actual_models),
        "provider": provider_contract,
        "source_hashes": runtime_source_hashes(matrix_path),
        "memory_dir": str((target / "memory").resolve()),
        "finished_at": utc_now(),
    }
    atomic_json(target / "call_log.json", call_log)
    atomic_json(target / "build.json", build_record)
    atomic_json(status_path, {
        "phase": "complete",
        "run_id": row["run_id"],
        "build": str((target / "build.json").resolve()),
        "finished_at": utc_now(),
    })
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--base-url", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES,
                        default="subscription-proxy")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_ORDER))
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(BENCHMARK_ORDER))
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--run-ids", nargs="+")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix_path = args.matrix.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    rows = load_matrix(matrix_path)
    if sha256_file(LOCOMO_EVALUATOR) != LOCKED_LOCOMO_EVALUATOR_SHA256:
        raise RunError("locked LoCoMo evaluator hash mismatch")

    if args.worker_run_id:
        matches = [row for row in rows if row["run_id"] == args.worker_run_id]
        if len(matches) != 1:
            raise RunError(f"worker run ID not found: {args.worker_run_id}")
        return execute_worker(
            matches[0], results_dir, args.base_url, matrix_path, args.provider,
        )

    selected = filter_rows(rows, args)
    if not selected:
        raise RunError("no matrix rows selected")
    if not args.execute:
        print(f"planned selection: {len(selected)} runs")
        for row in selected:
            print(row["run_id"])
        print("model requests sent: 0")
        return 0
    if not args.allow_model_requests:
        raise RunError("--execute requires --allow-model-requests")

    provider_contract = provider_preflight(
        args.provider,
        args.base_url,
        {row["model"] for row in selected},
    )
    status_path = results_dir / "status.json"
    completed = 0
    failures: list[dict[str, str]] = []
    atomic_json(results_dir / "execution_manifest.json", {
        "schema_version": 1,
        "started_at": utc_now(),
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "selected_run_ids": [row["run_id"] for row in selected],
        "base_url": args.base_url,
        "provider": provider_contract,
        "source_hashes": runtime_source_hashes(matrix_path),
        "reasoning_effort": provider_contract["requested_reasoning_effort"],
        "sequential": args.jobs == 1,
        "jobs": args.jobs,
    })
    if args.jobs < 1:
        raise RunError("--jobs must be >= 1")

    pending = []
    for position, row in enumerate(selected, start=1):
        target = run_dir(results_dir, row)
        if valid_complete(target, row):
            completed += 1
            continue
        pending.append((position, row, target))

    def launch(item):
        position, row, target = item
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--matrix", str(matrix_path),
            "--results-dir", str(results_dir),
            "--base-url", args.base_url,
            "--provider", args.provider,
            "--worker-run-id", row["run_id"],
        ]
        log_path = target / "run.log"
        target.mkdir(parents=True, exist_ok=True)
        print(f"[{position}/{len(selected)}] {row['run_id']}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
        return position, row, target, log_path, process.returncode

    atomic_json(status_path, {
        "phase": "running",
        "completed": completed,
        "selected_total": len(selected),
        "pending": len(pending),
        "jobs": args.jobs,
        "failed": failures,
        "updated_at": utc_now(),
    })
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(launch, item) for item in pending]
        for future in as_completed(futures):
            position, row, target, log_path, returncode = future.result()
            if returncode != 0 or not valid_complete(target, row):
                failure = {
                    "run_id": row["run_id"],
                    "reason": f"worker exit={returncode}",
                    "log": str(log_path),
                }
                failures.append(failure)
            else:
                completed += 1
            atomic_json(status_path, {
                "phase": "running",
                "last_finished": row["run_id"],
                "position": position,
                "completed": completed,
                "selected_total": len(selected),
                "failed": failures,
                "jobs": args.jobs,
                "updated_at": utc_now(),
            })

    if failures and not args.continue_on_error:
        atomic_json(status_path, {
            "phase": "failed",
            "completed": completed,
            "selected_total": len(selected),
            "failed": failures,
            "updated_at": utc_now(),
        })
        return 1

    atomic_json(status_path, {
        "phase": "complete" if not failures else "partial",
        "completed": completed,
        "selected_total": len(selected),
        "failed": failures,
        "finished_at": utc_now(),
    })
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
