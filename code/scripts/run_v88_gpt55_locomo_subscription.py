#!/usr/bin/env python3
"""Run full LoCoMo with GPT-5.5 subscription access and reasoning disabled."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_v88_gpt55_locomo as shared  # noqa: E402
from src.adapters.question_checkpoint import memory_sha256  # noqa: E402


MODEL = "gpt-5.5"
REASONING_EFFORT = "none"
DEFAULT_OUT = (
    ROOT / "results" / "gpt55-locomo-v88-calendar-thinking-off-fast-20260714"
)
DEFAULT_PROXY_BASE = "http://127.0.0.1:8199/v1"
SAMPLE_WORKERS = 2
REQUEST_CONCURRENCY = 1
PROXY_CONCURRENCY = 2
RUN_RETRIES = 2
SDK_MAX_RETRIES = 0
SDK_HTTP_TIMEOUT_S = 600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src" / "nativemem.py",
        ROOT / "src" / "v8_memory.py",
        ROOT / "src" / "adapters" / "run_nativemem.py",
        ROOT / "src" / "adapters" / "question_checkpoint.py",
        ROOT / "src" / "chatgpt_proxy.py",
        ROOT / "scripts" / "run_v88_gpt55_locomo.py",
        Path(__file__).resolve(),
        shared.DATA,
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def fingerprint(config: dict, sources: dict[str, str]) -> str:
    encoded = json.dumps(
        {"config": config, "sources": sources},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def health_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("subscription proxy must be a loopback HTTP endpoint")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path}/healthz", "", "")
    )


def read_proxy_health(base_url: str) -> dict:
    request = urllib.request.Request(health_url(base_url), method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"subscription proxy health check failed: {exc}") from exc


def validate_proxy_health(health: dict, expected_log: Path) -> None:
    if health.get("status") != "ok" or health.get("auth_readable") is not True:
        raise RuntimeError("subscription proxy authentication is not ready")
    if health.get("max_concurrency") != PROXY_CONCURRENCY:
        raise RuntimeError(
            f"subscription proxy concurrency must equal {PROXY_CONCURRENCY}"
        )
    if health.get("max_attempts") != 2 or health.get("read_timeout_s") != 150.0:
        raise RuntimeError("subscription proxy retry/timeout policy differs")
    if health.get("requested_reasoning_effort") != REASONING_EFFORT:
        raise RuntimeError("subscription proxy is not fixed to reasoning.effort=none")
    raw_log = health.get("request_log")
    if not raw_log or Path(raw_log).expanduser().resolve() != expected_log.resolve():
        raise RuntimeError("subscription proxy request log differs from this result root")


def summarize_proxy_log(path: Path) -> dict:
    summary = {
        "successes": 0,
        "errors": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "thinking_off_verified": True,
    }
    if not path.exists():
        return summary
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid proxy log JSON on line {line_number}"
                ) from exc
            if record.get("status") == "success":
                summary["successes"] += 1
                usage = record.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
                summary["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                summary["cached_prompt_tokens"] += details.get("cached_tokens", 0) or 0
                summary["completion_tokens"] += usage.get("completion_tokens", 0) or 0
                summary["reasoning_tokens"] += reasoning_tokens
                if (
                    record.get("requested_reasoning_effort") != "none"
                    or record.get("actual_reasoning_effort") not in (None, "none")
                    or reasoning_tokens != 0
                ):
                    summary["thinking_off_verified"] = False
            elif record.get("status") == "error":
                summary["errors"] += 1
    return summary


def build_config(args: argparse.Namespace, proxy_log: Path) -> dict:
    return {
        "samples": args.samples,
        "model": MODEL,
        "provider": "chatgpt_pro_subscription",
        "proxy_base_url": args.proxy_base_url,
        "proxy_request_log": str(proxy_log),
        "reasoning_effort": REASONING_EFFORT,
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "sample_workers": SAMPLE_WORKERS,
        "request_concurrency": REQUEST_CONCURRENCY,
        "proxy_concurrency": PROXY_CONCURRENCY,
        "retries": RUN_RETRIES,
        "sdk_max_retries": SDK_MAX_RETRIES,
        "sdk_http_timeout_s": SDK_HTTP_TIMEOUT_S,
        "imported_sample0_memory": getattr(
            args, "imported_sample0_memory", None
        ),
        "max_sessions": None,
        "questions_limit": None,
    }


def prepare_shared_args(args: argparse.Namespace) -> None:
    args.model = MODEL
    args.api_key = "subscription-proxy"
    args.base_url = args.proxy_base_url
    args.sample_workers = SAMPLE_WORKERS
    args.request_concurrency = REQUEST_CONCURRENCY
    args.retries = RUN_RETRIES
    args.nativemem_openai_max_retries = SDK_MAX_RETRIES
    args.nativemem_http_timeout = SDK_HTTP_TIMEOUT_S
    args.max_sessions = None
    args.questions_limit = None
    args.force = False


def import_sample0_memory(source_root: Path, output_dir: Path) -> dict:
    source_root = source_root.expanduser().resolve()
    source = source_root / "memory_sample0"
    if not source.is_dir():
        raise RuntimeError(f"sample 0 memory is missing at {source}")

    source_hash = memory_sha256(source)
    destination = output_dir / "memory_sample0"
    if destination.exists():
        if not destination.is_dir() or memory_sha256(destination) != source_hash:
            raise RuntimeError("existing imported sample 0 memory differs")
    else:
        shutil.copytree(source, destination)
        if memory_sha256(destination) != source_hash:
            raise RuntimeError("copied sample 0 memory failed hash verification")

    old_manifest = source_root / "run_manifest.json"
    old_fingerprint = None
    if old_manifest.exists():
        old_fingerprint = json.loads(
            old_manifest.read_text(encoding="utf-8")
        ).get("fingerprint")
    provenance = {
        "source_result_root": str(source_root),
        "source_memory": str(source),
        "memory_sha256": source_hash,
        "source_fingerprint": old_fingerprint,
    }

    # This incomplete artifact makes the shared launcher preserve the imported
    # memory on the first attempt.  archive_partial() retains it as provenance.
    seed_output = output_dir / "sample0_questions.json"
    checkpoint = output_dir / "sample0_questions.checkpoint.json"
    if not seed_output.exists() and not checkpoint.exists():
        shared.atomic_json(seed_output, [{
            "question_id": "_build_stats",
            "build_time_s": 0.0,
            "num_memories": sum(
                1 for path in destination.rglob("*.md")
                if "raw" not in path.parts
            ),
            "notes": (
                "imported existing sample 0 memory; "
                f"sha256={source_hash}; source={source_root}"
            ),
            "build_calls": None,
            "build_tokens_in": None,
            "build_tokens_out": None,
            "build_llm_time_s": None,
        }])
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--samples", type=shared.parse_samples, default=shared.parse_samples("0-9")
    )
    parser.add_argument("--proxy-base-url", default=DEFAULT_PROXY_BASE)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--import-sample0-memory-from", type=Path)
    args = parser.parse_args(argv)
    if not args.allow_model_requests:
        parser.error("subscription LoCoMo execution requires --allow-model-requests")

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error(
            f"manifest exists at {manifest_path}; pass --resume or use a new output dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_log = output_dir / "proxy_requests.jsonl"

    args.imported_sample0_memory = None
    if args.import_sample0_memory_from is not None:
        if 0 not in args.samples:
            parser.error("sample 0 memory import requires sample 0 in --samples")
        args.imported_sample0_memory = import_sample0_memory(
            args.import_sample0_memory_from, output_dir
        )

    health = read_proxy_health(args.proxy_base_url)
    validate_proxy_health(health, proxy_log)
    prepare_shared_args(args)

    lock_handle = (output_dir / ".launcher.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another launcher is using {output_dir}") from exc

    config = build_config(args, proxy_log)
    sources = source_hashes()
    run_fingerprint = fingerprint(config, sources)
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != run_fingerprint:
            raise SystemExit(
                "existing run manifest has a different configuration or source hash"
            )

    manifest = {
        "schema_version": 1,
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": MODEL,
        "provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "thinking_off": True,
        "git_commit": shared.git_head(),
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "config": config,
        "proxy_health": health,
        "source_hashes": sources,
        "fingerprint": run_fingerprint,
        "samples": {},
        "status": "running",
    }
    if previous is not None:
        manifest["created_at"] = previous.get("created_at", manifest["created_at"])
        manifest["samples"].update(previous.get("samples", {}))
    shared.atomic_json(manifest_path, manifest)

    lock = threading.Lock()
    failed_samples = []
    with ThreadPoolExecutor(
        max_workers=min(SAMPLE_WORKERS, len(args.samples))
    ) as executor:
        futures = {
            executor.submit(
                shared.run_sample, sample, args, output_dir, manifest, lock
            ): sample
            for sample in args.samples
        }
        for future in as_completed(futures):
            sample_id, ok, detail = future.result()
            print(
                f"sample {sample_id}: {'complete' if ok else 'failed'} "
                f"({detail})",
                flush=True,
            )
            if not ok:
                failed_samples.append(sample_id)

    proxy_summary = summarize_proxy_log(proxy_log)
    manifest["proxy_summary"] = proxy_summary
    manifest["finished_at"] = utc_now()
    if failed_samples:
        manifest["status"] = "failed"
        manifest["failed_samples"] = sorted(failed_samples)
    elif not proxy_summary["thinking_off_verified"]:
        manifest["status"] = "failed"
        manifest["failure"] = "thinking_off_evidence_mismatch"
    else:
        manifest["status"] = "complete"
    shared.atomic_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
