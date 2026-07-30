#!/usr/bin/env python3
"""Run the frozen v8.8+calendar method on full LoCoMo with GPT-5.5.

The launcher never deletes an existing valid result.  Incomplete artifacts are
renamed into ``partials/`` before a retry, and ``run_manifest.json`` records the
exact configuration and per-sample state.  Each sample is an independent
checkpoint, so an interrupted run can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402

DATA = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUT = ROOT / "results" / "v88-calendar-gpt55-locomo-20260714"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_samples(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            lo, hi = sorted((a, b))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    vals = sorted(out)
    if not vals or any(x < 0 or x > 9 for x in vals):
        raise argparse.ArgumentTypeError("samples must be in 0..9")
    return vals


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def expected_questions(sample: int, questions_limit: int | None = None) -> int:
    with DATA.open() as f:
        qa = json.load(f)[sample]["qa"]
    return min(len(qa), questions_limit) if questions_limit else len(qa)


def validate_sample(path: Path, sample: int, questions_limit: int | None = None,
                    expected_model: str | None = None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with path.open() as f:
            records = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return False, f"invalid_json:{type(exc).__name__}"
    if not isinstance(records, list):
        return False, "not_a_list"
    builds = [r for r in records if r.get("question_id") == "_build_stats"]
    qs = [r for r in records if r.get("question_id") != "_build_stats"]
    want = expected_questions(sample, questions_limit)
    if len(builds) != 1:
        return False, f"build_stats={len(builds)}"
    notes = str(builds[0].get("notes", ""))
    if (expected_model and f"model={expected_model}" not in notes
            and not notes.startswith("reused existing memory dir:")):
        return False, "builder_model_mismatch"
    if len(qs) != want:
        return False, f"questions={len(qs)},expected={want}"
    expected_ids = {f"s{sample}_q{i}" for i in range(want)}
    got_ids = {str(r.get("question_id")) for r in qs}
    if got_ids != expected_ids:
        return False, "question_ids_mismatch"
    missing_answers = [r.get("question_id") for r in qs if not r.get("answer")]
    if missing_answers:
        return False, f"missing_answers={len(missing_answers)}"
    return True, "complete"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src" / "nativemem.py",
        ROOT / "src" / "v8_memory.py",
        ROOT / "src" / "adapters" / "run_nativemem.py",
        ROOT / "src" / "chatgpt_proxy.py",
        ROOT / "src" / "openai_gpt55_flex_gateway.py",
        ROOT / "src" / "openai_gpt55_flex_gateway_evidence.py",
        ROOT / "scripts" / "gpt55_run_proxy.py",
        DATA,
        Path(__file__).resolve(),
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def fingerprint(config: dict, sources: dict[str, str]) -> str:
    payload = json.dumps({"config": config, "sources": sources}, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def memory_reusable(out_dir: Path, sample: int) -> bool:
    memory = out_dir / f"memory_sample{sample}"
    if not memory.is_dir():
        return False
    output = out_dir / f"sample{sample}_questions.json"
    if output.exists():
        try:
            records = json.loads(output.read_text())
            if any(r.get("question_id") == "_build_stats" for r in records):
                return True
        except Exception:  # noqa: BLE001
            pass
    log = out_dir / f"sample{sample}.log"
    if log.exists():
        log_text = log.read_text(errors="replace")
        if (
            "[nativemem] build done:" in log_text
            or "[nativemem] reused existing memory dir:" in log_text
        ):
            return True
    return False


def archive_partial(out_dir: Path, sample: int, preserve_memory: bool = False) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    partials = out_dir / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"sample{sample}_questions.json",
             out_dir / f"sample{sample}.log"]
    if not preserve_memory:
        paths.append(out_dir / f"memory_sample{sample}")
    for path in paths:
        if path.exists():
            target = partials / f"{path.name}.{stamp}"
            shutil.move(str(path), str(target))


def experiment_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "NATIVEMEM_PROMPT": "v8",
        "NATIVEMEM_STORE_MODE": "oneshot",
        "NATIVEMEM_V8_SINGLE": "1",
        "NATIVEMEM_CHUNK_TURNS": "6",
        "NATIVEMEM_V8_SEGMENT": "fixed",
        "NATIVEMEM_V8_TIDY": "on",
        "NATIVEMEM_V8_SECTIONS": "on",
        "NATIVEMEM_V8_ARTICLE": "off",
        "NATIVEMEM_V8_TIDY_COMBINED": "off",
        "NATIVEMEM_V8_VERIFY": "on",
        "NATIVEMEM_V8_MERGE_LINES": "on",
        "NATIVEMEM_V8_MAX_TOPICS": "30",
        "NATIVEMEM_V8_MAP": "dir",
        "NATIVEMEM_V8_MAP_INLINE": "8",
        "NATIVEMEM_V8_MAX_ROUNDS": "12",
        "NATIVEMEM_V8_MAX_TOKENS": "1200",
        "NATIVEMEM_V8_READ_CONTEXT": "1",
        "NATIVEMEM_V8_REWRITE_MIN": "8",
        "NATIVEMEM_V8_MAX_DEPTH": "4",
        "NATIVEMEM_TOPK": "20",
        "NATIVEMEM_TRUST_PROXY": "0",
        "NATIVEMEM_V8_CONCURRENCY": str(args.request_concurrency),
        "BUILDER_MODEL": args.model,
        "BUILDER_BASE": args.base_url,
        "BUILDER_KEY": args.api_key,
        # src.nativemem historically used ALIYUN_KEY for its OpenAI-compatible
        # client.  Keep this explicit for old checkouts and remove any provider
        # proxy inherited from the shell.
        "ALIYUN_KEY": args.api_key,
    })
    env.pop("NATIVEMEM_V9_PIPELINE", None)
    env.pop("NATIVEMEM_V9_SCRIBE_MODE", None)
    env.pop("MODEL", None)
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"
    if hasattr(args, "nativemem_openai_max_retries"):
        env["NATIVEMEM_OPENAI_MAX_RETRIES"] = str(
            args.nativemem_openai_max_retries
        )
    if hasattr(args, "nativemem_http_timeout"):
        env["NATIVEMEM_HTTP_TIMEOUT"] = str(args.nativemem_http_timeout)
    return env


def artifact_metadata(out_dir: Path, sample: int) -> dict:
    output = out_dir / f"sample{sample}_questions.json"
    records = json.loads(output.read_text())
    qs = [r for r in records if r.get("question_id") != "_build_stats"]
    memory = out_dir / f"memory_sample{sample}"
    return {
        "json_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "questions": len(qs),
        "empty_answers": sum(not str(r.get("answer", "")).strip() for r in qs),
        "memory_markdown_files": sum(1 for p in memory.rglob("*.md")) if memory.exists() else 0,
    }


def run_sample(sample: int, args: argparse.Namespace, out_dir: Path,
               manifest: dict, lock: threading.Lock) -> tuple[int, bool, str]:
    out = out_dir / f"sample{sample}_questions.json"
    valid, reason = validate_sample(out, sample, args.questions_limit, args.model)
    if valid and not args.force:
        with lock:
            manifest["samples"][str(sample)] = {
                "status": "complete", "validated_at": utc_now(), "reason": reason}
            atomic_json(out_dir / "run_manifest.json", manifest)
        return sample, True, "already complete"

    reuse_memory = memory_reusable(out_dir, sample)
    archive_partial(out_dir, sample, preserve_memory=reuse_memory)

    def command(reuse: bool) -> list[str]:
        cmd = [sys.executable, "src/adapters/run_nativemem.py",
               "--sample", str(sample), "--output", str(out)]
        if reuse:
            cmd += ["--memory-dir", str(out_dir / f"memory_sample{sample}")]
        if args.max_sessions is not None:
            cmd += ["--max-sessions", str(args.max_sessions)]
        if args.questions_limit is not None:
            cmd += ["--questions-limit", str(args.questions_limit)]
        return cmd

    cmd = command(reuse_memory)

    with lock:
        manifest["samples"][str(sample)] = {
            "status": "running", "started_at": utc_now(), "command": cmd}
        atomic_json(out_dir / "run_manifest.json", manifest)

    log_path = out_dir / f"sample{sample}.log"
    last_error = ""
    for attempt in range(1, args.retries + 1):
        cmd = command(reuse_memory)
        with log_path.open("a") as log:
            log.write(f"\n[{utc_now()}] attempt={attempt}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, env=experiment_env(args),
                                  stdout=log, stderr=subprocess.STDOUT)
        valid, reason = validate_sample(out, sample, args.questions_limit, args.model)
        if proc.returncode == 0 and valid:
            with lock:
                manifest["samples"][str(sample)] = {
                    "status": "complete", "finished_at": utc_now(),
                    "attempt": attempt, "validation": reason,
                    "output": str(out.relative_to(ROOT)),
                    "log": str(log_path.relative_to(ROOT)),
                    "artifact": artifact_metadata(out_dir, sample),
                    "memory_reused": reuse_memory,
                }
                atomic_json(out_dir / "run_manifest.json", manifest)
            return sample, True, "complete"
        last_error = f"returncode={proc.returncode},validation={reason}"
        if attempt < args.retries:
            reuse_memory = memory_reusable(out_dir, sample)
            archive_partial(out_dir, sample, preserve_memory=reuse_memory)

    with lock:
        manifest["samples"][str(sample)] = {
            "status": "failed", "finished_at": utc_now(), "error": last_error}
        atomic_json(out_dir / "run_manifest.json", manifest)
    return sample, False, last_error


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", type=parse_samples, default=parse_samples("0-9"))
    ap.add_argument("--gateway-root", type=Path, required=True)
    ap.add_argument("--allow-model-requests", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample-workers", type=int, default=1)
    ap.add_argument("--request-concurrency", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-sessions", type=int)
    ap.add_argument("--questions-limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.model = "gpt-5.5"
    args.api_key = "x"
    if not args.allow_model_requests:
        ap.error("formal LoCoMo execution requires --allow-model-requests")
    if args.sample_workers < 1 or args.request_concurrency < 1 or args.retries < 1:
        ap.error("worker/concurrency/retries values must be positive")

    out_dir = args.output_dir.expanduser().resolve()
    old_manifest = out_dir / "run_manifest.json"
    if old_manifest.exists() and not args.resume:
        ap.error(
            f"manifest exists at {old_manifest}; pass --resume or use a new "
            "--output-dir"
        )
    if args.resume and old_manifest.exists():
        try:
            existing = json.loads(old_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ap.error(f"cannot read existing manifest: {exc}")
        provider = existing.get("provider_evidence")
        if (
            not isinstance(provider, dict)
            or provider.get("schema") != "openai-gpt55-flex-invocations/v1"
            or provider.get("active_run_id") is not None
        ):
            ap.error(
                "existing output is not a closed Flex-evidence run; use a new "
                "--output-dir"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (out_dir / ".launcher.lock").open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another launcher is already using {out_dir}")
    config = {
        "samples": args.samples,
        "model": args.model,
        "provider": "openai_api_flex_via_exclusive_child_proxy",
        "gateway_root": str(args.gateway_root.expanduser().resolve()),
        "explicit_model_request_authorization": True,
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "sample_workers": args.sample_workers,
        "request_concurrency": args.request_concurrency,
        "max_sessions": args.max_sessions,
        "questions_limit": args.questions_limit,
    }
    sources = source_hashes()
    run_fingerprint = fingerprint(config, sources)
    previous: dict | None = None
    if old_manifest.exists():
        previous = json.loads(old_manifest.read_text())
        if previous.get("fingerprint") != run_fingerprint:
            raise SystemExit(
                "existing run_manifest.json has a different configuration or source "
                "hash; formal provider results require a new output directory. "
                "--force may rebuild samples only when the run fingerprint is unchanged"
            )

    invocation = flex_evidence.begin_child_invocation(
        args.gateway_root,
        out_dir / "provider_evidence" / f"invocation-{utc_now().replace(':', '')}",
    )
    args.base_url = invocation.base_url
    os.environ["CHATGPT_PROXY_LOG"] = str(invocation.log_path)
    manifest = {
        "schema_version": 1,
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": args.model,
        "git_commit": git_head(),
        "created_at": utc_now(),
        "output_dir": str(out_dir),
        "config": config,
        "source_hashes": sources,
        "fingerprint": run_fingerprint,
        "samples": {},
        "provider_evidence": {
            "schema": "openai-gpt55-flex-invocations/v1",
            "gateway_root": str(args.gateway_root.expanduser().resolve()),
            "active_run_id": invocation.run_id,
            "invocations": [],
        },
    }
    if previous is not None:
        try:
            manifest["created_at"] = previous.get(
                "created_at", manifest["created_at"]
            )
            manifest["samples"].update(previous.get("samples", {}))
            manifest["provider_evidence"]["invocations"] = previous[
                "provider_evidence"
            ]["invocations"]
        except BaseException:  # noqa: BLE001
            invocation.abort()
            raise
    atomic_json(old_manifest, manifest)

    try:
        lock = threading.Lock()
        failures = []
        with ThreadPoolExecutor(
            max_workers=min(args.sample_workers, len(args.samples))
        ) as executor:
            futures = {
                executor.submit(run_sample, sample, args, out_dir, manifest, lock): sample
                for sample in args.samples
            }
            for future in as_completed(futures):
                sample, ok, detail = future.result()
                print(
                    f"sample {sample}: {'complete' if ok else 'failed'} ({detail})",
                    flush=True,
                )
                if not ok:
                    failures.append(sample)
        evidence_record = invocation.finish()
    except Exception:
        invocation.abort()
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        atomic_json(old_manifest, manifest)
        raise

    manifest["provider_evidence"]["active_run_id"] = None
    manifest["provider_evidence"]["invocations"].append(evidence_record)
    manifest["finished_at"] = utc_now()
    manifest["status"] = "failed" if failures else "complete"
    manifest["failed_samples"] = failures
    atomic_json(old_manifest, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
