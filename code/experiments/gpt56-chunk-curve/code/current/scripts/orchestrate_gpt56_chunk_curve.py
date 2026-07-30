#!/usr/bin/env python3
"""Wait for the GPT-5.6 smoke build, then run Luna, Terra, and Sol builds."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_gpt56_chunk_curve.py"
DEFAULT_RESULTS = ROOT / "results" / "formal" / "gpt56-chunk-curve"
DEFAULT_SMOKE_BUILD = (
    DEFAULT_RESULTS / "beam-100k" / "100K-conv-1" / "luna" / "w32" /
    "build.json"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def valid_smoke(path: Path, run_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("status") == "complete"
        and record.get("run_id") == run_id
        and record.get("reasoning_effort") == "none"
        and record.get("usage", {}).get("reasoning_tokens") == 0
        and (path.parent / "memory" / "_SUCCESS.json").is_file()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8202/v1")
    parser.add_argument(
        "--provider", choices=("subscription-proxy", "frontier"),
        default="subscription-proxy",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--smoke-build", type=Path, default=DEFAULT_SMOKE_BUILD,
    )
    parser.add_argument(
        "--smoke-run-id",
        default="gpt56-beam-100k-100K-conv-1-luna-w32",
    )
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute or not args.allow_model_requests:
        print("dry run: would wait for smoke, then execute luna, terra, sol")
        print("model requests sent: 0")
        return 0

    results = args.results_dir.expanduser().resolve()
    smoke_build = args.smoke_build.expanduser().resolve()
    status_path = results / "orchestration_status.json"
    results.mkdir(parents=True, exist_ok=True)
    lock = (results / ".orchestration.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(
            f"another orchestrator is already using {results}"
        ) from exc

    atomic_json(status_path, {
        "phase": "waiting_for_smoke",
        "smoke_build": str(smoke_build),
        "smoke_run_id": args.smoke_run_id,
        "results_dir": str(results),
        "provider": args.provider,
        "base_url": args.base_url,
        "started_at": now(),
        "model_order": ["luna", "terra", "sol"],
    })
    while not valid_smoke(smoke_build, args.smoke_run_id):
        smoke_status = smoke_build.parent / "status.json"
        if smoke_status.is_file():
            state = json.loads(smoke_status.read_text(encoding="utf-8"))
            if state.get("phase") in {"failed", "interrupted"}:
                atomic_json(status_path, {
                    "phase": "blocked",
                    "reason": "smoke did not complete",
                    "smoke_status": state,
                    "updated_at": now(),
                })
                return 1
        time.sleep(max(5, args.poll_seconds))

    completed_tiers = []
    for tier in ("luna", "terra", "sol"):
        atomic_json(status_path, {
            "phase": "running",
            "current_tier": tier,
            "completed_tiers": completed_tiers,
            "updated_at": now(),
        })
        log_path = results / f"controller-{tier}.log"
        command = [
            sys.executable,
            str(RUNNER),
            "--models", tier,
            "--provider", args.provider,
            "--base-url", args.base_url,
            "--results-dir", str(results),
            "--execute",
            "--allow-model-requests",
        ]
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            atomic_json(status_path, {
                "phase": "failed",
                "current_tier": tier,
                "completed_tiers": completed_tiers,
                "runner_exit": process.returncode,
                "log": str(log_path),
                "updated_at": now(),
            })
            return process.returncode
        completed_tiers.append(tier)

    atomic_json(status_path, {
        "phase": "builds_complete",
        "completed_tiers": completed_tiers,
        "finished_at": now(),
    })
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
