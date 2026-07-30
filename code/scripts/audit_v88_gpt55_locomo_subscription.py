#!/usr/bin/env python3
"""Audit the completed GPT-5.5 subscription LoCoMo run.

This auditor is intentionally separate from ``audit_v88_gpt55_locomo``.
The latter proves an OpenAI API Flex transport contract; this module records
the weaker ChatGPT Pro subscription evidence without representing it as Flex.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_v88_gpt55_locomo as common  # noqa: E402
from scripts import run_v88_gpt55_locomo_subscription as runner  # noqa: E402
from src.adapters.question_checkpoint import memory_sha256  # noqa: E402


EXPECTED_SAMPLES = list(range(10))
EXPECTED_CATEGORY_COUNTS = common.EXPECTED_CATEGORY_COUNTS
IMPORTED_BUILD_ACCOUNTING_STATUS = (
    "unavailable_imported_subscription_memory"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_IMPORT_REQUIRED_HASHES = frozenset({
    "src/nativemem.py",
    "src/v8_memory.py",
    "src/adapters/run_nativemem.py",
    "src/chatgpt_proxy.py",
    "scripts/run_v88_gpt55_locomo.py",
    "scripts/run_v88_gpt55_locomo_subscription.py",
    "benchmarks/locomo/data/locomo10.json",
})


class AuditError(RuntimeError):
    """The subscription result does not satisfy its recorded contract."""


def _require_regular_tree(root: Path, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise AuditError(f"{label} is missing or unsafe: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AuditError(f"{label} contains a symbolic link: {path}")


def _normal_build_record(
    run_dir: Path,
    sample: int,
    current: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    try:
        build = common.resolve_build_record(
            run_dir, sample, current, state
        )
    except common.AuditError as exc:
        raise AuditError(str(exc)) from exc
    if "build_accounting" in build:
        raise AuditError(
            f"sample {sample} normal build carries imported build accounting"
        )
    for key in ("build_calls", "build_tokens_in", "num_memories"):
        value = build.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise AuditError(f"sample {sample} has invalid {key}")
    return build


def _memory_event_count(root: Path) -> int:
    return sum(
        1 for path in root.rglob("*.md")
        if path.is_file() and "raw" not in path.relative_to(root).parts
    )


def _source_import_evidence(
    source_root: Path,
    source_memory: Path,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    recorded_fingerprint: str,
    recorded_memory_hash: str,
) -> dict[str, Any]:
    source_config = source_manifest.get("config")
    source_hashes = source_manifest.get("source_hashes")
    if not isinstance(source_config, dict) or not isinstance(source_hashes, dict):
        raise AuditError(
            "sample 0 source manifest lacks config/source_hashes provenance"
        )
    if set(source_hashes) != SOURCE_IMPORT_REQUIRED_HASHES:
        raise AuditError(
            "sample 0 source manifest source-hash paths differ: "
            f"missing={sorted(SOURCE_IMPORT_REQUIRED_HASHES - set(source_hashes))}, "
            f"unexpected={sorted(set(source_hashes) - SOURCE_IMPORT_REQUIRED_HASHES)}"
        )
    for relative, digest in source_hashes.items():
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise AuditError("sample 0 source manifest has invalid source hashes")
    if runner.fingerprint(source_config, source_hashes) != recorded_fingerprint:
        raise AuditError("sample 0 source manifest fingerprint does not recompute")

    source_config_expected = {
        "samples": EXPECTED_SAMPLES,
        "model": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "reasoning_effort": "none",
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "max_sessions": None,
        "questions_limit": None,
    }
    source_config_mismatches = {
        key: {"expected": value, "actual": source_config.get(key)}
        for key, value in source_config_expected.items()
        if source_config.get(key) != value
    }
    if source_config_mismatches:
        raise AuditError(
            "sample 0 source method configuration differs: "
            f"{source_config_mismatches}"
        )
    expected_source_log = source_root / "proxy_requests.jsonl"
    if Path(str(source_config.get("proxy_request_log", ""))).resolve() != (
        expected_source_log
    ):
        raise AuditError("sample 0 source proxy log path differs")

    if source_manifest.get("status") != "failed":
        raise AuditError("sample 0 source run is not recorded as failed")
    source_samples = source_manifest.get("samples")
    source_sample = (
        source_samples.get("0", {}) if isinstance(source_samples, dict) else {}
    )
    if (
        not isinstance(source_sample, dict)
        or source_sample.get("status") != "failed"
        or not isinstance(source_sample.get("error"), str)
        or not source_sample["error"].strip()
        or not source_sample.get("finished_at")
    ):
        raise AuditError("sample 0 source failure state is incomplete")

    source_log = source_root / "sample0.log"
    if source_log.is_symlink() or not source_log.is_file():
        raise AuditError("sample 0 source build log is missing or unsafe")
    try:
        log_lines = source_log.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AuditError("sample 0 source build log is not UTF-8") from exc
    expected_start = f"[nativemem] building memory -> {source_memory}"
    build_markers = [
        line for line in log_lines
        if re.fullmatch(r"\[nativemem\] build done: [0-9]+(?:\.[0-9]+)?s", line)
    ]
    if expected_start not in log_lines or len(build_markers) != 1:
        raise AuditError("sample 0 source log lacks unique build-complete evidence")
    start_index = log_lines.index(expected_start)
    marker_index = log_lines.index(build_markers[0])
    if marker_index <= start_index:
        raise AuditError("sample 0 source build-complete evidence is out of order")

    current_matches: list[str] = []
    for relative, recorded_hash in sorted(source_hashes.items()):
        current_path = ROOT / relative
        current_hash = (
            common.sha256_file(current_path) if current_path.is_file() else None
        )
        if current_hash == recorded_hash:
            current_matches.append(relative)
    dataset_relative = "benchmarks/locomo/data/locomo10.json"
    if dataset_relative not in current_matches:
        raise AuditError("sample 0 source dataset hash differs from the current benchmark")

    return {
        "status": "source_build_complete_with_import_time_memory_hash",
        "source_run_status": source_manifest["status"],
        "source_sample_status": source_sample["status"],
        "source_sample_error": source_sample["error"],
        "source_log": {
            "path": str(source_log),
            "sha256": common.sha256_file(source_log),
            "build_done_marker": build_markers[0],
            "sha256_persisted_at_import": False,
        },
        "import_time_memory_sha256": recorded_memory_hash,
        "source_manifest_fingerprint_recomputed": True,
        "source_hashes": {
            "recorded": len(source_hashes),
            "current_worktree_matches": current_matches,
            "current_worktree_mismatches": sorted(
                set(source_hashes) - set(current_matches)
            ),
        },
        "limitation": (
            "the memory SHA-256 was persisted at import time; the source build-log "
            "SHA-256 is authenticated only at audit time"
        ),
    }


def authenticate_imported_sample0_build(
    run_dir: Path,
    manifest: dict[str, Any],
    current: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate imported memory while preserving unavailable accounting.

    The source process completed sample-0 memory construction but was stopped
    before its in-memory ``_build_stats`` record was written.  Consequently the
    exact build-call and build-token totals are unavailable.  They remain null;
    no estimate or placeholder count is introduced.
    """

    imported = manifest.get("config", {}).get("imported_sample0_memory")
    if not isinstance(imported, dict):
        raise AuditError("sample 0 has no imported-memory provenance")
    if state.get("memory_reused") is not True:
        raise AuditError("sample 0 import lacks manifest memory_reused evidence")

    source_root_value = imported.get("source_result_root")
    source_memory_value = imported.get("source_memory")
    recorded_memory_hash = imported.get("memory_sha256")
    recorded_fingerprint = imported.get("source_fingerprint")
    if not all(
        isinstance(value, str) and value
        for value in (
            source_root_value,
            source_memory_value,
            recorded_memory_hash,
            recorded_fingerprint,
        )
    ):
        raise AuditError("sample 0 imported-memory provenance is incomplete")
    if SHA256_PATTERN.fullmatch(recorded_memory_hash) is None:
        raise AuditError("sample 0 imported memory hash is invalid")
    if SHA256_PATTERN.fullmatch(recorded_fingerprint) is None:
        raise AuditError("sample 0 source fingerprint is invalid")

    source_root = Path(source_root_value).expanduser().resolve()
    source_memory = Path(source_memory_value).expanduser().resolve()
    if source_memory != source_root / "memory_sample0":
        raise AuditError("sample 0 source memory path differs from its result root")
    imported_memory = (run_dir / "memory_sample0").resolve()
    for path, label in (
        (source_memory, "sample 0 source memory"),
        (imported_memory, "sample 0 imported memory"),
    ):
        _require_regular_tree(path, label)
        if memory_sha256(path) != recorded_memory_hash:
            raise AuditError(f"{label} hash differs from the import record")

    source_manifest_path = source_root / "run_manifest.json"
    if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
        raise AuditError("sample 0 source manifest is missing or unsafe")
    source_manifest = common.read_json(source_manifest_path)
    if not isinstance(source_manifest, dict):
        raise AuditError("sample 0 source manifest is not an object")
    source_identity = {
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "fingerprint": recorded_fingerprint,
        "output_dir": str(source_root),
    }
    mismatches = {
        key: {"expected": value, "actual": source_manifest.get(key)}
        for key, value in source_identity.items()
        if source_manifest.get(key) != value
    }
    if mismatches:
        raise AuditError(
            f"sample 0 source manifest identity differs: {mismatches}"
        )
    source_import_evidence = _source_import_evidence(
        source_root,
        source_memory,
        source_manifest_path,
        source_manifest,
        recorded_fingerprint,
        recorded_memory_hash,
    )

    expected_note = f"reused existing memory dir: {imported_memory}"
    if str(current.get("notes", "")) != expected_note:
        raise AuditError("sample 0 build record does not identify imported memory")
    if any(current.get(key) is not None for key in (
        "build_calls", "build_tokens_in", "build_tokens_out", "build_llm_time_s"
    )):
        raise AuditError(
            "sample 0 imported build unexpectedly claims unavailable accounting"
        )
    memories = current.get("num_memories")
    if isinstance(memories, bool) or not isinstance(memories, int) or memories <= 0:
        raise AuditError("sample 0 imported build has invalid num_memories")
    actual_memories = _memory_event_count(imported_memory)
    if memories != actual_memories:
        raise AuditError(
            "sample 0 imported build num_memories differs from its memory tree"
        )
    build_time = current.get("build_time_s")
    if (
        isinstance(build_time, bool)
        or not isinstance(build_time, (int, float))
        or float(build_time) != 0.0
    ):
        raise AuditError("sample 0 imported build time must remain zero")

    result = copy.deepcopy(current)
    result["build_accounting"] = {
        "status": IMPORTED_BUILD_ACCOUNTING_STATUS,
        "reason": (
            "source process ended after memory construction and before the "
            "sample build record was persisted"
        ),
        "source_provider": "chatgpt_pro_subscription",
        "source_result_root": str(source_root),
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": common.sha256_file(source_manifest_path),
            "fingerprint": recorded_fingerprint,
            "status": source_manifest.get("status"),
        },
        "source_completion_evidence": source_import_evidence,
        "source_memory": {
            "path": str(source_memory),
            "sha256": recorded_memory_hash,
        },
        "imported_memory": {
            "path": str(imported_memory),
            "sha256": recorded_memory_hash,
        },
        "build_calls": None,
        "build_tokens_in": None,
        "build_tokens_out": None,
        "build_llm_time_s": None,
    }
    return result


def resolve_build_record(
    run_dir: Path,
    manifest: dict[str, Any],
    sample: int,
    current: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    notes = str(current.get("notes", ""))
    imported = manifest.get("config", {}).get("imported_sample0_memory")
    if sample == 0 and isinstance(imported, dict):
        if not notes.startswith("reused existing memory dir:"):
            raise AuditError(
                "sample 0 configured import did not use imported-memory authentication"
            )
        return authenticate_imported_sample0_build(
            run_dir, manifest, current, state
        )
    return _normal_build_record(run_dir, sample, current, state)


def validate_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    expected_top = {
        "schema_version": 1,
        "benchmark": "locomo",
        "method": "NativeMem-v8.8+calendar",
        "backbone": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "thinking_off": True,
        "output_dir": str(run_dir),
        "status": "complete",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_top.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise AuditError(f"subscription manifest identity differs: {mismatches}")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError("subscription manifest config is not an object")
    expected_config = {
        "samples": EXPECTED_SAMPLES,
        "model": "gpt-5.5",
        "provider": "chatgpt_pro_subscription",
        "reasoning_effort": "none",
        "chunk_turns": 6,
        "segment": "fixed",
        "single_model_retrieve_answer": True,
        "calendar": True,
        "sample_workers": 2,
        "request_concurrency": 1,
        "proxy_concurrency": 2,
        "retries": 2,
        "sdk_max_retries": 0,
        "sdk_http_timeout_s": 600,
        "max_sessions": None,
        "questions_limit": None,
    }
    config_mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if config_mismatches:
        raise AuditError(
            f"subscription method configuration differs: {config_mismatches}"
        )
    expected_proxy = run_dir / "proxy_requests.jsonl"
    if Path(str(config.get("proxy_request_log", ""))).resolve() != expected_proxy:
        raise AuditError("subscription proxy log path differs from the result root")
    proxy_base = str(config.get("proxy_base_url", ""))
    if not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):\d+/v1", proxy_base):
        raise AuditError("subscription proxy base URL is not a loopback /v1 URL")

    sources = manifest.get("source_hashes")
    expected_sources = runner.source_hashes()
    if sources != expected_sources:
        raise AuditError("subscription source hashes are stale or altered")
    if manifest.get("fingerprint") != runner.fingerprint(config, sources):
        raise AuditError("subscription manifest fingerprint differs")

    health = manifest.get("proxy_health")
    if not isinstance(health, dict):
        raise AuditError("subscription proxy health record is absent")
    if (
        health.get("status") != "ok"
        or health.get("auth_readable") is not True
        or health.get("max_concurrency") != 2
        or health.get("max_attempts") != 2
        or health.get("connect_timeout_s") != 20.0
        or health.get("read_timeout_s") != 150.0
        or health.get("requested_reasoning_effort") != "none"
        or Path(str(health.get("request_log", ""))).resolve() != expected_proxy
        or health.get("code_sha256") != sources.get("src/chatgpt_proxy.py")
    ):
        raise AuditError("subscription proxy health contract differs")


def audit_proxy_window(
    run_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    proxy_path = run_dir / "proxy_requests.jsonl"
    if proxy_path.is_symlink() or not proxy_path.is_file():
        raise AuditError("subscription proxy request log is missing or unsafe")
    try:
        entries = common.load_proxy_window(
            proxy_path, manifest.get("created_at"), manifest.get("finished_at")
        )
    except common.AuditError as exc:
        raise AuditError(str(exc)) from exc
    if not entries:
        raise AuditError("subscription proxy window contains no requests")

    max_attempts = manifest["proxy_health"]["max_attempts"]
    successes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    error_reasoning_observed = 0

    def require_parameters(
        entry: dict[str, Any], index: int, name: str
    ) -> list[str]:
        value = entry.get(name)
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            raise AuditError(
                f"subscription proxy entry {index} has invalid {name}"
            )
        return value

    def require_attempts(entry: dict[str, Any], index: int) -> int:
        attempts = entry.get("attempts")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or not 1 <= attempts <= max_attempts
        ):
            raise AuditError(f"subscription proxy entry {index} attempts differ")
        return attempts

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AuditError(
                f"subscription proxy entry {index} is not an object"
            )
        status = entry.get("status")
        if (
            entry.get("requested_model") != "gpt-5.5"
            or entry.get("requested_reasoning_effort") != "none"
        ):
            raise AuditError(
                f"subscription proxy entry {index} request identity differs"
            )
        require_attempts(entry, index)
        unsupported = require_parameters(entry, index, "unsupported_parameters")
        ignored = require_parameters(entry, index, "ignored_client_parameters")
        if unsupported != [] or ignored != ["max_output_tokens"]:
            raise AuditError(
                f"subscription proxy entry {index} parameter handling differs"
            )
        if status == "error":
            actual_model = entry.get("actual_model")
            actual_effort = entry.get("actual_reasoning_effort")
            reasoning_present = entry.get("reasoning_tokens_present")
            reasoning_tokens = entry.get("reasoning_tokens")
            http_status = entry.get("http_status")
            error_text = entry.get("error")
            http_status_valid = (
                http_status is None
                or (
                    not isinstance(http_status, bool)
                    and isinstance(http_status, int)
                    and 400 <= http_status <= 599
                )
                or (
                    not isinstance(http_status, bool)
                    and isinstance(http_status, int)
                    and 200 <= http_status <= 299
                    and isinstance(error_text, str)
                    and error_text.startswith("ChunkedEncodingError:")
                )
            )
            if (
                actual_model not in (None, "gpt-5.5")
                or actual_effort not in (None, "none")
                or not isinstance(reasoning_present, bool)
                or entry.get("response_id") is not None
                or not isinstance(error_text, str)
                or not error_text.strip()
                or not http_status_valid
            ):
                raise AuditError(
                    f"subscription proxy error {index} evidence differs"
                )
            if reasoning_present:
                if (
                    isinstance(reasoning_tokens, bool)
                    or not isinstance(reasoning_tokens, int)
                    or reasoning_tokens != 0
                ):
                    raise AuditError(
                        f"subscription proxy error {index} reasoning differs"
                    )
                error_reasoning_observed += 1
            elif reasoning_tokens is not None:
                raise AuditError(
                    f"subscription proxy error {index} claims unobserved reasoning"
                )
            errors.append(entry)
            continue
        if status != "success":
            raise AuditError(f"subscription proxy entry {index} has invalid status")
        response_id = entry.get("response_id")
        usage = entry.get("usage")
        completion = (
            usage.get("completion_tokens_details")
            if isinstance(usage, dict) else None
        )
        reasoning_tokens = (
            completion.get("reasoning_tokens")
            if isinstance(completion, dict)
            and "reasoning_tokens" in completion
            else None
        )
        if (
            entry.get("actual_model") != "gpt-5.5"
            or entry.get("actual_reasoning_effort") != "none"
            or reasoning_tokens != 0
            or not isinstance(response_id, str)
            or not response_id
            or response_id in response_ids
            or not isinstance(usage, dict)
        ):
            raise AuditError(f"subscription proxy success {index} differs")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AuditError(
                    f"subscription proxy success {index} has invalid usage.{key}"
                )
        if usage["total_tokens"] != (
            usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            raise AuditError(
                f"subscription proxy success {index} total tokens differ"
            )
        prompt_details = usage.get("prompt_tokens_details")
        cached_tokens = (
            prompt_details.get("cached_tokens")
            if isinstance(prompt_details, dict)
            and "cached_tokens" in prompt_details
            else None
        )
        if (
            isinstance(cached_tokens, bool)
            or not isinstance(cached_tokens, int)
            or not 0 <= cached_tokens <= usage["prompt_tokens"]
        ):
            raise AuditError(
                f"subscription proxy success {index} cached tokens differ"
            )
        response_ids.add(response_id)
        successes.append(entry)

    if not successes:
        raise AuditError("subscription proxy window contains no successful requests")

    stored = manifest.get("proxy_summary")
    if not isinstance(stored, dict):
        raise AuditError("subscription manifest has no final proxy summary")
    computed = {
        "successes": len(successes),
        "errors": len(errors),
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in successes),
        "cached_prompt_tokens": sum(
            row["usage"]["prompt_tokens_details"]["cached_tokens"]
            for row in successes
        ),
        "completion_tokens": sum(
            row["usage"]["completion_tokens"] for row in successes
        ),
        "reasoning_tokens": 0,
        "thinking_off_verified": True,
    }
    if stored != computed:
        raise AuditError("subscription manifest proxy summary differs from its window")

    return {
        "path": str(proxy_path),
        "sha256": common.sha256_file(proxy_path),
        "evidence_scope": "shared_proxy_window",
        "exact_run_response_id_linkage": "unavailable_not_persisted",
        "run_call_count_comparison": "not_performed_without_exact_linkage",
        "global_actual_model_consistency": True,
        "entries": len(entries),
        "successes": len(successes),
        "errors": len(errors),
        "actual_models": ["gpt-5.5"],
        "reasoning_tokens": 0,
        "thinking_off_verified": True,
        "thinking_off_evidence_scope": "all_successful_outputs_explicit",
        "failed_requests_used_as_thinking_off_proof": False,
        "errors_with_observed_reasoning": error_reasoning_observed,
        "errors_without_observed_reasoning": len(errors) - error_reasoning_observed,
        "unsupported_parameters": [],
        "ignored_client_parameters": ["max_output_tokens"],
        "requested_output_limit_enforced": False,
        "unique_response_ids": len(response_ids),
        "physical_http_attempts": sum(
            int(entry.get("attempts", 0)) for entry in entries
        ),
    }


def _audit_unlocked(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = common.read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise AuditError("subscription run manifest is not an object")
    validate_manifest(run_dir, manifest)

    expected = common.expected_records()
    observed: dict[str, dict[str, Any]] = {}
    combined: list[dict[str, Any]] = []
    build_records: list[dict[str, Any]] = []
    sample_audit: dict[str, Any] = {}
    imported_builds = 0
    configured_import = isinstance(
        manifest.get("config", {}).get("imported_sample0_memory"), dict
    )
    for sample in EXPECTED_SAMPLES:
        state = manifest.get("samples", {}).get(str(sample), {})
        if state.get("status") != "complete":
            raise AuditError(f"sample {sample} is not complete in the manifest")
        output = run_dir / f"sample{sample}_questions.json"
        if output.is_symlink() or not output.is_file():
            raise AuditError(f"sample {sample} output is missing or unsafe")
        records = common.read_json(output)
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            raise AuditError(f"sample {sample} output is not a record list")
        builds = [
            record for record in records
            if record.get("question_id") == "_build_stats"
        ]
        questions = [
            record for record in records
            if record.get("question_id") != "_build_stats"
        ]
        if len(builds) != 1:
            raise AuditError(f"sample {sample} has {len(builds)} build records")
        build = resolve_build_record(
            run_dir, manifest, sample, builds[0], state
        )
        build["sample"] = sample
        authenticated_import = sample == 0 and configured_import
        if authenticated_import:
            accounting = build.get("build_accounting")
            if (
                not isinstance(accounting, dict)
                or accounting.get("status") != IMPORTED_BUILD_ACCOUNTING_STATUS
            ):
                raise AuditError(
                    "sample 0 import lacks authenticated build accounting"
                )
            imported_builds += 1
        build_records.append(build)
        combined.append(build)

        expected_ids = [
            question_id for question_id in expected
            if question_id.startswith(f"s{sample}_")
        ]
        if len(questions) != len(expected_ids):
            raise AuditError(f"sample {sample} question count differs")
        for index, record in enumerate(questions):
            question_id = f"s{sample}_q{index}"
            if question_id in observed or question_id not in expected:
                raise AuditError(f"duplicate or unknown question id: {question_id}")
            if record.get("question_id") != question_id:
                raise AuditError(f"sample {sample} question ordering differs")
            reference = expected[question_id]
            for key in ("question", "gold", "category"):
                if record.get(key) != reference[key]:
                    raise AuditError(f"{question_id} has mismatched {key}")
            if not str(record.get("answer", "")).strip():
                raise AuditError(f"{question_id} has an empty answer")
            retrieval = record.get("retrieval")
            if not isinstance(retrieval, dict):
                raise AuditError(f"{question_id} lacks retrieval metadata")
            for key in ("calls", "steps", "tokens_in"):
                value = retrieval.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise AuditError(f"{question_id} has invalid retrieval {key}")
            observed[question_id] = record
            combined.append(record)

        artifact = state.get("artifact")
        if not isinstance(artifact, dict):
            raise AuditError(f"sample {sample} manifest artifact is absent")
        output_hash = common.sha256_file(output)
        if (
            artifact.get("json_sha256") != output_hash
            or artifact.get("questions") != len(questions)
            or artifact.get("empty_answers") != 0
        ):
            raise AuditError(f"sample {sample} manifest artifact differs")
        sample_audit[str(sample)] = {
            "questions": len(questions),
            "output_sha256": output_hash,
            "build_accounting": (
                build["build_accounting"]["status"]
                if authenticated_import else "recorded"
            ),
            "retrieval_calls": sum(
                int(record["retrieval"]["calls"]) for record in questions
            ),
        }

    missing = sorted(set(expected) - set(observed))
    if missing or len(observed) != 1_986:
        raise AuditError(
            f"subscription output is incomplete: observed={len(observed)}, "
            f"first_missing={missing[0] if missing else None}"
        )
    categories = Counter(int(record["category"]) for record in observed.values())
    if dict(sorted(categories.items())) != EXPECTED_CATEGORY_COUNTS:
        raise AuditError(f"category counts differ: {dict(categories)}")
    expected_imported_builds = 1 if configured_import else 0
    if imported_builds != expected_imported_builds:
        raise AuditError(
            "authenticated imported build count differs: "
            f"expected={expected_imported_builds}, found={imported_builds}"
        )

    proxy_window = audit_proxy_window(run_dir, manifest)
    report = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "LoCoMo",
        "method": "NativeMem-v8.8+calendar",
        "model": "gpt-5.5",
        "generation_provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "run_dir": str(run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": common.sha256_file(manifest_path),
        },
        "questions": len(observed),
        "cat1_4": sum(categories[key] for key in (1, 2, 3, 4)),
        "category_counts": dict(sorted(categories.items())),
        "empty_answers": 0,
        "samples": sample_audit,
        "source_hashes_match": True,
        "build_accounting": {
            "records": len(build_records),
            "recorded": len(build_records) - imported_builds,
            "unavailable_imported": imported_builds,
            "no_invented_counts": True,
        },
        "proxy_window": proxy_window,
        "combined_sha256": None,
        "scoring_input": None,
    }
    return combined, report


def audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    try:
        with common.stable_run_lock(run_dir):
            return _audit_unlocked(run_dir)
    except common.AuditError as exc:
        raise AuditError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    audit_path = run_dir / "audit.json"
    audit_path.unlink(missing_ok=True)
    try:
        with common.stable_run_lock(run_dir):
            combined, report = _audit_unlocked(run_dir)
            combined_path = run_dir / "questions_all.json"
            common.atomic_json(combined_path, combined)
            report["combined_sha256"] = common.sha256_file(combined_path)
            report["scoring_input"] = {
                "path": str(combined_path),
                "sha256": report["combined_sha256"],
            }
            common.atomic_json(audit_path, report)
    except (AuditError, common.AuditError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
