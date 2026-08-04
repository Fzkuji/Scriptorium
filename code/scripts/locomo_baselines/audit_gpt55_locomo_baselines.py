#!/usr/bin/env python3
"""Independently audit GPT-5.5 LoCoMo build+retrieval baseline artifacts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_gpt55_locomo_baselines as runner  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


class BaselineAuditError(RuntimeError):
    """Raised when a baseline run has incomplete or inconsistent evidence."""


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineAuditError(f"cannot read valid JSON from {path}: {exc}") from exc


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_time(value: object) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BaselineAuditError(f"invalid timestamp: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise BaselineAuditError(f"timestamp has no timezone: {value!r}")
    return timestamp


def resolve_recorded_path(value: object, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise BaselineAuditError(f"invalid recorded path: {value!r}")
    path = Path(value)
    return path if path.is_absolute() else base / path


def assert_runner_inactive(run_dir: Path):  # type: ignore[no-untyped-def]
    lock_path = run_dir / ".launcher.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise BaselineAuditError(f"baseline launcher is active in {run_dir}") from exc
    return handle


def validate_manifest_contract(
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[runner.MethodSpec, Path, list[dict[str, Any]]]:
    if manifest.get("schema_version") != 1:
        raise BaselineAuditError("run manifest schema_version differs")
    if manifest.get("benchmark") != "LoCoMo":
        raise BaselineAuditError("run manifest benchmark is not LoCoMo")
    scope = manifest.get("scope")
    if scope not in {"formal", "smoke"}:
        raise BaselineAuditError("run manifest scope differs")
    expected_task_scope = (
        "build_and_retrieve_only"
        if scope == "formal"
        else "smoke_build_and_retrieve_only"
    )
    if manifest.get("task_scope") != expected_task_scope:
        raise BaselineAuditError("run manifest task scope differs")
    expected_status = "inputs_complete" if scope == "formal" else "smoke_complete"
    if manifest.get("status") != expected_status:
        raise BaselineAuditError(
            f"run manifest status is {manifest.get('status')!r}, not {expected_status}"
        )
    if manifest.get("output_dir") != str(run_dir):
        raise BaselineAuditError("run manifest output_dir differs")
    method = manifest.get("method")
    if not isinstance(method, str) or method not in runner.METHOD_REGISTRY:
        raise BaselineAuditError(f"run manifest has an unknown method: {method!r}")
    if scope == "formal" and method not in runner.FORMAL_METHODS:
        raise BaselineAuditError("method is outside the preregistered formal matrix")
    spec = runner.METHOD_REGISTRY[method]
    expected_model = "gpt-5.5" if spec.expects_llm else "not_applicable"
    if manifest.get("builder_model") != expected_model:
        raise BaselineAuditError("run manifest builder model differs")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise BaselineAuditError("run manifest config is not an object")
    expected_config_keys = {
        "adapter",
        "adapter_extra_args",
        "answerer",
        "builder_model",
        "dataset",
        "effective_python",
        "effective_python_sha256",
        "expects_llm_calls",
        "formal_method_matrix",
        "gateway_contract",
        "judge",
        "legacy_proxy_log",
        "max_sessions",
        "method",
        "proxy_mode",
        "preregistration",
        "python",
        "python_sha256",
        "questions_limit",
        "retries",
        "sample_workers",
        "samples",
        "scope",
        "structured_preregistration",
        "task_scope",
        "upstream_base",
        "usage_tracking",
    }
    if set(config) != expected_config_keys:
        raise BaselineAuditError("run configuration fields differ from the contract")
    samples = list(range(10)) if scope == "formal" else config.get("samples")
    if not isinstance(samples, list) or not samples or any(
        not isinstance(value, int) or value < 0 or value > 9 for value in samples
    ):
        raise BaselineAuditError("run configuration sample selection is invalid")
    if samples != sorted(set(samples)):
        raise BaselineAuditError("run configuration samples are not unique and sorted")
    expected_builder = runner.builder_contract(spec, config.get("upstream_base", ""))
    if manifest.get("builder_contract") != expected_builder:
        raise BaselineAuditError("run builder contract differs")
    expected_provider = (
        config.get("gateway_contract")
        if spec.expects_llm
        else {"status": "not_applicable_llm_free_method"}
    )
    if manifest.get("provider_contract") != expected_provider:
        raise BaselineAuditError("run provider contract differs")
    preregistration = runner.load_formal_preregistration()
    expected_preregistration_record = runner.preregistration_record(preregistration)
    if manifest.get("preregistration") != expected_preregistration_record:
        raise BaselineAuditError("run preregistration record differs")
    expected_config = {
        "method": method,
        "scope": scope,
        "task_scope": expected_task_scope,
        "samples": samples,
        "builder_model": expected_model,
        "proxy_mode": "managed_exclusive" if spec.expects_llm else "not_applicable",
        "adapter": runner.source_key(spec.adapter),
        "adapter_extra_args": list(spec.extra_args),
        "expects_llm_calls": spec.expects_llm,
        "usage_tracking": spec.usage_tracking,
        "gateway_contract": expected_provider,
        "formal_method_matrix": list(runner.FORMAL_METHODS),
        "structured_preregistration": runner.STRUCTURED_PREREGISTRATION,
        "preregistration": expected_preregistration_record,
        "answerer": "not_run",
        "judge": "not_run",
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise BaselineAuditError(f"run configuration differs: {mismatches}")
    expected_matrix = {
        "formal_methods": list(runner.FORMAL_METHODS),
        "structured_preregistration": runner.STRUCTURED_PREREGISTRATION,
        "other_registered_methods": [
            name for name in runner.METHOD_REGISTRY if name not in runner.FORMAL_METHODS
        ],
        "other_method_scope": "smoke_or_blocked_only",
    }
    if manifest.get("experiment_matrix") != expected_matrix:
        raise BaselineAuditError("run experiment matrix differs")
    for key in ("sample_workers", "retries"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] < 1:
            raise BaselineAuditError(f"run configuration has invalid {key}")
    if config["sample_workers"] > spec.max_sample_workers:
        raise BaselineAuditError("run configuration exceeds method worker limit")
    if scope == "formal":
        if config.get("max_sessions") is not None or config.get("questions_limit") is not None:
            raise BaselineAuditError("formal run contains smoke limits")
    else:
        for key in ("max_sessions", "questions_limit"):
            if not isinstance(config.get(key), int) or config[key] < 1:
                raise BaselineAuditError(f"smoke run has invalid {key}")
    if manifest.get("config_sha256") != runner.canonical_hash(config):
        raise BaselineAuditError("run manifest config hash differs")
    if spec.expects_llm:
        try:
            validated_gateway = runner.flex_evidence.validate_recorded_contract(
                expected_provider
            )
        except runner.flex_evidence.EvidenceError as exc:
            raise BaselineAuditError(str(exc)) from exc
        if config.get("upstream_base") != validated_gateway.get("origin"):
            raise BaselineAuditError("recorded Flex gateway origin differs")
        for key, expected in {
            "gateway_source_sha256": runner.sha256_file(runner.FLEX_GATEWAY),
            "gateway_auditor_sha256": runner.sha256_file(
                runner.FLEX_GATEWAY_AUDITOR
            ),
            "gateway_evidence_source_sha256": runner.sha256_file(
                Path(runner.flex_evidence.__file__)
            ),
        }.items():
            if expected_provider.get(key) != expected:
                raise BaselineAuditError(f"recorded Flex {key} differs")
    elif config.get("upstream_base") != "not_applicable":
        raise BaselineAuditError("LLM-free method records a provider origin")
    if config.get("legacy_proxy_log") is not None:
        raise BaselineAuditError("legacy shared proxy log is not allowed")

    dataset_path = resolve_recorded_path(config.get("dataset"), base=ROOT).resolve()
    if runner.path_identity(dataset_path) != runner.path_identity(runner.DATASET):
        raise BaselineAuditError("run did not use the pinned LoCoMo dataset path")
    dataset = runner.load_dataset(dataset_path)
    data_record = manifest.get("dataset")
    if not isinstance(data_record, dict):
        raise BaselineAuditError("run manifest dataset record is not an object")
    expected_data_record = {
        "path": str(dataset_path),
        "sha256": runner.sha256_file(dataset_path),
        "samples": 10,
        "questions": runner.EXPECTED_QUESTIONS,
        "primary_cat1_4": runner.EXPECTED_PRIMARY,
        "adversarial_cat5": runner.EXPECTED_ADVERSARIAL,
        "categories": {
            str(key): value for key, value in runner.EXPECTED_CATEGORIES.items()
        },
    }
    if data_record != expected_data_record:
        raise BaselineAuditError("run manifest dataset record differs")

    current_sources = runner.compute_source_hashes(spec)
    if manifest.get("source_hashes") != current_sources:
        raise BaselineAuditError("source hashes are missing, stale, or altered")
    python = resolve_recorded_path(config.get("python"), base=ROOT).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise BaselineAuditError(f"recorded Python executable is unavailable: {python}")
    if config.get("python_sha256") != runner.sha256_file(python):
        raise BaselineAuditError("recorded Python executable hash differs")
    effective_python = resolve_recorded_path(
        config.get("effective_python"), base=ROOT
    ).absolute()
    if not effective_python.is_file() or not os.access(effective_python, os.X_OK):
        raise BaselineAuditError(
            f"recorded effective Python is unavailable: {effective_python}"
        )
    if config.get("effective_python_sha256") != runner.sha256_file(effective_python):
        raise BaselineAuditError("recorded effective Python hash differs")
    current_runtime = runner.runtime_fingerprint(spec, python, effective_python)
    if manifest.get("runtime") != current_runtime:
        raise BaselineAuditError("runtime package/source fingerprint differs")
    expected_fingerprint = runner.canonical_hash(
        {
            "config_sha256": manifest["config_sha256"],
            "dataset_sha256": expected_data_record["sha256"],
            "source_hashes": current_sources,
            "runtime": current_runtime,
        }
    )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise BaselineAuditError("run fingerprint differs")

    evaluation = manifest.get("evaluation")
    if evaluation != {
        "answerer": "not_run",
        "primary_judge": "not_run",
        "secondary_judge": "not_run",
    }:
        raise BaselineAuditError("run manifest claims an out-of-scope evaluation")
    created_at = parse_time(manifest.get("created_at"))
    if runner.parse_preregistration_time(preregistration["frozen_at"]) > created_at:
        raise BaselineAuditError("formal matrix was preregistered after run creation")
    finished_at = parse_time(manifest.get("finished_at"))
    if finished_at < created_at:
        raise BaselineAuditError("run manifest timestamps are reversed")
    return spec, dataset_path, dataset


def audit_samples(
    run_dir: Path,
    manifest: dict[str, Any],
    dataset: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    states = manifest.get("samples")
    selected = manifest["config"]["samples"]
    if not isinstance(states, dict) or set(states) != {str(i) for i in selected}:
        raise BaselineAuditError("run manifest sample states differ from configured scope")
    builds: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for sample in selected:
        dataset_item = dict(dataset[sample])
        if manifest["scope"] == "smoke":
            dataset_item["qa"] = list(
                dataset_item["qa"][: manifest["config"]["questions_limit"]]
            )
        checkpoint_path = run_dir / "checkpoints" / f"sample{sample}.json"
        checkpoint = read_json(checkpoint_path)
        if not isinstance(checkpoint, dict) or checkpoint.get("status") != "complete":
            raise BaselineAuditError(f"sample {sample} checkpoint is not complete")
        if checkpoint != states[str(sample)]:
            raise BaselineAuditError(
                f"sample {sample} checkpoint differs from run manifest"
            )
        if checkpoint.get("fingerprint") != manifest.get("fingerprint"):
            raise BaselineAuditError(f"sample {sample} fingerprint differs")
        expected_frozen_hashes = {
            "config_sha256": manifest["config_sha256"],
            "dataset_sha256": manifest["dataset"]["sha256"],
            "source_hashes_sha256": runner.canonical_hash(manifest["source_hashes"]),
        }
        if checkpoint.get("frozen_hashes") != expected_frozen_hashes:
            raise BaselineAuditError(f"sample {sample} frozen hashes differ")
        if checkpoint.get("sample") != sample:
            raise BaselineAuditError(f"sample {sample} checkpoint index differs")
        if not isinstance(checkpoint.get("attempts"), int) or checkpoint["attempts"] < 1:
            raise BaselineAuditError(f"sample {sample} attempt count is invalid")
        if checkpoint.get("adapter_extra_args") != manifest["config"]["adapter_extra_args"]:
            raise BaselineAuditError(f"sample {sample} adapter arguments differ")

        output = checkpoint.get("output")
        log = checkpoint.get("log")
        if not isinstance(output, dict) or output.get("path") != f"sample{sample}_questions.json":
            raise BaselineAuditError(f"sample {sample} output path differs")
        if not isinstance(log, dict):
            raise BaselineAuditError(f"sample {sample} log record is missing")
        output_path = run_dir / output["path"]
        log_path = resolve_recorded_path(log.get("path"), base=run_dir).resolve()
        try:
            log_path.relative_to(run_dir)
        except ValueError as exc:
            raise BaselineAuditError(f"sample {sample} log escapes run directory") from exc
        if not output_path.is_file() or output.get("sha256") != runner.sha256_file(output_path):
            raise BaselineAuditError(f"sample {sample} output hash differs")
        if output.get("bytes") != output_path.stat().st_size:
            raise BaselineAuditError(f"sample {sample} output size differs")
        if not log_path.is_file() or log.get("sha256") != runner.sha256_file(log_path):
            raise BaselineAuditError(f"sample {sample} log hash differs")
        if log.get("bytes") != log_path.stat().st_size:
            raise BaselineAuditError(f"sample {sample} log size differs")

        try:
            validation = runner.validate_sample_file(
                output_path,
                dataset_item,
                sample,
                log_path=log_path,
                expected_builder=manifest["builder_contract"],
            )
        except runner.BaselineRunError as exc:
            raise BaselineAuditError(
                f"sample {sample} independent validation failed: {exc}"
            ) from exc
        if checkpoint.get("validation") != validation:
            raise BaselineAuditError(f"sample {sample} validation summary differs")
        try:
            runner.validate_diagnostic_records(
                checkpoint.get("diagnostics"),
                run_dir,
                required=runner.METHOD_REGISTRY[manifest["method"]].requires_diagnostics,
            )
        except runner.BaselineRunError as exc:
            raise BaselineAuditError(f"sample {sample} diagnostics failed: {exc}") from exc
        rows = read_json(output_path)
        build = dict(rows[0])
        build["sample_index"] = sample
        builds.append(build)
        questions.extend(rows[1:])
        report[str(sample)] = {
            "questions": validation["questions"],
            "num_memories": validation["num_memories"],
            "build_calls": validation["build_calls"],
            "retrieval_calls": validation["retrieval_calls"],
            "output_sha256": output["sha256"],
            "log_sha256": log["sha256"],
        }
    expected_questions = sum(
        len(dataset[sample]["qa"])
        if manifest["scope"] == "formal"
        else min(
            len(dataset[sample]["qa"]), manifest["config"]["questions_limit"]
        )
        for sample in selected
    )
    if len(questions) != expected_questions:
        raise BaselineAuditError(
            f"sample outputs contain {len(questions)} questions, expected {expected_questions}"
        )
    ids = [record.get("question_id") for record in questions]
    if len(set(ids)) != expected_questions:
        raise BaselineAuditError("sample outputs contain duplicate question IDs")
    categories = Counter(int(record["category"]) for record in questions)
    if manifest["scope"] == "formal" and dict(sorted(categories.items())) != runner.EXPECTED_CATEGORIES:
        raise BaselineAuditError("sample output category inventory differs")
    if spec := runner.METHOD_REGISTRY.get(manifest.get("method", "")):
        if spec.expects_llm and spec.usage_tracking == "in_process":
            no_build_calls = [
                sample
                for sample, value in report.items()
                if value["build_calls"] <= 0
            ]
            if no_build_calls:
                raise BaselineAuditError(
                    f"samples lack in-process build-call evidence: {no_build_calls}"
                )
    return builds, questions, report


def audit_assembly(
    run_dir: Path,
    manifest: dict[str, Any],
    spec: runner.MethodSpec,
    dataset_path: Path,
    dataset: list[dict[str, Any]],
    builds: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    questions_path = run_dir / "questions.json"
    inputs_manifest_path = run_dir / "inputs_manifest.json"
    actual = read_json(questions_path)
    expected = [*builds, *questions]
    if actual != expected:
        raise BaselineAuditError("questions.json differs from independently assembled records")
    if len(actual) != 10 + runner.EXPECTED_QUESTIONS:
        raise BaselineAuditError("questions.json record count differs")

    assembly_manifest = read_json(inputs_manifest_path)
    if not isinstance(assembly_manifest, dict):
        raise BaselineAuditError("inputs_manifest.json is not an object")
    expected_header = {
        "schema_version": 1,
        "benchmark": "LoCoMo",
        "method": manifest["method"],
        "status": "inputs_complete",
    }
    for key, value in expected_header.items():
        if assembly_manifest.get(key) != value:
            raise BaselineAuditError(f"assembler manifest differs in {key}")
    parse_time(assembly_manifest.get("created_at"))
    expected_dataset = {
        "path": str(dataset_path),
        "sha256": runner.sha256_file(dataset_path),
        "conversations": 10,
        "questions": runner.EXPECTED_QUESTIONS,
        "primary_questions_cat1_4": runner.EXPECTED_PRIMARY,
        "adversarial_questions_cat5": runner.EXPECTED_ADVERSARIAL,
        "categories": {
            str(key): value for key, value in runner.EXPECTED_CATEGORIES.items()
        },
    }
    if assembly_manifest.get("dataset") != expected_dataset:
        raise BaselineAuditError("assembler manifest dataset record differs")
    if assembly_manifest.get("adapter") != {
        "path": str(spec.adapter.resolve()),
        "sha256": runner.sha256_file(spec.adapter),
    }:
        raise BaselineAuditError("assembler manifest adapter record differs")
    if assembly_manifest.get("assembler") != {
        "path": str(runner.ASSEMBLER.resolve()),
        "sha256": runner.sha256_file(runner.ASSEMBLER),
    }:
        raise BaselineAuditError("assembler manifest source record differs")
    if assembly_manifest.get("contract") != {
        "path": str(runner.CONTRACT.resolve()),
        "sha256": runner.sha256_file(runner.CONTRACT),
    }:
        raise BaselineAuditError("assembler manifest contract record differs")
    if assembly_manifest.get("builder") != manifest["builder_contract"]:
        raise BaselineAuditError("assembler manifest builder record differs")
    inputs = assembly_manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 10:
        raise BaselineAuditError("assembler manifest does not contain ten inputs")
    for sample, (record, dataset_item) in enumerate(zip(inputs, dataset)):
        path = run_dir / f"sample{sample}_questions.json"
        expected_input = {
            "sample_index": sample,
            "path": str(path),
            "sha256": runner.sha256_file(path),
            "questions": len(dataset_item["qa"]),
        }
        if record != expected_input:
            raise BaselineAuditError(f"assembler input record {sample} differs")
    expected_output = {
        "path": str(questions_path),
        "sha256": runner.sha256_file(questions_path),
        "records": 10 + runner.EXPECTED_QUESTIONS,
        "build_records": 10,
        "question_records": runner.EXPECTED_QUESTIONS,
    }
    if assembly_manifest.get("output") != expected_output:
        raise BaselineAuditError("assembler manifest output record differs")
    if assembly_manifest.get("evaluation") != {
        "answerer": "pending",
        "primary_judge": "pending",
        "secondary_judge": "pending",
    }:
        raise BaselineAuditError("assembler manifest evaluation state differs")

    assembly = manifest.get("assembly")
    if not isinstance(assembly, dict) or assembly.get("status") != "inputs_complete":
        raise BaselineAuditError("run manifest assembly is not inputs_complete")
    expected_run_assembly = {
        "status": "inputs_complete",
        "questions": {
            "path": questions_path.name,
            "sha256": runner.sha256_file(questions_path),
            "records": len(actual),
            "question_records": runner.EXPECTED_QUESTIONS,
            "primary_cat1_4": runner.EXPECTED_PRIMARY,
            "adversarial_cat5": runner.EXPECTED_ADVERSARIAL,
        },
        "manifest": {
            "path": inputs_manifest_path.name,
            "sha256": runner.sha256_file(inputs_manifest_path),
        },
        "log": assembly.get("log"),
    }
    assembly_log = assembly.get("log")
    if not isinstance(assembly_log, dict) or assembly_log.get("path") != "assembly.log":
        raise BaselineAuditError("run manifest assembly log record differs")
    log_path = run_dir / "assembly.log"
    if not log_path.is_file() or assembly_log.get("sha256") != runner.sha256_file(log_path):
        raise BaselineAuditError("assembly log hash differs")
    if "returncode=0" not in log_path.read_text(encoding="utf-8", errors="replace"):
        raise BaselineAuditError("assembly log does not record returncode=0")
    if assembly != expected_run_assembly:
        raise BaselineAuditError("run manifest assembly hashes or counts differ")
    return {
        "questions_path": str(questions_path),
        "questions_sha256": expected_output["sha256"],
        "inputs_manifest_path": str(inputs_manifest_path),
        "inputs_manifest_sha256": runner.sha256_file(inputs_manifest_path),
        "records": len(actual),
        "questions": runner.EXPECTED_QUESTIONS,
        "primary_cat1_4": runner.EXPECTED_PRIMARY,
        "adversarial_cat5": runner.EXPECTED_ADVERSARIAL,
    }


def audit_proxy_evidence(
    manifest: dict[str, Any], spec: runner.MethodSpec
) -> dict[str, Any]:
    evidence = manifest.get("proxy_evidence")
    if not isinstance(evidence, dict):
        raise BaselineAuditError("run manifest proxy evidence is missing")
    expected_mode = "managed_exclusive" if spec.expects_llm else "not_applicable"
    if evidence.get("mode") != expected_mode:
        raise BaselineAuditError("proxy evidence mode differs")
    if evidence.get("exact_run_linkage") is not bool(spec.expects_llm):
        raise BaselineAuditError("proxy exact-linkage declaration differs")
    invocations = evidence.get("invocations")
    if not isinstance(invocations, list):
        raise BaselineAuditError("proxy invocation records are invalid")
    if not spec.expects_llm:
        if invocations:
            raise BaselineAuditError("LLM-free method has proxy invocations")
        return {
            "mode": "not_applicable",
            "exact_run_linkage": False,
            "entries": 0,
        }
    if not invocations:
        raise BaselineAuditError("LLM method has no exclusive proxy invocation")
    run_dir = Path(manifest["output_dir"]).resolve()
    proxy_root = run_dir / "proxy"
    successes: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    gateway_request_ids: set[str] = set()
    run_ids: set[str] = set()
    provider_contract = manifest.get("provider_contract")
    if not isinstance(provider_contract, dict):
        raise BaselineAuditError("run provider contract is missing")
    expected_upstream = provider_contract.get("origin")
    expected_gateway_root = provider_contract.get("result_root")
    for index, invocation in enumerate(invocations):
        status = invocation.get("status") if isinstance(invocation, dict) else None
        if not isinstance(invocation, dict) or status not in {
            "captured",
            "interrupted_captured",
        }:
            raise BaselineAuditError(f"proxy invocation {index} is not captured")
        if invocation.get("exact_run_linkage") is not True:
            raise BaselineAuditError(f"proxy invocation {index} lacks exact linkage")
        if invocation.get("wrapper_sha256") != runner.sha256_file(runner.RUN_PROXY):
            raise BaselineAuditError(f"proxy invocation {index} wrapper hash differs")
        if invocation.get("gateway_source_sha256") != runner.sha256_file(
            runner.FLEX_GATEWAY
        ):
            raise BaselineAuditError(f"proxy invocation {index} gateway source differs")
        if (
            invocation.get("gateway_root_marker_sha256")
            != provider_contract.get("root_marker_sha256")
            or invocation.get("gateway_result_root") != expected_gateway_root
            or invocation.get("provider_model") != runner.FLEX_PROVIDER_MODEL
            or invocation.get("service_tier") != runner.FLEX_SERVICE_TIER
        ):
            raise BaselineAuditError(f"proxy invocation {index} gateway contract differs")
        if invocation.get("upstream") != expected_upstream:
            raise BaselineAuditError(f"proxy invocation {index} upstream differs")
        run_id = invocation.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in run_ids
        ):
            raise BaselineAuditError(f"proxy invocation {index} run_id is invalid")
        run_ids.add(run_id)
        path_values = {
            key: invocation.get(key) for key in ("log", "process_log", "ready")
        }
        if any(
            not isinstance(value, str) or Path(value).is_absolute()
            for value in path_values.values()
        ):
            raise BaselineAuditError(
                f"proxy invocation {index} paths must be relative"
            )
        log_path = resolve_recorded_path(path_values["log"], base=run_dir).resolve()
        process_log = resolve_recorded_path(
            path_values["process_log"], base=run_dir
        ).resolve()
        ready_path = resolve_recorded_path(
            path_values["ready"], base=run_dir
        ).resolve()
        paths = (log_path, process_log, ready_path)
        if len({runner.path_identity(path) for path in paths}) != len(paths):
            raise BaselineAuditError(f"proxy invocation {index} paths collide")
        if any(not runner.is_within(path, proxy_root) for path in paths):
            raise BaselineAuditError(f"proxy invocation {index} path escapes proxy directory")
        if (
            not log_path.is_file()
            or invocation.get("log_sha256") != runner.sha256_file(log_path)
        ):
            raise BaselineAuditError(f"proxy invocation {index} log hash differs")
        if invocation.get("log_bytes") != log_path.stat().st_size:
            raise BaselineAuditError(f"proxy invocation {index} log size differs")
        if (
            not process_log.is_file()
            or invocation.get("process_log_sha256")
            != runner.sha256_file(process_log)
            or invocation.get("process_log_bytes") != process_log.stat().st_size
        ):
            raise BaselineAuditError(f"proxy invocation {index} process log differs")
        if (
            not ready_path.is_file()
            or invocation.get("ready_sha256") != runner.sha256_file(ready_path)
            or invocation.get("ready_bytes") != ready_path.stat().st_size
        ):
            raise BaselineAuditError(f"proxy invocation {index} ready record differs")
        ready = read_json(ready_path)
        if not isinstance(ready, dict):
            raise BaselineAuditError(f"proxy invocation {index} ready JSON is invalid")
        port = ready.get("port")
        expected_base = (
            f"http://127.0.0.1:{port}/v1"
            if isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536
            else None
        )
        if (
            ready.get("run_id") != run_id
            or ready.get("pid") != invocation.get("pid")
            or ready.get("upstream") != expected_upstream
            or Path(str(ready.get("log"))).resolve() != log_path
            or ready.get("base_url") != expected_base
            or invocation.get("base_url") != expected_base
        ):
            raise BaselineAuditError(f"proxy invocation {index} ready metadata differs")
        ready_started = parse_time(ready.get("started_at"))
        started = parse_time(invocation.get("started_at"))
        finished = parse_time(invocation.get("finished_at"))
        if not ready_started <= started <= finished:
            raise BaselineAuditError(f"proxy invocation {index} timestamps are reversed")
        if status == "captured":
            if not isinstance(invocation.get("returncode"), int):
                raise BaselineAuditError(
                    f"proxy invocation {index} return code is invalid"
                )
        elif invocation.get("recovery_action") not in {
            "original_process_not_running",
            "terminated_original_process",
            "original_pid_reused",
        }:
            raise BaselineAuditError(
                f"proxy invocation {index} interruption recovery differs"
            )
        health = invocation.get("health")
        upstream_health = health.get("upstream_health") if isinstance(health, dict) else None
        if (
            not isinstance(health, dict)
            or health.get("status") != "ok"
            or health.get("run_id") != run_id
            or health.get("exclusive_log") != str(log_path)
            or health.get("upstream") != expected_upstream
            or not isinstance(upstream_health, dict)
            or upstream_health.get("status") != "ok"
            or upstream_health.get("schema") != runner.FLEX_HEALTH_SCHEMA
            or upstream_health.get("requested_model")
            != runner.FLEX_REQUESTED_MODEL
            or upstream_health.get("provider_model")
            != runner.FLEX_PROVIDER_MODEL
            or upstream_health.get("service_tier")
            != runner.FLEX_SERVICE_TIER
            or not isinstance(upstream_health.get("budget"), dict)
            or upstream_health["budget"].get("max_cost_usd")
            != provider_contract.get("max_cost_usd")
        ):
            raise BaselineAuditError(f"proxy invocation {index} health evidence differs")
        entries: list[dict[str, Any]] = []
        with log_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BaselineAuditError(
                        f"proxy invocation {index} has invalid line {line_number}"
                    ) from exc
                if not isinstance(entry, dict) or entry.get("run_id") != run_id:
                    raise BaselineAuditError(f"proxy invocation {index} run linkage differs")
                request_started = parse_time(entry.get("started_at"))
                request_finished = parse_time(entry.get("finished_at"))
                if not started <= request_started <= request_finished <= finished:
                    raise BaselineAuditError(
                        f"proxy invocation {index} request timestamps differ"
                    )
                if entry.get("status") != "success" or entry.get("http_status") != 200:
                    raise BaselineAuditError(f"proxy invocation {index} contains a request error")
                if (
                    entry.get("requested_model") != "gpt-5.5"
                    or entry.get("actual_model") != "gpt-5.5"
                    or entry.get("requested_service_tier")
                    not in (None, runner.FLEX_SERVICE_TIER)
                    or entry.get("provider_actual_model")
                    != runner.FLEX_PROVIDER_MODEL
                    or entry.get("service_tier") != runner.FLEX_SERVICE_TIER
                ):
                    raise BaselineAuditError(
                        f"proxy invocation {index} model/tier differs from GPT-5.5 Flex"
                    )
                gateway_request_id = entry.get("gateway_request_id")
                if (
                    not isinstance(gateway_request_id, str)
                    or not gateway_request_id
                    or gateway_request_id in gateway_request_ids
                ):
                    raise BaselineAuditError(
                        f"proxy invocation {index} gateway request ID is invalid"
                    )
                gateway_request_ids.add(gateway_request_id)
                response_id = entry.get("response_id")
                if not isinstance(response_id, str) or not response_id or response_id in response_ids:
                    raise BaselineAuditError(f"proxy invocation {index} response ID is invalid")
                response_ids.add(response_id)
                request_sha256 = entry.get("request_sha256")
                if (
                    not isinstance(request_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
                    or entry.get("error") is not None
                ):
                    raise BaselineAuditError(f"proxy invocation {index} request hash is missing")
                for key in (
                    "gateway_request_sha256",
                    "provider_request_sha256",
                ):
                    if not isinstance(entry.get(key), str) or re.fullmatch(
                        r"[0-9a-f]{64}", entry[key]
                    ) is None:
                        raise BaselineAuditError(
                            f"proxy invocation {index} {key} is invalid"
                        )
                entries.append(entry)
        if invocation.get("provider_window_error") is not None:
            raise BaselineAuditError(
                f"proxy invocation {index} Flex provider window failed"
            )
        try:
            window_report = runner.flex_evidence.audit_window(
                invocation.get("provider_window"),
                consumer_records=entries,
            )
        except runner.flex_evidence.EvidenceError as exc:
            raise BaselineAuditError(
                f"proxy invocation {index} Flex evidence failed: {exc}"
            ) from exc
        window_contract = invocation["provider_window"].get("contract", {})
        if (
            window_contract.get("result_root") != expected_gateway_root
            or window_contract.get("origin") != expected_upstream
            or window_contract.get("root_marker_sha256")
            != provider_contract.get("root_marker_sha256")
        ):
            raise BaselineAuditError(
                f"proxy invocation {index} provider-window root differs"
            )
        successes.extend(entries)
        reports.append({
            "run_id": run_id,
            "status": status,
            "log": str(log_path),
            "log_sha256": invocation["log_sha256"],
            "requests": len(entries),
            "flex_gateway": window_report,
        })
    if not successes:
        raise BaselineAuditError("LLM method has no successful exclusive proxy requests")
    return {
        "mode": "managed_exclusive",
        "exact_run_linkage": True,
        "actual_model_verification": "per_request",
        "provider_model_verification": "per_gateway_request_id",
        "service_tier_verification": "per_gateway_request_id",
        "billing_verification": "bounded_append_only_gateway_segment",
        "entries": len(successes),
        "successes": len(successes),
        "errors": 0,
        "invocations": reports,
    }


def _audit_unlocked(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise BaselineAuditError("run_manifest.json is not an object")
    spec, dataset_path, dataset = validate_manifest_contract(run_dir, manifest)
    builds, questions, sample_report = audit_samples(run_dir, manifest, dataset)
    if manifest["scope"] == "formal":
        assembly_report = audit_assembly(
            run_dir,
            manifest,
            spec,
            dataset_path,
            dataset,
            builds,
            questions,
        )
    else:
        if manifest.get("assembly") != {"status": "not_applicable_smoke_scope"}:
            raise BaselineAuditError("smoke run assembly state differs")
        forbidden = [
            run_dir / "questions.json",
            run_dir / "inputs_manifest.json",
            run_dir / "assembly.log",
        ]
        if any(path.exists() for path in forbidden):
            raise BaselineAuditError("smoke run contains formal assembly artifacts")
        assembly_report = {"status": "not_applicable_smoke_scope"}
    proxy_report = audit_proxy_evidence(manifest, spec)
    tracked_calls = sum(
        value["build_calls"] + value["retrieval_calls"]
        for value in sample_report.values()
    )
    if spec.expects_llm and spec.usage_tracking == "in_process":
        if proxy_report["successes"] < tracked_calls:
            raise BaselineAuditError(
                "exclusive proxy calls do not cover in-process tracked calls"
            )
        proxy_report["call_count_comparison"] = {
            "mode": "accumulated_proxy_calls_cover_final_artifact",
            "tracked_final_artifact_calls": tracked_calls,
            "exclusive_proxy_successes": proxy_report["successes"],
            "exact_match": proxy_report["successes"] == tracked_calls,
        }
    elif spec.expects_llm:
        proxy_report["call_count_comparison"] = {
            "mode": "not_available_subprocess_untracked",
            "exclusive_proxy_successes": proxy_report["successes"],
        }
    else:
        proxy_report["call_count_comparison"] = {"mode": "not_applicable"}
    categories = Counter(int(record["category"]) for record in questions)
    primary = sum(categories.get(category, 0) for category in (1, 2, 3, 4))
    adversarial = categories.get(5, 0)
    return {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "LoCoMo",
        "method": manifest["method"],
        "task_scope": manifest["task_scope"],
        "builder_model": manifest["builder_model"],
        "run_dir": str(run_dir),
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": runner.sha256_file(manifest_path),
            "fingerprint": manifest["fingerprint"],
        },
        "source_hashes_match": True,
        "dataset_sha256": manifest["dataset"]["sha256"],
        "preregistration": manifest["preregistration"],
        "experiment_matrix": manifest["experiment_matrix"],
        "samples": sample_report,
        "scope": {
            "kind": manifest["scope"],
            "samples": manifest["config"]["samples"],
            "sample_count": len(manifest["config"]["samples"]),
            "questions": len(questions),
            "primary_cat1_4": primary,
            "adversarial_cat5": adversarial,
            "categories": {
                str(key): value for key, value in sorted(categories.items())
            },
        },
        "assembly": assembly_report,
        "proxy_evidence": proxy_report,
        "evaluation": {
            "answerer": "not_run",
            "primary_judge": "not_run",
            "secondary_judge": "not_run",
        },
    }


def audit(run_dir: Path) -> dict[str, Any]:
    """Audit a run while holding the same non-blocking lock as the launcher."""
    run_dir = run_dir.expanduser().resolve()
    lock_handle = assert_runner_inactive(run_dir)
    try:
        return _audit_unlocked(run_dir)
    finally:
        lock_handle.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit one GPT-5.5 LoCoMo build+retrieval baseline run"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    audit_path = run_dir / "audit.json"
    lock_handle = None
    try:
        lock_handle = assert_runner_inactive(run_dir)
        try:
            audit_path.unlink()
        except FileNotFoundError:
            pass
        report = _audit_unlocked(run_dir)
        atomic_json(audit_path, report)
    except (BaselineAuditError, runner.BaselineRunError, OSError) as exc:
        try:
            audit_path.unlink()
        except FileNotFoundError:
            pass
        parser.error(str(exc))
    finally:
        if lock_handle is not None:
            lock_handle.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
