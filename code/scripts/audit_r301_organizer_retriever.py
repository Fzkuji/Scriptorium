#!/usr/bin/env python3
"""Independently audit an R301 organizer x retriever artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import tempfile
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import controlled_locomo_answer_contract as answer_contract  # noqa: E402
from src.evaluation.durable_model_ledger import (  # noqa: E402
    ledger_state,
    normalize_usage,
    read_ledger,
    sha256_file,
)
from src.evaluation.metrics import f1_set  # noqa: E402
from src.evaluation.visible_token_audit import (  # noqa: E402
    audit_visible_token_trace,
)
from src.evaluation.visible_token_budget import (  # noqa: E402
    DeliveryResult,
    snapshot_memory_path,
)
from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402
from src import openrouter_gateway_evidence  # noqa: E402
from src import v8_memory  # noqa: E402


RUN_SCHEMA = "nativemem.r301-run.v1"
ORGANIZER_SCHEMA = "nativemem.r301-organizer.v1"
QUESTION_SCHEMA = "nativemem.r301-question.v1"
ANALYSIS_SCHEMA = "nativemem.r301-interaction.v1"
PREREG_SCHEMA = "nativemem.r301-preregistration.v1"
CELLS = ("O55-R55", "O4o-R55", "O55-R4o", "O4o-R4o")
ORGANIZERS = ("O55", "O4o")
RETRIEVERS = ("R55", "R4o")
REPLICATES = (1, 2, 3)
VISIBLE_BUDGET = 20_000
PREREG = ROOT / "paper/refine-logs/R301_PREREGISTRATION.json"
M4_PREREG = ROOT / "paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json"
EXPECTED_M4_SHA256 = (
    "b706535a61972c368547caf6611bac397959ae3fbfe71142adb8bfb95b449c22"
)
EXPECTED_DATA_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
EXPECTED_MAPPING_SHA256 = (
    "3291df579bc20d0678f99b3361ac6d287beee583e4d3dff93b996f4522e56318"
)


class AuditFailure(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def value_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing JSONL: {path}")
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), f"incomplete JSONL: {path}")
    output = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditFailure(f"invalid JSONL {path}:{line_number}") from exc
        require(isinstance(value, dict), f"non-object JSONL: {path}:{line_number}")
        output.append(value)
    return output


def assert_safe_tree(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"invalid tree: {root}")
    names: dict[str, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        key = relative.casefold()
        require(
            key not in names or names[key] == relative,
            f"case-colliding paths: {names.get(key)} / {relative}",
        )
        names[key] = relative
        require(not path.is_symlink(), f"symlink is forbidden: {relative}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            require(stat.st_nlink == 1, f"hardlink is forbidden: {relative}")
            inode = (stat.st_dev, stat.st_ino)
            require(inode not in inodes, f"shared inode: {inodes.get(inode)} / {relative}")
            inodes[inode] = relative
        else:
            require(path.is_dir(), f"special node is forbidden: {relative}")


def tree_descriptor(root: Path) -> dict[str, Any]:
    assert_safe_tree(root)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat(follow_symlinks=False).st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {"file_count": len(files), "tree_sha256": value_sha256(files)}


def validate_preregistration() -> dict[str, Any]:
    prereg = load_json(PREREG)
    require(
        prereg.get("schema") == PREREG_SCHEMA and prereg.get("status") == "frozen",
        "R301 preregistration identity differs",
    )
    content = dict(prereg)
    recorded = content.pop("protocol_content_sha256", None)
    require(recorded == value_sha256(content), "R301 preregistration hash differs")
    binding = prereg.get("m4_comparison_preregistration")
    require(
        binding
        == {
            "path": "paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json",
            "sha256": EXPECTED_M4_SHA256,
            "section": "m5_interaction",
        },
        "M4 binding differs",
    )
    require(sha256_file(M4_PREREG) == EXPECTED_M4_SHA256, "M4 prereg hash differs")
    m4 = load_json(M4_PREREG)
    require(
        m4.get("status") == "frozen_before_formal_scores"
        and m4.get("m5_interaction")
        == {
            "estimand": "(O55_R55 - O4o_R55) - (O55_R4o - O4o_R4o)",
            "organizer_replicates_minimum": 3,
            "resampling_unit": "LoCoMo conversation",
            "confidence": 0.95,
            "support_rule": (
                "matched organizer-effect direction in both retriever columns "
                "and clustered interaction interval excludes zero"
            ),
            "failure_rule": (
                "otherwise report organizer or retriever capacity effects only"
            ),
        },
        "M4 m5_interaction differs",
    )
    common = m4.get("common_protocol")
    require(
        isinstance(common, dict)
        and common.get("continuous_estimand")
        == "question_weighted_mean_left_minus_right"
        and common.get("clustered_bootstrap_repetitions") == 10_000
        and common.get("clustered_bootstrap_seed") == 20_260_714
        and common.get("confidence") == 0.95
        and common.get("pairing")
        == "exact question_id; LoCoMo resampling cluster is conversation"
        and common.get("missingness")
        == "no imputation; incomplete or unequal question sets invalidate a formal comparison",
        "M4 common clustered-comparison protocol differs",
    )
    require(tuple(prereg.get("cells", ())) == CELLS, "cell matrix differs")
    require(
        tuple(prereg.get("organizer_replicates", ())) == REPLICATES,
        "replicate matrix differs",
    )
    require(
        prereg.get("visible_token_gate", {}).get("hard_cap_tokens")
        == VISIBLE_BUDGET,
        "visible-token cap differs",
    )
    require(
        prereg.get("visible_token_gate", {}).get("tokenizer")
        == answer_contract.formal_token_counter().identity,
        "tokenizer preregistration differs",
    )
    return prereg


def source_hashes() -> dict[str, str]:
    paths = {
        "runner": ROOT / "scripts/run_r301_organizer_retriever.py",
        "auditor": Path(__file__).resolve(),
        "proxy": ROOT / "scripts/controlled_r301_model_proxy.py",
        "gpt55_proxy": ROOT / "scripts/controlled_gpt55_run_proxy.py",
        "preregistration": PREREG,
        "m4_comparison_preregistration": M4_PREREG,
        "answer_contract": Path(answer_contract.__file__).resolve(),
        "visible_token_budget": ROOT / "src/evaluation/visible_token_budget.py",
        "visible_token_audit": ROOT / "src/evaluation/visible_token_audit.py",
        "durable_model_ledger": ROOT / "src/evaluation/durable_model_ledger.py",
        "openrouter_gateway_evidence": ROOT / "src/openrouter_gateway_evidence.py",
        "openai_gpt55_flex_gateway": ROOT / "src/openai_gpt55_flex_gateway.py",
        "openai_gpt55_flex_gateway_evidence": (
            ROOT / "src/openai_gpt55_flex_gateway_evidence.py"
        ),
        "metrics": ROOT / "src/evaluation/metrics.py",
        "r207_runner": ROOT / "scripts/run_r207_path_control.py",
        "r207_auditor": ROOT / "scripts/audit_r207_path_control.py",
        "v8_memory": ROOT / "src/v8_memory.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_bank(bank: Mapping[str, Any], sample: int) -> None:
    require(
        bank.get("schema") == "nativemem.canonical-entry-bank.v1"
        and bank.get("sample") == sample,
        f"sample {sample}: bank identity differs",
    )
    entries = bank.get("entries")
    require(isinstance(entries, list) and entries, f"sample {sample}: empty bank")
    keys = {
        "entry_id",
        "sample",
        "session",
        "chunk",
        "ordinal",
        "when",
        "summary",
        "summary_inline",
        "dia_ids",
    }
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == keys, "entry keys differ")
        entry_id = entry.get("entry_id")
        require(
            isinstance(entry_id, str) and entry_id and entry_id not in seen,
            "entry ID differs",
        )
        seen.add(entry_id)
        require(entry.get("sample") == sample, "entry sample differs")
        require(
            isinstance(entry.get("dia_ids"), list)
            and all(re.fullmatch(r"D\d+:\d+", item) for item in entry["dia_ids"]),
            "entry source IDs differ",
        )
    require(
        bank.get("entry_count") == len(entries)
        and bank.get("entries_sha256") == value_sha256(entries),
        "bank count/hash differs",
    )


def build_turn_index(conversation: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    output = {}
    for key, turns in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if match is None or not isinstance(turns, list):
            continue
        date = str(conversation.get(f"session_{match.group(1)}_date_time", ""))
        for turn in turns:
            require(isinstance(turn, dict), "conversation turn is invalid")
            dia_id = str(turn.get("dia_id", ""))
            require(
                re.fullmatch(r"D\d+:\d+", dia_id) is not None and dia_id not in output,
                "conversation source ID differs",
            )
            output[dia_id] = {
                "date": date,
                "speaker": str(turn.get("speaker", "")),
                "text": str(turn.get("text", "")),
            }
    require(output, "conversation turn index is empty")
    return output


def resolve_sources(source_ids: Sequence[str], turn_index: Mapping[str, Mapping[str, str]]) -> str:
    output = []
    for source_id in source_ids:
        require(source_id in turn_index, f"source absent from conversation: {source_id}")
        turn = turn_index[source_id]
        output.append(
            f"[{source_id}] ({turn['date']}) {turn['speaker']}: {turn['text']}"
        )
    return "\n".join(output)


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    value = json.loads(stripped)
    require(isinstance(value, dict), "model JSON response is not an object")
    return value


def validate_topic_path(value: str) -> None:
    path = Path(value)
    require(
        isinstance(value, str)
        and value
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == value
        and len(path.parts) <= 4
        and "\x00" not in value
        and all(
            part not in {"", ".", ".."}
            and len(part.encode("utf-8")) <= 120
            for part in path.parts
        ),
        f"unsafe topic path: {value!r}",
    )
    require(
        v8_memory._sanitize_topic(value) == value,
        f"topic path changes during materialization: {value!r}",
    )


def audit_inputs(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], int]:
    input_manifest = load_json(root / "inputs/manifest.json")
    records = input_manifest.get("samples")
    require(isinstance(records, list) and records, "input manifest samples differ")
    require(
        input_manifest.get("sample_count") == len(records)
        and input_manifest.get("records_sha256") == value_sha256(records),
        "input manifest summary differs",
    )
    inputs = {}
    total_questions = 0
    canonical_inputs = []
    for record in records:
        require(isinstance(record, dict), "input record is not an object")
        sample = record.get("sample")
        require(isinstance(sample, int) and sample not in inputs, "input sample differs")
        path = root / str(record.get("path", ""))
        require(path.is_file() and sha256_file(path) == record.get("sha256"), "input hash differs")
        payload = load_json(path)
        require(
            payload.get("schema") == "nativemem.r301-frozen-input.v1"
            and payload.get("sample") == sample
            and payload.get("sample_id") == record.get("sample_id"),
            "frozen input identity differs",
        )
        validate_bank(payload.get("bank", {}), sample)
        questions = payload.get("questions")
        require(isinstance(questions, list) and questions, "input question list differs")
        require(
            record.get("question_count") == len(questions)
            and record.get("bank_entries_sha256")
            == payload["bank"]["entries_sha256"],
            "input record summary differs",
        )
        question_ids = set()
        for question in questions:
            require(isinstance(question, dict), "input question is invalid")
            question_id = question.get("question_id")
            require(
                isinstance(question_id, str)
                and question_id
                and question_id not in question_ids,
                "input question ID differs",
            )
            question_ids.add(question_id)
            require(
                isinstance(question.get("gold_answer"), str)
                and isinstance(question.get("gold_source_ids"), list)
                and isinstance(question.get("source_recall_eligible"), bool),
                "input scoring fields differ",
            )
        inputs[sample] = payload
        total_questions += len(questions)
        canonical_inputs.append(
            {
                "sample": sample,
                "sample_id": payload["sample_id"],
                "conversation": payload["conversation"],
                "bank": payload["bank"],
            }
        )
    require(
        input_manifest.get("question_count") == total_questions,
        "input question count differs",
    )
    require(
        manifest["config"].get("input_bundle_sha256") == value_sha256(canonical_inputs),
        "manifest input bundle hash differs",
    )
    return inputs, total_questions


def audit_organizer(
    root: Path,
    *,
    sample: int,
    organizer_id: str,
    replicate: int,
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    directory = (
        root
        / f"organizers/sample-{sample:02d}/{organizer_id}/replicate-{replicate:02d}"
    )
    artifact = load_json(directory / "organizer.json")
    require(
        artifact.get("schema") == ORGANIZER_SCHEMA
        and artifact.get("status") == "complete"
        and artifact.get("sample") == sample
        and artifact.get("organizer_id") == organizer_id
        and artifact.get("replicate") == replicate,
        "organizer identity differs",
    )
    placements = artifact.get("placements")
    require(isinstance(placements, list), "organizer placements differ")
    expected_ids = [entry["entry_id"] for entry in bank["entries"]]
    require(
        [item.get("entry_id") for item in placements] == expected_ids,
        "organizer changed entry membership/order",
    )
    for placement in placements:
        require(
            isinstance(placement, dict)
            and set(placement) == {"entry_id", "topic_path"},
            "placement keys differ",
        )
        validate_topic_path(placement["topic_path"])
    paths = {item["topic_path"] for item in placements}
    require(len(paths) <= 30, "organizer topic count exceeds limit")
    require(
        artifact.get("bank_entries_sha256") == bank["entries_sha256"]
        and artifact.get("entry_count") == len(bank["entries"])
        and artifact.get("placements_sha256") == value_sha256(placements)
        and artifact.get("topic_count") == len(paths)
        and artifact.get("maximum_depth")
        == max(len(Path(path).parts) for path in paths)
        and artifact.get("materialization")
        == "src.v8_memory.write_events + dedup_topic_files"
        and artifact.get("timeline_generation")
        == "src.v8_memory.write_events deterministic timeline"
        and artifact.get("model_maintenance_calls") == 0,
        "organizer summary differs",
    )
    memory = directory / "memory"
    require(artifact.get("memory_tree") == tree_descriptor(memory), "memory tree differs")
    by_id = {entry["entry_id"]: entry for entry in bank["entries"]}
    with tempfile.TemporaryDirectory(prefix="r301-rematerialize-", dir=root.parent) as raw:
        rebuilt = Path(raw) / "memory"
        rebuilt.mkdir()
        for placement in placements:
            entry = by_id[placement["entry_id"]]
            v8_memory.write_events(
                str(rebuilt),
                [
                    {
                        "when": entry["when"],
                        "summary": entry["summary"],
                        "summary_inline": entry["summary_inline"],
                        "dia_ids": entry["dia_ids"],
                        "topic": placement["topic_path"],
                    }
                ],
            )
        removed = int(v8_memory.dedup_topic_files(str(rebuilt)) or 0)
        require(
            artifact.get("exact_duplicate_entries_removed") == removed,
            "deterministic exact-dedup count differs",
        )
        require(
            tree_descriptor(rebuilt) == tree_descriptor(memory),
            "independent deterministic rematerialization differs",
        )
    return artifact


def reconstruct_deliveries(trace: Sequence[Mapping[str, Any]]) -> dict[str, DeliveryResult]:
    output = {}
    for record in trace:
        if record.get("record_type") != "delivery":
            continue
        delivered = record.get("delivered")
        output[str(record["event_id"])] = DeliveryResult(
            event_id=str(record["event_id"]),
            kind=str(record["kind"]),
            decision=str(record["decision"]),
            delivered_text=(
                str(delivered["text"]) if isinstance(delivered, dict) else None
            ),
            delivered_tokens=(
                int(delivered["tokens"]) if isinstance(delivered, dict) else 0
            ),
            cumulative_visible_tokens=int(record["cumulative_visible_tokens"]),
            exhausted=bool(record["exhausted"]),
            finalizer_required=bool(record["finalizer_required"]),
        )
    return output


def ledger_calls(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    starts = {
        str(record["logical_call_id"]): record
        for record in records
        if record.get("event") == "model_call_started"
    }
    terminals = {
        str(record["logical_call_id"]): record
        for record in records
        if record.get("event") in {"model_call_finished", "model_call_failed"}
    }
    require(set(starts) == set(terminals), "model call ledger is incomplete")
    require(
        all(record.get("event") == "model_call_finished" for record in terminals.values()),
        "model call ledger contains a failure",
    )
    return starts, terminals


def validate_prefix(prefix: Mapping[str, Any], path: Path) -> None:
    require(Path(str(prefix.get("path"))).resolve() == path.resolve(), "proxy path differs")
    offset = prefix.get("byte_offset")
    require(isinstance(offset, int) and 0 <= offset <= path.stat().st_size, "proxy offset differs")
    require(
        prefix.get("prefix_sha256") == hashlib.sha256(path.read_bytes()[:offset]).hexdigest(),
        "proxy prefix hash differs",
    )


def audit_proxy_lifecycle(
    root: Path,
    prereg: Mapping[str, Any],
    *,
    run_fingerprint: str,
    config: Mapping[str, Any],
) -> tuple[dict[Path, dict[str, str]], list[dict[str, Any]]]:
    proxy_root = root / "proxy"
    require(proxy_root.is_dir() and not proxy_root.is_symlink(), "formal proxy root differs")
    ready_paths = sorted(proxy_root.rglob("launch-*.ready.json"))
    require(ready_paths, "formal proxy has no launch artifacts")
    logs: dict[Path, dict[str, str]] = {}
    flex_reports: list[dict[str, Any]] = []
    gpt55_contract = config.get("gpt55_gateway")
    require(isinstance(gpt55_contract, dict), "formal GPT-5.5 gateway contract is absent")
    try:
        flex_evidence.validate_recorded_contract(gpt55_contract)
    except flex_evidence.EvidenceError as exc:
        raise AuditFailure(str(exc)) from exc
    require(
        config.get("gpt55_upstream") == gpt55_contract.get("origin"),
        "formal GPT-5.5 upstream differs from its gateway contract",
    )
    gpt4o_profile = prereg["organizers"]["O4o"]
    for ready_path in ready_paths:
        label = ready_path.parent.name
        require(label in {"gpt55", "gpt4o"}, "proxy launch label differs")
        ready = load_json(ready_path)
        log = Path(str(ready.get("log", ""))).resolve()
        launch = ready_path.name.removesuffix(".ready.json")
        expected_run_id = f"{run_fingerprint}:{label}:{launch}"
        stopped_path = ready_path.with_name(
            ready_path.name.removesuffix(".ready.json") + ".stopped.json"
        )
        stopped = load_json(stopped_path)
        port = ready.get("port")
        common_ready = (
            ready.get("run_id") == expected_run_id
            and isinstance(port, int)
            and 0 < port < 65536
            and ready.get("base_url") == f"http://127.0.0.1:{port}/v1"
            and log.is_file()
            and not log.is_symlink()
            and log.parent == ready_path.parent
        )
        if label == "gpt55":
            require(
                common_ready
                and ready.get("upstream") == gpt55_contract["origin"]
                and ready.get("base_wrapper_sha256")
                == sha256_file(ROOT / "scripts/gpt55_run_proxy.py")
                and ready.get("controlled_wrapper_sha256")
                == sha256_file(ROOT / "scripts/controlled_gpt55_run_proxy.py"),
                "GPT-5.5 proxy ready provenance differs",
            )
        else:
            require(
                common_ready
                and ready.get("schema")
                == "nativemem.r301-exclusive-proxy-ready.v1"
                and ready.get("expected_requested_model")
                == gpt4o_profile["requested_model"]
                and ready.get("accepted_actual_model_regex")
                == gpt4o_profile["accepted_actual_model_regex"]
                and ready.get("upstream")
                == str(config.get("gpt4o_upstream")).rstrip("/")
                and ready.get("api_key_env")
                == config.get("gpt4o_api_key_env")
                and ready.get("wrapper_sha256")
                == sha256_file(ROOT / "scripts/controlled_r301_model_proxy.py")
                and ready.get("api_key_recorded") is False,
                "GPT-4o-mini proxy ready provenance differs",
            )
        require(
            stopped.get("schema") == "nativemem.r301-exclusive-proxy-stop.v1"
            and stopped.get("status")
            in {"normal_stop", "interrupted_recovered", "already_exited"}
            and stopped.get("run_id") == ready.get("run_id")
            and stopped.get("pid") == ready.get("pid")
            and Path(str(stopped.get("ready"))).resolve() == ready_path.resolve()
            and stopped.get("ready_sha256") == sha256_file(ready_path)
            and Path(str(stopped.get("log"))).resolve() == log
            and stopped.get("log_sha256") == sha256_file(log),
            "proxy stop provenance differs",
        )
        if label == "gpt55":
            window_start = ready_path.with_name(
                ready_path.name.removesuffix(".ready.json")
                + ".flex-window-start.json"
            )
            require(
                Path(str(stopped.get("provider_window_start"))).resolve()
                == window_start.resolve()
                and stopped.get("provider_window_start_sha256")
                == sha256_file(window_start)
                and stopped.get("gateway_contract") == gpt55_contract,
                "GPT-5.5 provider-window start provenance differs",
            )
            require(
                load_json(window_start).get("contract") == gpt55_contract,
                "GPT-5.5 provider-window start contract differs",
            )
            try:
                flex_reports.append(
                    flex_evidence.audit_window(
                        stopped.get("provider_window"),
                        consumer_records=load_jsonl(log),
                    )
                )
            except flex_evidence.EvidenceError as exc:
                raise AuditFailure(str(exc)) from exc
        else:
            require(
                stopped.get("gateway_contract") is None
                and stopped.get("provider_window_start") is None
                and stopped.get("provider_window_start_sha256") is None
                and stopped.get("provider_window") is None,
                "GPT-4o-mini proxy contains GPT-5.5 provider evidence",
            )
        pid = ready.get("pid")
        require(isinstance(pid, int) and pid > 0, "proxy PID differs")
        process = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            capture_output=True,
            check=False,
        )
        command = process.stdout.strip()
        require(
            not command
            or str(
                ROOT
                / "scripts"
                / (
                    "controlled_gpt55_run_proxy.py"
                    if label == "gpt55"
                    else "controlled_r301_model_proxy.py"
                )
            )
            not in command
            or str(ready_path) not in command,
            "exclusive proxy is still running after the stop manifest",
        )
        require(log not in logs, "proxy log is assigned to multiple launches")
        logs[log] = {"run_id": expected_run_id, "label": label}
    for directory in (proxy_root / "gpt55", proxy_root / "gpt4o"):
        require(directory.is_dir(), "formal proxy model directory differs")
        expected_names = {
            path.name for path in directory.glob("launch-*.ready.json")
        } | {path.name for path in directory.glob("launch-*.jsonl")} | {
            path.name for path in directory.glob("launch-*.stopped.json")
        }
        if directory.name == "gpt55":
            expected_names |= {
                path.name
                for path in directory.glob("launch-*.flex-window-start.json")
            }
        require(
            {path.name for path in directory.iterdir()} == expected_names,
            "proxy directory contains an unknown artifact",
        )
    require(flex_reports, "formal GPT-5.5 proxy has no provider window")
    return logs, flex_reports


def audit_calls(
    root: Path,
    *,
    mode: str,
    prereg: Mapping[str, Any],
    config: Mapping[str, Any],
    run_fingerprint: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    list[dict[str, Any]],
]:
    starts, terminals = ledger_calls(records)
    model_profiles = {
        **prereg["organizers"],
        **prereg["retrievers"],
        "A55": prereg["answerer"],
    }
    assigned_proxy_events: set[tuple[str, str]] = set()
    formal_logs, flex_reports = (
        audit_proxy_lifecycle(
            root,
            prereg,
            run_fingerprint=run_fingerprint,
            config=config,
        )
        if mode == "formal"
        else ({}, [])
    )
    for logical_id, start in starts.items():
        terminal = terminals[logical_id]
        request_path = root / str(start.get("request_path", ""))
        response_path = root / str(terminal.get("response_path", ""))
        require(
            request_path.is_file()
            and sha256_file(request_path) == start.get("request_sha256"),
            "request artifact hash differs",
        )
        require(
            response_path.is_file()
            and sha256_file(response_path) == terminal.get("response_sha256"),
            "response artifact hash differs",
        )
        request = load_json(request_path)
        response_wrapper = load_json(response_path)
        response = response_wrapper.get("response")
        require(isinstance(response, dict), "response payload differs")
        model_id = start.get("model_id")
        profile = model_profiles.get(model_id)
        require(isinstance(profile, dict), "model profile is unknown")
        actual = str(response.get("model", ""))
        require(
            logical_id.startswith(f"{run_fingerprint}:")
            and request.get("schema") == "nativemem.r301-model-request.v1"
            and request.get("logical_call_id") == logical_id
            and request.get("operation_id") == start.get("operation_id")
            and request.get("question_id") == start.get("question_id")
            and request.get("model_id") == model_id
            and start.get("requested_model") == profile["requested_model"]
            and start.get("accepted_actual_model_regex")
            == profile["accepted_actual_model_regex"]
            and response_wrapper.get("schema")
            == "nativemem.r301-model-response.v1"
            and response_wrapper.get("logical_call_id") == logical_id
            and terminal.get("operation_id") == start.get("operation_id")
            and terminal.get("question_id") == start.get("question_id")
            and terminal.get("model_id") == model_id,
            "request/ledger/response linkage differs",
        )
        require(
            request.get("requested_model") == profile["requested_model"]
            and request.get("accepted_actual_model_regex")
            == profile["accepted_actual_model_regex"]
            and re.fullmatch(profile["accepted_actual_model_regex"], actual)
            is not None,
            "requested/actual model binding differs",
        )
        require(
            terminal.get("response_model") == actual
            and terminal.get("response_id") == response.get("id")
            and terminal.get("usage") == normalize_usage(response.get("usage")),
            "ledger response identity differs",
        )
        payload = request.get("payload")
        require(isinstance(payload, dict), "model request payload differs")
        visible = {
            "messages": payload.get("messages"),
            **(
                {"response_format": payload["response_format"]}
                if "response_format" in payload
                else {}
            ),
        }
        tokenizer = answer_contract.formal_token_counter()
        require(
            request.get("model_visible_sha256") == value_sha256(visible)
            and request.get("local_visible_tokens")
            == tokenizer.count(canonical_json(visible))
            and request.get("tokenizer") == tokenizer.identity,
            "local model-visible accounting differs",
        )
        serialized = canonical_json(payload)
        require(
            '"gold_answer"' not in serialized
            and '"gold_source_ids"' not in serialized
            and '"category"' not in serialized,
            "post-model scoring fields leaked into a model request",
        )
        evidence = terminal.get("proxy_evidence")
        require(isinstance(evidence, dict), "proxy evidence is missing")
        if mode == "synthetic_no_network":
            require(
                evidence.get("mode") == "synthetic"
                and evidence.get("events") == []
                and evidence.get("client_http_attempts") == 0
                and evidence.get("upstream_http_attempts") == 0,
                "synthetic call contains network evidence",
            )
        else:
            require(evidence.get("mode") == "exclusive_proxy", "formal proxy mode differs")
            events = evidence.get("events")
            require(isinstance(events, list) and len(events) == 1, "formal proxy events differ")
            event = events[0]
            log = Path(str(evidence.get("log_prefix", {}).get("path")))
            require(
                log.resolve() in formal_logs and log.is_file() and not log.is_symlink(),
                "formal proxy log differs",
            )
            validate_prefix(start["proxy_log_start"], log)
            validate_prefix(evidence["log_prefix"], log)
            log_events = load_jsonl(log)
            linked = [item for item in log_events if item.get("logical_call_id") == logical_id]
            require(linked == events, "embedded proxy event differs from log")
            require(
                event.get("status") == "success"
                and event.get("http_status") in range(200, 300)
                and event.get("run_id") == formal_logs[log.resolve()]["run_id"]
                and event.get("logical_call_id") == logical_id
                and event.get("question_id")
                == (request.get("question_id") or "organizer")
                and event.get("requested_model") == profile["requested_model"]
                and event.get("actual_model") == actual
                and event.get("response_id") == response.get("id")
                and event.get("request_sha256")
                == hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
                and event.get("client_http_attempts") == 1
                and isinstance(event.get("upstream_http_attempts"), int)
                and event["upstream_http_attempts"] >= 1
                and evidence.get("client_http_attempts") == 1
                and evidence.get("upstream_http_attempts")
                == event["upstream_http_attempts"]
                and evidence.get("unsupported_parameters")
                == sorted(set(event.get("unsupported_parameters", [])))
                and normalize_usage(event.get("usage")) == terminal.get("usage"),
                "formal proxy response binding differs",
            )
            if formal_logs[log.resolve()]["label"] == "gpt55":
                require(
                    isinstance(event.get("gateway_request_id"), str)
                    and bool(event["gateway_request_id"])
                    and isinstance(event.get("gateway_request_sha256"), str)
                    and re.fullmatch(
                        r"[0-9a-f]{64}", event["gateway_request_sha256"]
                    )
                    is not None
                    and isinstance(event.get("provider_request_sha256"), str)
                    and re.fullmatch(
                        r"[0-9a-f]{64}", event["provider_request_sha256"]
                    )
                    is not None
                    and event.get("provider_actual_model")
                    == "gpt-5.5-2026-04-23"
                    and event.get("service_tier") == "flex",
                    "GPT-5.5 Flex proxy response evidence differs",
                )
            key = (str(log.resolve()), str(event.get("event_id")))
            require(key not in assigned_proxy_events, "proxy event assigned twice")
            assigned_proxy_events.add(key)
    if mode == "formal":
        proxy_events = []
        for path in formal_logs:
            proxy_events.extend((str(path.resolve()), item) for item in load_jsonl(path))
        require(
            {(path, str(item.get("event_id"))) for path, item in proxy_events}
            == assigned_proxy_events,
            "formal proxy log contains an unassigned event",
        )
    else:
        require(not (root / "proxy").exists(), "synthetic run contains a proxy directory")
    return starts, terminals, flex_reports


def audit_question(
    root: Path,
    *,
    path: Path,
    record: Mapping[str, Any],
    input_sample: Mapping[str, Any],
    organizers: Mapping[tuple[int, str, int], Mapping[str, Any]],
    starts: Mapping[str, Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    result = load_json(path)
    require(
        result.get("schema") == QUESTION_SCHEMA and result.get("status") == "complete",
        "question result identity differs",
    )
    for field in ("cell", "replicate", "sample", "sample_id", "question_id"):
        require(result.get(field) == record.get(field), f"question inventory {field} differs")
    sample = int(result["sample"])
    cell = str(result["cell"])
    organizer_id, retriever_id = cell.split("-", 1)
    replicate = int(result["replicate"])
    organizer = organizers[(sample, organizer_id, replicate)]
    bank = input_sample["bank"]
    questions = {
        item["question_id"]: item for item in input_sample["questions"]
    }
    require(result["question_id"] in questions, "result question is not frozen")
    question = questions[result["question_id"]]
    require(
        result.get("question") == question["question"]
        and result.get("question_index") == question["question_index"]
        and result.get("category") == question["category"]
        and result.get("bank_entries_sha256") == bank["entries_sha256"]
        and result.get("placements_sha256") == organizer["placements_sha256"],
        "question input binding differs",
    )
    memory = (
        root
        / f"organizers/sample-{sample:02d}/{organizer_id}/replicate-{replicate:02d}/memory"
    )
    memory_snapshot = snapshot_memory_path(memory).descriptor
    require(
        result.get("memory_before") == memory_snapshot
        and result.get("memory_after") == memory_snapshot
        and result.get("memory_unchanged") is True,
        "question before/after memory hash differs",
    )
    trace_path = path.parent / "visible_tokens.jsonl"
    trace_manifest = path.parent / "visible_tokens.manifest.json"
    visible_audit = audit_visible_token_trace(
        trace_path,
        manifest_path=trace_manifest,
        memory_before_path=memory,
        memory_after_path=memory,
    )
    require(visible_audit.get("audit_status") == "pass", "visible-token audit failed")
    require(
        visible_audit.get("configured_budget_tokens") == VISIBLE_BUDGET
        and visible_audit.get("tokenizer")
        == prereg["visible_token_gate"]["tokenizer"],
        "visible-token policy differs",
    )
    trace = load_jsonl(trace_path)
    deliveries = reconstruct_deliveries(trace)
    tool_trace = result.get("tool_trace")
    require(isinstance(tool_trace, list) and len(tool_trace) >= 4, "tool trace differs")
    require(
        [item.get("ordinal") for item in tool_trace]
        == list(range(1, len(tool_trace) + 1)),
        "tool trace order differs",
    )
    for item in tool_trace:
        delivery = deliveries.get(str(item.get("event_id")))
        require(delivery is not None, "tool trace event is absent from token trace")
        require(
            item.get("decision") == delivery.decision
            and item.get("delivered_tokens") == delivery.delivered_tokens,
            "tool trace delivery summary differs",
        )
        trace_record = next(
            row for row in trace if row.get("event_id") == item.get("event_id")
        )
        require(
            item.get("raw_sha256") == trace_record["raw"]["sha256"],
            "tool trace raw hash differs",
        )
    selected_paths = result.get("selected_topic_paths")
    selected_entry_ids = result.get("selected_entry_ids")
    selected_source_ids = result.get("selected_source_ids")
    require(
        isinstance(selected_paths, list)
        and isinstance(selected_entry_ids, list)
        and isinstance(selected_source_ids, list)
        and len(set(selected_paths)) == len(selected_paths)
        and len(set(selected_entry_ids)) == len(selected_entry_ids)
        and len(set(selected_source_ids)) == len(selected_source_ids),
        "ordered retrieval selections differ",
    )
    placement_by_id = {
        item["entry_id"]: item["topic_path"] for item in organizer["placements"]
    }
    entry_by_id = {item["entry_id"]: item for item in bank["entries"]}
    opened_expected = [
        entry_id
        for entry_id in placement_by_id
        if placement_by_id[entry_id] in set(selected_paths)
    ]
    require(result.get("opened_entry_ids") == opened_expected, "opened entries differ")
    require(set(selected_entry_ids).issubset(opened_expected), "selected entry was not opened")
    allowed_sources = {
        source_id
        for entry_id in selected_entry_ids
        for source_id in entry_by_id[entry_id]["dia_ids"]
    }
    require(set(selected_source_ids).issubset(allowed_sources), "selected source differs")
    turn_index = build_turn_index(input_sample["conversation"])
    resolved_text = resolve_sources(selected_source_ids, turn_index)
    source_record = next(
        row for row in trace if row.get("event_id") == "source-0001-resolve-selected"
    )
    require(
        source_record.get("kind") == "source_resolution"
        and source_record.get("raw", {}).get("text") == resolved_text,
        "source-resolution payload differs",
    )
    answer_event_ids = ["tool-0003-selection-trace"] + [
        f"tool-entry-{index:04d}" for index in range(1, len(selected_entry_ids) + 1)
    ] + ["source-0001-resolve-selected"]
    answer_deliveries = [deliveries[event_id] for event_id in answer_event_ids]
    prompt = answer_contract.assemble_answer_prompt(
        question=question["question"], deliveries=answer_deliveries
    )
    calls = result.get("model_calls")
    require(isinstance(calls, list) and len(calls) == 3, "question model calls differ")
    call_ids = [call.get("logical_call_id") for call in calls]
    require(all(call_id in starts for call_id in call_ids), "question call linkage differs")
    for call, call_id in zip(calls, call_ids, strict=True):
        terminal = terminals[str(call_id)]
        require(
            call.get("response_id") == terminal.get("response_id")
            and call.get("actual_model") == terminal.get("response_model")
            and call.get("usage") == terminal.get("usage")
            and call.get("latency_s") == terminal.get("latency_s")
            and call.get("proxy_evidence") == terminal.get("proxy_evidence"),
            "question call summary differs from durable ledger",
        )
    navigation_request = load_json(root / starts[str(call_ids[0])]["request_path"])
    navigation_messages = navigation_request.get("payload", {}).get("messages")
    require(
        navigation_request.get("model_id") == retriever_id
        and isinstance(navigation_messages, list)
        and len(navigation_messages) == 2,
        "navigation request identity differs",
    )
    navigation_user = json.loads(navigation_messages[-1]["content"])
    require(
        navigation_user
        == {
            "retriever_id": retriever_id,
            "question": question["question"],
            "path_inventory_text": deliveries["tool-0001-list-paths"].delivered_text,
            "maximum_paths": 4,
        },
        "navigation request bypassed the delivered inventory",
    )
    selection_request = load_json(root / starts[str(call_ids[1])]["request_path"])
    selection_messages = selection_request.get("payload", {}).get("messages")
    require(
        selection_request.get("model_id") == retriever_id
        and isinstance(selection_messages, list)
        and len(selection_messages) == 2,
        "selection request identity differs",
    )
    selection_user = json.loads(selection_messages[-1]["content"])
    require(
        selection_user
        == {
            "retriever_id": retriever_id,
            "question": question["question"],
            "opened_entries_text": deliveries["tool-0002-open-paths"].delivered_text,
            "maximum_entries": 8,
        },
        "selection request bypassed the delivered opened-entry observation",
    )
    answer_request_path = root / starts[str(call_ids[-1])]["request_path"]
    answer_request = load_json(answer_request_path)
    require(
        answer_request.get("model_id") == "A55"
        and answer_request.get("payload", {}).get("messages")
        == [{"role": "user", "content": prompt}]
        and result.get("answer_prompt_sha256")
        == hashlib.sha256(prompt.encode()).hexdigest(),
        "fixed-answer prompt boundary differs",
    )
    answer_response_path = root / terminals[str(call_ids[-1])]["response_path"]
    answer_response = load_json(answer_response_path)["response"]
    raw_answer = answer_response["choices"][0]["message"]["content"]
    answer = answer_contract.extract_answer(raw_answer)
    require(
        result.get("answer") == answer
        and result.get("gold_answer_post_model_only") == question["gold_answer"]
        and result.get("fixed_answerer_f1_set")
        == f1_set(answer, question["gold_answer"]),
        "fixed-answer score differs",
    )
    require(
        result.get("diagnostics")
        == {
            "navigation_model_calls": 2,
            "navigation_tool_calls": 2,
            "navigation_visible_tokens": (
                deliveries["tool-0001-list-paths"].delivered_tokens
                + deliveries["tool-0002-open-paths"].delivered_tokens
            ),
            "navigation_provider_prompt_tokens": sum(
                int(call["usage"].get("prompt_tokens") or 0) for call in calls[:2]
            ),
            "navigation_latency_s": round(
                sum(float(call["latency_s"]) for call in calls[:2]), 6
            ),
        },
        "navigation diagnostic accounting differs",
    )
    gold_sources = question["gold_source_ids"]
    eligible = question["source_recall_eligible"]
    recall = set(gold_sources).issubset(selected_source_ids) if eligible else None
    source_delivered = source_record.get("delivered")
    source_resolution_recall = (
        bool(recall)
        and source_record.get("decision") == "delivered"
        and isinstance(source_delivered, dict)
        and all(
            f"[{source_id}]" in str(source_delivered.get("text", ""))
            for source_id in gold_sources
        )
        if eligible
        else None
    )
    require(
        result.get("gold_source_ids_post_model_only") == gold_sources
        and result.get("source_recall_eligible") == eligible
        and result.get("mapped_source_recall_all_gold") == recall,
        "mapped-source recall differs",
    )
    first_ordinal = first_path = None
    for ordinal, topic_path in enumerate(selected_paths, start=1):
        if any(
            placement_by_id[entry_id] == topic_path
            and set(entry_by_id[entry_id]["dia_ids"]) & set(gold_sources)
            for entry_id in opened_expected
        ):
            first_ordinal = ordinal
            first_path = topic_path
            break
    require(
        result.get("first_relevant_file")
        == {
            "path": first_path,
            "opened_path_ordinal": first_ordinal,
            "first_path_hit": (first_ordinal == 1 if eligible else None),
        },
        "first relevant file differs",
    )
    budget = result.get("visible_budget")
    require(
        isinstance(budget, dict)
        and budget.get("configured_tokens") == VISIBLE_BUDGET
        and budget.get("visible_tokens") == visible_audit["cumulative_visible_tokens"]
        and budget.get("source_resolution_tokens")
        == visible_audit["cumulative_source_resolution_tokens"]
        and budget.get("trace_sha256") == sha256_file(trace_path)
        and budget.get("tokenizer") == prereg["visible_token_gate"]["tokenizer"],
        "question visible-token summary differs",
    )
    models = result.get("models")
    require(
        isinstance(models, dict)
        and models.get("answerer_requested") == "gpt-5.5"
        and models.get("answerer_actual") == calls[-1]["actual_model"]
        and models.get("retriever_requested")
        == prereg["retrievers"][retriever_id]["requested_model"],
        "question model binding differs",
    )
    stage = result.get("stage_evidence")
    require(isinstance(stage, dict), "stage evidence is missing")
    canonical_all = (
        all(
            any(source in entry["dia_ids"] for entry in bank["entries"])
            for source in gold_sources
        )
        if eligible
        else None
    )
    require(
        stage.get("gold_source_mapping_complete") == eligible
        and stage.get("gold_source_in_canonical_entries") == canonical_all
        and stage.get("gold_source_survived_maintenance") == canonical_all
        and stage.get("gold_source_path_valid") == canonical_all
        and stage.get("retrieval_reached_gold_source") == recall
        and stage.get("source_resolution_returned_gold_content")
        == source_resolution_recall,
        "M4 stage evidence differs",
    )
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def interaction(means: Mapping[str, float]) -> float:
    return (means["O55-R55"] - means["O4o-R55"]) - (
        means["O55-R4o"] - means["O4o-R4o"]
    )


def reconstruct_analysis(results: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for result in results:
        grouped[(result["sample_id"], result["question_id"], result["cell"])].append(
            float(result["fixed_answerer_f1_set"])
        )
    question_cell = {}
    for key, values in grouped.items():
        require(len(values) == 3, f"question/cell replicate count differs: {key}")
        question_cell[key] = sum(values) / 3
    samples = sorted({key[0] for key in question_cell})
    cell_values = {
        cell: [value for key, value in question_cell.items() if key[2] == cell]
        for cell in CELLS
    }
    require(all(cell_values.values()), "interaction cell is empty")
    cell_means = {
        cell: sum(values) / len(values) for cell, values in cell_values.items()
    }
    by_sample = {sample: {cell: [] for cell in CELLS} for sample in samples}
    for (sample, _question, cell), value in question_cell.items():
        by_sample[sample][cell].append(value)
    rng = random.Random(20_260_714)
    bootstrap = []
    for _ in range(10_000):
        selected = [rng.choice(samples) for _ in samples]
        means = {}
        for cell in CELLS:
            values = [value for sample in selected for value in by_sample[sample][cell]]
            means[cell] = sum(values) / len(values)
        bootstrap.append(interaction(means))
    interval = {
        "lower": percentile(bootstrap, 0.025),
        "upper": percentile(bootstrap, 0.975),
    }
    delta_r55 = cell_means["O55-R55"] - cell_means["O4o-R55"]
    delta_r4o = cell_means["O4o-R4o"] - cell_means["O55-R4o"]
    matched = delta_r55 > 0 and delta_r4o > 0
    passed = matched and interval["lower"] > 0
    formal = mode == "formal"
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "outcome": "fixed_answerer_f1_set",
        "record_count": len(results),
        "question_cell_count": len(question_cell),
        "conversation_cluster_count": len(samples),
        "organizer_replicates": list(REPLICATES),
        "m4_comparison_preregistration": {
            "path": "paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json",
            "sha256": EXPECTED_M4_SHA256,
            "section": "m5_interaction",
        },
        "replicate_aggregation": "equal-weight mean within cell and question",
        "cell_means": cell_means,
        "matched_direction": {
            "O55_minus_O4o_under_R55": delta_r55,
            "O4o_minus_O55_under_R4o": delta_r4o,
            "both_strict": matched,
        },
        "interaction": {
            "formula": "(O55-R55 - O4o-R55) - (O55-R4o - O4o-R4o)",
            "point_estimate": interaction(cell_means),
            "confidence_interval_95": interval,
            "bootstrap_cluster": "conversation",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_714,
            "interval_method": "two-sided percentile",
        },
        "decision": {
            "matched_direction_required": True,
            "positive_interval_lower_bound_required": True,
            "rule_passed": passed,
            "claim_status": (
                "model_specific_organization_supported"
                if formal and passed
                else "model_specific_organization_not_supported"
                if formal
                else "synthetic_validation_only"
            ),
            "failure_action": (
                "delete matching claim and report organizer or retriever capacity effects only"
            ),
        },
    }


def audit_run(output_dir: Path) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    require(ROOT in root.parents, "R301 artifact must be under repository root")
    assert_safe_tree(root)
    prereg = validate_preregistration()
    manifest = load_json(root / "run_manifest.json")
    complete = load_json(root / "complete.json")
    require(
        manifest.get("schema") == RUN_SCHEMA
        and manifest.get("status") == "running"
        and complete.get("schema") == RUN_SCHEMA
        and complete.get("status") == "complete",
        "run/complete identity differs",
    )
    mode = complete.get("mode")
    require(mode in {"formal", "synthetic_no_network"}, "run mode differs")
    require(
        manifest.get("run_fingerprint") == complete.get("run_fingerprint"),
        "run fingerprint linkage differs",
    )
    require(manifest.get("source_hashes") == source_hashes(), "source hashes differ")
    config = manifest.get("config")
    require(isinstance(config, dict), "run config is missing")
    require(
        config.get("preregistration_sha256") == sha256_file(PREREG)
        and config.get("preregistration_content_sha256")
        == prereg["protocol_content_sha256"]
        and config.get("m4_comparison_preregistration_sha256")
        == EXPECTED_M4_SHA256
        and config.get("cells") == list(CELLS)
        and config.get("organizer_replicates") == list(REPLICATES)
        and config.get("visible_budget_tokens") == VISIBLE_BUDGET,
        "run config differs from preregistration",
    )
    expected_fingerprint = value_sha256(
        {"schema": RUN_SCHEMA, "config": config, "source_hashes": source_hashes()}
    )
    require(
        manifest.get("run_fingerprint") == expected_fingerprint,
        "run fingerprint differs",
    )
    inputs, question_count = audit_inputs(root, manifest)
    if mode == "formal":
        require(
            config.get("mode") == "formal"
            and config.get("network_requests_authorized") is True,
            "formal request authorization record differs",
        )
        require(set(inputs) == set(range(10)) and question_count == 1540, "formal scope differs")
        require(
            sha256_file(ROOT / "benchmarks/locomo/data/locomo10.json")
            == EXPECTED_DATA_SHA256
            and sha256_file(
                ROOT
                / "results/paper-experiments-20260714/evidence-mapping/v1/"
                "evidence_mapping.v1.questions.jsonl"
            )
            == EXPECTED_MAPPING_SHA256,
            "formal source dataset/mapping hash differs",
        )
        r207_root = Path(str(config.get("r207_root")))
        import audit_r207_path_control  # noqa: PLC0415

        r207 = audit_r207_path_control.audit(r207_root)
        require(
            r207.get("status") == "pass"
            and r207.get("sample_count") == 10
            and r207.get("run_fingerprint") == config.get("r207_run_fingerprint"),
            "bound R207 source no longer passes audit",
        )
        for sample, payload in inputs.items():
            source_path = r207_root / f"samples/sample-{sample}/canonical_entry_bank.json"
            require(load_json(source_path) == payload["bank"], "frozen bank differs from R207")
    else:
        require(
            set(inputs) == {0}
            and question_count == 1
            and config.get("mode") == "synthetic_no_network"
            and config.get("network_requests_authorized") is False
            and config.get("gpt55_gateway") is None
            and config.get("gpt55_upstream") is None,
            "synthetic scope or gateway binding differs",
        )
    records = read_ledger(root / "operations.jsonl")
    require(
        records
        and all(
            record.get("run_id") == manifest["run_fingerprint"]
            for record in records
        ),
        "durable ledger run identity differs",
    )
    state = ledger_state(records)
    require(state["failed_model_calls"] == 0, "ledger has failed calls")
    starts, terminals, flex_reports = audit_calls(
        root,
        mode=mode,
        prereg=prereg,
        config=config,
        run_fingerprint=manifest["run_fingerprint"],
        records=records,
    )
    gateway_report: dict[str, Any] | None = None
    if mode == "formal":
        start_binding = config.get("openrouter_gateway")
        final_binding = complete.get("openrouter_gateway")
        require(
            isinstance(start_binding, dict) and isinstance(final_binding, dict),
            "formal OpenRouter gateway binding is absent",
        )
        require(
            all(
                final_binding.get(name) == start_binding.get(name)
                for name in start_binding
            ),
            "formal OpenRouter final binding changed its frozen start fields",
        )
        try:
            openrouter_gateway_evidence.validate_binding(
                start_binding,
                result_root=Path(str(start_binding.get("result_root", ""))),
                base_url=str(start_binding.get("base_url", "")),
            )
            gateway_response_ids = [
                str(terminal.get("response_id"))
                for terminal in terminals.values()
                if terminal.get("response_model")
                == openrouter_gateway_evidence.REQUESTED_MODEL
            ]
            gateway_report = openrouter_gateway_evidence.verify_response_ids(
                final_binding, gateway_response_ids
            )
        except openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise AuditFailure(str(exc)) from exc
    committed = state["committed_operations"]
    organizers = {}
    for sample, payload in inputs.items():
        for organizer_id in ORGANIZERS:
            for replicate in REPLICATES:
                organizer = audit_organizer(
                    root,
                    sample=sample,
                    organizer_id=organizer_id,
                    replicate=replicate,
                    bank=payload["bank"],
                )
                organizers[(sample, organizer_id, replicate)] = organizer
                operation_id = (
                    f"organizer/sample-{sample:02d}/{organizer_id}/"
                    f"replicate-{replicate:02d}"
                )
                require(operation_id in committed, "organizer operation is not committed")
                call_id = organizer.get("organizer_call", {}).get("logical_call_id")
                require(call_id in starts and call_id in terminals, "organizer call linkage differs")
                organizer_call = organizer["organizer_call"]
                terminal = terminals[str(call_id)]
                require(
                    organizer_call.get("response_id") == terminal.get("response_id")
                    and organizer_call.get("actual_model")
                    == terminal.get("response_model")
                    and organizer_call.get("usage") == terminal.get("usage")
                    and organizer_call.get("latency_s") == terminal.get("latency_s")
                    and organizer_call.get("proxy_evidence")
                    == terminal.get("proxy_evidence"),
                    "organizer call summary differs from durable ledger",
                )
                request = load_json(root / starts[str(call_id)]["request_path"])
                messages = request.get("payload", {}).get("messages")
                require(
                    request.get("model_id") == organizer_id
                    and isinstance(messages, list)
                    and len(messages) == 2,
                    "organizer request identity differs",
                )
                organizer_input = json.loads(messages[-1]["content"])
                expected_input = {
                    "organizer_id": organizer_id,
                    "replicate": replicate,
                    "current_path_inventory": [],
                    "entries": [
                        {
                            "entry_id": entry["entry_id"],
                            "when": entry["when"],
                            "summary": entry["summary"],
                            "summary_inline": entry["summary_inline"],
                            "dia_ids": entry["dia_ids"],
                        }
                        for entry in payload["bank"]["entries"]
                    ],
                }
                require(organizer_input == expected_input, "organizer input bank differs")
                response = load_json(root / terminals[str(call_id)]["response_path"])[
                    "response"
                ]
                raw_content = response["choices"][0]["message"]["content"]
                require(
                    parse_json_response(raw_content)
                    == {"placements": organizer["placements"]},
                    "organizer response/placement artifact differs",
                )
    inventory = load_jsonl(root / "analysis/question_results.jsonl")
    require(
        len(inventory) == question_count * len(CELLS) * len(REPLICATES),
        "question inventory count differs",
    )
    keys = [
        (item.get("cell"), item.get("replicate"), item.get("sample"), item.get("question_id"))
        for item in inventory
    ]
    require(len(set(keys)) == len(keys), "question inventory contains duplicates")
    expected_keys = {
        (cell, replicate, sample, question["question_id"])
        for sample, payload in inputs.items()
        for question in payload["questions"]
        for cell in CELLS
        for replicate in REPLICATES
    }
    require(set(keys) == expected_keys, "question inventory matrix differs")
    results = []
    for item in inventory:
        path = root / str(item.get("result_path", ""))
        require(
            path.is_file()
            and sha256_file(path) == item.get("result_sha256"),
            "question result hash differs",
        )
        sample = int(item["sample"])
        result = audit_question(
            root,
            path=path,
            record=item,
            input_sample=inputs[sample],
            organizers=organizers,
            starts=starts,
            terminals=terminals,
            prereg=prereg,
        )
        require(
            result["operation_id"] in committed,
            "question operation is not committed",
        )
        commit = committed[result["operation_id"]]
        require(
            commit.get("artifact_path") == str(path.relative_to(root))
            and commit.get("artifact_sha256") == sha256_file(path),
            "question commit artifact differs",
        )
        results.append(result)
    expected_operations = len(organizers) + len(results)
    require(
        len(committed) == expected_operations,
        "ledger contains missing or additional committed operations",
    )
    expected_calls = len(organizers) + 3 * len(results)
    require(
        state["successful_model_calls"] == expected_calls
        and len(starts) == expected_calls,
        "model call count differs",
    )
    analysis = reconstruct_analysis(results, mode)
    recorded_analysis = load_json(root / "analysis/interaction.json")
    require(recorded_analysis == analysis, "interaction analysis differs")
    require(
        complete.get("network_requests")
        == (expected_calls if mode == "formal" else 0)
        and complete.get("sample_count") == len(inputs)
        and complete.get("question_count") == question_count
        and complete.get("cell_count") == 4
        and complete.get("organizer_replicates") == 3
        and complete.get("organizer_artifact_count") == len(organizers)
        and complete.get("question_result_count") == len(results)
        and complete.get("model_call_count") == expected_calls
        and complete.get("failed_model_calls") == 0
        and complete.get("operations_sha256")
        == sha256_file(root / "operations.jsonl")
        and complete.get("input_manifest_sha256")
        == sha256_file(root / "inputs/manifest.json")
        and complete.get("question_inventory_sha256")
        == sha256_file(root / "analysis/question_results.jsonl")
        and complete.get("interaction_sha256")
        == sha256_file(root / "analysis/interaction.json")
        and complete.get("interaction_decision") == analysis["decision"],
        "complete manifest summary differs",
    )
    return {
        "schema": "nativemem.r301-audit.v1",
        "status": "pass",
        "artifact_dir": str(root),
        "run_fingerprint": manifest["run_fingerprint"],
        "mode": mode,
        "sample_count": len(inputs),
        "question_count": question_count,
        "cell_count": len(CELLS),
        "organizer_replicates": len(REPLICATES),
        "organizer_artifact_count": len(organizers),
        "question_result_count": len(results),
        "model_call_count": expected_calls,
        "network_requests": complete["network_requests"],
        "visible_token_trace_count": len(results),
        "interaction": analysis["interaction"],
        "decision": analysis["decision"],
        "openrouter_gateway_evidence": gateway_report,
        "openai_gpt55_flex_gateway_evidence": flex_reports,
        "m4_comparison_preregistration_sha256": EXPECTED_M4_SHA256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_run(args.artifact_dir)
    if args.report:
        answer_contract.atomic_json_replace(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditFailure, OSError, ValueError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
