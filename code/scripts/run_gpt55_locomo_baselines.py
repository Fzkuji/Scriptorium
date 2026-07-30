#!/usr/bin/env python3
"""Run retrieval-only LoCoMo baselines with the GPT-5.5 builder.

The launcher deliberately stops at build + retrieval.  It writes one atomic
checkpoint per conversation and invokes ``assemble_locomo_baseline_inputs.py``
only after all ten pinned LoCoMo conversations validate.  Answering and judging
are outside this script's scope.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import locomo_baseline_contract as contract


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402

DATASET = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
ASSEMBLER = ROOT / "scripts" / "assemble_locomo_baseline_inputs.py"
AUDITOR = ROOT / "scripts" / "audit_gpt55_locomo_baselines.py"
CONTRACT = ROOT / "scripts" / "locomo_baseline_contract.py"
RUN_PROXY = ROOT / "scripts" / "gpt55_run_proxy.py"
FLEX_GATEWAY = ROOT / "src" / "openai_gpt55_flex_gateway.py"
FLEX_GATEWAY_AUDITOR = ROOT / "scripts" / "audit_openai_gpt55_flex_gateway.py"
DEFAULT_PYTHON = "/opt/miniconda3/bin/python3"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "gpt55-locomo-baselines-20260714"
PREREGISTRATION = DEFAULT_OUTPUT_ROOT / "formal_matrix_preregistration.json"
FLEX_ROOT_MARKER_NAME = "openai_gpt55_flex_root.json"
FLEX_STATE_NAME = "flex_cost_state.json"
FLEX_REQUEST_LOG_NAME = "flex_requests.jsonl"
FLEX_READY_NAME = "gateway_ready.json"
FLEX_ROOT_SCHEMA = "openai-gpt55-flex-result-root/v1"
FLEX_STATE_SCHEMA = "openai-gpt55-flex-cost-state/v1"
FLEX_HEALTH_SCHEMA = "openai-gpt55-flex-health/v1"
FLEX_PROVIDER_MODEL = "gpt-5.5-2026-04-23"
FLEX_REQUESTED_MODEL = "gpt-5.5"
FLEX_SERVICE_TIER = "flex"
EXPECTED_CATEGORIES = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
EXPECTED_QUESTIONS = 1986
EXPECTED_PRIMARY = 1540
EXPECTED_ADVERSARIAL = 446
REQUIRED_BUILD_FIELDS = {
    "build_time_s",
    "num_memories",
    "build_calls",
    "build_tokens_in",
    "build_tokens_out",
    "build_llm_time_s",
    "notes",
}
REQUIRED_RETRIEVAL_FIELDS = {
    "latency_s",
    "k",
    "calls",
    "tokens_in",
    "tokens_out",
}


class BaselineRunError(RuntimeError):
    """Raised when an artifact cannot be accepted as a valid baseline run."""


@dataclass(frozen=True)
class MethodSpec:
    adapter: Path
    extra_args: tuple[str, ...] = ()
    expects_llm: bool = True
    usage_tracking: str = "in_process"
    source_extras: tuple[Path, ...] = ()
    runtime_trees: tuple[Path, ...] = ()
    runtime_python_env: str | None = None
    runtime_python_default: Path | None = None
    max_sample_workers: int = 1
    requires_diagnostics: bool = False


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "full_context": MethodSpec(
        ROOT / "src/adapters/run_naive.py",
        ("--method", "full_context"),
        expects_llm=False,
        usage_tracking="not_applicable",
        max_sample_workers=10,
    ),
    "bm25": MethodSpec(
        ROOT / "src/adapters/run_naive.py",
        ("--method", "bm25"),
        expects_llm=False,
        usage_tracking="not_applicable",
        max_sample_workers=10,
    ),
    "mem0": MethodSpec(
        ROOT / "src/adapters/run_mem0.py",
        runtime_trees=(ROOT / "third_party/mem0",),
    ),
    "amem": MethodSpec(
        ROOT / "src/adapters/run_amem.py",
        runtime_trees=(ROOT / "third_party/amem",),
    ),
    "lightmem": MethodSpec(
        ROOT / "src/adapters/run_lightmem.py",
        ("--k", "20"),
        runtime_trees=(ROOT / "third_party/lightmem",),
    ),
    "nemori": MethodSpec(
        ROOT / "src/adapters/run_nemori.py",
        runtime_trees=(ROOT / "third_party/nemori",),
    ),
    "zep": MethodSpec(
        ROOT / "src/adapters/run_zep.py",
        runtime_trees=(ROOT / "third_party/graphiti",),
    ),
    "memoryos": MethodSpec(
        ROOT / "src/adapters/run_memoryos.py",
        runtime_trees=(ROOT / "third_party/memoryos/memoryos-pypi",),
    ),
    "memos": MethodSpec(
        ROOT / "src/adapters/run_memos.py",
        runtime_trees=(ROOT / "third_party/memos",),
    ),
    "simplemem": MethodSpec(
        ROOT / "src/adapters/run_simplemem.py",
        runtime_trees=(ROOT / "third_party/simplemem",),
    ),
    "hindsight": MethodSpec(
        ROOT / "src/adapters/run_hindsight.py",
        runtime_trees=(ROOT / "third_party/hindsight/hindsight-api-slim",),
        runtime_python_default=ROOT / ".venv-hindsight/bin/python",
    ),
    "memmachine": MethodSpec(
        ROOT / "src/adapters/run_memmachine.py",
        expects_llm=False,
        usage_tracking="not_applicable",
        runtime_trees=(ROOT / "third_party/memmachine",),
        runtime_python_default=ROOT / "third_party/memmachine_venv/bin/python",
    ),
    "mirix": MethodSpec(
        ROOT / "src/adapters/run_mirix.py",
        usage_tracking="subprocess_untracked",
        source_extras=(ROOT / "src/adapters/_mirix_server_shim.py",),
        runtime_trees=(ROOT / "third_party/mirix",),
        runtime_python_env="MIRIX_PY",
        runtime_python_default=ROOT / "third_party/mirix-venv/bin/python",
        requires_diagnostics=True,
    ),
    "emem": MethodSpec(
        ROOT / "src/adapters/run_emem.py",
        runtime_trees=(ROOT / "third_party/emem",),
    ),
    "evermemos": MethodSpec(
        ROOT / "src/adapters/run_evermemos.py",
        usage_tracking="subprocess_untracked",
        source_extras=(ROOT / "src/adapters/_evermemos_server_shim.py",),
        runtime_trees=(ROOT / "third_party/evermemos",),
        runtime_python_env="EVERMEMOS_PY",
        runtime_python_default=ROOT / "third_party/evermemos_venv/bin/python",
        requires_diagnostics=True,
    ),
}
FORMAL_METHODS = ("full_context", "bm25", "mem0", "zep")
STRUCTURED_PREREGISTRATION = {
    "method": "zep",
    "implementation": "Graphiti OSS temporal context graph",
    "claim_boundary": "Graphiti OSS core, not the commercial Zep platform",
    "adapter_surface": "Graphiti.add_episode and Graphiti.search edge facts",
}
PREREGISTRATION_PUBLIC_EVIDENCE = [
    {
        "title": "Graphiti official repository",
        "url": "https://github.com/getzep/graphiti",
        "supports": "open-source temporal context graph engine and core API surface",
    },
    {
        "title": "Zep: A Temporal Knowledge Graph Architecture for Agent Memory",
        "url": "https://arxiv.org/abs/2501.13956",
        "supports": "temporal knowledge-graph memory architecture",
    },
]
PREREGISTRATION_SELECTION_BASIS = [
    "Graphiti is publicly positioned as an open-source temporal context graph engine.",
    "The local adapter directly calls Graphiti.add_episode and Graphiti.search.",
    "The adapter uses an in-process Kuzu graph and local embedding shim, allowing the formal implementation surface to be fingerprinted and audited.",
]
PREREGISTRATION_DOWNSTREAM_GATES = {
    "R002_source_mapping": "required_not_yet_audited",
    "R004_visible_token_gate": "required_not_yet_audited",
    "formal_answer_generation": "not_started",
    "judging": "not_started",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


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


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineRunError(f"cannot read valid JSON from {path}: {exc}") from exc


def parse_preregistration_time(value: object) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BaselineRunError(
            f"formal matrix preregistration has an invalid frozen_at: {value!r}"
        ) from exc
    if timestamp.tzinfo is None:
        raise BaselineRunError("formal matrix preregistration frozen_at has no timezone")
    return timestamp


def load_formal_preregistration(
    path: Path = PREREGISTRATION,
) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise BaselineRunError("formal matrix preregistration is not an object")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "frozen_at",
        "benchmark",
        "task_scope",
        "formal_methods",
        "structured_method",
        "selection",
        "non_formal_methods",
        "non_formal_scope",
        "downstream_gates",
        "prohibited_until_gates_pass",
    }
    if set(payload) != expected_keys:
        raise BaselineRunError("formal matrix preregistration fields differ")
    fixed = {
        "schema_version": 1,
        "artifact_type": "formal_experiment_matrix_preregistration",
        "status": "frozen",
        "benchmark": "LoCoMo",
        "task_scope": "retrieval_inputs_before_answer_generation",
        "formal_methods": list(FORMAL_METHODS),
        "structured_method": STRUCTURED_PREREGISTRATION,
        "non_formal_methods": [
            method for method in METHOD_REGISTRY if method not in FORMAL_METHODS
        ],
        "non_formal_scope": "smoke_or_blocked_only",
        "downstream_gates": PREREGISTRATION_DOWNSTREAM_GATES,
        "prohibited_until_gates_pass": ["formal_answer_generation", "judging"],
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in fixed.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise BaselineRunError(
            f"formal matrix preregistration differs from the frozen contract: {mismatches}"
        )
    selection = payload.get("selection")
    expected_selection = {
        "selection_used_test_answers": False,
        "basis": PREREGISTRATION_SELECTION_BASIS,
        "public_evidence": PREREGISTRATION_PUBLIC_EVIDENCE,
    }
    if selection != expected_selection:
        raise BaselineRunError("formal matrix preregistration selection evidence differs")
    parse_preregistration_time(payload.get("frozen_at"))
    return payload


def preregistration_record(
    payload: dict[str, Any], path: Path = PREREGISTRATION
) -> dict[str, Any]:
    return {
        "path": source_key(path),
        "sha256": sha256_file(path),
        "frozen_at": payload["frozen_at"],
        "status": payload["status"],
    }


def source_key(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def required_source_paths(spec: MethodSpec) -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        AUDITOR,
        ASSEMBLER,
        CONTRACT,
        RUN_PROXY,
        FLEX_GATEWAY,
        FLEX_GATEWAY_AUDITOR,
        Path(flex_evidence.__file__),
        PREREGISTRATION,
        spec.adapter,
        ROOT / "src/adapters/_usage_tracker.py",
        ROOT / "src/evaluation/llm_clients.py",
        *spec.source_extras,
    ]
    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.expanduser().resolve()] = None
    return list(unique)


def sha256_tree(path: Path) -> dict[str, Any]:
    """Hash relevant source/config files while excluding environments and caches."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise BaselineRunError(f"required runtime source tree does not exist: {resolved}")
    suffixes = {
        ".cfg",
        ".c",
        ".cc",
        ".cpp",
        ".conf",
        ".css",
        ".csv",
        ".go",
        ".graphql",
        ".h",
        ".hpp",
        ".html",
        ".in",
        ".ini",
        ".j2",
        ".java",
        ".jinja",
        ".jinja2",
        ".js",
        ".jsx",
        ".json",
        ".kt",
        ".lock",
        ".md",
        ".mustache",
        ".prompt",
        ".proto",
        ".py",
        ".rs",
        ".rst",
        ".scss",
        ".sh",
        ".sql",
        ".template",
        ".tmpl",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
    excluded = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".venv",
        "venv",
    }
    records: list[dict[str, str]] = []
    for candidate in sorted(resolved.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
            continue
        relative = candidate.relative_to(resolved)
        if any(
            part in excluded
            or part.endswith("_venv")
            or part.endswith("-venv")
            for part in relative.parts
        ):
            continue
        records.append({"path": str(relative), "sha256": sha256_file(candidate)})
    if not records:
        raise BaselineRunError(f"runtime source tree has no fingerprinted files: {resolved}")
    return {
        "path": source_key(resolved),
        "included_suffixes": sorted(suffixes),
        "excluded_directory_names": sorted(excluded),
        "excluded_venv_name_suffixes": ["-venv", "_venv"],
        "files": len(records),
        "sha256": canonical_hash(records),
    }


def compute_source_hashes(spec: MethodSpec) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in required_source_paths(spec):
        if not path.is_file():
            raise BaselineRunError(f"required source file does not exist: {path}")
        result[source_key(path)] = sha256_file(path)
    return result


def resolve_runtime_python(spec: MethodSpec, primary: Path) -> Path:
    candidate: Path | None = None
    if spec.runtime_python_env and os.environ.get(spec.runtime_python_env):
        candidate = Path(os.environ[spec.runtime_python_env])
    elif spec.runtime_python_default is not None and spec.runtime_python_default.exists():
        candidate = spec.runtime_python_default
    if candidate is None:
        return primary
    return resolve_python(str(candidate))


def package_inventory(python: Path) -> dict[str, Any]:
    code = (
        "import hashlib,importlib.metadata as m,json,sys;"
        "names=('METADATA','RECORD','direct_url.json','INSTALLER','entry_points.txt');"
        "h=lambda d,n:(hashlib.sha256(s.encode()).hexdigest() "
        "if (s:=d.read_text(n)) is not None else None);"
        "d=sorted(({'name':x.metadata.get('Name') or x.metadata.get('Summary') or '',"
        "'version':x.version,'metadata':{n:h(x,n) for n in names}} "
        "for x in m.distributions()),key=lambda x:(x['name'].casefold(),x['version']));"
        "print(json.dumps({'executable':sys.executable,'version':sys.version,'packages':d}))"
    )
    try:
        payload = subprocess.check_output(
            [str(python), "-c", code], text=True, stderr=subprocess.STDOUT, timeout=60
        )
        data = json.loads(payload)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise BaselineRunError(f"cannot fingerprint runtime {python}: {exc}") from exc
    return {
        "requested_executable": str(python),
        "resolved_executable": str(Path(data["executable"]).absolute()),
        "python_sha256": sha256_file(Path(data["executable"]).absolute()),
        "python_version": data["version"],
        "packages": data["packages"],
        "packages_sha256": canonical_hash(data["packages"]),
    }


def runtime_fingerprint(
    spec: MethodSpec,
    primary: Path,
    effective: Path | None = None,
) -> dict[str, Any]:
    runtime_python = effective or resolve_runtime_python(spec, primary)
    return {
        "primary": package_inventory(primary),
        "effective": package_inventory(runtime_python),
        "source_trees": [sha256_tree(path) for path in spec.runtime_trees],
    }


def expected_gold(qa: dict[str, Any]) -> str:
    try:
        return contract.expected_gold(qa)
    except contract.ContractError as exc:
        raise BaselineRunError(str(exc)) from exc


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        return contract.load_dataset(path)
    except contract.ContractError as exc:
        raise BaselineRunError(str(exc)) from exc


def _legacy_load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or len(payload) != 10:
        raise BaselineRunError("LoCoMo dataset must contain exactly ten conversations")
    categories: Counter[int] = Counter()
    total = 0
    for sample, item in enumerate(payload):
        if not isinstance(item, dict) or not isinstance(item.get("qa"), list):
            raise BaselineRunError(f"dataset sample {sample} has no QA list")
        for qa in item["qa"]:
            if not isinstance(qa, dict):
                raise BaselineRunError(f"dataset sample {sample} contains invalid QA")
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise BaselineRunError(f"dataset sample {sample} has an empty question")
            try:
                category = int(qa["category"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BaselineRunError(
                    f"dataset sample {sample} has an invalid category"
                ) from exc
            expected_gold(qa)
            categories[category] += 1
            total += 1
    if total != EXPECTED_QUESTIONS:
        raise BaselineRunError(
            f"LoCoMo dataset has {total} questions, expected {EXPECTED_QUESTIONS}"
        )
    if dict(sorted(categories.items())) != EXPECTED_CATEGORIES:
        raise BaselineRunError(
            f"LoCoMo category inventory differs: {dict(sorted(categories.items()))}"
        )
    return payload


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_number(value: object) -> bool:
    return _is_number(value) and float(value) >= 0


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_LIST_FAILURES = re.compile(
    r"\b(?:failed_sessions|failed|failures)\s*=\s*\[([^\]]*)\]", re.IGNORECASE
)
_COUNT_FAILURES = re.compile(
    r"\b(?:failed_turns|pages_failed|llm_errors|cascade_failed|"
    r"failed_retryable|failed_permanent)\s*=\s*(-?\d+)",
    re.IGNORECASE,
)
_BUILD_ERROR = re.compile(r"\bbuild_error\s*=\s*([^;,]+)", re.IGNORECASE)
_LOG_FAILURES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"traceback \(most recent call last\)",
        r"\busing fallback\b",
        r"\bfalling back\b",
        r"\bper-session fallback\b",
        r"\bbuild error\b",
        r"\bsearch error\b",
        r"\bretrieve error\b",
        r"\brecall error\b",
        r"\bget_all(?:_memories)? failed\b",
        r"\bfinal flush failed\b",
        r"\bfact count failed\b",
        r"\bprobe failed\b",
        r"\bfailed after retr(?:y|ies)\b",
        r"\bsession\s+\S+\s+failed\b",
        r"\bturn failed\b",
        r"\bserver died\b",
        r"\bserver not healthy\b",
        r"\bdrain timeout\b",
        r"\[error\]",
    )
)


def validate_failure_markers(build: dict[str, Any], log_text: str) -> None:
    notes = str(build.get("notes", ""))
    if not notes.strip():
        raise BaselineRunError("_build_stats.notes is empty")
    if re.search(r"\bfallback\b", notes, re.IGNORECASE):
        raise BaselineRunError("_build_stats records a fallback")
    for match in _LIST_FAILURES.finditer(notes):
        if match.group(1).strip():
            raise BaselineRunError(f"_build_stats records {match.group(0)}")
    for match in _COUNT_FAILURES.finditer(notes):
        if int(match.group(1)) != 0:
            raise BaselineRunError(f"_build_stats records {match.group(0)}")
    build_error = _BUILD_ERROR.search(notes)
    if build_error and build_error.group(1).strip().lower() not in {"none", "null", ""}:
        raise BaselineRunError(f"_build_stats records {build_error.group(0)}")
    for key in (
        "failed_sessions",
        "failed_turns",
        "failed_pages",
        "pages_failed",
        "build_error",
        "error",
        "fallback",
    ):
        if key not in build:
            continue
        value = build[key]
        if value not in (None, False, 0, "", [], {}):
            raise BaselineRunError(f"_build_stats.{key} reports failure: {value!r}")
    for pattern in _LOG_FAILURES:
        match = pattern.search(log_text)
        if match:
            raise BaselineRunError(
                f"attempt log contains failure marker: {match.group(0)!r}"
            )


def validate_sample_records(
    rows: Any,
    dataset_item: dict[str, Any],
    sample: int,
    *,
    log_text: str = "",
    expected_builder: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        return contract.validate_sample_records(
            rows,
            dataset_item,
            sample,
            log_text=log_text,
            expected_builder=expected_builder,
        )
    except contract.ContractError as exc:
        raise BaselineRunError(str(exc)) from exc


def _legacy_validate_sample_records(
    rows: Any,
    dataset_item: dict[str, Any],
    sample: int,
    *,
    log_text: str = "",
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise BaselineRunError(f"sample {sample} output is not a non-empty list")
    if not isinstance(rows[0], dict) or rows[0].get("question_id") != "_build_stats":
        raise BaselineRunError(f"sample {sample} lacks leading _build_stats")
    if any(
        isinstance(record, dict) and record.get("question_id") == "_build_stats"
        for record in rows[1:]
    ):
        raise BaselineRunError(f"sample {sample} contains multiple _build_stats")
    build = rows[0]
    missing_build = sorted(REQUIRED_BUILD_FIELDS - set(build))
    if missing_build:
        raise BaselineRunError(
            f"sample {sample} _build_stats misses fields: {missing_build}"
        )
    if not _nonnegative_number(build["build_time_s"]):
        raise BaselineRunError(f"sample {sample} has invalid build_time_s")
    if not isinstance(build["num_memories"], int) or isinstance(
        build["num_memories"], bool
    ) or build["num_memories"] <= 0:
        raise BaselineRunError(f"sample {sample} has no built memories")
    for key in ("build_calls", "build_tokens_in", "build_tokens_out"):
        if not _nonnegative_integer(build[key]):
            raise BaselineRunError(f"sample {sample} has invalid {key}")
    if not _nonnegative_number(build["build_llm_time_s"]):
        raise BaselineRunError(f"sample {sample} has invalid build_llm_time_s")
    validate_failure_markers(build, log_text)

    expected = dataset_item.get("qa")
    if not isinstance(expected, list):
        raise BaselineRunError(f"dataset sample {sample} has no QA list")
    questions = rows[1:]
    if len(questions) != len(expected):
        raise BaselineRunError(
            f"sample {sample} has {len(questions)} questions; expected {len(expected)}"
        )
    for index, (record, source) in enumerate(zip(questions, expected)):
        question_id = f"s{sample}_q{index}"
        if not isinstance(record, dict):
            raise BaselineRunError(f"{question_id} is not an object")
        if record.get("question_id") != question_id:
            raise BaselineRunError(f"{question_id} has a mismatched question_id")
        if record.get("question") != source.get("question"):
            raise BaselineRunError(f"{question_id} question differs from dataset")
        if record.get("gold") != expected_gold(source):
            raise BaselineRunError(f"{question_id} gold differs from dataset")
        if record.get("category") != source.get("category"):
            raise BaselineRunError(f"{question_id} category differs from dataset")
        if "answer" in record:
            raise BaselineRunError(f"{question_id} contains an answerer output")
        memories = record.get("memories")
        if not isinstance(memories, list) or not memories:
            raise BaselineRunError(f"{question_id} has empty memories")
        for memory_index, memory in enumerate(memories):
            if not isinstance(memory, dict):
                raise BaselineRunError(
                    f"{question_id} memory {memory_index} is not an object"
                )
            text = memory.get("text")
            if not isinstance(text, str) or not text.strip():
                raise BaselineRunError(
                    f"{question_id} memory {memory_index} has empty text"
                )
            lowered = text.strip().lower()
            if lowered.startswith(("[error]", "[fallback]")) or lowered in {
                "no memory",
                "no memories",
                "no relevant memory",
                "no relevant memory found.",
            }:
                raise BaselineRunError(
                    f"{question_id} memory {memory_index} is a failure placeholder"
                )
            if "date" not in memory or not (
                memory["date"] is None or isinstance(memory["date"], str)
            ):
                raise BaselineRunError(
                    f"{question_id} memory {memory_index} has invalid date metadata"
                )
        retrieval = record.get("retrieval")
        if not isinstance(retrieval, dict):
            raise BaselineRunError(f"{question_id} lacks retrieval metadata")
        missing_retrieval = sorted(REQUIRED_RETRIEVAL_FIELDS - set(retrieval))
        if missing_retrieval:
            raise BaselineRunError(
                f"{question_id} retrieval misses fields: {missing_retrieval}"
            )
        if not _nonnegative_number(retrieval["latency_s"]):
            raise BaselineRunError(f"{question_id} has invalid retrieval latency")
        if not isinstance(retrieval["k"], int) or isinstance(
            retrieval["k"], bool
        ) or retrieval["k"] <= 0:
            raise BaselineRunError(f"{question_id} has invalid retrieval k")
        for key in ("calls", "tokens_in", "tokens_out"):
            if not _nonnegative_integer(retrieval[key]):
                raise BaselineRunError(f"{question_id} has invalid retrieval {key}")
        for key in ("error", "build_error", "fallback"):
            if retrieval.get(key) not in (None, False, 0, "", [], {}):
                raise BaselineRunError(
                    f"{question_id} retrieval reports {key}: {retrieval[key]!r}"
                )
    return {
        "questions": len(questions),
        "num_memories": build["num_memories"],
        "build_calls": build["build_calls"],
        "retrieval_calls": sum(record["retrieval"]["calls"] for record in questions),
    }


def validate_sample_file(
    path: Path,
    dataset_item: dict[str, Any],
    sample: int,
    *,
    log_path: Path | None = None,
    expected_builder: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_text = ""
    if log_path is not None:
        if not log_path.is_file():
            raise BaselineRunError(f"sample {sample} attempt log is missing")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return validate_sample_records(
        read_json(path),
        dataset_item,
        sample,
        log_text=log_text,
        expected_builder=expected_builder,
    )


def builder_contract(spec: MethodSpec, upstream_base: str) -> dict[str, str]:
    if not spec.expects_llm:
        return {
            "requested_model": "not_applicable",
            "base_url": "not_applicable",
            "usage_tracking": spec.usage_tracking,
        }
    return {
        "requested_model": "gpt-5.5",
        "base_url": (
            "managed-exclusive-proxy->openai-api-flex-gateway->"
            f"{upstream_base.rstrip('/')}/v1"
        ),
        "usage_tracking": spec.usage_tracking,
    }


def load_flex_gateway_contract(result_root: Path) -> dict[str, Any]:
    """Validate the only allowed GPT-5.5 provider root."""
    try:
        evidence = flex_evidence.active_contract(result_root)
    except flex_evidence.EvidenceError as exc:
        raise BaselineRunError(str(exc)) from exc
    return {
        **evidence,
        "gateway_source_sha256": sha256_file(FLEX_GATEWAY),
        "gateway_auditor_sha256": sha256_file(FLEX_GATEWAY_AUDITOR),
        "gateway_evidence_source_sha256": sha256_file(Path(flex_evidence.__file__)),
    }


def resolve_python(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" in value:
        resolved = candidate.absolute()
    else:
        found = shutil.which(value)
        if found is None:
            raise BaselineRunError(f"Python executable not found: {value}")
        resolved = Path(found).absolute()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise BaselineRunError(f"Python executable is not executable: {resolved}")
    return resolved


def path_identity(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False)).casefold()


def is_within(path: Path, directory: Path) -> bool:
    candidate = path.expanduser().resolve(strict=False)
    parent = directory.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_run_paths(
    args: argparse.Namespace,
    python: Path,
    runtime_python: Path,
) -> None:
    out_dir = args.output_dir.expanduser().resolve(strict=False)
    protected = {
        "dataset": args.dataset,
        "python": python,
        "effective_python": runtime_python,
        "adapter": METHOD_REGISTRY[args.method].adapter,
        "assembler": ASSEMBLER,
        "auditor": AUDITOR,
        "contract": CONTRACT,
        "run_proxy": RUN_PROXY,
        "flex_gateway": FLEX_GATEWAY,
        "flex_gateway_auditor": FLEX_GATEWAY_AUDITOR,
        "preregistration": PREREGISTRATION,
    }
    for index, path in enumerate(required_source_paths(METHOD_REGISTRY[args.method])):
        protected[f"source_{index}"] = path
    for index, path in enumerate(METHOD_REGISTRY[args.method].runtime_trees):
        protected[f"runtime_tree_{index}"] = path
    if args.proxy_log is not None:
        protected["legacy_proxy_log"] = args.proxy_log
    if getattr(args, "gateway_root", None) is not None:
        protected["flex_gateway_result_root"] = args.gateway_root
    for label, path in protected.items():
        if is_within(path, out_dir):
            raise BaselineRunError(f"{label} must be outside output directory: {path}")
    if path_identity(out_dir) == path_identity(ROOT) or is_within(ROOT, out_dir):
        raise BaselineRunError("output directory cannot be the repository or its parent")
    reserved_output_roots = (
        ROOT / "benchmarks",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "third_party",
    )
    for reserved in reserved_output_roots:
        if is_within(out_dir, reserved):
            raise BaselineRunError(
                f"output directory cannot be inside protected source/data tree: {reserved}"
            )


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def start_managed_proxy(args: argparse.Namespace, out_dir: Path) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    try:
        provider_window = flex_evidence.capture_start(args.gateway_root)
    except flex_evidence.EvidenceError as exc:
        raise BaselineRunError(f"cannot capture Flex provider start: {exc}") from exc
    run_id = uuid.uuid4().hex
    proxy_dir = out_dir / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    log_path = proxy_dir / f"{run_id}.jsonl"
    ready_path = proxy_dir / f"{run_id}.ready.json"
    process_log = proxy_dir / f"{run_id}.process.log"
    command = [
        str(resolve_python(args.python)),
        str(RUN_PROXY),
        "--port",
        "0",
        "--upstream",
        args.upstream_base,
        "--log",
        str(log_path),
        "--ready",
        str(ready_path),
        "--run-id",
        run_id,
    ]
    handle = process_log.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    deadline = time.monotonic() + 30
    ready: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BaselineRunError(
                f"managed proxy exited during startup; see {process_log}"
            )
        if ready_path.is_file():
            value = read_json(ready_path)
            if isinstance(value, dict) and value.get("run_id") == run_id:
                ready = value
                break
        time.sleep(0.1)
    if ready is None:
        process.terminate()
        raise BaselineRunError("managed proxy did not become ready")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(
            f"http://127.0.0.1:{ready['port']}/healthz", timeout=10
        ) as response:
            health = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        process.terminate()
        raise BaselineRunError(f"managed proxy health check failed: {exc}") from exc
    if health.get("status") != "ok" or health.get("run_id") != run_id:
        process.terminate()
        raise BaselineRunError("managed proxy health response differs")
    upstream_health = health.get("upstream_health")
    if (
        not isinstance(upstream_health, dict)
        or upstream_health.get("status") != "ok"
        or upstream_health.get("schema") != FLEX_HEALTH_SCHEMA
        or upstream_health.get("requested_model") != FLEX_REQUESTED_MODEL
        or upstream_health.get("provider_model") != FLEX_PROVIDER_MODEL
        or upstream_health.get("service_tier") != FLEX_SERVICE_TIER
        or not isinstance(upstream_health.get("budget"), dict)
        or upstream_health["budget"].get("max_cost_usd")
        != args.gateway_contract["max_cost_usd"]
    ):
        process.terminate()
        raise BaselineRunError("fixed Flex gateway health evidence differs")
    record = {
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "base_url": ready["base_url"],
        "upstream": args.upstream_base.rstrip("/"),
        "log": str(log_path.relative_to(out_dir)),
        "ready": str(ready_path.relative_to(out_dir)),
        "process_log": str(process_log.relative_to(out_dir)),
        "pid": process.pid,
        "health": health,
        "ready_sha256": sha256_file(ready_path),
        "ready_bytes": ready_path.stat().st_size,
        "wrapper_sha256": sha256_file(RUN_PROXY),
        "gateway_source_sha256": sha256_file(FLEX_GATEWAY),
        "gateway_root_marker_sha256": args.gateway_contract[
            "root_marker_sha256"
        ],
        "gateway_result_root": args.gateway_contract["result_root"],
        "provider_model": FLEX_PROVIDER_MODEL,
        "service_tier": FLEX_SERVICE_TIER,
        "provider_window": provider_window,
        "exact_run_linkage": True,
    }
    return process, record


def finish_managed_proxy(
    process: subprocess.Popen[str], record: dict[str, Any], out_dir: Path
) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_path = out_dir / record["log"]
    process_log = out_dir / record["process_log"]
    record.update({
        "status": "captured",
        "finished_at": utc_now(),
        "returncode": process.returncode,
        "log_sha256": sha256_file(log_path),
        "log_bytes": log_path.stat().st_size,
        "process_log_sha256": sha256_file(process_log),
        "process_log_bytes": process_log.stat().st_size,
    })
    try:
        record["provider_window"] = flex_evidence.capture_end(
            record["provider_window"]
        )
    except flex_evidence.EvidenceError as exc:
        record["provider_window_error"] = str(exc)


def _proxy_record_path(
    record: dict[str, Any], key: str, out_dir: Path
) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise BaselineRunError(f"managed proxy {key} path is invalid")
    path = (out_dir / value).resolve()
    if not is_within(path, out_dir / "proxy"):
        raise BaselineRunError(f"managed proxy {key} path escapes proxy directory")
    return path


def _process_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def recover_interrupted_managed_proxies(
    manifest: dict[str, Any], out_dir: Path
) -> None:
    """Stop and capture exclusive proxies left by an interrupted launcher."""
    evidence = manifest.get("proxy_evidence")
    invocations = evidence.get("invocations") if isinstance(evidence, dict) else None
    if not isinstance(invocations, list):
        raise BaselineRunError("managed proxy invocation inventory is invalid")
    for record in invocations:
        if not isinstance(record, dict):
            raise BaselineRunError("managed proxy invocation record is invalid")
        if record.get("status") != "running":
            continue
        run_id = record.get("run_id")
        pid = record.get("pid")
        if not isinstance(run_id, str) or not run_id:
            raise BaselineRunError("interrupted managed proxy run_id is invalid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise BaselineRunError("interrupted managed proxy pid is invalid")
        command = _process_command(pid)
        action = "original_process_not_running"
        if command and str(RUN_PROXY) in command and run_id in command:
            action = "terminated_original_process"
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise BaselineRunError(
                    f"cannot stop interrupted managed proxy pid {pid}"
                ) from exc
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if not _process_command(pid):
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise BaselineRunError(
                        f"cannot kill interrupted managed proxy pid {pid}"
                    ) from exc
                kill_deadline = time.monotonic() + 5
                while time.monotonic() < kill_deadline and _process_command(pid):
                    time.sleep(0.1)
                if _process_command(pid):
                    raise BaselineRunError(
                        f"interrupted managed proxy pid {pid} did not stop"
                    )
        elif command:
            action = "original_pid_reused"
        log_path = _proxy_record_path(record, "log", out_dir)
        process_log = _proxy_record_path(record, "process_log", out_dir)
        ready_path = _proxy_record_path(record, "ready", out_dir)
        for path in (log_path, process_log, ready_path):
            if not path.is_file():
                raise BaselineRunError(
                    f"interrupted managed proxy evidence is missing: {path}"
                )
        record.update(
            {
                "status": "interrupted_captured",
                "finished_at": utc_now(),
                "recovery_action": action,
                "log_sha256": sha256_file(log_path),
                "log_bytes": log_path.stat().st_size,
                "process_log_sha256": sha256_file(process_log),
                "process_log_bytes": process_log.stat().st_size,
                "ready_sha256": sha256_file(ready_path),
                "ready_bytes": ready_path.stat().st_size,
            }
        )
        try:
            record["provider_window"] = flex_evidence.capture_end(
                record.get("provider_window", {})
            )
        except flex_evidence.EvidenceError as exc:
            record["provider_window_error"] = str(exc)


def experiment_env(
    args: argparse.Namespace,
    dataset_path: Path,
    spec: MethodSpec,
    diagnostics_dir: Path,
) -> dict[str, str]:
    passthrough = {
        "ALL_PROXY",
        "CUDA_VISIBLE_DEVICES",
        "CURL_CA_BUNDLE",
        "DYLD_LIBRARY_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "HUGGINGFACE_HUB_CACHE",
        "KMP_DUPLICATE_LIB_OK",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "MKL_NUM_THREADS",
        "NO_PROXY",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "REQUESTS_CA_BUNDLE",
        "SENTENCE_TRANSFORMERS_HOME",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "TZ",
        "USER",
        "VECLIB_MAXIMUM_THREADS",
        "XDG_CACHE_HOME",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    env = {key: value for key, value in os.environ.items() if key in passthrough}
    local_gateway_key = "r110-local-flex-gateway"
    env.update(
        {
            "BUILDER_MODEL": "gpt-5.5",
            "BUILDER_BASE": args.base_url,
            "BUILDER_KEY": local_gateway_key,
            "ALIYUN_KEY": local_gateway_key,
            "OPENAI_API_KEY": local_gateway_key,
            "OPENAI_BASE_URL": args.base_url,
            "OPENAI_API_BASE": args.base_url,
            "HINDSIGHT_API_LLM_PROVIDER": "openai",
            "HINDSIGHT_API_LLM_MODEL": "gpt-5.5",
            "HINDSIGHT_API_LLM_API_KEY": local_gateway_key,
            "HINDSIGHT_API_LLM_BASE_URL": args.base_url,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
            # Real adapters use the pinned repository path.  The variable lets
            # offline fake adapters consume the exact same validation dataset.
            "LOCOMO_BASELINE_DATASET": str(dataset_path),
            "LOCOMO_DIAGNOSTICS_DIR": str(diagnostics_dir),
        }
    )
    if spec.runtime_python_env:
        env[spec.runtime_python_env] = str(args.runtime_python)
    return env


def collect_diagnostics(path: Path, *, required: bool) -> list[dict[str, Any]]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if required and not files:
        raise BaselineRunError(f"required adapter diagnostics are missing from {path}")
    records: list[dict[str, Any]] = []
    fatal = re.compile(
        r"traceback \(most recent call last\)|unhandled exception|server died",
        re.IGNORECASE,
    )
    for candidate in files:
        payload = candidate.read_bytes()
        text_payload = payload.decode("utf-8", errors="replace")
        if fatal.search(text_payload):
            raise BaselineRunError(f"diagnostic file contains a fatal marker: {candidate}")
        records.append({
            "path": str(candidate),
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        })
    return records


def validate_diagnostic_records(
    records: object, out_dir: Path, *, required: bool
) -> None:
    if not isinstance(records, list) or (required and not records):
        raise BaselineRunError("checkpoint diagnostics are missing")
    recorded_paths: set[Path] = set()
    parent_dirs: set[Path] = set()
    fatal = re.compile(
        r"traceback \(most recent call last\)|unhandled exception|server died",
        re.IGNORECASE,
    )
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise BaselineRunError("checkpoint diagnostic record is invalid")
        path = Path(record["path"]).expanduser().resolve()
        if not is_within(path, out_dir / "diagnostics"):
            raise BaselineRunError("checkpoint diagnostic path escapes diagnostics directory")
        if not path.is_file():
            raise BaselineRunError(f"checkpoint diagnostic file is missing: {path}")
        if record.get("sha256") != sha256_file(path) or record.get("bytes") != path.stat().st_size:
            raise BaselineRunError(f"checkpoint diagnostic hash differs: {path}")
        if fatal.search(path.read_text(encoding="utf-8", errors="replace")):
            raise BaselineRunError(f"checkpoint diagnostic contains a fatal marker: {path}")
        recorded_paths.add(path)
        parent_dirs.add(path.parent)
    actual_paths = {
        candidate.resolve()
        for directory in parent_dirs
        for candidate in directory.rglob("*")
        if candidate.is_file()
    }
    if actual_paths != recorded_paths:
        raise BaselineRunError("checkpoint diagnostic inventory differs")


def archive_path(path: Path, partials_dir: Path, reason: str) -> None:
    if not path.exists():
        return
    partials_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason).strip("._")[:80]
    if not safe_reason:
        safe_reason = "invalid"
    reason_hash = sha256_bytes(reason.encode("utf-8"))[:12]
    target = partials_dir / f"{path.name}.{safe_reason}.{reason_hash}.{stamp}"
    os.replace(path, target)


def invalidate_derived(out_dir: Path) -> None:
    for name in ("questions.json", "inputs_manifest.json", "audit.json"):
        try:
            (out_dir / name).unlink()
        except FileNotFoundError:
            pass


def sample_checkpoint_path(out_dir: Path, sample: int) -> Path:
    return out_dir / "checkpoints" / f"sample{sample}.json"


def completed_sample_state(
    out_dir: Path,
    sample: int,
    dataset_item: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    checkpoint_path = sample_checkpoint_path(out_dir, sample)
    output_path = out_dir / f"sample{sample}_questions.json"
    if not checkpoint_path.exists() or not output_path.exists():
        return None, "checkpoint_or_output_missing"
    try:
        checkpoint = read_json(checkpoint_path)
        if not isinstance(checkpoint, dict) or checkpoint.get("status") != "complete":
            return None, "checkpoint_not_complete"
        if checkpoint.get("fingerprint") != manifest["fingerprint"]:
            return None, "checkpoint_fingerprint_mismatch"
        expected_hashes = {
            "config_sha256": manifest["config_sha256"],
            "dataset_sha256": manifest["dataset"]["sha256"],
            "source_hashes_sha256": canonical_hash(manifest["source_hashes"]),
        }
        if checkpoint.get("frozen_hashes") != expected_hashes:
            return None, "checkpoint_frozen_hashes_mismatch"
        if checkpoint.get("sample") != sample:
            return None, "checkpoint_sample_mismatch"
        if checkpoint.get("output", {}).get("path") != output_path.name:
            return None, "checkpoint_output_path_mismatch"
        if checkpoint.get("output", {}).get("sha256") != sha256_file(output_path):
            return None, "checkpoint_output_hash_mismatch"
        log_value = checkpoint.get("log", {}).get("path")
        if not isinstance(log_value, str):
            return None, "checkpoint_log_path_missing"
        log_path = out_dir / log_value
        if not log_path.is_file():
            return None, "checkpoint_log_missing"
        if checkpoint.get("log", {}).get("sha256") != sha256_file(log_path):
            return None, "checkpoint_log_hash_mismatch"
        summary = validate_sample_file(
            output_path,
            dataset_item,
            sample,
            log_path=log_path,
            expected_builder=manifest["builder_contract"],
        )
        if checkpoint.get("validation") != summary:
            return None, "checkpoint_validation_mismatch"
        validate_diagnostic_records(
            checkpoint.get("diagnostics"),
            out_dir,
            required=METHOD_REGISTRY[manifest["method"]].requires_diagnostics,
        )
    except (BaselineRunError, OSError, TypeError, ValueError) as exc:
        return None, f"checkpoint_validation_failed:{exc}"
    return checkpoint, "complete"


def run_sample(
    sample: int,
    args: argparse.Namespace,
    spec: MethodSpec,
    python: Path,
    dataset: list[dict[str, Any]],
    out_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_lock: threading.Lock,
) -> tuple[int, bool, str]:
    fingerprint = manifest["fingerprint"]
    dataset_item = scoped_dataset_item(dataset[sample], args)
    existing, reason = completed_sample_state(
        out_dir, sample, dataset_item, manifest
    )
    if (
        existing is not None
        and spec.expects_llm
        and spec.usage_tracking == "in_process"
        and existing["validation"]["build_calls"] <= 0
    ):
        existing = None
        reason = "checkpoint_has_no_in_process_build_calls"
    if existing is not None:
        with manifest_lock:
            manifest["samples"][str(sample)] = existing
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
        return sample, True, "resumed_complete"

    output_path = out_dir / f"sample{sample}_questions.json"
    checkpoint_path = sample_checkpoint_path(out_dir, sample)
    if output_path.exists():
        archive_path(output_path, out_dir / "partials", reason)
    if checkpoint_path.exists():
        archive_path(checkpoint_path, out_dir / "partials", reason)
    invalidate_derived(out_dir)

    old_state = manifest.get("samples", {}).get(str(sample), {})
    previous_attempts = int(old_state.get("attempts", 0) or 0)
    last_error = "not_started"
    for local_attempt in range(1, args.retries + 1):
        attempt = previous_attempts + local_attempt
        token = uuid.uuid4().hex
        temporary_output = out_dir / "attempts" / f"sample{sample}.{token}.json"
        temporary_output.parent.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "logs" / f"sample{sample}.attempt{attempt}.{token}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_dir = out_dir / "diagnostics" / f"sample{sample}" / token
        diagnostics_dir.mkdir(parents=True, exist_ok=False)
        command = [
            str(python),
            str(spec.adapter),
            "--sample",
            str(sample),
            "--output",
            str(temporary_output),
            *spec.extra_args,
        ]
        if args.scope == "smoke":
            command.extend(["--max-sessions", str(args.max_sessions)])
            command.extend(["--questions-limit", str(args.questions_limit)])
        running_state = {
            "status": "running",
            "sample": sample,
            "fingerprint": fingerprint,
            "frozen_hashes": {
                "config_sha256": manifest["config_sha256"],
                "dataset_sha256": manifest["dataset"]["sha256"],
                "source_hashes_sha256": canonical_hash(manifest["source_hashes"]),
            },
            "attempts": attempt,
            "started_at": utc_now(),
            "command": command,
            "adapter_extra_args": list(spec.extra_args),
        }
        with manifest_lock:
            manifest["samples"][str(sample)] = running_state
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)

        returncode: int | None = None
        validation: dict[str, Any] | None = None
        try:
            with log_path.open("x", encoding="utf-8") as log:
                log.write(f"[{utc_now()}] sample={sample} attempt={attempt}\n")
                log.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
                log.flush()
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=experiment_env(args, args.dataset, spec, diagnostics_dir),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                returncode = process.returncode
                log.write(f"[{utc_now()}] returncode={returncode}\n")
                log.flush()
                os.fsync(log.fileno())
            if returncode != 0:
                raise BaselineRunError(f"adapter return code is {returncode}")
            if not temporary_output.is_file():
                raise BaselineRunError("adapter did not create its output")
            try:
                normalized = contract.normalize_sample_records(
                    read_json(temporary_output),
                    dataset_item,
                    sample,
                    builder_model=manifest["builder_contract"]["requested_model"],
                    builder_base_url=manifest["builder_contract"]["base_url"],
                    usage_tracking=spec.usage_tracking,
                )
            except contract.ContractError as exc:
                raise BaselineRunError(str(exc)) from exc
            atomic_json(temporary_output, normalized)
            validation = validate_sample_file(
                temporary_output,
                dataset_item,
                sample,
                log_path=log_path,
                expected_builder=manifest["builder_contract"],
            )
            if (
                spec.expects_llm
                and spec.usage_tracking == "in_process"
                and validation["build_calls"] <= 0
            ):
                raise BaselineRunError(
                    "LLM baseline has no in-process build-call evidence"
                )
            diagnostics = collect_diagnostics(
                diagnostics_dir, required=spec.requires_diagnostics
            )
            os.replace(temporary_output, output_path)
            checkpoint = {
                "schema_version": 1,
                "status": "complete",
                "sample": sample,
                "fingerprint": fingerprint,
                "frozen_hashes": running_state["frozen_hashes"],
                "attempts": attempt,
                "started_at": running_state["started_at"],
                "finished_at": utc_now(),
                "adapter_extra_args": list(spec.extra_args),
                "output": {
                    "path": output_path.name,
                    "sha256": sha256_file(output_path),
                    "bytes": output_path.stat().st_size,
                },
                "log": {
                    "path": str(log_path.relative_to(out_dir)),
                    "sha256": sha256_file(log_path),
                    "bytes": log_path.stat().st_size,
                },
                "validation": validation,
                "diagnostics": diagnostics,
            }
            atomic_json(checkpoint_path, checkpoint)
            with manifest_lock:
                manifest["samples"][str(sample)] = checkpoint
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
            return sample, True, "complete"
        except (BaselineRunError, OSError, subprocess.SubprocessError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if temporary_output.exists():
                archive_path(temporary_output, out_dir / "partials", "invalid_attempt")
            failed_attempt = {
                "schema_version": 1,
                "status": "failed",
                "sample": sample,
                "fingerprint": fingerprint,
                "frozen_hashes": running_state["frozen_hashes"],
                "attempts": attempt,
                "started_at": running_state["started_at"],
                "finished_at": utc_now(),
                "adapter_extra_args": list(spec.extra_args),
                "returncode": returncode,
                "error": last_error,
                "log": {
                    "path": str(log_path.relative_to(out_dir)),
                    "sha256": sha256_file(log_path) if log_path.is_file() else None,
                },
            }
            atomic_json(checkpoint_path, failed_attempt)
            with manifest_lock:
                manifest["samples"][str(sample)] = failed_attempt
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
    return sample, False, last_error


def validate_assembled(
    out_dir: Path,
    method: str,
    spec: MethodSpec,
    dataset: list[dict[str, Any]],
    expected_builder: dict[str, str],
) -> dict[str, Any]:
    questions_path = out_dir / "questions.json"
    assembly_manifest_path = out_dir / "inputs_manifest.json"
    payload = read_json(questions_path)
    assembly_manifest = read_json(assembly_manifest_path)
    if not isinstance(payload, list) or len(payload) != 10 + EXPECTED_QUESTIONS:
        raise BaselineRunError("assembled questions.json has an invalid record count")
    builds = payload[:10]
    questions = payload[10:]
    for sample, build in enumerate(builds):
        if not isinstance(build, dict) or build.get("question_id") != "_build_stats":
            raise BaselineRunError(f"assembled build record {sample} is invalid")
        if build.get("sample_index") != sample:
            raise BaselineRunError(f"assembled build record {sample} lacks sample_index")
    expected_ids: list[str] = []
    categories: Counter[int] = Counter()
    cursor = 0
    for sample, item in enumerate(dataset):
        for question_index, source in enumerate(item["qa"]):
            question_id = f"s{sample}_q{question_index}"
            expected_ids.append(question_id)
            record = questions[cursor]
            cursor += 1
            if not isinstance(record, dict) or record.get("question_id") != question_id:
                raise BaselineRunError(f"assembled question order differs at {question_id}")
            if record.get("question") != source.get("question"):
                raise BaselineRunError(f"assembled {question_id} question differs")
            if record.get("gold") != expected_gold(source):
                raise BaselineRunError(f"assembled {question_id} gold differs")
            if record.get("category") != source.get("category"):
                raise BaselineRunError(f"assembled {question_id} category differs")
            if record.get("evidence") != source.get("evidence"):
                raise BaselineRunError(f"assembled {question_id} evidence differs")
            categories[int(record["category"])] += 1
    if len(set(expected_ids)) != EXPECTED_QUESTIONS:
        raise BaselineRunError("assembled question IDs are not unique")
    if dict(sorted(categories.items())) != EXPECTED_CATEGORIES:
        raise BaselineRunError("assembled category inventory differs")
    if not isinstance(assembly_manifest, dict):
        raise BaselineRunError("inputs_manifest.json is not an object")
    if assembly_manifest.get("status") != "inputs_complete":
        raise BaselineRunError("assembler manifest is not inputs_complete")
    if assembly_manifest.get("method") != method:
        raise BaselineRunError("assembler manifest method differs")
    if assembly_manifest.get("output", {}).get("sha256") != sha256_file(questions_path):
        raise BaselineRunError("assembler manifest output hash differs")
    if assembly_manifest.get("adapter", {}).get("sha256") != sha256_file(spec.adapter):
        raise BaselineRunError("assembler manifest adapter hash differs")
    if assembly_manifest.get("contract") != {
        "path": str(CONTRACT.resolve()),
        "sha256": sha256_file(CONTRACT),
    }:
        raise BaselineRunError("assembler manifest contract hash differs")
    if assembly_manifest.get("builder") != expected_builder:
        raise BaselineRunError("assembler manifest builder provenance differs")
    return {
        "status": "inputs_complete",
        "questions": {
            "path": questions_path.name,
            "sha256": sha256_file(questions_path),
            "records": len(payload),
            "question_records": EXPECTED_QUESTIONS,
            "primary_cat1_4": EXPECTED_PRIMARY,
            "adversarial_cat5": EXPECTED_ADVERSARIAL,
        },
        "manifest": {
            "path": assembly_manifest_path.name,
            "sha256": sha256_file(assembly_manifest_path),
        },
    }


def assemble_inputs(
    args: argparse.Namespace,
    python: Path,
    spec: MethodSpec,
    dataset: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    invalidate_derived(out_dir)
    log_path = out_dir / "assembly.log"
    command = [
        str(python),
        str(ASSEMBLER),
        "--method",
        args.method,
        "--input-dir",
        str(out_dir),
        "--output",
        str(out_dir / "questions.json"),
        "--manifest",
        str(out_dir / "inputs_manifest.json"),
        "--dataset",
        str(args.dataset),
        "--adapter",
        str(spec.adapter),
        "--builder-model",
        builder_contract(spec, args.upstream_base)["requested_model"],
        "--builder-base-url",
        builder_contract(spec, args.upstream_base)["base_url"],
        "--usage-tracking",
        spec.usage_tracking,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{utc_now()}] returncode={process.returncode}\n")
        log.flush()
        os.fsync(log.fileno())
    if process.returncode != 0:
        invalidate_derived(out_dir)
        raise BaselineRunError(
            f"assembler returned {process.returncode}; see {log_path}"
        )
    result = validate_assembled(
        out_dir,
        args.method,
        spec,
        dataset,
        builder_contract(spec, args.upstream_base),
    )
    result["log"] = {
        "path": log_path.name,
        "sha256": sha256_file(log_path),
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full build+retrieve LoCoMo baselines with GPT-5.5"
    )
    parser.add_argument("--method", required=True, choices=sorted(METHOD_REGISTRY))
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--samples", default=None, help="smoke-only comma/range selection")
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--questions-limit", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--gateway-root",
        type=Path,
        default=None,
        help="active OpenAI GPT-5.5 Flex gateway result root (required for LLM methods)",
    )
    parser.add_argument(
        "--allow-model-requests",
        action="store_true",
        help="required acknowledgement before an LLM baseline may send requests",
    )
    parser.add_argument(
        "--proxy-log",
        type=Path,
        default=None,
        help="deprecated; formal runs always use a managed exclusive log",
    )
    parser.add_argument("--sample-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if args.proxy_log is not None:
        parser.error("--proxy-log is not accepted; the launcher manages an exclusive log")
    if args.sample_workers < 1 or args.retries < 1:
        parser.error("sample-workers and retries must be positive")
    spec = METHOD_REGISTRY[args.method]
    if spec.expects_llm and not args.allow_model_requests:
        parser.error("LLM baselines require --allow-model-requests")
    if args.scope == "formal" and args.method not in FORMAL_METHODS:
        parser.error(
            "formal scope is preregistered for: " + ", ".join(FORMAL_METHODS)
        )
    if spec.expects_llm and args.gateway_root is None:
        parser.error("LLM baselines require --gateway-root for the fixed Flex gateway")
    if args.sample_workers > spec.max_sample_workers:
        parser.error(
            f"{args.method} allows at most {spec.max_sample_workers} sample worker(s)"
        )
    if args.scope == "formal":
        if args.samples is not None:
            parser.error("formal scope always uses samples 0-9")
        args.samples = list(range(10))
        args.max_sessions = None
        args.questions_limit = None
    else:
        if args.max_sessions < 1 or args.questions_limit < 1:
            parser.error("smoke limits must be positive")
        try:
            args.samples = parse_sample_selection(args.samples or "0")
        except (argparse.ArgumentTypeError, ValueError) as exc:
            parser.error(str(exc))
    args.dataset = args.dataset.expanduser().resolve()
    if path_identity(args.dataset) != path_identity(DATASET):
        parser.error(f"dataset must be the pinned LoCoMo file: {DATASET}")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.gateway_root = (
        args.gateway_root.expanduser().resolve()
        if args.gateway_root is not None
        else None
    )
    args.upstream_base = "not_applicable"
    args.gateway_contract = None
    return args


def parse_sample_selection(value: str) -> list[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)
            if stop < start:
                raise argparse.ArgumentTypeError("sample ranges must be ascending")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(part))
    if not selected or min(selected) < 0 or max(selected) > 9:
        raise argparse.ArgumentTypeError("samples must be in 0..9")
    return sorted(selected)


def scoped_dataset_item(
    item: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if args.scope == "formal":
        return item
    result = dict(item)
    result["qa"] = list(item["qa"][: args.questions_limit])
    return result


def initialize_manifest(
    args: argparse.Namespace,
    spec: MethodSpec,
    python: Path,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    del dataset
    preregistration = load_formal_preregistration()
    preregistration_frozen_at = parse_preregistration_time(
        preregistration["frozen_at"]
    )
    created_at = utc_now()
    if preregistration_frozen_at > datetime.fromisoformat(created_at):
        raise BaselineRunError(
            "formal matrix preregistration was frozen after run initialization"
        )
    preregistration_source = preregistration_record(preregistration)
    source_hashes = compute_source_hashes(spec)
    runtime = runtime_fingerprint(spec, python, args.runtime_python)
    data_hash = sha256_file(args.dataset)
    task_scope = (
        "build_and_retrieve_only"
        if args.scope == "formal"
        else "smoke_build_and_retrieve_only"
    )
    expected_builder = builder_contract(spec, args.upstream_base)
    gateway_contract = (
        args.gateway_contract
        if spec.expects_llm
        else {"status": "not_applicable_llm_free_method"}
    )
    config = {
        "method": args.method,
        "scope": args.scope,
        "task_scope": task_scope,
        "samples": args.samples,
        "max_sessions": args.max_sessions,
        "questions_limit": args.questions_limit,
        "builder_model": expected_builder["requested_model"],
        "upstream_base": args.upstream_base.rstrip("/"),
        "gateway_contract": gateway_contract,
        "proxy_mode": "managed_exclusive" if spec.expects_llm else "not_applicable",
        "python": str(python),
        "python_sha256": sha256_file(python),
        "effective_python": str(args.runtime_python),
        "effective_python_sha256": sha256_file(args.runtime_python),
        "adapter": source_key(spec.adapter),
        "adapter_extra_args": list(spec.extra_args),
        "expects_llm_calls": spec.expects_llm,
        "usage_tracking": spec.usage_tracking,
        "sample_workers": args.sample_workers,
        "retries": args.retries,
        "dataset": str(args.dataset),
        "legacy_proxy_log": str(args.proxy_log) if args.proxy_log is not None else None,
        "answerer": "not_run",
        "judge": "not_run",
        "formal_method_matrix": list(FORMAL_METHODS),
        "structured_preregistration": STRUCTURED_PREREGISTRATION,
        "preregistration": preregistration_source,
    }
    config_hash = canonical_hash(config)
    fingerprint = canonical_hash(
        {
            "config_sha256": config_hash,
            "dataset_sha256": data_hash,
            "source_hashes": source_hashes,
            "runtime": runtime,
        }
    )
    return {
        "schema_version": 1,
        "benchmark": "LoCoMo",
        "method": args.method,
        "scope": args.scope,
        "task_scope": task_scope,
        "status": "initialized",
        "builder_model": expected_builder["requested_model"],
        "builder_contract": expected_builder,
        "provider_contract": gateway_contract,
        "git_commit": git_head(),
        "created_at": created_at,
        "updated_at": created_at,
        "output_dir": str(args.output_dir),
        "config": config,
        "config_sha256": config_hash,
        "dataset": {
            "path": str(args.dataset),
            "sha256": data_hash,
            "samples": 10,
            "questions": EXPECTED_QUESTIONS,
            "primary_cat1_4": EXPECTED_PRIMARY,
            "adversarial_cat5": EXPECTED_ADVERSARIAL,
            "categories": {str(key): value for key, value in EXPECTED_CATEGORIES.items()},
        },
        "source_hashes": source_hashes,
        "runtime": runtime,
        "fingerprint": fingerprint,
        "preregistration": preregistration_source,
        "experiment_matrix": {
            "formal_methods": list(FORMAL_METHODS),
            "structured_preregistration": STRUCTURED_PREREGISTRATION,
            "other_registered_methods": [
                method for method in METHOD_REGISTRY if method not in FORMAL_METHODS
            ],
            "other_method_scope": "smoke_or_blocked_only",
        },
        "samples": {},
        "proxy_evidence": {
            "mode": "managed_exclusive" if spec.expects_llm else "not_applicable",
            "exact_run_linkage": bool(spec.expects_llm),
            "invocations": [],
        },
        "assembly": {"status": "pending"},
        "evaluation": {
            "answerer": "not_run",
            "primary_judge": "not_run",
            "secondary_judge": "not_run",
        },
    }


def merge_existing_manifest(
    new_manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    if not manifest_path.exists():
        return new_manifest
    old = read_json(manifest_path)
    if not isinstance(old, dict):
        raise BaselineRunError("existing run_manifest.json is not an object")
    if old.get("fingerprint") != new_manifest["fingerprint"]:
        try:
            (manifest_path.parent / "audit.json").unlink()
        except FileNotFoundError:
            pass
        raise BaselineRunError(
            "existing run manifest has stale data/config/source hashes; use a new output directory"
        )
    for key in (
        "benchmark",
        "method",
        "task_scope",
        "scope",
        "builder_model",
        "builder_contract",
        "provider_contract",
        "config_sha256",
        "dataset",
        "preregistration",
        "experiment_matrix",
        "source_hashes",
        "runtime",
    ):
        if old.get(key) != new_manifest.get(key):
            raise BaselineRunError(f"existing run manifest differs in {key}")
    new_manifest["created_at"] = old.get("created_at", new_manifest["created_at"])
    if isinstance(old.get("samples"), dict):
        new_manifest["samples"] = old["samples"]
    old_proxy = old.get("proxy_evidence", {})
    if isinstance(old_proxy, dict) and isinstance(old_proxy.get("invocations"), list):
        new_manifest["proxy_evidence"]["invocations"] = old_proxy["invocations"]
    return new_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    managed_proxy: subprocess.Popen[str] | None = None
    proxy_record: dict[str, Any] | None = None
    provider_lock = None
    try:
        spec = METHOD_REGISTRY[args.method]
        python = resolve_python(args.python)
        args.runtime_python = resolve_runtime_python(spec, python)
        if spec.expects_llm:
            try:
                provider_lock = flex_evidence.acquire_consumer_lock(
                    args.gateway_root
                )
            except flex_evidence.EvidenceError as exc:
                raise BaselineRunError(str(exc)) from exc
            args.gateway_contract = load_flex_gateway_contract(args.gateway_root)
            args.upstream_base = args.gateway_contract["origin"]
        else:
            args.gateway_contract = None
            args.upstream_base = "not_applicable"
        validate_run_paths(args, python, args.runtime_python)
        dataset = load_dataset(args.dataset)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (args.output_dir / ".launcher.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BaselineRunError(
                f"another launcher is active in {args.output_dir}"
            ) from exc

        manifest_path = args.output_dir / "run_manifest.json"
        try:
            (args.output_dir / "audit.json").unlink()
        except FileNotFoundError:
            pass
        manifest = merge_existing_manifest(
            initialize_manifest(args, spec, python, dataset), manifest_path
        )
        recover_interrupted_managed_proxies(manifest, args.output_dir)
        invalidate_derived(args.output_dir)
        manifest["status"] = "running"
        manifest["assembly"] = {"status": "pending"}
        if spec.expects_llm:
            managed_proxy, proxy_record = start_managed_proxy(args, args.output_dir)
            args.base_url = proxy_record["base_url"]
            manifest["proxy_evidence"]["invocations"].append(proxy_record)
        else:
            args.base_url = "not_applicable"
        atomic_json(manifest_path, manifest)

        manifest_lock = threading.Lock()
        failures: list[tuple[int, str]] = []
        try:
            with ThreadPoolExecutor(max_workers=min(args.sample_workers, 10)) as executor:
                futures = {
                    executor.submit(
                        run_sample,
                        sample,
                        args,
                        spec,
                        python,
                        dataset,
                        args.output_dir,
                        manifest,
                        manifest_path,
                        manifest_lock,
                    ): sample
                    for sample in args.samples
                }
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        _, accepted, detail = future.result()
                    except Exception as exc:  # noqa: BLE001
                        accepted = False
                        detail = f"internal error: {type(exc).__name__}: {exc}"
                    print(
                        f"sample {sample}: {'complete' if accepted else 'failed'} ({detail})",
                        flush=True,
                    )
                    if not accepted:
                        failures.append((sample, detail))
        finally:
            if managed_proxy is not None and proxy_record is not None:
                finish_managed_proxy(managed_proxy, proxy_record, args.output_dir)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)

        if failures:
            invalidate_derived(args.output_dir)
            manifest["status"] = "failed"
            manifest["finished_at"] = utc_now()
            manifest["failures"] = [
                {"sample": sample, "error": detail} for sample, detail in failures
            ]
            manifest["assembly"] = {"status": "not_created_due_to_sample_failures"}
            atomic_json(manifest_path, manifest)
            return 1

        for sample in args.samples:
            checkpoint, reason = completed_sample_state(
                args.output_dir,
                sample,
                scoped_dataset_item(dataset[sample], args),
                manifest,
            )
            if checkpoint is None:
                raise BaselineRunError(
                    f"post-run validation rejected sample {sample}: {reason}"
                )
            manifest["samples"][str(sample)] = checkpoint
        if args.scope == "formal":
            manifest["assembly"] = assemble_inputs(
                args, python, spec, dataset, args.output_dir
            )
            manifest["status"] = "inputs_complete"
        else:
            manifest["assembly"] = {"status": "not_applicable_smoke_scope"}
            manifest["status"] = "smoke_complete"
        manifest["finished_at"] = utc_now()
        manifest.pop("failures", None)
        atomic_json(manifest_path, manifest)
        print(
            f"{args.method}: {args.scope} scope complete for samples {args.samples}; "
            "answerer and judges were not invoked",
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        if managed_proxy is not None and proxy_record is not None and managed_proxy.poll() is None:
            finish_managed_proxy(managed_proxy, proxy_record, args.output_dir)
        # A stale/incompatible existing manifest is rejected before this
        # invocation owns the run.  Preserve all of that run's artifacts.
        if (
            "args" in locals()
            and isinstance(args.output_dir, Path)
            and "manifest" in locals()
            and isinstance(manifest, dict)
        ):
            invalidate_derived(args.output_dir)
            manifest_path = args.output_dir / "run_manifest.json"
            manifest["status"] = "failed"
            manifest["finished_at"] = utc_now()
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["assembly"] = {"status": "not_complete"}
            atomic_json(manifest_path, manifest)
        print(f"baseline launcher failed: {exc}", file=os.sys.stderr)
        return 1
    finally:
        if provider_lock is not None:
            provider_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
