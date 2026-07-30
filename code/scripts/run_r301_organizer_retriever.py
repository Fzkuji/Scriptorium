#!/usr/bin/env python3
"""Run the preregistered R301 fixed-entry organizer x retriever experiment.

Formal mode consumes an independently audited all-ten R207 artifact.  The
synthetic mode exercises the same 2 x 2 x 3 matrix, fixed answer boundary,
20,000-token gate, durable ledger, resume checks, and interaction analysis
without a network request.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import controlled_locomo_answer_contract as answer_contract  # noqa: E402
from src.evaluation.durable_model_ledger import (  # noqa: E402
    HashChainLedger,
    atomic_json as durable_atomic_json,
    ledger_state,
    normalize_usage,
    proxy_evidence,
    proxy_prefix,
    read_ledger,
    sha256_file,
)
from src.evaluation.metrics import f1_set  # noqa: E402
from src.evaluation.visible_token_budget import (  # noqa: E402
    DeliveryResult,
    TokenCounter,
    VisibleTokenBudgetGate,
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
DEFAULT_PREREG = ROOT / "paper/refine-logs/R301_PREREGISTRATION.json"
M4_COMPARISON_PREREG = (
    ROOT / "paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json"
)
EXPECTED_M4_COMPARISON_SHA256 = (
    "b706535a61972c368547caf6611bac397959ae3fbfe71142adb8bfb95b449c22"
)
DEFAULT_DATA = ROOT / "benchmarks/locomo/data/locomo10.json"
DEFAULT_MAPPING = (
    ROOT
    / "results/paper-experiments-20260714/evidence-mapping/v1/"
    "evidence_mapping.v1.questions.jsonl"
)
PROXY_SCRIPT = ROOT / "scripts/controlled_r301_model_proxy.py"
GPT55_PROXY_SCRIPT = ROOT / "scripts/controlled_gpt55_run_proxy.py"
EXPECTED_DATA_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
EXPECTED_MAPPING_SHA256 = (
    "3291df579bc20d0678f99b3361ac6d287beee583e4d3dff93b996f4522e56318"
)
ORGANIZER_IDS = ("O55", "O4o")
RETRIEVER_IDS = ("R55", "R4o")
CELLS = ("O55-R55", "O4o-R55", "O55-R4o", "O4o-R4o")
REPLICATES = (1, 2, 3)
VISIBLE_BUDGET = 20_000

ORGANIZER_SYSTEM = """R301_ORGANIZER
Assign every fixed entry to one topic path for future question retrieval.
Return one JSON object with key placements. placements must preserve the input
entry order and contain only entry_id and topic_path. Do not alter entry text,
source identifiers, dates, or entry identifiers. Topic paths are relative,
slash-separated, at most four components, and use at most thirty distinct
paths. Do not include commentary."""

NAVIGATION_SYSTEM = """R301_NAVIGATE
Choose up to four topic paths that are most useful for the question. You see
only the path inventory that passed the visible-token gate. Return one JSON
object with key topic_paths in priority order. Do not answer the question."""

SELECTION_SYSTEM = """R301_SELECT
Select up to eight entry IDs from the opened-path observation that are useful
for the question. Return one JSON object with ordered entry_ids and source_ids.
Every source ID must occur in a selected entry. Do not answer the question."""


class R301Error(RuntimeError):
    """Raised when an R301 contract cannot be proven."""


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


def utc_now() -> str:
    return answer_contract.utc_now()


def read_json(path: Path) -> dict[str, Any]:
    value = answer_contract.read_json(path)
    if not isinstance(value, dict):
        raise R301Error(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise R301Error(f"JSONL artifact is unavailable: {path}")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise R301Error(f"JSONL artifact has an incomplete line: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R301Error(f"invalid JSONL {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise R301Error(f"non-object JSONL {path}:{line_number}")
        records.append(value)
    return records


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    try:
        answer_contract.atomic_json_no_clobber(path, value)
    except FileExistsError as exc:
        raise R301Error(f"refusing to overwrite artifact: {path}") from exc


def atomic_jsonl_no_clobber(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    answer_contract.reject_symlink_components(path)
    if path.exists() or path.is_symlink():
        raise R301Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write((canonical_json(record) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def safe_relative(value: str, *, label: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise R301Error(f"unsafe {label}: {value!r}")
    return path


def assert_safe_tree(root: Path) -> None:
    answer_contract.reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise R301Error(f"tree is not a regular directory: {root}")
    names: dict[str, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        key = relative.casefold()
        if key in names and names[key] != relative:
            raise R301Error(f"case-colliding paths: {names[key]} / {relative}")
        names[key] = relative
        if path.is_symlink():
            raise R301Error(f"symlink is forbidden: {relative}")
        if path.is_file():
            stat = path.stat(follow_symlinks=False)
            if stat.st_nlink != 1:
                raise R301Error(f"hardlink is forbidden: {relative}")
            inode = (stat.st_dev, stat.st_ino)
            if inode in inodes:
                raise R301Error(
                    f"shared file inode: {inodes[inode]} / {relative}"
                )
            inodes[inode] = relative
        elif not path.is_dir():
            raise R301Error(f"special filesystem node is forbidden: {relative}")


def tree_descriptor(root: Path) -> dict[str, Any]:
    assert_safe_tree(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat(follow_symlinks=False).st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"file_count": len(files), "tree_sha256": value_sha256(files)}


def validate_preregistration(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_PREREG.resolve():
        raise R301Error("R301 must use the frozen repository preregistration")
    payload = read_json(path)
    if payload.get("schema") != PREREG_SCHEMA or payload.get("status") != "frozen":
        raise R301Error("R301 preregistration identity differs")
    unhashed = dict(payload)
    recorded_hash = unhashed.pop("protocol_content_sha256", None)
    if recorded_hash != value_sha256(unhashed):
        raise R301Error("R301 preregistration content hash differs")
    expected_plan = ROOT / str(payload.get("plan", {}).get("path", ""))
    if (
        not expected_plan.is_file()
        or sha256_file(expected_plan) != payload["plan"].get("sha256")
    ):
        raise R301Error("experiment-plan hash differs from preregistration")
    if tuple(payload.get("organizers", {})) != ORGANIZER_IDS:
        raise R301Error("organizer matrix differs")
    if tuple(payload.get("retrievers", {})) != RETRIEVER_IDS:
        raise R301Error("retriever matrix differs")
    if tuple(payload.get("cells", ())) != CELLS:
        raise R301Error("cell order differs")
    if tuple(payload.get("organizer_replicates", ())) != REPLICATES:
        raise R301Error("organizer replicates differ")
    gate = payload.get("visible_token_gate")
    if not isinstance(gate, dict) or gate.get("hard_cap_tokens") != VISIBLE_BUDGET:
        raise R301Error("visible-token budget differs")
    if gate.get("tokenizer") != answer_contract.formal_token_counter().identity:
        raise R301Error("formal tokenizer identity differs")
    answerer = payload.get("answerer")
    if (
        not isinstance(answerer, dict)
        or answerer.get("requested_model") != "gpt-5.5"
        or answerer.get("temperature") != 0
        or answerer.get("fixed_across_cells") is not True
        or answerer.get("prompt_template_sha256")
        != hashlib.sha256(
            answer_contract.ANSWER_PROMPT.encode("utf-8")
        ).hexdigest()
        or answerer.get("prompt_may_use") != "DeliveryResult.delivered_text_only"
    ):
        raise R301Error("fixed-answerer preregistration differs")
    interaction = payload.get("interaction")
    if not isinstance(interaction, dict) or interaction.get("formula") != (
        "(O55-R55 - O4o-R55) - (O55-R4o - O4o-R4o)"
    ):
        raise R301Error("interaction formula differs")
    if (
        interaction.get("bootstrap_cluster") != "conversation"
        or interaction.get("bootstrap_resamples") != 10_000
        or interaction.get("bootstrap_seed") != 20_260_714
        or interaction.get("confidence_level") != 0.95
    ):
        raise R301Error("interaction resampling policy differs")
    binding = payload.get("m4_comparison_preregistration")
    expected_binding = {
        "path": "paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json",
        "sha256": EXPECTED_M4_COMPARISON_SHA256,
        "section": "m5_interaction",
    }
    if binding != expected_binding or sha256_file(
        M4_COMPARISON_PREREG
    ) != EXPECTED_M4_COMPARISON_SHA256:
        raise R301Error("M4 comparison preregistration binding differs")
    m4 = read_json(M4_COMPARISON_PREREG)
    if m4.get("status") != "frozen_before_formal_scores" or m4.get(
        "m5_interaction"
    ) != {
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
    }:
        raise R301Error("M4 m5_interaction rule differs")
    common = m4.get("common_protocol")
    if not isinstance(common, dict) or any(
        (
            common.get("continuous_estimand")
            != "question_weighted_mean_left_minus_right",
            common.get("clustered_bootstrap_repetitions") != 10_000,
            common.get("clustered_bootstrap_seed") != 20_260_714,
            common.get("confidence") != 0.95,
            common.get("pairing")
            != "exact question_id; LoCoMo resampling cluster is conversation",
            common.get("missingness")
            != "no imputation; incomplete or unequal question sets invalidate a formal comparison",
        )
    ):
        raise R301Error("M4 common clustered-comparison protocol differs")
    return payload


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "auditor": ROOT / "scripts/audit_r301_organizer_retriever.py",
        "proxy": PROXY_SCRIPT,
        "gpt55_proxy": GPT55_PROXY_SCRIPT,
        "preregistration": DEFAULT_PREREG,
        "m4_comparison_preregistration": M4_COMPARISON_PREREG,
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
    output = {}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise R301Error(f"required source is missing or a symlink: {path}")
        output[name] = sha256_file(path)
    return output


def parse_json_object(text: str, *, label: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise R301Error(f"{label} response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise R301Error(f"{label} response is not an object")
    return value


def build_turn_index(conversation: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for key, turns in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if match is None or not isinstance(turns, list):
            continue
        date = str(conversation.get(f"session_{match.group(1)}_date_time", ""))
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise R301Error("conversation turn is not an object")
            dia_id = str(turn.get("dia_id", ""))
            if re.fullmatch(r"D\d+:\d+", dia_id) is None or dia_id in output:
                raise R301Error(f"invalid or duplicate source ID: {dia_id!r}")
            output[dia_id] = {
                "date": date,
                "speaker": str(turn.get("speaker", "")),
                "text": str(turn.get("text", "")),
            }
    if not output:
        raise R301Error("conversation contains no source-indexed turns")
    return output


def resolve_sources(
    source_ids: Sequence[str], turn_index: Mapping[str, Mapping[str, str]]
) -> str:
    blocks = []
    for source_id in source_ids:
        turn = turn_index.get(source_id)
        if turn is None:
            raise R301Error(f"selected source is absent from conversation: {source_id}")
        blocks.append(
            f"[{source_id}] ({turn['date']}) {turn['speaker']}: {turn['text']}"
        )
    return "\n".join(blocks)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    requested_model: str
    accepted_actual_model_regex: str
    temperature: float

    def accepts(self, actual_model: str) -> bool:
        return re.fullmatch(self.accepted_actual_model_regex, actual_model) is not None


def model_specs(preregistration: Mapping[str, Any]) -> dict[str, ModelSpec]:
    output: dict[str, ModelSpec] = {}
    for collection in ("organizers", "retrievers"):
        for model_id, value in preregistration[collection].items():
            output[model_id] = ModelSpec(
                model_id=model_id,
                requested_model=str(value["requested_model"]),
                accepted_actual_model_regex=str(value["accepted_actual_model_regex"]),
                temperature=float(value["temperature"]),
            )
    answerer = preregistration["answerer"]
    output["A55"] = ModelSpec(
        model_id="A55",
        requested_model=str(answerer["requested_model"]),
        accepted_actual_model_regex=str(answerer["accepted_actual_model_regex"]),
        temperature=float(answerer["temperature"]),
    )
    return output


class ModelClient(Protocol):
    def complete(
        self,
        *,
        payload: Mapping[str, Any],
        logical_call_id: str,
        question_id: str | None,
        spec: ModelSpec,
    ) -> dict[str, Any]: ...


class HttpModelClient:
    def __init__(self, endpoints: Mapping[str, str], timeout_seconds: int = 360):
        self.endpoints = dict(endpoints)
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def complete(
        self,
        *,
        payload: Mapping[str, Any],
        logical_call_id: str,
        question_id: str | None,
        spec: ModelSpec,
    ) -> dict[str, Any]:
        endpoint = self.endpoints[spec.requested_model].rstrip("/")
        body = canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer x",
                "X-Controlled-Logical-Call-ID": logical_call_id,
                "X-Controlled-Question-ID": question_id or "organizer",
            },
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            raise R301Error(f"model HTTP {exc.code}: {raw[:1000]!r}") from exc
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R301Error("model response is not JSON") from exc
        if not isinstance(value, dict):
            raise R301Error("model response root is not an object")
        return value


class FakeModelClient:
    """Deterministic model substitute used only by synthetic validation."""

    def complete(
        self,
        *,
        payload: Mapping[str, Any],
        logical_call_id: str,
        question_id: str | None,
        spec: ModelSpec,
    ) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise R301Error("fake request has no messages")
        system = str(messages[0].get("content", ""))
        user = str(messages[-1].get("content", ""))
        if system.startswith("R301_ORGANIZER"):
            data = json.loads(user)
            placements = []
            for index, entry in enumerate(data["entries"]):
                relevant = "Pixel" in entry["summary_inline"]
                if data["organizer_id"] == "O55":
                    path = "people/ari/pets" if relevant else f"animals/misc-{index}"
                else:
                    path = "animals/pixel" if relevant else f"people/misc-{index}"
                placements.append({"entry_id": entry["entry_id"], "topic_path": path})
            content = canonical_json({"placements": placements})
        elif system.startswith("R301_NAVIGATE"):
            data = json.loads(user)
            try:
                visible_paths = json.loads(data["path_inventory_text"] or "{}").get(
                    "paths", []
                )
            except json.JSONDecodeError:
                visible_paths = []
            prefix = "people/" if data["retriever_id"] == "R55" else "animals/"
            candidates = [
                item["topic_path"]
                for item in visible_paths
                if item["topic_path"].startswith(prefix)
            ]
            content = canonical_json({"topic_paths": candidates[:1]})
        elif system.startswith("R301_SELECT"):
            data = json.loads(user)
            text = str(data.get("opened_entries_text") or "")
            entry_ids = re.findall(r"entry_id=([^\s]+)", text)[:8]
            source_ids = []
            for value in re.findall(r"source_ids=([^\n]+)", text):
                source_ids.extend(re.findall(r"D\d+:\d+", value))
            content = canonical_json(
                {"entry_ids": entry_ids, "source_ids": list(dict.fromkeys(source_ids))}
            )
        else:
            content = "<answer>Pixel</answer>" if "Pixel" in user else "<answer>Unknown</answer>"
        actual = (
            "openai/gpt-4o-mini-2024-07-18"
            if spec.requested_model == "openai/gpt-4o-mini"
            else "gpt-5.5"
        )
        prompt_tokens = max(1, len(canonical_json(payload).encode("utf-8")) // 4)
        completion_tokens = max(1, len(content.encode("utf-8")) // 4)
        return {
            "id": f"fake-{hashlib.sha256(logical_call_id.encode()).hexdigest()[:24]}",
            "model": actual,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def response_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise R301Error("model response must contain one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
        raise R301Error("model response did not finish with stop")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise R301Error("model response content is empty")
    return content


class CallRecorder:
    def __init__(
        self,
        *,
        root: Path,
        ledger: HashChainLedger,
        tokenizer: TokenCounter,
        client: ModelClient,
        proxy_logs: Mapping[str, Path | None],
        formal: bool,
    ) -> None:
        self.root = root
        self.ledger = ledger
        self.tokenizer = tokenizer
        self.client = client
        self.proxy_logs = dict(proxy_logs)
        self.formal = formal
        self.ordinals: defaultdict[str, int] = defaultdict(int)

    def call(
        self,
        *,
        operation_id: str,
        question_id: str | None,
        spec: ModelSpec,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> dict[str, Any]:
        self.ordinals[operation_id] += 1
        ordinal = self.ordinals[operation_id]
        logical_call_id = (
            f"{self.ledger.run_id}:{operation_id}:call-{ordinal:04d}"
        )
        filename = hashlib.sha256(logical_call_id.encode()).hexdigest()[:32]
        request_path = self.root / "calls" / f"{filename}.request.json"
        response_path = self.root / "calls" / f"{filename}.response.json"
        payload = {
            "model": spec.requested_model,
            "messages": [dict(message) for message in messages],
            "temperature": spec.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
            if spec.model_id != "A55"
            else None,
        }
        if payload["response_format"] is None:
            payload.pop("response_format")
        visible_payload = {
            "messages": payload["messages"],
            **(
                {"response_format": payload["response_format"]}
                if "response_format" in payload
                else {}
            ),
        }
        request_artifact = {
            "schema": "nativemem.r301-model-request.v1",
            "logical_call_id": logical_call_id,
            "operation_id": operation_id,
            "question_id": question_id,
            "model_id": spec.model_id,
            "requested_model": spec.requested_model,
            "accepted_actual_model_regex": spec.accepted_actual_model_regex,
            "payload": payload,
            "model_visible_sha256": value_sha256(visible_payload),
            "local_visible_tokens": self.tokenizer.count(
                canonical_json(visible_payload)
            ),
            "tokenizer": self.tokenizer.identity,
        }
        request_sha = durable_atomic_json(request_path, request_artifact)
        proxy_log = self.proxy_logs.get(spec.requested_model)
        started_prefix = proxy_prefix(proxy_log)
        self.ledger.append(
            "model_call_started",
            operation_id=operation_id,
            logical_call_id=logical_call_id,
            question_id=question_id,
            model_id=spec.model_id,
            requested_model=spec.requested_model,
            accepted_actual_model_regex=spec.accepted_actual_model_regex,
            request_path=str(request_path.relative_to(self.root)),
            request_sha256=request_sha,
            local_visible_tokens=request_artifact["local_visible_tokens"],
            tokenizer=self.tokenizer.identity,
            proxy_log_start=started_prefix,
        )
        started = time.monotonic()
        try:
            response = self.client.complete(
                payload=payload,
                logical_call_id=logical_call_id,
                question_id=question_id,
                spec=spec,
            )
            response_artifact = {
                "schema": "nativemem.r301-model-response.v1",
                "logical_call_id": logical_call_id,
                "response": response,
            }
            response_sha = durable_atomic_json(response_path, response_artifact)
            evidence = proxy_evidence(
                proxy_log,
                logical_call_id=logical_call_id,
                formal=self.formal,
            )
            actual_model = str(response.get("model", ""))
            response_id = str(response.get("id", ""))
            if not response_id or not spec.accepts(actual_model):
                raise R301Error(
                    f"actual model is not bound to {spec.model_id}: {actual_model!r}"
                )
            usage = normalize_usage(response.get("usage"))
            if self.formal:
                successful = [
                    event
                    for event in evidence["events"]
                    if event.get("status") == "success"
                ]
                if len(successful) != 1:
                    raise R301Error("formal call lacks one successful proxy event")
                event = successful[0]
                if (
                    event.get("logical_call_id") != logical_call_id
                    or event.get("question_id") != (question_id or "organizer")
                    or event.get("requested_model") != spec.requested_model
                    or event.get("actual_model") != actual_model
                    or event.get("response_id") != response_id
                    or event.get("request_sha256")
                    != hashlib.sha256(
                        canonical_json(payload).encode("utf-8")
                    ).hexdigest()
                    or normalize_usage(event.get("usage")) != usage
                ):
                    raise R301Error("proxy and response identities differ")
            content = response_content(response)
            latency_s = round(time.monotonic() - started, 6)
            self.ledger.append(
                "model_call_finished",
                operation_id=operation_id,
                logical_call_id=logical_call_id,
                question_id=question_id,
                model_id=spec.model_id,
                response_path=str(response_path.relative_to(self.root)),
                response_sha256=response_sha,
                response_id=response_id,
                response_model=actual_model,
                usage=usage,
                latency_s=latency_s,
                proxy_evidence=evidence,
            )
            return {
                "content": content,
                "logical_call_id": logical_call_id,
                "response_id": response_id,
                "requested_model": spec.requested_model,
                "actual_model": actual_model,
                "usage": usage,
                "latency_s": latency_s,
                "proxy_evidence": evidence,
            }
        except BaseException as exc:
            if not response_path.exists():
                response_sha = None
            self.ledger.append(
                "model_call_failed",
                operation_id=operation_id,
                logical_call_id=logical_call_id,
                question_id=question_id,
                model_id=spec.model_id,
                response_path=(
                    str(response_path.relative_to(self.root))
                    if response_path.exists()
                    else None
                ),
                response_sha256=(
                    sha256_file(response_path) if response_path.exists() else None
                ),
                latency_s=round(time.monotonic() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def validate_bank(bank: Mapping[str, Any], *, sample: int) -> dict[str, Any]:
    if (
        bank.get("schema") != "nativemem.canonical-entry-bank.v1"
        or bank.get("sample") != sample
    ):
        raise R301Error(f"sample {sample} canonical-bank identity differs")
    entries = bank.get("entries")
    if not isinstance(entries, list) or not entries:
        raise R301Error(f"sample {sample} canonical bank is empty")
    expected_keys = {
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
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise R301Error("canonical entry fields differ")
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id or entry_id in seen:
            raise R301Error("canonical entry IDs are invalid")
        seen.add(entry_id)
        if entry.get("sample") != sample:
            raise R301Error("canonical entry sample differs")
        source_ids = entry.get("dia_ids")
        if not isinstance(source_ids, list) or not all(
            isinstance(value, str) and re.fullmatch(r"D\d+:\d+", value)
            for value in source_ids
        ):
            raise R301Error("canonical entry source IDs are invalid")
    if (
        bank.get("entry_count") != len(entries)
        or bank.get("entries_sha256") != value_sha256(entries)
    ):
        raise R301Error("canonical bank hash/count differs")
    return dict(bank)


def validate_topic_path(value: Any) -> str:
    if not isinstance(value, str):
        raise R301Error("organizer topic path is not a string")
    path = safe_relative(value, label="topic path")
    if len(path.parts) > 4 or any(
        not part or part in {".", ".."} or len(part.encode("utf-8")) > 120
        for part in path.parts
    ):
        raise R301Error(f"organizer topic path is invalid: {value!r}")
    if value != path.as_posix() or "\x00" in value:
        raise R301Error(f"organizer topic path is not canonical: {value!r}")
    if v8_memory._sanitize_topic(value) != value:
        raise R301Error(f"organizer topic path changes during materialization: {value!r}")
    return value


def validate_placements(
    raw: Mapping[str, Any], bank: Mapping[str, Any]
) -> list[dict[str, str]]:
    if set(raw) != {"placements"} or not isinstance(raw["placements"], list):
        raise R301Error("organizer output must contain only placements")
    expected_ids = [entry["entry_id"] for entry in bank["entries"]]
    placements = []
    for item in raw["placements"]:
        if not isinstance(item, dict) or set(item) != {"entry_id", "topic_path"}:
            raise R301Error("organizer placement fields differ")
        placements.append(
            {
                "entry_id": str(item["entry_id"]),
                "topic_path": validate_topic_path(item["topic_path"]),
            }
        )
    if [item["entry_id"] for item in placements] != expected_ids:
        raise R301Error("organizer changed, omitted, duplicated, or reordered entries")
    if len({item["topic_path"] for item in placements}) > 30:
        raise R301Error("organizer exceeded the preregistered topic count")
    return placements


def materialize_organizer(
    *,
    target: Path,
    sample: int,
    organizer_id: str,
    replicate: int,
    bank: Mapping[str, Any],
    placements: Sequence[Mapping[str, str]],
    call: Mapping[str, Any],
) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        raise R301Error(f"organizer artifact already exists: {target}")
    memory = target / "memory"
    memory.mkdir(parents=True)
    entries_by_id = {entry["entry_id"]: entry for entry in bank["entries"]}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for placement in placements:
        entry = entries_by_id[placement["entry_id"]]
        grouped[placement["topic_path"]].append(entry)
        v8_memory.write_events(
            str(memory),
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
    duplicate_lines_removed = int(v8_memory.dedup_topic_files(str(memory)) or 0)
    tree = tree_descriptor(memory)
    artifact = {
        "schema": ORGANIZER_SCHEMA,
        "status": "complete",
        "sample": sample,
        "organizer_id": organizer_id,
        "replicate": replicate,
        "bank_entries_sha256": bank["entries_sha256"],
        "entry_count": len(bank["entries"]),
        "placements": list(placements),
        "placements_sha256": value_sha256(list(placements)),
        "topic_count": len(grouped),
        "maximum_depth": max(len(Path(path).parts) for path in grouped),
        "exact_duplicate_entries_removed": duplicate_lines_removed,
        "materialization": "src.v8_memory.write_events + dedup_topic_files",
        "timeline_generation": "src.v8_memory.write_events deterministic timeline",
        "model_maintenance_calls": 0,
        "organizer_call": dict(call),
        "memory_tree": tree,
    }
    atomic_json_no_clobber(target / "organizer.json", artifact)
    return artifact


def organizer_input(bank: Mapping[str, Any], organizer_id: str, replicate: int) -> str:
    return canonical_json(
        {
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
                for entry in bank["entries"]
            ],
        }
    )


def render_path_inventory(organizer: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts: defaultdict[str, int] = defaultdict(int)
    for placement in organizer["placements"]:
        counts[placement["topic_path"]] += 1
    return [
        {"topic_path": path, "entry_count": count}
        for path, count in sorted(counts.items())
    ]


def opened_entries(
    organizer: Mapping[str, Any],
    bank: Mapping[str, Any],
    selected_paths: Sequence[str],
) -> list[dict[str, Any]]:
    selected = set(selected_paths)
    by_id = {entry["entry_id"]: entry for entry in bank["entries"]}
    output = []
    for placement in organizer["placements"]:
        if placement["topic_path"] in selected:
            entry = by_id[placement["entry_id"]]
            output.append({**entry, "topic_path": placement["topic_path"]})
    return output


def render_opened_entries(entries: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        " ".join(
            (
                f"entry_id={entry['entry_id']}",
                f"topic_path={entry['topic_path']}",
                f"when={entry['when']}",
                f"summary={entry['summary_inline']}",
                f"source_ids={','.join(entry['dia_ids'])}",
            )
        )
        for entry in entries
    )


def _ordered_strings(value: Any, *, label: str, limit: int) -> list[str]:
    if not isinstance(value, list) or len(value) > limit or not all(
        isinstance(item, str) and item for item in value
    ):
        raise R301Error(f"invalid ordered {label}")
    if len(set(value)) != len(value):
        raise R301Error(f"duplicate {label}")
    return list(value)


def selected_evidence_text(entry: Mapping[str, Any]) -> str:
    return (
        f"entry_id={entry['entry_id']}\n"
        f"topic_path={entry['topic_path']}\n"
        f"when={entry['when']}\n"
        f"summary={entry['summary_inline']}\n"
        f"source_ids={','.join(entry['dia_ids'])}"
    )


def execute_question(
    *,
    target: Path,
    run_id: str,
    operation_id: str,
    cell: str,
    organizer_id: str,
    retriever_id: str,
    replicate: int,
    sample: int,
    sample_id: str,
    question: Mapping[str, Any],
    bank: Mapping[str, Any],
    organizer: Mapping[str, Any],
    memory_root: Path,
    turn_index: Mapping[str, Mapping[str, str]],
    preregistration_sha256: str,
    tokenizer: TokenCounter,
    recorder: CallRecorder,
    specs: Mapping[str, ModelSpec],
) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        raise R301Error(f"question artifact already exists: {target}")
    target.mkdir(parents=True)
    memory_before = snapshot_memory_path(memory_root)
    gate_run_id = (
        f"{run_id}:{cell}:replicate-{replicate:02d}:"
        f"{sample_id}:{question['question_id']}"
    )
    gate = VisibleTokenBudgetGate(
        trace_path=target / "visible_tokens.jsonl",
        manifest_path=target / "visible_tokens.manifest.json",
        run_id=gate_run_id,
        configured_budget_tokens=VISIBLE_BUDGET,
        tokenizer=tokenizer,
        memory_before=memory_before,
        overflow_policy="truncate",
        metadata={
            "experiment": "R301",
            "cell": cell,
            "organizer_id": organizer_id,
            "retriever_id": retriever_id,
            "replicate": replicate,
            "sample": sample,
            "sample_id": sample_id,
            "question_id": question["question_id"],
            "preregistration_sha256": preregistration_sha256,
        },
    )
    deliveries_for_answer: list[DeliveryResult] = []
    tool_trace: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    try:
        inventory = render_path_inventory(organizer)
        inventory_text = canonical_json({"paths": inventory})
        inventory_delivery = gate.deliver_tool_result(
            event_id="tool-0001-list-paths",
            raw_text=inventory_text,
            tool_name="list_paths",
            tool_call_id=f"{question['question_id']}:list-paths",
            metadata={"path_count": len(inventory)},
        )
        tool_trace.append(
            {
                "ordinal": 1,
                "tool_name": "list_paths",
                "event_id": inventory_delivery.event_id,
                "decision": inventory_delivery.decision,
                "delivered_tokens": inventory_delivery.delivered_tokens,
                "raw_sha256": hashlib.sha256(inventory_text.encode()).hexdigest(),
            }
        )
        navigation_payload = {
            "retriever_id": retriever_id,
            "question": question["question"],
            "path_inventory_text": inventory_delivery.delivered_text,
            "maximum_paths": 4,
        }
        navigation_call = recorder.call(
            operation_id=operation_id,
            question_id=question["question_id"],
            spec=specs[retriever_id],
            messages=[
                {"role": "system", "content": NAVIGATION_SYSTEM},
                {"role": "user", "content": canonical_json(navigation_payload)},
            ],
            max_tokens=400,
        )
        calls.append(navigation_call)
        navigation = parse_json_object(
            navigation_call["content"], label="navigation"
        )
        if set(navigation) != {"topic_paths"}:
            raise R301Error("navigation output fields differ")
        selected_paths = _ordered_strings(
            navigation["topic_paths"], label="topic paths", limit=4
        )
        available_paths = {item["topic_path"] for item in inventory}
        if not set(selected_paths).issubset(available_paths):
            raise R301Error("retriever selected an unavailable topic path")
        opened = opened_entries(organizer, bank, selected_paths)
        opened_text = render_opened_entries(opened)
        opened_delivery = gate.deliver_tool_result(
            event_id="tool-0002-open-paths",
            raw_text=opened_text,
            tool_name="open_paths",
            tool_call_id=f"{question['question_id']}:open-paths",
            metadata={"topic_paths": selected_paths, "entry_count": len(opened)},
        )
        tool_trace.append(
            {
                "ordinal": 2,
                "tool_name": "open_paths",
                "event_id": opened_delivery.event_id,
                "decision": opened_delivery.decision,
                "delivered_tokens": opened_delivery.delivered_tokens,
                "topic_paths": selected_paths,
                "entry_ids": [entry["entry_id"] for entry in opened],
                "raw_sha256": hashlib.sha256(opened_text.encode()).hexdigest(),
            }
        )
        selection_payload = {
            "retriever_id": retriever_id,
            "question": question["question"],
            "opened_entries_text": opened_delivery.delivered_text,
            "maximum_entries": 8,
        }
        selection_call = recorder.call(
            operation_id=operation_id,
            question_id=question["question_id"],
            spec=specs[retriever_id],
            messages=[
                {"role": "system", "content": SELECTION_SYSTEM},
                {"role": "user", "content": canonical_json(selection_payload)},
            ],
            max_tokens=500,
        )
        calls.append(selection_call)
        selection = parse_json_object(selection_call["content"], label="selection")
        if set(selection) != {"entry_ids", "source_ids"}:
            raise R301Error("selection output fields differ")
        selected_entry_ids = _ordered_strings(
            selection["entry_ids"], label="entry IDs", limit=8
        )
        opened_by_id = {entry["entry_id"]: entry for entry in opened}
        if not set(selected_entry_ids).issubset(opened_by_id):
            raise R301Error("retriever selected an unopened entry")
        selected_entries = [opened_by_id[value] for value in selected_entry_ids]
        selected_source_ids = _ordered_strings(
            selection["source_ids"], label="source IDs", limit=64
        )
        allowed_sources = {
            source_id for entry in selected_entries for source_id in entry["dia_ids"]
        }
        if (
            not set(selected_source_ids).issubset(allowed_sources)
            or not all(re.fullmatch(r"D\d+:\d+", value) for value in selected_source_ids)
        ):
            raise R301Error("retriever selected a source outside selected entries")
        selection_text = canonical_json(
            {
                "selected_topic_paths": selected_paths,
                "selected_entry_ids": selected_entry_ids,
                "selected_source_ids": selected_source_ids,
            }
        )
        selection_delivery = gate.deliver_tool_result(
            event_id="tool-0003-selection-trace",
            raw_text=selection_text,
            tool_name="retrieval_selection",
            tool_call_id=f"{question['question_id']}:selection",
            metadata={
                "entry_ids": selected_entry_ids,
                "source_ids": selected_source_ids,
            },
        )
        deliveries_for_answer.append(selection_delivery)
        tool_trace.append(
            {
                "ordinal": 3,
                "tool_name": "retrieval_selection",
                "event_id": selection_delivery.event_id,
                "decision": selection_delivery.decision,
                "delivered_tokens": selection_delivery.delivered_tokens,
                "entry_ids": selected_entry_ids,
                "source_ids": selected_source_ids,
                "raw_sha256": hashlib.sha256(selection_text.encode()).hexdigest(),
            }
        )
        for index, entry in enumerate(selected_entries, start=1):
            text = selected_evidence_text(entry)
            delivery = gate.deliver_tool_result(
                event_id=f"tool-entry-{index:04d}",
                raw_text=text,
                tool_name="selected_entry",
                tool_call_id=f"{question['question_id']}:entry:{index}",
                metadata={
                    "entry_id": entry["entry_id"],
                    "topic_path": entry["topic_path"],
                    "source_ids": entry["dia_ids"],
                },
            )
            deliveries_for_answer.append(delivery)
            tool_trace.append(
                {
                    "ordinal": len(tool_trace) + 1,
                    "tool_name": "selected_entry",
                    "event_id": delivery.event_id,
                    "decision": delivery.decision,
                    "delivered_tokens": delivery.delivered_tokens,
                    "entry_id": entry["entry_id"],
                    "source_ids": entry["dia_ids"],
                    "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        resolved_text = resolve_sources(selected_source_ids, turn_index)
        source_delivery = gate.deliver_source_resolution(
            event_id="source-0001-resolve-selected",
            raw_text=resolved_text,
            source_ids=selected_source_ids,
            metadata={"resolver": "exact_frozen_locomo_turn_resolver"},
        )
        deliveries_for_answer.append(source_delivery)
        tool_trace.append(
            {
                "ordinal": len(tool_trace) + 1,
                "tool_name": "resolve_sources",
                "event_id": source_delivery.event_id,
                "kind": "source_resolution",
                "decision": source_delivery.decision,
                "delivered_tokens": source_delivery.delivered_tokens,
                "source_ids": selected_source_ids,
                "raw_sha256": hashlib.sha256(resolved_text.encode()).hexdigest(),
            }
        )
        prompt = answer_contract.assemble_answer_prompt(
            question=question["question"], deliveries=deliveries_for_answer
        )
        answer_call = recorder.call(
            operation_id=operation_id,
            question_id=question["question_id"],
            spec=specs["A55"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        calls.append(answer_call)
        answer = answer_contract.extract_answer(answer_call["content"])
        memory_after = snapshot_memory_path(memory_root)
        provider_usage = {
            "calls": [
                {
                    "logical_call_id": call["logical_call_id"],
                    "requested_model": call["requested_model"],
                    "actual_model": call["actual_model"],
                    "usage": call["usage"],
                }
                for call in calls
            ]
        }
        gate_manifest = gate.finalize(
            memory_after=memory_after,
            actual_model_usage=provider_usage,
            metadata={"answer_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        )
    except BaseException:
        gate.close_incomplete()
        raise
    gold_sources = list(question["gold_source_ids"])
    source_eligible = bool(question["source_recall_eligible"])
    selected_source_set = set(selected_source_ids)
    all_gold_recalled = (
        set(gold_sources).issubset(selected_source_set) if source_eligible else None
    )
    source_resolution_all_gold = (
        bool(all_gold_recalled)
        and source_delivery.decision == "delivered"
        and all(
            f"[{source_id}]" in (source_delivery.delivered_text or "")
            for source_id in gold_sources
        )
        if source_eligible
        else None
    )
    first_relevant_ordinal = None
    first_relevant_path = None
    for ordinal, topic_path in enumerate(selected_paths, start=1):
        if any(
            entry["topic_path"] == topic_path
            and set(entry["dia_ids"]) & set(gold_sources)
            for entry in opened
        ):
            first_relevant_ordinal = ordinal
            first_relevant_path = topic_path
            break
    first_path_hit = first_relevant_ordinal == 1 if source_eligible else None
    score = f1_set(answer, question["gold_answer"])
    result = {
        "schema": QUESTION_SCHEMA,
        "status": "complete",
        "run_id": gate_run_id,
        "operation_id": operation_id,
        "cell": cell,
        "organizer_id": organizer_id,
        "retriever_id": retriever_id,
        "replicate": replicate,
        "sample": sample,
        "sample_id": sample_id,
        "question_id": question["question_id"],
        "question_index": question["question_index"],
        "category": question["category"],
        "question": question["question"],
        "bank_entries_sha256": bank["entries_sha256"],
        "placements_sha256": organizer["placements_sha256"],
        "selected_topic_paths": selected_paths,
        "opened_entry_ids": [entry["entry_id"] for entry in opened],
        "selected_entry_ids": selected_entry_ids,
        "selected_source_ids": selected_source_ids,
        "gold_source_ids_post_model_only": gold_sources,
        "source_recall_eligible": source_eligible,
        "mapped_source_recall_all_gold": all_gold_recalled,
        "first_relevant_file": {
            "path": first_relevant_path,
            "opened_path_ordinal": first_relevant_ordinal,
            "first_path_hit": first_path_hit,
        },
        "tool_trace": tool_trace,
        "model_calls": calls,
        "models": {
            "organizer": organizer["organizer_call"]["actual_model"],
            "retriever_requested": specs[retriever_id].requested_model,
            "retriever_actual": sorted(
                {call["actual_model"] for call in calls[:2]}
            ),
            "answerer_requested": "gpt-5.5",
            "answerer_actual": answer_call["actual_model"],
        },
        "answer": answer,
        "gold_answer_post_model_only": question["gold_answer"],
        "fixed_answerer_f1_set": score,
        "diagnostics": {
            "navigation_model_calls": 2,
            "navigation_tool_calls": 2,
            "navigation_visible_tokens": (
                inventory_delivery.delivered_tokens + opened_delivery.delivered_tokens
            ),
            "navigation_provider_prompt_tokens": sum(
                int(call["usage"].get("prompt_tokens") or 0) for call in calls[:2]
            ),
            "navigation_latency_s": round(
                sum(float(call["latency_s"]) for call in calls[:2]), 6
            ),
        },
        "answer_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "visible_budget": {
            "configured_tokens": VISIBLE_BUDGET,
            "visible_tokens": gate_manifest["summary"]["cumulative_visible_tokens"],
            "source_resolution_tokens": gate_manifest["summary"][
                "cumulative_source_resolution_tokens"
            ],
            "exhausted": gate_manifest["summary"]["exhausted"],
            "trace_sha256": gate_manifest["trace_sha256"],
            "tokenizer": tokenizer.identity,
        },
        "memory_before": memory_before.descriptor,
        "memory_after": memory_after.descriptor,
        "memory_unchanged": memory_before.sha256 == memory_after.sha256,
        "stage_evidence": {
            "schema_version": "r301-stage-evidence-v1",
            "gold_source_mapping_complete": source_eligible,
            "gold_source_in_canonical_entries": (
                all(
                    any(source in entry["dia_ids"] for entry in bank["entries"])
                    for source in gold_sources
                )
                if source_eligible
                else None
            ),
            "gold_source_survived_maintenance": (
                all(
                    any(source in entry["dia_ids"] for entry in bank["entries"])
                    for source in gold_sources
                )
                if source_eligible
                else None
            ),
            "gold_source_path_valid": (
                all(
                    any(source in entry["dia_ids"] for entry in bank["entries"])
                    for source in gold_sources
                )
                if source_eligible
                else None
            ),
            "retrieval_reached_gold_source": all_gold_recalled,
            "source_resolution_returned_gold_content": source_resolution_all_gold,
            "trace_ids": {
                "mapping": [
                    f"gold-mapping:{question['question_id']}:{question['mapping_sha256']}"
                ],
                "canonical_entries": [
                    f"canonical-bank:{bank['entries_sha256']}"
                ],
                "maintenance": [
                    f"maintenance-zero:{organizer['memory_tree']['tree_sha256']}"
                ],
                "paths": [
                    f"organizer-paths:{organizer['placements_sha256']}"
                ],
                "retrieval": [
                    f"retrieval-selection:{value_sha256(selected_entry_ids)}"
                ],
                "source_resolution": [
                    f"source-resolution:{hashlib.sha256(resolved_text.encode()).hexdigest()}"
                ],
            },
        },
    }
    atomic_json_no_clobber(target / "result.json", result)
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise R301Error("cannot compute an empty percentile")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def interaction_from_cell_means(means: Mapping[str, float]) -> float:
    return (means["O55-R55"] - means["O4o-R55"]) - (
        means["O55-R4o"] - means["O4o-R4o"]
    )


def analyze_interaction(
    records: Sequence[Mapping[str, Any]], *, formal: bool
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[
            (record["sample_id"], record["question_id"], record["cell"])
        ].append(float(record["fixed_answerer_f1_set"]))
    question_cell: dict[tuple[str, str, str], float] = {}
    for key, values in grouped.items():
        if len(values) != len(REPLICATES):
            raise R301Error(f"question/cell lacks three organizer replicates: {key}")
        question_cell[key] = sum(values) / len(values)
    samples = sorted({key[0] for key in question_cell})
    cells = {
        cell: [value for key, value in question_cell.items() if key[2] == cell]
        for cell in CELLS
    }
    if any(not values for values in cells.values()):
        raise R301Error("interaction matrix is incomplete")
    cell_means = {cell: sum(values) / len(values) for cell, values in cells.items()}
    point = interaction_from_cell_means(cell_means)
    by_sample: dict[str, dict[str, list[float]]] = {
        sample: {cell: [] for cell in CELLS} for sample in samples
    }
    for (sample, _question, cell), value in question_cell.items():
        by_sample[sample][cell].append(value)
    rng = random.Random(20_260_714)
    bootstrapped = []
    for _ in range(10_000):
        selected_samples = [rng.choice(samples) for _ in samples]
        replicate_means = {}
        for cell in CELLS:
            values = [
                value
                for sample in selected_samples
                for value in by_sample[sample][cell]
            ]
            replicate_means[cell] = sum(values) / len(values)
        bootstrapped.append(interaction_from_cell_means(replicate_means))
    interval = {
        "lower": percentile(bootstrapped, 0.025),
        "upper": percentile(bootstrapped, 0.975),
    }
    delta_r55 = cell_means["O55-R55"] - cell_means["O4o-R55"]
    delta_r4o_matched = cell_means["O4o-R4o"] - cell_means["O55-R4o"]
    matched = delta_r55 > 0 and delta_r4o_matched > 0
    interval_excludes_zero_positive = interval["lower"] > 0
    rule_passed = matched and interval_excludes_zero_positive
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": "complete",
        "outcome": "fixed_answerer_f1_set",
        "record_count": len(records),
        "question_cell_count": len(question_cell),
        "conversation_cluster_count": len(samples),
        "organizer_replicates": list(REPLICATES),
        "m4_comparison_preregistration": {
            "path": "paper/refine-logs/M4_COMPARISON_PREREGISTRATION.json",
            "sha256": EXPECTED_M4_COMPARISON_SHA256,
            "section": "m5_interaction",
        },
        "replicate_aggregation": "equal-weight mean within cell and question",
        "cell_means": cell_means,
        "matched_direction": {
            "O55_minus_O4o_under_R55": delta_r55,
            "O4o_minus_O55_under_R4o": delta_r4o_matched,
            "both_strict": matched,
        },
        "interaction": {
            "formula": "(O55-R55 - O4o-R55) - (O55-R4o - O4o-R4o)",
            "point_estimate": point,
            "confidence_interval_95": interval,
            "bootstrap_cluster": "conversation",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_714,
            "interval_method": "two-sided percentile",
        },
        "decision": {
            "matched_direction_required": True,
            "positive_interval_lower_bound_required": True,
            "rule_passed": rule_passed,
            "claim_status": (
                "model_specific_organization_supported"
                if formal and rule_passed
                else "model_specific_organization_not_supported"
                if formal
                else "synthetic_validation_only"
            ),
            "failure_action": (
                "delete matching claim and report organizer or retriever capacity effects only"
            ),
        },
    }


def _synthetic_inputs() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    conversation = {
        "speaker_a": "Ari",
        "speaker_b": "Bo",
        "session_1_date_time": "1 January 2025",
        "session_1": [
            {"speaker": "Ari", "dia_id": "D1:1", "text": "I adopted a cat named Pixel."},
            {"speaker": "Bo", "dia_id": "D1:2", "text": "I moved to Rome."},
            {"speaker": "Ari", "dia_id": "D1:3", "text": "It rained yesterday."},
        ],
    }
    entries = [
        {
            "entry_id": "synthetic-entry-pixel",
            "sample": 0,
            "session": 1,
            "chunk": 1,
            "ordinal": 1,
            "when": "2025-01-01",
            "summary": "Ari adopted Pixel",
            "summary_inline": "Ari adopted a cat named Pixel [D1:1]",
            "dia_ids": ["D1:1"],
        },
        {
            "entry_id": "synthetic-entry-rome",
            "sample": 0,
            "session": 1,
            "chunk": 1,
            "ordinal": 2,
            "when": "2025-01-01",
            "summary": "Bo moved to Rome",
            "summary_inline": "Bo moved to Rome [D1:2]",
            "dia_ids": ["D1:2"],
        },
        {
            "entry_id": "synthetic-entry-rain",
            "sample": 0,
            "session": 1,
            "chunk": 1,
            "ordinal": 3,
            "when": "2024-12-31",
            "summary": "It rained",
            "summary_inline": "It rained yesterday [D1:3]",
            "dia_ids": ["D1:3"],
        },
    ]
    bank = {
        "schema": "nativemem.canonical-entry-bank.v1",
        "sample": 0,
        "entry_count": len(entries),
        "entries_sha256": value_sha256(entries),
        "source_identity_sha256": value_sha256(
            [
                {"entry_id": entry["entry_id"], "dia_ids": entry["dia_ids"]}
                for entry in entries
            ]
        ),
        "entries": entries,
    }
    mapping_payload = {
        "question_id": "synthetic:q000",
        "normalized_source_ids": ["D1:1"],
        "source_recall_eligible": True,
    }
    question = {
        "question_id": "synthetic:q000",
        "question_index": 0,
        "category": 1,
        "question": "What is the cat's name?",
        "gold_answer": "Pixel",
        "gold_source_ids": ["D1:1"],
        "source_recall_eligible": True,
        "mapping_sha256": value_sha256(mapping_payload),
    }
    samples = [
        {
            "sample": 0,
            "sample_id": "synthetic-conversation-0",
            "conversation": conversation,
            "bank": bank,
            "bank_path": None,
            "bank_file_sha256": value_sha256(bank),
        }
    ]
    return samples, {0: bank}, {0: [question]}


def load_formal_inputs(
    r207_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    import audit_r207_path_control  # noqa: PLC0415

    r207_report = audit_r207_path_control.audit(r207_root)
    if (
        r207_report.get("status") != "pass"
        or r207_report.get("sample_count") != 10
    ):
        raise R301Error("formal R207 source audit did not pass all ten samples")
    manifest = read_json(r207_root / "run_manifest.json")
    if manifest.get("config", {}).get("data", {}).get("mode") != "locomo":
        raise R301Error("formal R301 requires a formal LoCoMo R207 artifact")
    if sha256_file(DEFAULT_DATA) != EXPECTED_DATA_SHA256:
        raise R301Error("LoCoMo dataset hash differs")
    if sha256_file(DEFAULT_MAPPING) != EXPECTED_MAPPING_SHA256:
        raise R301Error("R002 evidence mapping hash differs")
    raw_dataset = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
    if not isinstance(raw_dataset, list) or len(raw_dataset) != 10:
        raise R301Error("LoCoMo dataset inventory differs")
    mappings: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    with DEFAULT_MAPPING.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("benchmark") != "LoCoMo" or record.get("qa_scoring_eligible") is not True:
                continue
            sample = int(record["sample_index"])
            mappings[sample].append(record)
    if sum(len(values) for values in mappings.values()) != 1540 or set(mappings) != set(range(10)):
        raise R301Error("R002 primary LoCoMo inventory differs")
    samples = []
    banks = {}
    questions = {}
    for sample in range(10):
        source = raw_dataset[sample]
        conversation = source.get("conversation")
        qa = source.get("qa")
        sample_id = str(source.get("sample_id", ""))
        if not isinstance(conversation, dict) or not isinstance(qa, list) or not sample_id:
            raise R301Error(f"LoCoMo sample {sample} shape differs")
        bank_path = r207_root / f"samples/sample-{sample}/canonical_entry_bank.json"
        bank = validate_bank(read_json(bank_path), sample=sample)
        banks[sample] = bank
        sample_questions = []
        for mapping in sorted(mappings[sample], key=lambda item: int(item["question_index"])):
            index = int(mapping["question_index"])
            if index < 0 or index >= len(qa):
                raise R301Error("R002 question index is outside raw LoCoMo")
            raw_question = qa[index]
            if (
                mapping.get("sample_id") != sample_id
                or mapping.get("question") != raw_question.get("question")
                or mapping.get("category") != raw_question.get("category")
                or "answer" not in raw_question
            ):
                raise R301Error("R002 question differs from raw LoCoMo")
            sample_questions.append(
                {
                    "question_id": str(mapping["question_id"]),
                    "question_index": index,
                    "category": int(mapping["category"]),
                    "question": str(mapping["question"]),
                    "gold_answer": str(raw_question["answer"]),
                    "gold_source_ids": [
                        str(value) for value in mapping["normalized_source_ids"]
                    ],
                    "source_recall_eligible": bool(mapping["source_recall_eligible"]),
                    "mapping_sha256": value_sha256(mapping),
                }
            )
        questions[sample] = sample_questions
        samples.append(
            {
                "sample": sample,
                "sample_id": sample_id,
                "conversation": conversation,
                "bank": bank,
                "bank_path": str(bank_path.relative_to(ROOT)),
                "bank_file_sha256": sha256_file(bank_path),
            }
        )
    return samples, banks, questions, r207_report


class ManagedProxy:
    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        label: str,
        upstream: str,
        api_key_env: str | None,
        requested_model: str,
        actual_regex: str,
        flex_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.label = label
        self.upstream = upstream
        self.api_key_env = api_key_env
        self.requested_model = requested_model
        self.actual_regex = actual_regex
        self.flex_contract = dict(flex_contract) if flex_contract is not None else None
        self.process: subprocess.Popen[Any] | None = None
        self.log: Path | None = None
        self.ready: Path | None = None
        self.window_start_path: Path | None = None
        self.provider_window_start: dict[str, Any] | None = None
        self.base_url: str | None = None

        if self.flex_contract is not None:
            try:
                flex_evidence.validate_recorded_contract(self.flex_contract)
            except flex_evidence.EvidenceError as exc:
                raise R301Error(str(exc)) from exc
            if self.label != "gpt55" or self.upstream != self.flex_contract["origin"]:
                raise R301Error("GPT-5.5 proxy does not match the Flex gateway contract")
        elif self.api_key_env is None:
            raise R301Error("non-Flex proxy requires an API-key environment name")

    @staticmethod
    def stopped_path(ready: Path) -> Path:
        return ready.with_name(
            ready.name.removesuffix(".ready.json") + ".stopped.json"
        )

    @staticmethod
    def flex_start_path(ready: Path) -> Path:
        return ready.with_name(
            ready.name.removesuffix(".ready.json") + ".flex-window-start.json"
        )

    @staticmethod
    def process_command(pid: int) -> str | None:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            capture_output=True,
            check=False,
        )
        command = result.stdout.strip()
        return command if result.returncode == 0 and command else None

    @classmethod
    def recover_stale(cls, ready_path: Path) -> None:
        stopped_path = cls.stopped_path(ready_path)
        if stopped_path.exists():
            stopped = read_json(stopped_path)
            if (
                stopped.get("ready_sha256") != sha256_file(ready_path)
                or not Path(str(stopped.get("log", ""))).is_file()
                or stopped.get("log_sha256")
                != sha256_file(Path(str(stopped["log"])))
            ):
                raise R301Error("existing proxy stop manifest differs")
            return
        ready = read_json(ready_path)
        is_flex = ready_path.parent.name == "gpt55"
        pid = ready.get("pid")
        log = Path(str(ready.get("log", "")))
        if not isinstance(pid, int) or pid <= 0 or not log.is_file():
            raise R301Error("stale proxy ready artifact is incomplete")
        command = cls.process_command(pid)
        status = "already_exited"
        if command is not None:
            proxy_script = GPT55_PROXY_SCRIPT if is_flex else PROXY_SCRIPT
            expected_fragments = (
                str(proxy_script),
                str(ready_path),
                str(log),
                str(ready.get("run_id")),
            )
            if not all(fragment in command for fragment in expected_fragments):
                raise R301Error("stale proxy PID now belongs to another command")
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and cls.process_command(pid) is not None:
                time.sleep(0.05)
            if cls.process_command(pid) is not None:
                os.killpg(pid, signal.SIGKILL)
                time.sleep(0.05)
            if cls.process_command(pid) is not None:
                raise R301Error("stale exclusive proxy did not terminate")
            status = "interrupted_recovered"
        provider_window = None
        gateway_contract = None
        window_start_path = None
        if is_flex:
            window_start_path = cls.flex_start_path(ready_path)
            provider_window_start = read_json(window_start_path)
            try:
                provider_window = flex_evidence.capture_end(provider_window_start)
                flex_evidence.audit_window(
                    provider_window,
                    consumer_records=load_jsonl(log),
                )
            except flex_evidence.EvidenceError as exc:
                raise R301Error(str(exc)) from exc
            gateway_contract = provider_window["contract"]
        atomic_json_no_clobber(
            stopped_path,
            {
                "schema": "nativemem.r301-exclusive-proxy-stop.v1",
                "status": status,
                "run_id": ready["run_id"],
                "pid": pid,
                "ready": str(ready_path),
                "ready_sha256": sha256_file(ready_path),
                "log": str(log),
                "log_sha256": sha256_file(log),
                "gateway_contract": gateway_contract,
                "provider_window_start": (
                    str(window_start_path) if window_start_path is not None else None
                ),
                "provider_window_start_sha256": (
                    sha256_file(window_start_path)
                    if window_start_path is not None
                    else None
                ),
                "provider_window": provider_window,
                "stopped_at": utc_now(),
            },
        )

    def start(self) -> None:
        directory = self.root / "proxy" / self.label
        directory.mkdir(parents=True, exist_ok=True)
        launch = len(list(directory.glob("launch-*.ready.json"))) + 1
        self.log = directory / f"launch-{launch:03d}.jsonl"
        self.ready = directory / f"launch-{launch:03d}.ready.json"
        script = PROXY_SCRIPT
        command = [
            sys.executable,
            str(script),
            "--port",
            "0",
            "--upstream",
            self.upstream,
        ]
        if self.flex_contract is not None:
            script = GPT55_PROXY_SCRIPT
            command[1] = str(script)
            try:
                self.provider_window_start = flex_evidence.capture_start(
                    Path(self.flex_contract["result_root"])
                )
            except flex_evidence.EvidenceError as exc:
                raise R301Error(str(exc)) from exc
            if self.provider_window_start["contract"] != self.flex_contract:
                raise R301Error("live Flex gateway differs from the frozen R301 contract")
            self.window_start_path = self.flex_start_path(self.ready)
            atomic_json_no_clobber(
                self.window_start_path,
                self.provider_window_start,
            )
        else:
            assert self.api_key_env is not None
            command.extend(
                [
                    "--api-key-env",
                    self.api_key_env,
                    "--expected-requested-model",
                    self.requested_model,
                    "--accepted-actual-model-regex",
                    self.actual_regex,
                ]
            )
        command.extend(
            [
                "--log",
                str(self.log),
                "--ready",
                str(self.ready),
                "--run-id",
                f"{self.run_id}:{self.label}:launch-{launch:03d}",
            ]
        )
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise R301Error(f"{self.label} proxy exited before readiness")
                if self.ready.is_file():
                    ready = read_json(self.ready)
                    self.base_url = str(ready["base_url"])
                    return
                time.sleep(0.05)
            raise R301Error(f"{self.label} proxy readiness timed out")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is None or self.ready is None or self.log is None:
            return
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        provider_window = None
        if self.flex_contract is not None:
            if self.provider_window_start is None or self.window_start_path is None:
                raise R301Error("GPT-5.5 proxy has no Flex provider-window start")
            try:
                provider_window = flex_evidence.capture_end(
                    self.provider_window_start
                )
                flex_evidence.audit_window(
                    provider_window,
                    consumer_records=load_jsonl(self.log),
                )
            except flex_evidence.EvidenceError as exc:
                raise R301Error(str(exc)) from exc
        stopped = self.stopped_path(self.ready)
        if not stopped.exists():
            ready = read_json(self.ready)
            atomic_json_no_clobber(
                stopped,
                {
                    "schema": "nativemem.r301-exclusive-proxy-stop.v1",
                    "status": "normal_stop",
                    "run_id": ready["run_id"],
                    "pid": self.process.pid,
                    "ready": str(self.ready),
                    "ready_sha256": sha256_file(self.ready),
                    "log": str(self.log),
                    "log_sha256": sha256_file(self.log),
                    "gateway_contract": (
                        dict(self.flex_contract)
                        if self.flex_contract is not None
                        else None
                    ),
                    "provider_window_start": (
                        str(self.window_start_path)
                        if self.window_start_path is not None
                        else None
                    ),
                    "provider_window_start_sha256": (
                        sha256_file(self.window_start_path)
                        if self.window_start_path is not None
                        else None
                    ),
                    "provider_window": provider_window,
                    "stopped_at": utc_now(),
                },
            )


def recover_stale_proxies(output: Path) -> None:
    proxy_root = output / "proxy"
    if not proxy_root.exists():
        return
    if proxy_root.is_symlink() or not proxy_root.is_dir():
        raise R301Error("proxy root is not a regular directory")
    for ready in sorted(proxy_root.rglob("launch-*.ready.json")):
        ManagedProxy.recover_stale(ready)


def operation_artifact_ok(
    root: Path, committed: Mapping[str, Any], expected_relative: str
) -> bool:
    path = root / expected_relative
    return (
        committed.get("artifact_path") == expected_relative
        and path.is_file()
        and not path.is_symlink()
        and committed.get("artifact_sha256") == sha256_file(path)
    )


def begin_operation(
    ledger: HashChainLedger,
    *,
    operation_id: str,
    kind: str,
    input_sha256: str,
    metadata: Mapping[str, Any],
) -> None:
    ledger.append(
        "operation_started",
        operation_id=operation_id,
        kind=kind,
        operation_input_sha256=input_sha256,
        metadata=dict(metadata),
    )


def commit_operation(
    ledger: HashChainLedger,
    *,
    operation_id: str,
    kind: str,
    input_sha256: str,
    root: Path,
    artifact_path: Path,
) -> None:
    ledger.append(
        "operation_committed",
        operation_id=operation_id,
        kind=kind,
        operation_input_sha256=input_sha256,
        artifact_path=str(artifact_path.relative_to(root)),
        artifact_sha256=sha256_file(artifact_path),
    )


def fail_operation(
    ledger: HashChainLedger,
    *,
    operation_id: str,
    kind: str,
    input_sha256: str,
    exc: BaseException,
) -> None:
    ledger.append(
        "operation_failed",
        operation_id=operation_id,
        kind=kind,
        operation_input_sha256=input_sha256,
        error=f"{type(exc).__name__}: {exc}",
    )


def create_or_resume_manifest(
    *,
    output: Path,
    formal: bool,
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    sources: Mapping[str, str],
    samples: Sequence[Mapping[str, Any]],
    r207_root: Path | None,
    r207_report: Mapping[str, Any] | None,
    gpt55_gateway: Mapping[str, Any] | None,
    openrouter_gateway: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = {
        "mode": "formal" if formal else "synthetic_no_network",
        "network_requests_authorized": formal,
        "samples": [sample["sample"] for sample in samples],
        "cells": list(CELLS),
        "organizer_replicates": list(REPLICATES),
        "visible_budget_tokens": VISIBLE_BUDGET,
        "preregistration_path": str(preregistration_path.relative_to(ROOT)),
        "preregistration_sha256": sha256_file(preregistration_path),
        "preregistration_content_sha256": preregistration["protocol_content_sha256"],
        "m4_comparison_preregistration_sha256": EXPECTED_M4_COMPARISON_SHA256,
        "r207_root": str(r207_root) if r207_root is not None else None,
        "r207_run_fingerprint": (
            r207_report.get("run_fingerprint") if r207_report is not None else None
        ),
        "gpt55_gateway": dict(gpt55_gateway) if formal else None,
        "gpt55_upstream": (
            str(gpt55_gateway["origin"])
            if formal and gpt55_gateway is not None
            else None
        ),
        "gpt4o_upstream": args.gpt4o_upstream if formal else None,
        "gpt4o_api_key_env": args.gpt4o_api_key_env if formal else None,
        "openrouter_gateway": dict(openrouter_gateway) if formal else None,
        "api_keys_recorded": False,
        "bank_inputs": [
            {
                "sample": sample["sample"],
                "sample_id": sample["sample_id"],
                "bank_path": sample["bank_path"],
                "bank_file_sha256": sample["bank_file_sha256"],
                "entries_sha256": sample["bank"]["entries_sha256"],
            }
            for sample in samples
        ],
        "input_bundle_sha256": value_sha256(
            [
                {
                    "sample": sample["sample"],
                    "sample_id": sample["sample_id"],
                    "conversation": sample["conversation"],
                    "bank": sample["bank"],
                }
                for sample in samples
            ]
        ),
    }
    fingerprint = value_sha256(
        {
            "schema": RUN_SCHEMA,
            "config": config,
            "source_hashes": sources,
        }
    )
    path = output / "run_manifest.json"
    expected = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "created_at": None,
        "run_fingerprint": fingerprint,
        "config": config,
        "source_hashes": dict(sources),
    }
    if path.exists():
        manifest = read_json(path)
        comparable = dict(manifest)
        comparable["created_at"] = None
        if comparable != expected:
            raise R301Error("resume manifest differs; use a new output root")
        return manifest
    manifest = dict(expected)
    manifest["created_at"] = utc_now()
    atomic_json_no_clobber(path, manifest)
    return manifest


def capture_formal_openrouter_gateway(
    result_root: Path, *, upstream: str
) -> dict[str, Any]:
    """Validate the marked loopback gateway before any formal R301 call."""
    try:
        return openrouter_gateway_evidence.capture_binding(
            result_root, base_url=upstream
        )
    except openrouter_gateway_evidence.GatewayEvidenceError as exc:
        raise R301Error(str(exc)) from exc


def freeze_inputs(
    *,
    output: Path,
    samples: Sequence[Mapping[str, Any]],
    questions: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    records = []
    for sample in samples:
        sample_index = int(sample["sample"])
        relative = f"inputs/sample-{sample_index:02d}.json"
        path = output / relative
        payload = {
            "schema": "nativemem.r301-frozen-input.v1",
            "sample": sample_index,
            "sample_id": sample["sample_id"],
            "conversation": sample["conversation"],
            "bank": sample["bank"],
            "questions": list(questions[sample_index]),
        }
        if path.exists():
            if read_json(path) != payload:
                raise R301Error("frozen input bundle differs on resume")
        else:
            atomic_json_no_clobber(path, payload)
        records.append(
            {
                "sample": sample_index,
                "sample_id": sample["sample_id"],
                "path": relative,
                "sha256": sha256_file(path),
                "bank_entries_sha256": sample["bank"]["entries_sha256"],
                "question_count": len(questions[sample_index]),
            }
        )
    manifest = {
        "schema": "nativemem.r301-frozen-input-manifest.v1",
        "samples": records,
        "sample_count": len(records),
        "question_count": sum(record["question_count"] for record in records),
        "records_sha256": value_sha256(records),
    }
    path = output / "inputs/manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise R301Error("frozen input manifest differs on resume")
    else:
        atomic_json_no_clobber(path, manifest)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().absolute()
    answer_contract.reject_symlink_components(output)
    if output.is_symlink():
        raise R301Error("output root must not be a symlink")
    formal = not args.synthetic_sanity
    prereg_path = args.preregistration.expanduser().resolve()
    preregistration = validate_preregistration(prereg_path)
    tokenizer = answer_contract.formal_token_counter()
    specs = model_specs(preregistration)
    if formal:
        if not args.allow_model_requests:
            raise R301Error("formal R301 requires --allow-model-requests")
        if args.gateway_root is None:
            raise R301Error("formal R301 requires --gateway-root")
        if args.samples != list(range(10)):
            raise R301Error("formal R301 requires --samples 0-9")
        if args.r207_root is None:
            raise R301Error("formal R301 requires --r207-root")
        r207_root = args.r207_root.expanduser().resolve()
        samples, banks, questions, r207_report = load_formal_inputs(r207_root)
    else:
        if args.allow_model_requests:
            raise R301Error("synthetic R301 forbids --allow-model-requests")
        if args.samples != [0]:
            raise R301Error("synthetic R301 requires sample zero only")
        samples, banks, questions = _synthetic_inputs()
        r207_root = None
        r207_report = None
    sources = _source_hashes()
    if output.exists() and (output / "complete.json").is_file():
        with answer_contract.FileLock(output / ".r301.lock"):
            if formal:
                ready_paths = sorted(
                    (output / "proxy/gpt55").glob("launch-*.ready.json")
                )
                needs_flex_recovery = any(
                    not ManagedProxy.stopped_path(path).exists()
                    for path in ready_paths
                )
                flex_lock = None
                try:
                    if needs_flex_recovery:
                        flex_lock = flex_evidence.acquire_consumer_lock(
                            args.gateway_root
                        )
                    recover_stale_proxies(output)
                except flex_evidence.EvidenceError as exc:
                    raise R301Error(str(exc)) from exc
                finally:
                    if flex_lock is not None:
                        flex_lock.close()
            from audit_r301_organizer_retriever import audit_run  # noqa: PLC0415

            return audit_run(output)
    output.mkdir(parents=True, exist_ok=True)
    with answer_contract.FileLock(output / ".r301.lock"), contextlib.ExitStack() as locks:
        gpt55_contract: dict[str, Any] | None = None
        openrouter_binding: dict[str, Any] | None = None
        if formal:
            try:
                flex_lock = flex_evidence.acquire_consumer_lock(args.gateway_root)
                locks.callback(flex_lock.close)
                live_gpt55_contract = flex_evidence.active_contract(args.gateway_root)
                existing_manifest_path = output / "run_manifest.json"
                if existing_manifest_path.exists():
                    existing_manifest = read_json(existing_manifest_path)
                    stored_contract = existing_manifest.get("config", {}).get(
                        "gpt55_gateway"
                    )
                    if not isinstance(stored_contract, dict):
                        raise R301Error(
                            "resume manifest lacks GPT-5.5 Flex gateway binding"
                        )
                    flex_evidence.validate_recorded_contract(stored_contract)
                    if live_gpt55_contract != stored_contract:
                        raise R301Error(
                            "live GPT-5.5 Flex gateway differs from the resume binding"
                        )
                    gpt55_contract = dict(stored_contract)
                else:
                    gpt55_contract = live_gpt55_contract
            except flex_evidence.EvidenceError as exc:
                raise R301Error(str(exc)) from exc
            if args.openrouter_gateway_root is None:
                raise R301Error(
                    "formal R301 requires --openrouter-gateway-root"
                )
            if not args.gpt4o_upstream:
                raise R301Error(
                    "formal R301 requires --gpt4o-upstream pointing to the "
                    "marked loopback OpenRouter gateway"
                )
            try:
                live_binding = capture_formal_openrouter_gateway(
                    args.openrouter_gateway_root,
                    upstream=args.gpt4o_upstream,
                )
                if existing_manifest_path.exists():
                    existing_manifest = read_json(existing_manifest_path)
                    stored_binding = existing_manifest.get("config", {}).get(
                        "openrouter_gateway"
                    )
                    if not isinstance(stored_binding, dict):
                        raise R301Error(
                            "resume manifest lacks OpenRouter gateway binding"
                        )
                    openrouter_gateway_evidence.validate_binding(
                        stored_binding,
                        result_root=args.openrouter_gateway_root,
                        base_url=args.gpt4o_upstream,
                    )
                    if any(
                        live_binding.get(name) != stored_binding.get(name)
                        for name in (
                            "result_root",
                            "base_url",
                            "root_marker",
                            "state_path",
                            "request_log_path",
                            "max_cost_usd",
                        )
                    ):
                        raise R301Error(
                            "live OpenRouter gateway differs from the resume binding"
                        )
                    openrouter_binding = dict(stored_binding)
                else:
                    openrouter_binding = live_binding
            except openrouter_gateway_evidence.GatewayEvidenceError as exc:
                raise R301Error(str(exc)) from exc
        manifest = create_or_resume_manifest(
            output=output,
            formal=formal,
            preregistration=preregistration,
            preregistration_path=prereg_path,
            sources=sources,
            samples=samples,
            r207_root=r207_root,
            r207_report=r207_report,
            gpt55_gateway=gpt55_contract,
            openrouter_gateway=openrouter_binding,
            args=args,
        )
        input_manifest = freeze_inputs(
            output=output, samples=samples, questions=questions
        )
        if input_manifest["question_count"] != sum(
            len(values) for values in questions.values()
        ):
            raise R301Error("frozen input question count differs")
        if formal:
            recover_stale_proxies(output)
        ledger_path = output / "operations.jsonl"
        records = read_ledger(ledger_path)
        state = ledger_state(records)
        ledger = HashChainLedger(ledger_path, run_id=manifest["run_fingerprint"])
        committed = state["committed_operations"]
        proxies: list[ManagedProxy] = []
        if formal:
            gpt55_proxy = ManagedProxy(
                root=output,
                run_id=manifest["run_fingerprint"],
                label="gpt55",
                upstream=str(gpt55_contract["origin"]),
                api_key_env=None,
                requested_model=specs["A55"].requested_model,
                actual_regex=specs["A55"].accepted_actual_model_regex,
                flex_contract=gpt55_contract,
            )
            gpt4o_proxy = ManagedProxy(
                root=output,
                run_id=manifest["run_fingerprint"],
                label="gpt4o",
                upstream=args.gpt4o_upstream,
                api_key_env=args.gpt4o_api_key_env,
                requested_model=specs["O4o"].requested_model,
                actual_regex=specs["O4o"].accepted_actual_model_regex,
            )
            for proxy in (gpt55_proxy, gpt4o_proxy):
                proxy.start()
                proxies.append(proxy)
            assert gpt55_proxy.base_url and gpt4o_proxy.base_url
            client: ModelClient = HttpModelClient(
                {
                    specs["A55"].requested_model: gpt55_proxy.base_url,
                    specs["O4o"].requested_model: gpt4o_proxy.base_url,
                }
            )
            proxy_logs = {
                specs["A55"].requested_model: gpt55_proxy.log,
                specs["O4o"].requested_model: gpt4o_proxy.log,
            }
        else:
            client = FakeModelClient()
            proxy_logs = {
                specs["A55"].requested_model: None,
                specs["O4o"].requested_model: None,
            }
        recorder = CallRecorder(
            root=output,
            ledger=ledger,
            tokenizer=tokenizer,
            client=client,
            proxy_logs=proxy_logs,
            formal=formal,
        )
        organizers: dict[tuple[int, str, int], dict[str, Any]] = {}
        try:
            for sample_record in samples:
                sample = int(sample_record["sample"])
                bank = banks[sample]
                for organizer_id in ORGANIZER_IDS:
                    for replicate in REPLICATES:
                        operation_id = (
                            f"organizer/sample-{sample:02d}/{organizer_id}/"
                            f"replicate-{replicate:02d}"
                        )
                        relative = (
                            f"organizers/sample-{sample:02d}/{organizer_id}/"
                            f"replicate-{replicate:02d}/organizer.json"
                        )
                        committed_record = committed.get(operation_id)
                        if committed_record is not None:
                            if not operation_artifact_ok(output, committed_record, relative):
                                raise R301Error("committed organizer artifact changed")
                            organizer = read_json(output / relative)
                            organizers[(sample, organizer_id, replicate)] = organizer
                            continue
                        input_sha = value_sha256(
                            {
                                "bank_entries_sha256": bank["entries_sha256"],
                                "organizer_id": organizer_id,
                                "replicate": replicate,
                                "preregistration": preregistration[
                                    "protocol_content_sha256"
                                ],
                            }
                        )
                        begin_operation(
                            ledger,
                            operation_id=operation_id,
                            kind="organize_fixed_entries",
                            input_sha256=input_sha,
                            metadata={
                                "sample": sample,
                                "organizer_id": organizer_id,
                                "replicate": replicate,
                            },
                        )
                        try:
                            call = recorder.call(
                                operation_id=operation_id,
                                question_id=None,
                                spec=specs[organizer_id],
                                messages=[
                                    {"role": "system", "content": ORGANIZER_SYSTEM},
                                    {
                                        "role": "user",
                                        "content": organizer_input(
                                            bank, organizer_id, replicate
                                        ),
                                    },
                                ],
                                max_tokens=max(1000, len(bank["entries"]) * 80),
                            )
                            placements = validate_placements(
                                parse_json_object(call["content"], label="organizer"),
                                bank,
                            )
                            target = (output / relative).parent
                            organizer = materialize_organizer(
                                target=target,
                                sample=sample,
                                organizer_id=organizer_id,
                                replicate=replicate,
                                bank=bank,
                                placements=placements,
                                call=call,
                            )
                            commit_operation(
                                ledger,
                                operation_id=operation_id,
                                kind="organize_fixed_entries",
                                input_sha256=input_sha,
                                root=output,
                                artifact_path=output / relative,
                            )
                        except BaseException as exc:
                            fail_operation(
                                ledger,
                                operation_id=operation_id,
                                kind="organize_fixed_entries",
                                input_sha256=input_sha,
                                exc=exc,
                            )
                            raise
                        organizers[(sample, organizer_id, replicate)] = organizer

            result_records: list[dict[str, Any]] = []
            result_locations: dict[str, Path] = {}
            for sample_record in samples:
                sample = int(sample_record["sample"])
                sample_id = str(sample_record["sample_id"])
                bank = banks[sample]
                turn_index = build_turn_index(sample_record["conversation"])
                for cell in CELLS:
                    organizer_id, retriever_id = cell.split("-", 1)
                    for replicate in REPLICATES:
                        organizer = organizers[(sample, organizer_id, replicate)]
                        memory_root = (
                            output
                            / f"organizers/sample-{sample:02d}/{organizer_id}/"
                            f"replicate-{replicate:02d}/memory"
                        )
                        for question in questions[sample]:
                            question_slug = hashlib.sha256(
                                question["question_id"].encode()
                            ).hexdigest()[:20]
                            operation_id = (
                                f"question/{cell}/replicate-{replicate:02d}/"
                                f"sample-{sample:02d}/{question['question_id']}"
                            )
                            relative = (
                                f"questions/{cell}/replicate-{replicate:02d}/"
                                f"sample-{sample:02d}/{question_slug}/result.json"
                            )
                            committed_record = committed.get(operation_id)
                            if committed_record is not None:
                                if not operation_artifact_ok(
                                    output, committed_record, relative
                                ):
                                    raise R301Error("committed question artifact changed")
                                result_records.append(read_json(output / relative))
                                result_locations[operation_id] = output / relative
                                continue
                            input_sha = value_sha256(
                                {
                                    "cell": cell,
                                    "replicate": replicate,
                                    "question_id": question["question_id"],
                                    "question": question["question"],
                                    "bank_entries_sha256": bank["entries_sha256"],
                                    "placements_sha256": organizer["placements_sha256"],
                                    "mapping_sha256": question["mapping_sha256"],
                                }
                            )
                            begin_operation(
                                ledger,
                                operation_id=operation_id,
                                kind="retrieve_and_fixed_answer",
                                input_sha256=input_sha,
                                metadata={
                                    "cell": cell,
                                    "replicate": replicate,
                                    "sample": sample,
                                    "question_id": question["question_id"],
                                },
                            )
                            try:
                                result = execute_question(
                                    target=(output / relative).parent,
                                    run_id=manifest["run_fingerprint"],
                                    operation_id=operation_id,
                                    cell=cell,
                                    organizer_id=organizer_id,
                                    retriever_id=retriever_id,
                                    replicate=replicate,
                                    sample=sample,
                                    sample_id=sample_id,
                                    question=question,
                                    bank=bank,
                                    organizer=organizer,
                                    memory_root=memory_root,
                                    turn_index=turn_index,
                                    preregistration_sha256=sha256_file(prereg_path),
                                    tokenizer=tokenizer,
                                    recorder=recorder,
                                    specs=specs,
                                )
                                commit_operation(
                                    ledger,
                                    operation_id=operation_id,
                                    kind="retrieve_and_fixed_answer",
                                    input_sha256=input_sha,
                                    root=output,
                                    artifact_path=output / relative,
                                )
                            except BaseException as exc:
                                fail_operation(
                                    ledger,
                                    operation_id=operation_id,
                                    kind="retrieve_and_fixed_answer",
                                    input_sha256=input_sha,
                                    exc=exc,
                                )
                                raise
                            result_records.append(result)
                            result_locations[operation_id] = output / relative
            expected_results = sum(len(values) for values in questions.values()) * 12
            if len(result_records) != expected_results:
                raise R301Error("question-result inventory differs")
            analysis = analyze_interaction(result_records, formal=formal)
            analysis_path = output / "analysis/interaction.json"
            if analysis_path.exists():
                if read_json(analysis_path) != analysis:
                    raise R301Error("existing interaction analysis differs")
            else:
                atomic_json_no_clobber(analysis_path, analysis)
            inventory_path = output / "analysis/question_results.jsonl"
            inventory_records = [
                {
                    "cell": record["cell"],
                    "replicate": record["replicate"],
                    "sample": record["sample"],
                    "sample_id": record["sample_id"],
                    "question_id": record["question_id"],
                    "result_path": str(
                        result_locations[record["operation_id"]].relative_to(output)
                    ),
                    "result_sha256": sha256_file(
                        result_locations[record["operation_id"]]
                    ),
                }
                for record in result_records
            ]
            inventory_records.sort(
                key=lambda item: (
                    item["cell"],
                    item["replicate"],
                    item["sample"],
                    item["question_id"],
                )
            )
            if inventory_path.exists():
                existing = [
                    json.loads(line)
                    for line in inventory_path.read_text(encoding="utf-8").splitlines()
                ]
                if existing != inventory_records:
                    raise R301Error("existing result inventory differs")
            else:
                atomic_jsonl_no_clobber(inventory_path, inventory_records)
            final_state = ledger_state(ledger.records)
            complete = {
                "schema": RUN_SCHEMA,
                "status": "complete",
                "run_fingerprint": manifest["run_fingerprint"],
                "completed_at": utc_now(),
                "mode": "formal" if formal else "synthetic_no_network",
                "network_requests": (
                    final_state["successful_model_calls"] if formal else 0
                ),
                "sample_count": len(samples),
                "question_count": sum(len(values) for values in questions.values()),
                "cell_count": len(CELLS),
                "organizer_replicates": len(REPLICATES),
                "organizer_artifact_count": len(organizers),
                "question_result_count": len(result_records),
                "model_call_count": final_state["successful_model_calls"],
                "failed_model_calls": final_state["failed_model_calls"],
                "operations_sha256": sha256_file(ledger_path),
                "input_manifest_sha256": sha256_file(
                    output / "inputs/manifest.json"
                ),
                "question_inventory_sha256": sha256_file(inventory_path),
                "interaction_sha256": sha256_file(analysis_path),
                "interaction_decision": analysis["decision"],
                "openrouter_gateway": (
                    openrouter_gateway_evidence.finalize_binding(
                        openrouter_binding
                    )
                    if formal and openrouter_binding is not None
                    else None
                ),
            }
            atomic_json_no_clobber(output / "complete.json", complete)
        finally:
            stop_error: BaseException | None = None
            for proxy in reversed(proxies):
                try:
                    proxy.stop()
                except BaseException as exc:
                    if stop_error is None:
                        stop_error = exc
            if stop_error is not None:
                raise stop_error
        from audit_r301_organizer_retriever import audit_run  # noqa: PLC0415

        report = audit_run(output)
        answer_contract.atomic_json_replace(output / "audit.json", report)
        return report


def parse_samples(value: str) -> list[int]:
    values: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(item) for item in part.split("-", 1))
            values.update(range(min(left, right), max(left, right) + 1))
        else:
            values.add(int(part))
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("samples must be non-negative")
    return sorted(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--r207-root", type=Path)
    parser.add_argument("--samples", type=parse_samples, default=list(range(10)))
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--gpt4o-upstream")
    parser.add_argument("--openrouter-gateway-root", type=Path)
    parser.add_argument("--gpt4o-api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args(argv)
    if args.synthetic_sanity and args.samples == list(range(10)):
        args.samples = [0]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        R301Error,
        answer_contract.ControlledAnswerError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
