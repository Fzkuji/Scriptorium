#!/usr/bin/env python3
"""Resume-safe BEAM rubric-nugget evaluation.

The evaluator consumes ``evaluation_input.json`` produced by
``audit_v88_gpt55_beam.py``.  Each rubric nugget is judged independently with
the vendored BEAM prompt and committed to its own atomic JSON file.  Invalid or
failed judge responses never receive a numeric score.

Formal evaluation requires an explicit profile.  ``primary`` is OpenRouter's
``openai/gpt-4o-mini`` and is comparable only inside this project's unified
protocol.  It is not directly comparable to the published BEAM metric or the
Mem0 reproduction.  ``secondary`` is GPT-5.5 through a validated Flex gateway
root and is labeled
non-comparable, for example:

    python3 scripts/evaluate_v88_gpt55_beam.py \
      results/.../evaluation_input.json --profile secondary \
      --gateway-root /absolute/path/to/flex-gateway-result \
      --allow-model-requests --resume

This script intentionally reports nugget-only results.  It does not label the
vendored LLM event-alignment heuristic as the official BEAM tau-b times F1
metric.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from src import openai_gpt55_flex_gateway_evidence as flex_evidence
from src import openrouter_gateway_evidence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PROMPTS_PATH = (
    ROOT / "third_party" / "mem0-benchmarks" / "benchmarks" / "beam" / "prompts.py"
)
EXPECTED_QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
VALID_SCORES = {0.0, 0.5, 1.0}
LOCK_FILENAME = ".evaluation.lock"
ATTEMPT_LEDGER_SCHEMA = 1
FORMAL_SELECTION = {"100K": list(range(10)), "1M": list(range(35))}
FORMAL_QUESTION_COUNT = 900
FORMAL_QUESTION_TYPE_COUNT = 90
FORMAL_QUESTIONS_PER_CONVERSATION = 20
FORMAL_QUESTIONS_PER_TYPE_PER_CONVERSATION = 2
TEST_PRIMARY_BASE_URL = "http://127.0.0.1:1/v1"
TEST_SECONDARY_BASE_URL = "http://127.0.0.1:1/v1"
PRIMARY_RESPONSE_MODEL_RE = re.compile(
    r"^(?:openai/)?gpt-4o-mini(?:-(?P<date>\d{4}-\d{2}-\d{2}))?$"
)
JUDGE_PROFILES: dict[str, dict[str, Any]] = {
    "primary": {
        "provider": "OpenRouter",
        "model": "openai/gpt-4o-mini",
        "expected_response_model": "openai/gpt-4o-mini",
        # Used only by injected no-network unit fixtures. Formal CLI execution
        # replaces this with the marked OpenRouter gateway binding.
        "base_url": TEST_PRIMARY_BASE_URL,
        "comparison_label": "primary_project_unified_protocol_openrouter_gpt4o_mini",
        "comparison_scope": "project_unified_protocol_only",
        "comparable_to_project_unified_protocol": True,
        "comparable_to_published_beam_official": False,
        "comparable_to_mem0": False,
    },
    "secondary": {
        "provider": "OpenAI API Flex",
        "model": "gpt-5.5",
        "expected_response_model": "gpt-5.5",
        # Used only by injected no-network unit fixtures. Formal CLI execution
        # replaces this with the origin bound to --gateway-root.
        "base_url": TEST_SECONDARY_BASE_URL,
        "comparison_label": "secondary_non_comparable_gpt55",
        "comparison_scope": "sensitivity_analysis_only",
        "comparable_to_project_unified_protocol": False,
        "comparable_to_published_beam_official": False,
        "comparable_to_mem0": False,
    },
}


def _load_prompt_module() -> Any:
    spec = importlib.util.spec_from_file_location("vendored_beam_prompts", PROMPTS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import BEAM prompts from {PROMPTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BEAM_PROMPTS = _load_prompt_module()


class EvaluationError(RuntimeError):
    """Raised for invalid input, state, or judge output."""


class JudgeResponseError(EvaluationError):
    """A judge response was received but was unusable."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def utc_now() -> str:
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


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def should_trust_environment_proxy(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host not in {"localhost", "127.0.0.1", "::1"}


def acquire_output_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / LOCK_FILENAME).open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise EvaluationError(
            f"another evaluator is already using {output_dir}"
        ) from exc
    return handle


def release_output_lock(handle: Any | None) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def parse_strict_judgment(content: str) -> dict[str, Any]:
    """Parse exactly ``score`` and ``reason`` without score coercion."""
    if not isinstance(content, str) or not content.strip():
        raise EvaluationError("judge returned empty content")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EvaluationError("judge content is not a JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"score", "reason"}:
        raise EvaluationError("judge JSON must contain exactly score and reason")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise EvaluationError("judge score is not numeric")
    score = float(score)
    if score not in VALID_SCORES:
        raise EvaluationError("judge score must be exactly 0, 0.5, or 1")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise EvaluationError("judge reason is empty or not a string")
    return {"score": score, "reason": reason.strip()}


def _validate_input_provenance(path: Path, value: dict[str, Any]) -> None:
    """Bind a formal input to the auditor report and completed runner manifest."""
    input_path = path.expanduser().resolve()
    run_dir_value = value.get("run_dir")
    if not isinstance(run_dir_value, str) or not run_dir_value.strip():
        raise EvaluationError("evaluation input has no audited run directory")
    run_dir = Path(run_dir_value).expanduser().resolve()
    expected_input = run_dir / "evaluation_input.json"
    if input_path != expected_input:
        raise EvaluationError(
            "formal evaluation input is not the audited run's evaluation_input.json"
        )

    audit_path = run_dir / "audit.json"
    manifest_path = run_dir / "run_manifest.json"
    if not audit_path.is_file():
        raise EvaluationError("formal evaluation input has no sibling audit.json")
    if not manifest_path.is_file():
        raise EvaluationError("formal evaluation input has no run_manifest.json")
    audit = read_json(audit_path)
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise EvaluationError("formal audit report has invalid schema")
    exact_audit = {
        "status": "passed",
        "benchmark": "BEAM",
        "formal_scope_verified": True,
        "selected_conversations": FORMAL_SELECTION,
        "conversation_count": len(
            [index for indices in FORMAL_SELECTION.values() for index in indices]
        ),
        "questions": FORMAL_QUESTION_COUNT,
        "run_dir": str(run_dir),
        "evaluation_input_path": str(expected_input),
        "run_manifest_path": str(manifest_path),
    }
    for key, expected in exact_audit.items():
        if audit.get(key) != expected:
            raise EvaluationError(f"formal audit report has mismatched {key}")
    input_sha256 = sha256_file(input_path)
    if audit.get("evaluation_input_sha256") != input_sha256:
        raise EvaluationError("formal audit does not match evaluation_input.json")

    manifest_sha256 = sha256_file(manifest_path)
    if value.get("run_manifest_sha256") != manifest_sha256:
        raise EvaluationError("evaluation input run-manifest hash is stale")
    if audit.get("run_manifest_sha256") != manifest_sha256:
        raise EvaluationError("formal audit run-manifest hash is stale")
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 2
        or manifest.get("benchmark") != "BEAM"
        or manifest.get("status") != "complete"
        or manifest.get("failed_conversations") not in (None, [])
    ):
        raise EvaluationError("formal run manifest is not complete and failure-free")


def load_evaluation_input(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise EvaluationError("evaluation input schema version is not 1")
    exact = {
        "benchmark": "BEAM",
        "metric_scope": "rubric_nugget_only",
        "method": "NativeMem-v8.8+calendar",
        "answer_model": "gpt-5.5",
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise EvaluationError(f"evaluation input has mismatched {key}")
    if value.get("formal_scope_verified") is not True:
        raise EvaluationError("evaluation input is not a formal-scope audit artifact")
    if value.get("selected_conversations") != FORMAL_SELECTION:
        raise EvaluationError("evaluation input does not contain the formal BEAM scope")
    _validate_input_provenance(path, value)
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise EvaluationError("evaluation input has no records")
    if value.get("question_count") != len(records):
        raise EvaluationError("evaluation input question count mismatch")
    if len(records) != FORMAL_QUESTION_COUNT:
        raise EvaluationError("formal BEAM input must contain exactly 900 questions")
    if any(not isinstance(record, dict) for record in records):
        raise EvaluationError("evaluation input contains a non-object record")
    ids = [record.get("question_id") for record in records]
    if any(not isinstance(question_id, str) or not question_id for question_id in ids):
        raise EvaluationError("evaluation input has a missing question id")
    if len(set(ids)) != len(ids):
        raise EvaluationError("evaluation input has duplicate question ids")
    types = Counter(record.get("question_type") for record in records)
    if set(types) != set(EXPECTED_QUESTION_TYPES):
        raise EvaluationError(
            f"evaluation input does not contain all ten BEAM types: {dict(types)}"
        )
    if value.get("question_type_counts") != dict(types):
        raise EvaluationError("evaluation input question-type counts mismatch")
    if dict(types) != {
        question_type: FORMAL_QUESTION_TYPE_COUNT
        for question_type in EXPECTED_QUESTION_TYPES
    }:
        raise EvaluationError("formal BEAM input must contain 90 questions per type")
    expected_conversations = {
        (chat_size, conv_idx)
        for chat_size, indices in FORMAL_SELECTION.items()
        for conv_idx in indices
    }
    records_by_conversation: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    nugget_count = 0
    for record in records:
        chat_size = record.get("chat_size")
        if not isinstance(chat_size, str):
            raise EvaluationError(f"{record['question_id']} has invalid chat size")
        conv_idx = record.get("conversation_index")
        if isinstance(conv_idx, bool) or not isinstance(conv_idx, int):
            raise EvaluationError(
                f"{record['question_id']} has invalid conversation index"
            )
        conversation_key = (chat_size, conv_idx)
        if conversation_key not in expected_conversations:
            raise EvaluationError(
                f"{record['question_id']} is outside the formal conversation scope"
            )
        question_index = record.get("question_index")
        if isinstance(question_index, bool) or not isinstance(question_index, int):
            raise EvaluationError(f"{record['question_id']} has invalid question index")
        question_type = record.get("question_type")
        expected_id = f"{chat_size}_{conv_idx}_q{question_index}_{question_type}"
        if record["question_id"] != expected_id:
            raise EvaluationError(
                f"formal question id mismatch: expected {expected_id!r}"
            )
        records_by_conversation[conversation_key].append(record)
        for key in ("question", "answer"):
            if not str(record.get(key, "")).strip():
                raise EvaluationError(f"{record['question_id']} has empty {key}")
        rubric = record.get("rubric")
        if (
            not isinstance(rubric, list)
            or not rubric
            or any(not isinstance(item, str) or not item.strip() for item in rubric)
        ):
            raise EvaluationError(f"{record['question_id']} has invalid rubric")
        nugget_count += len(rubric)
    if value.get("rubric_nugget_count") != nugget_count:
        raise EvaluationError("evaluation input rubric nugget count mismatch")
    if set(records_by_conversation) != expected_conversations:
        raise EvaluationError("formal input does not cover every selected conversation")
    expected_indices = set(range(FORMAL_QUESTIONS_PER_CONVERSATION))
    expected_type_counts = {
        question_type: FORMAL_QUESTIONS_PER_TYPE_PER_CONVERSATION
        for question_type in EXPECTED_QUESTION_TYPES
    }
    for conversation_key, conversation_records in records_by_conversation.items():
        if len(conversation_records) != FORMAL_QUESTIONS_PER_CONVERSATION:
            raise EvaluationError(
                f"formal conversation {conversation_key} does not contain "
                f"{FORMAL_QUESTIONS_PER_CONVERSATION} questions"
            )
        indices = [record["question_index"] for record in conversation_records]
        if len(set(indices)) != len(indices) or set(indices) != expected_indices:
            raise EvaluationError(
                f"formal conversation {conversation_key} has invalid question indices"
            )
        type_counts = Counter(
            record["question_type"] for record in conversation_records
        )
        if dict(type_counts) != expected_type_counts:
            raise EvaluationError(
                f"formal conversation {conversation_key} has invalid type inventory"
            )
    return value


def job_id(question_id: str, nugget_index: int) -> str:
    if not question_id.replace("_", "").replace("-", "").isalnum():
        digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16]
        question_id = f"question_{digest}"
    return f"{question_id}__n{nugget_index:03d}"


def judgment_path(output_dir: Path, identifier: str) -> Path:
    return output_dir / "judgments" / f"{identifier}.json"


def job_attempt_ledger_path(output_dir: Path, identifier: str) -> Path:
    return output_dir / "attempt_ledgers" / f"{identifier}.jsonl"


def append_job_attempt_event(
    output_dir: Path, identifier: str, event: dict[str, Any]
) -> dict[str, Any]:
    """Append and fsync one per-job event before state may advance."""
    path = job_attempt_ledger_path(output_dir, identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = {
        "schema_version": ATTEMPT_LEDGER_SCHEMA,
        "event_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "job_id": identifier,
        **copy.deepcopy(event),
    }
    raw = (json.dumps(persisted, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("BEAM attempt ledger append wrote zero bytes")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return persisted


def read_job_attempt_events(output_dir: Path, identifier: str) -> list[dict[str, Any]]:
    path = job_attempt_ledger_path(output_dir, identifier)
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise EvaluationError(
            f"attempt ledger for {identifier} has a non-durable partial tail"
        )
    events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            event = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"attempt ledger for {identifier} has invalid line {line_number}"
            ) from exc
        if (
            not isinstance(event, dict)
            or event.get("schema_version") != ATTEMPT_LEDGER_SCHEMA
            or event.get("job_id") != identifier
            or not isinstance(event.get("event_id"), str)
            or not event["event_id"]
            or event["event_id"] in event_ids
        ):
            raise EvaluationError(
                f"attempt ledger for {identifier} has invalid event identity"
            )
        event_ids.add(event["event_id"])
        events.append(event)
    return events


def job_attempt_ledger_descriptor(
    output_dir: Path, identifier: str
) -> dict[str, Any]:
    path = job_attempt_ledger_path(output_dir, identifier)
    events = read_job_attempt_events(output_dir, identifier)
    return {
        "schema_version": ATTEMPT_LEDGER_SCHEMA,
        "path": path.relative_to(output_dir).as_posix(),
        "event_count": len(events),
        "sha256": sha256_file(path) if path.exists() else stable_hash([]),
    }


def immediate_proxy_evidence(
    config: dict[str, Any], response_evidence: dict[str, Any] | None
) -> dict[str, Any]:
    response_id = (
        response_evidence.get("response_id")
        if isinstance(response_evidence, dict)
        else None
    )
    if config["profile"] == "primary":
        status = "not_applicable_primary_openrouter"
    elif response_id:
        status = "response_id_pending_post_audit_linkage"
    else:
        status = "unlinked_transport_or_missing_identity"
    return {
        "status": status,
        "profile": config["profile"],
        "proxy_log": config.get("proxy_request_log"),
        "response_id": response_id,
    }


def completed_job_attempts(
    output_dir: Path,
    identifier: str,
    *,
    input_sha256: str,
    config_hash: str,
) -> list[dict[str, Any]]:
    """Reconstruct completed attempts and reject any unresolved started call."""
    events = read_job_attempt_events(output_dir, identifier)
    attempts: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for event in events:
        if (
            event.get("input_sha256") != input_sha256
            or event.get("config_hash") != config_hash
        ):
            raise EvaluationError(
                f"attempt ledger for {identifier} has mismatched run identity"
            )
        event_type = event.get("event")
        attempt_number = event.get("attempt_number")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise EvaluationError(
                f"attempt ledger for {identifier} has invalid attempt number"
            )
        if event_type == "attempt_started":
            if pending is not None or attempt_number != len(attempts) + 1:
                raise EvaluationError(
                    f"attempt ledger for {identifier} has invalid start ordering"
                )
            pending = event
            continue
        if event_type != "attempt_finished" or pending is None:
            raise EvaluationError(
                f"attempt ledger for {identifier} has invalid terminal ordering"
            )
        if (
            attempt_number != pending["attempt_number"]
            or event.get("started_event_id") != pending["event_id"]
        ):
            raise EvaluationError(
                f"attempt ledger for {identifier} has mismatched terminal event"
            )
        status = event.get("status")
        response_evidence = event.get("response_evidence")
        if status not in {"complete", "failed"}:
            raise EvaluationError(
                f"attempt ledger for {identifier} has invalid terminal status"
            )
        if status == "complete" and not isinstance(response_evidence, dict):
            raise EvaluationError(
                f"attempt ledger for {identifier} lacks successful response evidence"
            )
        if status == "failed" and response_evidence is not None and not isinstance(
            response_evidence, dict
        ):
            raise EvaluationError(
                f"attempt ledger for {identifier} has invalid failure evidence"
            )
        attempt = {
            "attempt_number": attempt_number,
            "started_at": pending.get("started_at"),
            "finished_at": event.get("finished_at"),
            "status": status,
            "error_type": event.get("error_type"),
            "error_message": event.get("error_message"),
            "response_evidence": response_evidence,
            "proxy_evidence": event.get("proxy_evidence"),
        }
        attempts.append(attempt)
        pending = None
    if pending is not None:
        raise EvaluationError(
            "attempt ledger contains an unresolved provider call; refusing resume: "
            f"job={identifier}, attempt={pending['attempt_number']}"
        )
    if any(item["status"] == "complete" for item in attempts[:-1]):
        raise EvaluationError(
            f"attempt ledger for {identifier} continued after a successful response"
        )
    return attempts


def prompt_for(record: dict[str, Any], nugget: str) -> str:
    return BEAM_PROMPTS.get_beam_nugget_judge_prompt(
        record["question"], nugget, record["answer"]
    )


def _usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def validate_usage_record(value: object) -> dict[str, int]:
    expected_keys = {"prompt_tokens", "completion_tokens", "total_tokens"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise EvaluationError("judge usage record has invalid fields")
    normalized: dict[str, int] = {}
    for key in expected_keys:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise EvaluationError(f"judge usage record has invalid {key}")
        normalized[key] = item
    if normalized["total_tokens"] != (
        normalized["prompt_tokens"] + normalized["completion_tokens"]
    ):
        raise EvaluationError("judge usage total does not match token components")
    return normalized


def validate_raw_response_record(value: object) -> dict[str, Any]:
    """Parse the canonical raw judge response stored beside derived fields."""
    if not isinstance(value, dict) or set(value) != {
        "id",
        "model",
        "choices",
        "usage",
    }:
        raise EvaluationError("stored raw judge response has invalid fields")
    response_id = value["id"]
    response_model = value["model"]
    if not isinstance(response_id, str) or not response_id.strip():
        raise EvaluationError("stored raw judge response has no response id")
    if not isinstance(response_model, str) or not response_model.strip():
        raise EvaluationError("stored raw judge response has no response model")
    choices = value["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise EvaluationError("stored raw judge response has invalid choices")
    choice = choices[0]
    if not isinstance(choice, dict) or set(choice) != {"finish_reason", "message"}:
        raise EvaluationError("stored raw judge response has invalid choice fields")
    if choice["finish_reason"] != "stop":
        raise EvaluationError("stored raw judge response has non-stop finish reason")
    message = choice["message"]
    if not isinstance(message, dict) or set(message) != {"content", "refusal"}:
        raise EvaluationError("stored raw judge response has invalid message fields")
    refusal = message["refusal"]
    if refusal not in (None, ""):
        raise EvaluationError("stored raw judge response contains a refusal")
    parsed = parse_strict_judgment(message["content"])
    return {
        **parsed,
        "response_id": response_id.strip(),
        "response_model": response_model.strip(),
        "finish_reason": "stop",
        "usage": validate_usage_record(value["usage"]),
    }


def response_evidence_sha256(value: dict[str, Any]) -> str:
    return stable_hash(
        {
            key: value.get(key)
            for key in (
                "raw_response",
                "requested_model",
                "response_model",
                "response_id",
                "finish_reason",
                "usage",
            )
        }
    )


def response_model_matches(profile: str, value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    model = value.strip()
    if profile == "secondary":
        return model == "gpt-5.5"
    if profile == "primary":
        match = PRIMARY_RESPONSE_MODEL_RE.fullmatch(model)
        if match is None:
            return False
        date_suffix = match.group("date")
        if date_suffix is None:
            return True
        try:
            datetime.strptime(date_suffix, "%Y-%m-%d")
        except ValueError:
            return False
        return True
    return False


def call_judge(
    client: Any,
    *,
    model: str,
    record: dict[str, Any],
    nugget: str,
    max_tokens: int,
    expected_response_model: str,
    judge_profile: str,
) -> dict[str, Any]:
    prompt = prompt_for(record, nugget)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BEAM_PROMPTS.BEAM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    response_model = str(getattr(response, "model", "") or "").strip()
    response_id = str(getattr(response, "id", "") or "").strip()
    evidence: dict[str, Any] = {
        "requested_model": model,
        "response_model": response_model or None,
        "response_id": response_id or None,
    }
    if not response_model or not response_id:
        raise JudgeResponseError(
            "judge response is missing model or response id", evidence
        )
    usage = _usage_dict(response)
    evidence["usage"] = usage
    if not response_model_matches(judge_profile, response_model):
        raise JudgeResponseError(
            f"judge response model {response_model!r} differs from required "
            f"{expected_response_model!r}",
            evidence,
        )
    choices = getattr(response, "choices", None) or []
    if len(choices) != 1:
        raise JudgeResponseError(
            "judge response does not contain exactly one choice", evidence
        )
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    evidence["finish_reason"] = finish_reason
    if finish_reason != "stop":
        raise JudgeResponseError(
            f"judge response finish reason is {finish_reason!r}", evidence
        )
    message = getattr(choice, "message", None)
    refusal = str(getattr(message, "refusal", "") or "").strip()
    evidence["refusal"] = refusal or None
    if refusal:
        raise JudgeResponseError("judge refused the evaluation request", evidence)
    raw_content = getattr(message, "content", None)
    evidence["raw_response"] = raw_content
    try:
        parsed = parse_strict_judgment(raw_content)
    except EvaluationError as exc:
        raise JudgeResponseError(str(exc), evidence) from exc
    try:
        normalized_usage = validate_usage_record(usage)
    except EvaluationError as exc:
        raise JudgeResponseError(str(exc), evidence) from exc
    raw_response = {
        "id": response_id,
        "model": response_model,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": raw_content,
                    "refusal": refusal or None,
                },
            }
        ],
        "usage": normalized_usage,
    }
    complete = {
        **parsed,
        "raw_response": raw_response,
        "requested_model": model,
        "response_model": response_model,
        "response_id": response_id,
        "finish_reason": "stop",
        "usage": normalized_usage,
    }
    complete["response_evidence_sha256"] = response_evidence_sha256(complete)
    return complete


def evaluation_config(
    *,
    profile: str,
    base_url: str,
    max_tokens: int,
    max_retries: int,
    proxy_log: Path | None,
    openrouter_gateway: dict[str, Any] | None = None,
    flex_gateway_contract: dict[str, Any] | None = None,
    formal_transport_contract: bool = False,
) -> dict[str, Any]:
    spec = JUDGE_PROFILES[profile]
    if profile == "primary" and formal_transport_contract:
        if not isinstance(openrouter_gateway, dict):
            raise EvaluationError(
                "formal primary evaluation lacks OpenRouter gateway evidence"
            )
        base_url = str(openrouter_gateway["base_url"])
        transport_contract = "marked_openrouter_gateway"
    elif profile == "primary":
        transport_contract = "injected_test_judge"
    elif formal_transport_contract:
        if openrouter_gateway is not None:
            raise EvaluationError(
                "secondary evaluation cannot bind an OpenRouter gateway"
            )
        if not isinstance(flex_gateway_contract, dict):
            raise EvaluationError(
                "formal secondary evaluation lacks Flex gateway evidence"
            )
        try:
            flex_evidence.validate_recorded_contract(flex_gateway_contract)
        except flex_evidence.EvidenceError as exc:
            raise EvaluationError(str(exc)) from exc
        if proxy_log is not None:
            raise EvaluationError(
                "formal secondary evaluation does not accept a shared proxy log"
            )
        base_url = str(flex_gateway_contract["base_url"])
        transport_contract = "openai_gpt55_flex_gateway"
    else:
        if openrouter_gateway is not None or flex_gateway_contract is not None:
            raise EvaluationError("injected secondary evaluation claims gateway evidence")
        transport_contract = "secondary_local_proxy"
    return {
        "formal": True,
        "profile": profile,
        "comparison_label": spec["comparison_label"],
        "comparison_scope": spec["comparison_scope"],
        "comparable_to_project_unified_protocol": spec[
            "comparable_to_project_unified_protocol"
        ],
        "comparable_to_published_beam_official": spec[
            "comparable_to_published_beam_official"
        ],
        "comparable_to_mem0": spec["comparable_to_mem0"],
        "metric": "BEAM rubric-nugget mean",
        "metric_scope": "rubric_nugget_only",
        "pass_threshold": 0.5,
        "model": spec["model"],
        "provider": spec["provider"],
        "base_url": base_url.rstrip("/"),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "expected_response_model": spec["expected_response_model"],
        "proxy_request_log": str(proxy_log) if proxy_log else None,
        "transport_contract": transport_contract,
        "openrouter_gateway": openrouter_gateway,
        "openai_flex_gateway": flex_gateway_contract,
        "evaluator_source": str(SCRIPT_PATH.relative_to(ROOT)),
        "evaluator_source_sha256": sha256_file(SCRIPT_PATH),
        "prompt_source": str(PROMPTS_PATH.relative_to(ROOT)),
        "prompt_source_sha256": sha256_file(PROMPTS_PATH),
        "gateway_evidence_source_sha256": sha256_file(
            ROOT / "src/openrouter_gateway_evidence.py"
        ),
        "flex_gateway_evidence_source_sha256": sha256_file(
            ROOT / "src/openai_gpt55_flex_gateway_evidence.py"
        ),
        "system_prompt_sha256": hashlib.sha256(
            BEAM_PROMPTS.BEAM_JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "event_ordering": {
            "scope": "nugget-only",
            "official_tau_b_times_f1": "not_computed",
        },
    }


def load_or_create_state(
    output_dir: Path,
    *,
    input_path: Path,
    input_sha256: str,
    config: dict[str, Any],
    total_jobs: int,
    resume: bool,
) -> dict[str, Any]:
    state_path = output_dir / "run_state.json"
    config_hash = stable_hash(config)
    if state_path.exists():
        if not resume:
            raise EvaluationError(
                f"evaluation state exists at {state_path}; pass --resume or use "
                "a different --output-dir"
            )
        state = read_json(state_path)
        if state.get("schema_version") != 1 or state.get("benchmark") != "BEAM":
            raise EvaluationError("stored evaluation state has invalid identity")
        if state.get("config") != config or stable_hash(state["config"]) != state.get(
            "config_hash"
        ):
            raise EvaluationError("stored judge configuration is not self-consistent")
        if state.get("input_sha256") != input_sha256:
            raise EvaluationError("evaluation input changed since this run started")
        if state.get("config_hash") != config_hash:
            raise EvaluationError("judge configuration changed since this run started")
        if state.get("total_nuggets") != total_jobs:
            raise EvaluationError("stored nugget count differs from evaluation input")
        return state
    unexpected = (
        [path for path in output_dir.iterdir() if path.name != LOCK_FILENAME]
        if output_dir.exists()
        else []
    )
    if unexpected:
        raise EvaluationError(
            f"output directory is non-empty without run_state.json: {output_dir}"
        )
    state = {
        "schema_version": 1,
        "benchmark": "BEAM",
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "config": config,
        "config_hash": config_hash,
        "total_nuggets": total_jobs,
        "completed_nuggets": 0,
        "failed_nuggets": 0,
    }
    atomic_json(state_path, state)
    return state


def validate_stored_judgment(
    value: dict[str, Any],
    *,
    identifier: str,
    record: dict[str, Any],
    nugget_index: int,
    nugget: str,
    input_sha256: str,
    config_hash: str,
    config: dict[str, Any],
) -> None:
    if stable_hash(config) != config_hash:
        raise EvaluationError("judge configuration hash is not self-consistent")
    exact = {
        "schema_version": 1,
        "job_id": identifier,
        "question_id": record["question_id"],
        "nugget_index": nugget_index,
        "nugget": nugget,
        "input_sha256": input_sha256,
        "config_hash": config_hash,
        "judge_profile": config["profile"],
        "judge_provider": config["provider"],
        "comparison_label": config["comparison_label"],
        "requested_model": config["model"],
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise EvaluationError(f"stored judgment {identifier} has mismatched {key}")
    if value.get("status") not in {"running", "failed", "complete"}:
        raise EvaluationError(f"stored judgment {identifier} has invalid status")
    attempts = value.get("attempts")
    if not isinstance(attempts, list):
        raise EvaluationError(f"stored judgment {identifier} has invalid attempts")
    required_attempt_keys = {
        "attempt_number",
        "started_at",
        "finished_at",
        "status",
        "error_type",
        "error_message",
        "response_evidence",
        "proxy_evidence",
    }
    for attempt_number, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, dict)
            or set(attempt) != required_attempt_keys
            or attempt.get("attempt_number") != attempt_number
            or attempt.get("status") not in {"complete", "failed"}
            or not isinstance(attempt.get("started_at"), str)
            or not isinstance(attempt.get("finished_at"), str)
        ):
            raise EvaluationError(
                f"stored judgment {identifier} has malformed attempt {attempt_number}"
            )
        proxy_evidence = attempt.get("proxy_evidence")
        if not isinstance(proxy_evidence, dict) or set(proxy_evidence) != {
            "status", "profile", "proxy_log", "response_id"
        }:
            raise EvaluationError(
                f"stored judgment {identifier} has malformed proxy attempt evidence"
            )
        if proxy_evidence["profile"] != config["profile"]:
            raise EvaluationError(
                f"stored judgment {identifier} attempt profile differs"
            )
        response_evidence = attempt.get("response_evidence")
        if isinstance(response_evidence, dict):
            if response_evidence.get("requested_model") != config["model"]:
                raise EvaluationError(
                    f"stored judgment {identifier} attempt used an unexpected model"
                )
            for identity_key in ("response_model", "response_id"):
                identity = response_evidence.get(identity_key)
                if identity is not None and (
                    not isinstance(identity, str) or not identity.strip()
                ):
                    raise EvaluationError(
                        f"stored judgment {identifier} attempt has invalid {identity_key}"
                    )
            usage = response_evidence.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise EvaluationError(
                        f"stored judgment {identifier} attempt has invalid usage"
                    )
                for token_key in (
                    "prompt_tokens", "completion_tokens", "total_tokens"
                ):
                    token_value = usage.get(token_key)
                    if token_value is not None and (
                        isinstance(token_value, bool)
                        or not isinstance(token_value, int)
                        or token_value < 0
                    ):
                        raise EvaluationError(
                            f"stored judgment {identifier} attempt has invalid tokens"
                        )
            if proxy_evidence["response_id"] != response_evidence.get("response_id"):
                raise EvaluationError(
                    f"stored judgment {identifier} proxy response id differs"
                )
        if attempt["status"] == "complete":
            if (
                not isinstance(response_evidence, dict)
                or attempt["error_type"] is not None
                or attempt["error_message"] is not None
            ):
                raise EvaluationError(
                    f"stored judgment {identifier} has invalid successful attempt"
                )
        elif (
            not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error_message"), str)
        ):
            raise EvaluationError(
                f"stored judgment {identifier} has invalid failed attempt"
            )
    successful = [attempt for attempt in attempts if attempt["status"] == "complete"]
    if successful and (len(successful) != 1 or successful[0] is not attempts[-1]):
        raise EvaluationError(
            f"stored judgment {identifier} has invalid successful attempt ordering"
        )
    logical_calls = value.get("logical_judge_calls")
    physical_attempts = value.get("physical_http_attempts")
    expected_calls = len(attempts) + int(value.get("status") == "running")
    if (
        isinstance(logical_calls, bool)
        or not isinstance(logical_calls, int)
        or logical_calls < 0
        or physical_attempts != logical_calls
        or logical_calls != expected_calls
    ):
        raise EvaluationError(
            f"stored judgment {identifier} has invalid attempt counters"
        )
    ledger = value.get("attempt_ledger")
    if (
        not isinstance(ledger, dict)
        or set(ledger) != {"schema_version", "path", "event_count", "sha256"}
        or ledger.get("schema_version") != ATTEMPT_LEDGER_SCHEMA
        or not isinstance(ledger.get("path"), str)
        or ledger.get("path") != f"attempt_ledgers/{identifier}.jsonl"
        or isinstance(ledger.get("event_count"), bool)
        or not isinstance(ledger.get("event_count"), int)
        or ledger["event_count"] < 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(ledger.get("sha256", "")))
    ):
        raise EvaluationError(
            f"stored judgment {identifier} has invalid attempt ledger descriptor"
        )
    if value.get("status") == "complete":
        if not attempts or attempts[-1]["status"] != "complete":
            raise EvaluationError(
                f"stored judgment {identifier} lacks a successful final attempt"
            )
        score = value.get("score")
        if isinstance(score, bool) or score not in VALID_SCORES:
            raise EvaluationError(f"stored judgment {identifier} has invalid score")
        for key in ("reason", "requested_model", "response_model", "response_id"):
            if not str(value.get(key, "")).strip():
                raise EvaluationError(f"stored judgment {identifier} has missing {key}")
        if not response_model_matches(config["profile"], value.get("response_model")):
            raise EvaluationError(
                f"stored judgment {identifier} has unexpected response model"
            )
        if value.get("finish_reason") != "stop":
            raise EvaluationError(
                f"stored judgment {identifier} has non-stop finish reason"
            )
        parsed = validate_raw_response_record(value.get("raw_response"))
        raw_exact = {
            "response_model": value.get("response_model"),
            "response_id": value.get("response_id"),
            "finish_reason": value.get("finish_reason"),
            "usage": value.get("usage"),
        }
        if any(parsed[key] != expected for key, expected in raw_exact.items()):
            raise EvaluationError(
                f"stored judgment {identifier} metadata differs from its raw response"
            )
        if float(score) != parsed["score"] or value.get("reason") != parsed["reason"]:
            raise EvaluationError(
                f"stored judgment {identifier} differs from its raw response"
            )
        if value.get("response_evidence_sha256") != response_evidence_sha256(value):
            raise EvaluationError(
                f"stored judgment {identifier} has invalid response evidence hash"
            )
        attempt_result = attempts[-1]["response_evidence"]
        for key in (
            "score", "reason", "raw_response", "requested_model", "response_model",
            "response_id", "finish_reason", "usage", "response_evidence_sha256",
        ):
            if attempt_result.get(key) != value.get(key):
                raise EvaluationError(
                    f"stored judgment {identifier} differs from final attempt evidence"
                )
    elif value.get("status") == "failed":
        if not attempts or attempts[-1]["status"] != "failed":
            raise EvaluationError(
                f"stored judgment {identifier} lacks a failed final attempt"
            )
        if value.get("error") != attempts[-1]:
            raise EvaluationError(
                f"stored judgment {identifier} error differs from final attempt"
            )
    elif "score" in value:
        raise EvaluationError(
            f"non-complete judgment {identifier} must not contain a score"
        )


def _job_payload_from_attempts(
    *,
    base: dict[str, Any],
    attempts: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    identifier = base["job_id"]
    descriptor = job_attempt_ledger_descriptor(output_dir, identifier)
    counters = {
        "logical_judge_calls": len(attempts),
        "physical_http_attempts": len(attempts),
        "attempt_ledger": descriptor,
    }
    if attempts and attempts[-1]["status"] == "complete":
        result = attempts[-1]["response_evidence"]
        return {
            **base,
            "status": "complete",
            "completed_at": attempts[-1]["finished_at"],
            "attempts": attempts,
            **counters,
            **result,
        }
    failed = {
        **base,
        "status": "failed",
        "failed_at": attempts[-1]["finished_at"] if attempts else utc_now(),
        "attempts": attempts,
        **counters,
    }
    if attempts:
        failed["error"] = attempts[-1]
    return failed


def validate_job_attempt_ledger(
    output_dir: Path,
    value: dict[str, Any],
    *,
    input_sha256: str,
    config_hash: str,
) -> None:
    identifier = value["job_id"]
    attempts = completed_job_attempts(
        output_dir,
        identifier,
        input_sha256=input_sha256,
        config_hash=config_hash,
    )
    if attempts != value.get("attempts"):
        raise EvaluationError(
            f"stored judgment {identifier} differs from its attempt ledger"
        )
    if job_attempt_ledger_descriptor(output_dir, identifier) != value.get(
        "attempt_ledger"
    ):
        raise EvaluationError(
            f"stored judgment {identifier} has stale attempt ledger metadata"
        )


def _job_base(
    *,
    identifier: str,
    record: dict[str, Any],
    nugget_index: int,
    nugget: str,
    input_sha256: str,
    config_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": identifier,
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "chat_size": record["chat_size"],
        "conversation_index": record["conversation_index"],
        "nugget_index": nugget_index,
        "nugget": nugget,
        "input_sha256": input_sha256,
        "config_hash": config_hash,
        "judge_profile": config["profile"],
        "judge_provider": config["provider"],
        "comparison_label": config["comparison_label"],
        "requested_model": config["model"],
    }


def judge_one_job(
    client: Any,
    *,
    output_dir: Path,
    record: dict[str, Any],
    nugget_index: int,
    nugget: str,
    input_sha256: str,
    config_hash: str,
    config: dict[str, Any],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    identifier = job_id(record["question_id"], nugget_index)
    path = judgment_path(output_dir, identifier)
    base = _job_base(
        identifier=identifier,
        record=record,
        nugget_index=nugget_index,
        nugget=nugget,
        input_sha256=input_sha256,
        config_hash=config_hash,
        config=config,
    )
    previous: dict[str, Any] = read_json(path) if path.exists() else {}
    attempts = completed_job_attempts(
        output_dir,
        identifier,
        input_sha256=input_sha256,
        config_hash=config_hash,
    )
    if previous:
        # A terminal ledger event may have been fsynced immediately before a
        # process exit.  Recover only a running snapshot; completed/failed
        # snapshots must already match their immutable ledger.
        if previous.get("status") == "running" and attempts:
            previous = _job_payload_from_attempts(
                base=base, attempts=attempts, output_dir=output_dir, config=config
            )
            atomic_json(path, previous)
        validate_stored_judgment(
            previous,
            identifier=identifier,
            record=record,
            nugget_index=nugget_index,
            nugget=nugget,
            input_sha256=input_sha256,
            config_hash=config_hash,
            config=config,
        )
        validate_job_attempt_ledger(
            output_dir,
            previous,
            input_sha256=input_sha256,
            config_hash=config_hash,
        )
        if previous.get("status") == "complete":
            return previous
    if len(attempts) >= config["max_retries"]:
        return previous or _job_payload_from_attempts(
            base=base, attempts=attempts, output_dir=output_dir, config=config
        )
    for attempt_number in range(len(attempts) + 1, config["max_retries"] + 1):
        started_at = utc_now()
        started = append_job_attempt_event(
            output_dir,
            identifier,
            {
                "event": "attempt_started",
                "attempt_number": attempt_number,
                "started_at": started_at,
                "input_sha256": input_sha256,
                "config_hash": config_hash,
            },
        )
        atomic_json(
            path,
            {
                **base,
                "status": "running",
                "started_at": started_at,
                "attempts": attempts,
                "logical_judge_calls": attempt_number,
                "physical_http_attempts": attempt_number,
                "attempt_ledger": job_attempt_ledger_descriptor(
                    output_dir, identifier
                ),
            },
        )
        try:
            result = call_judge(
                client,
                model=config["model"],
                record=record,
                nugget=nugget,
                max_tokens=config["max_tokens"],
                expected_response_model=config["expected_response_model"],
                judge_profile=config["profile"],
            )
        except Exception as exc:  # noqa: BLE001
            response_evidence = (
                copy.deepcopy(exc.evidence)
                if isinstance(exc, JudgeResponseError)
                else None
            )
            append_job_attempt_event(
                output_dir,
                identifier,
                {
                    "event": "attempt_finished",
                    "attempt_number": attempt_number,
                    "started_event_id": started["event_id"],
                    "finished_at": utc_now(),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2000],
                    "response_evidence": response_evidence,
                    "proxy_evidence": immediate_proxy_evidence(
                        config, response_evidence
                    ),
                    "input_sha256": input_sha256,
                    "config_hash": config_hash,
                },
            )
            attempts = completed_job_attempts(
                output_dir,
                identifier,
                input_sha256=input_sha256,
                config_hash=config_hash,
            )
            failed = _job_payload_from_attempts(
                base=base, attempts=attempts, output_dir=output_dir, config=config
            )
            atomic_json(path, failed)
            if attempt_number < config["max_retries"]:
                sleep_fn(min(8.0, float(2 ** (attempt_number - 1))))
                continue
            return failed
        append_job_attempt_event(
            output_dir,
            identifier,
            {
                "event": "attempt_finished",
                "attempt_number": attempt_number,
                "started_event_id": started["event_id"],
                "finished_at": utc_now(),
                "status": "complete",
                "error_type": None,
                "error_message": None,
                "response_evidence": result,
                "proxy_evidence": immediate_proxy_evidence(config, result),
                "input_sha256": input_sha256,
                "config_hash": config_hash,
            },
        )
        attempts = completed_job_attempts(
            output_dir,
            identifier,
            input_sha256=input_sha256,
            config_hash=config_hash,
        )
        complete = _job_payload_from_attempts(
            base=base, attempts=attempts, output_dir=output_dir, config=config
        )
        atomic_json(path, complete)
        return complete
    raise AssertionError("unreachable retry loop")


def load_all_judgments(
    output_dir: Path,
    evaluation_input: dict[str, Any],
    *,
    input_sha256: str,
    config_hash: str,
    config: dict[str, Any],
    recover_running: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    judgments: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for record in evaluation_input["records"]:
        for nugget_index, nugget in enumerate(record["rubric"]):
            identifier = job_id(record["question_id"], nugget_index)
            path = judgment_path(output_dir, identifier)
            if not path.exists():
                missing.append(identifier)
                continue
            value = read_json(path)
            if recover_running and value.get("status") == "running":
                attempts = completed_job_attempts(
                    output_dir,
                    identifier,
                    input_sha256=input_sha256,
                    config_hash=config_hash,
                )
                if attempts:
                    base = _job_base(
                        identifier=identifier,
                        record=record,
                        nugget_index=nugget_index,
                        nugget=nugget,
                        input_sha256=input_sha256,
                        config_hash=config_hash,
                        config=config,
                    )
                    value = _job_payload_from_attempts(
                        base=base,
                        attempts=attempts,
                        output_dir=output_dir,
                        config=config,
                    )
                    atomic_json(path, value)
            validate_stored_judgment(
                value,
                identifier=identifier,
                record=record,
                nugget_index=nugget_index,
                nugget=nugget,
                input_sha256=input_sha256,
                config_hash=config_hash,
                config=config,
            )
            validate_job_attempt_ledger(
                output_dir,
                value,
                input_sha256=input_sha256,
                config_hash=config_hash,
            )
            judgments[identifier] = value
    return judgments, missing


def _group_metrics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [evaluation["score"] for evaluation in evaluations]
    passed = sum(score >= 0.5 for score in scores)
    return {
        "questions": len(scores),
        "rubric_nuggets": sum(
            len(evaluation["nugget_scores"]) for evaluation in evaluations
        ),
        "avg_score": mean(scores) if scores else None,
        "pass_threshold": 0.5,
        "passed": passed,
        "pass_rate": passed / len(scores) if scores else None,
        "pass_accuracy_percent": passed / len(scores) * 100 if scores else None,
    }


def aggregate_results(
    evaluation_input: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    incomplete_jobs: list[str] = []
    response_models: Counter[str] = Counter()
    response_ids: set[str] = set()
    for record in evaluation_input["records"]:
        nugget_scores: list[dict[str, Any]] = []
        for nugget_index, nugget in enumerate(record["rubric"]):
            identifier = job_id(record["question_id"], nugget_index)
            judgment = judgments.get(identifier)
            if not judgment or judgment.get("status") != "complete":
                incomplete_jobs.append(identifier)
                continue
            response_id = judgment["response_id"]
            if response_id in response_ids:
                raise EvaluationError(f"judge response id is duplicated: {response_id}")
            response_ids.add(response_id)
            if judgment.get("requested_model") != config["model"]:
                raise EvaluationError(
                    f"judge requested model mismatch for {identifier}"
                )
            if not response_model_matches(
                config["profile"], judgment.get("response_model")
            ):
                raise EvaluationError(f"judge response model mismatch for {identifier}")
            response_models[judgment["response_model"]] += 1
            nugget_scores.append(
                {
                    "nugget_index": nugget_index,
                    "nugget": nugget,
                    "score": float(judgment["score"]),
                    "reason": judgment["reason"],
                    "requested_model": judgment["requested_model"],
                    "response_model": judgment["response_model"],
                    "response_id": response_id,
                }
            )
        complete = len(nugget_scores) == len(record["rubric"])
        score = mean(item["score"] for item in nugget_scores) if complete else None
        evaluations.append(
            {
                "question_id": record["question_id"],
                "chat_size": record["chat_size"],
                "conversation_index": record["conversation_index"],
                "question_type": record["question_type"],
                "difficulty": record.get("difficulty"),
                "question": record["question"],
                "answer": record["answer"],
                "status": "complete" if complete else "incomplete",
                "score": score,
                "judgment": (
                    "PASS"
                    if score is not None and score >= 0.5
                    else "FAIL"
                    if score is not None
                    else None
                ),
                "nugget_scores": nugget_scores,
            }
        )

    complete_evaluations = [
        evaluation for evaluation in evaluations if evaluation["status"] == "complete"
    ]
    all_complete = len(complete_evaluations) == len(evaluations)
    metrics: dict[str, Any] | None = None
    if all_complete:
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evaluation in complete_evaluations:
            by_type[evaluation["question_type"]].append(evaluation)
            by_split[evaluation["chat_size"]].append(evaluation)
        if set(by_type) != set(EXPECTED_QUESTION_TYPES):
            raise EvaluationError(
                "complete evaluations do not cover all ten BEAM types"
            )
        metrics = {
            "metric_scope": "rubric_nugget_only",
            "overall": _group_metrics(complete_evaluations),
            "by_question_type": {
                question_type: _group_metrics(by_type[question_type])
                for question_type in EXPECTED_QUESTION_TYPES
            },
            "by_chat_size": {
                chat_size: _group_metrics(values)
                for chat_size, values in sorted(by_split.items())
            },
            "event_ordering": {
                **_group_metrics(by_type["event_ordering"]),
                "score_used": "rubric_nugget_mean",
                "official_tau_b_times_f1": None,
                "official_metric_status": "not_computed",
                "included_in_overall_as": "rubric_nugget_mean",
                "reason": (
                    "No verified official tau-b times F1 implementation is used. "
                    "The vendored event-alignment heuristic computes a different "
                    "quantity and is excluded."
                ),
            },
        }
    return {
        "schema_version": 1,
        "benchmark": "BEAM",
        "status": "complete" if all_complete else "incomplete",
        "judge_profile": config["profile"],
        "judge_provider": config["provider"],
        "comparison_label": config["comparison_label"],
        "comparison_scope": config["comparison_scope"],
        "comparable_to_project_unified_protocol": config[
            "comparable_to_project_unified_protocol"
        ],
        "comparable_to_published_beam_official": config[
            "comparable_to_published_beam_official"
        ],
        "comparable_to_mem0": config["comparable_to_mem0"],
        "metric_scope": "rubric_nugget_only",
        "question_count": len(evaluations),
        "complete_questions": len(complete_evaluations),
        "incomplete_jobs": sorted(set(incomplete_jobs)),
        "judge_response_models": dict(response_models),
        "judge_response_ids": len(response_ids),
        "metrics": metrics,
        "evaluations": evaluations,
    }


def validate_formal_config(config: dict[str, Any]) -> None:
    profile = config.get("profile")
    if profile not in JUDGE_PROFILES:
        raise EvaluationError(f"unknown judge profile in config: {profile!r}")
    spec = JUDGE_PROFILES[profile]
    exact = {
        "formal": True,
        "comparison_label": spec["comparison_label"],
        "comparison_scope": spec["comparison_scope"],
        "comparable_to_project_unified_protocol": spec[
            "comparable_to_project_unified_protocol"
        ],
        "comparable_to_published_beam_official": spec[
            "comparable_to_published_beam_official"
        ],
        "comparable_to_mem0": spec["comparable_to_mem0"],
        "metric": "BEAM rubric-nugget mean",
        "metric_scope": "rubric_nugget_only",
        "pass_threshold": 0.5,
        "model": spec["model"],
        "provider": spec["provider"],
        "expected_response_model": spec["expected_response_model"],
        "evaluator_source": str(SCRIPT_PATH.relative_to(ROOT)),
        "evaluator_source_sha256": sha256_file(SCRIPT_PATH),
        "prompt_source": str(PROMPTS_PATH.relative_to(ROOT)),
        "prompt_source_sha256": sha256_file(PROMPTS_PATH),
        "gateway_evidence_source_sha256": sha256_file(
            ROOT / "src/openrouter_gateway_evidence.py"
        ),
        "flex_gateway_evidence_source_sha256": sha256_file(
            ROOT / "src/openai_gpt55_flex_gateway_evidence.py"
        ),
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise EvaluationError(f"formal judge config has mismatched {key}")
    if not str(config.get("base_url", "")).strip():
        raise EvaluationError("formal judge config has no base URL")
    transport_contract = config.get("transport_contract")
    gateway_binding = config.get("openrouter_gateway")
    flex_gateway_binding = config.get("openai_flex_gateway")
    if profile == "primary" and transport_contract == "marked_openrouter_gateway":
        if not isinstance(gateway_binding, dict) or flex_gateway_binding is not None:
            raise EvaluationError("primary gateway binding is absent")
        try:
            openrouter_gateway_evidence.validate_binding(
                gateway_binding,
                result_root=Path(str(gateway_binding.get("result_root", ""))),
                base_url=str(gateway_binding.get("base_url", "")),
            )
        except openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise EvaluationError(str(exc)) from exc
        if config.get("base_url") != gateway_binding.get("base_url"):
            raise EvaluationError("primary gateway base URL differs")
        if config.get("max_retries") != 1:
            raise EvaluationError(
                "marked OpenRouter gateway evaluation forbids automatic retries"
            )
    elif profile == "primary" and transport_contract == "injected_test_judge":
        if (
            gateway_binding is not None
            or flex_gateway_binding is not None
            or config.get("base_url") != spec["base_url"]
        ):
            raise EvaluationError(
                "formal judge config has a non-canonical base URL or transport"
            )
    elif profile == "secondary" and transport_contract == "openai_gpt55_flex_gateway":
        if gateway_binding is not None or not isinstance(flex_gateway_binding, dict):
            raise EvaluationError("secondary Flex gateway binding differs")
        try:
            flex_evidence.validate_recorded_contract(flex_gateway_binding)
        except flex_evidence.EvidenceError as exc:
            raise EvaluationError(str(exc)) from exc
        if config.get("base_url") != flex_gateway_binding.get("base_url"):
            raise EvaluationError("secondary Flex gateway base URL differs")
    elif profile == "secondary":
        if (
            transport_contract != "secondary_local_proxy"
            or gateway_binding is not None
            or flex_gateway_binding is not None
            or config.get("base_url") != spec["base_url"]
        ):
            raise EvaluationError(
                "formal judge config has a non-canonical base URL or transport"
            )
    else:
        raise EvaluationError("formal judge transport contract is invalid")
    for key in ("max_tokens", "max_retries"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise EvaluationError(f"formal judge config has invalid {key}")
    proxy_log = config.get("proxy_request_log")
    if (
        profile == "secondary"
        and transport_contract == "secondary_local_proxy"
        and not str(proxy_log or "").strip()
    ):
        raise EvaluationError("secondary profile requires a frozen proxy request log")
    if (
        profile == "secondary"
        and transport_contract == "openai_gpt55_flex_gateway"
        and proxy_log is not None
    ):
        raise EvaluationError("secondary Flex profile must not claim a shared proxy log")
    if profile == "primary" and proxy_log is not None:
        raise EvaluationError("primary profile must not claim local proxy evidence")


def _proxy_matches(
    path: Path,
    response_ids: set[str],
    *,
    requested_model: str,
    actual_model: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"judge proxy log is missing: {path}")
    matched: dict[str, dict[str, Any]] = {}
    entries = 0
    upstream_attempts = 0
    terminal_errors = 0
    error_evidence = []
    all_unsupported: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entries += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"invalid judge proxy log line {line_number}"
                ) from exc
            status = entry.get("status")
            attempts = entry.get("attempts")
            unsupported = entry.get("unsupported_parameters")
            if (
                status not in {"success", "error"}
                or isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or not 1 <= attempts <= 8
                or not isinstance(unsupported, list)
                or any(not isinstance(value, str) or not value for value in unsupported)
                or len(set(unsupported)) != len(unsupported)
            ):
                raise EvaluationError(
                    f"judge proxy log line {line_number} has invalid attempts"
                )
            upstream_attempts += attempts
            all_unsupported.update(unsupported)
            if status == "error":
                terminal_errors += 1
                if entry.get("response_id") is not None or not isinstance(
                    entry.get("error"), str
                ):
                    raise EvaluationError(
                        f"judge proxy log line {line_number} has invalid error evidence"
                    )
                error_evidence.append({
                    "attempts": attempts,
                    "unsupported_parameters": unsupported,
                    "error": entry["error"],
                    "http_status": entry.get("http_status"),
                    "http_request_id": entry.get("http_request_id"),
                })
            response_id = str(entry.get("response_id", "")).strip()
            if response_id not in response_ids:
                continue
            if response_id in matched:
                raise EvaluationError(
                    f"duplicate matched judge proxy response id: {response_id}"
                )
            if entry.get("status") != "success":
                raise EvaluationError(
                    f"matched judge proxy response {response_id} is not successful"
                )
            if (
                entry.get("requested_model") != requested_model
                or entry.get("actual_model") != actual_model
            ):
                raise EvaluationError(
                    f"matched judge proxy response {response_id} has wrong model"
                )
            matched[response_id] = entry
    missing = response_ids - set(matched)
    if missing:
        raise EvaluationError(
            f"{len(missing)} judge response ids are absent from the proxy log; "
            f"first={sorted(missing)[0]}"
        )
    matched_upstream_attempts = sum(entry["attempts"] for entry in matched.values())
    matched_unsupported = sorted({
        value for entry in matched.values()
        for value in entry["unsupported_parameters"]
    })
    matched_evidence = [{
        "response_id": response_id,
        "attempts": entry["attempts"],
        "unsupported_parameters": entry["unsupported_parameters"],
        "http_request_id": entry.get("http_request_id"),
    } for response_id, entry in sorted(matched.items())]
    return {
        "path": str(path),
        "entries": entries,
        "matched_response_ids": len(matched),
        "matched_proxy_requests": len(matched),
        "matched_proxy_upstream_attempts": matched_upstream_attempts,
        "matched_proxy_internal_retries": matched_upstream_attempts - len(matched),
        "matched_evidence_sha256": stable_hash(matched_evidence),
        "unrelated_entries_ignored": entries - len(matched),
        "shared_proxy_upstream_attempts": upstream_attempts,
        "shared_proxy_terminal_errors": terminal_errors,
        "shared_proxy_error_evidence_sha256": stable_hash(error_evidence),
        "content_hash_linkage": "unavailable_not_persisted",
        "prompt_response_content_verified": False,
        "unsupported_parameters": matched_unsupported,
        "shared_unsupported_parameters": sorted(all_unsupported),
        "requested_output_limit_enforced": (
            "max_output_tokens" not in matched_unsupported
        ),
        "requested_models": [requested_model],
        "actual_models": [actual_model],
    }


def judgment_provider_response_ids(
    judgments: dict[str, dict[str, Any]],
) -> set[str]:
    response_ids: list[str] = []
    for judgment in judgments.values():
        attempts = judgment.get("attempts")
        if not isinstance(attempts, list):
            raise EvaluationError("BEAM judgment attempts are invalid")
        for attempt in attempts:
            evidence = attempt.get("response_evidence") if isinstance(attempt, dict) else None
            response_id = evidence.get("response_id") if isinstance(evidence, dict) else None
            if response_id is not None:
                if not isinstance(response_id, str) or not response_id:
                    raise EvaluationError("BEAM provider response ID is invalid")
                response_ids.append(response_id)
    if len(response_ids) != len(set(response_ids)):
        raise EvaluationError("BEAM provider response IDs are duplicated")
    return set(response_ids)


def audit_secondary_flex_evidence(
    output_dir: Path, response_ids: set[str]
) -> dict[str, Any]:
    root = output_dir / "provider_evidence"
    if not root.is_dir() or root.is_symlink():
        raise EvaluationError("secondary Flex evidence root is missing")
    directories = sorted(root.glob("invocation-*"))
    expected_names = {
        f"invocation-{index:04d}" for index in range(1, len(directories) + 1)
    }
    if not directories or {path.name for path in root.iterdir()} != expected_names:
        raise EvaluationError("secondary Flex invocation inventory differs")
    provider_response_ids: list[str] = []
    gateway_request_ids: list[str] = []
    invocations: list[dict[str, Any]] = []
    for index, directory in enumerate(directories, start=1):
        if (
            directory.name != f"invocation-{index:04d}"
            or not directory.is_dir()
            or directory.is_symlink()
        ):
            raise EvaluationError("secondary Flex invocation directory differs")
        expected_files = {
            flex_evidence.CHILD_READY_NAME,
            flex_evidence.CHILD_LOG_NAME,
            flex_evidence.WINDOW_NAME,
            flex_evidence.INVOCATION_NAME,
        }
        if {path.name for path in directory.iterdir()} != expected_files:
            raise EvaluationError("secondary Flex invocation files differ")
        invocation_path = directory / flex_evidence.INVOCATION_NAME
        invocation = read_json(invocation_path)
        if not isinstance(invocation, dict):
            raise EvaluationError("secondary Flex invocation is invalid")
        for field, filename in (
            ("window", flex_evidence.WINDOW_NAME),
            ("consumer_log", flex_evidence.CHILD_LOG_NAME),
            ("child_ready", flex_evidence.CHILD_READY_NAME),
        ):
            binding = invocation.get(field)
            if (
                not isinstance(binding, dict)
                or Path(str(binding.get("path"))).resolve() != directory / filename
            ):
                raise EvaluationError(f"secondary Flex {field} path differs")
        try:
            report = flex_evidence.audit_invocation(invocation)
            consumers = flex_evidence.load_child_proxy_records(
                directory / flex_evidence.CHILD_LOG_NAME,
                expected_run_id=str(invocation.get("run_id")),
            )
        except flex_evidence.EvidenceError as exc:
            raise EvaluationError(str(exc)) from exc
        provider_response_ids.extend(str(row["response_id"]) for row in consumers)
        gateway_request_ids.extend(str(row["gateway_request_id"]) for row in consumers)
        invocations.append(
            {
                "path": str(invocation_path.relative_to(output_dir)),
                "sha256": sha256_file(invocation_path),
                "requests": report["requests"],
                "segment_sha256": report["segment_sha256"],
                "committed_cost_nanos": report["committed_cost_nanos"],
            }
        )
    if (
        len(provider_response_ids) != len(set(provider_response_ids))
        or set(provider_response_ids) != response_ids
    ):
        raise EvaluationError(
            "BEAM judge responses do not equal bounded Flex provider responses"
        )
    if len(gateway_request_ids) != len(set(gateway_request_ids)):
        raise EvaluationError("BEAM Flex gateway request IDs are duplicated")
    return {
        "status": "passed",
        "transport": "exclusive-child-proxy-to-openai-gpt55-flex-gateway",
        "evidence_root": str(root),
        "invocations": invocations,
        "requests": len(provider_response_ids),
        "response_ids": len(response_ids),
        "gateway_request_ids_sha256": stable_hash(gateway_request_ids),
        "committed_cost_nanos": sum(
            int(row["committed_cost_nanos"]) for row in invocations
        ),
    }


def _results_payload(
    evaluation_input: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    *,
    input_sha256: str,
    config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    results = aggregate_results(evaluation_input, judgments, config)
    results["input_sha256"] = input_sha256
    results["judge_config"] = config
    results["judge_config_hash"] = config_hash
    return results


def write_result_artifacts(output_dir: Path, results: dict[str, Any]) -> None:
    atomic_json(output_dir / "results.json", results)
    if results.get("metrics") is not None:
        atomic_json(output_dir / "metrics.json", results["metrics"])
    else:
        (output_dir / "metrics.json").unlink(missing_ok=True)


def invalidate_derived_artifacts(
    output_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    """Invalidate derived formal artifacts before a new or resumed evaluation."""
    for filename in ("results.json", "metrics.json", "evaluation_audit.json"):
        (output_dir / filename).unlink(missing_ok=True)
    for key in (
        "finished_at",
        "results_sha256",
        "metrics_sha256",
        "post_evaluation_audit",
        "post_evaluation_audit_error",
        "evaluation_audit_sha256",
        "openrouter_gateway_final",
        "openrouter_gateway_evidence",
        "openai_flex_gateway_evidence",
    ):
        state.pop(key, None)
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_json(output_dir / "run_state.json", state)
    return state


def persist_post_audit_failure(
    output_dir: Path,
    state: dict[str, Any],
    results: dict[str, Any] | None,
    exc: BaseException,
) -> None:
    """Leave no passed audit or numeric aggregate after post-audit failure."""
    error = f"{type(exc).__name__}: {exc}"[:2000]
    (output_dir / "metrics.json").unlink(missing_ok=True)
    (output_dir / "evaluation_audit.json").unlink(missing_ok=True)
    failed_results = dict(results or {})
    failed_results.update(
        {
            "schema_version": failed_results.get("schema_version", 1),
            "benchmark": "BEAM",
            "status": "failed_post_evaluation_audit",
            "metrics": None,
            "post_evaluation_audit": "failed",
            "post_evaluation_audit_error": error,
        }
    )
    atomic_json(output_dir / "results.json", failed_results)
    state.setdefault("schema_version", 1)
    state.setdefault("benchmark", "BEAM")
    state.update(
        {
            "status": "failed",
            "post_evaluation_audit": "failed",
            "post_evaluation_audit_error": error,
            "results_sha256": sha256_file(output_dir / "results.json"),
            "metrics_sha256": None,
            "updated_at": utc_now(),
        }
    )
    state.pop("evaluation_audit_sha256", None)
    atomic_json(output_dir / "run_state.json", state)


def post_evaluation_audit(
    input_path: Path, output_dir: Path, *, proxy_log: Path | None = None
) -> dict[str, Any]:
    evaluation_input = load_evaluation_input(input_path)
    input_sha256 = sha256_file(input_path)
    state_path = output_dir / "run_state.json"
    results_path = output_dir / "results.json"
    metrics_path = output_dir / "metrics.json"
    state = read_json(state_path)
    if state.get("schema_version") != 1 or state.get("benchmark") != "BEAM":
        raise EvaluationError("post-audit found invalid run state identity")
    if state.get("status") != "complete":
        raise EvaluationError("post-audit requires a complete evaluation state")
    if state.get("input_sha256") != input_sha256:
        raise EvaluationError("post-audit input hash differs from run state")
    config = state.get("config")
    if not isinstance(config, dict):
        raise EvaluationError("post-audit state has no judge configuration")
    config_hash = stable_hash(config)
    if state.get("config_hash") != config_hash:
        raise EvaluationError("post-audit judge configuration hash mismatch")
    validate_formal_config(config)
    judgments, missing = load_all_judgments(
        output_dir,
        evaluation_input,
        input_sha256=input_sha256,
        config_hash=config_hash,
        config=config,
    )
    if missing:
        raise EvaluationError(f"post-audit found {len(missing)} missing judgments")
    incomplete = [
        identifier
        for identifier, value in judgments.items()
        if value.get("status") != "complete"
    ]
    if incomplete:
        raise EvaluationError(
            f"post-audit found {len(incomplete)} incomplete judgments"
        )
    expected_files = {f"{identifier}.json" for identifier in judgments}
    judgment_dir = output_dir / "judgments"
    actual_files = (
        {path.name for path in judgment_dir.glob("*.json")}
        if judgment_dir.is_dir()
        else set()
    )
    if actual_files != expected_files:
        raise EvaluationError(
            "post-audit judgment file inventory mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_ledgers = {f"{identifier}.jsonl" for identifier in judgments}
    ledger_dir = output_dir / "attempt_ledgers"
    actual_ledgers = (
        {path.name for path in ledger_dir.glob("*.jsonl")}
        if ledger_dir.is_dir()
        else set()
    )
    if actual_ledgers != expected_ledgers:
        raise EvaluationError(
            "post-audit attempt-ledger inventory mismatch: "
            f"missing={sorted(expected_ledgers - actual_ledgers)}, "
            f"extra={sorted(actual_ledgers - expected_ledgers)}"
        )
    recomputed = _results_payload(
        evaluation_input,
        judgments,
        input_sha256=input_sha256,
        config=config,
        config_hash=config_hash,
    )
    if recomputed.get("status") != "complete" or recomputed.get("metrics") is None:
        raise EvaluationError("post-audit recomputation is incomplete")
    stored_results = read_json(results_path)
    if stored_results != recomputed:
        raise EvaluationError("stored results differ from recomputed judgments")
    if read_json(metrics_path) != recomputed["metrics"]:
        raise EvaluationError("stored metrics differ from recomputed metrics")
    if state.get("results_sha256") != sha256_file(results_path):
        raise EvaluationError("post-audit results hash mismatch")
    if state.get("metrics_sha256") != sha256_file(metrics_path):
        raise EvaluationError("post-audit metrics hash mismatch")
    if (
        state.get("completed_nuggets") != len(judgments)
        or state.get("failed_nuggets") != 0
        or state.get("missing_nuggets") != 0
    ):
        raise EvaluationError("post-audit state nugget counts are inconsistent")
    response_ids = {value["response_id"] for value in judgments.values()}
    response_models = Counter(value["response_model"] for value in judgments.values())
    proxy_report: dict[str, Any]
    if config.get("transport_contract") == "openai_gpt55_flex_gateway":
        if proxy_log is not None:
            raise EvaluationError(
                "secondary Flex post-audit does not accept a shared proxy log"
            )
        proxy_report = audit_secondary_flex_evidence(
            output_dir, judgment_provider_response_ids(judgments)
        )
        if state.get("openai_flex_gateway_evidence") != proxy_report:
            raise EvaluationError(
                "stored OpenAI Flex gateway evidence differs from post-audit"
            )
    elif config["profile"] == "secondary":
        recorded_proxy = Path(config["proxy_request_log"]).expanduser().resolve()
        if proxy_log is not None and proxy_log.expanduser().resolve() != recorded_proxy:
            raise EvaluationError(
                "post-audit proxy log differs from the frozen secondary profile path"
            )
        proxy_report = _proxy_matches(
            recorded_proxy,
            response_ids,
            requested_model=config["model"],
            actual_model=config["expected_response_model"],
        )
    else:
        if proxy_log is not None:
            raise EvaluationError("primary post-audit does not accept a proxy log")
        if config.get("transport_contract") == "marked_openrouter_gateway":
            final_binding = state.get("openrouter_gateway_final")
            if not isinstance(final_binding, dict):
                raise EvaluationError(
                    "primary post-audit lacks final OpenRouter gateway binding"
                )
            try:
                proxy_report = openrouter_gateway_evidence.verify_response_ids(
                    final_binding, response_ids
                )
            except openrouter_gateway_evidence.GatewayEvidenceError as exc:
                raise EvaluationError(str(exc)) from exc
            if state.get("openrouter_gateway_evidence") != proxy_report:
                raise EvaluationError(
                    "stored OpenRouter gateway evidence differs from post-audit"
                )
        else:
            proxy_report = {
                "status": "synthetic_injected_judge_only",
                "matched_response_ids": 0,
            }
    return {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "BEAM",
        "judge_profile": config["profile"],
        "judge_provider": config["provider"],
        "comparison_label": config["comparison_label"],
        "comparison_scope": config["comparison_scope"],
        "comparable_to_project_unified_protocol": config[
            "comparable_to_project_unified_protocol"
        ],
        "comparable_to_published_beam_official": config[
            "comparable_to_published_beam_official"
        ],
        "comparable_to_mem0": config["comparable_to_mem0"],
        "input_sha256": input_sha256,
        "config_sha256": config_hash,
        "evaluator_source_sha256": config["evaluator_source_sha256"],
        "prompt_source_sha256": config["prompt_source_sha256"],
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "questions": recomputed["question_count"],
        "rubric_nuggets": len(judgments),
        "judge_response_ids": len(response_ids),
        "judge_response_models": dict(response_models),
        "attempt_accounting": {
            "logical_judge_calls": sum(
                value["logical_judge_calls"] for value in judgments.values()
            ),
            "physical_http_attempts": sum(
                value["physical_http_attempts"] for value in judgments.values()
            ),
            "per_job_append_only_ledgers": len(actual_ledgers),
        },
        "proxy_evidence": proxy_report,
        "event_ordering": {
            "scope": "nugget-only",
            "official_tau_b_times_f1": "not_computed",
        },
    }


def resolve_profile(args: argparse.Namespace) -> argparse.Namespace:
    spec = JUDGE_PROFILES[args.profile]
    args.model = spec["model"]
    args.expected_response_model = spec["expected_response_model"]
    environment_key = (
        "BEAM_PRIMARY_JUDGE_BASE_URL"
        if args.profile == "primary"
        else "BEAM_SECONDARY_JUDGE_BASE_URL"
    )
    if args.profile == "primary":
        if args.gateway_root is not None:
            raise EvaluationError(
                "--gateway-root is valid only for secondary GPT-5.5 judging"
            )
        if args.openrouter_gateway_root is None:
            raise EvaluationError(
                "primary formal profile requires --openrouter-gateway-root"
            )
        gateway_base = args.base_url or os.environ.get(environment_key)
        if not gateway_base:
            raise EvaluationError(
                "primary formal profile requires --base-url for the marked "
                "loopback OpenRouter gateway"
            )
        try:
            args.openrouter_gateway = openrouter_gateway_evidence.capture_binding(
                args.openrouter_gateway_root, base_url=gateway_base
            )
        except openrouter_gateway_evidence.GatewayEvidenceError as exc:
            raise EvaluationError(str(exc)) from exc
        args.base_url = str(args.openrouter_gateway["base_url"])
        args.api_key = "local-openrouter-gateway"
        if args.proxy_log is not None:
            raise EvaluationError("primary profile must not use --proxy-log")
        if args.max_retries != 1:
            raise EvaluationError(
                "primary marked-gateway evaluation requires --max-retries 1"
            )
    else:
        if args.base_url or os.environ.get(environment_key):
            raise EvaluationError(
                "secondary formal profile forbids base-URL overrides; use --gateway-root"
            )
        if args.gateway_root is None:
            raise EvaluationError(
                "secondary formal profile requires --gateway-root"
            )
        if args.openrouter_gateway_root is not None:
            raise EvaluationError(
                "--openrouter-gateway-root is valid only for primary judging"
            )
        args.openrouter_gateway = None
        if args.proxy_log is not None or os.environ.get("CHATGPT_PROXY_LOG"):
            raise EvaluationError(
                "secondary formal profile forbids shared proxy logs"
            )
        args.proxy_log = None
        args.api_key = "local-openai-flex-gateway"
    return args


def _next_flex_invocation_dir(output_dir: Path) -> tuple[Path, int]:
    root = output_dir / "provider_evidence"
    root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in root.iterdir():
        match = re.fullmatch(r"invocation-(\d{4})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    return root / f"invocation-{number:04d}", number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score audited BEAM answers with rubric nuggets"
    )
    parser.add_argument("evaluation_input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", choices=tuple(JUDGE_PROFILES), required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--proxy-log", type=Path)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--openrouter-gateway-root", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.audit_only and not args.allow_model_requests:
        parser.error("formal evaluation requires --allow-model-requests")
    if args.max_tokens < 1 or args.max_retries < 1:
        parser.error("--max-tokens and --max-retries must be positive")
    lock_handle = None
    flex_invocation = None
    try:
        input_path = args.evaluation_input.expanduser().resolve()
        profile_model = JUDGE_PROFILES[args.profile]["model"]
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else (
                input_path.parent
                / (
                    f"evaluation-{args.profile}-"
                    f"{profile_model.replace('/', '_').replace('.', '_')}"
                )
            )
        )
        lock_handle = acquire_output_lock(output_dir)
        if args.audit_only:
            try:
                report = post_evaluation_audit(
                    input_path, output_dir, proxy_log=args.proxy_log
                )
                if report["judge_profile"] != args.profile:
                    raise EvaluationError("post-audit profile differs from --profile")
                audit_path = output_dir / "evaluation_audit.json"
                atomic_json(audit_path, report)
                state = read_json(output_dir / "run_state.json")
                state["post_evaluation_audit"] = "passed"
                state["evaluation_audit_sha256"] = sha256_file(audit_path)
                state["updated_at"] = utc_now()
                atomic_json(output_dir / "run_state.json", state)
            except Exception as exc:  # noqa: BLE001
                state_path = output_dir / "run_state.json"
                state: dict[str, Any] = {}
                if state_path.is_file():
                    try:
                        state_candidate = read_json(state_path)
                    except (OSError, json.JSONDecodeError):
                        state_candidate = None
                    if isinstance(state_candidate, dict):
                        state = state_candidate
                results_path = output_dir / "results.json"
                previous_results = None
                if results_path.is_file():
                    try:
                        candidate = read_json(results_path)
                    except (OSError, json.JSONDecodeError):
                        candidate = None
                    if isinstance(candidate, dict):
                        previous_results = candidate
                persist_post_audit_failure(output_dir, state, previous_results, exc)
                raise EvaluationError(
                    f"post-evaluation audit failed: {type(exc).__name__}: {exc}"
                ) from exc
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        args = resolve_profile(args)
        if (
            args.profile == "primary"
            and args.resume
            and (output_dir / "run_state.json").is_file()
        ):
            existing_state = read_json(output_dir / "run_state.json")
            stored_binding = existing_state.get("config", {}).get(
                "openrouter_gateway"
            )
            if not isinstance(stored_binding, dict):
                raise EvaluationError(
                    "resumed primary evaluation lacks its frozen gateway binding"
                )
            try:
                openrouter_gateway_evidence.validate_binding(
                    stored_binding,
                    result_root=args.openrouter_gateway_root,
                    base_url=args.base_url,
                )
            except openrouter_gateway_evidence.GatewayEvidenceError as exc:
                raise EvaluationError(str(exc)) from exc
            live_binding = args.openrouter_gateway
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
                raise EvaluationError(
                    "live OpenRouter gateway differs from the resume binding"
                )
            args.openrouter_gateway = stored_binding
        if args.profile == "secondary":
            evidence_dir, invocation_number = _next_flex_invocation_dir(output_dir)
            try:
                flex_invocation = flex_evidence.begin_child_invocation(
                    args.gateway_root,
                    evidence_dir,
                    run_id=(
                        f"beam-secondary-{invocation_number:04d}-"
                        f"{uuid.uuid4().hex[:16]}"
                    ),
                )
            except flex_evidence.EvidenceError as exc:
                raise EvaluationError(str(exc)) from exc
            args.base_url = flex_invocation.base_url
            args.flex_gateway_contract = flex_invocation.contract
            if args.resume and (output_dir / "run_state.json").is_file():
                existing_state = read_json(output_dir / "run_state.json")
                stored_flex = existing_state.get("config", {}).get(
                    "openai_flex_gateway"
                )
                if not isinstance(stored_flex, dict):
                    raise EvaluationError(
                        "resumed secondary evaluation lacks its Flex gateway binding"
                    )
                try:
                    flex_evidence.validate_recorded_contract(stored_flex)
                except flex_evidence.EvidenceError as exc:
                    raise EvaluationError(str(exc)) from exc
                stable_fields = (
                    "result_root",
                    "base_url",
                    "origin",
                    "provider_model",
                    "service_tier",
                    "max_cost_nanos",
                    "root_marker_sha256",
                    "ready_canonical_sha256",
                )
                if any(
                    stored_flex.get(field)
                    != flex_invocation.contract.get(field)
                    for field in stable_fields
                ):
                    raise EvaluationError(
                        "live Flex gateway differs from the resume binding"
                    )
                args.flex_gateway_contract = stored_flex
        else:
            args.flex_gateway_contract = None
        evaluation_input = load_evaluation_input(input_path)
        input_sha256 = sha256_file(input_path)
        config = evaluation_config(
            profile=args.profile,
            base_url=(
                str(args.flex_gateway_contract["base_url"])
                if args.flex_gateway_contract is not None
                else args.base_url
            ),
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            proxy_log=args.proxy_log,
            openrouter_gateway=args.openrouter_gateway,
            flex_gateway_contract=args.flex_gateway_contract,
            formal_transport_contract=True,
        )
        validate_formal_config(config)
        total_jobs = evaluation_input["rubric_nugget_count"]
        state = load_or_create_state(
            output_dir,
            input_path=input_path,
            input_sha256=input_sha256,
            config=config,
            total_jobs=total_jobs,
            resume=args.resume,
        )
        state = invalidate_derived_artifacts(output_dir, state)
        from openai import OpenAI

        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            http_client=httpx.Client(
                trust_env=should_trust_environment_proxy(args.base_url),
                timeout=180,
            ),
            max_retries=0,
        )
        initial_judgments, initial_missing = load_all_judgments(
            output_dir,
            evaluation_input,
            input_sha256=input_sha256,
            config_hash=state["config_hash"],
            config=config,
            recover_running=True,
        )
        state.update(
            {
                "updated_at": utc_now(),
                "completed_nuggets": sum(
                    value.get("status") == "complete"
                    for value in initial_judgments.values()
                ),
                "failed_nuggets": sum(
                    value.get("status") == "failed"
                    for value in initial_judgments.values()
                ),
                "missing_nuggets": len(initial_missing),
            }
        )
        atomic_json(output_dir / "run_state.json", state)
        for record in evaluation_input["records"]:
            for nugget_index, nugget in enumerate(record["rubric"]):
                judgment = judge_one_job(
                    client,
                    output_dir=output_dir,
                    record=record,
                    nugget_index=nugget_index,
                    nugget=nugget,
                    input_sha256=input_sha256,
                    config_hash=state["config_hash"],
                    config=config,
                )
                state.update(
                    {
                        "updated_at": utc_now(),
                        "last_job_id": judgment["job_id"],
                        "last_job_status": judgment["status"],
                    }
                )
                atomic_json(output_dir / "run_state.json", state)

        judgments, missing = load_all_judgments(
            output_dir,
            evaluation_input,
            input_sha256=input_sha256,
            config_hash=state["config_hash"],
            config=config,
        )
        results = _results_payload(
            evaluation_input,
            judgments,
            input_sha256=input_sha256,
            config=config,
            config_hash=state["config_hash"],
        )
        write_result_artifacts(output_dir, results)
        failed = [
            identifier
            for identifier, value in judgments.items()
            if value.get("status") != "complete"
        ]
        gateway_final = None
        gateway_evidence = None
        if config.get("transport_contract") == "marked_openrouter_gateway":
            try:
                gateway_final = openrouter_gateway_evidence.finalize_binding(
                    config["openrouter_gateway"]
                )
                gateway_evidence = openrouter_gateway_evidence.verify_response_ids(
                    gateway_final,
                    [
                        value["response_id"]
                        for value in judgments.values()
                        if value.get("status") == "complete"
                    ],
                )
            except openrouter_gateway_evidence.GatewayEvidenceError as exc:
                raise EvaluationError(str(exc)) from exc
        elif config.get("transport_contract") == "openai_gpt55_flex_gateway":
            if flex_invocation is None:
                raise EvaluationError("secondary Flex invocation is absent")
            try:
                flex_invocation.finish()
            except flex_evidence.EvidenceError as exc:
                raise EvaluationError(str(exc)) from exc
            flex_invocation = None
            gateway_evidence = audit_secondary_flex_evidence(
                output_dir, judgment_provider_response_ids(judgments)
            )
        state.update(
            {
                "status": "complete" if results["status"] == "complete" else "failed",
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "completed_nuggets": sum(
                    value.get("status") == "complete" for value in judgments.values()
                ),
                "failed_nuggets": len(failed),
                "missing_nuggets": len(missing),
                "results_sha256": sha256_file(output_dir / "results.json"),
                "metrics_sha256": (
                    sha256_file(output_dir / "metrics.json")
                    if (output_dir / "metrics.json").is_file()
                    else None
                ),
                "openrouter_gateway_final": gateway_final,
                "openrouter_gateway_evidence": (
                    gateway_evidence
                    if config.get("transport_contract") == "marked_openrouter_gateway"
                    else None
                ),
                "openai_flex_gateway_evidence": (
                    gateway_evidence
                    if config.get("transport_contract")
                    == "openai_gpt55_flex_gateway"
                    else None
                ),
            }
        )
        atomic_json(output_dir / "run_state.json", state)
        if state["status"] == "complete":
            try:
                audit_report = post_evaluation_audit(
                    input_path, output_dir, proxy_log=args.proxy_log
                )
                audit_path = output_dir / "evaluation_audit.json"
                atomic_json(audit_path, audit_report)
                state["post_evaluation_audit"] = "passed"
                state["evaluation_audit_sha256"] = sha256_file(audit_path)
                atomic_json(output_dir / "run_state.json", state)
            except Exception as exc:  # noqa: BLE001
                persist_post_audit_failure(output_dir, state, results, exc)
                raise EvaluationError(
                    f"post-evaluation audit failed: {type(exc).__name__}: {exc}"
                ) from exc
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "output_dir": str(output_dir),
                    "completed_nuggets": state["completed_nuggets"],
                    "failed_nuggets": state["failed_nuggets"],
                    "missing_nuggets": state["missing_nuggets"],
                    "judge_profile": config["profile"],
                    "comparison_label": config["comparison_label"],
                    "post_evaluation_audit": state.get("post_evaluation_audit"),
                    "metrics": results["metrics"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if state["status"] == "complete" else 1
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    finally:
        if flex_invocation is not None:
            try:
                flex_invocation.finish()
            except Exception:  # noqa: BLE001 - primary error remains authoritative
                flex_invocation.abort()
        release_output_lock(lock_handle)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
