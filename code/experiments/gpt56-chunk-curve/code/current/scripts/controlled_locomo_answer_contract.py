#!/usr/bin/env python3
"""Shared, strict contract for controlled LoCoMo answer generation.

The module intentionally accepts only a question string and a sequence of
``DeliveryResult`` objects at the final prompt boundary.  Dataset gold,
evidence, category, and pre-gate memory text are not valid inputs to
``assemble_answer_prompt``.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.evaluation.prompts import ANSWER_PROMPT
from src.evaluation.visible_token_budget import DeliveryResult, TokenCounter


SCHEMA_VERSION = "controlled-locomo-answer-v1"
PREREG_SCHEMA_VERSION = "controlled-answer-protocol-preregistration-v1"
LEDGER_SCHEMA_VERSION = "controlled-answer-ledger-v1"
EXPECTED_MODEL = "gpt-5.5"
EXPECTED_TIKTOKEN_VERSION = "0.12.0"
EXPECTED_ENCODING = "o200k_base"
EXPECTED_HARD_BUDGET = 20_000
EXPECTED_QUESTIONS = 1_986
EXPECTED_PRIMARY = 1_540
EXPECTED_ADVERSARIAL = 446
EXPECTED_CATEGORIES = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
CAT5_CANONICAL_ABSTENTION = "Not mentioned in the conversation"
FORMAL_METHODS = ("full_context", "bm25", "mem0", "zep")
HARD_CAP_METHODS = ("bm25", "mem0", "zep")
FULL_CONTEXT_POLICY = "full_context_unbounded_accounted"
HARD_CAP_POLICY = "hard_cap"
ZERO_HASH = "0" * 64


class ControlledAnswerError(RuntimeError):
    """Raised when a controlled-answer artifact violates the contract."""


class DuplicateJsonKeyError(ValueError):
    pass


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControlledAnswerError(f"cannot read strict JSON from {path}: {exc}") from exc


def path_identity(path: Path) -> str:
    return unicodedata.normalize(
        "NFC", os.path.normcase(os.path.realpath(os.path.abspath(path)))
    ).casefold()


def reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ControlledAnswerError(f"path contains a symlink component: {current}")


def ensure_distinct_paths(paths: Mapping[str, Path], *, reject_symlinks: bool = True) -> None:
    items = list(paths.items())
    for name, path in items:
        if reject_symlinks:
            reject_symlink_components(path)
            if path.is_symlink():
                raise ControlledAnswerError(f"{name} must not be a symlink: {path}")
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if path_identity(left) == path_identity(right):
                raise ControlledAnswerError(
                    f"path collision between {left_name} and {right_name}"
                )
            if left.exists() and right.exists() and os.path.samefile(left, right):
                raise ControlledAnswerError(
                    f"inode collision between {left_name} and {right_name}"
                )


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    """Publish JSON atomically without overwriting a file or symlink."""

    reject_symlink_components(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if published:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a mutable diagnostic state file atomically."""

    reject_symlink_components(path)
    if path.is_symlink():
        raise ControlledAnswerError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class RenderedMemory:
    input_index: int
    rendered_rank: int
    rendered_text: str


def render_memories_individually(memories: Sequence[Any]) -> list[RenderedMemory]:
    """Reproduce ``format_memories`` order, but retain one event per memory."""

    if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
        raise ControlledAnswerError("memories must be a sequence, not a scalar string")
    entries: list[dict[str, Any]] = []
    for input_index, memory in enumerate(memories):
        if isinstance(memory, str):
            text = memory
            date = ""
        elif isinstance(memory, Mapping):
            text_value = memory.get("text", memory.get("memory", ""))
            if not isinstance(text_value, str):
                raise ControlledAnswerError(
                    f"memory {input_index} text is not a string"
                )
            text = text_value
            date = str(memory.get("date", memory.get("created_at", "")) or "")
        else:
            raise ControlledAnswerError(f"memory {input_index} is not renderable")
        if not text.strip():
            raise ControlledAnswerError(f"memory {input_index} has empty text")
        entries.append({"input_index": input_index, "text": text, "date": date})
    if not entries:
        raise ControlledAnswerError("controlled formal input has no memories")
    if any(entry["date"] for entry in entries):
        entries.sort(key=lambda entry: entry["date"])
    rendered: list[RenderedMemory] = []
    for rank, entry in enumerate(entries):
        line = (
            f"({entry['date']}) {entry['text']}"
            if entry["date"]
            else entry["text"]
        )
        # The separator is part of the later memory's gate event.  Concatenating
        # delivered payloads therefore reproduces format_memories exactly while
        # ensuring no prompt-visible newline bypasses token accounting.
        text = line if rank == 0 else f"\n{line}"
        rendered.append(
            RenderedMemory(
                input_index=int(entry["input_index"]),
                rendered_rank=rank,
                rendered_text=text,
            )
        )
    return rendered


def delivered_memory_block(deliveries: Sequence[DeliveryResult]) -> str:
    """Create context exclusively from gate-returned payloads."""

    if not isinstance(deliveries, Sequence):
        raise TypeError("deliveries must be a sequence of DeliveryResult")
    texts: list[str] = []
    for index, delivery in enumerate(deliveries):
        if not isinstance(delivery, DeliveryResult):
            raise TypeError(f"delivery {index} is not a DeliveryResult")
        if delivery.delivered_text is not None:
            texts.append(delivery.delivered_text)
    return "".join(texts)


def assemble_answer_prompt(
    *, question: str, deliveries: Sequence[DeliveryResult]
) -> str:
    """Final model-boundary assembly; raw records are not accepted."""

    if not isinstance(question, str) or not question.strip():
        raise ControlledAnswerError("question must be a non-empty string")
    return ANSWER_PROMPT.format(
        memories=delivered_memory_block(deliveries),
        question=question,
    )


def extract_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "Answer:" in text:
        return text.rsplit("Answer:", 1)[-1].strip()
    return text


def formal_token_counter() -> TokenCounter:
    """Resolve and verify the preregistered local tokenizer exactly."""

    try:
        version = importlib.metadata.version("tiktoken")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ControlledAnswerError("formal answering requires tiktoken==0.12.0") from exc
    if version != EXPECTED_TIKTOKEN_VERSION:
        raise ControlledAnswerError(
            f"formal tokenizer version differs: {version} != {EXPECTED_TIKTOKEN_VERSION}"
        )
    tokenizer = TokenCounter.resolve(
        requested_model=EXPECTED_MODEL,
        fallback_encoding=EXPECTED_ENCODING,
        allow_byte_fallback=False,
    )
    expected = {
        "implementation": "tiktoken",
        "implementation_version": EXPECTED_TIKTOKEN_VERSION,
        "encoding_name": EXPECTED_ENCODING,
        "requested_model": EXPECTED_MODEL,
        "resolution": "tiktoken_named_fallback",
        "fallback_encoding": EXPECTED_ENCODING,
        "fallback_reason": "requested_model_not_in_tiktoken_mapping",
        "provider_exact": False,
        "counting_note": (
            "Local tiktoken count used for an enforceable experiment budget; "
            "it is not a provider-reported exact count."
        ),
    }
    if tokenizer.identity != expected:
        raise ControlledAnswerError(
            f"formal tokenizer identity differs: {tokenizer.identity!r}"
        )
    return tokenizer


def protocol_content_hash(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("protocol_content_sha256", None)
    return canonical_hash(content)


def validate_preregistration(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ControlledAnswerError(f"preregistration is not a regular file: {path}")
    if expected_file_sha256 is not None and sha256_file(path) != expected_file_sha256:
        raise ControlledAnswerError("preregistration file SHA-256 differs")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ControlledAnswerError("preregistration root is not an object")
    if payload.get("schema_version") != PREREG_SCHEMA_VERSION:
        raise ControlledAnswerError("preregistration schema differs")
    if payload.get("status") != "frozen":
        raise ControlledAnswerError("preregistration is not frozen")
    if payload.get("protocol_content_sha256") != protocol_content_hash(payload):
        raise ControlledAnswerError("preregistration content hash differs")
    if payload.get("formal_methods") != list(FORMAL_METHODS):
        raise ControlledAnswerError("preregistered formal method matrix differs")
    answerer = payload.get("answerer")
    if (
        not isinstance(answerer, dict)
        or answerer.get("model") != EXPECTED_MODEL
        or answerer.get("temperature") != 0
        or answerer.get("prompt_template_sha256")
        != sha256_bytes(ANSWER_PROMPT.encode("utf-8"))
        or answerer.get("shared_across_all_rows") is not True
        or answerer.get("provider_hard_output_cap_claimed") is not False
        or answerer.get("unsupported_parameter_and_actual_usage_must_be_recorded")
        is not True
        or not isinstance(answerer.get("answer_max_tokens_requested"), int)
        or isinstance(answerer.get("answer_max_tokens_requested"), bool)
        or answerer["answer_max_tokens_requested"] <= 0
    ):
        raise ControlledAnswerError("preregistered answerer differs")
    tokenizer = payload.get("tokenizer")
    if tokenizer != formal_token_counter().identity:
        raise ControlledAnswerError("preregistered tokenizer identity differs")
    gate = payload.get("visible_token_gate")
    if not isinstance(gate, dict):
        raise ControlledAnswerError("preregistered visible-token gate is missing")
    expected_gate = {
        "hard_cap_tokens": EXPECTED_HARD_BUDGET,
        "hard_cap_methods": ["bm25", "mem0", "zep", "nativemem"],
        "full_context_policy": FULL_CONTEXT_POLICY,
        "overflow_policy": "truncate",
        "rendering": "format_memories_equivalent_individual_events_v1",
        "event_order": "stable_chronological_if_any_date_else_input_order",
        "prompt_may_use": "DeliveryResult.delivered_text_only",
        "source_resolution_tokens_for_baselines": 0,
        "one_trace_and_manifest_per_question": True,
        "provider_exact": False,
    }
    if gate != expected_gate:
        raise ControlledAnswerError("preregistered visible-token gate differs")
    expected_inventory = {
        "questions": EXPECTED_QUESTIONS,
        "primary_cat1_4": EXPECTED_PRIMARY,
        "adversarial_cat5": EXPECTED_ADVERSARIAL,
        "categories": {str(key): value for key, value in EXPECTED_CATEGORIES.items()},
    }
    if payload.get("inventory") != expected_inventory:
        raise ControlledAnswerError("preregistered inventory differs")
    scoring = payload.get("scoring_labels")
    if not isinstance(scoring, dict) or scoring != {
        "source": "raw LoCoMo dataset, reconstructed by question_id",
        "categories_1_to_4": "raw_dataset.answer",
        "category_5": (
            "raw_dataset.answer when explicitly present; otherwise "
            "Not mentioned in the conversation"
        ),
        "adversarial_answer_role": "distractor_not_gold",
        "category_5_reported_separately": True,
        "baseline_input_gold_is_not_authoritative": True,
    }:
        raise ControlledAnswerError("preregistered scoring-label policy differs")
    rows = payload.get("rows")
    if not isinstance(rows, list) or [row.get("method") for row in rows] != list(
        FORMAL_METHODS
    ):
        raise ControlledAnswerError("preregistered rows differ")
    for row in rows:
        method = row["method"]
        expected_policy = FULL_CONTEXT_POLICY if method == "full_context" else HARD_CAP_POLICY
        if (
            row.get("budget_policy") != expected_policy
            or not isinstance(row.get("model_context_limit_tokens"), int)
            or isinstance(row.get("model_context_limit_tokens"), bool)
            or row["model_context_limit_tokens"] <= 0
            or row.get("answer_completion_reservation_tokens")
            != answerer["answer_max_tokens_requested"]
        ):
            raise ControlledAnswerError(f"{method} preregistered policy differs")
        if method == "full_context":
            if (
                row.get("budget_tokens") != "unbounded"
                or row.get("matched_cap_claim_allowed") is not False
                or row.get("full_context_name_requires_no_truncation") is not True
            ):
                raise ControlledAnswerError("full-context must be unbounded-accounted")
        elif (
            row.get("budget_tokens") != EXPECTED_HARD_BUDGET
            or row.get("matched_cap_claim_allowed") is not True
            or row.get("full_context_name_requires_no_truncation") is not False
        ):
            raise ControlledAnswerError(f"{method} budget differs")
    context_limits = {row["model_context_limit_tokens"] for row in rows}
    if len(context_limits) != 1:
        raise ControlledAnswerError("preregistered rows use different context limits")
    expected_future = {
        name: {
            "budget_policy": HARD_CAP_POLICY,
            "budget_tokens": EXPECTED_HARD_BUDGET,
            "status": status,
        }
        for name, status in {
            "nativemem": "requires_canonical_questions_adapter",
            "R203": "requires_condition_runner_adapter",
            "R301": "requires_normalized_evidence_bundle_adapter",
        }.items()
    }
    if payload.get("future_integrations") != expected_future:
        raise ControlledAnswerError("preregistered future-integration policy differs")
    if payload.get("claims") != {
        "legacy_6000_token_claim_retained": False,
        "hard_cap_unit": "local tiktoken o200k_base tokens",
        "full_context_is_matched_cap": False,
    }:
        raise ControlledAnswerError("preregistered claims differ")
    return payload


class FileLock:
    """Advisory exclusive lock with symlink/inode path checks."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "FileLock":
        reject_symlink_components(self.path)
        if self.path.is_symlink():
            raise ControlledAnswerError(f"lock path is a symlink: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise ControlledAnswerError(f"another answer runner holds {self.path}") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.handle is not None:
            self.handle.close()
        return False


class DurableLedger:
    """Append-only, fsynced, hash-chained JSONL ledger."""

    def __init__(self, path: Path, *, run_id: str):
        reject_symlink_components(path)
        if path.is_symlink():
            raise ControlledAnswerError(f"ledger path is a symlink: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id
        records = audit_ledger(path, expected_run_id=run_id) if path.exists() else []
        self.sequence = len(records)
        self.last_hash = records[-1]["record_hash"] if records else ZERO_HASH
        self.handle = path.open("a", encoding="utf-8")

    def append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "created_at": utc_now(),
            "event": event,
            "payload": json.loads(canonical_json(dict(payload))),
            "previous_record_hash": self.last_hash,
        }
        record["record_hash"] = canonical_hash(record)
        self.handle.write(canonical_json(record) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.sequence += 1
        self.last_hash = record["record_hash"]
        return record

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self) -> "DurableLedger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


def audit_ledger(path: Path, *, expected_run_id: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ControlledAnswerError(f"ledger is missing or not regular: {path}")
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ControlledAnswerError(f"ledger has an incomplete final line: {path}")
    previous = ZERO_HASH
    records: list[dict[str, Any]] = []
    for sequence, line in enumerate(data.decode("utf-8").splitlines()):
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {value}")
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise ControlledAnswerError(
                f"ledger line {sequence + 1} is invalid: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ControlledAnswerError(f"ledger line {sequence + 1} is not an object")
        if record.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ControlledAnswerError("ledger schema differs")
        if record.get("sequence") != sequence:
            raise ControlledAnswerError("ledger sequence is not contiguous")
        if expected_run_id is not None and record.get("run_id") != expected_run_id:
            raise ControlledAnswerError("ledger run ID differs")
        if record.get("previous_record_hash") != previous:
            raise ControlledAnswerError("ledger hash chain differs")
        stored = record.get("record_hash")
        unhashed = dict(record)
        unhashed.pop("record_hash", None)
        if stored != canonical_hash(unhashed):
            raise ControlledAnswerError("ledger record hash differs")
        previous = str(stored)
        records.append(record)
    return records


def validate_question_record(record: Mapping[str, Any]) -> None:
    required = {
        "question_id",
        "question",
        "gold",
        "category",
        "evidence",
        "memories",
        "retrieval",
    }
    if not required.issubset(record):
        raise ControlledAnswerError(
            f"question record misses fields: {sorted(required - set(record))}"
        )
    question_id = record.get("question_id")
    if not isinstance(question_id, str) or re.fullmatch(r"s\d+_q\d+", question_id) is None:
        raise ControlledAnswerError(f"invalid question_id: {question_id!r}")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise ControlledAnswerError(f"{question_id} has an invalid question")
    category = record.get("category")
    if not isinstance(category, int) or isinstance(category, bool) or category not in range(1, 6):
        raise ControlledAnswerError(f"{question_id} has an invalid category")
    if not isinstance(record.get("gold"), str):
        raise ControlledAnswerError(f"{question_id} has invalid gold")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ControlledAnswerError(f"{question_id} has invalid evidence")
    render_memories_individually(record.get("memories"))


def canonical_scoring_reference(
    dataset: Sequence[Mapping[str, Any]], question_id: str
) -> dict[str, Any]:
    """Rebuild scoring labels from raw LoCoMo, never from baseline ``gold``.

    Most category-5 rows expose only ``adversarial_answer``.  That field is a
    distractor, not a correct answer.  The canonical target is abstention unless
    the raw QA row provides an explicit ``answer``.
    """

    match = re.fullmatch(r"s(\d+)_q(\d+)", question_id)
    if match is None:
        raise ControlledAnswerError(f"invalid question ID: {question_id!r}")
    sample, question_index = (int(value) for value in match.groups())
    try:
        qa = dataset[sample]["qa"][question_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ControlledAnswerError(
            f"question ID is outside the raw dataset: {question_id}"
        ) from exc
    if not isinstance(qa, Mapping) or qa.get("question") is None:
        raise ControlledAnswerError(f"raw dataset row is invalid: {question_id}")
    category = qa.get("category")
    if not isinstance(category, int) or isinstance(category, bool):
        raise ControlledAnswerError(f"raw category is invalid: {question_id}")
    if "answer" in qa:
        canonical_gold = str(qa["answer"])
        canonical_gold_source = "raw_dataset.answer"
    elif category == 5 and "adversarial_answer" in qa:
        canonical_gold = CAT5_CANONICAL_ABSTENTION
        canonical_gold_source = "cat5_canonical_abstention"
    else:
        raise ControlledAnswerError(
            f"raw dataset lacks a canonical answer: {question_id}"
        )
    distractor = (
        str(qa["adversarial_answer"])
        if "adversarial_answer" in qa
        else None
    )
    return {
        "question_id": question_id,
        "question": str(qa["question"]),
        "category": category,
        "canonical_gold": canonical_gold,
        "canonical_gold_source": canonical_gold_source,
        "adversarial_distractor": distractor,
        "evidence": list(qa.get("evidence", [])),
    }


def prompt_source_hashes(root: Path) -> dict[str, str]:
    files = {
        "contract": Path(__file__).resolve(),
        "prompts": root / "src/evaluation/prompts.py",
        "visible_token_budget": root / "src/evaluation/visible_token_budget.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def retry_call(
    operation: Callable[[int], Any],
    *,
    attempts: int,
) -> Any:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation(attempt)
        except Exception as exc:  # noqa: BLE001
            last = exc
    assert last is not None
    raise last


__all__ = [
    "ANSWER_PROMPT",
    "CAT5_CANONICAL_ABSTENTION",
    "ControlledAnswerError",
    "DeliveryResult",
    "DurableLedger",
    "EXPECTED_ADVERSARIAL",
    "EXPECTED_CATEGORIES",
    "EXPECTED_ENCODING",
    "EXPECTED_HARD_BUDGET",
    "EXPECTED_MODEL",
    "EXPECTED_PRIMARY",
    "EXPECTED_QUESTIONS",
    "EXPECTED_TIKTOKEN_VERSION",
    "FORMAL_METHODS",
    "FULL_CONTEXT_POLICY",
    "FileLock",
    "HARD_CAP_METHODS",
    "HARD_CAP_POLICY",
    "LEDGER_SCHEMA_VERSION",
    "PREREG_SCHEMA_VERSION",
    "RenderedMemory",
    "SCHEMA_VERSION",
    "TokenCounter",
    "assemble_answer_prompt",
    "atomic_json_no_clobber",
    "atomic_json_replace",
    "audit_ledger",
    "canonical_hash",
    "canonical_json",
    "canonical_scoring_reference",
    "delivered_memory_block",
    "ensure_distinct_paths",
    "extract_answer",
    "formal_token_counter",
    "path_identity",
    "prompt_source_hashes",
    "protocol_content_hash",
    "read_json",
    "reject_symlink_components",
    "render_memories_individually",
    "sha256_bytes",
    "sha256_file",
    "utc_now",
    "validate_preregistration",
    "validate_question_record",
]
