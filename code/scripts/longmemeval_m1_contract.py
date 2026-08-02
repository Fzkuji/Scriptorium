#!/usr/bin/env python3
"""Strict contract for the LongMemEval-S M1 controlled baseline inputs.

This module has no network or model-call code.  It validates the pinned
LongMemEval-S dataset, renders one item's private history, applies the frozen
visible-token policy, and provides filesystem primitives used by the runner
and auditor.
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
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.evaluation.prompts import ANSWER_PROMPT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
)
EXPECTED_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
EXPECTED_ITEMS = 500
EXPECTED_ABSTENTION = 30
EXPECTED_TYPES = {
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}

SCHEMA_VERSION = "longmemeval-m1-input-v1"
PREREG_SCHEMA_VERSION = "longmemeval-m1-preregistration-v1"
CHECKPOINT_SCHEMA_VERSION = "longmemeval-m1-checkpoint-v1"
LEDGER_SCHEMA_VERSION = "longmemeval-m1-attempt-ledger-v1"
RUN_MANIFEST_SCHEMA_VERSION = "longmemeval-m1-run-manifest-v1"
BACKEND_PLAN_SCHEMA_VERSION = "longmemeval-m1-backend-plan-v1"

FORMAL_METHODS = ("full_context", "bm25", "mem0", "graphiti")
PREPARABLE_METHODS = ("full_context", "bm25")
PLANNED_BACKENDS = ("mem0", "graphiti")
VISIBLE_BUDGET_TOKENS = 20_000
MODEL_CONTEXT_LIMIT_TOKENS = 128_000
ANSWER_RESERVATION_TOKENS = 4_096
MAX_RENDERED_PROMPT_TOKENS = (
    MODEL_CONTEXT_LIMIT_TOKENS - ANSWER_RESERVATION_TOKENS
)
EXPECTED_TIKTOKEN_VERSION = "0.12.0"
EXPECTED_ENCODING = "o200k_base"
EXPECTED_MODEL = "gpt-5.5"
BM25_VERSION = "0.2.2"
MEM0_VERSION = "2.0.10"
QDRANT_VERSION = "1.18.0"
GRAPHITI_VERSION = "0.29.2"
KUZU_VERSION = "0.11.3"
SENTENCE_TRANSFORMERS_VERSION = "5.6.0"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ZERO_HASH = "0" * 64


class ContractError(RuntimeError):
    """Raised when an M1 input or execution environment violates the contract."""


class DuplicateJsonKeyError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


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
        raise ContractError(f"cannot read strict JSON from {path}: {exc}") from exc


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
            raise ContractError(f"path contains a symlink component: {current}")


def ensure_distinct_paths(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    for name, path in items:
        reject_symlink_components(path)
        if path.is_symlink():
            raise ContractError(f"{name} must not be a symlink: {path}")
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if path_identity(left) == path_identity(right):
                raise ContractError(
                    f"path collision between {left_name} and {right_name}"
                )
            if left.exists() and right.exists() and os.path.samefile(left, right):
                raise ContractError(
                    f"inode collision between {left_name} and {right_name}"
                )


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
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
    reject_symlink_components(path)
    if path.is_symlink():
        raise ContractError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl_fsync(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    reject_symlink_components(path)
    if path.is_symlink():
        raise ContractError(f"attempt ledger must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("schema_version", LEDGER_SCHEMA_VERSION)
    payload.setdefault("event_id", str(uuid.uuid4()))
    payload.setdefault("created_at", utc_now())
    line = canonical_json(payload) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink():
        raise ContractError(f"attempt ledger must not be a symlink: {path}")
    events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read attempt ledger {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ContractError(f"attempt ledger line {line_number} is blank")
        try:
            event = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, DuplicateJsonKeyError) as exc:
            raise ContractError(
                f"attempt ledger line {line_number} is invalid JSON"
            ) from exc
        event_id = event.get("event_id") if isinstance(event, dict) else None
        if (
            not isinstance(event, dict)
            or event.get("schema_version") != LEDGER_SCHEMA_VERSION
            or not isinstance(event_id, str)
            or not event_id
            or event_id in event_ids
        ):
            raise ContractError(
                f"attempt ledger line {line_number} has invalid identity"
            )
        event_ids.add(event_id)
        events.append(event)
    return events


@contextmanager
def advisory_lock(path: Path):
    reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ContractError(f"lock path must not be a symlink: {path}")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"another process holds lock {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("utf-8"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class FormalTokenizer:
    identity: dict[str, Any]
    _encoding: Any

    @classmethod
    def resolve(cls) -> "FormalTokenizer":
        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ContractError("formal M1 inputs require tiktoken==0.12.0") from exc
        try:
            version = importlib.metadata.version("tiktoken")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ContractError("tiktoken distribution metadata is missing") from exc
        if version != EXPECTED_TIKTOKEN_VERSION:
            raise ContractError(
                f"tiktoken version differs: {version} != {EXPECTED_TIKTOKEN_VERSION}"
            )
        encoding = tiktoken.get_encoding(EXPECTED_ENCODING)
        identity = {
            "implementation": "tiktoken",
            "implementation_version": EXPECTED_TIKTOKEN_VERSION,
            "encoding_name": EXPECTED_ENCODING,
            "requested_model": EXPECTED_MODEL,
            "resolution": "explicit_named_encoding",
            "disallowed_special": [],
            "special_token_policy": "encode_literal_special_strings_as_ordinary_text",
            "provider_exact": False,
        }
        return cls(identity=identity, _encoding=encoding)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("tokenized value must be a string")
        return list(self._encoding.encode(text, disallowed_special=()))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, limit: int) -> str:
        if limit < 0:
            raise ValueError("token limit must be nonnegative")
        token_ids = self.encode(text)
        if len(token_ids) <= limit:
            return text
        prefix = token_ids[:limit]
        while prefix:
            candidate = self._encoding.decode(prefix)
            if text.startswith(candidate) and self.count(candidate) <= limit:
                return candidate
            prefix.pop()
        return ""


def package_version(distribution: str, expected: str) -> str:
    try:
        observed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError(f"required distribution is not installed: {distribution}") from exc
    if observed != expected:
        raise ContractError(
            f"{distribution} version differs: {observed} != {expected}"
        )
    return observed


def strict_dependency_snapshot() -> dict[str, str]:
    return {
        "tiktoken": package_version("tiktoken", EXPECTED_TIKTOKEN_VERSION),
        "rank-bm25": package_version("rank-bm25", BM25_VERSION),
        "mem0ai": package_version("mem0ai", MEM0_VERSION),
        "qdrant-client": package_version("qdrant-client", QDRANT_VERSION),
        "graphiti-core": package_version("graphiti-core", GRAPHITI_VERSION),
        "kuzu": package_version("kuzu", KUZU_VERSION),
        "sentence-transformers": package_version(
            "sentence-transformers", SENTENCE_TRANSFORMERS_VERSION
        ),
    }


def method_configuration(method: str) -> dict[str, Any]:
    common = {
        "history_isolation": "one_dataset_item_one_private_history",
        "source_mapping": {
            "requirement": "R002",
            "granularity": "session",
            "turn_recall_claimed": False,
        },
        "answerer": {
            "status": "deferred_shared_protocol_integration",
            "requested_model": EXPECTED_MODEL,
            "model_context_limit_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": ANSWER_RESERVATION_TOKENS,
            "max_rendered_prompt_tokens": MAX_RENDERED_PROMPT_TOKENS,
        },
    }
    if method == "full_context":
        return {
            **common,
            "public_name": "full_context",
            "implementation": "all LongMemEval haystack sessions in dataset order",
            "budget_policy": "full_context_unbounded_accounted",
            "truncation_allowed": False,
            "dependencies": {"tiktoken": EXPECTED_TIKTOKEN_VERSION},
        }
    if method == "bm25":
        return {
            **common,
            "public_name": "BM25",
            "implementation": "rank_bm25.BM25Okapi over one document per session",
            "tokenization": "lowercase_ascii_alphanumeric_v1",
            "ranking_tie_break": "dataset_session_order",
            "budget_policy": "hard_visible_total",
            "visible_budget_tokens": VISIBLE_BUDGET_TOKENS,
            "overflow_policy": "truncate_current_then_stop",
            "dependencies": {
                "tiktoken": EXPECTED_TIKTOKEN_VERSION,
                "rank-bm25": BM25_VERSION,
            },
        }
    if method == "mem0":
        return {
            **common,
            "public_name": "Mem0 OSS",
            "implementation": "mem0ai OSS with local embedded Qdrant",
            "scope_interfaces": ["formal_500", "smoke_exactly_1"],
            "budget_policy": "hard_visible_total",
            "visible_budget_tokens": VISIBLE_BUDGET_TOKENS,
            "dependencies": {
                "mem0ai": MEM0_VERSION,
                "qdrant-client": QDRANT_VERSION,
                "sentence-transformers": SENTENCE_TRANSFORMERS_VERSION,
            },
            "vector_store": {
                "provider": "qdrant",
                "mode": "local_embedded_path_per_item",
                "collection_name_template": "lme_m1_{dataset_index}_{question_id}",
            },
            "embedder": {
                "provider": "huggingface_local",
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMS,
            },
            "builder_llm": {
                "provider": "openai_compatible",
                "model": EXPECTED_MODEL,
                "endpoint_binding": "deferred_to_shared_controlled_gpt55_protocol",
                "temperature": 0,
            },
        }
    if method == "graphiti":
        return {
            **common,
            "public_name": "Graphiti OSS",
            "implementation": "graphiti-core OSS with embedded Kuzu",
            "claim_boundary": "Graphiti OSS core, not the commercial Zep platform",
            "scope_interfaces": ["formal_500", "smoke_exactly_1"],
            "budget_policy": "hard_visible_total",
            "visible_budget_tokens": VISIBLE_BUDGET_TOKENS,
            "dependencies": {
                "graphiti-core": GRAPHITI_VERSION,
                "kuzu": KUZU_VERSION,
                "sentence-transformers": SENTENCE_TRANSFORMERS_VERSION,
            },
            "graph_store": {
                "provider": "kuzu",
                "mode": "local_embedded_path_per_item",
            },
            "embedder": {
                "provider": "local_shim",
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMS,
            },
            "builder_llm": {
                "client": "OpenAIGenericClient",
                "model": EXPECTED_MODEL,
                "small_model": EXPECTED_MODEL,
                "structured_output_mode": "json_object",
                "endpoint_binding": "deferred_to_shared_controlled_gpt55_protocol",
                "temperature": 0,
            },
            "retrieval_surface": "Graphiti.search entity-edge facts",
        }
    raise ContractError(f"unsupported formal method: {method}")


def _validate_turn(turn: object, *, item_index: int, session_index: int, turn_index: int) -> None:
    if not isinstance(turn, dict):
        raise ContractError(
            f"item {item_index} session {session_index} turn {turn_index} is not an object"
        )
    if not isinstance(turn.get("role"), str) or not turn["role"].strip():
        raise ContractError(
            f"item {item_index} session {session_index} turn {turn_index} lacks role"
        )
    if not isinstance(turn.get("content"), str):
        raise ContractError(
            f"item {item_index} session {session_index} turn {turn_index} lacks content"
        )


def validate_dataset(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if sha256_file(path) != EXPECTED_DATASET_SHA256:
        raise ContractError("LongMemEval-S dataset SHA-256 differs from the pinned file")
    payload = read_json(path)
    if not isinstance(payload, list) or len(payload) != EXPECTED_ITEMS:
        raise ContractError("LongMemEval-S must contain exactly 500 items")
    question_ids: set[str] = set()
    history_hashes: set[str] = set()
    abstention = 0
    types: Counter[str] = Counter()
    for item_index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ContractError(f"dataset item {item_index} is not an object")
        required = {
            "question_id",
            "question_type",
            "question",
            "question_date",
            "answer",
            "answer_session_ids",
            "haystack_sessions",
            "haystack_dates",
            "haystack_session_ids",
        }
        if set(item) != required:
            raise ContractError(
                f"dataset item {item_index} fields differ: {sorted(set(item) ^ required)}"
            )
        question_id = item["question_id"]
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in question_ids
        ):
            raise ContractError(f"dataset item {item_index} has invalid question id")
        question_ids.add(question_id)
        if question_id.endswith("_abs"):
            abstention += 1
        question_type = item["question_type"]
        if question_type not in EXPECTED_TYPES:
            raise ContractError(f"dataset item {item_index} has unknown question type")
        types[question_type] += 1
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ContractError(f"dataset item {item_index} has empty question")
        sessions = item["haystack_sessions"]
        dates = item["haystack_dates"]
        session_ids = item["haystack_session_ids"]
        if (
            not isinstance(sessions, list)
            or not sessions
            or not isinstance(dates, list)
            or not isinstance(session_ids, list)
            or len(sessions) != len(dates)
            or len(sessions) != len(session_ids)
        ):
            raise ContractError(f"dataset item {item_index} session arrays differ")
        if any(not isinstance(value, str) or not value for value in dates):
            raise ContractError(f"dataset item {item_index} has invalid session dates")
        if (
            any(not isinstance(value, str) or not value for value in session_ids)
        ):
            raise ContractError(f"dataset item {item_index} has invalid session ids")
        for session_index, session in enumerate(sessions):
            if not isinstance(session, list) or not session:
                raise ContractError(
                    f"dataset item {item_index} session {session_index} is empty"
                )
            for turn_index, turn in enumerate(session):
                _validate_turn(
                    turn,
                    item_index=item_index,
                    session_index=session_index,
                    turn_index=turn_index,
                )
        references = item["answer_session_ids"]
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value for value in references)
            or len(set(references)) != len(references)
            or not set(references).issubset(session_ids)
        ):
            raise ContractError(
                f"dataset item {item_index} has invalid answer session ids"
            )
        history_hash = history_content_hash(item)
        if history_hash in history_hashes:
            raise ContractError("LongMemEval-S items do not have unique histories")
        history_hashes.add(history_hash)
    if abstention != EXPECTED_ABSTENTION:
        raise ContractError(f"LongMemEval-S abstention count is {abstention}, expected 30")
    if dict(types) != EXPECTED_TYPES:
        raise ContractError(f"LongMemEval-S type counts differ: {dict(types)}")
    return payload


def history_content_hash(item: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "haystack_session_ids": item["haystack_session_ids"],
            "haystack_dates": item["haystack_dates"],
            "haystack_sessions": item["haystack_sessions"],
        }
    )


def history_owner_hash(item: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "question_id": item["question_id"],
            "history_content_sha256": history_content_hash(item),
        }
    )


def item_directory_name(index: int, question_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", question_id):
        raise ContractError(f"question id is unsafe for a path: {question_id!r}")
    return f"{index:04d}_{question_id}"


def render_session(
    *,
    session_index: int,
    session_id: str,
    date: str,
    turns: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"[Session {session_index + 1} | session_id={session_id} | date={date}]"
    ]
    lines.extend(f"{turn['role']}: {turn['content']}" for turn in turns)
    return "\n".join(lines)


def rendered_sessions(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (session_id, date, turns) in enumerate(
        zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
        )
    ):
        text = render_session(
            session_index=index,
            session_id=session_id,
            date=date,
            turns=turns,
        )
        result.append(
            {
                "dataset_session_index": index,
                "source_session_id": session_id,
                "date": date,
                "text": text,
                "text_sha256": sha256_text(text),
            }
        )
    return result


def lexical_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def rank_bm25_sessions(
    item: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> list[int]:
    package_version("rank-bm25", BM25_VERSION)
    from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

    corpus = [lexical_tokens(str(session["text"])) for session in sessions]
    index = BM25Okapi(corpus)
    scores = index.get_scores(lexical_tokens(str(item["question"])))
    return sorted(range(len(sessions)), key=lambda value: (-float(scores[value]), value))


def _event_payload(
    *,
    rank: int,
    session: Mapping[str, Any],
    raw_text: str,
    delivered_text: str,
    raw_tokens: int,
    delivered_tokens: int,
    decision: str,
    cumulative: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "dataset_session_index": session["dataset_session_index"],
        "source_session_id": session["source_session_id"],
        "source_session_occurrence_key": (
            f"{int(session['dataset_session_index']):04d}:"
            f"{session['source_session_id']}"
        ),
        "source_mapping_granularity": "session",
        "turn_recall_claimed": False,
        "source_document_sha256": session["text_sha256"],
        "raw_text_sha256": sha256_text(raw_text),
        "raw_utf8_bytes": len(raw_text.encode("utf-8")),
        "raw_tokens": raw_tokens,
        "delivered_text_sha256": sha256_text(delivered_text),
        "delivered_utf8_bytes": len(delivered_text.encode("utf-8")),
        "delivered_tokens": delivered_tokens,
        "decision": decision,
        "cumulative_visible_tokens": cumulative,
    }


def _source_mapping(
    item: Mapping[str, Any], source_session_ids: Sequence[str]
) -> dict[str, Any]:
    references = list(item["answer_session_ids"])
    observed = list(dict.fromkeys(source_session_ids))
    matched = [value for value in references if value in observed]
    return {
        "requirement": "R002",
        "granularity": "session",
        "turn_recall_claimed": False,
        "reference_answer_session_ids": references,
        "delivered_source_session_ids": observed,
        "matched_answer_session_ids": matched,
        "session_recall": len(matched) / len(references),
    }


def _base_answer_input(
    *,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    tokenizer: FormalTokenizer,
    context: str,
    context_events: list[dict[str, Any]],
    budget: dict[str, Any],
) -> dict[str, Any]:
    prompt = ANSWER_PROMPT.format(memories=context, question=item["question"])
    prompt_tokens = tokenizer.count(prompt)
    retokenized_context_tokens = tokenizer.count(context)
    accounted_visible_tokens = budget["cumulative_visible_tokens"]
    if prompt_tokens > MAX_RENDERED_PROMPT_TOKENS:
        raise ContractError(
            f"item {item_index} rendered prompt exceeds declared input limit: "
            f"{prompt_tokens} > {MAX_RENDERED_PROMPT_TOKENS}"
        )
    source_ids = [event["source_session_id"] for event in context_events]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "LongMemEval-S",
        "method": method,
        "dataset_index": item_index,
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "question_date": item["question_date"],
        "abstention": str(item["question_id"]).endswith("_abs"),
        "history_owner_question_id": item["question_id"],
        "history_content_sha256": history_content_hash(item),
        "history_owner_sha256": history_owner_hash(item),
        "context": {
            "rendering": "session_header_role_content_v1",
            "text": context,
            "text_sha256": sha256_text(context),
            "utf8_bytes": len(context.encode("utf-8")),
            "visible_tokens": accounted_visible_tokens,
            "retokenized_context_tokens": retokenized_context_tokens,
            "source_session_ids": source_ids,
            "events": context_events,
            "budget": budget,
        },
        "answer_protocol_interface": {
            "status": "reserved_not_executed",
            "requested_model": EXPECTED_MODEL,
            "prompt_template": "scripts.evaluation.prompts.ANSWER_PROMPT",
            "prompt_template_sha256": sha256_text(ANSWER_PROMPT),
            "rendered_prompt_sha256": sha256_text(prompt),
            "rendered_prompt_tokens": prompt_tokens,
            "model_context_limit_tokens": MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": ANSWER_RESERVATION_TOKENS,
            "max_rendered_prompt_tokens": MAX_RENDERED_PROMPT_TOKENS,
            "model_calls": 0,
        },
    }


def prepare_full_context_item(
    item: Mapping[str, Any], item_index: int, tokenizer: FormalTokenizer
) -> tuple[dict[str, Any], dict[str, Any]]:
    sessions = rendered_sessions(item)
    pieces: list[str] = []
    events: list[dict[str, Any]] = []
    cumulative = 0
    for rank, session in enumerate(sessions):
        raw_text = str(session["text"]) if rank == 0 else f"\n\n{session['text']}"
        tokens = tokenizer.count(raw_text)
        cumulative += tokens
        pieces.append(raw_text)
        events.append(
            _event_payload(
                rank=rank,
                session=session,
                raw_text=raw_text,
                delivered_text=raw_text,
                raw_tokens=tokens,
                delivered_tokens=tokens,
                decision="delivered_full",
                cumulative=cumulative,
            )
        )
    context = "".join(pieces)
    if [event["source_session_id"] for event in events] != list(
        item["haystack_session_ids"]
    ):
        raise ContractError(f"item {item_index} full context omitted a session")
    # Full-context has one final prompt-boundary value.  Retokenizing the
    # concatenation avoids overstating it due to independent BPE boundaries
    # between per-session audit events.
    cumulative = tokenizer.count(context)
    answer_input = _base_answer_input(
        method="full_context",
        item=item,
        item_index=item_index,
        tokenizer=tokenizer,
        context=context,
        context_events=events,
        budget={
            "policy": "full_context_unbounded_accounted",
            "configured_visible_budget_tokens": None,
            "truncation_allowed": False,
            "truncated_events": 0,
            "cumulative_visible_tokens": cumulative,
        },
    )
    checkpoint_private = {
        "source_mapping": _source_mapping(item, item["haystack_session_ids"]),
        "raw_session_count": len(sessions),
        "delivered_session_count": len(sessions),
    }
    return answer_input, checkpoint_private


def prepare_bm25_item(
    item: Mapping[str, Any], item_index: int, tokenizer: FormalTokenizer
) -> tuple[dict[str, Any], dict[str, Any]]:
    sessions = rendered_sessions(item)
    ranking = rank_bm25_sessions(item, sessions)
    events: list[dict[str, Any]] = []
    pieces: list[str] = []
    cumulative = 0
    truncated = 0
    for rank, session_index in enumerate(ranking):
        session = sessions[session_index]
        raw_text = str(session["text"]) if not pieces else f"\n\n{session['text']}"
        raw_tokens = tokenizer.count(raw_text)
        remaining = VISIBLE_BUDGET_TOKENS - cumulative
        if remaining <= 0:
            break
        delivered_text = tokenizer.truncate(raw_text, remaining)
        delivered_tokens = tokenizer.count(delivered_text)
        if not delivered_text or delivered_tokens <= 0:
            break
        # A session is counted as delivered for R002 only when its stable
        # session identifier is itself visible in the delivered prefix.
        if f"session_id={session['source_session_id']}" not in delivered_text:
            break
        decision = (
            "delivered_full" if delivered_text == raw_text else "delivered_truncated"
        )
        if decision == "delivered_truncated":
            truncated += 1
        cumulative += delivered_tokens
        pieces.append(delivered_text)
        events.append(
            _event_payload(
                rank=rank,
                session=session,
                raw_text=raw_text,
                delivered_text=delivered_text,
                raw_tokens=raw_tokens,
                delivered_tokens=delivered_tokens,
                decision=decision,
                cumulative=cumulative,
            )
        )
        if decision == "delivered_truncated":
            break
    context = "".join(pieces)
    if cumulative > VISIBLE_BUDGET_TOKENS:
        raise ContractError(f"item {item_index} exceeds the 20K visible-token gate")
    source_ids = [event["source_session_id"] for event in events]
    answer_input = _base_answer_input(
        method="bm25",
        item=item,
        item_index=item_index,
        tokenizer=tokenizer,
        context=context,
        context_events=events,
        budget={
            "policy": "hard_visible_total",
            "configured_visible_budget_tokens": VISIBLE_BUDGET_TOKENS,
            "overflow_policy": "truncate_current_then_stop",
            "truncated_events": truncated,
            "cumulative_visible_tokens": cumulative,
            "exhausted": cumulative == VISIBLE_BUDGET_TOKENS,
        },
    )
    checkpoint_private = {
        "source_mapping": _source_mapping(item, source_ids),
        "raw_session_count": len(sessions),
        "ranked_session_ids": [sessions[index]["source_session_id"] for index in ranking],
        "delivered_session_count": len(source_ids),
    }
    return answer_input, checkpoint_private


def prepare_item(
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    tokenizer: FormalTokenizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method == "full_context":
        return prepare_full_context_item(item, item_index, tokenizer)
    if method == "bm25":
        return prepare_bm25_item(item, item_index, tokenizer)
    raise ContractError(f"method {method!r} is planned but not prepared without models")


def validate_answer_input_allowlist(payload: Mapping[str, Any]) -> None:
    prohibited = {
        "answer",
        "gold",
        "answer_session_ids",
        "reference_answer_session_ids",
        "matched_answer_session_ids",
        "hypothesis",
        "judge",
    }

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in prohibited:
                    raise ContractError(f"answer input contains prohibited field {path}{key}")
                walk(nested, f"{path}{key}.")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{path}{index}.")

    walk(payload, "")


def validate_item_ledger(
    events: Sequence[Mapping[str, Any]],
    *,
    method: str,
    item_index: int,
    question_id: str,
    require_complete: bool,
) -> dict[str, Any] | None:
    starts: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if (
            event.get("method") != method
            or event.get("dataset_index") != item_index
            or event.get("question_id") != question_id
        ):
            raise ContractError(f"item {item_index} attempt ledger identity mismatch")
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ContractError(f"item {item_index} attempt ledger lacks attempt id")
        event_type = event.get("event")
        if event_type == "attempt_started":
            if attempt_id in starts:
                raise ContractError(f"item {item_index} has duplicate attempt start")
            starts[attempt_id] = event
        elif event_type == "attempt_completed":
            if attempt_id in terminals:
                raise ContractError(f"item {item_index} has duplicate attempt terminal")
            terminals[attempt_id] = event
        else:
            raise ContractError(f"item {item_index} has unknown ledger event {event_type!r}")
    orphan = set(starts) - set(terminals)
    if orphan:
        raise ContractError(
            f"item {item_index} has unresolved attempt {sorted(orphan)[0]}; refusing resume"
        )
    if set(terminals) - set(starts):
        raise ContractError(f"item {item_index} terminal lacks a prior start")
    if len(terminals) > 1:
        raise ContractError(f"item {item_index} has more than one completed attempt")
    if require_complete and len(terminals) != 1:
        raise ContractError(f"item {item_index} lacks one completed attempt")
    return dict(next(iter(terminals.values()))) if terminals else None


def backend_plan_indices(scope: str, item_index: int | None) -> list[int]:
    if scope == "formal":
        if item_index is not None:
            raise ContractError("formal backend plan must cover all 500 items")
        return list(range(EXPECTED_ITEMS))
    if scope == "smoke":
        if item_index is None or not 0 <= item_index < EXPECTED_ITEMS:
            raise ContractError("smoke backend plan requires one valid --item-index")
        return [item_index]
    raise ContractError("backend plan scope must be formal or smoke")


def backend_workspace_descriptor(
    *, method: str, output_root: Path, item_index: int, question_id: str
) -> dict[str, Any]:
    item_name = item_directory_name(item_index, question_id)
    workspace = output_root / method / "items" / item_name / "backend"
    if method == "mem0":
        return {
            "workspace": str(workspace),
            "history_db": str(workspace / "history.db"),
            "qdrant_path": str(workspace / "qdrant"),
            "collection_name": f"lme_m1_{item_index}_{question_id}",
            "history_owner_question_id": question_id,
        }
    if method == "graphiti":
        return {
            "workspace": str(workspace),
            "kuzu_path": str(workspace / "kuzu"),
            "group_id": f"lme_m1_{item_index}_{question_id}",
            "history_owner_question_id": question_id,
        }
    raise ContractError(f"unsupported planned backend: {method}")


def regular_tree_inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"artifact root is not a regular directory: {root}")
    inventory: list[dict[str, Any]] = []
    inodes: dict[tuple[int, int], str] = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ContractError(f"artifact tree contains symlink: {relative}")
        stat = path.stat(follow_symlinks=False)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"artifact tree contains special node: {relative}")
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes:
            raise ContractError(
                f"artifact tree contains hardlink alias: {inodes[inode]} and {relative}"
            )
        inodes[inode] = relative
        inventory.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": stat.st_size,
            }
        )
    return inventory


def inventory_root(entries: Iterable[Mapping[str, Any]]) -> str:
    return canonical_hash(
        [
            {
                "path": entry["path"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
            for entry in entries
        ]
    )
