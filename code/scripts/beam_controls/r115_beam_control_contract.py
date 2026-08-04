#!/usr/bin/env python3
"""Frozen, network-free contract for the R115 BEAM controlled rows.

This module contains constants, canonical serialization, strict artifact I/O,
and the task-specific prompt.  It has no provider client and performs no
network access on import.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
HF_DATASET = "Mohammadta/BEAM"
HF_CONFIG = "default"
HF_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
HF_CACHE = ROOT / "benchmarks/beam/hf_cache"
HF_REVISION_CACHE = (
    HF_CACHE
    / "Mohammadta___beam/default/0.0.0/3205395e897e7318c7b094ef4e6047b9b82dbb03"
)
ARROW_FILES = {
    "100K": HF_REVISION_CACHE / "beam-100K.arrow",
    "1M": HF_REVISION_CACHE / "beam-1M.arrow",
}
ARROW_SHA256 = {
    "100K": "7b78964d628b866242dcb5df03bb6e3140d60cc8b366c7c87cbf43cc708007d0",
    "1M": "3d9029d389cd589135fd0c0d7ad09d9160532d35e1f240f2d45136ccef600c95",
}
ARROW_SCHEMA_SHA256 = (
    "bb8ec631757e93aa0dea1312d89449206c53381922fb2b854c5899a68e6ea0d4"
)
DATASET_INFO_PATH = HF_REVISION_CACHE / "dataset_info.json"
DATASET_INFO_SHA256 = (
    "0e499c10085f663e2fcaf007d8f123b07d3f3b31d7a15652dbeb9668d83eca07"
)
EXPECTED_SPLIT_ROWS = {"100K": 20, "1M": 35}
FORMAL_SELECTION = {"100K": list(range(10)), "1M": list(range(35))}
FORMAL_CONVERSATIONS = 45
QUESTIONS_PER_CONVERSATION = 20
FORMAL_QUESTIONS = 900
FORMAL_METHODS = ("full_context", "bm25", "mem0")

QUESTION_TYPES = (
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
GOLD_FIELD_BY_TYPE = {
    "abstention": "ideal_response",
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "information_extraction": "answer",
    "instruction_following": "expected_compliance",
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "preference_following": "expected_compliance",
    "summarization": "ideal_summary",
    "temporal_reasoning": "answer",
}

REQUESTED_MODEL = "gpt-5.5"
ACTUAL_MODEL_PATTERN = r"^gpt-5\.5(?:-2026-04-23)?$"
SYNTHETIC_BASE_URL = "http://127.0.0.1:1/v1"
MODEL_CONTEXT_LIMIT_TOKENS = 1_050_000
ANSWER_MAX_TOKENS = 4_096
MAX_RENDERED_PROMPT_TOKENS = MODEL_CONTEXT_LIMIT_TOKENS - ANSWER_MAX_TOKENS
VISIBLE_BUDGET_TOKENS = 20_000
TIKTOKEN_VERSION = "0.12.0"
TOKEN_ENCODING = "o200k_base"

BM25_TOP_K = 40
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
BM25_INDEX_VERSION = "r115-bm25-frozen-inverted-index-v1"
MEM0_TOP_K = 40
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
MEM0_VERSION = "2.0.10"
QDRANT_VERSION = "1.18.0"

SCHEMA = "r115-beam-controls/v1"
PREREG_SCHEMA = "r115-beam-controls-preregistration/v1"
CHECKPOINT_SCHEMA = "r115-beam-controls-checkpoint/v1"
INVENTORY_SCHEMA = "r115-beam-controls-inventory/v1"
QUESTION_SCHEMA = "r115-beam-controls-question/v1"
AUDIT_SCHEMA = "r115-beam-controls-audit/v1"
PREFLIGHT_SCHEMA = "r115-beam-controls-preflight/v1"
PREFLIGHT_AUDIT_SCHEMA = "r115-beam-controls-preflight-audit/v1"
RENDER_VERSION = "r115-beam-render-v1"
BM25_VERSION = "r115-bm25-turn-v1"
MEM0_SURFACE_VERSION = "r115-mem0-session-v1"
SOURCE_POLICY_VERSION = "r002-beam-turn-map-v1"
FULL_CONTEXT_POLICY = "complete-render-or-block-v1"
EXACT_PROMPT_COUNT_VERSION = "tiktoken-encode-with-unstable-prefix-v1"
BLOCKED_STATUS = "blocked_over_context"

EVIDENCE_MAPPING_ROOT = (
    ROOT / "results/paper-experiments-20260714/evidence-mapping/v1"
)
EVIDENCE_MAPPING_MANIFEST = (
    EVIDENCE_MAPPING_ROOT / "evidence_mapping.v1.manifest.json"
)
EVIDENCE_MAPPING_AUDIT = EVIDENCE_MAPPING_ROOT / "evidence_mapping.v1.audit.json"
EVIDENCE_MAPPING_QUESTIONS = (
    EVIDENCE_MAPPING_ROOT / "evidence_mapping.v1.questions.jsonl"
)

ANSWER_PROMPT = """You answer one BEAM benchmark question using only the supplied conversation evidence.

Instructions:
1. Use the timestamps and source identifiers in the evidence when the question
   requires ordering, updates, contradictions, or relative-time resolution.
2. Follow every format, preference, and instruction requested by the question.
3. Include every requested part. Do not impose a short-answer limit.
4. If the supplied evidence does not contain the answer, respond exactly:
   I don't have enough information to answer this question.
5. Put only the final response inside <answer></answer> tags.

Evidence:
{memories}

Question:
{question}

Answer:"""


class ContractError(RuntimeError):
    """Raised when an R115 input or artifact violates the frozen contract."""


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
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read strict JSON from {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ContractError(
                        f"JSONL has an incomplete line at {line_number}: {path}"
                    )
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=lambda raw: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON value: {raw}")
                    ),
                )
                if not isinstance(value, dict):
                    raise ContractError(
                        f"JSONL line {line_number} is not an object: {path}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"cannot read strict JSONL from {path}: {exc}") from exc
    return rows


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
            raise ContractError(f"path contains a symbolic-link component: {current}")


def ensure_distinct_paths(paths: Mapping[str, Path]) -> None:
    values = list(paths.items())
    for name, path in values:
        reject_symlink_components(path)
        if path.is_symlink():
            raise ContractError(f"{name} is a symbolic link: {path}")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if path_identity(left) == path_identity(right):
                raise ContractError(f"path collision: {left_name}, {right_name}")
            if left.exists() and right.exists() and os.path.samefile(left, right):
                raise ContractError(f"inode collision: {left_name}, {right_name}")


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    reject_symlink_components(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    reject_symlink_components(path)
    if path.is_symlink():
        raise ContractError(f"refusing to replace symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
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
        temporary.unlink(missing_ok=True)


def content_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return canonical_hash(payload)


def normalize_time_anchor(raw: Any) -> str:
    if raw is None or not str(raw).strip():
        raise ContractError("BEAM turn lacks a time anchor")
    text = str(raw).strip()
    iso = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if iso:
        year, month, day = map(int, iso.groups())
        return datetime(year, month, day).date().isoformat()
    named = re.search(
        r"\b([A-Za-z]+)[\s\-/]+(\d{1,2})(?:st|nd|rd|th)?"
        r"[\s,\-/]+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if named:
        candidate = " ".join(named.groups())
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
    raise ContractError(f"unparseable BEAM time anchor: {text!r}")


def flatten_source_ids(value: Any) -> list[str]:
    output: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, bool):
            raise ContractError("boolean BEAM source identifier")
        if isinstance(item, (str, int)):
            text = str(item).strip()
            if text:
                output.append(text)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
            return
        raise ContractError(f"unsupported BEAM source identifier: {item!r}")

    visit(value)
    return output


def parse_probing_questions(raw: Any) -> dict[str, Any]:
    value = raw
    if isinstance(raw, str):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ContractError("invalid BEAM probing_questions") from exc
    if not isinstance(value, dict):
        raise ContractError("BEAM probing_questions is not an object")
    return value


def extract_rubric_nuggets(question: Mapping[str, Any]) -> list[str]:
    raw = question.get("rubric", [])
    if isinstance(raw, dict):
        raw = raw.get("nuggets", [])
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    output: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            item = (
                item.get("description")
                or item.get("nugget")
                or item.get("criterion")
                or item.get("text")
            )
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output


def extract_questions(
    item: Mapping[str, Any], *, require_formal_inventory: bool = True
) -> list[dict[str, Any]]:
    grouped = parse_probing_questions(item.get("probing_questions", {}))
    unknown = sorted(set(grouped) - set(QUESTION_TYPES))
    if unknown:
        raise ContractError(f"unknown BEAM question types: {unknown}")
    questions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for question_type in QUESTION_TYPES:
        values = grouped.get(question_type, [])
        if isinstance(values, (dict, str)):
            values = [values]
        if not isinstance(values, list):
            raise ContractError(f"BEAM {question_type} question list is invalid")
        for raw in values:
            question = {"question": raw} if isinstance(raw, str) else dict(raw)
            text = str(
                question.get("question_text", question.get("question", "")) or ""
            ).strip()
            if not text:
                raise ContractError(f"empty BEAM {question_type} question")
            gold_field = GOLD_FIELD_BY_TYPE[question_type]
            if gold_field not in question:
                raise ContractError(
                    f"BEAM {question_type} question lacks {gold_field}"
                )
            question.update(
                {
                    "question_type": question_type,
                    "question_text": text,
                    "gold_field": gold_field,
                    "gold": question[gold_field],
                    "rubric_nuggets": extract_rubric_nuggets(question),
                    "raw_source_chat_ids": question.get("source_chat_ids"),
                    "normalized_source_chat_ids": flatten_source_ids(
                        question.get("source_chat_ids")
                    ),
                }
            )
            questions.append(question)
            counts[question_type] += 1
    if require_formal_inventory and (
        len(questions) != QUESTIONS_PER_CONVERSATION
        or any(counts[value] != 2 for value in QUESTION_TYPES)
    ):
        raise ContractError(
            "BEAM conversation must contain two questions for every frozen type"
        )
    return questions


def question_id(chat_size: str, conversation_index: int, question_index: int,
                question_type: str) -> str:
    return (
        f"beam:{chat_size}:c{conversation_index:03d}:"
        f"q{question_index:02d}:{question_type}"
    )


def bm25_tokens(text: str) -> list[str]:
    return BM25_TOKEN_PATTERN.findall(text.lower())


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    tokenized = [bm25_tokens(text) for text in documents]
    query_terms = list(dict.fromkeys(bm25_tokens(query)))
    lengths = [len(value) for value in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency = Counter(
        token for tokens in tokenized for token in set(tokens)
    )
    count = len(tokenized)
    output: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B
                + BM25_B * (len(tokens) / average_length if average_length else 0.0)
            )
            score += inverse * frequency * (BM25_K1 + 1.0) / denominator
        output.append(score)
    return output


class FrozenBM25Index:
    """Conversation-level BM25 statistics with score-equivalent postings."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.count = len(documents)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for document_index, text in enumerate(documents):
            frequencies = Counter(bm25_tokens(text))
            self.lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                self.postings.setdefault(term, []).append(
                    (document_index, frequency)
                )
        self.average_length = (
            sum(self.lengths) / self.count if self.count else 0.0
        )

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * self.count
        query_terms = list(dict.fromkeys(bm25_tokens(query)))
        for term in query_terms:
            postings = self.postings.get(term, [])
            if not postings:
                continue
            df = len(postings)
            inverse = math.log(
                1.0 + (self.count - df + 0.5) / (df + 0.5)
            )
            for document_index, frequency in postings:
                denominator = frequency + BM25_K1 * (
                    1.0
                    - BM25_B
                    + BM25_B
                    * (
                        self.lengths[document_index] / self.average_length
                        if self.average_length
                        else 0.0
                    )
                )
                scores[document_index] += (
                    inverse * frequency * (BM25_K1 + 1.0) / denominator
                )
        return scores


def dependency_snapshot() -> dict[str, str]:
    expected = {
        "tiktoken": TIKTOKEN_VERSION,
        "mem0ai": MEM0_VERSION,
        "qdrant-client": QDRANT_VERSION,
    }
    result: dict[str, str] = {}
    for package, version in expected.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ContractError(f"required package is missing: {package}") from exc
        if installed != version:
            raise ContractError(
                f"{package} version differs: expected {version}, got {installed}"
            )
        result[package] = installed
    return result


def validate_preregistration(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("R115 preregistration is not an object")
    exact = {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen",
        "benchmark": "BEAM",
        "milestone": "R115",
        "formal_methods": list(FORMAL_METHODS),
        "formal_selection": FORMAL_SELECTION,
        "formal_question_count": FORMAL_QUESTIONS,
        "requested_model": REQUESTED_MODEL,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ContractError(f"R115 preregistration field differs: {key}")
    if value.get("preregistration_content_sha256") != content_hash(
        value, "preregistration_content_sha256"
    ):
        raise ContractError("R115 preregistration content hash differs")
    return value


def source_hashes() -> dict[str, str]:
    paths = {
        "scripts/beam_controls/r115_beam_control_contract.py": Path(__file__).resolve(),
        "scripts/beam_controls/run_r115_beam_controls.py": ROOT / "scripts/beam_controls/run_r115_beam_controls.py",
        "scripts/beam_controls/audit_r115_beam_controls.py": ROOT / "scripts/beam_controls/audit_r115_beam_controls.py",
        "src/evaluation/visible_token_budget.py": ROOT / "src/evaluation/visible_token_budget.py",
        "src/evaluation/visible_token_audit.py": ROOT / "src/evaluation/visible_token_audit.py",
        "src/evaluation/durable_model_ledger.py": ROOT / "src/evaluation/durable_model_ledger.py",
        "scripts/gateways/openai_gpt55_flex_gateway.py": ROOT / "scripts/gateways/openai_gpt55_flex_gateway.py",
        "scripts/gateways/openai_gpt55_flex_gateway_evidence.py": ROOT / "scripts/gateways/openai_gpt55_flex_gateway_evidence.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ContractError(f"R115 source files are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))
