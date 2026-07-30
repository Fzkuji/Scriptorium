#!/usr/bin/env python3
"""Run NativeMem v8.8+calendar on BEAM with one GPT-5.5 agent.

This entry point supports the HuggingFace ``Mohammadta/BEAM`` 100K and 1M
splits.  One NativeMem library is built per BEAM conversation, then the same
GPT-5.5 client navigates that library and answers every probing question.

The run is resumable at conversation and question granularity.  Each
conversation owns a directory containing ``checkpoint.json``, ``results.json``
and ``memory/``.  A successful build is committed by an atomic directory rename
and a marker inside the memory directory, so a process interruption cannot make
a partial build look complete.  Exceptions are recorded and never converted to
successful question or conversation states.

Examples (these commands make model calls; this module's tests do not):

    python3 scripts/run_v88_gpt55_beam.py \
      --chat-sizes 100K --conversations 0 --limit 1

    python3 scripts/run_v88_gpt55_beam.py \
      --chat-sizes 100K,1M --conversations all --resume

Optional data-only prefetch (no model call):

    python3 -c 'from datasets import load_dataset; r="3205395e897e7318c7b094ef4e6047b9b82dbb03"; load_dataset("Mohammadta/BEAM", name="default", split="100K", revision=r, cache_dir="benchmarks/beam/hf_cache"); load_dataset("Mohammadta/BEAM", name="default", split="1M", revision=r, cache_dir="benchmarks/beam/hf_cache")'

The separate 10M files can be prefetched, without running this adapter, with:

    python3 -c 'from datasets import load_dataset; load_dataset("Mohammadta/BEAM-10M", split="10M", cache_dir="benchmarks/beam/hf_cache")'

``--limit`` limits the number of selected conversations *per split*.  It is a
safety/development control; it does not truncate a conversation or its QA set.

The 10M release uses a separate repository and a plan-keyed chat schema.  The
normalizer in this file understands that schema (and the ``plans[].chat``
fallback), but the CLI intentionally accepts only 100K/1M until a 10M run is
explicitly added and validated.  No dataset is downloaded merely by importing
this module.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "results" / "v88-calendar-gpt55-beam-20260714"
DEFAULT_CACHE = ROOT / "benchmarks" / "beam" / "hf_cache"
HF_DATASET = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"
DEFAULT_DATASET_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
SUPPORTED_CHAT_SIZES = ("100K", "1M")
EXPECTED_SPLIT_ROWS = {"100K": 20, "1M": 35}
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
CHECKPOINT_SCHEMA_VERSION = 1
BUILD_MARKER = "_beam_build.json"
CONTEXT_POLICY_VERSION = "beam-local-context-v1"
CONTEXT_TOKENIZER_PACKAGE = "tiktoken"
CONTEXT_TOKENIZER_VERSION = "0.12.0"
CONTEXT_TOKENIZER_ENCODING = "o200k_base"
LOCAL_REQUEST_TOKEN_LIMIT = 96_000
TOTAL_TOOL_CONTENT_TOKEN_LIMIT = 48_000
PER_TOOL_CONTENT_TOKEN_LIMIT = 24_000
CONTEXT_OUTPUT_RESERVATION_TOKENS = 1_200
CONTEXT_SAFETY_MARGIN_TOKENS = 8_192
CONTEXT_TRACE_SCHEMA_VERSION = "beam-context-safety-trace-v1"
INTERACTION_TRACE_SCHEMA_VERSION = "beam-interaction-trace-v1"
TRUNCATION_ALGORITHM = "decoded-prefix-plus-marker-recount-v1"
TOOL_TRUNCATION_MARKER = (
    "\n[Tool output truncated by beam-local-context-v1. Request a "
    "narrower range or a more specific search.]"
)
COMPACT_EVIDENCE_TRUNCATION_MARKER = (
    "\n[Additional retrieved evidence omitted by beam-local-context-v1.]"
)
FINALIZE_INSTRUCTION = (
    "Tool use is now finished. Using only the evidence already retrieved "
    "above, answer the original BEAM question completely. Follow its "
    "requested format. If the evidence is genuinely absent, use the "
    "specified insufficient-information response. Return the final answer "
    "inside <answer></answer>."
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / "src" / "adapters" / "run_nativemem.py",
    ROOT / "src" / "nativemem.py",
    ROOT / "src" / "v8_memory.py",
    ROOT / "src" / "chatgpt_proxy.py",
    ROOT / "src" / "openai_gpt55_flex_gateway.py",
    ROOT / "src" / "openai_gpt55_flex_gateway_evidence.py",
    ROOT / "scripts" / "gpt55_run_proxy.py",
)


BEAM_SINGLE_PROMPT = """You answer one BEAM benchmark question by navigating a NativeMem library.
You may use list_memory, search_memory, read_memory, and read_original only.
The first three tools expose a read-only view of the memory-library root.

The library has two views:
- topics/: topic files or nested topic directories. Read the complete relevant
  files instead of relying on filename or keyword overlap alone.
- timeline/YYYY/MM/DD.md: atomic events ordered by date. An event ends with one
  or more [Dn:m] references and may link back to its topic file.

[Dn:m] references identify original conversation turns. Pass all references
that may matter to read_original in one call. The returned turns include their
observation date. Use the original turns to verify names, numbers, dates,
constraints, preferences, updates, and contradictions.

Current library structure:
{structure}

Question:
{question}

Required procedure:
1. Use the structure above to read every topic likely to contain evidence. Use
   search_memory with several alternative terms only when topic navigation is
   insufficient.
2. For temporal or ordering questions, inspect the relevant timeline files and
   nearby dates. For contradictions or updates, prefer the latest supported
   statement while preserving any distinction the question asks about.
3. Call read_original for all candidate [Dn:m] references before answering when
   references are available. Check every requested part of the question.
4. Give a complete BEAM-style answer. There is no short-answer word limit.
   Follow any format or preference requested by the question. If the evidence
   is genuinely absent after a relevant search, answer exactly: "I don't have
   enough information to answer this question."
5. Put the final answer inside <answer></answer>. Do not put analysis inside the
   tags.

Tool results may end with a deterministic truncation notice. When that occurs,
request a narrower file range or a more specific search instead of assuming
that omitted lines are absent.
"""


BEAM_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_memory",
            "description": (
                "List directories and Markdown files inside the read-only "
                "NativeMem library. Paths are relative to the library root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Case-insensitive literal search over Markdown files in the "
                "read-only NativeMem library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {
                        "type": "integer", "minimum": 1, "maximum": 200,
                        "default": 80,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": (
                "Read a line range from one Markdown file in the read-only "
                "NativeMem library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {
                        "type": "integer", "minimum": 1, "default": 1,
                    },
                    "end_line": {
                        "type": "integer", "minimum": 1, "default": 2000,
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    """Write JSON in one filesystem replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stable_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in SOURCE_FILES
    }


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def parse_chat_sizes(spec: str) -> list[str]:
    sizes: list[str] = []
    for raw in spec.split(","):
        size = raw.strip().upper()
        if not size:
            continue
        if size not in SUPPORTED_CHAT_SIZES:
            allowed = ",".join(SUPPORTED_CHAT_SIZES)
            raise argparse.ArgumentTypeError(
                f"unsupported chat size {size!r}; supported: {allowed}")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one chat size is required")
    return sizes


def parse_indices(spec: str, size: int | None = None) -> list[int] | None:
    """Parse ``all``, comma lists, ranges, or a mixture of lists/ranges."""
    if spec.strip().lower() == "all":
        return None if size is None else list(range(size))
    out: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                lo, hi = sorted((a, b))
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid conversation specification: {spec!r}") from exc
    values = sorted(out)
    if not values or any(i < 0 for i in values):
        raise argparse.ArgumentTypeError(
            "conversation indices must be non-negative")
    if size is not None and any(i >= size for i in values):
        raise IndexError(
            f"conversation selection {values} exceeds split size {size}")
    return values


def _plan_key(key: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(key))
    return (int(match.group(1)) if match else 10**9, str(key))


def _flatten_turns(items: Any) -> list[dict[str, Any]]:
    """Flatten the one extra list level used inside BEAM batch ``turns``."""
    flat: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return flat
    for item in items:
        if isinstance(item, dict):
            flat.append(dict(item))
        elif isinstance(item, list):
            flat.extend(_flatten_turns(item))
    return flat


def _unwrap_batch_dicts(batch_dicts: Any) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    if not isinstance(batch_dicts, list):
        return batches
    for batch in batch_dicts:
        if not isinstance(batch, dict):
            continue
        turns = _flatten_turns(batch.get("turns", []))
        if turns:
            batches.append(turns)
    return batches


def parse_beam_chat(chat_data: Any, plans: Any = None) -> list[list[dict[str, Any]]]:
    """Normalize the released BEAM chat layouts into ``batch -> turns``.

    Supported layouts:
    - 100K/500K/1M: a two-dimensional list of turn dictionaries.
    - batch dictionaries containing a ``turns`` field.
    - 10M: session dictionaries keyed by ``plan-1`` ... ``plan-10``;
      each value is a list of batch dictionaries.
    - 10M ``plans[].chat`` as a fallback when the top-level chat is absent.
    """
    batches: list[list[dict[str, Any]]] = []
    if isinstance(chat_data, list) and chat_data:
        first = chat_data[0]
        if isinstance(first, list):
            batches = [_flatten_turns(batch) for batch in chat_data]
            batches = [batch for batch in batches if batch]
        elif isinstance(first, dict) and "turns" in first:
            batches = _unwrap_batch_dicts(chat_data)
        elif isinstance(first, dict) and (
                "role" in first or "content" in first):
            one = _flatten_turns(chat_data)
            batches = [one] if one else []
        elif isinstance(first, dict):
            # BEAM-10M top-level plan map.
            for session in chat_data:
                if not isinstance(session, dict):
                    continue
                for key in sorted(session, key=_plan_key):
                    batches.extend(_unwrap_batch_dicts(session.get(key)))

    if not batches and isinstance(plans, list):
        # The 10M release also exposes each plan's chat in plans[].chat.
        for plan in sorted(
                (p for p in plans if isinstance(p, dict)),
                key=lambda p: _plan_key(str(p.get("plan_id", "")))):
            batches.extend(parse_beam_chat(plan.get("chat", [])))
    return batches


def normalize_time_anchor(raw: Any) -> str:
    """Normalize BEAM time anchors, including ``March-15-2024``, to ISO."""
    if raw is None:
        raise ValueError("missing BEAM time_anchor")
    text = str(raw).strip()
    if not text:
        raise ValueError("empty BEAM time_anchor")

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
                pass
    raise ValueError(f"unparseable BEAM time_anchor: {text!r}")


def conversation_to_native(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one HF conversation into the session schema used by v8 build."""
    batches = parse_beam_chat(item.get("chat", []), plans=item.get("plans"))
    if not batches:
        raise ValueError("BEAM conversation has no parseable chat batches")

    conv: dict[str, Any] = {}
    total_turns = 0
    batches_with_multiple_dates = 0
    source_id_map: dict[str, list[str]] = {}
    for session_no, batch in enumerate(batches, 1):
        turns: list[dict[str, str]] = []
        anchors: list[str] = []
        for turn in batch:
            content = str(turn.get("content", "") or "").strip()
            if not content:
                continue
            role = str(turn.get("role", "user") or "user").strip().lower()
            if role in ("human", "customer"):
                role = "user"
            elif role not in ("user", "assistant"):
                role = "assistant" if role in ("ai", "bot") else role
            raw_anchor = turn.get("time_anchor")
            if raw_anchor:
                anchors.append(str(raw_anchor))
            turn_no = len(turns) + 1
            dia_id = f"D{session_no}:{turn_no}"
            turns.append({
                "speaker": role,
                "text": content,
                "dia_id": dia_id,
            })
            if turn.get("id") is not None:
                source_id_map.setdefault(str(turn["id"]), []).append(dia_id)
        if not turns:
            raise ValueError(f"BEAM batch {session_no} contains no non-empty turns")
        if not anchors:
            raise ValueError(f"BEAM batch {session_no} has no time_anchor")
        dates = []
        for anchor in anchors:
            normalized = normalize_time_anchor(anchor)
            if normalized not in dates:
                dates.append(normalized)
        batches_with_multiple_dates += int(len(dates) > 1)
        conv[f"session_{session_no}"] = turns
        # Match the Mem0 runner: one observation date per batch, taken from the
        # earliest turn carrying an anchor.  We normalize before v8 so its
        # deterministic calendar strip is active for BEAM.
        conv[f"session_{session_no}_date_time"] = dates[0]
        total_turns += len(turns)

    return conv, {
        "sessions": len(batches),
        "turns": total_turns,
        "batches_with_multiple_dates": batches_with_multiple_dates,
        "source_id_map": source_id_map,
    }


def parse_probing_questions(raw: Any) -> dict[str, Any]:
    """Decode the HF string-repr field without evaluating executable code."""
    value = raw
    if isinstance(raw, str):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid BEAM probing_questions string") from exc
    if not isinstance(value, dict):
        raise ValueError("BEAM probing_questions must decode to an object")
    return value


def extract_rubric_nuggets(question: dict[str, Any]) -> list[str]:
    raw = question.get("rubric", [])
    if isinstance(raw, dict):
        raw = raw.get("nuggets", [])
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    nuggets: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            text = (item.get("description") or item.get("nugget")
                    or item.get("criterion") or item.get("text") or str(item))
        else:
            text = item
        text = str(text or "").strip()
        if text:
            nuggets.append(text)
    return nuggets


def extract_questions(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all BEAM questions in the official ten-type ordering."""
    grouped = parse_probing_questions(item.get("probing_questions", {}))
    ordered_types = list(QUESTION_TYPES)
    ordered_types.extend(sorted(k for k in grouped if k not in QUESTION_TYPES))
    questions: list[dict[str, Any]] = []
    for question_type in ordered_types:
        values = grouped.get(question_type, [])
        if isinstance(values, dict) or isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str):
                question = {"question_text": value}
            elif isinstance(value, dict):
                question = dict(value)
            else:
                continue
            text = str(question.get(
                "question_text", question.get("question", "")) or "").strip()
            if not text:
                raise ValueError(
                    f"empty probing question in type {question_type}")
            question["question_type"] = question_type
            question["question_text"] = text
            question["rubric_nuggets"] = extract_rubric_nuggets(question)
            gold_field = GOLD_FIELD_BY_TYPE.get(question_type)
            if gold_field is None:
                raise ValueError(
                    f"unknown BEAM question type has no gold mapping: "
                    f"{question_type}")
            if gold_field not in question:
                raise ValueError(
                    f"BEAM {question_type} question lacks {gold_field!r}")
            question["gold_field"] = gold_field
            question["gold"] = question[gold_field]
            questions.append(question)
    if not questions:
        raise ValueError("BEAM conversation has no probing questions")
    return questions


def conversation_digest(conv: dict[str, Any],
                        questions: Iterable[dict[str, Any]]) -> str:
    """Hash normalized input incrementally without duplicating a 1M row."""
    digest = hashlib.sha256()
    session_no = 1
    while f"session_{session_no}" in conv:
        digest.update(f"session:{session_no}\n".encode())
        digest.update(str(conv.get(
            f"session_{session_no}_date_time", "")).encode("utf-8"))
        digest.update(b"\n")
        for turn in conv[f"session_{session_no}"]:
            for key in ("dia_id", "speaker", "text"):
                digest.update(str(turn.get(key, "")).encode("utf-8"))
                digest.update(b"\0")
        session_no += 1
    for q in questions:
        relevant = {
            key: q.get(key)
            for key in (
                "question_type", "question_text", "difficulty",
                "gold_field", "gold", "rubric_nuggets", "plan_reference",
                "source_chat_ids", "abstention_type", "why_unanswerable",
            )
        }
        digest.update(json.dumps(
            relevant, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def frozen_method_config(args: argparse.Namespace) -> dict[str, Any]:
    """Configuration defining the v8.8+calendar method, excluding secrets."""
    return {
        "method": "NativeMem-v8.8+calendar",
        "model": args.model,
        "provider": (
            "openai_api_flex_via_exclusive_child_proxy"
            if getattr(args, "formal_provider", False)
            else "offline_fixture_no_model_requests"
        ),
        "gateway_root": (
            str(args.gateway_root.expanduser().resolve())
            if getattr(args, "formal_provider", False) else None
        ),
        "explicit_model_request_authorization": bool(
            getattr(args, "allow_model_requests", False)
        ),
        "single_model_retrieve_answer": True,
        "chunk_turns": 6,
        "segment": "fixed",
        "tidy": True,
        "sections": True,
        "article": False,
        "tidy_combined": False,
        "verify": True,
        "merge_lines": True,
        "max_topics": 30,
        "map": "dir",
        "map_inline": 8,
        "max_rounds": 12,
        "max_tokens": 1200,
        "read_context": 1,
        "rewrite_min": 8,
        "max_depth": 4,
        "top_k": 20,
        "memory_tool_input_error_policy": "return-validated-error-to-model-v2",
        "context_safety": {
            "policy_version": CONTEXT_POLICY_VERSION,
            "trace_schema_version": CONTEXT_TRACE_SCHEMA_VERSION,
            "interaction_trace_schema_version": INTERACTION_TRACE_SCHEMA_VERSION,
            "tokenizer_package": CONTEXT_TOKENIZER_PACKAGE,
            "tokenizer_version": CONTEXT_TOKENIZER_VERSION,
            "tokenizer_encoding": CONTEXT_TOKENIZER_ENCODING,
            "provider_exact": False,
            "local_request_token_limit": LOCAL_REQUEST_TOKEN_LIMIT,
            "total_tool_content_token_limit": TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
            "per_tool_content_token_limit": PER_TOOL_CONTENT_TOKEN_LIMIT,
            "output_reservation_tokens": CONTEXT_OUTPUT_RESERVATION_TOKENS,
            "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
            "truncation_algorithm": TRUNCATION_ALGORITHM,
            "overflow_action": "truncate_tool_prefix_then_compact_finalization",
        },
        "trust_system_proxy": False,
        "request_concurrency": args.request_concurrency,
        "provider_evidence_mode": (
            "bounded_gateway_segments_with_exclusive_child_proxy"
            if getattr(args, "formal_provider", False) else None
        ),
        "dataset": HF_DATASET,
        "dataset_revision": args.dataset_revision,
        "calendar": True,
        "build_import": ({
            "root": str(args.reuse_build_root),
            "run_manifest_sha256": args.reuse_build_manifest_sha256,
        } if getattr(args, "reuse_build_root", None) is not None else None),
        "source_sha256": source_fingerprints(),
    }


def configure_native_runtime(args: argparse.Namespace) -> None:
    """Set the frozen method before importing modules that construct clients."""
    values = {
        "NATIVEMEM_PROMPT": "v8",
        "NATIVEMEM_STORE_MODE": "oneshot",
        "NATIVEMEM_V8_SINGLE": "1",
        "NATIVEMEM_CHUNK_TURNS": "6",
        "NATIVEMEM_V8_SEGMENT": "fixed",
        "NATIVEMEM_V8_TIDY": "on",
        "NATIVEMEM_V8_SECTIONS": "on",
        "NATIVEMEM_V8_ARTICLE": "off",
        "NATIVEMEM_V8_TIDY_COMBINED": "off",
        "NATIVEMEM_V8_VERIFY": "on",
        "NATIVEMEM_V8_MERGE_LINES": "on",
        "NATIVEMEM_V8_MAX_TOPICS": "30",
        "NATIVEMEM_V8_MAP": "dir",
        "NATIVEMEM_V8_MAP_INLINE": "8",
        "NATIVEMEM_V8_MAX_ROUNDS": "12",
        "NATIVEMEM_V8_MAX_TOKENS": "1200",
        "NATIVEMEM_V8_READ_CONTEXT": "1",
        "NATIVEMEM_V8_REWRITE_MIN": "8",
        "NATIVEMEM_V8_MAX_DEPTH": "4",
        "NATIVEMEM_TOPK": "20",
        "NATIVEMEM_TRUST_PROXY": "0",
        "NATIVEMEM_V8_CONCURRENCY": str(args.request_concurrency),
        "MODEL": args.model,
        "BUILDER_MODEL": args.model,
        "BUILDER_BASE": args.base_url,
        "BUILDER_KEY": args.api_key,
        "ALIYUN_KEY": args.api_key,
    }
    os.environ.update(values)
    for key in ("NATIVEMEM_V9_PIPELINE", "NATIVEMEM_V9_SCRIBE_MODE",
                "NATIVEMEM_V9_SCRIBE_MODEL", "NATIVEMEM_V9_COMPOSER_MODEL"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"


def load_native(args: argparse.Namespace) -> Any:
    configure_native_runtime(args)
    native = importlib.import_module("src.adapters.run_nativemem")
    if native.ALIYUN_MODEL != args.model:
        raise RuntimeError(
            "NativeMem was imported before GPT-5.5 runtime configuration; "
            "run this entry point in a fresh Python process")
    return native


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _resolve_memory_path(memory_dir: Path, raw_path: Any) -> Path:
    root = memory_dir.resolve()
    text = str(raw_path or ".").strip()
    candidate = Path(text)
    if candidate.is_absolute():
        raise ValueError("memory tool paths must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("memory tool path leaves the memory directory")
    if not resolved.exists():
        raise FileNotFoundError(f"memory path does not exist: {text}")
    return resolved


def _memory_markdown_files(path: Path, root: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".md" else []
    return sorted(
        candidate for candidate in path.rglob("*.md")
        if candidate.resolve().is_relative_to(root)
        if not any(part.startswith(".") or part == "raw"
                   for part in candidate.relative_to(path).parts)
    )


def execute_memory_tool(tool_name: str, args: dict[str, Any],
                        memory_dir: Path) -> str:
    """Execute only structured, read-only operations inside ``memory_dir``."""
    root = memory_dir.resolve()
    target = _resolve_memory_path(root, args.get("path", "."))
    if tool_name == "list_memory":
        recursive = bool(args.get("recursive", False))
        if target.is_file():
            candidates = [target]
        elif recursive:
            candidates = sorted(target.rglob("*"))
        else:
            candidates = sorted(target.iterdir())
        lines: list[str] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                continue
            rel = resolved.relative_to(root)
            if any(part.startswith(".") or part == "raw" for part in rel.parts):
                continue
            if resolved.is_dir():
                lines.append(f"{rel.as_posix()}/")
            elif resolved.suffix == ".md":
                lines.append(rel.as_posix())
            if len(lines) >= 2000:
                lines.append("(listing truncated at 2000 entries)")
                break
        return "\n".join(lines) or "(no Markdown files)"

    if tool_name == "search_memory":
        query = str(args.get("query", "") or "").strip()
        if not query:
            raise ValueError("search_memory requires a non-empty query")
        try:
            max_results = max(1, min(200, int(args.get("max_results", 80))))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_results must be an integer") from exc
        needle = query.casefold()
        hits: list[str] = []
        for path in _memory_markdown_files(target, root):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, 1):
                    if needle in line.casefold():
                        rel = path.resolve().relative_to(root).as_posix()
                        hits.append(f"{rel}:{line_no}:{line.rstrip()}")
                        if len(hits) >= max_results:
                            hits.append(
                                f"(search truncated at {max_results} results)")
                            return "\n".join(hits)
        return "\n".join(hits) or "(no matches)"

    if tool_name == "read_memory":
        if not target.is_file() or target.suffix != ".md":
            raise ValueError("read_memory path must name a Markdown file")
        try:
            start = max(1, int(args.get("start_line", 1)))
            end = int(args.get("end_line", 2000))
        except (TypeError, ValueError) as exc:
            raise ValueError("line bounds must be integers") from exc
        if end < start:
            raise ValueError("end_line must be greater than or equal to start_line")
        end = min(end, start + 1999)
        lines: list[str] = []
        with target.open(encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if line_no < start:
                    continue
                if line_no > end:
                    break
                lines.append(f"{line_no}: {line.rstrip()}")
        return "\n".join(lines) or "(requested range is empty)"

    raise ValueError(f"unsupported memory tool: {tool_name}")


def execute_memory_tool_for_model(tool_name: str, args: dict[str, Any],
                                  memory_dir: Path) -> tuple[str, str | None]:
    """Return expected tool-input errors to the model instead of failing QA.

    A missing or malformed model-selected path is a normal navigation miss.
    The resolver validates confinement before any file access, so returning the
    validation message is safe and lets the model list/search and retry.  Only
    these deterministic input errors are converted; unexpected filesystem or
    implementation failures still abort the question.
    """
    try:
        return execute_memory_tool(tool_name, args, memory_dir), None
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError,
            ValueError) as exc:
        message = f"Error: {type(exc).__name__}: {exc}"
        return message, message


def _validate_final_response(response: Any, content: str) -> re.Match[str]:
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason != "stop":
        raise RuntimeError(
            f"final answer did not finish normally: {finish_reason!r}")
    refusal = str(getattr(choice.message, "refusal", "") or "").strip()
    if refusal:
        raise RuntimeError(f"model refused the final answer: {refusal[:300]}")
    match = re.search(
        r"<answer>(.*?)</answer>", content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match or not match.group(1).strip():
        raise RuntimeError(
            "final answer must contain a non-empty <answer></answer> block")
    return match


def _validate_tool_response(response: Any) -> None:
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason != "tool_calls":
        raise RuntimeError(
            "tool response did not finish with tool_calls: "
            f"{finish_reason!r}")
    refusal = str(getattr(choice.message, "refusal", "") or "").strip()
    if refusal:
        raise RuntimeError(f"model refused tool use: {refusal[:300]}")


def _load_context_tokenizer() -> tuple[Any, dict[str, Any]]:
    """Load the frozen local tokenizer; formal runs do not use a fallback."""
    try:
        version = importlib_metadata.version(CONTEXT_TOKENIZER_PACKAGE)
        tiktoken = importlib.import_module(CONTEXT_TOKENIZER_PACKAGE)
    except (importlib_metadata.PackageNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "BEAM context safety requires tiktoken==0.12.0; use the frozen "
            "/opt/miniconda3/bin/python3 interpreter"
        ) from exc
    if version != CONTEXT_TOKENIZER_VERSION:
        raise RuntimeError(
            "BEAM context tokenizer version mismatch: expected "
            f"{CONTEXT_TOKENIZER_VERSION}, received {version}"
        )
    return tiktoken.get_encoding(CONTEXT_TOKENIZER_ENCODING), {
        "package": CONTEXT_TOKENIZER_PACKAGE,
        "version": version,
        "encoding": CONTEXT_TOKENIZER_ENCODING,
        "model_alias": "gpt-5.5",
        "provider_exact": False,
    }


def _local_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))


def _jsonable_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        normalized.append({
            "id": str(getattr(tool_call, "id", "") or ""),
            "type": str(getattr(tool_call, "type", "function") or "function"),
            "function": {
                "name": str(getattr(function, "name", "") or ""),
                "arguments": str(getattr(function, "arguments", "") or "{}"),
            },
        })
    return normalized


def _request_local_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    return _request_descriptor(tokenizer, messages, tools)["local_tokens"]


def _request_descriptor(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": messages}
    if tools is not None:
        payload["tools"] = tools
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    local_tokens = _local_tokens(tokenizer, rendered)
    bounded_total = (
        local_tokens
        + CONTEXT_OUTPUT_RESERVATION_TOKENS
        + CONTEXT_SAFETY_MARGIN_TOKENS
    )
    return {
        "local_tokens": local_tokens,
        "payload_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "payload_utf8_bytes": len(rendered.encode("utf-8")),
        "bounded_total_tokens": bounded_total,
        "within_local_limit": bounded_total <= LOCAL_REQUEST_TOKEN_LIMIT,
    }


def _safe_decoded_prefix(
    text: str,
    token_ids: list[int],
    count: int,
    tokenizer: Any,
) -> tuple[str, int]:
    """Decode only a token prefix that is also a literal text prefix."""
    count = max(0, min(count, len(token_ids)))
    while count:
        candidate = tokenizer.decode(token_ids[:count])
        if text.startswith(candidate):
            return candidate, count
        count -= 1
    return "", 0


def _truncate_tool_content(
    text: str,
    *,
    tokenizer: Any,
    limit: int,
    marker: str = TOOL_TRUNCATION_MARKER,
) -> tuple[str, int, int, bool]:
    """Prefix-truncate text and recount the exact decoded delivery."""
    raw_tokens = tokenizer.encode(text, disallowed_special=())
    if len(raw_tokens) <= limit:
        return text, len(raw_tokens), len(raw_tokens), False
    marker_tokens = tokenizer.encode(marker, disallowed_special=())
    if limit <= 0:
        return "", len(raw_tokens), 0, True
    if len(marker_tokens) > limit:
        delivered, _ = _safe_decoded_prefix(
            marker, marker_tokens, limit, tokenizer
        )
        return delivered, len(raw_tokens), _local_tokens(tokenizer, delivered), True

    prefix_slots = max(0, limit - len(marker_tokens))
    while True:
        prefix, used_slots = _safe_decoded_prefix(
            text, raw_tokens, prefix_slots, tokenizer
        )
        delivered = prefix + marker
        delivered_count = _local_tokens(tokenizer, delivered)
        if delivered_count <= limit:
            return delivered, len(raw_tokens), delivered_count, True
        if used_slots <= 0:
            raise RuntimeError("truncation marker exceeds its verified token limit")
        prefix_slots = max(0, used_slots - max(1, delivered_count - limit))


def _compact_finalize_messages(
    *,
    tokenizer: Any,
    question: str,
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    header = (
        "Answer the BEAM question using only the retrieved tool evidence below. "
        "Follow the requested format and return the final answer inside "
        "<answer></answer>. If the evidence is genuinely absent, answer exactly: "
        "\"I don't have enough information to answer this question.\"\n\n"
        f"Question:\n{question}\n\nRetrieved tool evidence:\n"
    )
    blocks = [
        f"\n[{item['tool']} step={item['step']}]\n{item['delivered_text']}"
        for item in evidence
        if item.get("delivered_text")
    ]
    raw_evidence = "".join(blocks)
    raw = header + raw_evidence
    request_limit = (
        LOCAL_REQUEST_TOKEN_LIMIT
        - CONTEXT_OUTPUT_RESERVATION_TOKENS
        - CONTEXT_SAFETY_MARGIN_TOKENS
    )
    raw_request_tokens = _request_local_tokens(
        tokenizer, [{"role": "user", "content": raw}], None
    )
    delivered_evidence = raw_evidence
    truncated = False
    if raw_request_tokens > request_limit:
        truncated = True
        header_request_tokens = _request_local_tokens(
            tokenizer, [{"role": "user", "content": header}], None
        )
        if header_request_tokens > request_limit:
            raise RuntimeError(
                "BEAM question alone exceeds the compact-finalization limit"
            )
        evidence_limit = max(0, request_limit - header_request_tokens)
        while True:
            delivered_evidence, _, _, _ = _truncate_tool_content(
                raw_evidence,
                tokenizer=tokenizer,
                limit=evidence_limit,
                marker=COMPACT_EVIDENCE_TRUNCATION_MARKER,
            )
            delivered = header + delivered_evidence
            delivered_request_tokens = _request_local_tokens(
                tokenizer, [{"role": "user", "content": delivered}], None
            )
            if delivered_request_tokens <= request_limit:
                break
            if evidence_limit <= 0:
                raise RuntimeError(
                    "compact-finalization header cannot satisfy the request limit"
                )
            evidence_limit = max(
                0,
                evidence_limit - max(
                    1, delivered_request_tokens - request_limit
                ),
            )
    delivered = header + delivered_evidence
    raw_tokens = _local_tokens(tokenizer, raw)
    delivered_tokens = _local_tokens(tokenizer, delivered)
    delivered_request_tokens = _request_local_tokens(
        tokenizer, [{"role": "user", "content": delivered}], None
    )
    return [{"role": "user", "content": delivered}], {
        "schema_version": "beam-compact-finalization-v1",
        "raw_content_tokens": raw_tokens,
        "delivered_content_tokens": delivered_tokens,
        "raw_request_tokens": raw_request_tokens,
        "delivered_request_tokens": delivered_request_tokens,
        "truncated": truncated,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "delivered_sha256": hashlib.sha256(
            delivered.encode("utf-8")
        ).hexdigest(),
        "source_tool_trace_sha256": stable_hash(evidence),
    }


def answer_question(native: Any, question: str, memory_dir: Path,
                    turn_index: dict[str, Any]) -> dict[str, Any]:
    """Strict BEAM variant of v8's one-model retrieve+answer loop.

    It deliberately does not catch client exceptions.  The caller records them
    as failed question checkpoints, instead of producing a successful empty
    answer as the generic LoCoMo adapter does.
    """
    max_rounds = int(os.environ.get("NATIVEMEM_V8_MAX_ROUNDS", "12"))
    max_tokens = int(os.environ.get("NATIVEMEM_V8_MAX_TOKENS", "1200"))
    read_context = int(os.environ.get("NATIVEMEM_V8_READ_CONTEXT", "1"))
    structure = native._v8_structure_map(str(memory_dir))
    prompt = BEAM_SINGLE_PROMPT.format(question=question, structure=structure)
    messages: list[Any] = [{"role": "user", "content": prompt}]
    tools = BEAM_MEMORY_TOOLS + [native._V8_READ_TOOL]
    tokenizer, tokenizer_identity = _load_context_tokenizer()
    memories: list[str] = []
    tool_input_errors: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    response_trace: list[dict[str, Any]] = []
    raw_tool_tokens_total = 0
    delivered_tool_tokens_total = 0
    request_observations: list[dict[str, Any]] = []
    context_compaction_used = False
    compact_finalization: dict[str, Any] | None = None
    steps = 0
    final_text = ""
    finished = False
    termination_reason = "max_rounds"
    response: Any | None = None
    response_models: set[str] = set()
    response_ids: list[str] = []
    seen_tool_call_ids: set[str] = set()

    def observe_request(
        phase: str,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]] | None,
        *,
        attempt: int,
        sent: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        descriptor = _request_descriptor(
            tokenizer, request_messages, request_tools
        )
        record = {
            "sequence": len(request_observations) + 1,
            "phase": phase,
            "attempt": attempt,
            "sent": sent,
            "outcome": "pending" if sent else "not_sent",
            "reason": reason,
            "tools_enabled": request_tools is not None,
            "message_count": len(request_messages),
            "output_reservation_tokens": CONTEXT_OUTPUT_RESERVATION_TOKENS,
            "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
            **descriptor,
        }
        if sent and not descriptor["within_local_limit"]:
            raise RuntimeError("attempted to send a locally over-limit request")
        request_observations.append(record)
        return record

    def record_response(
        current_response: Any,
        phase: str,
        observation: dict[str, Any],
    ) -> None:
        nonlocal steps
        observation["outcome"] = "response_received"
        native.log_usage(current_response, phase=phase)
        model = str(getattr(current_response, "model", "") or "").strip()
        response_id = str(getattr(current_response, "id", "") or "").strip()
        if not response_id:
            raise RuntimeError("backend response id is missing")
        if response_id in response_ids:
            raise RuntimeError(f"backend response id is reused: {response_id}")
        if model != native.ALIYUN_MODEL:
            raise RuntimeError(
                "backend model mismatch: requested "
                f"{native.ALIYUN_MODEL!r}, received {model or '<missing>'!r}"
            )
        choice = current_response.choices[0]
        message = choice.message
        normalized_tool_calls = _jsonable_tool_calls(
            getattr(message, "tool_calls", None)
        )
        steps += 1
        response_trace.append({
            "step": steps,
            "phase": phase,
            "request_sequence": observation["sequence"],
            "response_id": response_id,
            "model": model,
            "finish_reason": getattr(choice, "finish_reason", None),
            "refusal": str(getattr(message, "refusal", "") or ""),
            "content": str(getattr(message, "content", "") or ""),
            "tool_calls": normalized_tool_calls,
        })
        response_models.add(model)
        response_ids.append(response_id)

    for _ in range(max_rounds):
        descriptor = _request_descriptor(tokenizer, messages, tools)
        if not descriptor["within_local_limit"]:
            observe_request(
                "retrieve",
                messages,
                tools,
                attempt=1,
                sent=False,
                reason="local_request_limit",
            )
            context_compaction_used = True
            termination_reason = "retrieval_request_limit"
            break
        observation = observe_request(
            "retrieve", messages, tools, attempt=1, sent=True
        )
        response = native.client.chat.completions.create(
            model=native.ALIYUN_MODEL,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        record_response(response, "beam_v8_single", observation)
        message = response.choices[0].message
        content = _strip_think(message.content or "")
        if content:
            final_text = content
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            _validate_final_response(response, content)
            finished = True
            termination_reason = "model_answer"
            break
        _validate_tool_response(response)
        normalized_tool_calls = _jsonable_tool_calls(tool_calls)
        for normalized in normalized_tool_calls:
            call_id = normalized["id"]
            if not call_id or call_id in seen_tool_call_ids:
                raise RuntimeError("tool-call ids must be non-empty and unique")
            seen_tool_call_ids.add(call_id)
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": normalized_tool_calls,
        })
        for tool_call, normalized_tool_call in zip(
                tool_calls, normalized_tool_calls, strict=True):
            argument_error: str | None = None
            try:
                decoded_args = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(decoded_args, dict):
                    raise ValueError("tool arguments must be a JSON object")
                tool_args = decoded_args
            except (json.JSONDecodeError, ValueError) as exc:
                tool_args = {}
                argument_error = f"Error: {type(exc).__name__}: {exc}"
            tool_error: str | None = argument_error
            if argument_error is not None:
                raw_tool_output = argument_error
            elif tool_call.function.name == "read_original":
                dia_ids = tool_args.get("dia_ids", [])
                if isinstance(dia_ids, str):
                    dia_ids = [dia_ids]
                original = native.read_turns(
                    turn_index, dia_ids, context=read_context)
                raw_tool_output = str(original or "(no matching original turns)")
            else:
                raw_tool_output, tool_error = execute_memory_tool_for_model(
                    tool_call.function.name, tool_args, memory_dir)
            if tool_error is not None:
                tool_input_errors.append({
                    "step": steps,
                    "tool": tool_call.function.name,
                    "arguments": tool_args,
                    "error": tool_error,
                })
            remaining = max(
                0, TOTAL_TOOL_CONTENT_TOKEN_LIMIT - delivered_tool_tokens_total
            )
            call_limit = min(PER_TOOL_CONTENT_TOKEN_LIMIT, remaining)
            delivered_output, raw_tokens, delivered_tokens, truncated = (
                _truncate_tool_content(
                    raw_tool_output, tokenizer=tokenizer, limit=call_limit
                )
            )
            raw_tool_tokens_total += raw_tokens
            delivered_tool_tokens_total += delivered_tokens
            if tool_call.function.name == "read_original" and delivered_output.strip():
                memories.append(delivered_output)
            trace_record = {
                "step": steps,
                "tool_call_id": normalized_tool_call["id"],
                "tool": tool_call.function.name,
                "arguments": tool_args,
                "raw_sha256": hashlib.sha256(
                    raw_tool_output.encode("utf-8")
                ).hexdigest(),
                "delivered_sha256": hashlib.sha256(
                    delivered_output.encode("utf-8")
                ).hexdigest(),
                "raw_tokens": raw_tokens,
                "delivered_tokens": delivered_tokens,
                "truncated": truncated,
                "remaining_tokens_before": remaining,
                "applied_token_limit": call_limit,
                "truncation_algorithm": TRUNCATION_ALGORITHM,
                "cumulative_delivered_tokens": delivered_tool_tokens_total,
                "delivered_text": delivered_output,
            }
            if tool_call.function.name == "read_original":
                trace_record["raw_text"] = raw_tool_output
            tool_trace.append(trace_record)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": delivered_output,
            })
        if delivered_tool_tokens_total >= TOTAL_TOOL_CONTENT_TOKEN_LIMIT:
            termination_reason = "tool_content_budget"
            break

    if not finished:
        # Match the current v8 single-agent safeguard: once the tool budget is
        # exhausted, keep the complete trace but make a tools-disabled call.
        # BEAM answers are not constrained to LoCoMo's 5-6-word format.
        finalize = {
            "role": "user",
            "content": FINALIZE_INSTRUCTION,
        }
        final_messages = messages + [finalize]
        full_descriptor = _request_descriptor(tokenizer, final_messages, None)
        if (
            context_compaction_used
            or not full_descriptor["within_local_limit"]
        ):
            observe_request(
                "finalize_full",
                final_messages,
                None,
                attempt=0,
                sent=False,
                reason=(
                    "prior_retrieval_overflow"
                    if context_compaction_used
                    else "local_request_limit"
                ),
            )
            context_compaction_used = True
            final_messages, compact_finalization = _compact_finalize_messages(
                tokenizer=tokenizer,
                question=question,
                evidence=tool_trace,
            )
            final_phase = "finalize_compact"
        else:
            final_phase = "finalize_full"
        final_descriptor = _request_descriptor(tokenizer, final_messages, None)
        if not final_descriptor["within_local_limit"]:
            raise RuntimeError(
                "local context safety could not construct a bounded final request"
            )
        last_error: BaseException | None = None
        for retry in range(3):
            observation = observe_request(
                final_phase,
                final_messages,
                None,
                attempt=retry + 1,
                sent=True,
            )
            try:
                response = native.client.chat.completions.create(
                    model=native.ALIYUN_MODEL,
                    messages=final_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                observation["outcome"] = "client_error"
                observation["reason"] = type(exc).__name__
                last_error = exc
                if retry < 2:
                    time.sleep(retry + 1)
                continue
            record_response(response, "beam_v8_single_finalize", observation)
            final_text = _strip_think(
                response.choices[0].message.content or "")
            _validate_final_response(response, final_text)
            finished = True
            termination_reason = final_phase
            break
        if not finished:
            if last_error:
                raise RuntimeError(
                    "tools-disabled final answer call failed") from last_error
            raise RuntimeError(
                f"retrieval did not produce a final answer in {max_rounds} rounds")
    if response is None:
        raise RuntimeError("retrieval produced no backend response")
    match = _validate_final_response(response, final_text)
    answer = match.group(1).strip()
    return {
        "answer": answer,
        "answer_format": "answer_tag",
        "memories": [{"text": text, "date": ""} for text in memories],
        "steps": steps,
        "tool_input_errors": tool_input_errors,
        "tool_trace": tool_trace,
        "interaction_trace": {
            "schema_version": INTERACTION_TRACE_SCHEMA_VERSION,
            "initial_prompt": prompt,
            "structure": structure,
            "tools": tools,
            "finalize_instruction": FINALIZE_INSTRUCTION,
            "responses": response_trace,
            "termination_reason": termination_reason,
        },
        "context_safety": {
            "policy_version": CONTEXT_POLICY_VERSION,
            "trace_schema_version": CONTEXT_TRACE_SCHEMA_VERSION,
            "tokenizer": tokenizer_identity,
            "local_request_token_limit": LOCAL_REQUEST_TOKEN_LIMIT,
            "total_tool_content_token_limit": TOTAL_TOOL_CONTENT_TOKEN_LIMIT,
            "per_tool_content_token_limit": PER_TOOL_CONTENT_TOKEN_LIMIT,
            "output_reservation_tokens": CONTEXT_OUTPUT_RESERVATION_TOKENS,
            "safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
            "raw_tool_tokens": raw_tool_tokens_total,
            "delivered_tool_tokens": delivered_tool_tokens_total,
            "truncated_tool_results": sum(
                bool(item["truncated"]) for item in tool_trace
            ),
            "tool_budget_exhausted": (
                delivered_tool_tokens_total >= TOTAL_TOOL_CONTENT_TOKEN_LIMIT
            ),
            "request_observations": request_observations,
            "request_local_tokens": [
                item["local_tokens"] for item in request_observations
            ],
            "max_request_local_tokens": max(
                item["local_tokens"] for item in request_observations
            ),
            "max_sent_request_local_tokens": max(
                item["local_tokens"]
                for item in request_observations
                if item["sent"]
            ),
            "rejected_request_candidates": sum(
                not item["sent"] for item in request_observations
            ),
            "context_compaction_used": context_compaction_used,
            "compact_finalization": compact_finalization,
        },
        "response_models": sorted(response_models),
        "response_ids": response_ids,
    }


def _memory_marker(memory_dir: Path) -> dict[str, Any] | None:
    path = memory_dir / BUILD_MARKER
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, dict) else None


def memory_tree_sha256(memory_dir: Path) -> tuple[int, str]:
    """Hash all persisted markdown paths and bytes in deterministic order."""
    digest = hashlib.sha256()
    paths = sorted(
        path for path in memory_dir.rglob("*.md")
        if "raw" not in path.relative_to(memory_dir).parts
    )
    for path in paths:
        digest.update(path.relative_to(memory_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return len(paths), digest.hexdigest()


def memory_payload_descriptor(memory_dir: Path) -> dict[str, Any]:
    """Hash the complete memory payload except the replaceable build marker."""
    root = memory_dir.resolve()
    files: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for path in sorted(memory_dir.rglob("*")):
        relative = path.relative_to(memory_dir).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"memory import contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"memory import contains a special filesystem node: {relative}")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"memory import path escapes its root: {relative}")
        stat = path.stat(follow_symlinks=False)
        if stat.st_nlink != 1:
            raise RuntimeError(f"memory import contains a hardlink: {relative}")
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            raise RuntimeError(f"memory import reuses an inode: {relative}")
        seen_inodes.add(inode)
        if relative == BUILD_MARKER:
            continue
        files.append({
            "path": relative,
            "size": stat.st_size,
            "sha256": file_sha256(path),
        })
    return {
        "file_count": len(files),
        "tree_sha256": stable_hash(files),
        "files": files,
    }


def _valid_memory_marker(memory_dir: Path, input_hash: str,
                         config_hash: str) -> dict[str, Any] | None:
    marker = _memory_marker(memory_dir)
    if not marker:
        return None
    if marker.get("schema_version") not in (1, 2):
        return None
    if (marker.get("input_hash") != input_hash
            or marker.get("config_hash") != config_hash):
        return None
    markdown_files, tree_hash = memory_tree_sha256(memory_dir)
    stats = marker.get("stats", {})
    if stats.get("status") != "complete":
        return None
    if (not isinstance(stats.get("chunks_total"), int)
            or stats.get("chunks_total", 0) <= 0
            or stats.get("chunks_with_events") != stats.get("chunks_total")):
        return None
    if markdown_files <= 0:
        return None
    if stats.get("markdown_files") != markdown_files:
        return None
    if stats.get("memory_sha256") != tree_hash:
        return None
    if marker.get("schema_version") == 2:
        provenance = marker.get("import_provenance")
        if (not isinstance(provenance, dict)
                or provenance.get("destination_payload_sha256")
                != memory_payload_descriptor(memory_dir).get("tree_sha256")):
            return None
    return marker


def _import_completed_memory_build(
    native: Any,
    conv: dict[str, Any],
    conv_dir: Path,
    input_hash: str,
    config_hash: str,
    source_memory: Path,
) -> dict[str, Any]:
    """Copy a completed build while preserving explicit source provenance."""
    source_memory = source_memory.expanduser().resolve()
    if (not source_memory.is_dir() or source_memory.is_symlink()
            or ROOT not in source_memory.parents):
        raise RuntimeError("build-import memory must be a regular repo directory")
    source_conv = source_memory.parent
    source_checkpoint_path = source_conv / "checkpoint.json"
    source_marker_path = source_memory / BUILD_MARKER
    if (not source_checkpoint_path.is_file()
            or source_checkpoint_path.is_symlink()
            or not source_marker_path.is_file()
            or source_marker_path.is_symlink()):
        raise RuntimeError("build-import source lacks checkpoint or marker")
    source_checkpoint = json.loads(
        source_checkpoint_path.read_text(encoding="utf-8"))
    source_marker = json.loads(source_marker_path.read_text(encoding="utf-8"))
    if source_marker.get("schema_version") != 1:
        raise RuntimeError("only an original schema-v1 build may be imported")
    if (source_marker.get("input_hash") != input_hash
            or source_checkpoint.get("input_hash") != input_hash):
        raise RuntimeError("build-import input identity mismatch")
    source_stats = source_marker.get("stats")
    source_build = source_checkpoint.get("build")
    if (not isinstance(source_stats, dict)
            or not isinstance(source_build, dict)
            or source_stats.get("status") != "complete"
            or source_build.get("status") != "complete"):
        raise RuntimeError("build-import source build is not complete")
    markdown_files, memory_hash = memory_tree_sha256(source_memory)
    if (markdown_files != source_stats.get("markdown_files")
            or memory_hash != source_stats.get("memory_sha256")
            or memory_hash != source_build.get("memory_sha256")):
        raise RuntimeError("build-import source memory hash mismatch")
    for key in (
        "events", "expected_dia_ids", "stored_dia_ids", "chunks_total",
        "chunks_with_events", "calls", "tokens_in", "tokens_out",
        "requested_models", "response_models", "finish_reasons",
    ):
        if source_stats.get(key) != source_build.get(key):
            raise RuntimeError(f"build-import marker/checkpoint mismatch: {key}")
    if (source_stats.get("requested_models") != ["gpt-5.5"]
            or source_stats.get("response_models") != ["gpt-5.5"]
            or source_stats.get("finish_reasons") != ["stop"]):
        raise RuntimeError("build-import model identity is invalid")
    expected = _expected_dia_ids(conv)
    stored = _stored_dia_ids(native, source_memory) & expected
    if (len(expected) != source_stats.get("expected_dia_ids")
            or len(stored) != source_stats.get("stored_dia_ids")):
        raise RuntimeError("build-import source-id coverage mismatch")
    current_hashes = source_fingerprints()
    source_hashes = source_checkpoint.get("config", {}).get(
        "source_sha256", {})
    for relative in (
        "src/adapters/run_nativemem.py", "src/nativemem.py",
        "src/v8_memory.py", "src/chatgpt_proxy.py",
    ):
        if source_hashes.get(relative) != current_hashes.get(relative):
            raise RuntimeError(
                f"build-import NativeMem source mismatch: {relative}")

    source_payload = memory_payload_descriptor(source_memory)
    destination = conv_dir / "memory"
    if destination.exists():
        raise RuntimeError("build-import destination memory already exists")
    staging = conv_dir / ".memory-importing"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source_memory, staging, symlinks=False)
    copied_payload = memory_payload_descriptor(staging)
    if copied_payload != source_payload:
        shutil.rmtree(staging)
        raise RuntimeError("build-import copy differs from source payload")
    provenance = {
        "schema": "nativemem.beam-build-import.v1",
        "source_memory": str(source_memory),
        "source_checkpoint": str(source_checkpoint_path.resolve()),
        "source_checkpoint_sha256": file_sha256(source_checkpoint_path),
        "source_marker_sha256": file_sha256(source_marker_path),
        "source_config_hash": source_marker.get("config_hash"),
        "source_runner_sha256": source_hashes.get(
            "scripts/run_v88_gpt55_beam.py"),
        "source_payload_sha256": source_payload["tree_sha256"],
        "destination_payload_sha256": copied_payload["tree_sha256"],
    }
    stats = dict(source_stats)
    stats.update({
        "reused_existing_build": True,
        "imported_existing_build": True,
        "import_provenance": provenance,
    })
    atomic_json(staging / BUILD_MARKER, {
        "schema_version": 2,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "completed_at": utc_now(),
        "import_provenance": provenance,
        "stats": stats,
    })
    os.replace(staging, destination)
    return stats


def _archive_path(path: Path, conv_dir: Path, label: str) -> None:
    if not path.exists():
        return
    partials = conv_dir / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    shutil.move(str(path), str(partials / f"{label}.{stamp}"))


def _archive_question_records(conv_dir: Path,
                              questions: dict[str, Any]) -> Path:
    partials = conv_dir / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = partials / f"questions-before-memory-rebuild.{stamp}.json"
    atomic_json(path, questions)
    return path


def _expected_dia_ids(conv: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    session_no = 1
    while f"session_{session_no}" in conv:
        out.update(str(t.get("dia_id")) for t in conv[f"session_{session_no}"]
                   if t.get("dia_id"))
        session_no += 1
    return out


def _stored_dia_ids(native: Any, memory_dir: Path) -> set[str]:
    out: set[str] = set()
    for path in memory_dir.rglob("*.md"):
        if "raw" in path.relative_to(memory_dir).parts:
            continue
        try:
            out.update(native.dia_ids_in(path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return out


def build_native_memory(native: Any, conv: dict[str, Any], conv_dir: Path,
                        input_hash: str, config_hash: str,
                        import_memory: Path | None = None) -> dict[str, Any]:
    """Build once and atomically publish ``memory/`` with a validation marker."""
    memory_dir = conv_dir / "memory"
    marker = _valid_memory_marker(memory_dir, input_hash, config_hash)
    if marker:
        reused = dict(marker.get("stats", {}))
        reused["reused_existing_build"] = True
        return reused
    if memory_dir.exists():
        _archive_path(memory_dir, conv_dir, "memory-without-valid-marker")
    if import_memory is not None and import_memory.exists():
        return _import_completed_memory_build(
            native, conv, conv_dir, input_hash, config_hash, import_memory)

    staging = conv_dir / ".memory-building"
    if staging.exists():
        # This path is created only by this adapter and cannot represent a
        # complete build (complete builds are atomically renamed to memory/).
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    native.tracker.reset("beam_build")
    started = time.monotonic()
    original_distill_call = native.v8_memory._distill_call
    original_distill_events = native.v8_memory.distill_events
    original_native_distill = getattr(native, "distill_events", None)
    chunk_audit: list[dict[str, int]] = []
    completion_resource = native.v8_memory.client.chat.completions
    original_completion_create = completion_resource.create
    requested_build_models: list[str] = []
    response_build_models: list[str] = []
    build_finish_reasons: list[str] = []
    build_refusals: list[str] = []

    def observed_completion_create(*call_args: Any,
                                   **call_kwargs: Any) -> Any:
        requested_build_models.append(str(call_kwargs.get("model", "") or ""))
        response = original_completion_create(*call_args, **call_kwargs)
        response_build_models.append(
            str(getattr(response, "model", "") or "").strip())
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        build_finish_reasons.append(str(
            getattr(choice, "finish_reason", "<missing>") or "<missing>"))
        message = getattr(choice, "message", None)
        build_refusals.append(str(
            getattr(message, "refusal", "") or "").strip())
        return response

    def strict_distill_call(*call_args: Any, **call_kwargs: Any) -> str:
        text = original_distill_call(*call_args, **call_kwargs)
        if text is None:
            phase = call_kwargs.get("phase", "v8_distill")
            raise RuntimeError(
                f"NativeMem model call exhausted retries during {phase}")
        return text

    def strict_distill_events(turns: Any, obs_date: Any, dia_ids: Any,
                              *call_args: Any, **call_kwargs: Any) -> Any:
        events = original_distill_events(
            turns, obs_date, dia_ids, *call_args, **call_kwargs)
        expected_chunk = {str(value) for value in dia_ids if value}
        referenced = {
            str(value)
            for event in (events or []) if isinstance(event, dict)
            for value in event.get("dia_ids", []) if value
        }
        matched = referenced & expected_chunk
        if not events:
            raise RuntimeError(
                "NativeMem distillation produced no event for a non-empty chunk")
        if not matched:
            raise RuntimeError(
                "NativeMem chunk events contain no dia_id from their source chunk")
        chunk_audit.append({
            "turns": len(expected_chunk),
            "events": len(events),
            "referenced_dia_ids": len(matched),
        })
        return events

    # The shared implementation treats an exhausted retry budget as an empty
    # event list so exploratory LoCoMo jobs can continue.  A resumable formal
    # BEAM checkpoint must not call that a successful build.  Override only for
    # this build invocation and restore the module immediately afterwards.
    native.v8_memory._distill_call = strict_distill_call
    native.v8_memory.distill_events = strict_distill_events
    completion_resource.create = observed_completion_create
    if original_native_distill is not None:
        native.distill_events = strict_distill_events
    try:
        build_time, event_count = native.build_memory(conv, str(staging))
        if event_count <= 0:
            raise RuntimeError("NativeMem build produced zero events")
        expected = _expected_dia_ids(conv)
        stored = _stored_dia_ids(native, staging)
        matched = stored & expected
        if not matched:
            raise RuntimeError(
                "NativeMem build contains no expected dia_id references")
        if not chunk_audit:
            raise RuntimeError("NativeMem build processed no non-empty chunks")
        expected_model = str(native.ALIYUN_MODEL)
        if (not requested_build_models
                or any(model != expected_model
                       for model in requested_build_models)):
            raise RuntimeError(
                "NativeMem build did not request only the configured model: "
                f"expected {expected_model!r}, got "
                f"{sorted(set(requested_build_models))}")
        if (not response_build_models
                or any(model != expected_model for model in response_build_models)):
            raise RuntimeError(
                "NativeMem build backend model mismatch: expected "
                f"{expected_model!r}, got "
                f"{sorted(set(response_build_models))}")
        if (not build_finish_reasons
                or any(reason != "stop" for reason in build_finish_reasons)):
            raise RuntimeError(
                "NativeMem build response did not finish normally: "
                f"{sorted(set(build_finish_reasons))}")
        if any(build_refusals):
            raise RuntimeError(
                "NativeMem build backend returned a refusal")
        markdown_files, memory_hash = memory_tree_sha256(staging)
        snapshot = native.tracker.snapshot("beam_build")
        stats = {
            "status": "complete",
            "build_time_s": round(build_time or (time.monotonic() - started), 3),
            "events": event_count,
            "markdown_files": markdown_files,
            "memory_sha256": memory_hash,
            "expected_dia_ids": len(expected),
            "stored_dia_ids": len(matched),
            "dia_id_coverage": round(len(matched) / len(expected), 6)
            if expected else 1.0,
            "chunks_total": len(chunk_audit),
            "chunks_with_events": len(chunk_audit),
            "chunk_audit": chunk_audit,
            "requested_models": sorted(set(requested_build_models)),
            "response_models": sorted(set(response_build_models)),
            "finish_reasons": sorted(set(build_finish_reasons)),
            "calls": snapshot.get("calls"),
            "tokens_in": snapshot.get("tokens_in"),
            "tokens_out": snapshot.get("tokens_out"),
            "llm_time_s": snapshot.get("llm_time_s"),
            "reused_existing_build": False,
        }
        atomic_json(staging / BUILD_MARKER, {
            "schema_version": 1,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "completed_at": utc_now(),
            "stats": stats,
        })
        os.replace(staging, memory_dir)
        return stats
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        native.v8_memory._distill_call = original_distill_call
        native.v8_memory.distill_events = original_distill_events
        completion_resource.create = original_completion_create
        if original_native_distill is not None:
            native.distill_events = original_native_distill


def _question_base(question: dict[str, Any], question_id: str, qi: int,
                   chat_size: str, conv_idx: int,
                   conversation_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_index": qi,
        "chat_size": chat_size,
        "conversation_index": conv_idx,
        "conversation_id": conversation_id,
        "question_type": question.get("question_type", "unknown"),
        "difficulty": question.get("difficulty", "unknown"),
        "question": question.get("question_text", ""),
        "gold_field": question.get("gold_field"),
        "gold": question.get("gold"),
        # Retain this compatibility alias while making the source field
        # explicit; BEAM does not use ideal_response for nine question types.
        "ideal_response": question.get("gold"),
        "rubric": list(question.get("rubric_nuggets", [])),
        "plan_reference": question.get("plan_reference"),
        "source_chat_ids": question.get("source_chat_ids", []),
        "abstention_type": question.get("abstention_type"),
        "why_unanswerable": question.get("why_unanswerable"),
    }


def _record_error(checkpoint: dict[str, Any], stage: str, exc: BaseException,
                  question_id: str | None = None) -> dict[str, Any]:
    value = {
        "time": utc_now(),
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
        "traceback": traceback.format_exc()[-8000:],
    }
    if question_id:
        value["question_id"] = question_id
    checkpoint.setdefault("errors", []).append(value)
    return value


def _result_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    questions = sorted(
        checkpoint.get("questions", {}).values(),
        key=lambda q: q.get("question_index", 10**9),
    )
    return {
        "metadata": {
            key: checkpoint.get(key)
            for key in (
                "schema_version", "benchmark", "method", "chat_size",
                "conversation_index", "conversation_id", "status",
                "created_at", "updated_at", "completed_at", "input_hash",
                "config_hash", "config", "source", "build",
            )
        },
        "questions": questions,
        "errors": checkpoint.get("errors", []),
    }


def save_conversation_state(conv_dir: Path,
                            checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_json(conv_dir / "checkpoint.json", checkpoint)
    atomic_json(conv_dir / "results.json", _result_payload(checkpoint))


def _load_or_initialize_checkpoint(
    conv_dir: Path,
    *,
    resume: bool,
    chat_size: str,
    conv_idx: int,
    conversation_id: str,
    input_hash: str,
    config: dict[str, Any],
    source: dict[str, Any],
    question_count: int,
) -> dict[str, Any]:
    config_hash = stable_hash(config)
    path = conv_dir / "checkpoint.json"
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"checkpoint exists at {path}; pass --resume or use a new output dir")
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("config_hash") != config_hash:
            raise RuntimeError(
                f"configuration mismatch for {chat_size}/{conv_idx}")
        if checkpoint.get("input_hash") != input_hash:
            raise RuntimeError(f"dataset input changed for {chat_size}/{conv_idx}")
        return checkpoint

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark": "BEAM",
        "method": "NativeMem-v8.8+calendar",
        "chat_size": chat_size,
        "conversation_index": conv_idx,
        "conversation_id": conversation_id,
        "status": "pending",
        "created_at": utc_now(),
        "input_hash": input_hash,
        "config_hash": config_hash,
        "config": config,
        "source": source,
        "question_count": question_count,
        "build": {"status": "pending"},
        "questions": {},
        "errors": [],
    }
    save_conversation_state(conv_dir, checkpoint)
    return checkpoint


def checkpoint_is_complete(checkpoint: dict[str, Any],
                           questions: list[dict[str, Any]],
                           memory_dir: Path | None = None) -> bool:
    if checkpoint.get("status") != "complete":
        return False
    if checkpoint.get("build", {}).get("status") != "complete":
        return False
    records = checkpoint.get("questions", {})
    if len(records) != len(questions):
        return False
    chat_size = checkpoint.get("chat_size")
    conv_idx = checkpoint.get("conversation_index")
    expected_ids = {
        f"{chat_size}_{conv_idx}_q{qi}_{q.get('question_type', 'unknown')}"
        for qi, q in enumerate(questions)
    }
    if set(records) != expected_ids:
        return False
    if memory_dir is None or not _valid_memory_marker(
            memory_dir,
            str(checkpoint.get("input_hash", "")),
            str(checkpoint.get("config_hash", ""))):
        return False
    memory_sha256 = checkpoint.get("build", {}).get("memory_sha256")
    if not memory_sha256:
        return False
    return all(
        record.get("status") == "complete"
        and str(record.get("answer", "")).strip()
        and record.get("memory_sha256") == memory_sha256
        for record in records.values()
    )


def run_conversation(
    native: Any,
    item: dict[str, Any],
    chat_size: str,
    conv_idx: int,
    out_dir: Path,
    args: argparse.Namespace,
    source: dict[str, Any],
) -> tuple[bool, str]:
    """Build and answer one conversation, persisting after every state change."""
    native_conv, input_stats = conversation_to_native(item)
    source_id_map = input_stats.pop("source_id_map")
    questions = extract_questions(item)
    if source.get("kind") == "huggingface":
        counts = {
            question_type: sum(
                q.get("question_type") == question_type for q in questions)
            for question_type in QUESTION_TYPES
        }
        if len(questions) != 20 or any(count != 2 for count in counts.values()):
            raise ValueError(
                "unexpected BEAM question inventory: expected two questions "
                f"for each of ten types, got {counts}")
        missing_rubrics = [
            q.get("question_text") for q in questions
            if not q.get("rubric_nuggets")
        ]
        if missing_rubrics:
            raise ValueError(
                f"{len(missing_rubrics)} BEAM questions have no rubric nuggets")
    source_id_map_hash = stable_hash(source_id_map)
    input_stats.update({
        "source_id_count": len(source_id_map),
        "source_id_occurrences": sum(len(values)
                                     for values in source_id_map.values()),
        "duplicate_source_id_count": sum(
            len(values) > 1 for values in source_id_map.values()),
        "source_id_map_sha256": source_id_map_hash,
    })
    input_hash = stable_hash({
        "conversation": conversation_digest(native_conv, questions),
        "source_id_map": source_id_map,
    })
    conversation_id = str(item.get(
        "conversation_id", f"{chat_size}_{conv_idx}"))
    config = frozen_method_config(args)
    config.update({
        "chat_size": chat_size,
        "conversation_index": conv_idx,
        "conversation_id": conversation_id,
        "conversation_seed": item.get("conversation_seed", {}),
        "input": input_stats,
    })
    conv_dir = out_dir / chat_size / f"conversation_{conv_idx:03d}"
    conv_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_or_initialize_checkpoint(
        conv_dir,
        resume=args.resume,
        chat_size=chat_size,
        conv_idx=conv_idx,
        conversation_id=conversation_id,
        input_hash=input_hash,
        config=config,
        source=source,
        question_count=len(questions),
    )
    source_map_path = conv_dir / "source_id_map.json"
    atomic_json(source_map_path, {
        "schema_version": 1,
        "description": (
            "BEAM conversation-level raw turn id to one or more NativeMem "
            "batch-local Dn:m references; duplicate raw ids remain explicit."
        ),
        "sha256": source_id_map_hash,
        "mapping": source_id_map,
    })
    if checkpoint_is_complete(checkpoint, questions, conv_dir / "memory"):
        # checkpoint.json is authoritative.  Re-materialize results.json so a
        # prior interruption between the two atomic writes cannot leave a
        # complete checkpoint paired with a stale/missing result file.
        save_conversation_state(conv_dir, checkpoint)
        return True, "already complete"

    checkpoint["status"] = "running"
    save_conversation_state(conv_dir, checkpoint)

    try:
        checkpoint["build"] = {
            **checkpoint.get("build", {}),
            "status": "running",
            "started_at": utc_now(),
        }
        save_conversation_state(conv_dir, checkpoint)
        build_kwargs: dict[str, Any] = {}
        reuse_root = getattr(args, "reuse_build_root", None)
        if reuse_root is not None:
            candidate = (
                reuse_root / chat_size / f"conversation_{conv_idx:03d}"
                / "memory"
            )
            if candidate.exists():
                build_kwargs["import_memory"] = candidate
        build_stats = build_native_memory(
            native, native_conv, conv_dir, input_hash,
            checkpoint["config_hash"], **build_kwargs)
        if (not build_stats.get("reused_existing_build", False)
                and checkpoint.get("questions")):
            archive = _archive_question_records(
                conv_dir, checkpoint["questions"])
            build_stats["invalidated_question_count"] = len(
                checkpoint["questions"])
            build_stats["invalidated_questions_archive"] = str(
                archive.relative_to(conv_dir))
            checkpoint["questions"] = {}
        checkpoint["build"] = {
            **build_stats,
            "status": "complete",
            "completed_at": utc_now(),
            "memory_dir": "memory",
        }
        save_conversation_state(conv_dir, checkpoint)
    except Exception as exc:  # noqa: BLE001
        error = _record_error(checkpoint, "build", exc)
        checkpoint["build"] = {
            **checkpoint.get("build", {}),
            "status": "failed",
            "failed_at": utc_now(),
            "error": error,
        }
        checkpoint["status"] = "failed"
        save_conversation_state(conv_dir, checkpoint)
        return False, f"build failed: {type(exc).__name__}: {exc}"

    turn_index = native.build_turn_index(native_conv)
    failed_questions: list[str] = []
    for qi, question in enumerate(questions):
        q_type = str(question.get("question_type", "unknown"))
        question_id = f"{chat_size}_{conv_idx}_q{qi}_{q_type}"
        previous = checkpoint.get("questions", {}).get(question_id, {})
        if (previous.get("status") == "complete"
                and str(previous.get("answer", "")).strip()
                and previous.get("memory_sha256")
                == checkpoint["build"].get("memory_sha256")):
            continue

        base = _question_base(
            question, question_id, qi, chat_size, conv_idx, conversation_id)
        base["memory_sha256"] = checkpoint["build"]["memory_sha256"]
        checkpoint.setdefault("questions", {})[question_id] = {
            **base,
            "status": "running",
            "started_at": utc_now(),
        }
        save_conversation_state(conv_dir, checkpoint)

        phase = f"beam_{chat_size}_{conv_idx}_q{qi}"
        native.tracker.reset(phase)
        started = time.monotonic()
        try:
            with native.tracker.bind_thread(phase):
                result = answer_question(
                    native,
                    str(question.get("question_text", "")),
                    conv_dir / "memory",
                    turn_index,
                )
            usage = native.tracker.snapshot(phase)
            checkpoint["questions"][question_id] = {
                **base,
                "status": "complete",
                "completed_at": utc_now(),
                "answer": result["answer"],
                "answer_format": result["answer_format"],
                "memories": result["memories"],
                "retrieval": {
                    "latency_s": round(time.monotonic() - started, 3),
                    "steps": result["steps"],
                    "calls": usage.get("calls"),
                    "tokens_in": usage.get("tokens_in"),
                    "tokens_out": usage.get("tokens_out"),
                    "llm_time_s": usage.get("llm_time_s"),
                    "response_models": result.get("response_models", []),
                    "response_ids": result.get("response_ids", []),
                    "tool_input_errors": result.get("tool_input_errors", []),
                    "tool_trace": result.get("tool_trace", []),
                    "interaction_trace": result.get("interaction_trace"),
                    "context_safety": result.get("context_safety"),
                },
            }
        except Exception as exc:  # noqa: BLE001
            error = _record_error(
                checkpoint, "question", exc, question_id=question_id)
            checkpoint["questions"][question_id] = {
                **base,
                "status": "failed",
                "failed_at": utc_now(),
                "error": error,
                "retrieval": {
                    "latency_s": round(time.monotonic() - started, 3),
                },
            }
            failed_questions.append(question_id)
        save_conversation_state(conv_dir, checkpoint)

    if failed_questions:
        checkpoint["status"] = "failed"
        checkpoint["failed_questions"] = failed_questions
        save_conversation_state(conv_dir, checkpoint)
        return False, f"{len(failed_questions)} question(s) failed"

    if not all(
        (record := checkpoint.get("questions", {}).get(
            f"{chat_size}_{conv_idx}_q{qi}_{q.get('question_type', 'unknown')}", {}
        )).get("status") == "complete"
        and str(record.get("answer", "")).strip()
        and record.get("memory_sha256") == checkpoint["build"].get(
            "memory_sha256")
        for qi, q in enumerate(questions)
    ):
        checkpoint["status"] = "failed"
        save_conversation_state(conv_dir, checkpoint)
        return False, "question checkpoint coverage mismatch"

    if not _valid_memory_marker(
            conv_dir / "memory", input_hash, checkpoint["config_hash"]):
        error = {
            "time": utc_now(),
            "stage": "finalize",
            "type": "RuntimeError",
            "message": "memory marker or content changed during QA",
            "traceback": "",
        }
        checkpoint.setdefault("errors", []).append(error)
        checkpoint["status"] = "failed"
        save_conversation_state(conv_dir, checkpoint)
        return False, error["message"]

    checkpoint["status"] = "complete"
    checkpoint["completed_at"] = utc_now()
    checkpoint.pop("failed_questions", None)
    save_conversation_state(conv_dir, checkpoint)
    return True, "complete"


def load_split(args: argparse.Namespace,
               chat_size: str) -> tuple[Any, dict[str, Any]]:
    """Load a selected HF split or a small local test fixture."""
    if args.fixture:
        payload = json.loads(args.fixture.read_text(encoding="utf-8"))
        rows = payload.get(chat_size) if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(
                f"fixture {args.fixture} has no list split {chat_size}")
        return rows, {
            "kind": "fixture",
            "path": str(args.fixture.resolve()),
            "split": chat_size,
            "rows": len(rows),
            "fingerprint": stable_hash(rows),
        }

    from datasets import load_dataset

    try:
        dataset = load_dataset(
            HF_DATASET,
            name="default",
            split=chat_size,
            cache_dir=str(args.dataset_cache_dir),
            revision=args.dataset_revision,
        )
    except ImportError as exc:
        if "socksio" in str(exc).lower():
            raise RuntimeError(
                "Hugging Face access inherited a SOCKS proxy but socksio is "
                "not installed; install httpx[socks] or unset ALL_PROXY, "
                "HTTP_PROXY, HTTPS_PROXY, and their lowercase variants for "
                "the dataset-loading process") from exc
        raise
    expected_rows = EXPECTED_SPLIT_ROWS[chat_size]
    if len(dataset) != expected_rows:
        raise ValueError(
            f"unexpected {chat_size} row count: expected {expected_rows}, "
            f"got {len(dataset)}")
    return dataset, {
        "kind": "huggingface",
        "dataset": HF_DATASET,
        "config": "default",
        "revision": args.dataset_revision,
        "split": chat_size,
        "rows": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NativeMem v8.8+calendar BEAM adapter (100K/1M, GPT-5.5, "
            "per-conversation checkpoints)"),
    )
    parser.add_argument(
        "--chat-sizes", required=True, type=parse_chat_sizes,
        help="comma-separated BEAM splits: 100K,1M")
    parser.add_argument(
        "--conversations", default="all",
        help="indices such as 0,2,5-7, or all")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed builds/questions from matching checkpoints")
    parser.add_argument(
        "--limit", type=int,
        help="maximum selected conversations per split (does not truncate QA)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--dataset-revision", default=DEFAULT_DATASET_REVISION,
        help="Hugging Face commit/revision (defaults to the audited commit)")
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--request-concurrency", type=int, default=2)
    parser.add_argument(
        "--reuse-build-root", type=Path,
        help=(
            "optional prior BEAM run root from which matching, completed, "
            "independently validated memory builds are copied; answers are "
            "never reused"
        ),
    )
    args = parser.parse_args(argv)
    args.model = "gpt-5.5"
    args.api_key = "x"
    args.formal_provider = args.fixture is None
    if args.formal_provider and args.gateway_root is None:
        parser.error("formal BEAM execution requires --gateway-root")
    if args.formal_provider and not args.allow_model_requests:
        parser.error("formal BEAM execution requires --allow-model-requests")
    if not args.formal_provider and (args.gateway_root or args.allow_model_requests):
        parser.error("offline --fixture mode cannot accept provider authorization")
    args.base_url = "offline-fixture-no-model-requests"
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.request_concurrency < 1:
        parser.error("--request-concurrency must be positive")
    # Validate syntax before any dataset download; range bounds are checked
    # after the selected split's actual row count is known.
    parse_indices(args.conversations)
    return args


def _save_preflight_failure_checkpoint(
    out_dir: Path,
    chat_size: str,
    conv_idx: int,
    item: dict[str, Any],
    args: argparse.Namespace,
    source: dict[str, Any],
    exc: BaseException,
) -> None:
    """Persist conversation-local failures that happen before normalization."""
    conv_dir = out_dir / chat_size / f"conversation_{conv_idx:03d}"
    checkpoint_path = conv_dir / "checkpoint.json"
    if checkpoint_path.exists():
        return
    config = frozen_method_config(args)
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark": "BEAM",
        "method": "NativeMem-v8.8+calendar",
        "chat_size": chat_size,
        "conversation_index": conv_idx,
        "conversation_id": str(item.get(
            "conversation_id", f"{chat_size}_{conv_idx}")),
        "status": "failed",
        "created_at": utc_now(),
        "config": config,
        "config_hash": stable_hash(config),
        "source": source,
        "build": {"status": "not_started"},
        "questions": {},
        "errors": [],
    }
    _record_error(checkpoint, "preflight", exc)
    save_conversation_state(conv_dir, checkpoint)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir.expanduser().resolve()
    args.reuse_build_manifest_sha256 = None
    if args.reuse_build_root is not None:
        reuse_root = args.reuse_build_root.expanduser().resolve()
        reuse_manifest = reuse_root / "run_manifest.json"
        if (reuse_root == out_dir or ROOT not in reuse_root.parents
                or not reuse_root.is_dir() or reuse_root.is_symlink()
                or not reuse_manifest.is_file()
                or reuse_manifest.is_symlink()):
            print(
                "--reuse-build-root must be a distinct regular BEAM run "
                "directory under this repository",
                file=sys.stderr,
            )
            return 2
        args.reuse_build_root = reuse_root
        args.reuse_build_manifest_sha256 = file_sha256(reuse_manifest)
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists() and not args.resume:
        print(
            f"manifest exists at {manifest_path}; pass --resume or use a new "
            "--output-dir",
            file=sys.stderr,
        )
        return 2
    if args.formal_provider and args.resume and manifest_path.exists():
        try:
            previous_provider = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("provider_evidence")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read existing manifest: {exc}", file=sys.stderr)
            return 2
        if (
            not isinstance(previous_provider, dict)
            or previous_provider.get("schema")
            != "openai-gpt55-flex-invocations/v1"
            or previous_provider.get("active_run_id") is not None
        ):
            print(
                "existing output is not a closed Flex-evidence run; use a new "
                "--output-dir",
                file=sys.stderr,
            )
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (out_dir / ".launcher.lock").open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"another launcher is already using {out_dir}",
            file=sys.stderr,
        )
        return 2
    invocation = None
    args.dataset_cache_dir = args.dataset_cache_dir.expanduser().resolve()

    method = frozen_method_config(args)
    fixture_identity = None
    if args.fixture:
        fixture_path = args.fixture.expanduser().resolve()
        fixture_identity = {
            "path": str(fixture_path),
            "sha256": file_sha256(fixture_path),
        }
        args.fixture = fixture_path
    run_fingerprint = stable_hash({
        "benchmark": "BEAM",
        "method": method,
        "fixture": fixture_identity,
    })
    selection_request = {
        "chat_sizes": args.chat_sizes,
        "conversations": args.conversations,
        "limit_per_split": args.limit,
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "BEAM",
        "method": method,
        "run_fingerprint": run_fingerprint,
        "git_commit": git_head(),
        "created_at": utc_now(),
        "status": "running",
        "selection": selection_request,
        "selection_history": [selection_request],
        "build_import": ({
            "root": str(args.reuse_build_root),
            "run_manifest_sha256": args.reuse_build_manifest_sha256,
        } if args.reuse_build_root is not None else None),
        "sources": {},
        "conversations": {},
        "errors": [],
    }
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("run_fingerprint") != run_fingerprint:
                print(
                    "existing manifest has a different method/code/dataset "
                    "configuration; use a new --output-dir",
                    file=sys.stderr,
                )
                return 2
            if previous.get("build_import") != manifest.get("build_import"):
                print(
                    "existing manifest has different build-import provenance; "
                    "use a new --output-dir",
                    file=sys.stderr,
                )
                return 2
            manifest["created_at"] = previous.get(
                "created_at", manifest["created_at"])
            manifest["conversations"].update(
                previous.get("conversations", {}))
            manifest["errors"].extend(previous.get("errors", []))
            manifest["sources"].update(previous.get("sources", {}))
            manifest["selection_history"] = [
                *previous.get("selection_history", [
                    previous.get("selection", {})]),
                selection_request,
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"cannot resume manifest {manifest_path}: {exc}", file=sys.stderr)
            return 2
    if args.formal_provider:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src import openai_gpt55_flex_gateway_evidence as flex_evidence

        invocation = flex_evidence.begin_child_invocation(
            args.gateway_root,
            out_dir / "provider_evidence" / (
                f"invocation-{utc_now().replace(':', '')}"
            ),
        )
        args.base_url = invocation.base_url
        os.environ["CHATGPT_PROXY_LOG"] = str(invocation.log_path)
        prior_invocations = (
            previous["provider_evidence"]["invocations"]
            if manifest_path.exists() else []
        )
        manifest["provider_evidence"] = {
            "schema": "openai-gpt55-flex-invocations/v1",
            "gateway_root": str(args.gateway_root.expanduser().resolve()),
            "active_run_id": invocation.run_id,
            "invocations": prior_invocations,
        }
    atomic_json(manifest_path, manifest)

    try:
        native = load_native(args)
    except Exception as exc:  # noqa: BLE001
        manifest["errors"].append({
            "time": utc_now(), "stage": "backend_init",
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()[-8000:],
        })
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        if invocation is not None:
            try:
                evidence_record = invocation.finish()
                manifest["provider_evidence"]["active_run_id"] = None
                manifest["provider_evidence"]["invocations"].append(
                    evidence_record
                )
            except Exception as evidence_exc:  # noqa: BLE001
                invocation.abort()
                manifest["errors"].append({
                    "time": utc_now(), "stage": "provider_evidence",
                    "type": type(evidence_exc).__name__,
                    "message": str(evidence_exc),
                })
        atomic_json(manifest_path, manifest)
        return 1
    failures: set[str] = set(manifest.get("failed_conversations", []))
    failures.update(
        key for key, record in manifest.get("conversations", {}).items()
        if record.get("status") != "complete"
    )
    for chat_size in args.chat_sizes:
        try:
            rows, source = load_split(args, chat_size)
            previous_source = manifest["sources"].get(chat_size)
            if previous_source is not None and previous_source != source:
                raise RuntimeError(
                    f"dataset source changed for {chat_size}; use a new "
                    "--output-dir")
            manifest["sources"][chat_size] = source
            failures.discard(f"{chat_size}:dataset")
            atomic_json(manifest_path, manifest)
            requested = parse_indices(args.conversations, size=len(rows))
            indices = (list(range(len(rows))) if requested is None
                       else requested)
            if args.limit is not None:
                indices = indices[:args.limit]
            if not indices:
                raise ValueError(
                    f"no selected conversation exists in {chat_size} "
                    f"(split has {len(rows)} rows)")
        except Exception as exc:  # noqa: BLE001
            key = f"{chat_size}:dataset"
            failures.add(key)
            manifest["errors"].append({
                "time": utc_now(), "stage": "dataset", "chat_size": chat_size,
                "type": type(exc).__name__, "message": str(exc),
            })
            atomic_json(manifest_path, manifest)
            continue

        for conv_idx in indices:
            key = f"{chat_size}:{conv_idx}"
            item: dict[str, Any] = {}
            try:
                item = dict(rows[conv_idx])
                ok, detail = run_conversation(
                    native, item, chat_size, conv_idx, out_dir, args, source)
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = f"{type(exc).__name__}: {exc}"
                _save_preflight_failure_checkpoint(
                    out_dir, chat_size, conv_idx, item, args, source, exc)
                manifest["errors"].append({
                    "time": utc_now(), "stage": "conversation",
                    "conversation": key, "type": type(exc).__name__,
                    "message": str(exc),
                })
            manifest["conversations"][key] = {
                "status": "complete" if ok else "failed",
                "detail": detail,
                "updated_at": utc_now(),
                "checkpoint": str(
                    (out_dir / chat_size / f"conversation_{conv_idx:03d}"
                     / "checkpoint.json").relative_to(ROOT)
                ) if out_dir.is_relative_to(ROOT) else str(
                    out_dir / chat_size / f"conversation_{conv_idx:03d}"
                    / "checkpoint.json"),
            }
            if not ok:
                failures.add(key)
            else:
                failures.discard(key)
            atomic_json(manifest_path, manifest)
            print(f"{key}: {'complete' if ok else 'failed'} ({detail})",
                  flush=True)

    if invocation is not None:
        try:
            evidence_record = invocation.finish()
        except Exception as exc:  # noqa: BLE001
            invocation.abort()
            failures.add("provider_evidence")
            manifest["errors"].append({
                "time": utc_now(), "stage": "provider_evidence",
                "type": type(exc).__name__, "message": str(exc),
            })
        else:
            manifest["provider_evidence"]["active_run_id"] = None
            manifest["provider_evidence"]["invocations"].append(evidence_record)
    manifest["status"] = "failed" if failures else "complete"
    manifest["failed_conversations"] = sorted(failures)
    manifest["finished_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
