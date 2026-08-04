#!/usr/bin/env python3
"""Prepare or execute the frozen R115 BEAM controlled rows.

Formal execution is deliberately difficult to start accidentally.  It requires
the exact 45-conversation selection, one of the three preregistered conditions,
an explicit model-call acknowledgement, a fresh or exactly resumable output
root, and a validated OpenAI GPT-5.5 Flex gateway result root.  The loopback
URL is derived from the signed gateway-ready artifact and is never accepted as
a free-form external URL.
Importing this module and the ``preregister`` command perform no network or
model requests.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Mapping, Protocol


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from baselines.beam_controls import r115_beam_control_contract as contract  # noqa: E402
from baselines.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402
from scripts.evaluation import durable_model_ledger as durable  # noqa: E402
from scripts.evaluation import visible_token_budget as visible  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/paper-experiments-20260714/r115-beam-controls"
DEFAULT_PREREGISTRATION = (
    ROOT / "paper/refine-logs/R115_BEAM_CONTROLS_PREREGISTRATION.json"
)
DEFAULT_PREFLIGHT = (
    ROOT / "results/paper-experiments-20260714/r115-formal-preflight.json"
)


class RunError(RuntimeError):
    """Raised when an R115 run cannot preserve the frozen contract."""


class Backend(Protocol):
    def operation(self, operation_id: str) -> ContextManager[None]: ...

    def ingest(self, session: Mapping[str, Any]) -> dict[str, Any]: ...

    def retrieve(self, question: str) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


def _flatten_turns(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return output
    for item in value:
        if isinstance(item, dict):
            output.append(dict(item))
        elif isinstance(item, list):
            output.extend(_flatten_turns(item))
    return output


def parse_chat(chat: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(chat, list) or not chat:
        raise RunError("BEAM chat is empty")
    first = chat[0]
    if isinstance(first, list):
        batches = [_flatten_turns(value) for value in chat]
    elif isinstance(first, dict) and "turns" in first:
        batches = [_flatten_turns(value.get("turns")) for value in chat]
    elif isinstance(first, dict) and ("role" in first or "content" in first):
        batches = [_flatten_turns(chat)]
    else:
        raise RunError("BEAM chat layout is unsupported for R115")
    if not batches or any(not batch for batch in batches):
        raise RunError("BEAM chat contains an empty batch")
    return batches


def normalize_conversation(
    item: Mapping[str, Any],
    *,
    chat_size: str,
    conversation_index: int,
    require_formal_inventory: bool = True,
) -> dict[str, Any]:
    batches = parse_chat(item.get("chat"))
    sessions: list[dict[str, Any]] = []
    source_id_map: dict[str, list[str]] = {}
    seen_dia_ids: set[str] = set()
    for session_index, batch in enumerate(batches):
        turns: list[dict[str, Any]] = []
        dates: list[str] = []
        for raw in batch:
            text = str(raw.get("content", "") or "").strip()
            if not text:
                continue
            anchor = raw.get("time_anchor")
            if anchor:
                normalized_date = contract.normalize_time_anchor(anchor)
                if normalized_date not in dates:
                    dates.append(normalized_date)
            role = str(raw.get("role", "user") or "user").strip().lower()
            if role in {"human", "customer"}:
                role = "user"
            elif role in {"ai", "bot"}:
                role = "assistant"
            if role not in {"user", "assistant", "system", "tool"}:
                raise RunError(f"unsupported BEAM role: {role}")
            dia_id = f"D{session_index + 1}:{len(turns) + 1}"
            if dia_id in seen_dia_ids:
                raise RunError(f"duplicate normalized turn id: {dia_id}")
            seen_dia_ids.add(dia_id)
            source_ids = []
            if raw.get("id") is not None:
                source_id = str(raw["id"])
                source_ids.append(source_id)
                source_id_map.setdefault(source_id, []).append(dia_id)
            turns.append(
                {
                    "dia_id": dia_id,
                    "source_chat_ids": source_ids,
                    "role": role,
                    "text": text,
                    "raw_turn_sha256": contract.canonical_hash(dict(raw)),
                }
            )
        if not turns:
            raise RunError(f"BEAM session {session_index} has no nonempty turns")
        if not dates:
            raise RunError(f"BEAM session {session_index} has no time anchor")
        rendered_turns = [
            render_turn(
                chat_size=chat_size,
                conversation_index=conversation_index,
                session_index=session_index,
                date=dates[0],
                turn=turn,
            )
            for turn in turns
        ]
        sessions.append(
            {
                "session_index": session_index,
                "date": dates[0],
                "all_dates": dates,
                "turns": turns,
                "source_chat_ids": contract.dedupe(
                    source_id
                    for turn in turns
                    for source_id in turn["source_chat_ids"]
                ),
                "document_sha256": contract.sha256_text("\n\n".join(rendered_turns)),
            }
        )
    questions = contract.extract_questions(
        item, require_formal_inventory=require_formal_inventory
    )
    value = {
        "chat_size": chat_size,
        "conversation_index": conversation_index,
        "conversation_id": str(
            item.get("conversation_id", f"{chat_size}_{conversation_index}")
        ),
        "sessions": sessions,
        "source_id_map": source_id_map,
        "questions": questions,
    }
    value["normalized_conversation_sha256"] = contract.canonical_hash(value)
    return value


def render_turn(
    *,
    chat_size: str,
    conversation_index: int,
    session_index: int,
    date: str,
    turn: Mapping[str, Any],
) -> str:
    source_ids = ",".join(str(value) for value in turn["source_chat_ids"]) or "none"
    return (
        f"[BEAM {chat_size} conversation={conversation_index:03d} "
        f"session={session_index:04d} date={date} "
        f"dia_id={turn['dia_id']} source_chat_ids={source_ids} "
        f"role={turn['role']}]\n{turn['text']}"
    )


def render_full_context(conversation: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    for session in conversation["sessions"]:
        for turn in session["turns"]:
            blocks.append(
                render_turn(
                    chat_size=str(conversation["chat_size"]),
                    conversation_index=int(conversation["conversation_index"]),
                    session_index=int(session["session_index"]),
                    date=str(session["date"]),
                    turn=turn,
                )
            )
    return "\n\n".join(blocks)


def turn_chunks(conversation: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for session in conversation["sessions"]:
        for turn in session["turns"]:
            text = render_turn(
                chat_size=str(conversation["chat_size"]),
                conversation_index=int(conversation["conversation_index"]),
                session_index=int(session["session_index"]),
                date=str(session["date"]),
                turn=turn,
            )
            output.append(
                {
                    "chunk_id": str(turn["dia_id"]),
                    "session_index": int(session["session_index"]),
                    "date": str(session["date"]),
                    "source_chat_ids": list(turn["source_chat_ids"]),
                    "text": text,
                    "text_sha256": contract.sha256_text(text),
                }
            )
    return output


def build_bm25_index(
    chunks: list[dict[str, Any]],
) -> contract.FrozenBM25Index:
    return contract.FrozenBM25Index(
        [str(value["text"]) for value in chunks]
    )


def rank_bm25(
    question: str,
    chunks: list[dict[str, Any]],
    index: contract.FrozenBM25Index | None = None,
) -> list[dict[str, Any]]:
    frozen_index = index or build_bm25_index(chunks)
    scores = frozen_index.scores(question)
    ranked = sorted(
        zip(chunks, scores), key=lambda value: (-value[1], value[0]["chunk_id"])
    )[: contract.BM25_TOP_K]
    return [
        {**dict(chunk), "rank": rank, "score": score}
        for rank, (chunk, score) in enumerate(ranked)
    ]


def load_mapping_records(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = contract.read_jsonl(path)
    selected = [row for row in rows if row.get("benchmark") == "BEAM"]
    if len(selected) != contract.FORMAL_QUESTIONS:
        raise RunError("R002 mapping does not contain exactly 900 BEAM questions")
    mapping: dict[str, dict[str, Any]] = {}
    for row in selected:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in mapping:
            raise RunError("R002 mapping has a missing or duplicate BEAM question_id")
        mapping[question_id] = row
    return mapping, {
        "path": str(path.resolve()),
        "sha256": contract.sha256_file(path),
        "rows": len(selected),
    }


def load_splits(
    *, fixture: Path | None, cache_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if fixture is not None:
        payload = contract.read_json(fixture)
        if not isinstance(payload, dict):
            raise RunError("BEAM fixture is not an object")
        sources: dict[str, dict[str, Any]] = {}
        splits: dict[str, Any] = {}
        for chat_size in ("100K", "1M"):
            rows = payload.get(chat_size, [])
            if not isinstance(rows, list):
                raise RunError(f"fixture split {chat_size} is not a list")
            splits[chat_size] = rows
            sources[chat_size] = {
                "kind": "fixture",
                "path": str(fixture.resolve()),
                "sha256": contract.sha256_file(fixture),
                "rows": len(rows),
                "fingerprint": contract.canonical_hash(rows),
            }
        return splits, sources

    if cache_dir.expanduser().resolve() != contract.HF_CACHE.resolve():
        raise RunError("formal R115 uses only the frozen BEAM Arrow cache root")
    import pyarrow  # type: ignore[import-not-found]
    import pyarrow.ipc  # type: ignore[import-not-found]

    if contract.sha256_file(contract.DATASET_INFO_PATH) != contract.DATASET_INFO_SHA256:
        raise RunError("BEAM dataset_info.json hash differs")
    splits = {}
    sources = {}
    for chat_size in ("100K", "1M"):
        arrow_path = contract.ARROW_FILES[chat_size]
        if contract.sha256_file(arrow_path) != contract.ARROW_SHA256[chat_size]:
            raise RunError(f"BEAM {chat_size} Arrow hash differs")
        with pyarrow.memory_map(str(arrow_path), "r") as source:
            table = pyarrow.ipc.open_stream(source).read_all()
        schema_sha256 = contract.sha256_bytes(table.schema.serialize().to_pybytes())
        if schema_sha256 != contract.ARROW_SCHEMA_SHA256:
            raise RunError(f"BEAM {chat_size} Arrow schema differs")
        if table.num_rows != contract.EXPECTED_SPLIT_ROWS[chat_size]:
            raise RunError(f"BEAM {chat_size} Arrow row count differs")
        rows = table.to_pylist()
        splits[chat_size] = rows
        sources[chat_size] = {
            "kind": "pinned_huggingface_arrow_ipc",
            "dataset": contract.HF_DATASET,
            "config": contract.HF_CONFIG,
            "revision": contract.HF_REVISION,
            "split": chat_size,
            "path": str(arrow_path.resolve()),
            "sha256": contract.ARROW_SHA256[chat_size],
            "schema_sha256": schema_sha256,
            "rows": len(rows),
            "dataset_info_path": str(contract.DATASET_INFO_PATH.resolve()),
            "dataset_info_sha256": contract.DATASET_INFO_SHA256,
        }
    return splits, sources


def selected_indices(splits: Mapping[str, Any], *, fixture: bool) -> dict[str, list[int]]:
    if not fixture:
        return {key: list(value) for key, value in contract.FORMAL_SELECTION.items()}
    return {
        chat_size: list(range(len(splits.get(chat_size, []))))
        for chat_size in ("100K", "1M")
        if len(splits.get(chat_size, []))
    }


def build_inventory(
    *,
    splits: Mapping[str, Any],
    selections: Mapping[str, list[int]],
    mapping: Mapping[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    conversations: dict[str, dict[str, Any]] = {}
    for chat_size in ("100K", "1M"):
        for conversation_index in selections.get(chat_size, []):
            conversation = normalize_conversation(
                splits[chat_size][conversation_index],
                chat_size=chat_size,
                conversation_index=conversation_index,
                require_formal_inventory=mapping is not None,
            )
            conversation_key = f"{chat_size}:c{conversation_index:03d}"
            conversations[conversation_key] = conversation
            for question_index, question in enumerate(conversation["questions"]):
                question_id = contract.question_id(
                    chat_size,
                    conversation_index,
                    question_index,
                    str(question["question_type"]),
                )
                r002 = mapping.get(question_id) if mapping is not None else None
                if mapping is not None:
                    if not isinstance(r002, dict):
                        raise RunError(f"R002 mapping misses {question_id}")
                    exact = {
                        "chat_size": chat_size,
                        "conversation_index": conversation_index,
                        "question_index": question_index,
                        "question_type": question["question_type"],
                        "question": question["question_text"],
                        "normalized_source_ids": question[
                            "normalized_source_chat_ids"
                        ],
                    }
                    for key, expected in exact.items():
                        if r002.get(key) != expected:
                            raise RunError(f"R002 field {key} differs for {question_id}")
                inventory.append(
                    {
                        "question_id": question_id,
                        "chat_size": chat_size,
                        "conversation_index": conversation_index,
                        "conversation_id": conversation["conversation_id"],
                        "question_index": question_index,
                        "question_type": question["question_type"],
                        "question": question["question_text"],
                        "gold_field": question["gold_field"],
                        "gold": question["gold"],
                        "rubric_nuggets": question["rubric_nuggets"],
                        "raw_source_chat_ids": question["raw_source_chat_ids"],
                        "gold_source_ids": (
                            r002["normalized_source_ids"] if r002 else []
                        ),
                        "source_recall_eligible": bool(
                            r002 and r002["source_recall_eligible"]
                        ),
                        "source_recall_exclusion_reasons": (
                            r002["source_recall_exclusion_reasons"] if r002 else []
                        ),
                        "normalized_conversation_sha256": conversation[
                            "normalized_conversation_sha256"
                        ],
                        "r002_record_sha256": (
                            contract.canonical_hash(r002) if r002 else None
                        ),
                    }
                )
    return inventory, conversations


def _dataset_files_from_mapping_manifest() -> dict[str, Any]:
    value = contract.read_json(contract.EVIDENCE_MAPPING_MANIFEST)
    try:
        source = value["benchmarks"]["BEAM"]["source"]
    except (KeyError, TypeError) as exc:
        raise RunError("R002 mapping manifest lacks BEAM source identity") from exc
    if (
        source.get("dataset") != contract.HF_DATASET
        or source.get("config") != contract.HF_CONFIG
        or source.get("revision") != contract.HF_REVISION
    ):
        raise RunError("R002 mapping manifest BEAM dataset identity differs")
    for descriptor in source.get("files", {}).values():
        path = ROOT / descriptor["path"]
        if contract.sha256_file(path) != descriptor["sha256"]:
            raise RunError(f"pinned BEAM cache file differs: {path}")
    return source


def create_preregistration(output_path: Path) -> dict[str, Any]:
    source = _dataset_files_from_mapping_manifest()
    audit = contract.read_json(contract.EVIDENCE_MAPPING_AUDIT)
    if audit.get("status") != "passed":
        raise RunError("R002 mapping audit has not passed")
    payload: dict[str, Any] = {
        "schema_version": contract.PREREG_SCHEMA,
        "status": "frozen",
        "frozen_at": contract.utc_now(),
        "benchmark": "BEAM",
        "milestone": "R115",
        "scope_note": "limited controls for the requested BEAM scale evaluation",
        "formal_methods": list(contract.FORMAL_METHODS),
        "formal_selection": contract.FORMAL_SELECTION,
        "formal_conversation_count": contract.FORMAL_CONVERSATIONS,
        "formal_question_count": contract.FORMAL_QUESTIONS,
        "dataset": source,
        "dataset_arrow_contract": {
            "loader": "pyarrow.ipc.open_stream(memory_map).read_all",
            "schema_sha256": contract.ARROW_SCHEMA_SHA256,
            "files": {
                chat_size: {
                    "path": str(path.resolve()),
                    "sha256": contract.ARROW_SHA256[chat_size],
                    "rows": contract.EXPECTED_SPLIT_ROWS[chat_size],
                }
                for chat_size, path in contract.ARROW_FILES.items()
            },
            "dataset_info": {
                "path": str(contract.DATASET_INFO_PATH.resolve()),
                "sha256": contract.DATASET_INFO_SHA256,
            },
        },
        "r002": {
            "manifest_path": str(contract.EVIDENCE_MAPPING_MANIFEST.resolve()),
            "manifest_sha256": contract.sha256_file(
                contract.EVIDENCE_MAPPING_MANIFEST
            ),
            "audit_path": str(contract.EVIDENCE_MAPPING_AUDIT.resolve()),
            "audit_sha256": contract.sha256_file(contract.EVIDENCE_MAPPING_AUDIT),
            "questions_path": str(contract.EVIDENCE_MAPPING_QUESTIONS.resolve()),
            "questions_sha256": contract.sha256_file(
                contract.EVIDENCE_MAPPING_QUESTIONS
            ),
            "source_granularity": "turn",
            "source_recall_denominator": 804,
        },
        "requested_model": contract.REQUESTED_MODEL,
        "answer_protocol": {
            "temperature": 0,
            "max_completion_tokens": contract.ANSWER_MAX_TOKENS,
            "prompt_template": "scripts.r115_beam_control_contract.ANSWER_PROMPT",
            "prompt_template_sha256": contract.sha256_text(contract.ANSWER_PROMPT),
            "answer_tag_parser": "single_nonempty_answer_element_v1",
        },
        "context_policy": {
            "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
            "answer_completion_reservation_tokens": contract.ANSWER_MAX_TOKENS,
            "max_rendered_prompt_tokens": contract.MAX_RENDERED_PROMPT_TOKENS,
            "full_context": contract.FULL_CONTEXT_POLICY,
            "full_context_truncation_allowed": False,
            "full_context_blocked_records_remain_in_inventory": True,
            "blocked_denominator_policy": {
                "all_900": "blocked records count as no-answer/incorrect",
                "secondary": "also report answered-only accuracy and block rate",
            },
            "exact_prompt_counting": {
                "version": contract.EXACT_PROMPT_COUNT_VERSION,
                "equivalence": (
                    "encode_with_unstable freezes only tokens stable under every "
                    "question continuation; the residual plus each serialized "
                    "question suffix is then encoded exactly"
                ),
            },
            "hard_visible_total_methods": ["bm25", "mem0"],
            "hard_visible_total_tokens": contract.VISIBLE_BUDGET_TOKENS,
            "tokenizer": {
                "package": "tiktoken",
                "version": contract.TIKTOKEN_VERSION,
                "encoding": contract.TOKEN_ENCODING,
                "provider_exact": False,
            },
        },
        "conditions": {
            "full_context": {
                "render_version": contract.RENDER_VERSION,
                "unit": "all nonempty raw conversation turns",
                "overflow": contract.BLOCKED_STATUS,
                "truncation": "forbidden",
            },
            "bm25": {
                "version": contract.BM25_VERSION,
                "index_version": contract.BM25_INDEX_VERSION,
                "chunking": "one nonempty raw turn",
                "top_k": contract.BM25_TOP_K,
                "k1": contract.BM25_K1,
                "b": contract.BM25_B,
                "token_pattern": contract.BM25_TOKEN_PATTERN.pattern,
                "source_resolution": "exact selected raw turns",
            },
            "mem0": {
                "surface": contract.MEM0_SURFACE_VERSION,
                "package": "mem0ai",
                "version": contract.MEM0_VERSION,
                "top_k": contract.MEM0_TOP_K,
                "workspace": "one isolated workspace per BEAM conversation",
                "ingestion": "one add call per BEAM session with all raw turns",
                "source_resolution": "all raw turns bound to returned memory metadata",
                "embedder": contract.EMBEDDING_MODEL,
                "embedding_dims": contract.EMBEDDING_DIMS,
            },
        },
        "structured_control_decision": {
            "method": "Graphiti OSS",
            "status": "excluded_before_formal_answers",
            "reason": (
                "The frozen LoCoMo/LongMemEval Graphiti surface binds each edge "
                "to session episodes. BEAM R002 requires turn-level sources. "
                "Changing Graphiti to one episode per BEAM turn would not be the "
                "same preregistered surface; retaining session episodes cannot "
                "audit turn-level delivered evidence."
            ),
            "claim_boundary": "no BEAM Graphiti result or comparison",
        },
        "execution_gate": {
            "formal_model_calls_require_explicit_allow": True,
            "base_url": "derived_only_from_validated_Flex_gateway_root",
            "provider_contract": (
                "baselines.gateways.openai_gpt55_flex_gateway_evidence.active_contract"
            ),
            "exclusive_prefix_window": (
                "capture_start/capture_end plus exact consumer request IDs"
            ),
            "formal_selection_must_equal_all_45": True,
            "resume": "only exact identity with no ledger orphan or failure",
            "formal_scoring": "separate existing project-profile evaluator",
        },
        "model_calls": 0,
        "network_calls": 0,
        "supersedes_interrupted_preflight": {
            "status": "no_artifact_written",
            "reason": (
                "The earlier local preflight was interrupted while repeatedly "
                "encoding the same 1M context once per question. It made zero "
                "model and network calls."
            ),
        },
    }
    payload["preregistration_content_sha256"] = contract.content_hash(
        payload, "preregistration_content_sha256"
    )
    contract.atomic_json_no_clobber(output_path, payload)
    return payload


def create_formal_preflight(
    *, preregistration_path: Path, output_path: Path, cache_dir: Path
) -> dict[str, Any]:
    """Reconstruct every formal input without instantiating a model client."""

    preregistration = contract.validate_preregistration(preregistration_path)
    dependencies = contract.dependency_snapshot()
    tokenizer = visible.TokenCounter.resolve(
        requested_model=contract.REQUESTED_MODEL,
        fallback_encoding=contract.TOKEN_ENCODING,
        allow_byte_fallback=False,
    )
    if (
        tokenizer.identity.get("implementation_version") != contract.TIKTOKEN_VERSION
        or tokenizer.identity.get("encoding_name") != contract.TOKEN_ENCODING
    ):
        raise RunError("R115 preflight tokenizer identity differs")
    splits, sources = load_splits(fixture=None, cache_dir=cache_dir)
    mapping, mapping_descriptor = load_mapping_records(
        contract.EVIDENCE_MAPPING_QUESTIONS
    )
    inventory, conversations = build_inventory(
        splits=splits,
        selections=contract.FORMAL_SELECTION,
        mapping=mapping,
    )
    if len(inventory) != contract.FORMAL_QUESTIONS:
        raise RunError("R115 formal preflight did not reconstruct 900 questions")
    records: list[dict[str, Any]] = []
    for conversation_key, conversation in conversations.items():
        full_context = render_full_context(conversation)
        prompt_token_count, prompt_counter = exact_prompt_token_counter(
            tokenizer, full_context
        )
        chunks = turn_chunks(conversation)
        bm25_index = build_bm25_index(chunks)
        items = [
            item
            for item in inventory
            if item["chat_size"] == conversation["chat_size"]
            and item["conversation_index"] == conversation["conversation_index"]
        ]
        questions = []
        for item in items:
            prompt_tokens = prompt_token_count(str(item["question"]))
            ranked = rank_bm25(
                str(item["question"]), chunks, index=bm25_index
            )
            questions.append(
                {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "r002_record_sha256": item["r002_record_sha256"],
                    "source_recall_eligible": item["source_recall_eligible"],
                    "full_context_rendered_prompt_tokens": prompt_tokens,
                    "full_context_fit": (
                        prompt_tokens <= contract.MAX_RENDERED_PROMPT_TOKENS
                    ),
                    "bm25_ranked_chunk_ids": [row["chunk_id"] for row in ranked],
                    "bm25_ranked_records_sha256": contract.canonical_hash(ranked),
                }
            )
        records.append(
            {
                "conversation_key": conversation_key,
                "normalized_conversation_sha256": conversation[
                    "normalized_conversation_sha256"
                ],
                "sessions": len(conversation["sessions"]),
                "turns": len(chunks),
                "source_id_count": len(conversation["source_id_map"]),
                "full_context_sha256": contract.sha256_text(full_context),
                "full_context_tokens": tokenizer.count(full_context),
                "full_context_prompt_counter": prompt_counter,
                "questions": questions,
            }
        )
    flat_questions = [question for record in records for question in record["questions"]]
    payload: dict[str, Any] = {
        "schema_version": contract.PREFLIGHT_SCHEMA,
        "status": "passed",
        "benchmark": "BEAM",
        "milestone": "R115",
        "formal_methods": list(contract.FORMAL_METHODS),
        "formal_selection": contract.FORMAL_SELECTION,
        "conversation_count": len(records),
        "question_count": len(flat_questions),
        "full_context_fit_count": sum(
            question["full_context_fit"] for question in flat_questions
        ),
        "full_context_blocked_over_context_count": sum(
            not question["full_context_fit"] for question in flat_questions
        ),
        "source_recall_denominator": sum(
            question["source_recall_eligible"] for question in flat_questions
        ),
        "dataset_sources": sources,
        "mapping": mapping_descriptor,
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": contract.sha256_file(preregistration_path),
            "content_sha256": preregistration[
                "preregistration_content_sha256"
            ],
        },
        "dependencies": dependencies,
        "tokenizer": tokenizer.identity,
        "source_hashes": contract.source_hashes(),
        "inventory_sha256": contract.canonical_hash(inventory),
        "conversation_records": records,
        "graphiti": preregistration["structured_control_decision"],
        "execution_status": "not_started",
        "scoring_status": "not_started",
        "model_calls": 0,
        "network_calls": 0,
    }
    payload["preflight_content_sha256"] = contract.content_hash(
        payload, "preflight_content_sha256"
    )
    contract.atomic_json_no_clobber(output_path, payload)
    return payload


def answer_messages(question: str, evidence: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": contract.ANSWER_PROMPT.format(
                memories=evidence, question=question
            ),
        }
    ]


def exact_prompt_token_counter(
    tokenizer: visible.TokenCounter, evidence: str
) -> tuple[Callable[[str], int], dict[str, Any]]:
    """Prepare an exact counter for questions sharing one large context.

    ``encode_with_unstable`` returns only prefix tokens that cannot change for
    any continuation. Retaining those tokens and encoding the undecided byte
    suffix with each serialized question is exactly equivalent to encoding the
    complete canonical request.
    """

    marker = "R115_EXACT_QUESTION_8f76f5c55c7e4ad19329"
    rendered = contract.canonical_json(
        {"messages": answer_messages(marker, evidence)}
    )
    if rendered.count(marker) != 1:
        raise RunError("exact prompt marker collides with BEAM evidence")
    prefix, suffix = rendered.split(marker)
    if tokenizer.identity.get("implementation") != "tiktoken":
        return (
            lambda question: rendered_request_tokens(
                tokenizer, answer_messages(question, evidence)
            ),
            {
                "version": "direct-full-request-v1",
                "prefix_sha256": contract.sha256_text(prefix),
                "stable_prefix_tokens": None,
                "residual_sha256": None,
                "suffix_sha256": contract.sha256_text(suffix),
            },
        )

    import tiktoken  # type: ignore[import-not-found]

    encoding = tiktoken.get_encoding(str(tokenizer.identity["encoding_name"]))
    stable_ids, _ = encoding.encode_with_unstable(prefix)
    prefix_bytes = prefix.encode("utf-8")
    stable_count = len(stable_ids)
    while stable_count >= 0:
        stable_bytes = encoding.decode_bytes(stable_ids[:stable_count])
        if not prefix_bytes.startswith(stable_bytes):
            raise RunError("tiktoken stable prefix is not a request prefix")
        try:
            stable_bytes.decode("utf-8")
            residual = prefix_bytes[len(stable_bytes) :].decode("utf-8")
            break
        except UnicodeDecodeError:
            stable_count -= 1
    else:  # pragma: no cover - the empty prefix is valid UTF-8
        raise RunError("cannot find a UTF-8-aligned stable prompt prefix")
    stable_ids = stable_ids[:stable_count]

    def count(question: str) -> int:
        escaped_question = json.dumps(question, ensure_ascii=False)[1:-1]
        assembled = prefix + escaped_question + suffix
        expected = contract.canonical_json(
            {"messages": answer_messages(question, evidence)}
        )
        if assembled != expected:
            raise RunError("incremental prompt serialization is not exact")
        return len(stable_ids) + len(
            encoding.encode(residual + escaped_question + suffix)
        )

    return count, {
        "version": contract.EXACT_PROMPT_COUNT_VERSION,
        "prefix_sha256": contract.sha256_text(prefix),
        "stable_prefix_tokens": len(stable_ids),
        "stable_prefix_bytes": len(encoding.decode_bytes(stable_ids)),
        "residual_sha256": contract.sha256_text(residual),
        "residual_bytes": len(residual.encode("utf-8")),
        "suffix_sha256": contract.sha256_text(suffix),
    }


def rendered_request_tokens(
    tokenizer: visible.TokenCounter, messages: list[dict[str, str]]
) -> int:
    return tokenizer.count(contract.canonical_json({"messages": messages}))


def parse_answer(content: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", content, flags=re.DOTALL)
    if len(matches) != 1 or not matches[0].strip():
        raise RunError("GPT-5.5 response lacks one nonempty <answer> element")
    return matches[0].strip()


def usage_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    output = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for record in records:
        if record.get("event") != "model_call_finished":
            continue
        output["calls"] += 1
        usage = record.get("usage")
        if not isinstance(usage, Mapping):
            raise RunError("model ledger response lacks usage")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RunError(f"model ledger usage lacks {key}")
            output[key] += value
    return output


def call_answer(
    *,
    client: Any,
    observer: durable.DurableModelObserver,
    operation_id: str,
    question: str,
    evidence: str,
) -> dict[str, Any]:
    messages = answer_messages(question, evidence)
    started = time.monotonic()
    with observer.operation(operation_id):
        response = client.chat.completions.create(
            model=contract.REQUESTED_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=contract.ANSWER_MAX_TOKENS,
        )
    content = response.choices[0].message.content or ""
    answer = parse_answer(content)
    usage = durable.normalize_usage(getattr(response, "usage", None))
    if any(usage[key] is None for key in usage):
        raise RunError("answer response lacks provider usage")
    serialized = durable.response_dict(response)
    gateway_meta = serialized.get("flex_gateway_meta")
    return {
        "answer": answer,
        "raw_response_content": content,
        "response_id": str(getattr(response, "id", "") or ""),
        "actual_model": str(getattr(response, "model", "") or ""),
        "provider_usage": usage,
        "latency_s": round(time.monotonic() - started, 6),
        "messages_sha256": contract.canonical_hash(messages),
        "flex_gateway_meta": (
            dict(gateway_meta) if isinstance(gateway_meta, Mapping) else None
        ),
    }


def flex_consumer_records(
    *, ledger_records: Iterable[Mapping[str, Any]], model_root: Path
) -> list[dict[str, Any]]:
    """Extract non-secret provider bindings from committed response artifacts."""

    output: list[dict[str, Any]] = []
    for event in ledger_records:
        if event.get("event") != "model_call_finished":
            continue
        response_path = model_root / str(event.get("response_path", ""))
        response_artifact = contract.read_json(response_path)
        response = response_artifact.get("response")
        if not isinstance(response, Mapping):
            raise RunError("model response artifact is invalid")
        metadata = response.get("flex_gateway_meta")
        if not isinstance(metadata, Mapping):
            raise RunError("formal model response lacks Flex gateway metadata")
        record = {
            "gateway_request_id": metadata.get("request_id"),
            "response_id": response.get("id"),
            "gateway_request_sha256": metadata.get("request_sha256"),
            "provider_request_sha256": metadata.get("provider_request_sha256"),
            "provider_actual_model": metadata.get("provider_actual_model"),
            "service_tier": metadata.get("service_tier"),
            "actual_model": response.get("model"),
            "operation_id": event.get("operation_id"),
            "logical_call_id": event.get("logical_call_id"),
            "response_artifact_sha256": contract.sha256_file(response_path),
        }
        if (
            not isinstance(record["gateway_request_id"], str)
            or not record["gateway_request_id"]
            or record["response_id"] != event.get("response_id")
        ):
            raise RunError("formal Flex consumer binding is incomplete")
        output.append(record)
    return output


def source_recall(
    *, gold_source_ids: list[str], delivered_source_ids: list[str], eligible: bool
) -> dict[str, Any]:
    gold = set(gold_source_ids)
    delivered = set(delivered_source_ids)
    return {
        "eligible": eligible,
        "gold_source_count": len(gold),
        "delivered_gold_source_count": len(gold & delivered),
        "recall": (len(gold & delivered) / len(gold) if eligible and gold else None),
    }


def _source_texts(
    conversation: Mapping[str, Any], source_ids: Iterable[str]
) -> list[dict[str, Any]]:
    wanted = set(str(value) for value in source_ids)
    output: list[dict[str, Any]] = []
    for session in conversation["sessions"]:
        for turn in session["turns"]:
            matched = [
                value for value in turn["source_chat_ids"] if value in wanted
            ]
            if not matched:
                continue
            output.append(
                {
                    "source_chat_ids": matched,
                    "dia_id": turn["dia_id"],
                    "session_index": session["session_index"],
                    "text": render_turn(
                        chat_size=str(conversation["chat_size"]),
                        conversation_index=int(conversation["conversation_index"]),
                        session_index=int(session["session_index"]),
                        date=str(session["date"]),
                        turn=turn,
                    ),
                }
            )
    return output


def prepare_capped_evidence(
    *,
    method: str,
    question_record: Mapping[str, Any],
    conversation: Mapping[str, Any],
    retrieval_records: list[dict[str, Any]],
    item_dir: Path,
    tokenizer: visible.TokenCounter,
    memory_before: visible.MemorySnapshot,
    memory_after: visible.MemorySnapshot,
    actual_model_usage: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    trace_path = item_dir / "visible_tokens.jsonl"
    gate = visible.VisibleTokenBudgetGate(
        trace_path=trace_path,
        run_id=f"r115:{method}:{question_record['question_id']}",
        configured_budget_tokens=contract.VISIBLE_BUDGET_TOKENS,
        tokenizer=tokenizer,
        memory_before=memory_before,
        overflow_policy="truncate",
        metadata={
            "benchmark": "BEAM",
            "method": method,
            "question_id": question_record["question_id"],
        },
    )
    delivered_blocks: list[str] = []
    delivered_sources: list[str] = []
    delivery_rows: list[dict[str, Any]] = []
    try:
        for retrieval in retrieval_records:
            rank = int(retrieval["rank"])
            source_ids = contract.dedupe(retrieval.get("source_chat_ids", []))
            descriptor = (
                f"[{method} retrieval rank={rank} "
                f"source_chat_ids={','.join(source_ids) or 'none'}]\n"
                f"{retrieval['text']}"
            )
            tool = gate.deliver_tool_result(
                event_id=f"retrieval-{rank:04d}",
                raw_text=descriptor,
                tool_name=f"{method}_retrieve",
                metadata={
                    "rank": rank,
                    "record_id": retrieval.get("record_id"),
                    "source_chat_ids": source_ids,
                    "raw_sha256": contract.sha256_text(descriptor),
                },
            )
            if tool.delivered_text is not None:
                delivered_blocks.append(tool.delivered_text)
            row = {
                "rank": rank,
                "record_id": retrieval.get("record_id"),
                "raw_text_sha256": contract.sha256_text(descriptor),
                "raw_source_chat_ids": source_ids,
                "tool_delivery": {
                    "decision": tool.decision,
                    "tokens": tool.delivered_tokens,
                },
                "source_deliveries": [],
            }
            if gate.exhausted:
                delivery_rows.append(row)
                break
            for source_index, source in enumerate(
                _source_texts(conversation, source_ids)
            ):
                source_delivery = gate.deliver_source_resolution(
                    event_id=f"source-{rank:04d}-{source_index:04d}",
                    raw_text=str(source["text"]),
                    source_ids=list(source["source_chat_ids"]),
                    metadata={
                        "rank": rank,
                        "dia_id": source["dia_id"],
                        "session_index": source["session_index"],
                    },
                )
                row["source_deliveries"].append(
                    {
                        "dia_id": source["dia_id"],
                        "source_chat_ids": source["source_chat_ids"],
                        "decision": source_delivery.decision,
                        "tokens": source_delivery.delivered_tokens,
                    }
                )
                if (
                    source_delivery.decision == "delivered"
                    and source_delivery.delivered_text is not None
                ):
                    delivered_blocks.append(source_delivery.delivered_text)
                    delivered_sources.extend(source["source_chat_ids"])
                elif source_delivery.delivered_text is not None:
                    delivered_blocks.append(source_delivery.delivered_text)
                if gate.exhausted:
                    break
            delivery_rows.append(row)
            if gate.exhausted:
                break
        manifest = gate.finalize(
            memory_after=memory_after,
            actual_model_usage=actual_model_usage,
            metadata={"retrieval_record_count": len(retrieval_records)},
        )
    except BaseException:
        gate.close_incomplete()
        raise
    evidence = "\n\n".join(delivered_blocks)
    return evidence, {
        "budget": contract.VISIBLE_BUDGET_TOKENS,
        "visible_tokens": manifest["summary"]["cumulative_visible_tokens"],
        "exhausted": manifest["summary"]["exhausted"],
        "trace_path": str(trace_path.relative_to(item_dir)),
        "trace_sha256": contract.sha256_file(trace_path),
        "manifest_path": str(
            trace_path.with_suffix(".jsonl.manifest.json").relative_to(item_dir)
        ),
        "manifest_sha256": contract.sha256_file(
            trace_path.with_suffix(".jsonl.manifest.json")
        ),
        "delivered_evidence_sha256": contract.sha256_text(evidence),
        "delivered_source_chat_ids": contract.dedupe(delivered_sources),
        "deliveries": delivery_rows,
    }


class Mem0Backend:
    def __init__(
        self,
        *,
        workspace: Path,
        conversation: Mapping[str, Any],
        client_base_url: str,
        proxy_log: Path | None,
        ledger: durable.HashChainLedger,
        model_root: Path,
        tokenizer: visible.TokenCounter,
        formal: bool,
    ) -> None:
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mem0 import Memory  # type: ignore[import-not-found]
        from mem0.configs.base import MemoryConfig  # type: ignore[import-not-found]
        from mem0.memory import main as mem0_main  # type: ignore[import-not-found]
        from mem0.memory import telemetry as mem0_telemetry  # type: ignore[import-not-found]
        from openai import OpenAI  # type: ignore[import-not-found]
        import httpx  # type: ignore[import-not-found]

        mem0_telemetry.MEM0_TELEMETRY = False
        mem0_main.MEM0_TELEMETRY = False
        workspace.mkdir(parents=True, exist_ok=True)
        collection = hashlib.sha256(
            f"{conversation['chat_size']}:{conversation['conversation_index']}".encode()
        ).hexdigest()[:24]
        config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "collection_name": f"r115_{collection}",
                    "embedding_model_dims": contract.EMBEDDING_DIMS,
                    "path": str(workspace / "qdrant"),
                    "on_disk": True,
                },
            },
            llm={
                "provider": "openai",
                "config": {
                    "model": contract.REQUESTED_MODEL,
                    "temperature": 0,
                    "api_key": "controlled-local-proxy",
                    "openai_base_url": client_base_url,
                    "is_reasoning_model": False,
                },
            },
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": contract.EMBEDDING_MODEL,
                    "embedding_dims": contract.EMBEDDING_DIMS,
                    "model_kwargs": {
                        "local_files_only": True,
                        "trust_remote_code": False,
                    },
                },
            },
            history_db_path=str(workspace / "history.sqlite"),
            version="v1.1",
        )
        ambient_openrouter = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            self.memory = Memory(config=config)
        finally:
            if ambient_openrouter is not None:
                os.environ["OPENROUTER_API_KEY"] = ambient_openrouter
        self.memory.llm.client.close()
        self.http_client = httpx.Client(trust_env=False)
        self.memory.llm.client = OpenAI(
            api_key="controlled-local-proxy",
            base_url=client_base_url,
            http_client=self.http_client,
        )
        if str(self.memory.llm.client.base_url).rstrip("/") != client_base_url.rstrip(
            "/"
        ):
            raise RunError("Mem0 client is not bound to the dedicated proxy")
        self.owner = f"beam:{conversation['chat_size']}:c{conversation['conversation_index']:03d}"
        self.conversation = conversation
        self.observer = durable.DurableModelObserver(
            ledger=ledger,
            artifact_root=model_root,
            token_counter=tokenizer,
            expected_model=contract.REQUESTED_MODEL,
            proxy_log=proxy_log,
            formal=formal,
        )
        self.observer.install(self.memory.llm.client.chat.completions)

    def operation(self, operation_id: str) -> ContextManager[None]:
        return self.observer.operation(operation_id)

    def ingest(self, session: Mapping[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": turn["role"], "content": turn["text"]}
            for turn in session["turns"]
            if turn["role"] in {"user", "assistant"}
        ]
        result = self.memory.add(
            messages,
            user_id=self.owner,
            metadata={
                "source_chat_ids": list(session["source_chat_ids"]),
                "source_dia_ids": [turn["dia_id"] for turn in session["turns"]],
                "session_index": session["session_index"],
                "session_date": session["date"],
                "source_document_sha256": session["document_sha256"],
                "owner": self.owner,
            },
            infer=True,
        )
        rows = result.get("results") if isinstance(result, Mapping) else None
        if not isinstance(rows, list):
            raise RunError("Mem0 add returned an invalid result")
        return {
            "session_index": session["session_index"],
            "source_chat_ids": session["source_chat_ids"],
            "source_document_sha256": session["document_sha256"],
            "result_count": len(rows),
            "memory_ids": [row.get("id") for row in rows if isinstance(row, Mapping)],
        }

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        value = self.memory.search(
            question,
            top_k=contract.MEM0_TOP_K,
            filters={"user_id": self.owner},
            threshold=0.1,
            rerank=False,
            explain=False,
        )
        rows = value.get("results") if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise RunError("Mem0 search returned an invalid result")
        output: list[dict[str, Any]] = []
        all_sources = set(self.conversation["source_id_map"])
        sessions = {
            int(session["session_index"]): session
            for session in self.conversation["sessions"]
        }
        for rank, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise RunError("Mem0 search row is invalid")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise RunError("Mem0 search row lacks metadata")
            sources = contract.dedupe(metadata.get("source_chat_ids", []))
            session_index = metadata.get("session_index")
            if (
                not sources
                or any(value not in all_sources for value in sources)
                or not isinstance(session_index, int)
                or session_index not in sessions
                or metadata.get("source_document_sha256")
                != sessions[session_index]["document_sha256"]
                or metadata.get("owner") != self.owner
            ):
                raise RunError("Mem0 search row has an invalid source binding")
            text = row.get("memory")
            if not isinstance(text, str) or not text.strip():
                raise RunError("Mem0 search row has empty memory text")
            output.append(
                {
                    "rank": rank,
                    "record_id": str(row.get("id", "")),
                    "text": text,
                    "score": row.get("score"),
                    "source_chat_ids": sources,
                    "session_index": session_index,
                    "source_document_sha256": metadata[
                        "source_document_sha256"
                    ],
                    "backend_metadata_sha256": contract.canonical_hash(dict(metadata)),
                }
            )
        return output

    def close(self) -> None:
        self.observer.restore()
        self.memory.llm.client.close()
        vector_client = getattr(
            getattr(self.memory, "vector_store", None), "client", None
        )
        if vector_client is not None and hasattr(vector_client, "close"):
            vector_client.close()
        self.memory.close()


def _append_operation(
    ledger: durable.HashChainLedger, operation_id: str, callback: Callable[[], Any]
) -> Any:
    ledger.append("operation_started", operation_id=operation_id)
    try:
        result = callback()
    except BaseException as exc:
        ledger.append(
            "operation_failed",
            operation_id=operation_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    ledger.append(
        "operation_committed",
        operation_id=operation_id,
        result_sha256=contract.canonical_hash(result),
    )
    return result


def _record_for_inventory(
    *,
    item: Mapping[str, Any],
    method: str,
    status: str,
    evidence: str | None,
    answer: Mapping[str, Any] | None,
    retrieval: list[dict[str, Any]],
    visible_accounting: Mapping[str, Any] | None,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    model_usage: Mapping[str, Any],
    latency_s: float,
) -> dict[str, Any]:
    delivered_sources = (
        list(visible_accounting["delivered_source_chat_ids"])
        if visible_accounting is not None
        else (
            contract.dedupe(
                source
                for row in retrieval
                for source in row.get("source_chat_ids", [])
            )
            if status == "answered"
            else []
        )
    )
    record: dict[str, Any] = {
        "schema_version": contract.QUESTION_SCHEMA,
        "benchmark": "BEAM",
        "milestone": "R115",
        "method": method,
        **dict(item),
        "status": status,
        "answer": answer["answer"] if answer else None,
        "raw_response_content": answer["raw_response_content"] if answer else None,
        "response_id": answer["response_id"] if answer else None,
        "actual_model": answer["actual_model"] if answer else None,
        "answer_messages_sha256": answer["messages_sha256"] if answer else None,
        "provider_usage": answer["provider_usage"] if answer else None,
        "flex_gateway_meta": answer["flex_gateway_meta"] if answer else None,
        "retrieval_records": retrieval,
        "delivered_evidence_sha256": (
            contract.sha256_text(evidence) if evidence is not None else None
        ),
        "delivered_evidence": evidence,
        "delivered_source_chat_ids": delivered_sources,
        "source_recall": source_recall(
            gold_source_ids=list(item["gold_source_ids"]),
            delivered_source_ids=delivered_sources,
            eligible=bool(item["source_recall_eligible"]),
        ),
        "visible_accounting": dict(visible_accounting or {}),
        "memory_before": dict(before),
        "memory_after": dict(after),
        "memory_changed": before["sha256"] != after["sha256"],
        "model_usage_cumulative": dict(model_usage),
        "latency_s": latency_s,
        "created_at": contract.utc_now(),
    }
    record["question_record_content_sha256"] = contract.content_hash(
        record, "question_record_content_sha256"
    )
    return record


def run_conversation(
    *,
    method: str,
    conversation: Mapping[str, Any],
    inventory_items: list[dict[str, Any]],
    condition_dir: Path,
    client: Any,
    provider_base_url: str,
    tokenizer: visible.TokenCounter,
    formal: bool,
    backend_factory: Callable[..., Backend] | None = None,
) -> dict[str, Any]:
    chat_size = str(conversation["chat_size"])
    conversation_index = int(conversation["conversation_index"])
    conversation_dir = (
        condition_dir / chat_size / f"conversation_{conversation_index:03d}"
    )
    checkpoint_path = conversation_dir / "checkpoint.json"
    identity = {
        "schema_version": contract.CHECKPOINT_SCHEMA,
        "method": method,
        "chat_size": chat_size,
        "conversation_index": conversation_index,
        "normalized_conversation_sha256": conversation[
            "normalized_conversation_sha256"
        ],
        "question_ids": [item["question_id"] for item in inventory_items],
    }
    if checkpoint_path.exists():
        checkpoint = contract.read_json(checkpoint_path)
        if not isinstance(checkpoint, dict) or any(
            checkpoint.get(key) != value for key, value in identity.items()
        ):
            raise RunError("existing conversation checkpoint identity differs")
        if checkpoint.get("status") == "complete":
            return checkpoint
    else:
        conversation_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = {
            **identity,
            "status": "running",
            "created_at": contract.utc_now(),
            "completed_question_ids": [],
            "build": None,
        }
        contract.atomic_json_no_clobber(checkpoint_path, checkpoint)

    model_root = conversation_dir / "model_evidence"
    model_root.mkdir(exist_ok=True)
    ledger = durable.HashChainLedger(
        model_root / "ledger.jsonl",
        run_id=f"r115:{method}:{chat_size}:c{conversation_index:03d}",
    )
    ledger.path.touch(exist_ok=True)
    ledger_snapshot = durable.ledger_state(ledger.records)
    committed_operations = set(ledger_snapshot["committed_operations"])
    observer = durable.DurableModelObserver(
        ledger=ledger,
        artifact_root=model_root,
        token_counter=tokenizer,
        expected_model=contract.REQUESTED_MODEL,
        proxy_log=None,
        formal=False,
    )
    backend: Backend | None = None
    workspace: Path | None = None
    build_started = time.monotonic()
    try:
        if method == "mem0":
            workspace = conversation_dir / "workspace"
            before = visible.snapshot_memory_bytes(b"", label="empty_mem0_workspace")
            existing_build = checkpoint.get("build")
            build_complete = (
                isinstance(existing_build, dict)
                and existing_build.get("status") == "complete"
            )
            expected_build_operations = {
                f"build-session-{int(session['session_index']):04d}"
                for session in conversation["sessions"]
            }
            if not build_complete and (
                workspace.exists()
                or expected_build_operations & committed_operations
            ):
                raise RunError(
                    "Mem0 build artifacts exist without a committed checkpoint; "
                    "use a new output root"
                )
            if build_complete and not expected_build_operations.issubset(
                committed_operations
            ):
                raise RunError("Mem0 build checkpoint lacks committed operations")
            factory = backend_factory or Mem0Backend
            backend = factory(
                workspace=workspace,
                conversation=conversation,
                client_base_url=provider_base_url,
                proxy_log=None,
                ledger=ledger,
                model_root=model_root,
                tokenizer=tokenizer,
                formal=False,
            )
            if build_complete:
                after_build = visible.snapshot_memory_path(workspace)
                if (
                    existing_build.get("workspace") != str(workspace.resolve())
                    or existing_build.get("memory_after") != after_build.descriptor
                ):
                    raise RunError("resumed Mem0 workspace differs from checkpoint")
            else:
                receipts = []
                for session in conversation["sessions"]:
                    operation_id = f"build-session-{int(session['session_index']):04d}"
                    receipts.append(
                        _append_operation(
                            ledger,
                            operation_id,
                            lambda session=session, operation_id=operation_id: (
                                _backend_ingest(backend, operation_id, session)
                            ),
                        )
                    )
                after_build = visible.snapshot_memory_path(workspace)
                checkpoint["build"] = {
                    "status": "complete",
                    "receipts": receipts,
                    "workspace": str(workspace.resolve()),
                    "memory_before": before.descriptor,
                    "memory_after": after_build.descriptor,
                    "build_latency_s": round(time.monotonic() - build_started, 6),
                    "model_usage": usage_summary(ledger.records),
                }
        else:
            payload = contract.canonical_json(conversation).encode("utf-8")
            before = visible.snapshot_memory_bytes(payload, label="immutable_raw_beam")
            after_build = before
            observer.install(client.chat.completions)
            checkpoint["build"] = {
                "status": "not_applicable_read_only",
                "memory_before": before.descriptor,
                "memory_after": after_build.descriptor,
                "build_latency_s": 0.0,
                "model_usage": usage_summary(ledger.records),
            }
        contract.atomic_json_replace(checkpoint_path, checkpoint)

        full_context = render_full_context(conversation)
        full_context_prompt_count, _ = exact_prompt_token_counter(
            tokenizer, full_context
        )
        chunks = turn_chunks(conversation)
        bm25_index = build_bm25_index(chunks) if method == "bm25" else None
        for item in inventory_items:
            question_id = str(item["question_id"])
            question_path = conversation_dir / "questions" / f"{item['question_index']:02d}.json"
            if question_path.exists():
                existing = contract.read_json(question_path)
                if (
                    not isinstance(existing, dict)
                    or existing.get("question_id") != question_id
                    or existing.get("question_record_content_sha256")
                    != contract.content_hash(existing, "question_record_content_sha256")
                ):
                    raise RunError(f"existing question record differs: {question_id}")
                if question_id not in checkpoint["completed_question_ids"]:
                    raise RunError("question exists without checkpoint commit")
                continue
            if question_id in checkpoint["completed_question_ids"]:
                raise RunError("checkpoint references a missing question record")
            question_operations = {
                f"answer-q{int(item['question_index']):02d}",
                f"retrieve-q{int(item['question_index']):02d}",
            }
            if question_operations & committed_operations:
                raise RunError(
                    "committed model/retrieval work lacks a question checkpoint; "
                    "use a new output root"
                )

            started = time.monotonic()
            retrieval: list[dict[str, Any]] = []
            evidence: str | None
            answer: dict[str, Any] | None
            accounting: dict[str, Any] | None = None
            question_dir = conversation_dir / "question_evidence" / f"q{item['question_index']:02d}"
            question_dir.mkdir(parents=True, exist_ok=False)
            before_question = (
                visible.snapshot_memory_path(workspace)
                if workspace is not None
                else after_build
            )

            if method == "full_context":
                evidence = full_context
                prompt_tokens = full_context_prompt_count(str(item["question"]))
                if prompt_tokens > contract.MAX_RENDERED_PROMPT_TOKENS:
                    status = contract.BLOCKED_STATUS
                    answer = None
                    context_preflight = {
                        "policy": contract.FULL_CONTEXT_POLICY,
                        "complete_render_sha256": contract.sha256_text(evidence),
                        "complete_render_tokens": tokenizer.count(evidence),
                        "rendered_prompt_tokens": prompt_tokens,
                        "answer_completion_reservation_tokens": contract.ANSWER_MAX_TOKENS,
                        "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
                        "fit": False,
                        "truncation_applied": False,
                    }
                else:
                    status = "answered"
                    operation_id = f"answer-q{int(item['question_index']):02d}"
                    answer = _append_operation(
                        ledger,
                        operation_id,
                        lambda operation_id=operation_id, item=item: call_answer(
                            client=client,
                            observer=observer,
                            operation_id=operation_id,
                            question=str(item["question"]),
                            evidence=full_context,
                        ),
                    )
                    context_preflight = {
                        "policy": contract.FULL_CONTEXT_POLICY,
                        "complete_render_sha256": contract.sha256_text(evidence),
                        "complete_render_tokens": tokenizer.count(evidence),
                        "rendered_prompt_tokens": prompt_tokens,
                        "answer_completion_reservation_tokens": contract.ANSWER_MAX_TOKENS,
                        "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
                        "fit": True,
                        "truncation_applied": False,
                    }
                retrieval = [
                    {
                        "rank": index,
                        "record_id": chunk["chunk_id"],
                        "text_sha256": chunk["text_sha256"],
                        "source_chat_ids": chunk["source_chat_ids"],
                    }
                    for index, chunk in enumerate(chunks)
                ]
                accounting = context_preflight
                if status == "answered":
                    accounting["delivered_source_chat_ids"] = contract.dedupe(
                        source
                        for chunk in chunks
                        for source in chunk["source_chat_ids"]
                    )
                else:
                    accounting["delivered_source_chat_ids"] = []
            else:
                if method == "bm25":
                    assert bm25_index is not None
                    retrieval = [
                        {
                            "rank": row["rank"],
                            "record_id": row["chunk_id"],
                            "text": (
                                f"BM25 score={row['score']:.12f}; "
                                f"turn={row['chunk_id']}"
                            ),
                            "score": row["score"],
                            "source_chat_ids": row["source_chat_ids"],
                            "raw_turn_sha256": row["text_sha256"],
                        }
                        for row in rank_bm25(
                            str(item["question"]), chunks, index=bm25_index
                        )
                    ]
                else:
                    assert backend is not None
                    operation_id = f"retrieve-q{int(item['question_index']):02d}"
                    retrieval = _append_operation(
                        ledger,
                        operation_id,
                        lambda operation_id=operation_id, item=item: _backend_retrieve(
                            backend, operation_id, str(item["question"])
                        ),
                    )
                after_retrieval = (
                    visible.snapshot_memory_path(workspace)
                    if workspace is not None
                    else after_build
                )
                if after_retrieval.descriptor != before_question.descriptor:
                    raise RunError(
                        "retrieval mutated the frozen memory workspace; "
                        "formal answer call refused"
                    )
                evidence, accounting = prepare_capped_evidence(
                    method=method,
                    question_record=item,
                    conversation=conversation,
                    retrieval_records=retrieval,
                    item_dir=question_dir,
                    tokenizer=tokenizer,
                    memory_before=before_question,
                    memory_after=after_retrieval,
                    actual_model_usage=usage_summary(ledger.records),
                )
                operation_id = f"answer-q{int(item['question_index']):02d}"
                answer = _append_operation(
                    ledger,
                    operation_id,
                    lambda operation_id=operation_id, item=item, evidence=evidence: call_answer(
                        client=client,
                        observer=(backend.observer if method == "mem0" else observer),  # type: ignore[attr-defined]
                        operation_id=operation_id,
                        question=str(item["question"]),
                        evidence=evidence,
                    ),
                )
                status = "answered"

            after_question = (
                visible.snapshot_memory_path(workspace)
                if workspace is not None
                else after_build
            )
            if after_question.descriptor != before_question.descriptor:
                raise RunError("answering mutated the frozen memory workspace")
            record_evidence = evidence if status == "answered" else None
            record = _record_for_inventory(
                item=item,
                method=method,
                status=status,
                evidence=record_evidence,
                answer=answer,
                retrieval=retrieval,
                visible_accounting=accounting,
                before=before_question.descriptor,
                after=after_question.descriptor,
                model_usage=usage_summary(ledger.records),
                latency_s=round(time.monotonic() - started, 6),
            )
            contract.atomic_json_no_clobber(question_path, record)
            checkpoint["completed_question_ids"].append(question_id)
            checkpoint["model_usage"] = usage_summary(ledger.records)
            checkpoint["last_question_id"] = question_id
            contract.atomic_json_replace(checkpoint_path, checkpoint)

        if len(checkpoint["completed_question_ids"]) != len(inventory_items):
            raise RunError("conversation question checkpoint is incomplete")
        state = durable.ledger_state(ledger.records)
        checkpoint.update(
            {
                "status": "complete",
                "completed_at": contract.utc_now(),
                "model_ledger_path": str(
                    (model_root / "ledger.jsonl").relative_to(conversation_dir)
                ),
                "model_ledger_sha256": contract.sha256_file(
                    model_root / "ledger.jsonl"
                ),
                "model_ledger_state": state,
                "model_usage": usage_summary(ledger.records),
                "memory_after_run": (
                    visible.snapshot_memory_path(workspace).descriptor
                    if workspace is not None
                    else after_build.descriptor
                ),
            }
        )
        contract.atomic_json_replace(checkpoint_path, checkpoint)
        return checkpoint
    finally:
        if backend is not None:
            backend.close()
        else:
            observer.restore()


def _backend_ingest(
    backend: Backend, operation_id: str, session: Mapping[str, Any]
) -> dict[str, Any]:
    with backend.operation(operation_id):
        return backend.ingest(session)


def _backend_retrieve(
    backend: Backend, operation_id: str, question: str
) -> list[dict[str, Any]]:
    with backend.operation(operation_id):
        return backend.retrieve(question)


def _openai_client(base_url: str) -> Any:
    from openai import OpenAI  # type: ignore[import-not-found]
    import httpx  # type: ignore[import-not-found]

    return OpenAI(
        api_key="controlled-local-proxy",
        base_url=base_url,
        http_client=httpx.Client(trust_env=False, timeout=3600.0),
        max_retries=0,
    )


def collect_condition_consumers(condition_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_root in sorted(condition_dir.glob("*/conversation_*/model_evidence")):
        ledger_path = model_root / "ledger.jsonl"
        ledger_records = durable.read_ledger(ledger_path)
        durable.ledger_state(ledger_records)
        records.extend(
            flex_consumer_records(
                ledger_records=ledger_records,
                model_root=model_root,
            )
        )
    request_ids = [record["gateway_request_id"] for record in records]
    if len(request_ids) != len(set(request_ids)):
        raise RunError("condition reuses a Flex gateway request ID")
    return records


def _window_consumers(
    window: Mapping[str, Any], records: Iterable[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    segment = window.get("segment")
    if not isinstance(segment, Mapping) or not isinstance(
        segment.get("request_ids"), list
    ):
        raise RunError("closed Flex window lacks request IDs")
    wanted = set(str(value) for value in segment["request_ids"])
    return [record for record in records if record.get("gateway_request_id") in wanted]


def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    formal = args.fixture is None
    if formal and not args.allow_model_requests:
        raise RunError("formal R115 execution requires --allow-model-requests")
    if not formal and args.allow_model_requests:
        raise RunError("synthetic R115 execution forbids --allow-model-requests")
    if formal:
        if args.gateway_root is None:
            raise RunError("formal R115 execution requires --gateway-root")
        provider_contract = flex_evidence.active_contract(args.gateway_root)
        provider_base_url = str(provider_contract["base_url"])
    else:
        provider_contract = {
            "schema": "r115-synthetic-provider/v1",
            "status": "synthetic_no_network",
        }
        provider_base_url = contract.SYNTHETIC_BASE_URL
    preregistration = contract.validate_preregistration(args.preregistration)
    tokenizer = visible.TokenCounter.resolve(
        requested_model=contract.REQUESTED_MODEL,
        fallback_encoding=contract.TOKEN_ENCODING,
        allow_byte_fallback=False,
    )
    if (
        tokenizer.identity.get("implementation_version") != contract.TIKTOKEN_VERSION
        or tokenizer.identity.get("encoding_name") != contract.TOKEN_ENCODING
        or tokenizer.identity.get("provider_exact") is not False
    ):
        raise RunError("R115 tokenizer identity differs")
    if formal:
        contract.dependency_snapshot()
    splits, sources = load_splits(
        fixture=args.fixture, cache_dir=args.dataset_cache_dir
    )
    selections = selected_indices(splits, fixture=not formal)
    if formal and selections != contract.FORMAL_SELECTION:
        raise RunError("formal R115 selection must equal all 45 conversations")
    mapping, mapping_descriptor = (
        load_mapping_records(contract.EVIDENCE_MAPPING_QUESTIONS)
        if formal
        else (None, {"kind": "fixture_without_r002"})
    )
    inventory, conversations = build_inventory(
        splits=splits, selections=selections, mapping=mapping
    )
    if formal and len(inventory) != contract.FORMAL_QUESTIONS:
        raise RunError("formal R115 inventory must contain exactly 900 questions")
    output_root = args.output_root.expanduser().resolve()
    condition_dir = output_root / args.condition
    manifest_path = condition_dir / "run_manifest.json"
    condition_inventory_path = condition_dir / "condition_inventory.json"
    identity = {
        "schema_version": contract.SCHEMA,
        "benchmark": "BEAM",
        "milestone": "R115",
        "method": args.condition,
        "scope": "formal_900" if formal else "fixture",
        "formal": formal,
        "selection": selections,
        "question_count": len(inventory),
        "requested_model": contract.REQUESTED_MODEL,
        "base_url": provider_base_url,
        "preregistration_path": str(args.preregistration.resolve()),
        "preregistration_sha256": contract.sha256_file(args.preregistration),
        "preregistration_content_sha256": preregistration[
            "preregistration_content_sha256"
        ],
        "dataset_sources": sources,
        "mapping": mapping_descriptor,
        "source_hashes": contract.source_hashes(),
        "tokenizer": tokenizer.identity,
        "provider_contract": provider_contract,
    }
    identity["run_identity_sha256"] = contract.canonical_hash(identity)
    if manifest_path.exists():
        if not args.resume:
            raise RunError("R115 run manifest exists; pass --resume or use a new root")
        manifest = contract.read_json(manifest_path)
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value for key, value in identity.items()
        ):
            raise RunError("existing R115 run identity differs")
        if manifest.get("status") == "complete":
            return manifest
    else:
        if condition_dir.exists() and any(condition_dir.iterdir()):
            raise RunError("R115 condition directory is nonempty without a manifest")
        condition_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            **identity,
            "status": "running",
            "created_at": contract.utc_now(),
            "completed_conversations": [],
            "provider_windows": [],
            "provider_window_start": None,
        }
        contract.atomic_json_no_clobber(manifest_path, manifest)
        inventory_artifact = {
            "schema_version": contract.INVENTORY_SCHEMA,
            "method": args.condition,
            "selection": selections,
            "question_count": len(inventory),
            "records": inventory,
        }
        inventory_artifact["inventory_content_sha256"] = contract.content_hash(
            inventory_artifact, "inventory_content_sha256"
        )
        contract.atomic_json_no_clobber(
            condition_inventory_path, inventory_artifact
        )
    if not condition_inventory_path.is_file():
        raise RunError("R115 condition inventory is missing")
    manifest["condition_inventory_sha256"] = contract.sha256_file(
        condition_inventory_path
    )
    lock_handle = (condition_dir / ".runner.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RunError("another R115 runner is active") from exc
    if formal:
        existing_consumers = collect_condition_consumers(condition_dir)
        open_window = manifest.get("provider_window_start")
        if open_window is not None:
            closed = flex_evidence.capture_end(open_window)
            flex_evidence.audit_window(
                closed,
                consumer_records=_window_consumers(closed, existing_consumers),
            )
            manifest.setdefault("provider_windows", []).append(closed)
        manifest["provider_window_start"] = flex_evidence.capture_start(
            args.gateway_root
        )
        contract.atomic_json_replace(manifest_path, manifest)
    client = (
        args.client_factory()
        if args.client_factory
        else _openai_client(provider_base_url)
    )
    try:
        for conversation_key, conversation in conversations.items():
            if conversation_key in manifest["completed_conversations"]:
                continue
            items = [
                item
                for item in inventory
                if item["chat_size"] == conversation["chat_size"]
                and item["conversation_index"] == conversation["conversation_index"]
            ]
            checkpoint = run_conversation(
                method=args.condition,
                conversation=conversation,
                inventory_items=items,
                condition_dir=condition_dir,
                client=client,
                provider_base_url=provider_base_url,
                tokenizer=tokenizer,
                formal=formal,
                backend_factory=args.backend_factory,
            )
            if checkpoint.get("status") != "complete":
                raise RunError(f"conversation did not complete: {conversation_key}")
            manifest["completed_conversations"].append(conversation_key)
            contract.atomic_json_replace(manifest_path, manifest)
        expected = [
            f"{chat_size}:c{index:03d}"
            for chat_size in ("100K", "1M")
            for index in selections.get(chat_size, [])
        ]
        if manifest["completed_conversations"] != expected:
            raise RunError("R115 completed conversation inventory differs")
        if formal:
            closed = flex_evidence.capture_end(manifest["provider_window_start"])
            consumers = collect_condition_consumers(condition_dir)
            flex_evidence.audit_window(
                closed,
                consumer_records=_window_consumers(closed, consumers),
            )
            manifest.setdefault("provider_windows", []).append(closed)
            manifest["provider_window_start"] = None
            covered = {
                request_id
                for window in manifest["provider_windows"]
                for request_id in window["segment"]["request_ids"]
            }
            if covered != {
                record["gateway_request_id"] for record in consumers
            }:
                raise RunError("Flex provider windows do not cover all model calls")
        manifest.update(
            {
                "status": "complete",
                "completed_at": contract.utc_now(),
                "completed_question_count": len(inventory),
                "provider_evidence_status": (
                    "closed_and_audited" if formal else "synthetic_not_applicable"
                ),
            }
        )
        contract.atomic_json_replace(manifest_path, manifest)
        return manifest
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--output", type=Path, default=DEFAULT_PREREGISTRATION)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--dataset-cache-dir", type=Path, default=contract.HF_CACHE)
    run = subparsers.add_parser("run")
    run.add_argument("--condition", choices=contract.FORMAL_METHODS, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    run.add_argument("--dataset-cache-dir", type=Path, default=contract.HF_CACHE)
    run.add_argument(
        "--gateway-root",
        type=Path,
        help="active OpenAI GPT-5.5 Flex gateway result root",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-model-requests", action="store_true")
    run.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.client_factory = None
    args.backend_factory = None
    if args.command == "run":
        args.output_root = args.output_root.expanduser().resolve()
        args.preregistration = args.preregistration.expanduser().resolve()
        args.dataset_cache_dir = args.dataset_cache_dir.expanduser().resolve()
        args.gateway_root = (
            args.gateway_root.expanduser().resolve() if args.gateway_root else None
        )
        args.fixture = args.fixture.expanduser().resolve() if args.fixture else None
    elif args.command == "preflight":
        args.preregistration = args.preregistration.expanduser().resolve()
        args.output = args.output.expanduser().resolve()
        args.dataset_cache_dir = args.dataset_cache_dir.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "preregister":
            value = create_preregistration(args.output.expanduser().resolve())
        elif args.command == "preflight":
            value = create_formal_preflight(
                preregistration_path=args.preregistration,
                output_path=args.output,
                cache_dir=args.dataset_cache_dir,
            )
        else:
            value = run_condition(args)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (
        RunError,
        contract.ContractError,
        durable.DurableLedgerError,
        flex_evidence.EvidenceError,
    ) as exc:
        print(f"R115 failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
