#!/usr/bin/env python3
"""Independent offline auditor for an executed R115 BEAM control condition.

The auditor does not import the R115 runner.  It reloads the pinned dataset,
reconstructs the exact selection, canonical conversation rendering, question
inventory, BM25 ranking, context-fit decisions, source mappings, visible-token
traces, durable model ledgers, proxy linkage, and completeness.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import r115_beam_control_contract as contract  # noqa: E402
from src import openai_gpt55_flex_gateway as flex_gateway  # noqa: E402
from src import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402
from src.evaluation import durable_model_ledger as durable  # noqa: E402
from src.evaluation import visible_token_audit  # noqa: E402
from src.evaluation import visible_token_budget as visible  # noqa: E402


class AuditError(RuntimeError):
    """Raised when an R115 artifact cannot be admitted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


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


def _parse_chat(value: Any) -> list[list[dict[str, Any]]]:
    require(isinstance(value, list) and value, "BEAM chat is empty")
    first = value[0]
    if isinstance(first, list):
        batches = [_flatten_turns(item) for item in value]
    elif isinstance(first, dict) and "turns" in first:
        batches = [_flatten_turns(item.get("turns")) for item in value]
    elif isinstance(first, dict) and ("role" in first or "content" in first):
        batches = [_flatten_turns(value)]
    else:
        raise AuditError("BEAM chat layout is unsupported")
    require(bool(batches) and all(batches), "BEAM chat has an empty batch")
    return batches


def _normalize_date(raw: Any) -> str:
    require(raw is not None and bool(str(raw).strip()), "BEAM time anchor is missing")
    text = str(raw).strip()
    iso = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if iso:
        from datetime import datetime

        year, month, day = map(int, iso.groups())
        return datetime(year, month, day).date().isoformat()
    named = re.search(
        r"\b([A-Za-z]+)[\s\-/]+(\d{1,2})(?:st|nd|rd|th)?"
        r"[\s,\-/]+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if named:
        from datetime import datetime

        candidate = " ".join(named.groups())
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
    raise AuditError(f"unparseable BEAM time anchor: {text!r}")


def _flatten_sources(value: Any) -> list[str]:
    output: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, bool):
            raise AuditError("boolean BEAM source identifier")
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
        raise AuditError("unsupported BEAM source identifier")

    visit(value)
    return output


def _parse_questions(
    raw: Any, *, require_formal_inventory: bool
) -> list[dict[str, Any]]:
    value = raw
    if isinstance(raw, str):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AuditError("invalid BEAM probing_questions") from exc
    require(isinstance(value, dict), "BEAM probing_questions is not an object")
    require(
        set(value).issubset(contract.QUESTION_TYPES),
        "BEAM question type inventory differs",
    )
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for question_type in contract.QUESTION_TYPES:
        rows = value.get(question_type, [])
        if isinstance(rows, (dict, str)):
            rows = [rows]
        require(isinstance(rows, list), f"{question_type} rows are invalid")
        for raw_question in rows:
            question = (
                {"question": raw_question}
                if isinstance(raw_question, str)
                else dict(raw_question)
            )
            text = str(
                question.get("question_text", question.get("question", "")) or ""
            ).strip()
            require(bool(text), f"empty {question_type} question")
            gold_field = contract.GOLD_FIELD_BY_TYPE[question_type]
            require(gold_field in question, f"{question_type} lacks {gold_field}")
            rubric = question.get("rubric", [])
            if isinstance(rubric, dict):
                rubric = rubric.get("nuggets", [])
            if not isinstance(rubric, list):
                rubric = [rubric] if rubric else []
            nuggets = []
            for item in rubric:
                if isinstance(item, dict):
                    item = (
                        item.get("description")
                        or item.get("nugget")
                        or item.get("criterion")
                        or item.get("text")
                    )
                if str(item or "").strip():
                    nuggets.append(str(item).strip())
            question.update(
                {
                    "question_type": question_type,
                    "question_text": text,
                    "gold_field": gold_field,
                    "gold": question[gold_field],
                    "rubric_nuggets": nuggets,
                    "raw_source_chat_ids": question.get("source_chat_ids"),
                    "normalized_source_chat_ids": _flatten_sources(
                        question.get("source_chat_ids")
                    ),
                }
            )
            output.append(question)
            counts[question_type] += 1
    if require_formal_inventory:
        require(len(output) == 20, "BEAM question count differs")
        require(
            all(counts[value] == 2 for value in contract.QUESTION_TYPES),
            "BEAM per-type question count differs",
        )
    return output


def _render_turn(
    *,
    chat_size: str,
    conversation_index: int,
    session_index: int,
    date: str,
    turn: Mapping[str, Any],
) -> str:
    sources = ",".join(turn["source_chat_ids"]) or "none"
    return (
        f"[BEAM {chat_size} conversation={conversation_index:03d} "
        f"session={session_index:04d} date={date} "
        f"dia_id={turn['dia_id']} source_chat_ids={sources} "
        f"role={turn['role']}]\n{turn['text']}"
    )


def reconstruct_conversation(
    item: Mapping[str, Any],
    *,
    chat_size: str,
    conversation_index: int,
    require_formal_inventory: bool,
) -> dict[str, Any]:
    sessions = []
    source_map: dict[str, list[str]] = {}
    for session_index, batch in enumerate(_parse_chat(item.get("chat"))):
        turns = []
        dates: list[str] = []
        for raw in batch:
            text = str(raw.get("content", "") or "").strip()
            if not text:
                continue
            if raw.get("time_anchor"):
                date = _normalize_date(raw["time_anchor"])
                if date not in dates:
                    dates.append(date)
            role = str(raw.get("role", "user") or "user").strip().lower()
            role = {"human": "user", "customer": "user", "ai": "assistant", "bot": "assistant"}.get(role, role)
            require(role in {"user", "assistant", "system", "tool"}, "role differs")
            dia_id = f"D{session_index + 1}:{len(turns) + 1}"
            sources = []
            if raw.get("id") is not None:
                source_id = str(raw["id"])
                sources = [source_id]
                source_map.setdefault(source_id, []).append(dia_id)
            turns.append(
                {
                    "dia_id": dia_id,
                    "source_chat_ids": sources,
                    "role": role,
                    "text": text,
                    "raw_turn_sha256": contract.canonical_hash(dict(raw)),
                }
            )
        require(bool(turns) and bool(dates), "session lacks turns or time anchor")
        rendered = [
            _render_turn(
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
                "source_chat_ids": list(
                    dict.fromkeys(
                        source for turn in turns for source in turn["source_chat_ids"]
                    )
                ),
                "document_sha256": contract.sha256_text("\n\n".join(rendered)),
            }
        )
    value = {
        "chat_size": chat_size,
        "conversation_index": conversation_index,
        "conversation_id": str(
            item.get("conversation_id", f"{chat_size}_{conversation_index}")
        ),
        "sessions": sessions,
        "source_id_map": source_map,
        "questions": _parse_questions(
            item.get("probing_questions"),
            require_formal_inventory=require_formal_inventory,
        ),
    }
    value["normalized_conversation_sha256"] = contract.canonical_hash(value)
    return value


def _full_render(conversation: Mapping[str, Any]) -> str:
    return "\n\n".join(
        _render_turn(
            chat_size=conversation["chat_size"],
            conversation_index=conversation["conversation_index"],
            session_index=session["session_index"],
            date=session["date"],
            turn=turn,
        )
        for session in conversation["sessions"]
        for turn in session["turns"]
    )


def _chunks(conversation: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for session in conversation["sessions"]:
        for turn in session["turns"]:
            text = _render_turn(
                chat_size=conversation["chat_size"],
                conversation_index=conversation["conversation_index"],
                session_index=session["session_index"],
                date=session["date"],
                turn=turn,
            )
            output.append(
                {
                    "chunk_id": turn["dia_id"],
                    "session_index": session["session_index"],
                    "date": session["date"],
                    "source_chat_ids": turn["source_chat_ids"],
                    "text": text,
                    "text_sha256": contract.sha256_text(text),
                }
            )
    return output


def _bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.count = len(chunks)
        self.lengths = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for document_index, chunk in enumerate(chunks):
            frequencies = Counter(_bm25_tokenize(str(chunk["text"])))
            self.lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                self.postings.setdefault(term, []).append(
                    (document_index, frequency)
                )
        self.average = sum(self.lengths) / self.count if self.count else 0.0

    def scores(self, question: str) -> list[float]:
        scores = [0.0] * self.count
        for term in list(dict.fromkeys(_bm25_tokenize(question))):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            df = len(postings)
            inverse = math.log(
                1.0 + (self.count - df + 0.5) / (df + 0.5)
            )
            for document_index, frequency in postings:
                denominator = frequency + 1.5 * (
                    1.0
                    - 0.75
                    + 0.75
                    * (
                        self.lengths[document_index] / self.average
                        if self.average
                        else 0.0
                    )
                )
                scores[document_index] += inverse * frequency * 2.5 / denominator
        return scores


def _bm25_rank(
    question: str,
    chunks: list[dict[str, Any]],
    index: _BM25Index | None = None,
) -> list[dict[str, Any]]:
    scores = (index or _BM25Index(chunks)).scores(question)
    return [
        {**chunk, "rank": rank, "score": score}
        for rank, (chunk, score) in enumerate(
            sorted(zip(chunks, scores), key=lambda row: (-row[1], row[0]["chunk_id"]))[:40]
        )
    ]


def _load_dataset(*, fixture: Path | None, cache_dir: Path) -> dict[str, Any]:
    if fixture is not None:
        value = contract.read_json(fixture)
        require(isinstance(value, dict), "fixture is not an object")
        return {key: value.get(key, []) for key in ("100K", "1M")}
    require(
        cache_dir.expanduser().resolve() == contract.HF_CACHE.resolve(),
        "formal audit requires the frozen BEAM Arrow cache root",
    )
    import pyarrow  # type: ignore[import-not-found]
    import pyarrow.ipc  # type: ignore[import-not-found]

    require(
        contract.sha256_file(contract.DATASET_INFO_PATH)
        == contract.DATASET_INFO_SHA256,
        "BEAM dataset_info.json hash differs",
    )
    output = {}
    for chat_size in ("100K", "1M"):
        arrow_path = contract.ARROW_FILES[chat_size]
        require(
            contract.sha256_file(arrow_path) == contract.ARROW_SHA256[chat_size],
            f"BEAM {chat_size} Arrow hash differs",
        )
        with pyarrow.memory_map(str(arrow_path), "r") as source:
            table = pyarrow.ipc.open_stream(source).read_all()
        require(
            contract.sha256_bytes(table.schema.serialize().to_pybytes())
            == contract.ARROW_SCHEMA_SHA256,
            f"BEAM {chat_size} Arrow schema differs",
        )
        require(
            table.num_rows == contract.EXPECTED_SPLIT_ROWS[chat_size],
            f"BEAM {chat_size} Arrow row count differs",
        )
        output[chat_size] = table.to_pylist()
    return output


def _load_r002() -> dict[str, dict[str, Any]]:
    rows = contract.read_jsonl(contract.EVIDENCE_MAPPING_QUESTIONS)
    output = {}
    for row in rows:
        if row.get("benchmark") != "BEAM":
            continue
        key = row.get("question_id")
        require(isinstance(key, str) and key not in output, "R002 identity differs")
        output[key] = row
    require(len(output) == 900, "R002 BEAM count differs")
    return output


def _expected_inventory(
    *, datasets: Mapping[str, Any], selection: Mapping[str, list[int]], formal: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    r002 = _load_r002() if formal else {}
    inventory = []
    conversations = {}
    for chat_size in ("100K", "1M"):
        for conversation_index in selection.get(chat_size, []):
            conversation = reconstruct_conversation(
                datasets[chat_size][conversation_index],
                chat_size=chat_size,
                conversation_index=conversation_index,
                require_formal_inventory=formal,
            )
            conversations[f"{chat_size}:c{conversation_index:03d}"] = conversation
            for question_index, question in enumerate(conversation["questions"]):
                question_id = (
                    f"beam:{chat_size}:c{conversation_index:03d}:"
                    f"q{question_index:02d}:{question['question_type']}"
                )
                mapping = r002.get(question_id)
                if formal:
                    require(isinstance(mapping, dict), f"R002 misses {question_id}")
                    for key, expected in {
                        "chat_size": chat_size,
                        "conversation_index": conversation_index,
                        "question_index": question_index,
                        "question_type": question["question_type"],
                        "question": question["question_text"],
                        "normalized_source_ids": question[
                            "normalized_source_chat_ids"
                        ],
                    }.items():
                        require(mapping.get(key) == expected, f"R002 {key} differs")
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
                        "gold_source_ids": mapping["normalized_source_ids"] if mapping else [],
                        "source_recall_eligible": bool(
                            mapping and mapping["source_recall_eligible"]
                        ),
                        "source_recall_exclusion_reasons": (
                            mapping["source_recall_exclusion_reasons"] if mapping else []
                        ),
                        "normalized_conversation_sha256": conversation[
                            "normalized_conversation_sha256"
                        ],
                        "r002_record_sha256": (
                            contract.canonical_hash(mapping) if mapping else None
                        ),
                    }
                )
    return inventory, conversations


def _request_payload(
    *, question: str, evidence: str
) -> tuple[list[dict[str, str]], str]:
    messages = [
        {
            "role": "user",
            "content": contract.ANSWER_PROMPT.format(
                memories=evidence, question=question
            ),
        }
    ]
    return messages, contract.canonical_hash(messages)


def _exact_prompt_token_counter(
    tokenizer: visible.TokenCounter, evidence: str
) -> tuple[Callable[[str], int], dict[str, Any]]:
    """Independently reconstruct exact shared-prefix prompt token counts."""

    marker = "R115_EXACT_QUESTION_8f76f5c55c7e4ad19329"
    marker_messages, _ = _request_payload(question=marker, evidence=evidence)
    rendered = contract.canonical_json({"messages": marker_messages})
    require(rendered.count(marker) == 1, "exact prompt marker collision")
    prefix, suffix = rendered.split(marker)
    if tokenizer.identity.get("implementation") != "tiktoken":
        return (
            lambda question: tokenizer.count(
                contract.canonical_json(
                    {"messages": _request_payload(question=question, evidence=evidence)[0]}
                )
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
        require(
            prefix_bytes.startswith(stable_bytes),
            "tiktoken stable prefix is not a request prefix",
        )
        try:
            stable_bytes.decode("utf-8")
            residual = prefix_bytes[len(stable_bytes) :].decode("utf-8")
            break
        except UnicodeDecodeError:
            stable_count -= 1
    else:  # pragma: no cover
        raise AuditError("cannot find a UTF-8-aligned stable prompt prefix")
    stable_ids = stable_ids[:stable_count]

    def count(question: str) -> int:
        escaped_question = json.dumps(question, ensure_ascii=False)[1:-1]
        assembled = prefix + escaped_question + suffix
        messages, _ = _request_payload(question=question, evidence=evidence)
        require(
            assembled == contract.canonical_json({"messages": messages}),
            "incremental prompt serialization is not exact",
        )
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


def _audit_answer_call(
    *,
    ledger: list[dict[str, Any]],
    model_root: Path,
    operation_id: str,
    question: str,
    evidence: str,
    record: Mapping[str, Any],
    proxy_log: Path | None,
    formal: bool,
) -> None:
    starts = [
        row
        for row in ledger
        if row.get("event") == "model_call_started"
        and row.get("operation_id") == operation_id
    ]
    finishes = [
        row
        for row in ledger
        if row.get("event") == "model_call_finished"
        and row.get("operation_id") == operation_id
    ]
    require(len(starts) == len(finishes) == 1, f"{operation_id} call count differs")
    start, finish = starts[0], finishes[0]
    require(
        start.get("logical_call_id") == finish.get("logical_call_id"),
        f"{operation_id} logical call differs",
    )
    request_path = model_root / str(start.get("request_path", ""))
    response_path = model_root / str(finish.get("response_path", ""))
    require(request_path.is_file() and response_path.is_file(), "call artifacts missing")
    require(contract.sha256_file(request_path) == start["request_sha256"], "request hash differs")
    require(contract.sha256_file(response_path) == finish["response_sha256"], "response hash differs")
    request = contract.read_json(request_path)
    response = contract.read_json(response_path)
    messages, messages_sha = _request_payload(question=question, evidence=evidence)
    expected_visible = {"messages": messages}
    require(
        request.get("requested_model") == contract.REQUESTED_MODEL
        and request.get("model_visible_payload") == expected_visible
        and request.get("model_visible_sha256")
        == contract.sha256_text(durable.canonical_json(expected_visible)),
        f"{operation_id} request differs",
    )
    options = request.get("request_options")
    require(
        isinstance(options, dict)
        and options.get("temperature") == 0
        and options.get("max_tokens") == contract.ANSWER_MAX_TOKENS,
        f"{operation_id} request options differ",
    )
    body = response.get("response")
    require(isinstance(body, dict), "response body missing")
    require(
        body.get("id") == finish.get("response_id") == record.get("response_id")
        and body.get("model") == finish.get("response_model") == record.get("actual_model")
        and re.fullmatch(contract.ACTUAL_MODEL_PATTERN, str(body.get("model", ""))),
        f"{operation_id} response identity differs",
    )
    require(
        durable.normalize_usage(body.get("usage"))
        == finish.get("usage")
        == record.get("provider_usage"),
        f"{operation_id} usage differs",
    )
    require(record.get("answer_messages_sha256") == messages_sha, "message hash differs")
    content = body.get("choices", [{}])[0].get("message", {}).get("content")
    require(content == record.get("raw_response_content"), "response content differs")
    matches = re.findall(r"<answer>(.*?)</answer>", str(content), flags=re.DOTALL)
    require(len(matches) == 1 and matches[0].strip() == record.get("answer"), "answer parse differs")
    if formal:
        metadata = body.get("flex_gateway_meta")
        require(
            isinstance(metadata, dict)
            and metadata == record.get("flex_gateway_meta")
            and metadata.get("provider_actual_model") == flex_gateway.PROVIDER_MODEL
            and metadata.get("service_tier") == flex_gateway.SERVICE_TIER
            and metadata.get("returned_alias") == contract.REQUESTED_MODEL
            and isinstance(metadata.get("request_id"), str)
            and bool(metadata["request_id"]),
            "Flex response metadata differs",
        )


def _flex_consumers(
    *, ledger: Iterable[Mapping[str, Any]], model_root: Path
) -> list[dict[str, Any]]:
    output = []
    for event in ledger:
        if event.get("event") != "model_call_finished":
            continue
        response_path = model_root / str(event.get("response_path", ""))
        require(response_path.is_file(), "Flex consumer response artifact missing")
        artifact = contract.read_json(response_path)
        response = artifact.get("response")
        require(isinstance(response, dict), "Flex consumer response invalid")
        metadata = response.get("flex_gateway_meta")
        require(isinstance(metadata, dict), "Flex consumer metadata missing")
        output.append(
            {
                "gateway_request_id": metadata.get("request_id"),
                "response_id": response.get("id"),
                "gateway_request_sha256": metadata.get("request_sha256"),
                "provider_request_sha256": metadata.get(
                    "provider_request_sha256"
                ),
                "provider_actual_model": metadata.get("provider_actual_model"),
                "service_tier": metadata.get("service_tier"),
                "actual_model": response.get("model"),
                "operation_id": event.get("operation_id"),
                "logical_call_id": event.get("logical_call_id"),
                "response_artifact_sha256": contract.sha256_file(response_path),
            }
        )
    return output


def _audit_question(
    *,
    method: str,
    item: Mapping[str, Any],
    record: Mapping[str, Any],
    conversation: Mapping[str, Any],
    question_dir: Path,
    model_root: Path,
    ledger: list[dict[str, Any]],
    tokenizer: visible.TokenCounter,
    full_context_prompt_count: Callable[[str], int] | None,
    bm25_index: _BM25Index | None,
    proxy_log: Path | None,
    formal: bool,
) -> dict[str, Any]:
    require(record.get("schema_version") == contract.QUESTION_SCHEMA, "question schema differs")
    require(record.get("method") == method, "question method differs")
    for key, expected in item.items():
        require(record.get(key) == expected, f"question field differs: {key}")
    require(
        record.get("question_record_content_sha256")
        == contract.content_hash(record, "question_record_content_sha256"),
        "question content hash differs",
    )
    all_sources = set(conversation["source_id_map"])
    delivered_sources = record.get("delivered_source_chat_ids")
    require(
        isinstance(delivered_sources, list)
        and len(delivered_sources) == len(set(delivered_sources))
        and set(delivered_sources).issubset(all_sources),
        "delivered evidence contains another/future conversation source",
    )
    expected_recall = {
        "eligible": item["source_recall_eligible"],
        "gold_source_count": len(set(item["gold_source_ids"])),
        "delivered_gold_source_count": len(
            set(item["gold_source_ids"]) & set(delivered_sources)
        ),
        "recall": (
            len(set(item["gold_source_ids"]) & set(delivered_sources))
            / len(set(item["gold_source_ids"]))
            if item["source_recall_eligible"] and item["gold_source_ids"]
            else None
        ),
    }
    require(record.get("source_recall") == expected_recall, "source recall differs")
    require(
        record.get("memory_before") == record.get("memory_after")
        and record.get("memory_changed") is False,
        "question retrieval or answering mutated the frozen memory workspace",
    )
    chunks = _chunks(conversation)
    operation_id = f"answer-q{int(item['question_index']):02d}"
    if method == "full_context":
        full_context = _full_render(conversation)
        require(
            full_context_prompt_count is not None,
            "full-context prompt counter missing",
        )
        prompt_tokens = full_context_prompt_count(str(item["question"]))
        fit = prompt_tokens <= contract.MAX_RENDERED_PROMPT_TOKENS
        accounting = record.get("visible_accounting")
        require(isinstance(accounting, dict), "full-context accounting missing")
        expected_preflight = {
            "policy": contract.FULL_CONTEXT_POLICY,
            "complete_render_sha256": contract.sha256_text(full_context),
            "complete_render_tokens": tokenizer.count(full_context),
            "rendered_prompt_tokens": prompt_tokens,
            "answer_completion_reservation_tokens": contract.ANSWER_MAX_TOKENS,
            "model_context_limit_tokens": contract.MODEL_CONTEXT_LIMIT_TOKENS,
            "fit": fit,
            "truncation_applied": False,
            "delivered_source_chat_ids": (
                list(
                    dict.fromkeys(
                        source
                        for chunk in chunks
                        for source in chunk["source_chat_ids"]
                    )
                )
                if fit
                else []
            ),
        }
        require(accounting == expected_preflight, "full-context preflight differs")
        if fit:
            require(record.get("status") == "answered", "fit full-context not answered")
            require(record.get("delivered_evidence") == full_context, "full-context was truncated")
            _audit_answer_call(
                ledger=ledger,
                model_root=model_root,
                operation_id=operation_id,
                question=item["question"],
                evidence=full_context,
                record=record,
                proxy_log=proxy_log,
                formal=formal,
            )
        else:
            require(record.get("status") == contract.BLOCKED_STATUS, "overflow was not blocked")
            require(
                record.get("answer") is None
                and record.get("response_id") is None
                and record.get("delivered_evidence") is None,
                "blocked full-context has a fabricated answer/evidence",
            )
            require(
                not any(row.get("operation_id") == operation_id for row in ledger),
                "blocked full-context made an answer call",
            )
    else:
        require(record.get("status") == "answered", "capped method is not answered")
        retrieval = record.get("retrieval_records")
        require(isinstance(retrieval, list), "retrieval records missing")
        if method == "bm25":
            require(bm25_index is not None, "BM25 frozen index missing")
            expected = [
                {
                    "rank": row["rank"],
                    "record_id": row["chunk_id"],
                    "text": f"BM25 score={row['score']:.12f}; turn={row['chunk_id']}",
                    "score": row["score"],
                    "source_chat_ids": row["source_chat_ids"],
                    "raw_turn_sha256": row["text_sha256"],
                }
                for row in _bm25_rank(
                    item["question"], chunks, index=bm25_index
                )
            ]
            require(retrieval == expected, "BM25 ranking or chunking differs")
        else:
            require(len(retrieval) <= contract.MEM0_TOP_K, "Mem0 top-k differs")
            require(
                [row.get("rank") for row in retrieval] == list(range(len(retrieval))),
                "Mem0 ranks differ",
            )
            for row in retrieval:
                require(
                    isinstance(row.get("text"), str)
                    and bool(row["text"].strip())
                    and set(row.get("source_chat_ids", [])).issubset(all_sources)
                    and row.get("source_chat_ids"),
                    "Mem0 source trace differs",
                )
        accounting = record.get("visible_accounting")
        require(isinstance(accounting, dict), "visible accounting missing")
        trace = question_dir / str(accounting.get("trace_path", ""))
        trace_manifest = question_dir / str(accounting.get("manifest_path", ""))
        require(
            trace.is_file()
            and trace_manifest.is_file()
            and contract.sha256_file(trace) == accounting.get("trace_sha256")
            and contract.sha256_file(trace_manifest) == accounting.get("manifest_sha256"),
            "visible trace hash differs",
        )
        report = visible_token_audit.audit_visible_token_trace(
            trace,
            manifest_path=trace_manifest,
            require_complete=True,
        )
        require(report["audit_status"] == "pass", f"visible trace failed: {report['errors']}")
        require(
            report["configured_budget_tokens"] == contract.VISIBLE_BUDGET_TOKENS
            and report["cumulative_visible_tokens"] == accounting["visible_tokens"]
            and report["cumulative_visible_tokens"] <= contract.VISIBLE_BUDGET_TOKENS,
            "visible budget differs",
        )
        evidence = record.get("delivered_evidence")
        require(isinstance(evidence, str), "delivered evidence missing")
        require(
            contract.sha256_text(evidence)
            == record.get("delivered_evidence_sha256")
            == accounting.get("delivered_evidence_sha256"),
            "delivered evidence hash differs",
        )
        trace_rows = contract.read_jsonl(trace)
        reconstructed = "\n\n".join(
            row["delivered"]["text"]
            for row in trace_rows
            if row.get("record_type") == "delivery"
            and isinstance(row.get("delivered"), dict)
        )
        require(evidence == reconstructed, "delivered evidence does not match trace")
        traced_sources = list(
            dict.fromkeys(
                source
                for row in trace_rows
                if row.get("record_type") == "delivery"
                and row.get("kind") == "source_resolution"
                and row.get("decision") == "delivered"
                and isinstance(row.get("delivered"), dict)
                for source in row.get("metadata", {}).get("source_ids", [])
            )
        )
        require(traced_sources == delivered_sources, "delivered source trace differs")
        partial_sources = {
            source
            for row in trace_rows
            if row.get("record_type") == "delivery"
            and row.get("kind") == "source_resolution"
            and row.get("decision") == "truncated"
            for source in row.get("metadata", {}).get("source_ids", [])
        }
        require(
            partial_sources.isdisjoint(delivered_sources),
            "partially delivered source was counted as source recall",
        )
        _audit_answer_call(
            ledger=ledger,
            model_root=model_root,
            operation_id=operation_id,
            question=item["question"],
            evidence=evidence,
            record=record,
            proxy_log=proxy_log,
            formal=formal,
        )
    return {
        "question_id": item["question_id"],
        "status": record["status"],
        "source_recall_eligible": item["source_recall_eligible"],
        "source_recall": record["source_recall"]["recall"],
    }


def audit(
    *,
    condition_dir: Path,
    fixture: Path | None,
    cache_dir: Path,
    allow_fixture: bool,
) -> dict[str, Any]:
    condition_dir = condition_dir.expanduser().resolve()
    require(condition_dir.is_dir() and not condition_dir.is_symlink(), "condition directory missing")
    manifest_path = condition_dir / "run_manifest.json"
    manifest = contract.read_json(manifest_path)
    require(isinstance(manifest, dict), "run manifest invalid")
    method = manifest.get("method")
    require(method in contract.FORMAL_METHODS, "run method differs")
    formal = bool(manifest.get("formal"))
    require(formal == (fixture is None), "fixture/formal identity differs")
    require(formal or allow_fixture, "fixture audit requires --allow-fixture")
    expected_scope = "formal_900" if formal else "fixture"
    require(
        manifest.get("schema_version") == contract.SCHEMA
        and manifest.get("benchmark") == "BEAM"
        and manifest.get("milestone") == "R115"
        and manifest.get("scope") == expected_scope
        and manifest.get("status") == "complete"
        and manifest.get("requested_model") == contract.REQUESTED_MODEL,
        "run manifest frozen identity differs",
    )
    prereg_path = Path(str(manifest.get("preregistration_path", "")))
    prereg = contract.validate_preregistration(prereg_path)
    require(
        contract.sha256_file(prereg_path) == manifest.get("preregistration_sha256")
        and prereg["preregistration_content_sha256"]
        == manifest.get("preregistration_content_sha256"),
        "preregistration binding differs",
    )
    require(manifest.get("source_hashes") == contract.source_hashes(), "source code changed")
    provider_contract = manifest.get("provider_contract")
    if formal:
        validated_provider = flex_evidence.validate_recorded_contract(
            provider_contract
        )
        require(
            validated_provider.get("provider_model")
            == flex_gateway.PROVIDER_MODEL
            and validated_provider.get("service_tier")
            == flex_gateway.SERVICE_TIER,
            "formal Flex provider contract differs",
        )
        require(
            manifest.get("base_url") == validated_provider.get("base_url"),
            "formal base URL is not derived from the Flex provider contract",
        )
    else:
        require(
            provider_contract
            == {
                "schema": "r115-synthetic-provider/v1",
                "status": "synthetic_no_network",
            },
            "fixture provider identity differs",
        )
        require(
            manifest.get("base_url") == contract.SYNTHETIC_BASE_URL,
            "fixture base URL differs",
        )
    tokenizer = visible.TokenCounter.from_identity(manifest.get("tokenizer", {}))
    require(
        tokenizer.identity.get("implementation_version") == "0.12.0"
        and tokenizer.identity.get("encoding_name") == "o200k_base",
        "tokenizer identity differs",
    )
    selection = manifest.get("selection")
    require(isinstance(selection, dict), "selection missing")
    if formal:
        require(selection == contract.FORMAL_SELECTION, "formal all-45 selection differs")
        require(manifest.get("question_count") == 900, "formal question count differs")
        require(
            contract.sha256_file(contract.EVIDENCE_MAPPING_MANIFEST)
            == prereg["r002"]["manifest_sha256"]
            and contract.sha256_file(contract.EVIDENCE_MAPPING_AUDIT)
            == prereg["r002"]["audit_sha256"]
            and contract.sha256_file(contract.EVIDENCE_MAPPING_QUESTIONS)
            == prereg["r002"]["questions_sha256"],
            "R002 frozen files changed",
        )
    datasets = _load_dataset(fixture=fixture, cache_dir=cache_dir)
    expected_inventory, conversations = _expected_inventory(
        datasets=datasets, selection=selection, formal=formal
    )
    inventory_path = condition_dir / "condition_inventory.json"
    inventory = contract.read_json(inventory_path)
    require(isinstance(inventory, dict), "condition inventory invalid")
    require(
        inventory.get("schema_version") == contract.INVENTORY_SCHEMA
        and inventory.get("method") == method
        and inventory.get("selection") == selection
        and inventory.get("question_count") == len(expected_inventory)
        and inventory.get("records") == expected_inventory
        and inventory.get("inventory_content_sha256")
        == contract.content_hash(inventory, "inventory_content_sha256")
        and contract.sha256_file(inventory_path)
        == manifest.get("condition_inventory_sha256"),
        "condition inventory differs",
    )
    runner_lock = (condition_dir / ".runner.lock").open("r+")
    audit_lock = (condition_dir / ".audit.lock").open("w")
    try:
        fcntl.flock(audit_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(runner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        runner_lock.close()
        audit_lock.close()
        raise AuditError("R115 runner or auditor is active") from exc
    return _audit_locked_tail(
        condition_dir=condition_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        method=str(method),
        formal=formal,
        selection=selection,
        conversations=conversations,
        expected_inventory=expected_inventory,
        inventory_path=inventory_path,
        tokenizer=tokenizer,
        runner_lock=runner_lock,
        audit_lock=audit_lock,
    )


def audit_preflight(
    *, preflight_path: Path, cache_dir: Path, output_path: Path
) -> dict[str, Any]:
    """Independently reconstruct a zero-network formal preflight artifact."""

    preflight_path = preflight_path.expanduser().resolve()
    value = contract.read_json(preflight_path)
    require(isinstance(value, dict), "preflight is not an object")
    require(
        value.get("schema_version") == contract.PREFLIGHT_SCHEMA
        and value.get("status") == "passed"
        and value.get("benchmark") == "BEAM"
        and value.get("milestone") == "R115"
        and value.get("formal_methods") == list(contract.FORMAL_METHODS)
        and value.get("formal_selection") == contract.FORMAL_SELECTION
        and value.get("model_calls") == 0
        and value.get("network_calls") == 0
        and value.get("execution_status") == "not_started"
        and value.get("scoring_status") == "not_started",
        "preflight frozen identity differs",
    )
    require(
        value.get("preflight_content_sha256")
        == contract.content_hash(value, "preflight_content_sha256"),
        "preflight content hash differs",
    )
    prereg_descriptor = value.get("preregistration")
    require(isinstance(prereg_descriptor, dict), "preflight preregistration missing")
    prereg_path = Path(str(prereg_descriptor.get("path", "")))
    prereg = contract.validate_preregistration(prereg_path)
    require(
        contract.sha256_file(prereg_path) == prereg_descriptor.get("sha256")
        and prereg["preregistration_content_sha256"]
        == prereg_descriptor.get("content_sha256"),
        "preflight preregistration binding differs",
    )
    require(value.get("dependencies") == contract.dependency_snapshot(), "dependencies differ")
    require(value.get("source_hashes") == contract.source_hashes(), "preflight code changed")
    tokenizer = visible.TokenCounter.from_identity(value.get("tokenizer", {}))
    require(
        tokenizer.identity.get("implementation_version") == contract.TIKTOKEN_VERSION
        and tokenizer.identity.get("encoding_name") == contract.TOKEN_ENCODING,
        "preflight tokenizer differs",
    )
    datasets = _load_dataset(fixture=None, cache_dir=cache_dir)
    expected_sources = {
        chat_size: {
            "kind": "pinned_huggingface_arrow_ipc",
            "dataset": contract.HF_DATASET,
            "config": contract.HF_CONFIG,
            "revision": contract.HF_REVISION,
            "split": chat_size,
            "path": str(contract.ARROW_FILES[chat_size].resolve()),
            "sha256": contract.ARROW_SHA256[chat_size],
            "schema_sha256": contract.ARROW_SCHEMA_SHA256,
            "rows": len(dataset),
            "dataset_info_path": str(contract.DATASET_INFO_PATH.resolve()),
            "dataset_info_sha256": contract.DATASET_INFO_SHA256,
        }
        for chat_size, dataset in datasets.items()
    }
    require(value.get("dataset_sources") == expected_sources, "dataset sources differ")
    mapping_descriptor = {
        "path": str(contract.EVIDENCE_MAPPING_QUESTIONS.resolve()),
        "sha256": contract.sha256_file(contract.EVIDENCE_MAPPING_QUESTIONS),
        "rows": 900,
    }
    require(value.get("mapping") == mapping_descriptor, "mapping descriptor differs")
    inventory, conversations = _expected_inventory(
        datasets=datasets,
        selection=contract.FORMAL_SELECTION,
        formal=True,
    )
    require(
        value.get("inventory_sha256") == contract.canonical_hash(inventory),
        "preflight inventory hash differs",
    )
    records = []
    for conversation_key, conversation in conversations.items():
        full_context = _full_render(conversation)
        prompt_token_count, prompt_counter = _exact_prompt_token_counter(
            tokenizer, full_context
        )
        chunks = _chunks(conversation)
        bm25_index = _BM25Index(chunks)
        items = [
            item
            for item in inventory
            if item["chat_size"] == conversation["chat_size"]
            and item["conversation_index"] == conversation["conversation_index"]
        ]
        questions = []
        for item in items:
            prompt_tokens = prompt_token_count(str(item["question"]))
            ranked = _bm25_rank(
                item["question"], chunks, index=bm25_index
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
    require(value.get("conversation_records") == records, "preflight records differ")
    flat_questions = [question for record in records for question in record["questions"]]
    require(
        value.get("conversation_count") == 45
        and value.get("question_count") == 900
        and value.get("full_context_fit_count")
        == sum(question["full_context_fit"] for question in flat_questions)
        and value.get("full_context_blocked_over_context_count")
        == sum(not question["full_context_fit"] for question in flat_questions)
        and value.get("source_recall_denominator")
        == sum(question["source_recall_eligible"] for question in flat_questions),
        "preflight totals differ",
    )
    require(
        value.get("graphiti") == prereg["structured_control_decision"]
        and value["graphiti"].get("status") == "excluded_before_formal_answers",
        "Graphiti preregistered exclusion differs",
    )
    report = {
        "schema_version": contract.PREFLIGHT_AUDIT_SCHEMA,
        "status": "passed",
        "benchmark": "BEAM",
        "milestone": "R115",
        "formal_selection_verified": True,
        "conversation_count": 45,
        "question_count": 900,
        "full_context_fit_count": value["full_context_fit_count"],
        "full_context_blocked_over_context_count": value[
            "full_context_blocked_over_context_count"
        ],
        "source_recall_denominator": 804,
        "graphiti_status": "excluded_before_formal_answers",
        "preflight_path": str(preflight_path),
        "preflight_sha256": contract.sha256_file(preflight_path),
        "preflight_content_sha256": value["preflight_content_sha256"],
        "model_calls": 0,
        "network_calls": 0,
    }
    report["audit_content_sha256"] = contract.content_hash(
        report, "audit_content_sha256"
    )
    contract.atomic_json_no_clobber(output_path, report)
    return report


def _audit_locked_tail(
    *,
    condition_dir: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    method: str,
    formal: bool,
    selection: Mapping[str, list[int]],
    conversations: Mapping[str, dict[str, Any]],
    expected_inventory: list[dict[str, Any]],
    inventory_path: Path,
    tokenizer: visible.TokenCounter,
    runner_lock: Any,
    audit_lock: Any,
) -> dict[str, Any]:
    proxy_log = None
    audited = []
    provider_consumers: list[dict[str, Any]] = []
    try:
        expected_completed = []
        for conversation_key, conversation in conversations.items():
            expected_completed.append(conversation_key)
            chat_size = conversation["chat_size"]
            conversation_index = conversation["conversation_index"]
            conversation_dir = condition_dir / chat_size / f"conversation_{conversation_index:03d}"
            checkpoint = contract.read_json(conversation_dir / "checkpoint.json")
            items = [
                item
                for item in expected_inventory
                if item["chat_size"] == chat_size
                and item["conversation_index"] == conversation_index
            ]
            require(
                isinstance(checkpoint, dict)
                and checkpoint.get("schema_version") == contract.CHECKPOINT_SCHEMA
                and checkpoint.get("method") == method
                and checkpoint.get("status") == "complete"
                and checkpoint.get("normalized_conversation_sha256")
                == conversation["normalized_conversation_sha256"]
                and checkpoint.get("question_ids")
                == [item["question_id"] for item in items]
                and checkpoint.get("completed_question_ids")
                == [item["question_id"] for item in items],
                f"checkpoint differs: {conversation_key}",
            )
            model_root = conversation_dir / "model_evidence"
            ledger_path = model_root / "ledger.jsonl"
            require(
                contract.sha256_file(ledger_path) == checkpoint.get("model_ledger_sha256"),
                "model ledger hash differs",
            )
            ledger = durable.read_ledger(ledger_path)
            state = durable.ledger_state(ledger)
            require(state == checkpoint.get("model_ledger_state"), "model ledger state differs")
            require(
                not state["failed_model_calls"] and not state.get("failed_operations"),
                "model ledger has failures",
            )
            if formal:
                provider_consumers.extend(
                    _flex_consumers(ledger=ledger, model_root=model_root)
                )
            if method == "mem0":
                workspace = conversation_dir / "workspace"
                snapshot = visible.snapshot_memory_path(workspace).descriptor
                require(
                    snapshot == checkpoint.get("memory_after_run"),
                    "Mem0 workspace hash differs",
                )
            full_context_prompt_count = None
            if method == "full_context":
                full_context_prompt_count, _ = _exact_prompt_token_counter(
                    tokenizer, _full_render(conversation)
                )
            bm25_index = _BM25Index(_chunks(conversation)) if method == "bm25" else None
            for item in items:
                question_path = conversation_dir / "questions" / f"{item['question_index']:02d}.json"
                record = contract.read_json(question_path)
                require(isinstance(record, dict), "question record invalid")
                audited.append(
                    _audit_question(
                        method=method,
                        item=item,
                        record=record,
                        conversation=conversation,
                        question_dir=(
                            conversation_dir
                            / "question_evidence"
                            / f"q{item['question_index']:02d}"
                        ),
                        model_root=model_root,
                        ledger=ledger,
                        tokenizer=tokenizer,
                        full_context_prompt_count=full_context_prompt_count,
                        bm25_index=bm25_index,
                        proxy_log=proxy_log,
                        formal=formal,
                    )
                )
        require(
            manifest.get("completed_conversations") == expected_completed,
            "completed conversation inventory differs",
        )
        require(
            manifest.get("completed_question_count") == len(audited),
            "completed question count differs",
        )
        # Reject extra question records or conversation directories.
        actual_question_files = sorted(condition_dir.glob("*/conversation_*/questions/*.json"))
        require(len(actual_question_files) == len(audited), "extra/missing question files")
        actual_conversations = sorted(condition_dir.glob("*/conversation_*"))
        require(len(actual_conversations) == len(conversations), "extra/missing conversation directories")
        provider_audits = []
        if formal:
            require(
                manifest.get("provider_evidence_status") == "closed_and_audited"
                and manifest.get("provider_window_start") is None,
                "formal provider window is not closed",
            )
            windows = manifest.get("provider_windows")
            require(isinstance(windows, list) and windows, "provider windows missing")
            consumer_ids = [
                str(record.get("gateway_request_id")) for record in provider_consumers
            ]
            require(
                len(consumer_ids) == len(set(consumer_ids)),
                "duplicate consumer gateway request ID",
            )
            covered: list[str] = []
            for window in windows:
                segment = window.get("segment") if isinstance(window, dict) else None
                require(isinstance(segment, dict), "provider window segment missing")
                request_ids = segment.get("request_ids")
                require(isinstance(request_ids, list), "provider request IDs missing")
                wanted = set(str(value) for value in request_ids)
                subset = [
                    record
                    for record in provider_consumers
                    if record.get("gateway_request_id") in wanted
                ]
                provider_audits.append(
                    flex_evidence.audit_window(window, consumer_records=subset)
                )
                covered.extend(str(value) for value in request_ids)
            require(
                len(covered) == len(set(covered))
                and set(covered) == set(consumer_ids),
                "provider windows do not exactly cover consumer model calls",
            )
        else:
            require(
                manifest.get("provider_evidence_status")
                == "synthetic_not_applicable",
                "fixture provider evidence status differs",
            )
        report = {
            "schema_version": contract.AUDIT_SCHEMA,
            "status": "passed",
            "benchmark": "BEAM",
            "milestone": "R115",
            "method": method,
            "formal": formal,
            "selection": selection,
            "conversation_count": len(conversations),
            "question_count": len(audited),
            "answered_count": sum(row["status"] == "answered" for row in audited),
            "blocked_over_context_count": sum(
                row["status"] == contract.BLOCKED_STATUS for row in audited
            ),
            "source_recall_denominator": sum(
                row["source_recall_eligible"] for row in audited
            ),
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": contract.sha256_file(manifest_path),
            "condition_inventory_sha256": contract.sha256_file(inventory_path),
            "audited_question_ids_sha256": contract.canonical_hash(
                [row["question_id"] for row in audited]
            ),
            "scoring_status": "not_executed",
            "provider_request_count": len(provider_consumers),
            "provider_window_audits": provider_audits,
            "audit_network_calls": 0,
        }
        report["audit_content_sha256"] = contract.content_hash(
            report, "audit_content_sha256"
        )
        contract.atomic_json_no_clobber(condition_dir / "audit.json", report)
        return report
    finally:
        fcntl.flock(runner_lock.fileno(), fcntl.LOCK_UN)
        fcntl.flock(audit_lock.fileno(), fcntl.LOCK_UN)
        runner_lock.close()
        audit_lock.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("condition_dir", type=Path)
    run.add_argument("--dataset-cache-dir", type=Path, default=contract.HF_CACHE)
    run.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--allow-fixture", action="store_true", help=argparse.SUPPRESS)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("preflight_path", type=Path)
    preflight.add_argument("--dataset-cache-dir", type=Path, default=contract.HF_CACHE)
    preflight.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            output = args.output or args.preflight_path.with_name(
                f"{args.preflight_path.stem}.audit.json"
            )
            report = audit_preflight(
                preflight_path=args.preflight_path,
                cache_dir=args.dataset_cache_dir,
                output_path=output,
            )
        else:
            report = audit(
                condition_dir=args.condition_dir,
                fixture=args.fixture,
                cache_dir=args.dataset_cache_dir,
                allow_fixture=args.allow_fixture,
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (
        AuditError,
        contract.ContractError,
        durable.DurableLedgerError,
        flex_evidence.EvidenceError,
        OSError,
        ValueError,
    ) as exc:
        audit_path = (
            (args.output or args.preflight_path.with_name(
                f"{args.preflight_path.stem}.audit.json"
            )).expanduser().resolve()
            if args.command == "preflight"
            else args.condition_dir.expanduser().resolve() / "audit.json"
        )
        if audit_path.exists():
            audit_path.unlink()
        print(f"R115 audit failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
