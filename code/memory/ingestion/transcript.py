"""Read a Claude Code or Codex session transcript as source records.

Both hosts hand the hook a path to one JSONL session file, so nothing here
searches for transcripts or decodes the host's directory naming — the file
to read is always given.

What a session file holds is mostly not conversation. In one real 15366-line
transcript only 1059 user records were something a person typed; the other
13685 were tool results wearing the same `"type": "user"`. Recording those
as memory would bury a handful of stated preferences under a landfill of
file listings, so the readers below keep human turns and assistant prose and
drop everything else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..runtime.state import SourceRecord

CLAUDE_CODE = "claude-code"
CODEX = "codex"


def _text_from_content(content: Any) -> str:
    """The human-readable text of a message, whatever shape it arrived in.

    Claude Code stores a user message as a bare string or as a block list,
    and an assistant message always as blocks. Only `text` blocks are prose:
    `thinking` is not addressed to anyone, and `tool_use` is machinery.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _claude_code_turns(entries: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Human turns and assistant replies, in file order.

    A record is a person speaking when it carries `origin.kind == "human"`.
    That marker was present on every one of the 1169 typed and queued
    messages in the transcript this was checked against, and on none of the
    tool results, which makes it a better filter than `promptSource` — that
    field has several human-ish values and several that are not.
    """
    superseded: set[str] = set()
    seen: set[str] = set()
    for entry in entries:
        kind = entry.get("type")
        if kind not in ("user", "assistant"):
            continue
        for field in ("supersedesUuids", "retractedMessageUuids"):
            superseded.update(entry.get(field) or [])
        uuid = entry.get("uuid")
        if not uuid or uuid in seen:
            continue
        message = entry.get("message") or {}
        if kind == "user":
            origin = entry.get("origin") or {}
            if origin.get("kind") != "human":
                continue
        text = _text_from_content(message.get("content"))
        if not text:
            continue
        seen.add(uuid)
        yield {
            "message_id": uuid,
            "role": "user" if kind == "user" else "assistant",
            "content": text,
            "timestamp": entry.get("timestamp"),
            "thread_id": entry.get("sessionId") or "",
        }
    # A retraction can appear after the message it retracts, so this is
    # resolved once the whole file has been read rather than inline.
    if superseded:
        # Regenerating is cheaper than holding every turn in memory twice;
        # the caller filters on the returned ids.
        yield {"_superseded": superseded}


def _codex_turns(entries: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Codex wraps each turn in `payload` and has no per-message uuid.

    Ordinal position is the only stable identity available, so the message
    id is derived from it.
    """
    for index, entry in enumerate(entries):
        payload = entry.get("payload") or {}
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_from_content(payload.get("content"))
        if not text:
            continue
        yield {
            "message_id": str(payload.get("id") or f"msg_{index:06d}"),
            "role": role,
            "content": text,
            "timestamp": entry.get("timestamp") or payload.get("timestamp"),
            "thread_id": "",
        }


def _entries(path: Path) -> Iterator[dict[str, Any]]:
    """Parse line by line: a session file can reach hundreds of megabytes.

    A malformed line is skipped rather than fatal — a transcript being
    appended to while it is read can end mid-object.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def read_transcript(
    path: str | Path, *, provider: str | None = None
) -> list[SourceRecord]:
    """Every human turn and assistant reply in one session file.

    The provider is inferred from the file's shape when not given: Codex
    wraps turns in `payload`, Claude Code does not.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"transcript is not a file: {source}")

    rows = list(_entries(source))
    if provider is None:
        provider = CODEX if any("payload" in row for row in rows) else CLAUDE_CODE

    reader = _codex_turns if provider == CODEX else _claude_code_turns
    turns: list[dict[str, Any]] = []
    superseded: set[str] = set()
    for turn in reader(iter(rows)):
        if "_superseded" in turn:
            superseded = turn["_superseded"]
            continue
        turns.append(turn)

    thread = source.stem
    return [
        SourceRecord(
            provider=provider,
            thread_id=turn["thread_id"] or thread,
            message_id=turn["message_id"],
            ordinal=index,
            role=turn["role"],
            content=turn["content"],
            timestamp=turn["timestamp"],
        )
        for index, turn in enumerate(
            turn for turn in turns if turn["message_id"] not in superseded
        )
    ]
