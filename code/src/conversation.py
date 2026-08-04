"""LoCoMo conversation conversion and source lookup for Scriptorium."""

import hashlib
import re
from datetime import date
from typing import Any


def _turn_text(turn: dict[str, Any]) -> str:
    text = str(turn.get("text", turn.get("content", ""))).strip()
    query = str(turn.get("query", "")).strip()
    caption = str(turn.get("blip_caption", "")).strip()
    if query and caption:
        image = (
            f"[Sharing image - query: {query}. The image shows: {caption}]"
        )
    elif caption:
        image = f"[Sharing image. The image shows: {caption}]"
    elif query:
        image = f"[Sharing image - query: {query}.]"
    else:
        image = ""
    return "\n".join(part for part in (text, image) if part)


def benchmark_source_id(dia_id: str) -> str:
    """Map a LoCoMo evidence label to the opaque Scriptorium Source ID."""
    match = re.fullmatch(r"D(\d+):(\d+)", str(dia_id))
    if not match:
        return str(dia_id)
    thread_id = "thread_" + hashlib.sha256(
        f"locomo:D{match.group(1)}".encode()
    ).hexdigest()[:12]
    message_id = "msg_" + hashlib.sha256(
        f"locomo:{dia_id}".encode()
    ).hexdigest()[:12]
    return f"locomo/{thread_id}/{message_id}"


def add_benchmark_source_ids(turn_index):
    """Allow source resolution by benchmark labels and Scriptorium Source IDs."""
    augmented = dict(turn_index)
    augmented.update({
        benchmark_source_id(dia_id): turn
        for dia_id, turn in turn_index.items()
    })
    return augmented


def build_turn_index(conv: dict[str, Any]) -> dict[str, Any]:
    """Index source turns by the stable IDs written into Scriptorium."""
    result: dict[str, Any] = {}
    order = 0
    session_number = 1
    while f"session_{session_number}" in conv:
        observed = normalize_date(
            conv.get(f"session_{session_number}_date_time", "")
        )
        session = conv[f"session_{session_number}"]
        if isinstance(session, list):
            for position, turn in enumerate(session, start=1):
                if not isinstance(turn, dict):
                    continue
                source_id = benchmark_source_id(
                    str(turn.get("dia_id", f"D{session_number}:{position}"))
                )
                result[source_id] = {
                    "speaker": turn.get(
                        "speaker", turn.get("role", "user")
                    ),
                    "text": _turn_text(turn),
                    "order": order,
                    "date": observed,
                }
                order += 1
        session_number += 1
    return result


def read_turns(
    turn_index: dict[str, Any], source_ids: list[str], context: int = 1
) -> str:
    by_order = {
        int(record["order"]): record for record in turn_index.values()
    }
    wanted: set[int] = set()
    for source_id in source_ids:
        record = turn_index.get(str(source_id))
        if record is None:
            continue
        order = int(record["order"])
        wanted.update(
            candidate
            for candidate in range(order - context, order + context + 1)
            if candidate in by_order
        )
    return "\n".join(
        (
            f"({record['date']}) {record['speaker']}: {record['text']}"
            if record.get("date")
            else f"{record['speaker']}: {record['text']}"
        )
        for record in (by_order[order] for order in sorted(wanted))
    )


def normalize_date(raw: object) -> str:
    """Normalize benchmark timestamps without legacy runtime state."""
    value = str(raw or "").strip()
    if not value:
        return value
    match = re.search(
        r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", value
    )
    if match:
        try:
            return date(*(int(part) for part in match.groups())).isoformat()
        except ValueError:
            return value
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", value)
    if not match:
        return value
    months = {
        name: number
        for number, name in enumerate(
            (
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ),
            start=1,
        )
    }
    month = months.get(match.group(2).capitalize())
    if month is None:
        return value
    return f"{match.group(3)}-{month:02d}-{int(match.group(1)):02d}"


def session_content(
    session: object, session_number: int
) -> tuple[list[tuple[str, str]], list[str]]:
    turns: list[tuple[str, str]] = []
    refs: list[str] = []
    if not isinstance(session, list):
        return turns, refs
    for position, turn in enumerate(session, start=1):
        if isinstance(turn, dict):
            text = _turn_text(turn)
            if not text.strip():
                continue
            turns.append((
                str(turn.get("speaker", turn.get("role", "user"))),
                text,
            ))
            refs.append(str(turn.get("dia_id", f"D{session_number}:{position}")))
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            turns.append((str(turn[0]), str(turn[1])))
            refs.append(f"D{session_number}:{position}")
    return turns, refs
