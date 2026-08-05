"""Convert one BEAM conversation into the shape the LoCoMo runner reads.

BEAM ships as Arrow files of whole conversations: 100K, 500K and 1M tokens,
each cut into a handful of very large sessions and probed by twenty questions
across ten categories. LoCoMo is the opposite shape — many short sessions,
four categories — but the runner between them only needs sessions of turns
and questions with gold answers, so converting is enough to reuse the build,
query and evaluation path unchanged.

    python -m scripts.runners.beam.convert --size 100K --conversation 1 \
        --output benchmarks/beam/converted/beam100K-1.json

The result is a one-sample list, so `run_conversation.py --data <file>
--sample-id beam100K-1` runs it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

ARROW_DIR = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/beam/hf_cache/Mohammadta___beam/default/0.0.0"
    / "3205395e897e7318c7b094ef4e6047b9b82dbb03"
)
SIZES = ("100K", "500K", "1M")
# Fixed order so a category's number means the same thing in every run.
CATEGORIES = (
    "information_extraction",
    "multi_session_reasoning",
    "temporal_reasoning",
    "knowledge_update",
    "contradiction_resolution",
    "event_ordering",
    "preference_following",
    "instruction_following",
    "summarization",
    "abstention",
)
MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"),
        start=1,
    )
}


def parse_literal(value: object) -> Any:
    """BEAM stores lists and dicts as their Python repr inside strings."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def parse_anchor(raw: object) -> str:
    """`March-15-2024` into an ISO date the memory runtime already parses."""
    match = re.search(r"([A-Za-z]+)[-\s](\d{1,2})[-,\s]+(\d{4})", str(raw or ""))
    if not match:
        return ""
    month = MONTHS.get(match.group(1).lower())
    if not month:
        return ""
    try:
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    except ValueError:
        return ""


def convert_conversation(row: dict[str, Any], sample_id: str) -> dict[str, Any]:
    conversation: dict[str, Any] = {
        "speaker_a": "User",
        "speaker_b": "Assistant",
    }
    for number, session in enumerate(row["chat"], start=1):
        turns = []
        observed = ""
        for position, message in enumerate(session, start=1):
            observed = observed or parse_anchor(message.get("time_anchor"))
            turns.append({
                "speaker": (
                    "User" if message.get("role") == "user" else "Assistant"
                ),
                "dia_id": f"D{number}:{position}",
                "text": str(message.get("content", "")),
                # BEAM's own message id, kept so a memory can be traced back
                # to the row it came from.
                "beam_id": str(message.get("id", "")),
            })
        conversation[f"session_{number}"] = turns
        conversation[f"session_{number}_date_time"] = observed
    return conversation


def convert_questions(row: dict[str, Any]) -> list[dict[str, Any]]:
    probing = parse_literal(row["probing_questions"])
    if not isinstance(probing, dict):
        raise ValueError("probing_questions is not a category mapping")
    unknown = set(probing) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown BEAM categories: {sorted(unknown)}")
    questions = []
    for number, name in enumerate(CATEGORIES, start=1):
        for item in probing.get(name, []):
            # Abstention questions carry an ideal_response instead of an
            # answer; both are the text the judge compares against.
            gold = item.get("answer") or item.get("ideal_response") or ""
            questions.append({
                "question": str(item.get("question", "")),
                "answer": str(gold),
                "category": number,
                "beam_category": name,
                "abstention": name == "abstention",
                "rubric": parse_literal(item.get("rubric", [])),
                "difficulty": str(item.get("difficulty", "")),
                "evidence": [],
            })
    return questions


def convert(size: str, conversation_id: str) -> dict[str, Any]:
    from datasets import Dataset

    path = ARROW_DIR / f"beam-{size}.arrow"
    if not path.is_file():
        raise FileNotFoundError(f"BEAM data missing: {path}")
    dataset = Dataset.from_file(str(path))
    matches = [
        row for row in dataset
        if str(row["conversation_id"]) == str(conversation_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one BEAM conversation {conversation_id!r} in {size}, "
            f"found {len(matches)}"
        )
    row = matches[0]
    sample_id = f"beam{size}-{conversation_id}"
    return {
        "sample_id": sample_id,
        "conversation": convert_conversation(row, sample_id),
        "qa": convert_questions(row),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=SIZES, default="100K")
    parser.add_argument("--conversation", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    sample = convert(args.size, args.conversation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([sample], ensure_ascii=False), encoding="utf-8"
    )
    sessions = [
        value for key, value in sample["conversation"].items()
        if re.fullmatch(r"session_\d+", key)
    ]
    print(json.dumps({
        "sample_id": sample["sample_id"],
        "sessions": len(sessions),
        "messages": sum(len(session) for session in sessions),
        "characters": sum(
            len(turn["text"]) for session in sessions for turn in session
        ),
        "questions": len(sample["qa"]),
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
