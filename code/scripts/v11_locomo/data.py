"""LoCoMo sample loading and inventory."""

import json
import re
from pathlib import Path
from typing import Any


def load_sample(path: Path, sample_id: str) -> tuple[int, dict[str, Any]]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    matches = [
        (index, sample)
        for index, sample in enumerate(dataset)
        if sample.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one LoCoMo sample {sample_id!r}, found {len(matches)}"
        )
    return matches[0]


def sample_inventory(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = sample["conversation"]
    sessions = [
        value
        for key, value in conversation.items()
        if re.fullmatch(r"session_[0-9]+", key)
    ]
    categories = [int(question["category"]) for question in sample["qa"]]
    return {
        "sample_id": sample["sample_id"],
        "sessions": len(sessions),
        "messages": sum(len(session) for session in sessions),
        "questions": len(categories),
        "primary_questions": sum(
            category in {1, 2, 3, 4} for category in categories
        ),
        "adversarial_questions": sum(category == 5 for category in categories),
    }
