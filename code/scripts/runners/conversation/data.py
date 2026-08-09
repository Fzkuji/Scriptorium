"""Sample loading and inventory, for any benchmark shaped as sessions."""

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
            f"expected one sample {sample_id!r}, found {len(matches)}"
        )
    return matches[0]


def check_benchmark(sample: dict[str, Any], benchmark: str) -> None:
    """Refuse a sample whose questions are not the ones this benchmark scores.

    The two shapes look alike enough to run: a BEAM sample scored as LoCoMo
    silently keeps the four questions numbered 1-4 and reports them under
    LoCoMo's category names, and a LoCoMo sample scored as BEAM judges its
    unanswerable questions against the distractor they carry.
    """
    beam_shaped = any("beam_category" in question for question in sample["qa"])
    if benchmark == "beam" and not beam_shaped:
        raise ValueError(
            f"--benchmark beam needs questions carrying beam_category; "
            f"{sample['sample_id']} has none. Convert it first with "
            f"python -m scripts.runners.beam.convert"
        )
    if benchmark != "beam" and beam_shaped:
        raise ValueError(
            f"{sample['sample_id']} carries BEAM questions; "
            f"run it with --benchmark beam"
        )


def sample_inventory(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = sample["conversation"]
    sessions = [
        value
        for key, value in conversation.items()
        if re.fullmatch(r"session_[0-9]+", key)
    ]
    questions = sample["qa"]
    # A benchmark that marks its unanswerable questions says so; LoCoMo says
    # it by numbering them 5, which in another benchmark means something else.
    flagged = any("abstention" in question for question in questions)
    unanswerable = sum(
        bool(question["abstention"]) if flagged else int(question["category"]) == 5
        for question in questions
    )
    return {
        "sample_id": sample["sample_id"],
        "sessions": len(sessions),
        "messages": sum(len(session) for session in sessions),
        "questions": len(questions),
        "primary_questions": len(questions) - unanswerable,
        "adversarial_questions": unanswerable,
    }
