"""Pass rule and capacity-curve consistency for Writer calibration.

Each case reproduces a failure mode observed in the 2026-08-03 deepseek runs.
"""

import json
from pathlib import Path

import pytest

from scripts.model_capacity.calibrate_writer import evaluate_probe
from memory.runtime.capacity import (
    SCHEMA,
    WriterCapacity,
    find_capacity_inversions,
)

REFS = {"calibration/a-s001/msg-001"}
FACTS = {"ARCHIVE-ABCDEF012345"}


def write_memory(tmp_path: Path, body: str) -> Path:
    topics = tmp_path / "topics"
    topics.mkdir(parents=True, exist_ok=True)
    (topics / "notes.md").write_text(body, encoding="utf-8")
    return tmp_path


def test_full_coverage_passes_without_newly_created_blocks(tmp_path: Path):
    """A trial that extended existing blocks used to fail at coverage 1.0."""
    memory = write_memory(
        tmp_path, "calibration/a-s001/msg-001 ARCHIVE-ABCDEF012345\n"
    )

    result = evaluate_probe(
        memory,
        # status ok but count 0: the agent edited rather than created.
        audit=[{"status": "ok", "count": 0}],
        expected_refs=REFS,
        expected_facts=FACTS,
    )

    assert result["passed"] is True
    assert result["written_blocks"] == 0


def test_fabricated_fact_fails_even_with_full_coverage(tmp_path: Path):
    memory = write_memory(
        tmp_path,
        "calibration/a-s001/msg-001 ARCHIVE-ABCDEF012345\n"
        "ARCHIVE-999999999999 invented\n",
    )

    result = evaluate_probe(
        memory,
        audit=[{"status": "ok", "count": 1}],
        expected_refs=REFS,
        expected_facts=FACTS,
    )

    assert result["passed"] is False
    assert result["fact_coverage"] == 1.0
    assert result["fabricated_fact_count"] == 1
    assert result["fact_precision"] == 0.5


def test_partial_coverage_after_turn_limit_is_inconclusive(tmp_path: Path):
    memory = write_memory(tmp_path, "nothing useful\n")

    result = evaluate_probe(
        memory,
        audit=[{"status": "stopped"}],
        expected_refs=REFS,
        expected_facts=FACTS,
    )

    assert result["passed"] is False
    # Ran out of turns; says nothing about how much input it can handle.
    assert result["inconclusive"] is True


def test_full_coverage_is_not_inconclusive_even_if_turns_ran_out(tmp_path: Path):
    memory = write_memory(
        tmp_path, "calibration/a-s001/msg-001 ARCHIVE-ABCDEF012345\n"
    )

    result = evaluate_probe(
        memory,
        audit=[{"status": "stopped"}],
        expected_refs=REFS,
        expected_facts=FACTS,
    )

    assert result["passed"] is True
    assert result["inconclusive"] is False


def level(tokens: int, *, passing: bool, inconclusive: int = 0, trials: int = 2):
    value = 1.0 if passing else 0.0
    return {
        "candidate_tokens": tokens,
        "trials": trials,
        "pass_rate": value,
        "source_coverage_min": value,
        "fact_coverage_min": value,
        "inconclusive_trials": inconclusive,
    }


def test_sawtooth_curve_is_reported_as_an_inversion():
    """The observed 4096 fail / 32768 pass shape is not a capacity curve."""
    levels = [
        level(4096, passing=False),
        level(16384, passing=False),
        level(32768, passing=True),
    ]

    inversions = find_capacity_inversions(levels)

    assert (4096, 32768) in inversions
    assert (16384, 32768) in inversions


def test_monotonic_curve_has_no_inversions():
    levels = [
        level(4096, passing=True),
        level(16384, passing=True),
        level(32768, passing=False),
    ]

    assert find_capacity_inversions(levels) == []


def test_levels_cut_off_by_the_turn_budget_are_not_inversions():
    levels = [
        level(4096, passing=False, inconclusive=2, trials=2),
        level(32768, passing=True),
    ]

    assert find_capacity_inversions(levels) == []


def artifact(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "model": "m",
        "writer_protocol_sha256": "h",
        "safe_input_tokens": 8192,
        "tokenizer": {"implementation": "tiktoken"},
        **overrides,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_usable_artifact_loads(tmp_path: Path):
    loaded = WriterCapacity.load(
        artifact(tmp_path), model="m", writer_protocol_sha256="h"
    )

    assert loaded.safe_input_tokens == 8192


def test_inconsistent_artifact_is_refused(tmp_path: Path):
    path = artifact(tmp_path, status="inconsistent")

    with pytest.raises(ValueError, match="not usable"):
        WriterCapacity.load(path, model="m", writer_protocol_sha256="h")


def test_artifact_reporting_inversions_is_refused(tmp_path: Path):
    path = artifact(tmp_path, capacity_inversions=[
        {"failing_candidate_tokens": 4096, "passing_candidate_tokens": 32768}
    ])

    with pytest.raises(ValueError, match="inversion"):
        WriterCapacity.load(path, model="m", writer_protocol_sha256="h")
