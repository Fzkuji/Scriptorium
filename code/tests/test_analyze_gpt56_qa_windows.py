from __future__ import annotations

from pathlib import Path

import pytest


def _row(
    benchmark: str,
    question_id: str,
    score: float,
    *,
    correct: bool | None,
    passed: bool,
    visible_tokens: int,
) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "unit_id": "unit-1",
        "original_question_id": question_id,
        "question_sha256": f"hash-{question_id}",
        "quality": {
            "score": score,
            "correct": correct,
            "pass": passed,
        },
        "access": {
            "visible_tokens": visible_tokens,
            "retrieval_model_calls": 2,
            "accepted_logical_calls": 3,
        },
    }


def test_exact_mcnemar_p_is_two_sided() -> None:
    from scripts import analyze_gpt56_qa_windows as analyze

    assert analyze.exact_mcnemar_p(0, 0) == 1.0
    assert analyze.exact_mcnemar_p(1, 1) == 1.0
    assert analyze.exact_mcnemar_p(0, 6) == pytest.approx(0.03125)


def test_compare_cells_uses_paired_questions_and_candidate_minus_reference() -> None:
    from scripts import analyze_gpt56_qa_windows as analyze

    reference = [
        _row("locomo", "q0", 1.0, correct=True, passed=True, visible_tokens=100),
        _row("locomo", "q1", 0.0, correct=False, passed=False, visible_tokens=120),
        _row("beam-100k", "b0", 0.25, correct=None, passed=False, visible_tokens=80),
        _row("longmemeval-s", "l0", 0.0, correct=False, passed=False, visible_tokens=60),
    ]
    candidate = [
        _row("locomo", "q0", 0.0, correct=False, passed=False, visible_tokens=90),
        _row("locomo", "q1", 1.0, correct=True, passed=True, visible_tokens=100),
        _row("beam-100k", "b0", 0.75, correct=None, passed=True, visible_tokens=70),
        _row("longmemeval-s", "l0", 1.0, correct=True, passed=True, visible_tokens=50),
    ]

    result = analyze.compare_cells(reference, candidate)

    assert result["locomo_delta"] == 0.0
    assert result["locomo_candidate_only"] == 1
    assert result["locomo_reference_only"] == 1
    assert result["locomo_mcnemar_p"] == 1.0
    assert result["beam_delta"] == 0.5
    assert result["beam_candidate_only_pass"] == 1
    assert result["longmemeval_correct_delta"] == 1
    assert result["visible_tokens_delta"] == pytest.approx(-12.5)


def test_compare_cells_rejects_unpaired_question_identity() -> None:
    from scripts import analyze_gpt56_qa_windows as analyze

    reference = [
        _row("locomo", "q0", 1.0, correct=True, passed=True, visible_tokens=100)
    ]
    candidate = [
        {
            **_row(
                "locomo",
                "q0",
                1.0,
                correct=True,
                passed=True,
                visible_tokens=100,
            ),
            "question_sha256": "different",
        }
    ]

    with pytest.raises(analyze.QAWindowAnalysisError, match="identity"):
        analyze.compare_cells(reference, candidate)


def test_cell_spec_reuses_only_existing_terra_sol_w32(tmp_path: Path) -> None:
    from scripts import analyze_gpt56_qa_windows as analyze

    terra = analyze.cell_spec(
        "terra",
        32,
        new_qa_root=tmp_path / "new-qa",
        new_score_root=tmp_path / "new-scores",
        old_qa_root=tmp_path / "old-qa",
        old_score_root=tmp_path / "old-scores",
    )
    luna = analyze.cell_spec(
        "luna",
        32,
        new_qa_root=tmp_path / "new-qa",
        new_score_root=tmp_path / "new-scores",
        old_qa_root=tmp_path / "old-qa",
        old_score_root=tmp_path / "old-scores",
    )

    assert terra["qa_root"] == tmp_path / "old-qa" / "terra-r5"
    assert terra["score_root"] == tmp_path / "old-scores"
    assert luna["qa_root"] == tmp_path / "new-qa" / "luna-w32"
    assert luna["score_root"] == tmp_path / "new-scores" / "luna-w32"


def test_latex_outputs_keep_benchmark_results_in_separate_tables() -> None:
    from scripts import analyze_gpt56_qa_windows as analyze

    rows = []
    for tier in analyze.TIERS:
        for window in analyze.WINDOWS:
            rows.append(
                {
                    "tier": tier,
                    "write_turns": window,
                    "locomo_score": 0.9,
                    "longmemeval_correct": 2,
                    "beam_score": 0.6,
                }
            )

    tables = analyze.latex_tables(rows)

    assert set(tables) == {
        "qa_locomo_window_table.tex",
        "qa_longmemeval_window_table.tex",
        "qa_beam_window_table.tex",
    }
    assert "LoCoMo" in tables["qa_locomo_window_table.tex"]
    assert "BEAM" not in tables["qa_locomo_window_table.tex"]
    assert "LongMemEval-S" in tables["qa_longmemeval_window_table.tex"]
    assert "BEAM" in tables["qa_beam_window_table.tex"]
