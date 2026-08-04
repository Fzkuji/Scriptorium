"""Reading a comparison score, including the half-written file case."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from baselines.read_score import read_score  # noqa: E402

SCRIPT = CODE / "baselines" / "read_score.py"


def write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload))
    return path


def test_locomo_shape(tmp_path: Path):
    path = write(tmp_path, {"records": [
        {"judge_score": 1}, {"judge_score": 0}, {"judge_score": 1},
    ]})
    assert read_score(path) == (pytest.approx(66.67, abs=0.01), 3)


def test_longmemeval_shape(tmp_path: Path):
    """LongMemEval reports results/, not records/."""
    path = write(tmp_path, {"results": [{"judge_score": 1}, {"judge_score": 1}]})
    assert read_score(path) == (100.0, 2)


def test_unjudged_records_are_not_counted(tmp_path: Path):
    """evaluate.py writes answers before judging; those must not score 0."""
    path = write(tmp_path, {"records": [
        {"judge_score": 1}, {"answer": "Tokyo"}, {"answer": "Paris"},
    ]})
    assert read_score(path) == (100.0, 1)


def test_a_run_that_died_before_judging_is_an_error(tmp_path: Path):
    """This is what tells the sweep a system failed — the file exists but is unscored."""
    path = write(tmp_path, {"records": [{"answer": "Tokyo"}, {"answer": "Paris"}]})
    with pytest.raises(ValueError):
        read_score(path)


def test_cli_exits_non_zero_when_nothing_was_judged(tmp_path: Path):
    path = write(tmp_path, {"records": [{"answer": "Tokyo"}]})
    assert subprocess.run([sys.executable, str(SCRIPT), str(path)],
                          capture_output=True).returncode == 1


def test_cli_prints_score_and_count(tmp_path: Path):
    path = write(tmp_path, {"records": [{"judge_score": 1}, {"judge_score": 0}]})
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["50.0", "2"]
