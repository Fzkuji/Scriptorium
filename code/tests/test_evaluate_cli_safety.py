import json

import pytest

from src.evaluation import evaluate


def _input(path):
    path.write_text(
        json.dumps([{
            "question_id": "q0",
            "question": "question",
            "gold": "answer",
            "answer": "answer",
            "question_type": "multi-session",
            "abstention": False,
        }]),
        encoding="utf-8",
    )


def _argv(input_path, output_path):
    return [
        "--benchmark", "longmemeval",
        "--input", str(input_path),
        "--output", str(output_path),
        "--metrics", "em",
        "--skip-answerer",
        "--skip-judge",
    ]


@pytest.mark.parametrize("alias_kind", ["same", "hardlink", "symlink"])
def test_input_output_alias_is_rejected_without_overwrite(tmp_path, alias_kind):
    source = tmp_path / "input.json"
    _input(source)
    before = source.read_bytes()
    if alias_kind == "same":
        output = source
    elif alias_kind == "hardlink":
        output = tmp_path / "hard.json"
        output.hardlink_to(source)
    else:
        output = tmp_path / "link.json"
        output.symlink_to(source)

    with pytest.raises(ValueError, match="collision"):
        evaluate.main(_argv(source, output))
    assert source.read_bytes() == before


def test_case_and_unicode_aliases_are_rejected(tmp_path):
    source = tmp_path / "Café.JSON"
    _input(source)
    with pytest.raises(ValueError, match="collision"):
        evaluate.main(_argv(source, tmp_path / "café.json"))
    with pytest.raises(ValueError, match="collision"):
        evaluate._ensure_distinct_paths({
            "input": tmp_path / "café.json",
            "output": tmp_path / "café.json",
        })


def test_evaluate_lock_is_single_instance(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _input(source)
    with evaluate.exclusive_output_lock(output):
        with pytest.raises(RuntimeError, match="already locked"):
            evaluate.main(_argv(source, output))


def test_evaluate_resumes_existing_output_atomically(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _input(source)
    evaluate.main(_argv(source, output))
    first = json.loads(output.read_text(encoding="utf-8"))
    assert first["meta"]["status"] == "complete"
    first["results"][0]["resume_marker"] = "preserved"
    evaluate.save(first["results"], first["meta"], output)

    evaluate.main(_argv(source, output))
    resumed = json.loads(output.read_text(encoding="utf-8"))
    assert resumed["meta"]["status"] == "complete"
    assert resumed["results"][0]["resume_marker"] == "preserved"
    assert json.loads(source.read_text(encoding="utf-8"))[0]["question_id"] == "q0"


def test_evaluate_resume_rejects_changed_limit(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    _input(source)
    evaluate.main([*_argv(source, output), "--limit", "1"])

    with pytest.raises(ValueError, match="metadata differs at limit"):
        evaluate.main(_argv(source, output))
