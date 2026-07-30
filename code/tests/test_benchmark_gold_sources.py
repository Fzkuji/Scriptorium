import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load_script("freeze_benchmark_gold_sources.py")
AUDIT = load_script("audit_benchmark_gold_sources.py")


def test_frozen_locomo_normalization_is_explicit():
    assert FREEZE.normalize_anchor("D1:2") == (["D1:2"], "identity")
    assert FREEZE.normalize_anchor("D30:05") == (
        ["D30:5"],
        "strip_leading_zero",
    )
    assert FREEZE.normalize_anchor("D:11:26") == (
        ["D11:26"],
        "frozen_typo_D_colon_session_colon_turn",
    )
    assert FREEZE.normalize_anchor("D8:6; D9:17") == (
        ["D8:6", "D9:17"],
        "split_compound_anchor",
    )
    assert FREEZE.normalize_anchor("D9:1 D4:4 D4:6") == (
        ["D9:1", "D4:4", "D4:6"],
        "split_compound_anchor",
    )
    assert FREEZE.normalize_anchor("D") == ([], "unsupported_anchor_syntax")


def test_recursive_beam_source_ids_are_deterministic():
    raw = {
        "second": [[3, " 4 "]],
        "first": [1, {"nested": [2, 1]}],
    }
    assert FREEZE.flatten_beam_source_ids(raw) == [
        (1, "1", "integer_to_string"),
        (2, "2", "integer_to_string"),
        (1, "1", "integer_to_string"),
        (3, "3", "integer_to_string"),
        (" 4 ", "4", "strip_whitespace"),
    ]
    with pytest.raises(FREEZE.FreezeError, match="boolean"):
        FREEZE.flatten_beam_source_ids([True])


def test_collision_lock_and_symlink_guards(tmp_path):
    artifact = tmp_path / "artifact.json"
    assert FREEZE.collision_safe_atomic_write(artifact, b"one\n") == "written"
    assert FREEZE.collision_safe_atomic_write(artifact, b"one\n") == "unchanged"
    with pytest.raises(FREEZE.FreezeError, match="overwrite"):
        FREEZE.collision_safe_atomic_write(artifact, b"two\n")

    symlink = tmp_path / "artifact-link.json"
    symlink.symlink_to(artifact)
    with pytest.raises(FREEZE.FreezeError, match="symlink"):
        FREEZE.collision_safe_atomic_write(symlink, b"one\n")

    lock_dir = tmp_path / "locked"
    with FREEZE.exclusive_output_lock(lock_dir):
        with pytest.raises(FREEZE.FreezeError, match="locked"):
            with FREEZE.exclusive_output_lock(lock_dir):
                pass


def test_full_frozen_bundle_and_independent_audit(tmp_path):
    output_dir = tmp_path / "evidence-mapping-v1"
    assert FREEZE.main(["--output-dir", str(output_dir)]) == 0
    # An identical rerun is idempotent rather than silently replacing files.
    assert FREEZE.main(["--output-dir", str(output_dir)]) == 0

    report = AUDIT.audit(artifact_dir=output_dir)
    assert report["status"] == "passed"
    assert report["question_count"] == 3386
    assert report["source_recall_question_denominators"] == {
        "LoCoMo": 1533,
        "LongMemEval-S": 500,
        "BEAM": 804,
    }

    manifest = json.loads(
        (output_dir / FREEZE.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    locomo = manifest["benchmarks"]["LoCoMo"]["counts"]
    assert locomo["primary_qa_questions"] == 1540
    assert locomo["qa_scoring_question_denominator"] == 1540
    assert {
        entry["question_id"] for entry in locomo["source_recall_exclusions"]
    } == {
        "locomo:conv-26:q030",
        "locomo:conv-26:q046",
        "locomo:conv-42:q058",
        "locomo:conv-42:q088",
        "locomo:conv-47:q038",
        "locomo:conv-50:q039",
        "locomo:conv-50:q042",
    }
    assert locomo["normalization_reason_counts"] == {
        "frozen_typo_D_colon_session_colon_turn": 1,
        "identity": 2808,
        "split_compound_anchor": 4,
        "strip_leading_zero": 1,
        "unsupported_anchor_syntax": 1,
    }

    lme = manifest["benchmarks"]["LongMemEval-S"]["counts"]
    assert lme["turn_level_gold_mapping"] is False
    assert lme["gold_source_granularity"] == "session"
    assert lme["mapping_level"] == "session"
    assert lme["items_with_duplicate_haystack_session_ids"] == 13

    beam = manifest["benchmarks"]["BEAM"]["counts"]
    assert beam["questions"] == 900
    assert beam["empty_question_types"] == {
        "abstention": 90,
        "preference_following": 2,
        "summarization": 4,
    }
    assert beam["formal_scope"]["100K"]["conversation_indices"] == list(
        range(10)
    )
    assert beam["formal_scope"]["1M"]["conversation_indices"] == list(range(35))

    records = [
        json.loads(line)
        for line in (output_dir / FREEZE.QUESTIONS_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    beam_empty = [
        record
        for record in records
        if record["benchmark"] == "BEAM"
        and not record["source_recall_eligible"]
    ]
    assert len(beam_empty) == 96
    assert sum(
        record["source_recall_empty_class"] == "abstention"
        for record in beam_empty
    ) == 90
    assert sum(
        record["source_recall_empty_class"] == "other" for record in beam_empty
    ) == 6

    questions_path = output_dir / FREEZE.QUESTIONS_NAME
    questions_path.write_bytes(questions_path.read_bytes() + b"{}\n")
    with pytest.raises(AUDIT.AuditError, match="question hash"):
        AUDIT.audit(artifact_dir=output_dir)
