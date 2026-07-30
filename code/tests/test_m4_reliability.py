from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_locomo_human_packet import (
    allocate_categories,
    build_payloads,
    write_packet,
)
from scripts.freeze_m4_paired_inputs import calculate as calculate_pairings
from scripts.run_m4_failure_analysis import RULE_VERSION, calculate as calculate_failures
from scripts.run_m4_statistics import calculate as calculate_statistics
from scripts.run_m4_statistics import atomic_json_no_clobber
from scripts.score_locomo_human_agreement import calculate as calculate_agreement
from src.evaluation.m4_reliability import (
    ReliabilityError,
    analyze_comparison,
    attribute_failure,
    exact_mcnemar,
    holm_adjust,
    select_qualitative_cases,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def paired_records() -> list[dict[str, object]]:
    return [
        {"question_id": "s0_q0", "cluster_id": "s0", "left": 1, "right": 0},
        {"question_id": "s0_q1", "cluster_id": "s0", "left": 1, "right": 1},
        {"question_id": "s1_q0", "cluster_id": "s1", "left": 0, "right": 1},
        {"question_id": "s1_q1", "cluster_id": "s1", "left": 0, "right": 0},
    ]


def test_exact_mcnemar_and_cluster_bootstrap_are_deterministic() -> None:
    first = analyze_comparison(paired_records(), kind="binary", repetitions=100, seed=7)
    second = analyze_comparison(paired_records(), kind="binary", repetitions=100, seed=7)
    assert first == second
    assert first["paired_clustered_bootstrap"]["estimate"] == 0
    assert exact_mcnemar(paired_records())["p_value"] == 1.0


def test_mcnemar_rejects_nonbinary() -> None:
    records = paired_records()
    records[0]["left"] = 0.5
    with pytest.raises(ReliabilityError, match="binary"):
        exact_mcnemar(records)


def test_holm_adjustment_is_monotone_and_name_stable() -> None:
    adjusted = holm_adjust([("b", 0.02), ("a", 0.01), ("c", 0.5)])
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}


def test_statistics_calculate_adds_holm_within_family(tmp_path: Path) -> None:
    source = tmp_path / "paired.json"
    write_json(
        source,
        {
            "schema_version": 1,
            "status": "frozen",
            "analysis_id": "test",
            "comparisons": [
                {
                    "comparison_id": "a",
                    "family": "main",
                    "kind": "binary",
                    "left_method": "x",
                    "right_method": "y",
                    "metric": "correct",
                    "records": paired_records(),
                },
                {
                    "comparison_id": "cost",
                    "family": "cost",
                    "kind": "continuous",
                    "records": [
                        {**record, "left": index + 1, "right": index}
                        for index, record in enumerate(paired_records())
                    ],
                },
            ],
        },
    )
    result = calculate_statistics(source, repetitions=50, base_seed=11)
    assert result["results"][0]["statistics"]["exact_mcnemar"][
        "holm_adjusted_p_value"
    ] == 1.0
    assert "exact_mcnemar" not in result["results"][1]["statistics"]


def test_statistics_output_never_clobbers(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    atomic_json_no_clobber(output, {"value": 1})
    with pytest.raises(ReliabilityError, match="overwrite"):
        atomic_json_no_clobber(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}


def test_pairing_freeze_binds_artifacts_questions_clusters_and_metrics(
    tmp_path: Path,
) -> None:
    method_paths: list[Path] = []
    for method_index in range(2):
        path = tmp_path / f"method-{method_index}.json"
        write_json(
            path,
            {
                "meta": {"status": "complete", "benchmark": "locomo"},
                "results": [
                    {
                        "question_id": f"s{sample}_q0",
                        "judge_score": int((sample + method_index) % 2 == 0),
                        "retrieval": {"calls": sample + method_index + 1},
                    }
                    for sample in range(2)
                ],
            },
        )
        method_paths.append(path)
    spec = tmp_path / "spec.json"
    methods = []
    for method_index, path in enumerate(method_paths):
        methods.append(
            {
                "method_id": f"m{method_index}",
                "artifact": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_questions": 2,
                "required_meta": {"benchmark": "locomo"},
                "cluster_rule": "locomo_question_id",
                "metrics": {
                    "correct": "judge_score",
                    "calls": "retrieval.calls",
                },
            }
        )
    write_json(
        spec,
        {
            "schema_version": 1,
            "status": "frozen",
            "analysis_id": "pairing-test",
            "methods": methods,
            "comparisons": [
                {
                    "comparison_id": "accuracy",
                    "family": "main",
                    "left_method": "m0",
                    "right_method": "m1",
                    "metric": "correct",
                    "kind": "binary",
                },
                {
                    "comparison_id": "calls",
                    "family": "cost",
                    "left_method": "m0",
                    "right_method": "m1",
                    "metric": "calls",
                    "kind": "continuous",
                },
            ],
        },
    )
    result = calculate_pairings(spec)
    assert len(result["comparisons"]) == 2
    assert result["comparisons"][0]["records"][0]["cluster_id"] == "s0"
    method_paths[0].write_text(method_paths[0].read_text() + " ", encoding="utf-8")
    with pytest.raises(ReliabilityError, match="hash changed"):
        calculate_pairings(spec)


def failure_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "question_id": "s0_q0",
        "gold_source_mapping_complete": True,
        "score_correct": False,
        "primary_sensitivity_disagree": False,
        "gold_source_in_canonical_entries": True,
        "gold_source_survived_maintenance": True,
        "gold_source_path_valid": True,
        "retrieval_reached_gold_source": True,
        "source_resolution_returned_gold_content": True,
        "trace_ids": {"retrieval": ["trace-1"]},
    }
    record.update(updates)
    return record


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"gold_source_in_canonical_entries": False}, "extraction_omission"),
        ({"gold_source_survived_maintenance": False}, "maintenance_error"),
        ({"gold_source_path_valid": False}, "wrong_path"),
        ({"retrieval_reached_gold_source": False}, "navigation_miss"),
        ({"source_resolution_returned_gold_content": False}, "source_resolution_error"),
        ({"primary_sensitivity_disagree": True}, "judge_ambiguity"),
        ({}, "answer_error"),
    ],
)
def test_failure_rule_uses_earliest_stage(
    updates: dict[str, object], expected: str
) -> None:
    assert attribute_failure(failure_record(**updates))["primary_label"] == expected


def test_wrong_path_precedes_later_maintenance_failure() -> None:
    result = attribute_failure(
        failure_record(
            gold_source_path_valid=False,
            gold_source_survived_maintenance=False,
        )
    )
    assert result["primary_label"] == "wrong_path"
    assert "maintenance_error" in result["secondary_tags"]


def test_failure_rule_excludes_incomplete_mapping() -> None:
    result = attribute_failure(failure_record(gold_source_mapping_complete=False))
    assert result["eligible"] is False
    assert result["primary_label"] is None


def test_qualitative_selection_uses_hash_order() -> None:
    labels = [
        attribute_failure(failure_record(question_id=f"s0_q{i}")) for i in range(5)
    ]
    first = select_qualitative_cases(labels, per_label=2, seed="fixed")
    second = select_qualitative_cases(list(reversed(labels)), per_label=2, seed="fixed")
    assert first == second
    assert len(first) == 2


def test_failure_calculate_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "failure.json"
    write_json(
        source,
        {
            "schema_version": 1,
            "status": "frozen",
            "rule_version": RULE_VERSION,
            "records": [
                failure_record(question_id="s0_q0"),
                failure_record(question_id="s1_q0", score_correct=True),
            ],
        },
    )
    result = calculate_failures(source, per_label=1)
    assert result["records"] == 2
    assert result["primary_failure_counts"]["answer_error"] == 1
    assert result["correct_without_primary_failure"] == 1


def fake_locomo_score() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for sample in range(10):
        for offset in range(154):
            category = offset % 4 + 1
            records.append(
                {
                    "question_id": f"s{sample}_q{offset}",
                    "question": f"Question {sample}-{offset}?",
                    "gold": f"Gold {sample}-{offset}",
                    "answer": f"Answer {sample}-{offset}",
                    "category": category,
                    "judge_score": int(offset % 3 != 0),
                }
            )
    return {
        "meta": {
            "status": "complete",
            "benchmark": "locomo",
            "judge_profile": "protocol-primary-openrouter-gpt4o-mini",
        },
        "results": records,
    }


def test_category_allocation_is_exact_and_has_minimum() -> None:
    allocation = allocate_categories({1: 80, 2: 40, 3: 20, 4: 10})
    assert sum(allocation.values()) == 10
    assert all(value >= 1 for value in allocation.values())


def test_human_packet_is_deterministic_and_blinded(tmp_path: Path) -> None:
    source = tmp_path / "scores.json"
    write_json(source, fake_locomo_score())
    packet, key, templates = build_payloads(source)
    assert len(packet["items"]) == len(key["items"]) == len(templates) == 100
    public = json.dumps(packet)
    assert "question_id" not in public
    assert "judge_score" not in public
    output = tmp_path / "packet"
    manifest = write_packet(source, output, seed="locomo-human-validation-v1")
    assert manifest["external_labels_required"] == 2
    assert (output / "public/annotator_A.csv").read_bytes() == (
        output / "public/annotator_B.csv"
    ).read_bytes()


def fill_labels(template: Path, output: Path, *, alternate: bool = False) -> None:
    with template.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        row["verdict"] = "correct" if not alternate or index % 2 == 0 else "incorrect"
        row["confidence"] = "4"
        row["rationale"] = "independent label"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_human_agreement_requires_two_complete_label_sets(tmp_path: Path) -> None:
    source = tmp_path / "scores.json"
    write_json(source, fake_locomo_score())
    packet_dir = tmp_path / "packet"
    write_packet(source, packet_dir, seed="locomo-human-validation-v1")
    labels_a = tmp_path / "a.csv"
    labels_b = tmp_path / "b.csv"
    fill_labels(packet_dir / "public/annotator_A.csv", labels_a)
    fill_labels(packet_dir / "public/annotator_B.csv", labels_b, alternate=True)
    result = calculate_agreement(
        packet_dir / "private/private_key.json",
        labels_a,
        labels_b,
        annotator_a="external-a",
        annotator_b="external-b",
    )
    assert result["three_class_agreement"]["n"] == 100
    assert result["binary_agreement_excluding_any_uncertain"]["n"] == 100
    with pytest.raises(ReliabilityError, match="distinct"):
        calculate_agreement(
            packet_dir / "private/private_key.json",
            labels_a,
            labels_b,
            annotator_a="same",
            annotator_b="same",
        )


def test_independent_auditors_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.audit_locomo_human_packet import main as audit_packet_main
    from scripts.audit_m4_failure_analysis import main as audit_failure_main
    from scripts.audit_m4_statistics import main as audit_statistics_main

    paired_source = tmp_path / "paired.json"
    write_json(
        paired_source,
        {
            "schema_version": 1,
            "status": "frozen",
            "analysis_id": "audit-test",
            "comparisons": [
                {
                    "comparison_id": "binary",
                    "family": "main",
                    "kind": "binary",
                    "records": paired_records(),
                }
            ],
        },
    )
    statistics_result = tmp_path / "statistics.json"
    write_json(
        statistics_result,
        calculate_statistics(paired_source, repetitions=20, base_seed=3),
    )
    statistics_audit = tmp_path / "statistics-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_m4_statistics.py",
            "--input",
            str(paired_source),
            "--result",
            str(statistics_result),
            "--output",
            str(statistics_audit),
        ],
    )
    assert audit_statistics_main() == 0
    assert json.loads(statistics_audit.read_text())["status"] == "pass"

    failure_source = tmp_path / "failure.json"
    write_json(
        failure_source,
        {
            "schema_version": 1,
            "status": "frozen",
            "rule_version": RULE_VERSION,
            "records": [failure_record()],
        },
    )
    failure_result = tmp_path / "failure-result.json"
    write_json(failure_result, calculate_failures(failure_source, per_label=1))
    failure_audit = tmp_path / "failure-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_m4_failure_analysis.py",
            "--input",
            str(failure_source),
            "--result",
            str(failure_result),
            "--output",
            str(failure_audit),
        ],
    )
    assert audit_failure_main() == 0
    assert json.loads(failure_audit.read_text())["status"] == "pass"

    score_source = tmp_path / "score.json"
    write_json(score_source, fake_locomo_score())
    packet_dir = tmp_path / "human-packet"
    write_packet(score_source, packet_dir, seed="audit-packet")
    packet_audit = tmp_path / "packet-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_locomo_human_packet.py",
            "--source",
            str(score_source),
            "--packet-dir",
            str(packet_dir),
            "--output",
            str(packet_audit),
        ],
    )
    assert audit_packet_main() == 0
    assert json.loads(packet_audit.read_text())["status"] == "pass"
