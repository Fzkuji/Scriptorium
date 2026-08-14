from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_r203_readonly_views as auditor  # noqa: E402
import controlled_locomo_answer_contract as answer_contract  # noqa: E402
import r203_formal_contract as contract  # noqa: E402
import readonly_nativemem_control as control  # noqa: E402
import run_controlled_locomo_answers as controlled_answer  # noqa: E402
import run_r116_formal as r116_runner  # noqa: E402
import run_r203_readonly_views as runner  # noqa: E402


def _write_memory(path: Path, source_id: str) -> Path:
    (path / "topics").mkdir(parents=True)
    (path / "timeline/2025/01").mkdir(parents=True)
    (path / "topics/fact.md").write_text(
        f"Registered fixture evidence [{source_id}].\n", encoding="utf-8"
    )
    (path / "timeline/2025/01/01.md").write_text(
        f"Registered fixture timeline [{source_id}].\n", encoding="utf-8"
    )
    return path


def _fake_source_binding(memory: Path) -> dict:
    descriptor = control.memory_descriptor(memory)
    inventory = [
        {
            "sample_index": sample,
            "memory": descriptor,
            "sample_output": {
                "path": f"synthetic/sample{sample}_questions.json",
                "bytes": 1,
                "sha256": "0" * 64,
            },
        }
        for sample in contract.SAMPLES
    ]
    binding = {
        "schema_version": contract.SOURCE_BINDING_SCHEMA,
        "source_kind": "synthetic-test-only",
        "source_root": "synthetic-test-only",
        "run_manifest": {"synthetic": True},
        "independent_audit": {"synthetic": True},
        "source_inventory": inventory,
        "source_inventory_sha256": answer_contract.canonical_hash(inventory),
        "raw_dataset": contract.file_binding(contract.DATASET),
        "r002": {
            "manifest": {"sha256": contract.EXPECTED_MAPPING_MANIFEST_SHA256},
            "questions": {"sha256": contract.EXPECTED_MAPPING_QUESTIONS_SHA256},
            "audit": {"synthetic": True},
            "live_independent_audit": {"synthetic": True},
        },
        "source_questions": 1,
        "r203_primary_questions": contract.PRIMARY_QUESTIONS,
        "r203_condition_artifacts": contract.TOTAL_ARTIFACTS,
        "source_recall_denominator": contract.SOURCE_RECALL_DENOMINATOR,
        "provenance_claim_boundary": {
            "canonical_input": (
                "independently_audited_final_NativeMem_tree_at_R203_entry"
            ),
            "r203_maintenance_operation_count": 0,
            "does_not_claim": "NativeMem_builder_pre_maintenance_provenance",
        },
    }
    binding["binding_sha256"] = answer_contract.canonical_hash(binding)
    return binding


def test_v2_preregistration_changes_only_integrity_and_freezes_exact_scope():
    prereg = contract.validate_preregistration()

    assert prereg["schema_version"] == contract.PREREG_SCHEMA
    assert prereg["scope"]["primary_questions"] == 1540
    assert prereg["integrity"]["source_recall_denominator"] == 1533
    assert prereg["integrity_revision"]["experimental_design_changes"] == []
    assert (
        prereg["integrity_revision"]["previous_file_path"]
        == contract.relative_to_root(contract.SUPERSEDED_V1_PREREGISTRATION)
    )
    assert (
        answer_contract.sha256_file(contract.SUPERSEDED_V1_PREREGISTRATION)
        == contract.V1_PREREGISTRATION_FILE_SHA256
    )
    provenance = prereg["integrity"]["stage_provenance_interpretation"]
    assert provenance["r203_maintenance_operation_count"] == 0
    assert "does not establish" in provenance["claim_boundary"]

    specs = contract.question_specs()
    assert len(specs) == 1540
    assert sum(row["source_recall_eligible"] for row in specs) == 1533
    assert {
        category: sum(row["category"] == category for row in specs)
        for category in range(1, 5)
    } == contract.CATEGORY_COUNTS


def test_preflight_is_local_and_reports_only_concrete_source_blockers(tmp_path):
    report = contract.run_preflight(
        source_root=tmp_path / "missing-source", write_path=None
    )

    assert report["status"] == "blocked"
    assert report["model_requests"] == 0
    assert report["network_requests"] == 0
    assert report["scope"]["primary_questions"] == 1540
    assert report["scope"]["condition_artifacts"] == 6160
    assert report["scope"]["source_recall_denominator"] == 1533
    assert report["source"]["blockers"] == [
        f"source root is absent or unsafe: {tmp_path / 'missing-source'}"
    ]


def test_blocked_preflight_prevents_gateway_or_model_start(tmp_path, monkeypatch):
    blocked = {
        "schema_version": contract.PREFLIGHT_SCHEMA,
        "status": "blocked",
        "source": {"status": "blocked", "blockers": ["source incomplete"]},
        "model_requests": 0,
        "network_requests": 0,
    }
    blocked["preflight_content_sha256"] = answer_contract.canonical_hash(blocked)
    monkeypatch.setattr(contract, "run_preflight", lambda *args, **kwargs: blocked)
    acquired = False

    def forbidden(*args, **kwargs):
        nonlocal acquired
        acquired = True
        raise AssertionError("gateway must not be inspected for blocked source")

    monkeypatch.setattr(
        runner.controlled_answer.flex_evidence, "acquire_consumer_lock", forbidden
    )
    with pytest.raises(contract.R203FormalError, match="blocked"):
        runner.run_formal(
            output_dir=tmp_path / "formal",
            gateway_root=tmp_path / "gateway",
            allow_model_requests=True,
        )
    assert acquired is False


def test_one_formal_question_uses_fake_provider_and_reconstructs_m4(tmp_path):
    spec = contract.question_specs()[0]
    source_id = str(spec["gold_source_ids"][0])
    output = tmp_path / "formal-question"
    memory = _write_memory(
        output / "conditions/dual_source/memory_sample0", source_id
    )
    source_binding = _fake_source_binding(memory)
    copy_record = {
        "schema_version": "synthetic-copy-record",
        "rows": [],
    }
    copy_record["inventory_content_sha256"] = answer_contract.canonical_hash(
        copy_record
    )
    proxy_log = output / "synthetic_proxy.jsonl"
    run_id = "r203-formal-fake-provider"
    retrieval = r116_runner._ProxyLoggingScriptedCompletionResource(  # noqa: SLF001
        [
            [
                {
                    "name": "read_memory_file",
                    "arguments": {"path": "topics/fact.md"},
                }
            ],
            [
                {
                    "name": "resolve_sources",
                    "arguments": {"source_ids": [source_id]},
                }
            ],
            [],
        ],
        proxy_log=proxy_log,
        run_id=run_id,
    )
    answer_client = controlled_answer.FakeAnswerClient(
        proxy_log=proxy_log,
        run_id=run_id,
        response_text="<answer>fixture</answer>",
    )

    report = runner.execute_formal_question(
        output_dir=output,
        run_id=run_id,
        preregistration=contract.DEFAULT_PREREGISTRATION,
        source_binding=source_binding,
        copy_record=copy_record,
        condition="dual_source",
        spec=spec,
        completion_resource=retrieval,
        answer_client=answer_client,
        tokenizer=answer_contract.formal_token_counter(),
        proxy_log=proxy_log,
    )

    assert report["status"] == "passed"
    assert report["mapped_source_recall"] == 1.0
    assert report["stage_evidence"]["gold_source_in_canonical_entries"] is True
    assert report["stage_evidence"]["gold_source_survived_maintenance"] is True
    assert report["stage_evidence"]["retrieval_reached_gold_source"] is True
    stage = json.loads(
        (
            output
            / f"conditions/dual_source/questions/{spec['artifact_id']}"
            / "stage_provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert stage["maintenance"]["operation_count"] == 0
    assert "no_NativeMem_builder" in stage["r203_scope"]["claim_boundary"]
    assert "run_r203_readonly_views" not in auditor.__dict__


def test_all_40_condition_copies_are_byte_identical_per_sample(tmp_path):
    output = tmp_path / "copies"
    source_inventory = []
    for sample in contract.SAMPLES:
        source = _write_memory(
            tmp_path / "sources" / f"memory_sample{sample}", f"D1:{sample + 1}"
        )
        descriptor = control.memory_descriptor(source)
        source_inventory.append(
            {
                "sample_index": sample,
                "memory": descriptor,
                "sample_output": {"synthetic": True},
            }
        )
        for condition in contract.CONDITIONS:
            control.copy_memory_tree(
                source,
                contract.condition_memory_path(output, condition, sample),
            )
    binding = {
        "source_inventory": source_inventory,
        "source_inventory_sha256": answer_contract.canonical_hash(
            source_inventory
        ),
    }
    inventory = contract.copy_inventory(output)

    contract.validate_copy_inventory(
        output_dir=output, source_binding=binding, inventory=inventory
    )
    assert len(inventory) == 40
    for sample in contract.SAMPLES:
        assert len(
            {
                row["descriptor"]["sha256"]
                for row in inventory
                if row["sample_index"] == sample
            }
        ) == 1


def test_partial_question_and_orphan_proxy_fail_closed(tmp_path):
    output = tmp_path / "partial"
    partial = output / "conditions/dual_source/questions/unknown"
    partial.mkdir(parents=True)
    with pytest.raises(runner.R203Error, match="unknown|incomplete"):
        runner._audit_existing_formal_checkpoints(  # noqa: SLF001
            output_dir=output,
            specs=[],
            source_binding={},
            copy_record={},
        )

    proxy = output / "proxy/invocation-orphan"
    proxy.mkdir(parents=True)
    with pytest.raises(Exception, match="start record|orphan"):
        runner._assert_closed_proxy_inventory(output, "r203-orphan")  # noqa: SLF001


def test_formal_cli_rejects_missing_authority_and_arbitrary_base_url(tmp_path):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--formal",
                "--output-dir",
                str(tmp_path / "formal"),
                "--gateway-root",
                str(tmp_path / "gateway"),
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--formal",
                "--output-dir",
                str(tmp_path / "formal"),
                "--gateway-root",
                str(tmp_path / "gateway"),
                "--allow-model-requests",
                "--base-url",
                "http://127.0.0.1:8199/v1",
            ]
        )
