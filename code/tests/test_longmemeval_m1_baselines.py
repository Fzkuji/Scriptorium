import copy
import importlib.metadata
import os

import pytest

from scripts.longmemeval_m1 import audit_longmemeval_m1_baselines as auditor
from scripts.longmemeval_m1 import longmemeval_m1_contract as contract
from scripts.longmemeval_m1 import run_longmemeval_m1_baselines as runner


def _has_formal_runtime():
    expected = {
        "tiktoken": "0.12.0",
        "rank-bm25": "0.2.2",
        "mem0ai": "2.0.10",
        "qdrant-client": "1.18.0",
        "graphiti-core": "0.29.2",
        "kuzu": "0.11.3",
        "sentence-transformers": "5.6.0",
    }
    try:
        return all(importlib.metadata.version(name) == version
                   for name, version in expected.items())
    except importlib.metadata.PackageNotFoundError:
        return False


FORMAL_RUNTIME = _has_formal_runtime()
requires_formal_runtime = pytest.mark.skipif(
    not FORMAL_RUNTIME,
    reason="formal M1 tests require the pinned /opt/miniconda3 environment",
)


def test_canonical_dataset_inventory_and_independent_histories():
    data = contract.validate_dataset(contract.DEFAULT_DATASET)

    assert len(data) == 500
    assert sum(item["question_id"].endswith("_abs") for item in data) == 30
    assert len({item["question_id"] for item in data}) == 500
    assert len({contract.history_content_hash(item) for item in data}) == 500
    assert len({contract.history_owner_hash(item) for item in data}) == 500
    assert contract.sha256_file(contract.DEFAULT_DATASET) == (
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    )


@requires_formal_runtime
def test_formal_tokenizer_records_literal_special_token_policy():
    tokenizer = contract.FormalTokenizer.resolve()

    assert tokenizer.identity == {
        "implementation": "tiktoken",
        "implementation_version": "0.12.0",
        "encoding_name": "o200k_base",
        "requested_model": "gpt-5.5",
        "resolution": "explicit_named_encoding",
        "disallowed_special": [],
        "special_token_policy": "encode_literal_special_strings_as_ordinary_text",
        "provider_exact": False,
    }
    assert tokenizer.count("literal <|endoftext|> text") > 0


@requires_formal_runtime
def test_full_context_max_item_has_no_truncation_and_fits_declared_prompt_limit():
    data = contract.validate_dataset(contract.DEFAULT_DATASET)
    tokenizer = contract.FormalTokenizer.resolve()
    item = data[165]

    answer_input, private = contract.prepare_full_context_item(item, 165, tokenizer)

    assert item["question_id"] == "a11281a2"
    assert answer_input["context"]["visible_tokens"] == 106_384
    assert answer_input["answer_protocol_interface"]["rendered_prompt_tokens"] == 106_874
    assert answer_input["answer_protocol_interface"]["rendered_prompt_tokens"] <= 123_904
    assert answer_input["context"]["source_session_ids"] == item[
        "haystack_session_ids"
    ]
    assert len(answer_input["context"]["events"]) == len(item["haystack_sessions"])
    assert all(
        event["decision"] == "delivered_full"
        and event["raw_text_sha256"] == event["delivered_text_sha256"]
        and event["raw_tokens"] == event["delivered_tokens"]
        for event in answer_input["context"]["events"]
    )
    assert answer_input["context"]["budget"] == {
        "policy": "full_context_unbounded_accounted",
        "configured_visible_budget_tokens": None,
        "truncation_allowed": False,
        "truncated_events": 0,
        "cumulative_visible_tokens": 106_384,
    }
    assert private["source_mapping"]["granularity"] == "session"
    assert private["source_mapping"]["turn_recall_claimed"] is False
    contract.validate_answer_input_allowlist(answer_input)


@requires_formal_runtime
def test_bm25_gate_hashes_delivered_text_and_session_sources():
    data = contract.validate_dataset(contract.DEFAULT_DATASET)
    tokenizer = contract.FormalTokenizer.resolve()
    item = data[3]  # Contains a duplicated raw session ID occurrence.

    answer_input, private = contract.prepare_bm25_item(item, 3, tokenizer)
    context = answer_input["context"]

    assert 0 < context["visible_tokens"] <= 20_000
    assert context["visible_tokens"] == 20_000
    assert context["retokenized_context_tokens"] == 19_991
    assert context["budget"]["configured_visible_budget_tokens"] == 20_000
    assert context["budget"]["truncated_events"] == 1
    assert context["events"][-1]["decision"] == "delivered_truncated"
    assert context["source_session_ids"] == [
        event["source_session_id"] for event in context["events"]
    ]
    for event in context["events"]:
        assert event["source_session_id"] in item["haystack_session_ids"]
        assert event["source_session_occurrence_key"] == (
            f"{event['dataset_session_index']:04d}:{event['source_session_id']}"
        )
        assert event["source_mapping_granularity"] == "session"
        assert event["turn_recall_claimed"] is False
        assert len(event["raw_text_sha256"]) == 64
        assert len(event["delivered_text_sha256"]) == 64
        assert f"session_id={event['source_session_id']}" in context["text"]
    assert private["source_mapping"]["delivered_source_session_ids"] == list(
        dict.fromkeys(context["source_session_ids"])
    )
    contract.validate_answer_input_allowlist(answer_input)


@requires_formal_runtime
def test_preregistration_freezes_four_rows_before_answers(tmp_path):
    preregistration = tmp_path / "formal_matrix_preregistration.json"
    payload = runner.create_preregistration(
        dataset_path=contract.DEFAULT_DATASET,
        output_path=preregistration,
        model_context_limit_tokens=128_000,
        answer_reservation_tokens=4_096,
    )
    audited, data = auditor.audit_preregistration(
        preregistration, contract.DEFAULT_DATASET
    )

    assert audited == payload
    assert len(data) == 500
    assert payload["formal_methods"] == [
        "full_context", "bm25", "mem0", "graphiti"
    ]
    assert [row["configuration"]["public_name"] for row in payload["rows"]] == [
        "full_context", "BM25", "Mem0 OSS", "Graphiti OSS"
    ]
    assert payload["answer_protocol"]["status"] == "not_started"
    assert payload["model_calls"] == payload["network_calls"] == 0
    with pytest.raises(FileExistsError):
        runner.create_preregistration(
            dataset_path=contract.DEFAULT_DATASET,
            output_path=preregistration,
            model_context_limit_tokens=128_000,
            answer_reservation_tokens=4_096,
        )


@requires_formal_runtime
@pytest.mark.parametrize("method", ["mem0", "graphiti"])
def test_backend_formal_and_single_item_smoke_plans_are_no_execution(
    tmp_path, method
):
    preregistration = tmp_path / "prereg.json"
    runner.create_preregistration(
        dataset_path=contract.DEFAULT_DATASET,
        output_path=preregistration,
        model_context_limit_tokens=128_000,
        answer_reservation_tokens=4_096,
    )
    output_root = tmp_path / "future-runs"
    formal_path = tmp_path / f"{method}-formal.json"
    smoke_path = tmp_path / f"{method}-smoke.json"
    formal = runner.create_backend_plan(
        method=method,
        scope="formal",
        item_index=None,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=preregistration,
        output_root=output_root,
        output_path=formal_path,
    )
    smoke = runner.create_backend_plan(
        method=method,
        scope="smoke",
        item_index=7,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=preregistration,
        output_root=output_root,
        output_path=smoke_path,
    )

    assert formal["item_count"] == 500
    assert smoke["item_count"] == 1
    assert smoke["items"][0]["dataset_index"] == 7
    assert len({row["workspace"]["workspace"] for row in formal["items"]}) == 500
    assert formal["status"] == smoke["status"] == "planned_not_executed"
    assert formal["model_calls"] == smoke["model_calls"] == 0
    assert not output_root.exists()
    assert auditor.audit_backend_plan(
        plan_path=formal_path,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=preregistration,
    )["status"] == "passed"
    assert auditor.audit_backend_plan(
        plan_path=smoke_path,
        dataset_path=contract.DEFAULT_DATASET,
        preregistration_path=preregistration,
    )["status"] == "passed"


@requires_formal_runtime
def test_one_item_checkpoint_ledger_and_independent_audit(tmp_path):
    data = contract.validate_dataset(contract.DEFAULT_DATASET)
    tokenizer = contract.FormalTokenizer.resolve()
    run_dir = tmp_path / "full_context"
    (run_dir / "items").mkdir(parents=True)
    preregistration_sha256 = "a" * 64
    configuration_sha256 = contract.canonical_hash(
        contract.method_configuration("full_context")
    )

    summary = runner._process_item(
        method="full_context",
        item=data[0],
        item_index=0,
        run_dir=run_dir,
        preregistration_sha256=preregistration_sha256,
        configuration_sha256=configuration_sha256,
        tokenizer=tokenizer,
    )
    item_dir = run_dir / "items" / summary["item_dir"]
    audited = auditor._audit_one_item(
        method="full_context",
        item=data[0],
        item_index=0,
        item_dir=item_dir,
        preregistration_sha256=preregistration_sha256,
        configuration_sha256=configuration_sha256,
        tokenizer=tokenizer,
    )

    assert audited == summary
    assert len(contract.read_jsonl(item_dir / "attempts.jsonl")) == 2
    assert contract.read_json(item_dir / "checkpoint.json")["model_calls"] == 0


def test_orphan_attempt_fails_closed_before_item_preparation(tmp_path):
    item = contract.read_json(contract.DEFAULT_DATASET)[0]
    run_dir = tmp_path / "full_context"
    item_dir = run_dir / "items" / contract.item_directory_name(0, item["question_id"])
    item_dir.mkdir(parents=True)
    contract.append_jsonl_fsync(
        item_dir / "attempts.jsonl",
        {
            "event": "attempt_started",
            "attempt_id": "orphan-attempt",
            "method": "full_context",
            "dataset_index": 0,
            "question_id": item["question_id"],
        },
    )

    with pytest.raises(contract.ContractError, match="unresolved attempt"):
        runner._process_item(
            method="full_context",
            item=item,
            item_index=0,
            run_dir=run_dir,
            preregistration_sha256="a" * 64,
            configuration_sha256="b" * 64,
            tokenizer=object(),
        )
    assert not (item_dir / "answer_input.json").exists()


def test_path_alias_symlink_and_hardlink_are_rejected(tmp_path):
    first = tmp_path / "first.json"
    first.write_text("{}\n", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    os.link(first, hardlink)
    with pytest.raises(contract.ContractError, match="inode collision"):
        contract.ensure_distinct_paths({"first": first, "hardlink": hardlink})

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(contract.ContractError, match="symlink component"):
        contract.reject_symlink_components(alias / "output.json")


def test_tree_inventory_rejects_hardlink_and_symlink(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    first = root / "first"
    first.write_text("content", encoding="utf-8")
    second = root / "second"
    os.link(first, second)
    with pytest.raises(contract.ContractError, match="hardlink alias"):
        contract.regular_tree_inventory(root)
    second.unlink()
    second.symlink_to(first)
    with pytest.raises(contract.ContractError, match="symlink"):
        contract.regular_tree_inventory(root)


def test_run_auditor_allows_its_report_but_rejects_other_root_files(tmp_path):
    (tmp_path / "items").mkdir()
    (tmp_path / ".run.lock").write_text("pid=1\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "audit.json").write_text("{}\n", encoding="utf-8")
    auditor._root_allowlist(tmp_path, None)

    (tmp_path / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(auditor.AuditError, match="unexpected nodes"):
        auditor._root_allowlist(tmp_path, None)


@requires_formal_runtime
def test_preregistration_tamper_breaks_content_hash(tmp_path):
    path = tmp_path / "prereg.json"
    payload = runner.create_preregistration(
        dataset_path=contract.DEFAULT_DATASET,
        output_path=path,
        model_context_limit_tokens=128_000,
        answer_reservation_tokens=4_096,
    )
    tampered = copy.deepcopy(payload)
    tampered["context_policy"]["hard_visible_total_tokens"] = 19_999
    contract.atomic_json_replace(path, tampered)

    with pytest.raises(auditor.AuditError, match="content hash"):
        auditor.audit_preregistration(path, contract.DEFAULT_DATASET)


def test_backend_scope_requires_exact_formal_or_one_smoke_item():
    assert contract.backend_plan_indices("formal", None) == list(range(500))
    assert contract.backend_plan_indices("smoke", 499) == [499]
    with pytest.raises(contract.ContractError, match="all 500"):
        contract.backend_plan_indices("formal", 0)
    with pytest.raises(contract.ContractError, match="one valid"):
        contract.backend_plan_indices("smoke", None)
    with pytest.raises(contract.ContractError, match="one valid"):
        contract.backend_plan_indices("smoke", 500)


def test_mem0_and_graphiti_configs_pin_local_storage_and_models():
    mem0 = contract.method_configuration("mem0")
    graphiti = contract.method_configuration("graphiti")

    assert mem0["vector_store"] == {
        "provider": "qdrant",
        "mode": "local_embedded_path_per_item",
        "collection_name_template": "lme_m1_{dataset_index}_{question_id}",
    }
    assert mem0["builder_llm"]["model"] == "gpt-5.5"
    assert graphiti["public_name"] == "Graphiti OSS"
    assert graphiti["graph_store"] == {
        "provider": "kuzu",
        "mode": "local_embedded_path_per_item",
    }
    assert graphiti["claim_boundary"] == (
        "Graphiti OSS core, not the commercial Zep platform"
    )
    assert graphiti["builder_llm"]["model"] == "gpt-5.5"
    assert graphiti["builder_llm"]["small_model"] == "gpt-5.5"
