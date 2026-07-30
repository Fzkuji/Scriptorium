import copy
import json
from itertools import count

import pytest

from scripts import audit_v88_gpt55_scores as auditor
from scripts import audit_v88_gpt55_locomo_subscription as subscription_auditor
from scripts import score_v88_gpt55_benchmarks as scorer


def _usage(profile, response_id, *, response_model=None):
    from src.evaluation.llm_clients import emit_attempt_event

    requested = scorer.JUDGE_PROFILES[profile]["requested_model"]
    actual = response_model or requested
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "parse_attempts": 1,
        "logical_judge_calls": 1,
        "request_attempts": 1,
        "physical_http_attempts": 1,
        "failed_request_attempts": 0,
        "unknown_token_attempts": 0,
        "request_attempt_details": [{
            "parse_attempt": 1,
            "request_attempt": 1,
            "status": "accepted",
            "failure_type": None,
            "error_type": None,
            "error_message": None,
            "requested_model": requested,
            "response_model": actual,
            "response_id": response_id,
            "finish_reason": "stop",
            "refusal": None,
            "choice_count": 1,
            "prompt_tokens": 11,
            "completion_tokens": 2,
        }],
        "requested_models": [requested],
        "response_models": [actual],
        "response_ids": [response_id],
        "finish_reasons": ["stop"],
        "refusals": [None],
        "choice_counts": [1],
    }
    observed_attempt = copy.deepcopy(usage["request_attempt_details"][0])
    observed_attempt.pop("parse_attempt")
    observed_attempt.pop("logical_judge_call", None)
    emit_attempt_event({"event": "physical_http_attempt", "attempt": observed_attempt})
    emit_attempt_event({
        "event": "logical_judge_call",
        "logical_judge_call": 1,
        "status": "parse_accepted",
        "response_id": response_id,
        "score": 1,
    })
    return usage


def _write_proxy_log(path, response_ids=()):
    entries = [{
        "timestamp": "2026-07-14T00:00:00+00:00",
        "status": "success",
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": response_id,
        "attempts": 2,
        "unsupported_parameters": ["max_output_tokens"],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 2,
            "total_tokens": 13,
        },
    } for response_id in response_ids]
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )


def _make_source_audit(
    monkeypatch, tmp_path, source, benchmark, response_ids=()
):
    response_ids = tuple(response_ids)
    proxy_log = tmp_path / "proxy_requests.jsonl"
    _write_proxy_log(proxy_log, response_ids)
    if benchmark in scorer.LOCOMO_BENCHMARKS:
        manifest = {
            "status": "complete",
            "benchmark": "locomo",
            "method": "NativeMem-v8.8+calendar",
            "backbone": "gpt-5.5",
        }
        report = {
            "schema_version": 1,
            "status": "passed",
            "benchmark": "LoCoMo",
            "method": "NativeMem-v8.8+calendar",
            "model": "gpt-5.5",
            "questions": 1986,
            "cat1_4": 1540,
            "combined_sha256": scorer.sha256_file(source),
        }
    else:
        manifest = {
            "status": "complete",
            "completed": 500,
            "benchmark": "LongMemEval-S",
            "method": {"version": "v8.8+calendar"},
            "models": {
                "builder": "gpt-5.5",
                "retriever": "gpt-5.5",
                "answerer": "gpt-5.5",
            },
        }
        report = {
            "schema_version": 1,
            "status": "passed",
            "benchmark": "LongMemEval-S",
            "method": "NativeMem-v8.8+calendar",
            "model": "gpt-5.5",
            "questions": 500,
            "abstention_questions": 30,
            "evaluation_input_sha256": scorer.sha256_file(source),
        }
    manifest_path = tmp_path / "run_manifest.json"
    scorer.atomic_json(manifest_path, manifest)
    report.update({
        "run_dir": str(tmp_path),
        "source_hashes_match": True,
        "empty_answers": 0,
        "manifest": {
            "path": str(manifest_path),
            "sha256": scorer.sha256_file(manifest_path),
        },
        "scoring_input": {
            "path": str(source),
            "sha256": scorer.sha256_file(source),
        },
        "proxy_window": {
            "path": str(proxy_log),
            **scorer.UPSTREAM_PROXY_EVIDENCE,
            "entries": len(response_ids),
            "successes": len(response_ids),
            "errors": 0,
            "actual_models": ["gpt-5.5"],
        },
    })
    recomputed_report = copy.deepcopy(report)
    legacy_hash_key = (
        "combined_sha256"
        if benchmark in scorer.LOCOMO_BENCHMARKS
        else "evaluation_input_sha256"
    )
    recomputed_report[legacy_hash_key] = None
    recomputed_report["scoring_input"] = None
    recomputed_records = copy.deepcopy(scorer.read_json(source))
    monkeypatch.setattr(
        scorer,
        "_rerun_upstream_audit",
        lambda _benchmark, _run_dir: (
            copy.deepcopy(recomputed_records),
            copy.deepcopy(recomputed_report),
        ),
    )
    audit_path = tmp_path / "audit.json"
    scorer.atomic_json(audit_path, report)
    return audit_path, proxy_log


def test_locomo_official_selection_excludes_cat5_and_keeps_ten_builds():
    dataset = scorer.read_json(scorer.LOCOMO_DATA)
    raw = []
    for sample, conversation in enumerate(dataset):
        raw.append({
            "question_id": "_build_stats",
            "sample": sample,
            "build_calls": 1,
            "build_tokens_in": 1,
            "num_memories": 1,
        })
        for index, qa in enumerate(conversation["qa"]):
            raw.append({
                "question_id": f"s{sample}_q{index}",
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(qa["category"]),
                "answer": "test answer",
                "retrieval": {"calls": 1, "steps": 1, "tokens_in": 1},
            })

    selected = scorer.select_official_records(raw, "locomo")
    questions = [r for r in selected if r["question_id"] != "_build_stats"]

    assert len(selected) == 1550
    assert len(questions) == 1540
    assert {r["category"] for r in questions} == {1, 2, 3, 4}
    assert sum(r["question_id"] == "_build_stats" for r in selected) == 10


def test_locomo_selection_accepts_only_authenticated_imported_sample0_build():
    dataset = scorer.read_json(scorer.LOCOMO_DATA)
    sha = "a" * 64
    raw = []
    for sample, conversation in enumerate(dataset):
        build = {
            "question_id": "_build_stats",
            "sample": sample,
            "build_calls": 1,
            "build_tokens_in": 1,
            "num_memories": 1,
        }
        if sample == 0:
            build.update({
                "build_calls": None,
                "build_tokens_in": None,
                "build_accounting": {
                    "status": "unavailable_imported_subscription_memory",
                    "source_provider": "chatgpt_pro_subscription",
                    "source_manifest": {"sha256": sha},
                    "source_memory": {"sha256": sha},
                    "imported_memory": {"sha256": sha},
                    "build_calls": None,
                    "build_tokens_in": None,
                    "build_tokens_out": None,
                    "build_llm_time_s": None,
                },
            })
        raw.append(build)
        for index, qa in enumerate(conversation["qa"]):
            raw.append({
                "question_id": f"s{sample}_q{index}",
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(qa["category"]),
                "answer": "test answer",
                "retrieval": {"calls": 1, "steps": 1, "tokens_in": 1},
            })

    selected = scorer.select_official_records(raw, "locomo")
    assert len(selected) == 1550
    imported = next(
        record for record in selected
        if record.get("question_id") == "_build_stats"
        and record.get("sample") == 0
    )
    assert imported["build_calls"] is None

    tampered = copy.deepcopy(raw)
    tampered[0]["build_accounting"]["source_memory"]["sha256"] = "invalid"
    with pytest.raises(scorer.ScoringError, match="invalid build_calls"):
        scorer.select_official_records(tampered, "locomo")


def test_locomo_upstream_audit_dispatches_subscription_manifest(
    monkeypatch, tmp_path
):
    scorer.atomic_json(tmp_path / "run_manifest.json", {
        "provider": "chatgpt_pro_subscription",
        "formal_flex_result": False,
        "config": {"provider": "chatgpt_pro_subscription"},
    })
    expected = ([{"question_id": "q"}], {"status": "passed"})
    monkeypatch.setattr(subscription_auditor, "audit", lambda path: expected)

    assert scorer._rerun_upstream_audit("locomo", tmp_path) == expected
    assert "upstream_audit_locomo_subscription" in scorer.code_hashes()


def test_locomo_cat5_selection_rebuilds_gold_and_retains_distractor():
    dataset = scorer.read_json(scorer.LOCOMO_DATA)
    raw = []
    for sample, conversation in enumerate(dataset):
        raw.append({
            "question_id": "_build_stats",
            "sample": sample,
            "build_calls": 1,
            "build_tokens_in": 1,
            "num_memories": 1,
        })
        for index, qa in enumerate(conversation["qa"]):
            raw.append({
                "question_id": f"s{sample}_q{index}",
                "question": qa["question"],
                # This is the legacy upstream value and must be replaced.
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(qa["category"]),
                "answer": "Not mentioned in the conversation",
                "retrieval": {"calls": 1, "steps": 1, "tokens_in": 1},
            })

    selected = scorer.select_official_records(raw, "locomo-cat5")
    questions = [r for r in selected if r["question_id"] != "_build_stats"]
    expected = scorer._expected_locomo_cat5()

    assert len(questions) == 446
    assert sum(q["gold"] == scorer.CAT5_CANONICAL_GOLD for q in questions) == 444
    assert sum(q["cat5_abstention"] for q in questions) == 444
    assert sorted(
        q["gold"] for q in questions if not q["cat5_abstention"]
    ) == ["No", "No"]
    assert all(q["distractor"] == expected[q["question_id"]]["distractor"]
               for q in questions)
    assert all(q["category"] == 5 for q in questions)


def test_secondary_scoring_saves_partial_and_resumes_without_rejudging(
    monkeypatch, tmp_path
):
    selected = [{
        "question_id": "_build_stats",
        "build_calls": 1,
        "build_tokens_in": 10,
    }]
    selected.extend({
        "question_id": f"q{index}",
        "question": f"question {index}",
        "gold": f"gold {index}",
        "answer": f"answer {index}",
        "question_type": "multi-session",
        "abstention": False,
        "retrieval": {"calls": 1, "tokens_in": 2, "tokens_out": 1},
    } for index in range(3))
    source = tmp_path / "evaluation_input.json"
    output = tmp_path / "scores.json"
    scorer.atomic_json(source, selected)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval",
        ["resp-0", "resp-1", "resp-2"],
    )
    stale_audit = tmp_path / "custom-score-audit.json"
    scorer.atomic_json(stale_audit, {
        "status": "passed", "evaluation": str(output.resolve())
    })
    scorer.register_score_audit(output, stale_audit)
    calls = []

    def interrupted(*_args, **_kwargs):
        index = len(calls)
        calls.append(index)
        if index == 1:
            raise RuntimeError("interrupted")
        return 1, "yes", _usage("secondary", "resp-0")

    with pytest.raises(RuntimeError, match="interrupted"):
        scorer.score_selected_records(
            selected=selected,
            benchmark="longmemeval",
            profile_name="secondary",
            source_input=source,
            source_audit=source_audit,
            output=output,
            judge_fn=interrupted,
            save_every=1,
        )

    partial = scorer.read_json(output)
    assert partial["meta"]["status"] == "partial"
    assert partial["meta"]["completed_questions"] == 1
    assert partial["meta"]["comparison_status"] == "non-comparable"
    assert partial["meta"]["comparable_to_published_primary"] is False
    assert not stale_audit.exists()
    assert not scorer.score_audit_registry_path(output).exists()

    original_audit = scorer.read_json(source_audit)
    changed_audit = copy.deepcopy(original_audit)
    changed_audit["unexpected_change"] = True
    scorer.atomic_json(source_audit, changed_audit)
    with pytest.raises(scorer.ScoringError, match="recomputed upstream audit"):
        scorer.score_selected_records(
            selected=selected,
            benchmark="longmemeval",
            profile_name="secondary",
            source_input=source,
            source_audit=source_audit,
            output=output,
            judge_fn=lambda *_args, **_kwargs: pytest.fail("judge was called"),
            save_every=1,
        )
    scorer.atomic_json(source_audit, original_audit)

    resumed_calls = count(1)

    def resumed(*_args, **_kwargs):
        index = next(resumed_calls)
        score = index % 2
        return score, "yes" if score else "no", _usage(
            "secondary", f"resp-{index}"
        )

    result = scorer.score_selected_records(
        selected=selected,
        benchmark="longmemeval",
        profile_name="secondary",
        source_input=source,
        source_audit=source_audit,
        output=output,
        judge_fn=resumed,
        save_every=1,
    )

    assert result["meta"]["status"] == "complete"
    assert result["meta"]["completed_questions"] == 3
    assert next(resumed_calls) == 3
    assert [
        record["judge_usage"]["response_ids"][0]
        for record in result["results"] if "judge_usage" in record
    ] == ["resp-0", "resp-1", "resp-2"]
    assert not list(tmp_path.glob(".scores.json.*"))


def _longmemeval_raw_records():
    raw = []
    for index, item in enumerate(scorer.read_json(scorer.LONGMEMEVAL_DATA)):
        raw.append({
            "question_id": "_build_stats",
            "dataset_index": index,
            "source_question_id": item["question_id"],
            "build_time_s": 1,
            "build_calls": 1,
            "build_tokens_in": 10,
            "build_tokens_out": 2,
            "num_memories": 1,
        })
        raw.append({
            "question_id": item["question_id"],
            "dataset_index": index,
            "question": item["question"],
            "gold": item["answer"],
            "question_type": item["question_type"],
            "abstention": str(item["question_id"]).endswith("_abs"),
            "answer": str(item["answer"]),
            "retrieval": {
                "latency_s": 0.1,
                "steps": 1,
                "calls": 1,
                "tokens_in": 5,
                "tokens_out": 1,
            },
        })
    return raw


def test_full_longmemeval_score_audit_and_model_tamper_detection(
    monkeypatch, tmp_path
):
    raw = _longmemeval_raw_records()
    source = tmp_path / "evaluation_input.json"
    evaluation = tmp_path / "primary.json"
    scorer.atomic_json(source, raw)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )
    selected = scorer.select_official_records(raw, "longmemeval")
    response_ids = count()

    def fake_primary(*_args, **_kwargs):
        index = next(response_ids)
        return 1, "yes", _usage("primary", f"or-{index}")

    result = scorer.score_selected_records(
        selected=selected,
        benchmark="longmemeval",
        profile_name="primary",
        source_input=source,
        source_audit=source_audit,
        output=evaluation,
        judge_fn=fake_primary,
        save_every=125,
    )
    report = auditor.audit_scores(
        benchmark="longmemeval",
        source_input=source,
        source_audit=source_audit,
        evaluation=evaluation,
        profile_name="primary",
    )

    assert result["meta"]["completed_questions"] == 500
    assert report["status"] == "passed"
    assert report["questions"] == 500
    assert report["correct"] == 500
    assert report["comparison_status"] == "protocol-comparable"
    assert report["source_provenance"]["upstream_proxy_evidence"] == (
        scorer.UPSTREAM_PROXY_EVIDENCE
    )
    assert report["judge_usage"]["requested_models"] == {
        "openai/gpt-4o-mini": 500
    }
    assert report["judge_usage"]["finish_reasons"] == {"stop": 500}
    assert report["judge_usage"]["refusal_attempts"] == 0
    assert report["judge_usage"]["choice_counts"] == {1: 500}

    tampered = copy.deepcopy(result)
    question = next(r for r in tampered["results"] if "judge_usage" in r)
    question["judge_usage"]["response_models"] = ["gpt-5.5"]
    scorer.atomic_json(evaluation, tampered)
    with pytest.raises(auditor.ScoreAuditError, match="unexpected response model"):
        auditor.audit_scores(
            benchmark="longmemeval",
            source_input=source,
            source_audit=source_audit,
            evaluation=evaluation,
            profile_name="primary",
        )

    tampered = copy.deepcopy(result)
    tampered["meta"]["scope"] = "LongMemEval-S non-abstention only"
    scorer.atomic_json(evaluation, tampered)
    with pytest.raises(auditor.ScoreAuditError, match="meta.scope mismatch"):
        auditor.audit_scores(
            benchmark="longmemeval",
            source_input=source,
            source_audit=source_audit,
            evaluation=evaluation,
            profile_name="primary",
        )


def test_full_locomo_score_audit_uses_cat1_4_only(monkeypatch, tmp_path):
    dataset = scorer.read_json(scorer.LOCOMO_DATA)
    raw = []
    for sample, conversation in enumerate(dataset):
        raw.append({
            "question_id": "_build_stats",
            "sample": sample,
            "build_time_s": 1,
            "build_calls": 1,
            "build_tokens_in": 10,
            "build_tokens_out": 2,
            "num_memories": 1,
        })
        for index, qa in enumerate(conversation["qa"]):
            raw.append({
                "question_id": f"s{sample}_q{index}",
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(qa["category"]),
                "answer": str(qa.get("answer", "not mentioned")),
                "retrieval": {
                    "latency_s": 0.1,
                    "steps": 1,
                    "calls": 1,
                    "tokens_in": 5,
                    "tokens_out": 1,
                },
            })
    source = tmp_path / "questions_all.json"
    evaluation = tmp_path / "secondary.json"
    scorer.atomic_json(source, raw)
    proxy_response_ids = [f"local-{index}" for index in range(1540)]
    source_audit, proxy_log = _make_source_audit(
        monkeypatch, tmp_path, source, "locomo", proxy_response_ids
    )
    selected = scorer.select_official_records(raw, "locomo")
    response_ids = count()

    def fake_secondary(*_args, **_kwargs):
        index = next(response_ids)
        return 1, '{"label":"CORRECT"}', _usage(
            "secondary", f"local-{index}"
        )

    scorer.score_selected_records(
        selected=selected,
        benchmark="locomo",
        profile_name="secondary",
        source_input=source,
        source_audit=source_audit,
        output=evaluation,
        judge_fn=fake_secondary,
        proxy_log=proxy_log,
        save_every=500,
    )
    report = auditor.audit_scores(
        benchmark="locomo",
        source_input=source,
        source_audit=source_audit,
        evaluation=evaluation,
        profile_name="secondary",
        proxy_log=proxy_log,
    )

    assert report["questions"] == 1540
    assert report["correct"] == 1540
    assert report["comparison_status"] == "non-comparable"
    assert report["aggregate"]["n_main"] == 1540
    assert report["aggregate"]["n_adversarial"] == 0
    assert report["judge_usage"]["response_models"] == {"gpt-5.5": 1540}
    with proxy_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-07-14T02:00:00+00:00",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
            "response_id": "unrelated-after-score",
            "attempts": 1,
            "unsupported_parameters": ["max_output_tokens"],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }) + "\n")
    repeated = auditor.audit_scores(
        benchmark="locomo",
        source_input=source,
        source_audit=source_audit,
        evaluation=evaluation,
        profile_name="secondary",
        proxy_log=proxy_log,
    )
    assert repeated["judge_proxy_evidence"] == report["judge_proxy_evidence"]


def test_full_locomo_cat5_is_separate_and_never_passes_distractor_to_judge(
    monkeypatch, tmp_path
):
    dataset = scorer.read_json(scorer.LOCOMO_DATA)
    raw = []
    for sample, conversation in enumerate(dataset):
        raw.append({
            "question_id": "_build_stats",
            "sample": sample,
            "build_time_s": 1,
            "build_calls": 1,
            "build_tokens_in": 10,
            "build_tokens_out": 2,
            "num_memories": 1,
        })
        for index, qa in enumerate(conversation["qa"]):
            raw.append({
                "question_id": f"s{sample}_q{index}",
                "question": qa["question"],
                "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(qa["category"]),
                "answer": str(
                    qa.get("answer") or scorer.CAT5_CANONICAL_GOLD
                ),
                "retrieval": {
                    "latency_s": 0.1,
                    "steps": 1,
                    "calls": 1,
                    "tokens_in": 5,
                    "tokens_out": 1,
                },
            })
    source = tmp_path / "questions_all.json"
    evaluation = tmp_path / "cat5-primary.json"
    scorer.atomic_json(source, raw)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "locomo-cat5"
    )
    selected = scorer.select_official_records(raw, "locomo-cat5")
    questions = [r for r in selected if r["question_id"] != "_build_stats"]
    calls = count()

    def fake_cat5(*args, **kwargs):
        index = next(calls)
        expected = questions[index]
        assert kwargs == {"abstention": expected["cat5_abstention"]}
        assert args == (
            expected["question"], expected["gold"], expected["answer"]
        )
        return 1, '{"label":"CORRECT"}', _usage(
            "primary", f"cat5-{index}"
        )

    result = scorer.score_selected_records(
        selected=selected,
        benchmark="locomo-cat5",
        profile_name="primary",
        source_input=source,
        source_audit=source_audit,
        output=evaluation,
        judge_fn=fake_cat5,
        save_every=200,
    )
    report = auditor.audit_scores(
        benchmark="locomo-cat5",
        source_input=source,
        source_audit=source_audit,
        evaluation=evaluation,
        profile_name="primary",
    )

    assert next(calls) == 446
    assert result["meta"]["question_count"] == 446
    assert result["meta"]["scope"] == scorer.SCOPES["locomo-cat5"]
    assert report["questions"] == 446
    assert report["aggregate"]["n"] == 446
    assert report["aggregate"]["n_abstention"] == 444
    assert report["aggregate"]["n_explicit_answer"] == 2
    assert report["aggregate"]["excluded_from_cat1_4"] is True
    assert report["aggregate"]["semantic_judge_accuracy"] == 1
    assert report["aggregate"]["semantic_abstention_judge_accuracy"] == 1
    assert report["aggregate"]["explicit_answer_judge_accuracy"] == 1
    assert report["aggregate"]["lexical_refusal_rate_diagnostic"] == 1
    artifact_question = next(
        r for r in result["results"] if r.get("question_id") != "_build_stats"
    )
    assert artifact_question["distractor"]
    assert artifact_question["gold"] == scorer.CAT5_CANONICAL_GOLD


def test_locomo_cat5_profile_is_separate_from_published_primary():
    profile = scorer.judge_profile("locomo-cat5", "primary")
    assert profile["comparison_status"] == "separate-category-experiment"
    assert profile["comparable_to_published_primary"] is False
    assert "reported separately" in profile["note"]


def test_judge_usage_requires_requested_and_response_models():
    usage = _usage("secondary", "response-1")
    scorer.validate_judge_result(
        1, "yes", usage, "longmemeval", "secondary", "q"
    )

    wrong_requested = copy.deepcopy(usage)
    wrong_requested["requested_models"] = ["openai/gpt-4o-mini"]
    with pytest.raises(scorer.ScoringError, match="requested model"):
        scorer.validate_judge_result(
            1, "yes", wrong_requested, "longmemeval", "secondary", "q"
        )

    wrong_response = copy.deepcopy(usage)
    wrong_response["response_models"] = ["openai/gpt-4o-mini"]
    with pytest.raises(scorer.ScoringError, match="response model"):
        scorer.validate_judge_result(
            1, "yes", wrong_response, "longmemeval", "secondary", "q"
        )

    wrong_finish = copy.deepcopy(usage)
    wrong_finish["finish_reasons"] = ["length"]
    with pytest.raises(scorer.ScoringError, match="finish reason"):
        scorer.validate_judge_result(
            1, "yes", wrong_finish, "longmemeval", "secondary", "q"
        )

    refusal = copy.deepcopy(usage)
    refusal["refusals"] = ["cannot answer"]
    with pytest.raises(scorer.ScoringError, match="refusal"):
        scorer.validate_judge_result(
            1, "yes", refusal, "longmemeval", "secondary", "q"
        )

    wrong_choice_count = copy.deepcopy(usage)
    wrong_choice_count["choice_counts"] = [2]
    with pytest.raises(scorer.ScoringError, match="choice count"):
        scorer.validate_judge_result(
            1, "yes", wrong_choice_count, "longmemeval", "secondary", "q"
        )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-4o-mini", True),
        ("openai/gpt-4o-mini-2024-07-18", True),
        ("openai/gpt-4o-mini-2024-02-30", False),
        ("evil/gpt-4o-mini-fake", False),
        ("gpt-4o-mini-not-openai", False),
        ("gpt-4o-mini", False),
        ("OpenAI/GPT-4O-MINI", False),
    ],
)
def test_primary_response_model_requires_canonical_openai_name(model, expected):
    assert scorer.response_model_matches("primary", model) is expected


def test_judge_raw_must_agree_with_saved_score():
    with pytest.raises(scorer.ScoringError, match="disagrees with score"):
        scorer.validate_judge_result(
            1, "no", _usage("primary", "response-1"),
            "longmemeval", "primary", "q",
        )
    with pytest.raises(scorer.ScoringError, match="disagrees with score"):
        scorer.validate_judge_result(
            1, '{"label":"WRONG"}', _usage("primary", "response-2"),
            "locomo", "primary", "q",
        )


def test_longmemeval_aggregate_includes_abstention_in_types_and_overall():
    dataset = scorer.read_json(scorer.LONGMEMEVAL_DATA)
    records = [{
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "abstention": str(item["question_id"]).endswith("_abs"),
        "judge_score": int(str(item["question_id"]).endswith("_abs")),
    } for item in dataset]

    aggregate = scorer.recompute_aggregate(records, "longmemeval")

    assert aggregate["overall_acc"] == 30 / 500
    assert sum(block["n"] for block in aggregate["by_type"].values()) == 500
    assert aggregate["by_type"]["multi-session"]["n"] == 133
    assert aggregate["by_type"]["single-session-user"]["n"] == 70
    assert aggregate["abstention_acc"] == 1


def test_longmemeval_primary_is_not_published_primary_comparable():
    profile = scorer.judge_profile("longmemeval", "primary")
    assert profile["comparison_status"] == "protocol-comparable"
    assert profile["comparable_to_published_primary"] is False
    assert "gpt-4o-2024-08-06" in profile["note"]


def test_efficiency_counts_all_judge_request_attempts():
    records = [{
        "question_id": "q",
        "judge_usage": {
            "parse_attempts": 3,
            "request_attempts": 7,
            "prompt_tokens": 30,
            "completion_tokens": 6,
        },
    }]
    aggregate = scorer.recompute_aggregate(records, "longmemeval")
    assert aggregate["efficiency"]["judge_overhead"]["calls"] == 7


def test_selection_rejects_duplicate_or_mismatched_build_identity():
    raw = _longmemeval_raw_records()
    build = next(record for record in raw if record["question_id"] == "_build_stats")
    build["source_question_id"] = "fabricated"
    with pytest.raises(scorer.ScoringError, match="source question id mismatch"):
        scorer.select_official_records(raw, "longmemeval")


def test_source_audit_hash_is_required(monkeypatch, tmp_path):
    source = tmp_path / "evaluation_input.json"
    scorer.atomic_json(source, [])
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )
    report = scorer.read_json(source_audit)
    report["scoring_input"]["sha256"] = "0" * 64
    scorer.atomic_json(source_audit, report)

    with pytest.raises(scorer.ScoringError, match="authenticate"):
        scorer.validate_upstream_audit("longmemeval", source, source_audit)

    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = scorer.read_json(manifest_path)
    manifest["unexpected_change"] = True
    scorer.atomic_json(manifest_path, manifest)
    with pytest.raises(scorer.ScoringError, match="manifest hash"):
        scorer.validate_upstream_audit("longmemeval", source, source_audit)


def test_source_audit_must_match_reexecuted_records_and_report(
    monkeypatch, tmp_path
):
    source = tmp_path / "evaluation_input.json"
    scorer.atomic_json(source, [{"question_id": "original"}])
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )

    stored_report = scorer.read_json(source_audit)
    recomputed_report = copy.deepcopy(stored_report)
    recomputed_report["evaluation_input_sha256"] = None
    recomputed_report["scoring_input"] = None
    monkeypatch.setattr(
        scorer,
        "_rerun_upstream_audit",
        lambda _benchmark, _run_dir: (
            [{"question_id": "different"}], recomputed_report
        ),
    )
    with pytest.raises(scorer.ScoringError, match="records recomputed"):
        scorer.validate_upstream_audit("longmemeval", source, source_audit)

    recomputed_report["questions"] = 499
    monkeypatch.setattr(
        scorer,
        "_rerun_upstream_audit",
        lambda _benchmark, _run_dir: (
            scorer.read_json(source), recomputed_report
        ),
    )
    with pytest.raises(scorer.ScoringError, match="recomputed upstream audit"):
        scorer.validate_upstream_audit("longmemeval", source, source_audit)


def test_code_hashes_include_both_upstream_auditors():
    hashes = scorer.code_hashes()
    assert "upstream_audit_locomo" in hashes
    assert "upstream_audit_longmemeval" in hashes


def test_secondary_proxy_evidence_requires_exact_models_ids_and_usage(tmp_path):
    proxy_log = tmp_path / "proxy.jsonl"
    _write_proxy_log(proxy_log, ["response-1"])
    records = [{
        "question_id": "q",
        "judge_usage": _usage("secondary", "response-1"),
    }]
    evidence = scorer.validate_secondary_proxy_evidence(records, proxy_log)
    assert evidence["matched_response_ids"] == 1

    entries = [json.loads(line) for line in proxy_log.read_text().splitlines()]
    entries[0]["actual_model"] = "gpt-4o-mini"
    proxy_log.write_text(json.dumps(entries[0]) + "\n", encoding="utf-8")
    with pytest.raises(scorer.ScoringError, match="successful GPT-5.5"):
        scorer.validate_secondary_proxy_evidence(records, proxy_log)

    _write_proxy_log(proxy_log, ["different-response"])
    with pytest.raises(scorer.ScoringError, match="0 matching proxy records"):
        scorer.validate_secondary_proxy_evidence(records, proxy_log)


def test_secondary_proxy_evidence_tracks_ignored_output_limit_and_old_logs(
    tmp_path,
):
    proxy_log = tmp_path / "proxy.jsonl"
    records = [{
        "question_id": "q",
        "judge_usage": _usage("secondary", "response-1"),
    }]
    new_entry = {
        "timestamp": "2026-07-14T00:00:00+00:00",
        "status": "success",
        "requested_model": "gpt-5.5",
        "actual_model": "gpt-5.5",
        "response_id": "response-1",
        "attempts": 1,
        "unsupported_parameters": [],
        "ignored_client_parameters": ["max_output_tokens"],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 2,
            "total_tokens": 13,
        },
    }
    proxy_log.write_text(json.dumps(new_entry) + "\n", encoding="utf-8")

    evidence = scorer.validate_secondary_proxy_evidence(records, proxy_log)
    assert evidence["unsupported_parameters"] == []
    assert evidence["ignored_client_parameters"] == ["max_output_tokens"]
    assert evidence["requested_output_limit_enforced"] is False

    del new_entry["ignored_client_parameters"]
    new_entry["unsupported_parameters"] = ["max_output_tokens"]
    proxy_log.write_text(json.dumps(new_entry) + "\n", encoding="utf-8")
    old_evidence = scorer.validate_secondary_proxy_evidence(records, proxy_log)
    assert old_evidence["ignored_client_parameters"] == []
    assert old_evidence["requested_output_limit_enforced"] is False

    new_entry["ignored_client_parameters"] = ["max_output_tokens", 3]
    proxy_log.write_text(json.dumps(new_entry) + "\n", encoding="utf-8")
    with pytest.raises(scorer.ScoringError, match="invalid attempt evidence"):
        scorer.validate_secondary_proxy_evidence(records, proxy_log)


def test_secondary_proxy_evidence_ignores_append_after_frozen_prefix(tmp_path):
    proxy_log = tmp_path / "proxy.jsonl"
    _write_proxy_log(proxy_log, ["response-1"])
    records = [{
        "question_id": "q",
        "judge_usage": _usage("secondary", "response-1"),
    }]
    frozen = scorer.validate_secondary_proxy_evidence(records, proxy_log)
    with proxy_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-07-14T01:00:00+00:00",
            "status": "success",
            "requested_model": "gpt-5.5",
            "actual_model": "gpt-5.5",
            "response_id": "unrelated-later-response",
            "attempts": 1,
            "unsupported_parameters": ["max_output_tokens"],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }) + "\n")

    assert scorer.validate_secondary_proxy_evidence(
        records, proxy_log, frozen=frozen
    ) == frozen


def test_proxy_jsonl_ignores_partial_utf8_tail_but_rejects_bad_complete_line(
    tmp_path,
):
    proxy_log = tmp_path / "proxy.jsonl"
    complete = json.dumps({"response_id": "response-1"}).encode("utf-8") + b"\n"
    proxy_log.write_bytes(complete + b'{"error":"\xe4')
    entries, cutoff, _ = scorer._proxy_log_entries(proxy_log)
    assert entries == [
        {"response_id": "response-1"}
    ]
    assert cutoff == len(complete)

    proxy_log.write_bytes(complete + b'{"broken":\n')
    with pytest.raises(scorer.ScoringError, match="invalid proxy JSONL line 2"):
        scorer._proxy_log_entries(proxy_log)


def test_request_attempt_validation_and_secondary_transport_disclosure(tmp_path):
    proxy_log = tmp_path / "proxy.jsonl"
    _write_proxy_log(proxy_log, ["response-1"])
    usage = _usage("secondary", "response-1")
    transport_error = {
        "parse_attempt": 1,
        "request_attempt": 1,
        "status": "error",
        "failure_type": "transport_error",
        "error_type": "RuntimeError",
        "error_message": "connection reset",
        "requested_model": "gpt-5.5",
        "response_model": None,
        "response_id": None,
        "finish_reason": None,
        "refusal": None,
        "choice_count": None,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    usage["request_attempt_details"].insert(0, transport_error)
    usage["request_attempt_details"][1]["request_attempt"] = 2
    usage["request_attempts"] = 2
    usage["physical_http_attempts"] = 2
    usage["failed_request_attempts"] = 1
    usage["unknown_token_attempts"] = 1

    scorer.validate_judge_result(
        1, "yes", usage, "longmemeval", "secondary", "q"
    )
    evidence = scorer.validate_secondary_proxy_evidence(
        [{"question_id": "q", "judge_usage": usage}], proxy_log
    )
    assert evidence["request_attempts"] == 2
    assert evidence["matched_response_ids"] == 1
    assert evidence["unlinked_transport_errors"] == 1


def test_score_auditor_uses_same_evaluation_lock(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")

    with scorer.exclusive_output_lock(evaluation):
        with pytest.raises(scorer.ScoringError, match="already locked"):
            auditor.audit_scores(
                benchmark="longmemeval",
                source_input=tmp_path / "evaluation_input.json",
                source_audit=tmp_path / "audit.json",
                evaluation=evaluation,
                profile_name="primary",
            )


def test_failed_score_audit_cli_removes_old_passed_report(tmp_path):
    source = tmp_path / "evaluation_input.json"
    source.write_text("[]", encoding="utf-8")
    source_audit = tmp_path / "source.audit.json"
    source_audit.write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    output = tmp_path / "custom.audit.json"
    output.write_text('{"status":"passed"}', encoding="utf-8")

    with pytest.raises(SystemExit):
        auditor.main([
            "--benchmark", "longmemeval",
            "--input", str(source),
            "--source-audit", str(source_audit),
            "--evaluation", str(evaluation),
            "--judge", "primary",
            "--output", str(output),
        ])

    assert not output.exists()


def test_programmatic_restart_invalidates_audit_before_source_failure(tmp_path):
    source = tmp_path / "evaluation_input.json"
    source.write_text("[]", encoding="utf-8")
    source_audit = tmp_path / "source.audit.json"
    source_audit.write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    stale = scorer.default_score_audit_path(evaluation)
    scorer.atomic_json(stale, {
        "status": "passed", "evaluation": str(evaluation.resolve())
    })
    scorer.register_score_audit(evaluation, stale)

    with pytest.raises(scorer.ScoringError):
        scorer.score_selected_records(
            selected=[],
            benchmark="longmemeval",
            profile_name="primary",
            source_input=source,
            source_audit=source_audit,
            output=evaluation,
            judge_fn=lambda *_args, **_kwargs: pytest.fail("judge called"),
            restart=True,
        )

    assert not stale.exists()
    assert not scorer.score_audit_registry_path(evaluation).exists()


def test_cli_output_collision_is_rejected_without_overwrite(tmp_path):
    source = tmp_path / "evaluation_input.json"
    source.write_text("[]", encoding="utf-8")
    before = source.read_bytes()

    with pytest.raises(SystemExit):
        scorer.main([
            "--benchmark", "longmemeval",
            "--input", str(source),
            "--source-audit", str(tmp_path / "source.audit.json"),
            "--output", str(source),
            "--judge", "primary",
        ])
    assert source.read_bytes() == before

    with pytest.raises(SystemExit):
        auditor.main([
            "--benchmark", "longmemeval",
            "--input", str(source),
            "--source-audit", str(tmp_path / "source.audit.json"),
            "--evaluation", str(source),
            "--judge", "primary",
            "--output", str(source),
        ])
    assert source.read_bytes() == before


def test_primary_scoring_cli_requires_explicit_model_request_gate(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        scorer.main([
            "--benchmark", "longmemeval",
            "--input", str(tmp_path / "missing-input.json"),
            "--source-audit", str(tmp_path / "missing-audit.json"),
            "--output", str(tmp_path / "scores.json"),
            "--judge", "primary",
        ])
    assert exc_info.value.code == 2
    assert "--allow-model-requests" in capsys.readouterr().err


def _failed_usage(profile, attempts=3):
    from src.evaluation.llm_clients import emit_attempt_event

    requested = scorer.JUDGE_PROFILES[profile]["requested_model"]
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "parse_attempts": 0,
        "logical_judge_calls": 1,
        "request_attempts": attempts,
        "physical_http_attempts": attempts,
        "failed_request_attempts": attempts,
        "unknown_token_attempts": attempts,
        "request_attempt_details": [{
            "parse_attempt": 1,
            "logical_judge_call": 1,
            "request_attempt": index + 1,
            "status": "error",
            "failure_type": "transport_error",
            "error_type": "RuntimeError",
            "error_message": "connection reset",
            "requested_model": requested,
            "response_model": None,
            "response_id": None,
            "finish_reason": None,
            "refusal": None,
            "choice_count": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        } for index in range(attempts)],
        "requested_models": [],
        "response_models": [],
        "response_ids": [],
        "finish_reasons": [],
        "refusals": [],
        "choice_counts": [],
    }
    for detail in usage["request_attempt_details"]:
        observed = copy.deepcopy(detail)
        observed.pop("parse_attempt")
        observed.pop("logical_judge_call")
        emit_attempt_event({"event": "physical_http_attempt", "attempt": observed})
    emit_attempt_event({
        "event": "logical_judge_call",
        "logical_judge_call": 1,
        "status": "http_failed",
        "usage": copy.deepcopy(usage),
    })
    return usage


def test_failed_attempts_survive_resume_and_are_merged(monkeypatch, tmp_path):
    from src.evaluation.judges import JudgeCallError

    selected = [{
        "question_id": "q0",
        "question": "question",
        "gold": "gold",
        "answer": "answer",
        "question_type": "multi-session",
        "abstention": False,
        "retrieval": {"calls": 1, "tokens_in": 2, "tokens_out": 1},
    }]
    source = tmp_path / "evaluation_input.json"
    output = tmp_path / "scores.json"
    scorer.atomic_json(source, selected)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )

    with pytest.raises(JudgeCallError):
        scorer.score_selected_records(
            selected=selected,
            benchmark="longmemeval",
            profile_name="primary",
            source_input=source,
            source_audit=source_audit,
            output=output,
            judge_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                JudgeCallError("failed", _failed_usage("primary"))
            ),
            save_every=20,
        )

    result = scorer.score_selected_records(
        selected=selected,
        benchmark="longmemeval",
        profile_name="primary",
        source_input=source,
        source_audit=source_audit,
        output=output,
        judge_fn=lambda *_args, **_kwargs: (
            1, "yes", _usage("primary", "success-id")
        ),
        save_every=20,
    )
    usage = result["results"][0]["judge_usage"]
    assert usage["logical_judge_calls"] == 2
    assert usage["physical_http_attempts"] == 4
    assert usage["failed_request_attempts"] == 3
    ledger = scorer.read_attempt_ledger(scorer.attempt_ledger_path(output))
    assert sum(event["event"] == "judge_failure" for event in ledger) == 1


def test_ledger_replays_success_missing_from_snapshot(monkeypatch, tmp_path):
    selected = [{
        "question_id": f"q{index}",
        "question": f"question {index}",
        "gold": "gold",
        "answer": "answer",
        "question_type": "multi-session",
        "abstention": False,
        "retrieval": {"calls": 1, "tokens_in": 2, "tokens_out": 1},
    } for index in range(2)]
    source = tmp_path / "evaluation_input.json"
    output = tmp_path / "scores.json"
    scorer.atomic_json(source, selected)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )
    calls = []

    def interrupted(*_args, **_kwargs):
        index = len(calls)
        calls.append(index)
        if index == 1:
            raise KeyboardInterrupt()
        return 1, "yes", _usage("primary", "first-id")

    original_write = scorer._write_progress

    def lose_partial_snapshot(output_path, records, meta, benchmark, status):
        if status == "partial":
            raise KeyboardInterrupt()
        return original_write(output_path, records, meta, benchmark, status)

    monkeypatch.setattr(scorer, "_write_progress", lose_partial_snapshot)
    with pytest.raises(KeyboardInterrupt):
        scorer.score_selected_records(
            selected=selected,
            benchmark="longmemeval",
            profile_name="primary",
            source_input=source,
            source_audit=source_audit,
            output=output,
            judge_fn=interrupted,
            save_every=20,
        )
    monkeypatch.setattr(scorer, "_write_progress", original_write)

    resumed = []
    result = scorer.score_selected_records(
        selected=selected,
        benchmark="longmemeval",
        profile_name="primary",
        source_input=source,
        source_audit=source_audit,
        output=output,
        judge_fn=lambda *_args, **_kwargs: (
            resumed.append("called") or 0,
            "no",
            _usage("primary", "second-id"),
        ),
        save_every=20,
    )
    assert resumed == ["called"]
    assert [record["judge_raw"] for record in result["results"]] == ["yes", "no"]


@pytest.mark.parametrize("event_type", ["physical_http_attempt", "logical_judge_call"])
def test_resume_fails_closed_on_orphan_observed_ledger_event(
    monkeypatch, tmp_path, event_type
):
    selected = [{
        "question_id": "q0",
        "question": "question",
        "gold": "gold",
        "answer": "answer",
        "question_type": "multi-session",
        "abstention": False,
        "retrieval": {"calls": 1, "tokens_in": 2, "tokens_out": 1},
    }]
    source = tmp_path / "evaluation_input.json"
    output = tmp_path / "scores.json"
    scorer.atomic_json(source, selected)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )
    result = scorer.score_selected_records(
        selected=selected,
        benchmark="longmemeval",
        profile_name="primary",
        source_input=source,
        source_audit=source_audit,
        output=output,
        judge_fn=lambda *_args, **_kwargs: (
            1, "yes", _usage("primary", "complete-response")
        ),
    )
    run_key = result["meta"]["run_key"]
    event = {
        "event": event_type,
        "run_key": run_key,
        "invocation_id": "crashed-invocation",
        "question_id": "q0",
        "benchmark": "longmemeval",
        "judge_profile": scorer.judge_profile("longmemeval", "primary")["id"],
    }
    if event_type == "physical_http_attempt":
        usage = _usage("primary", "orphan-response")
        detail = copy.deepcopy(usage["request_attempt_details"][0])
        detail.pop("parse_attempt")
        detail.pop("logical_judge_call", None)
        event["attempt"] = detail
    else:
        event.update({
            "logical_judge_call": 1,
            "status": "parse_accepted",
            "response_id": "orphan-response",
            "score": 1,
        })
    scorer.append_attempt_event(scorer.attempt_ledger_path(output), event)

    calls = []
    with pytest.raises(scorer.ScoringError, match="unresolved observed call"):
        scorer.score_selected_records(
            selected=selected,
            benchmark="longmemeval",
            profile_name="primary",
            source_input=source,
            source_audit=source_audit,
            output=output,
            judge_fn=lambda *_args, **_kwargs: calls.append("called"),
        )
    assert calls == []


def test_reconcile_rejects_observation_from_different_invocation(
    monkeypatch, tmp_path
):
    selected = [{
        "question_id": "q0",
        "question": "question",
        "gold": "gold",
        "answer": "answer",
        "question_type": "multi-session",
        "abstention": False,
        "retrieval": {"calls": 1, "tokens_in": 2, "tokens_out": 1},
    }]
    source = tmp_path / "evaluation_input.json"
    output = tmp_path / "scores.json"
    scorer.atomic_json(source, selected)
    source_audit, _ = _make_source_audit(
        monkeypatch, tmp_path, source, "longmemeval"
    )
    result = scorer.score_selected_records(
        selected=selected,
        benchmark="longmemeval",
        profile_name="primary",
        source_input=source,
        source_audit=source_audit,
        output=output,
        judge_fn=lambda *_args, **_kwargs: (
            1, "yes", _usage("primary", "complete-response")
        ),
    )
    events = copy.deepcopy(
        scorer.read_attempt_ledger(scorer.attempt_ledger_path(output))
    )
    observation = next(
        event
        for event in events
        if event["event"] in {"physical_http_attempt", "logical_judge_call"}
    )
    observation["invocation_id"] = "different-invocation"

    with pytest.raises(scorer.ScoringError, match="crosses call identities"):
        scorer.reconcile_attempt_ledger(
            events,
            result["meta"]["run_key"],
            benchmark="longmemeval",
            profile_name="primary",
        )


def test_distinct_paths_reject_hardlink_case_and_unicode_aliases(tmp_path):
    source = tmp_path / "Input.json"
    source.write_text("[]", encoding="utf-8")
    hardlink = tmp_path / "hard.json"
    hardlink.hardlink_to(source)
    with pytest.raises(scorer.ScoringError, match="collision"):
        scorer.ensure_distinct_paths({"source": source, "output": hardlink})
    with pytest.raises(scorer.ScoringError, match="collision"):
        scorer.ensure_distinct_paths({
            "source": source, "output": tmp_path / "input.JSON"
        })
    composed = tmp_path / "café.json"
    decomposed = tmp_path / "café.json"
    with pytest.raises(scorer.ScoringError, match="collision"):
        scorer.ensure_distinct_paths({"source": composed, "output": decomposed})
