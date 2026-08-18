from types import SimpleNamespace

from src.retrieval.pipeline import (
    PipelineConfig,
    _distinct_evidence,
    _expand_source_block,
    _relation_neighbors,
    _timeline_queries,
    parse_timeline_events,
    m3_hybrid_mmr,
    question_contract,
    reciprocal_rank_fusion,
    retrieve_pipeline,
)


def row(event_id, path, line, content, refs=()):
    return {
        "event_id": event_id,
        "event": event_id,
        "path": path,
        "line": line,
        "date": "2025-01-01",
        "dates": ["2025-01-01"],
        "content": content,
        "refs": list(refs),
    }


class Index:
    def __init__(self, rows, events=()):
        self.rows = rows
        self.events = list(events)

    def search(self, _query, *, top_k):
        return self.rows[:top_k]


def test_rrf_rewards_candidates_found_by_both_retrievers():
    shared = row("shared", "topics/a.md", 1, "shared")
    fused = reciprocal_rank_fusion([
        [row("lexical", "topics/a.md", 2, "lexical"), shared],
        [shared, row("semantic", "topics/b.md", 1, "semantic")],
    ])

    assert fused[0]["event_id"] == "shared"
    assert fused[0]["ranks"] == {"0": 2, "1": 1}


def test_pipeline_is_bounded_balanced_and_adds_source_backlink():
    lexical = [
        row("t1", "topics/a.md", 1, "tea", ["D1:1"]),
        row("t2", "topics/a.md", 2, "coffee", ["D1:2"]),
        row("t3", "topics/a.md", 3, "water", ["D1:3"]),
        row("t4", "topics/b.md", 1, "juice", ["D2:1"]),
    ]
    semantic = [lexical[3], lexical[0]]
    source = SimpleNamespace(**row("D1:1", "sources/D1.md", 5, "original tea", ["D1:1"]))
    config = PipelineConfig(
        bm25_top_k=4,
        embedding_top_k=2,
        output_top_k=4,
        max_per_path=2,
        source_backlinks=1,
    )

    result = retrieve_pipeline(
        "drink",
        bm25=Index(lexical, events=[source]),
        embedding=Index(semantic),
        config=config,
    )

    assert len(result) == 4
    assert sum(item["path"] == "topics/a.md" for item in result) <= 2
    backlink = next(item for item in result if item["path"] == "sources/D1.md")
    assert backlink["backlink_for"] == "t1"
    assert [item["pipeline_rank"] for item in result] == [1, 2, 3, 4]


def test_pipeline_config_has_stable_fingerprint():
    assert PipelineConfig().fingerprint == PipelineConfig().fingerprint
    assert PipelineConfig(output_top_k=8).fingerprint != PipelineConfig().fingerprint
    assert PipelineConfig().answer_max_turns == 3
    assert PipelineConfig.for_version("p0-v3").multi_query is True
    assert PipelineConfig.for_version("p0-v3").output_top_k == 16
    assert PipelineConfig.for_version("p0b-r1").timeline_quota == 4
    assert PipelineConfig.for_version("m3").mmr_lambda == 0.7


def test_m3_hybrid_mmr_prefers_relevance_then_diversity():
    lexical = [
        {**row("a", "topics/a.md", 1, "alpha"), "similarity": 0.9},
        {**row("b", "topics/a.md", 2, "alpha duplicate"), "similarity": 0.8},
        {**row("c", "topics/c.md", 1, "different"), "similarity": 0.7},
    ]
    semantic = list(lexical)
    vectors = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.01],
        "c": [0.0, 1.0],
    }

    selected = m3_hybrid_mmr(lexical, semantic, vectors, limit=2)

    assert [item["event_id"] for item in selected] == ["a", "c"]
    assert selected[0]["retrieval_variant"] == "m3"


def test_question_contract_routes_preference_and_count_questions():
    preference = question_contract("Any tips for rearranging my bedroom furniture?")
    count = question_contract("How many museums did I visit?")

    assert preference["answer_shape"] == "preference"
    assert "preferences" in preference["queries"][-1]
    assert count["answer_shape"] == "count"
    assert count["state_terms"] == ["completed", "booked", "planned", "cancelled"]
    assert count["coverage_axis"] == "distinct_events"


def test_timeline_queries_remove_count_and_time_meta_terms():
    question = "How many times did I bake something in the past two weeks?"
    contract = question_contract(question, "2023-05-30")

    lanes = _timeline_queries(question, contract)

    assert "many" not in lanes[1]
    assert "weeks" not in lanes[1]
    assert "completed made did" in lanes[2]


def test_distinct_evidence_deduplicates_shared_source_refs():
    rows = [
        row("a", "timeline/a.md", 1, "first", ["source-1"]),
        row("b", "topics/b.md", 1, "duplicate", ["source-1"]),
        row("c", "timeline/c.md", 1, "second", ["source-2"]),
    ]

    selected = _distinct_evidence(rows, limit=3)

    assert [item["event_id"] for item in selected] == ["a", "c"]


def test_question_contract_normalizes_relative_time_from_question_date():
    point = question_contract(
        "What milestone did I mention four weeks ago?", "2023-03-29"
    )
    window = question_contract(
        "How much did I spend in the last four months?", "2023-04-01"
    )

    assert point["temporal"]["date_from"] == "2023-02-26"
    assert point["temporal"]["date_to"] == "2023-03-04"
    assert window["temporal"]["date_from"] == "2022-12-02"
    assert window["temporal"]["date_to"] == "2023-04-01"


def test_question_contract_normalizes_plural_and_aggregate_shape():
    contract = question_contract("How much total money did I spend at workshops?")

    assert contract["answer_shape"] == "count"
    assert any("workshop" in query.split() for query in contract["queries"])


def test_source_hit_expands_to_complete_message_block(tmp_path):
    source = tmp_path / "sources" / "thread.md"
    source.parent.mkdir()
    source.write_text(
        "<!-- source-id:msg1 -->\n"
        "[2025-01-01] assistant: Resources:\n\n"
        "1. First\n\n"
        "2. Exact answer\n\n"
        "<a id=\"next\"></a>\n"
        "<!-- source-id:msg2 -->\n"
        "[2025-01-02] user: next\n",
        encoding="utf-8",
    )
    expanded = _expand_source_block(tmp_path, row(
        "msg1", "sources/thread.md", 2, "Resources:", ["msg1"]
    ))

    assert "2. Exact answer" in expanded["content"]
    assert "user: next" not in expanded["content"]
    assert expanded["source_block_lines"] == 6


def test_timeline_adapter_preserves_date_topic_identity_and_source_ref(tmp_path):
    path = tmp_path / "timeline" / "2023" / "05" / "20.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# 2023-05-20\n\n"
        "The user visited Dr. Lee. "
        "[Topic](../../../topics/health.md#^abc123) "
        "[locomo/thread/msg1](../../../sources/locomo/thread.md#source-x)\n",
        encoding="utf-8",
    )

    events = parse_timeline_events(tmp_path)

    assert len(events) == 1
    assert events[0].event_id == "timeline:abc123"
    assert events[0].date == "2023-05-20"
    assert events[0].refs == ["locomo/thread/msg1"]


def test_relation_view_expands_topic_event_neighbors(tmp_path):
    (tmp_path / "relations.json").write_text(
        '{"outbound":{"seed":["neighbor"]},"backlinks":{}}',
        encoding="utf-8",
    )
    events = [
        SimpleNamespace(**row("seed:1", "topics/people/a.md", 1, "Alice")),
        SimpleNamespace(**row("neighbor:1", "topics/people/b.md", 1, "Bob")),
    ]

    rows = _relation_neighbors(
        tmp_path,
        [row("seed:1", "topics/people/a.md", 1, "Alice")],
        events,
    )

    assert [item["event_id"] for item in rows] == ["neighbor:1"]
    assert rows[0]["view"] == "relations"
