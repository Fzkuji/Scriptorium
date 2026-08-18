from src.retrieval.evidence_packet import build_evidence_packet, evidence_state


def test_evidence_state_is_conservative_about_plan_and_completion():
    assert evidence_state("The user planned to bake a cake.") == "planned"
    assert evidence_state("The user made an apple pie.") == "completed"
    assert evidence_state("Apple pie is one possible recipe.") == "unspecified"


def test_packet_preserves_candidate_order_and_text():
    rows = [
        {
            "content": "The user made pie.", "date": "2025-01-01",
            "refs": ["s1"], "path": "topics/cooking.md",
        },
        {
            "content": "The user planned cake.", "date": "2025-01-02",
            "refs": ["s2"], "path": "sources/thread.md",
        },
    ]
    packet, metrics = build_evidence_packet(
        rows, {"answer_shape": "count", "coverage_axis": "distinct_events"}
    )

    assert packet.index("The user made pie.") < packet.index("The user planned cake.")
    assert 'id="E01" rank="1"' in packet
    assert 'id="E02" rank="2"' in packet
    assert metrics["candidate_count"] == 2
    assert metrics["candidate_ranks"] == [1, 2]
    assert metrics["unique_source_refs"] == 2
