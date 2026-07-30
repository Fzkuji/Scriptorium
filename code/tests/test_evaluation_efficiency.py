from src.evaluation.evaluate import aggregate_efficiency


def test_efficiency_aggregates_every_independent_build_unit():
    records = [
        {
            "question_id": "_build_stats",
            "build_time_s": 10,
            "build_calls": 2,
            "build_tokens_in": 100,
            "build_tokens_out": 20,
            "num_memories": 3,
        },
        {
            "question_id": "_build_stats",
            "build_time_s": 15,
            "build_calls": 4,
            "build_tokens_in": 200,
            "build_tokens_out": 30,
            "num_memories": 5,
        },
        {
            "question_id": "q0",
            "retrieval": {
                "calls": 1,
                "tokens_in": 40,
                "tokens_out": 10,
                "latency_s": 2.5,
            },
        },
    ]

    result = aggregate_efficiency(records)

    assert result["build"] == {
        "units": 2,
        "time_s": 25,
        "calls": 6,
        "tokens_in": 300,
        "tokens_out": 50,
        "num_memories": 8,
    }
    assert result["retrieval"]["calls"] == 1
    assert result["system_total_tokens_k"] == 0.4
