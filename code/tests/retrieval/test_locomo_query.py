from src import conversation


def test_scriptorium_query_source_index_accepts_the_same_opaque_ids_as_topics():
    turn = {"speaker": "Melanie", "text": "painted a sunrise"}

    index = conversation.add_benchmark_source_ids({"D1:14": turn})

    ref = conversation.benchmark_source_id("D1:14")
    assert index[ref] is turn
