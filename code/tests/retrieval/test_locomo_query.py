from src import build as build_module


def test_scriptorium_query_source_index_accepts_the_same_opaque_ids_as_topics():
    turn = {"speaker": "Melanie", "text": "painted a sunrise"}

    index = build_module.add_benchmark_source_ids({"D1:14": turn})

    ref = build_module.benchmark_source_id("D1:14")
    assert index[ref] is turn
