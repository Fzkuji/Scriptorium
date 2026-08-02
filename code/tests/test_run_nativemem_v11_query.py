from src.nativemem_versions.v11 import adapter as v11_adapter


def test_v11_query_source_index_accepts_the_same_opaque_ids_as_topics():
    turn = {"speaker": "Melanie", "text": "painted a sunrise"}

    index = v11_adapter.add_benchmark_source_ids({"D1:14": turn})

    ref = v11_adapter.benchmark_source_id("D1:14")
    assert index[ref] is turn
