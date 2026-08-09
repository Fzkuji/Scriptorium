"""The process-wide index cache: reuse within a revision, rebuild on a
write, no growth from revisions a write has already superseded.

Exercises the cache directly with dummy factories rather than through
`search.nearest` or `read.read`, so it never needs a real or stubbed
embedding model.
"""

from memory.retrieval import index_cache
from memory.retrieval.index_cache import IndexCache


def _write(memory_dir, text: str) -> None:
    topics = memory_dir / "topics"
    topics.mkdir(exist_ok=True)
    (topics / "a.md").write_text(text, encoding="utf-8")


def test_the_same_revision_reuses_one_index(tmp_path):
    _write(tmp_path, "one\n")
    built: list[object] = []

    def factory():
        built.append(object())
        return built[-1]

    first = IndexCache(tmp_path).get_retrieval_index(("kind",), factory)
    second = IndexCache(tmp_path).get_retrieval_index(("kind",), factory)

    assert first is second
    assert len(built) == 1


def test_a_write_to_the_workspace_produces_a_new_index(tmp_path):
    _write(tmp_path, "one\n")

    before = IndexCache(tmp_path).get_retrieval_index(("kind",), object)
    _write(tmp_path, "two\n")
    after = IndexCache(tmp_path).get_retrieval_index(("kind",), object)

    assert before is not after


def test_the_superseded_entry_is_evicted_rather_than_accumulating(tmp_path):
    _write(tmp_path, "one\n")
    key = ("kind", str(tmp_path))

    def entries_for_key() -> int:
        return sum(1 for stamped in index_cache._indexes if stamped[1:] == key)

    IndexCache(tmp_path).get_retrieval_index(key, object)
    assert entries_for_key() == 1

    _write(tmp_path, "two\n")
    IndexCache(tmp_path).get_retrieval_index(key, object)

    assert entries_for_key() == 1
