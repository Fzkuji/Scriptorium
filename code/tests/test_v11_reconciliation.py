import pytest

from src.nativemem_versions.v11.reconciliation import (
    ReconciliationError,
    ReconciliationResult,
    apply_reconciliation,
    classify_topic_diff,
)
from src.nativemem_versions.v11.topic_markdown import MemoryUnit, parse_topic_tree
from src.v11_memory import MemoryWorkspace
from src.nativemem_versions.v11.memory import _make_reconciler
from types import SimpleNamespace


def unit(memory_id="mem_a", content="old fact", path="a.md", heading="A"):
    return MemoryUnit(
        memory_id=memory_id,
        content=content,
        when="2026-01-01",
        source_refs=("src_1",),
        source_links=("../sources/D1.md#d1-1",),
        topic_path=path,
        headings=(heading,),
        created_order=0,
    )


def test_classify_topic_diff_separates_structural_and_semantic_edits():
    structural = classify_topic_diff([unit()], [unit(path="moved.md", heading="Moved")])
    semantic = classify_topic_diff([unit()], [unit(content="rewritten fact")])

    assert structural.kind == "structural"
    assert structural.reconciler_required is False
    assert semantic.kind == "semantic"
    assert semantic.reconciler_required is True
    assert semantic.changed_ids == ("mem_a",)


def test_apply_reconciliation_accepts_exact_quotes_and_candidate_sources():
    result = ReconciliationResult(
        matches={"mem_a": "rewritten fact"},
        creates=(("new fact", "2026-02-01", ("src_2",)),),
        deleted_ids=(),
    )

    applied = apply_reconciliation(
        "rewritten fact. new fact.",
        [unit()],
        result,
        candidate_sources={"src_1", "src_2"},
    )

    assert applied.matches["mem_a"] == "rewritten fact"
    assert applied.creates[0][2] == ("src_2",)


@pytest.mark.parametrize("result, message", [
    (ReconciliationResult({"mem_a": "invented"}, (), ()), "exact quote"),
    (ReconciliationResult({"mem_a": "rewritten fact"}, (("new fact", "2026-02-01", ("src_fake",)),), ()), "candidate source"),
    (ReconciliationResult({}, (), ()), "missing old memory_id"),
])
def test_apply_reconciliation_rejects_invalid_or_implicit_changes(result, message):
    with pytest.raises(ReconciliationError, match=message):
        apply_reconciliation(
            "rewritten fact. new fact.",
            [unit()],
            result,
            candidate_sources={"src_1", "src_2"},
        )


def test_apply_reconciliation_allows_only_explicit_correction_deletion():
    result = ReconciliationResult({}, (), ("mem_a",))

    with pytest.raises(ReconciliationError, match="explicit correction"):
        apply_reconciliation(
            "corrected text",
            [unit()],
            result,
            candidate_sources={"src_1"},
        )

    applied = apply_reconciliation(
        "corrected text",
        [unit()],
        result,
        candidate_sources={"src_1"},
        allow_correction=True,
    )
    assert applied.deleted_ids == ("mem_a",)


def test_workspace_calls_reconciler_for_content_change_but_not_move(tmp_path):
    calls = []

    def reconcile(edited_text, old_units, candidate_sources):
        calls.append((edited_text, old_units, candidate_sources))
        return ReconciliationResult(
            {old_units[0].memory_id: "The user wakes at 6:30."}, (), ()
        )

    workspace = MemoryWorkspace(tmp_path, reconciler=reconcile)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "I wake at 6:30")],
        "refs": ["D1:1"],
    }])
    workspace.save_memory([{
        "when": "2026-01-01",
        "content": "The user wakes at 7:00.",
        "refs": ["D1:1"],
        "topic_path": "routine.md",
        "headings": ["Routine"],
    }])

    workspace.shell("sed -i '' 's/wakes at 7:00/wakes at 6:30/' topics/routine.md")
    assert len(calls) == 1
    workspace.shell("mkdir -p topics/moved && mv topics/routine.md topics/moved/routine.md")
    assert len(calls) == 1


def test_workspace_reconciles_new_free_text_and_materializes_new_id(tmp_path):
    calls = []

    def reconcile(edited_text, old_units, candidate_sources):
        calls.append(edited_text)
        return ReconciliationResult(
            {}, (("The user started a new job.", "2026-02-01", ("D1:2",)),), ()
        )

    workspace = MemoryWorkspace(tmp_path, reconciler=reconcile)
    workspace.archive_sessions([{
        "observation_date": "2026-01-01",
        "turns": [("user", "I moved."), ("user", "I started a new job.")],
        "refs": ["D1:1", "D1:2"],
    }])
    workspace.save_memory([{
        "when": "2026-01-01",
        "content": "The user moved.",
        "refs": ["D1:1"],
        "topic_path": "life.md",
        "headings": ["Life"],
    }])

    result = workspace.shell(
        "printf '%s\n' 'The user started a new job.' >> topics/life.md"
    )

    assert result.returncode == 0
    assert len(calls) == 1
    text = (tmp_path / "topics/life.md").read_text()
    assert "The user started a new job.[^mem_" in text
    assert "Sources: [D1:2]" in text


def test_workspace_retries_invalid_reconciliation_once(tmp_path):
    calls = []

    def reconcile(_text, old_units, _sources):
        calls.append(1)
        quote = "not present" if len(calls) == 1 else "Updated fact."
        return ReconciliationResult({old_units[0].memory_id: quote}, (), ())

    workspace = MemoryWorkspace(tmp_path, reconciler=reconcile)
    workspace.archive_sessions([{"observation_date": "2026-01-01", "turns": [("user", "fact")], "refs": ["D1:1"]}])
    workspace.save_memory([{"when": "2026-01-01", "content": "Old fact.", "refs": ["D1:1"], "topic_path": "a.md", "headings": ["A"]}])

    workspace.shell("sed -i '' 's/Old fact/Updated fact/' topics/a.md")

    assert len(calls) == 2


def test_workspace_rejects_source_edits_and_restores_stage(tmp_path):
    workspace = MemoryWorkspace(tmp_path)
    workspace.archive_sessions([{"observation_date": "2026-01-01", "turns": [("user", "fact")], "refs": ["D1:1"]}])
    workspace._refresh_stage()

    with pytest.raises(ValueError, match="Source Memory is append-only"):
        workspace.shell("printf '%s\n' 'tampered' > sources/D1.md")

    assert "tampered" not in (tmp_path / "sources/D1.md").read_text()


def test_llm_reconciler_parses_the_runtime_contract():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"matches":{"mem_a":"updated"},"creates":['
        '{"content":"new","when":null,"source_refs":["D1:2"]}],'
        '"deleted_ids":[],"organizational_quotes":[]}'
    ))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: response
    )))

    result = _make_reconciler(client, "test", None)(
        "updated new", [unit()], {"D1:1", "D1:2"}
    )

    assert result.matches == {"mem_a": "updated"}
    assert result.creates == (("new", None, ("D1:2",)),)


def test_workspace_explicit_correction_removes_only_derived_memory(tmp_path):
    def reconcile(_text, old_units, _sources):
        return ReconciliationResult({}, (), (old_units[0].memory_id,))

    workspace = MemoryWorkspace(tmp_path, reconciler=reconcile)
    workspace.archive_sessions([{"observation_date": "2026-01-01", "turns": [("user", "wrong")], "refs": ["D1:1"]}])
    workspace.save_memory([{"when": "2026-01-01", "content": "Wrong fact.", "refs": ["D1:1"], "topic_path": "a.md", "headings": ["A"]}])

    workspace.shell("printf '%s\n' '# A' > topics/a.md", allow_correction=True)

    assert "wrong" in (tmp_path / "sources/D1.md").read_text()
    assert "Wrong fact" not in (tmp_path / "topics/a.md").read_text()
    assert (tmp_path / "recent_events.jsonl").read_text() == ""


def test_workspace_reconciliation_reanchors_ids_after_split(tmp_path):
    def reconcile(_text, old_units, _sources):
        return ReconciliationResult(
            {old_units[0].memory_id: "First fact."},
            (("Second fact.", "2026-01-02", ("D1:2",)),),
            (),
        )

    workspace = MemoryWorkspace(tmp_path, reconciler=reconcile)
    workspace.archive_sessions([{"observation_date": "2026-01-01", "turns": [("user", "first"), ("user", "second")], "refs": ["D1:1", "D1:2"]}])
    workspace.save_memory([{"when": "2026-01-01", "content": "Combined fact.", "refs": ["D1:1"], "topic_path": "a.md", "headings": ["A"]}])
    old_id = parse_topic_tree(tmp_path / "topics")[0].memory_id

    workspace.shell("sed -i '' 's/Combined fact/First fact. Second fact/' topics/a.md")

    units = parse_topic_tree(tmp_path / "topics")
    assert [(row.memory_id, row.content) for row in units] == [
        (old_id, "First fact."),
        (units[1].memory_id, "Second fact."),
    ]


def test_workspace_reconciliation_supports_two_ids_on_merged_text(tmp_path):
    def reconcile(_text, old_units, _sources):
        return ReconciliationResult(
            {unit.memory_id: "Merged fact." for unit in old_units}, (), ()
        )

    workspace = MemoryWorkspace(tmp_path, reconciler=reconcile)
    workspace.archive_sessions([{"observation_date": "2026-01-01", "turns": [("user", "one"), ("user", "two")], "refs": ["D1:1", "D1:2"]}])
    workspace.save_memory([
        {"when": "2026-01-01", "content": "First fact.", "refs": ["D1:1"], "topic_path": "a.md", "headings": ["A"]},
        {"when": "2026-01-02", "content": "Second fact.", "refs": ["D1:2"], "topic_path": "a.md", "headings": ["A"]},
    ])
    old_ids = [row.memory_id for row in parse_topic_tree(tmp_path / "topics")]

    workspace.shell(
        "sed -i '' "
        f"-e 's/First fact.\\[\\^{old_ids[0]}\\]/Merged fact./' "
        f"-e '/Second fact.\\[\\^{old_ids[1]}\\]/d' topics/a.md"
    )

    units = parse_topic_tree(tmp_path / "topics")
    assert [row.memory_id for row in units] == old_ids
    assert [row.content for row in units] == ["Merged fact.", "Merged fact."]
