import os
import re
import nativemem

def test_measure_library_counts(tmp_path):
    (tmp_path / "a.md").write_text("## t\n[2023-01-01] one\n[2023-01-02] two\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("[2023-03-03] three\n")
    m = nativemem.measure_library(str(tmp_path))
    assert m["files"] == 2
    assert m["dirs"] == 1
    assert m["entries"] == 3
    assert m["bytes"] > 0

def test_grew_by_file_delta():
    old = {"files": 2, "dirs": 0, "bytes": 100, "entries": 5}
    new = {"files": 10, "dirs": 1, "bytes": 150, "entries": 9}
    # (10+1)-(2+0)=9 >= 8 → True
    assert nativemem.library_grew_past_threshold(old, new, growth_ratio=2.0, file_delta=8)

def test_grew_by_bytes_ratio():
    old = {"files": 2, "dirs": 0, "bytes": 100, "entries": 5}
    new = {"files": 3, "dirs": 0, "bytes": 250, "entries": 9}
    # file delta 1 < 8，但 250/100=2.5 >= 2.0 → True
    assert nativemem.library_grew_past_threshold(old, new, growth_ratio=2.0, file_delta=8)

def test_not_grown():
    old = {"files": 2, "dirs": 0, "bytes": 100, "entries": 5}
    new = {"files": 3, "dirs": 0, "bytes": 120, "entries": 6}
    assert not nativemem.library_grew_past_threshold(old, new, growth_ratio=2.0, file_delta=8)

def test_top_level_view_lists_direct_children(tmp_path):
    (tmp_path / "Caroline.md").write_text("## x\n[2023-01-01] a\n")
    (tmp_path / "Melanie.md").write_text("## y\n[2023-01-01] b\n")
    d = tmp_path / "projects"
    d.mkdir()
    (d / "p1.md").write_text("z")
    (d / "p2.md").write_text("z")
    (d / "p3.md").write_text("z")
    (tmp_path / ".hidden").write_text("nope")

    view = nativemem._top_level_view(str(tmp_path))
    assert "Caroline.md" in view
    assert "Melanie.md" in view
    assert "projects/  (3 items)" in view
    assert ".hidden" not in view

def test_top_level_view_empty(tmp_path):
    assert nativemem._top_level_view(str(tmp_path)) == "(empty memory)"

def _entries(n):
    return "".join(f"[2023-01-{i%28+1:02d}] fact {i}\n" for i in range(n))

def test_inspect_flags_big_file(tmp_path):
    (tmp_path / "Big.md").write_text("## t\n" + _entries(200))
    res = nativemem.inspect_structure(str(tmp_path), big=150, small=8, wide=20)
    kinds = {(w["kind"], w["target"]) for w in res["warnings"]}
    assert ("big", "Big.md") in kinds
    assert "⚠" in res["report"]

def test_inspect_flags_small_file(tmp_path):
    (tmp_path / "Tiny.md").write_text("## t\n" + _entries(3))
    res = nativemem.inspect_structure(str(tmp_path), big=150, small=8, wide=20)
    assert any(w["kind"] == "small" and w["target"] == "Tiny.md" for w in res["warnings"])

def test_inspect_flags_wide_dir(tmp_path):
    d = tmp_path / "groups"
    d.mkdir()
    for i in range(25):
        (d / f"f{i}.md").write_text(_entries(10))
    res = nativemem.inspect_structure(str(tmp_path), big=150, small=8, wide=20)
    assert any(w["kind"] == "wide" and w["target"].startswith("groups") for w in res["warnings"])

def test_inspect_clean_no_warnings(tmp_path):
    (tmp_path / "Ok.md").write_text("## t\n" + _entries(20))
    res = nativemem.inspect_structure(str(tmp_path), big=150, small=8, wide=20)
    assert res["warnings"] == []

def test_inspect_no_false_dup_from_md_suffix(tmp_path):
    """Regression: .md suffix pollution should NOT cause unrelated files to be flagged as duplicates.

    "career log.md" / "diet log.md" / "books log.md" / "travel log.md" share only the
    generic word "log" (3 chars, filtered out of the salient-word clustering signal in
    _find_merge_candidates, which requires len(w) >= 4). Their word-level Jaccard is
    2/6 = 0.3333 on the clean names ("career"/"log" vs "diet"/"log"), just BELOW the
    0.34 threshold, so they should never be flagged as duplicates.

    But with the raw ".md" filename fed into the similarity calc, the extra shared
    token "md" pushes the same pair to 3/6 = 0.5000, which is ABOVE the threshold and
    causes _find_merge_candidates to (wrongly) group all four files as duplicates.
    This is exactly the scenario the fix (stripping .md before computing similarity)
    is supposed to prevent, so this test fails on the pre-fix code and passes on the
    fixed code.
    """
    (tmp_path / "career log.md").write_text("## t\n" + _entries(20))
    (tmp_path / "diet log.md").write_text("## t\n" + _entries(20))
    (tmp_path / "books log.md").write_text("## t\n" + _entries(20))
    (tmp_path / "travel log.md").write_text("## t\n" + _entries(20))

    res = nativemem.inspect_structure(str(tmp_path), big=150, small=8, wide=20)
    # Should NOT have any dup warnings for these unrelated files
    dup_warnings = [w for w in res["warnings"] if w["kind"] == "dup"]
    assert len(dup_warnings) == 0, f"False dup warning: {dup_warnings}"


def test_tidy_local_noop_when_clean(tmp_path, monkeypatch):
    (tmp_path / "Ok.md").write_text("## t\n" + _entries(20))
    # 干净库：inspect 无 ⚠，consolidate_topics 也不该改动结构 → 不调 create
    monkeypatch.setattr(nativemem, "consolidate_topics", lambda *a, **k: 0)
    def boom(**kwargs):
        raise AssertionError("clean library must not call model in tidy_local")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", boom)
    n = nativemem.tidy_local(str(tmp_path), big=150, small=8, wide=20)
    assert n == 0


def test_reorganize_calls_model_with_report(tmp_path, monkeypatch):
    (tmp_path / "Big.md").write_text("## t\n" + _entries(200))
    seen = {"prompt": ""}
    class R:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": "done", "tool_calls": None})()})()]
            self.usage = type("U", (), {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2})()
    def fake_create(**kwargs):
        seen["prompt"] = kwargs["messages"][0]["content"]
        return R()
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)
    nativemem.reorganize_library(str(tmp_path), big=150, small=8, wide=20)
    # 报告里的 ⚠ 过大信息进了 prompt
    assert "⚠" in seen["prompt"] and "Big.md" in seen["prompt"]


def test_rebalance_prompt_summary_page_on(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_SUMMARY_PAGE", "on")
    (tmp_path / "Big.md").write_text("## t\n" + _entries(200))
    seen = {"prompt": ""}
    class R:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": "done", "tool_calls": None})()})()]
            self.usage = type("U", (), {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2})()
    def fake_create(**kwargs):
        seen["prompt"] = kwargs["messages"][0]["content"]
        return R()
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)
    nativemem.reorganize_library(str(tmp_path), big=150, small=8, wide=20)
    assert "摘要页" in seen["prompt"]


def test_rebalance_prompt_summary_page_off(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_SUMMARY_PAGE", "off")
    (tmp_path / "Big.md").write_text("## t\n" + _entries(200))
    seen = {"prompt": ""}
    class R:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": "done", "tool_calls": None})()})()]
            self.usage = type("U", (), {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2})()
    def fake_create(**kwargs):
        seen["prompt"] = kwargs["messages"][0]["content"]
        return R()
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)
    nativemem.reorganize_library(str(tmp_path), big=150, small=8, wide=20)
    assert "摘要页" not in seen["prompt"]
