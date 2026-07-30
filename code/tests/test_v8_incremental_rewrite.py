"""Incremental article rewrite: trigger by raw-line growth per file, not per
session. Raw (unarticled) lines start with `[YYYY-MM-DD]`; articled prose does
not (it starts with `## ` or `On [date]...`). See src/v8_memory.py."""
import os
import src.v8_memory as V8


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


def _fake(monkeypatch, content):
    calls = []

    def create(**k):
        calls.append(k["messages"])
        return _FakeResp(content)
    monkeypatch.setattr(V8.client.chat.completions, "create", create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    return calls


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# ---- raw-line counting -------------------------------------------------

def test_raw_count_pure_article_is_zero(tmp_path):
    p = str(tmp_path / "a.md")
    _write(p, "## Title\nOn [2023-05-25](timeline/2023/05/25.md), she adopted [D2:3].\n")
    assert V8._raw_line_count(p) == 0


def test_raw_count_pure_raw(tmp_path):
    p = str(tmp_path / "a.md")
    _write(p, "[2023-05-25] she adopted [D2:3]\n[2023-05-26] and visited [D2:8]\n")
    assert V8._raw_line_count(p) == 2


def test_raw_count_mixed_counts_only_tail_raw(tmp_path):
    p = str(tmp_path / "a.md")
    _write(p, "## Title\nOn [2023-05-25](timeline/2023/05/25.md), she adopted [D2:3].\n"
              "[2023-05-27] new note one [D3:1]\n[2023-05-28] new note two [D3:2]\n")
    assert V8._raw_line_count(p) == 2


def test_raw_count_linked_date_prefix_still_raw(tmp_path):
    # write_events runs link_dates_to_timeline, so a raw appended line's leading
    # [date] becomes [date](timeline/...). It's still a raw line.
    p = str(tmp_path / "a.md")
    _write(p, "[2023-05-25](timeline/2023/05/25.md) she adopted [D2:3]\n")
    assert V8._raw_line_count(p) == 1


# ---- trigger threshold -------------------------------------------------

def _seed_article_plus_raw(d, name, n_raw):
    """Articled file with n_raw raw tail lines appended (each unique dia_id)."""
    path = os.path.join(d, "topics", *name.split("/")) + ".md"
    body = "## Story\nOn [2023-05-01](timeline/2023/05/01.md), it began [D1:1].\n"
    for i in range(n_raw):
        body += f"[2023-05-{2+i:02d}] note {i} [D2:{i}]\n"
    _write(path, body)
    return path


def test_below_threshold_not_triggered(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_article_plus_raw(d, "t", 3)          # 3 raw < default 8
    before = open(p).read()
    calls = _fake(monkeypatch, "## rewritten [D1:1] [D2:0] [D2:1] [D2:2]\n")
    assert V8.rewrite_topic_articles(d, {"t"}) == 0
    assert calls == []                              # model never called
    assert open(p).read() == before                 # untouched


def test_at_threshold_triggered(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_article_plus_raw(d, "t", 8)           # 8 raw == default 8
    dia = [ln for ln in open(p) if "[D" in ln]
    # article must preserve all dia_ids
    ids = " ".join(sorted(set(V8._DIA_RE.findall(open(p).read()))))
    calls = _fake(monkeypatch, f"## rewritten article {ids}\n")
    assert V8.rewrite_topic_articles(d, {"t"}) == 1
    assert len(calls) == 1
    assert "rewritten article" in open(p).read()


def test_never_articled_small_file_triggered(tmp_path, monkeypatch):
    # a file that's all raw with >=3 lines and never articled -> trigger even
    # though raw < threshold.
    d = str(tmp_path)
    path = os.path.join(d, "topics", "t.md")
    _write(path, "[2023-05-01] a [D1:1]\n[2023-05-02] b [D1:2]\n[2023-05-03] c [D1:3]\n")
    calls = _fake(monkeypatch, "## art [D1:1] [D1:2] [D1:3]\n")
    assert V8.rewrite_topic_articles(d, {"t"}) == 1
    assert len(calls) == 1


def test_never_articled_tiny_file_not_triggered(tmp_path, monkeypatch):
    # all-raw but only 2 lines (< 3) and below threshold -> wait, accumulate.
    d = str(tmp_path)
    path = os.path.join(d, "topics", "t.md")
    _write(path, "[2023-05-01] a [D1:1]\n[2023-05-02] b [D1:2]\n")
    calls = _fake(monkeypatch, "## art [D1:1] [D1:2]\n")
    assert V8.rewrite_topic_articles(d, {"t"}) == 0
    assert calls == []


# ---- final sweep -------------------------------------------------------

def test_final_sweep_rewrites_any_raw(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_article_plus_raw(d, "t", 2)           # only 2 raw, below threshold
    ids = " ".join(sorted(set(V8._DIA_RE.findall(open(p).read()))))
    calls = _fake(monkeypatch, f"## final {ids}\n")
    # normal pass: skipped
    assert V8.rewrite_topic_articles(d, {"t"}) == 0
    # final sweep: forced
    assert V8.rewrite_topic_articles(d, {"t"}, final=True) == 1
    assert "final" in open(p).read()


def test_final_sweep_skips_pure_article(tmp_path, monkeypatch):
    d = str(tmp_path)
    path = os.path.join(d, "topics", "t.md")
    _write(path, "## Done\nOn [2023-05-01](timeline/2023/05/01.md), done [D1:1].\n")
    before = open(path).read()
    calls = _fake(monkeypatch, "## should-not-happen [D1:1]\n")
    assert V8.rewrite_topic_articles(d, {"t"}, final=True) == 0
    assert calls == []                              # no raw lines -> no call
    assert open(path).read() == before


# ---- hard check still holds -------------------------------------------

def test_dia_mismatch_keeps_original(tmp_path, monkeypatch):
    d = str(tmp_path)
    p = _seed_article_plus_raw(d, "t", 8)
    before = open(p).read()
    _fake(monkeypatch, "## lost refs [D1:1]\n")     # dropped all D2:* ids
    assert V8.rewrite_topic_articles(d, {"t"}) == 0
    assert open(p).read() == before


# ---- build wiring ------------------------------------------------------

def test_build_final_sweep_called(tmp_path, monkeypatch):
    import src.adapters.run_nativemem as R
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_V8_ARTICLE", "on")
    conv = {"session_1": [{"speaker": "C", "dia_id": "D1:1", "text": "hi"}],
            "session_1_date_time": "2023-05-07",
            "session_2": [{"speaker": "C", "dia_id": "D2:1", "text": "yo"}],
            "session_2_date_time": "2023-05-08"}
    monkeypatch.setattr(V8, "distill_events",
        lambda turns, obs, dids, known_topics=None, **k: [
            {"when": "2023-05-07", "summary": "x", "dia_ids": list(dids),
             "topic": "t"}])
    seen = []
    monkeypatch.setattr(V8, "rewrite_topic_articles",
                        lambda md, touched, final=False, **k: seen.append(final) or 0)
    R.build_memory(conv, str(tmp_path / "m"))
    # per-session calls (final=False) + one final sweep (final=True) at the end
    assert seen[-1] is True
    assert seen.count(True) == 1
