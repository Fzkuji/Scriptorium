import os, json
import src.adapters.run_nativemem as R
import src.v8_memory as V8

def test_v8_build_writes_dual_view(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    conv = {
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"},
        ],
        "session_1_date_time": "2023-05-07",
    }
    # fake distill：不联网，返回一个事件
    monkeypatch.setattr(V8, "distill_events",
        lambda text, obs, dids, known_topics=None, **k: [
            {"when": "2023-05-07", "summary": "去支持小组", "dia_ids": ["D1:3"], "topic": "support"}])
    d = str(tmp_path / "mem")
    R.build_memory(conv, d)
    assert os.path.exists(os.path.join(d, "timeline", "2023", "05", "07.md"))
    assert os.path.exists(os.path.join(d, "topics", "support.md"))
    assert "去支持小组" in open(os.path.join(d, "topics", "support.md")).read()


def test_v8_build_threads_known_topics_across_sessions(tmp_path, monkeypatch):
    # build_memory 的 v8 分支按 session 顺序调 distill_events，且把上一 session
    # 产出的话题名累积进 known_topics 传给下一 session —— 验证跨 session 真的
    # 传递了状态，而不是每个 session 各算各的。
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    conv = {
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to a LGBTQ support group"},
        ],
        "session_1_date_time": "2023-05-07",
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "researching adoption options"},
        ],
        "session_2_date_time": "2023-06-02",
    }

    received_known_topics = []

    def fake_distill(turns, obs, dids, known_topics=None, **k):
        received_known_topics.append(list(known_topics) if known_topics else [])
        n = len(received_known_topics)
        return [{"when": obs or "2023-01-01", "summary": f"事件{n}",
                 "dia_ids": list(dids), "topic": f"topic-{n}"}]

    # build_memory 调用的是模块限定名 v8_memory.distill_events（不是 import 进
    # run_nativemem 命名空间里的那个引用），所以要 patch 到 v8_memory 模块本身，
    # 一个 plain attribute patch 到 R 里 import 的 distill_events 不会生效。
    monkeypatch.setattr(V8, "distill_events", fake_distill)

    d = str(tmp_path / "mem")
    R.build_memory(conv, d)

    assert len(received_known_topics) == 2
    # session_1 是第一次调用，此时还没有任何话题积累
    assert received_known_topics[0] == []
    # session_2 调用时，应该已经带上了 session_1 产出的 topic-1（新格式带计数）
    assert any("topic-1" in item for item in received_known_topics[1])


def test_v8_build_dedups_each_session(tmp_path, monkeypatch):
    # 每 session 末（NATIVEMEM_V8_TIDY 默认 on）调 v8_memory.dedup_topic_files。
    # fake distill 每次返回同一条事件行，两次 chunk 会写出重复行，session 末
    # 去重把它并成一条。
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_CHUNK_TURNS", "1")  # 每 turn 一块 → 两次 distill
    conv = {
        "session_1": [
            {"speaker": "C", "dia_id": "D1:1", "text": "went to a support group"},
            {"speaker": "C", "dia_id": "D1:2", "text": "went to a support group again"},
        ],
        "session_1_date_time": "2023-05-07",
    }
    monkeypatch.setattr(V8, "distill_events",
        lambda turns, obs, dids, known_topics=None, **k: [
            {"when": "2023-05-07", "summary": "dup", "dia_ids": ["D1:1"], "topic": "t"}])
    d = str(tmp_path / "mem")
    R.build_memory(conv, d)
    body = open(os.path.join(d, "topics", "t.md")).read()
    assert body.count("dup ·") == 1   # session 末去重生效


def test_v8_build_triggers_consolidate_over_max_topics(tmp_path, monkeypatch):
    # topic 文件数 > NATIVEMEM_V8_MAX_TOPICS 时 session 末触发 consolidate。
    # 阈值设 2，fake distill 每 chunk 造一个不同 topic → 3 个文件 → 超阈值。
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_CHUNK_TURNS", "1")
    monkeypatch.setenv("NATIVEMEM_V8_MAX_TOPICS", "2")
    conv = {
        "session_1": [
            {"speaker": "C", "dia_id": "D1:1", "text": "a"},
            {"speaker": "C", "dia_id": "D1:2", "text": "b"},
            {"speaker": "C", "dia_id": "D1:3", "text": "c"},
        ],
        "session_1_date_time": "2023-05-07",
    }
    counter = {"n": 0}

    def fake_distill(turns, obs, dids, known_topics=None, **k):
        counter["n"] += 1
        return [{"when": "2023-05-07", "summary": "x",
                 "dia_ids": list(dids), "topic": f"topic-{counter['n']}"}]
    monkeypatch.setattr(V8, "distill_events", fake_distill)
    called = {"n": 0}
    monkeypatch.setattr(V8, "consolidate_topic_files",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or 0)
    d = str(tmp_path / "mem")
    R.build_memory(conv, d)
    assert called["n"] >= 1   # 3 > 2 → 触发合并


def test_v8_build_no_consolidate_under_max_topics(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    monkeypatch.setenv("NATIVEMEM_V8_MAX_TOPICS", "30")
    conv = {
        "session_1": [{"speaker": "C", "dia_id": "D1:1", "text": "a"}],
        "session_1_date_time": "2023-05-07",
    }
    monkeypatch.setattr(V8, "distill_events",
        lambda turns, obs, dids, known_topics=None, **k: [
            {"when": "2023-05-07", "summary": "x", "dia_ids": ["D1:1"], "topic": "t"}])
    called = {"n": 0}
    monkeypatch.setattr(V8, "consolidate_topic_files",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or 0)
    R.build_memory(conv, str(tmp_path / "mem"))
    assert called["n"] == 0   # 1 topic <= 30 → 不触发


def test_distill_receives_known_topics_with_counts(tmp_path, monkeypatch):
    # 块三：传给 distill 的 known_topics 从纯名字集升级为带条目数的清单。
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v8")
    seen = []

    def fake_distill(turns, obs, dids, known_topics=None, **k):
        seen.append(list(known_topics) if known_topics else [])
        return [{"when": obs or "2023-01-01", "summary": "e",
                 "dia_ids": list(dids), "topic": "Caroline-support"}]
    monkeypatch.setattr(V8, "distill_events", fake_distill)
    conv = {"session_1": [{"speaker": "C", "dia_id": "D1:1", "text": "support group"}],
            "session_1_date_time": "2023-05-07",
            "session_2": [{"speaker": "C", "dia_id": "D2:1", "text": "more support"}],
            "session_2_date_time": "2023-06-02"}
    R.build_memory(conv, str(tmp_path / "mem"))
    # session_2 传入的 known 应含带计数形态的 Caroline-support（session1 落了 1 条）
    joined = " ".join(seen[1])
    assert "Caroline-support" in joined and "1" in joined


def test_topics_snapshot_orders_by_count_desc(tmp_path):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "a", "dia_ids": ["D1:1"], "topic": "big"}])
    V8.write_events(d, [{"when": "2023-05-08", "summary": "b", "dia_ids": ["D1:2"], "topic": "big"}])
    V8.write_events(d, [{"when": "2023-05-09", "summary": "c", "dia_ids": ["D1:3"], "topic": "small"}])
    snap = R._topics_snapshot(d)
    assert snap[0][0] == "big" and snap[0][1] == 2
    assert snap[1] == ("small", 1)


def test_reload_known_topics_returns_current_names(tmp_path):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "a", "dia_ids": ["D1:1"], "topic": "keep"}])
    assert R._reload_known_topics(d) == {"keep"}
