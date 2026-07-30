import os
from src.v8_memory import (_sanitize_topic, write_events, dedup_topic_files,
                           _topics_dir_files, _topic_merge_candidates)


# ---- _sanitize_topic：分段清洗 / 两级封顶 / 兼容平铺 ----

def test_sanitize_flat_topic_unchanged():
    assert _sanitize_topic("Caroline-adoption") == "Caroline-adoption"
    assert _sanitize_topic("support group") == "support-group"


def test_sanitize_arbitrary_depth_kept():
    # 深度由模型决定：任意合法深度都保留
    assert _sanitize_topic("Caroline/adoption") == "Caroline/adoption"
    assert _sanitize_topic("Caroline/adoption/legal") == "Caroline/adoption/legal"
    assert _sanitize_topic("Melanie/family life") == "Melanie/family-life"


def test_sanitize_each_segment_cleaned_independently():
    # 每段单独清洗，不会把 / 拍成 -（老 bug）
    assert _sanitize_topic("A B/c d") == "A-B/c-d"


def test_sanitize_rejects_dotdot_and_absolute():
    # .. 被丢，绝对路径前导 / 产生的空段被丢
    assert _sanitize_topic("a/../b") == "a/b"
    assert _sanitize_topic("/etc/passwd") == "etc/passwd"
    assert _sanitize_topic("../../secret") == "secret"


def test_sanitize_caps_at_max_depth(monkeypatch):
    # 默认 MAX_DEPTH=4：第 5 段起折进第 4 段
    assert _sanitize_topic("a/b/c/d/e") == "a/b/c/d-e"
    monkeypatch.setenv("NATIVEMEM_V8_MAX_DEPTH", "2")
    assert _sanitize_topic("a/b/c/d") == "a/b-c-d"


def test_sanitize_empty_segment_falls_back():
    assert _sanitize_topic("//") == "misc"
    assert _sanitize_topic("") == "misc"
    # 空第一段回退，但保留有内容的第二段
    assert _sanitize_topic("/adoption") == "adoption"


# ---- write_events：topics/ 固定入口，门内层级由模型决定 ----

def test_write_events_creates_subdir(tmp_path):
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "researched adoption",
                      "dia_ids": ["D1:3"], "topic": "Caroline/adoption"}])
    p = os.path.join(d, "topics", "Caroline", "adoption.md")
    assert os.path.exists(p)
    assert "researched adoption" in open(p).read()


def test_write_events_three_level_walked(tmp_path):
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "kids school",
                      "dia_ids": ["D1:9"], "topic": "Melanie/family/kids"}])
    p = os.path.join(d, "topics", "Melanie", "family", "kids.md")
    assert os.path.exists(p)
    assert "Melanie/family/kids" in _topics_dir_files(d)


def test_write_events_flat_topic(tmp_path):
    # 一级话题名落在 topics/ 直属（老库平铺格式，天然兼容）
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "x", "dia_ids": ["D1:1"],
                      "topic": "support"}])
    assert os.path.exists(os.path.join(d, "topics", "support.md"))
    assert "support" in _topics_dir_files(d)


def test_write_events_root_has_exactly_two_views(tmp_path):
    # 根目录恒两个视图：timeline/（代码路由）+ topics/（模型门内自由）
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "x", "dia_ids": ["D1:1"],
                      "topic": "Caroline/adoption"}])
    assert sorted(os.listdir(d)) == ["timeline", "topics"]
    # 模型真起名 timeline 也只是 topics/timeline.md，和代码的 timeline/ 不冲突
    write_events(d, [{"when": "2023-05-07", "summary": "y", "dia_ids": ["D1:2"],
                      "topic": "timeline"}])
    assert os.path.exists(os.path.join(d, "topics", "timeline.md"))
    assert sorted(os.listdir(d)) == ["timeline", "topics"]


# ---- 遍历函数在两级结构下工作 ----

def test_topics_dir_files_walks_subdirs(tmp_path):
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-07", "summary": "a", "dia_ids": ["D1:1"], "topic": "Caroline/adoption"},
        {"when": "2023-05-07", "summary": "b", "dia_ids": ["D1:2"], "topic": "Melanie/family"},
        {"when": "2023-05-07", "summary": "c", "dia_ids": ["D1:3"], "topic": "hobbies"},
    ])
    names = _topics_dir_files(d)
    assert "Caroline/adoption" in names
    assert "Melanie/family" in names
    assert "hobbies" in names


def test_dedup_works_in_nested(tmp_path):
    d = str(tmp_path)
    ev = {"when": "2023-05-07", "summary": "dup", "dia_ids": ["D1:1"], "topic": "Caroline/adoption"}
    write_events(d, [ev])
    write_events(d, [ev])  # 完全相同的一行写两次
    removed = dedup_topic_files(d)
    # timeline 侧重复已被层1（写入路径精确去重）挡在门外，只剩 topic 文件那份被 dedup
    assert removed == 1
    p = os.path.join(d, "topics", "Caroline", "adoption.md")
    assert open(p).read().count("dup") == 1


def test_merge_candidates_use_hierarchical_names(tmp_path):
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-07", "summary": "a", "dia_ids": ["D1:1"], "topic": "Caroline/adoption-journey"},
        {"when": "2023-05-07", "summary": "b", "dia_ids": ["D1:2"], "topic": "Caroline/adoption-process"},
    ])
    cands = _topic_merge_candidates(d)
    flat = [name for g in cands for name in g]
    # 候选名带路径（能被 consolidate 用路径定位文件）
    assert any("Caroline/adoption" in n for n in flat)
