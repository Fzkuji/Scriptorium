import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8


def _seed(d):
    # topics/ 门内分层：Caroline/ 下 2 个文件、Melanie/ 1 个
    V8.write_events(d, [
        {"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "Caroline/support"},
        {"when": "2023-05-08", "summary": "B", "dia_ids": ["D1:2"], "topic": "Caroline/adoption"},
        {"when": "2023-06-02", "summary": "C", "dia_ids": ["D2:1"], "topic": "Melanie/art"},
    ])


def test_map_includes_timeline_and_person_dirs(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="dir")
    assert "timeline" in m          # 代码索引目录必须在
    assert "topics" in m            # 固定入口
    assert "Caroline" in m          # 门内模型建的人名目录也在


def test_map_declares_timeline_path_template(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="dir")
    # timeline 段头部固定声明纯数字三层路径模板，模型看地图即知年/月/日
    assert "timeline/YYYY/MM/DD.md" in m


def test_map_dir_mode_does_not_flatten_all_topic_files(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="dir")
    # dir 模式给每个目录汇总而非逐个 basename：Caroline/ 下 2 个文件折叠
    assert "support" not in m
    assert "2 files" in m or "2 file" in m


def test_map_files_mode_lists_topic_files_with_counts(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="files")
    assert "support" in m
    assert "1" in m


def test_map_no_hardcoded_dirname_walks_custom_dirs(tmp_path):
    # 模型自建的任意目录也要如实呈现(不写死 topics/timeline)
    d = str(tmp_path / "mem")
    os.makedirs(os.path.join(d, "people", "sub"), exist_ok=True)
    with open(os.path.join(d, "people", "sub", "x.md"), "w") as f:
        f.write("[2023-05-07] hi · [D1:1]\n")
    m = R._v8_structure_map(d, mode="dir")
    assert "people" in m


def test_map_empty(tmp_path):
    assert R._v8_structure_map(str(tmp_path / "empty"), mode="dir") == "(empty memory)"
