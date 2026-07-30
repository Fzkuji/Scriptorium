"""dia_id 区间引用语法：expand / dia_ids_in / read_turns 整段 / 守卫并集 / prompt 关键词。"""
import json

import src.v8_memory as V8
from src.v8_memory import (expand_dia_ids, dia_ids_in, build_turn_index,
                           read_turns, _refs_to_dia_ids)


# ---- expand_dia_ids：四种形态 + 容错 ----

def test_expand_single_is_identity():
    assert expand_dia_ids("D2:8") == ["D2:8"]


def test_expand_range_inclusive():
    assert expand_dia_ids("D2:3-6") == ["D2:3", "D2:4", "D2:5", "D2:6"]


def test_expand_enumeration():
    assert expand_dia_ids("D2:2,10") == ["D2:2", "D2:10"]


def test_expand_mixed():
    assert expand_dia_ids("D2:3-5,9,12") == [
        "D2:3", "D2:4", "D2:5", "D2:9", "D2:12"]


def test_expand_dedup_preserves_order():
    assert expand_dia_ids("D1:2, D1:1-3") == ["D1:2", "D1:1", "D1:3"]


def test_expand_reversed_range_still_works():
    assert expand_dia_ids("D1:5-3") == ["D1:3", "D1:4", "D1:5"]


def test_expand_invalid_returns_as_single_id():
    # 非法输入按原样单 id 处理，别崩
    assert expand_dia_ids("garbage") == ["garbage"]
    assert expand_dia_ids("") == []
    assert expand_dia_ids(None) == []


def test_dia_ids_in_extracts_and_expands_from_line():
    line = "[2023-05-25] chatted about the trip · [D2:3-5, D7:9]"
    assert dia_ids_in(line) == ["D2:3", "D2:4", "D2:5", "D7:9"]


# ---- read_turns：区间整段读，散句带窗口 ----

def _conv():
    return {"session_1": [
        {"speaker": "C", "dia_id": "D1:1", "text": "s1"},
        {"speaker": "M", "dia_id": "D1:2", "text": "s2"},
        {"speaker": "C", "dia_id": "D1:3", "text": "s3"},
        {"speaker": "M", "dia_id": "D1:4", "text": "s4"},
        {"speaker": "C", "dia_id": "D1:5", "text": "s5"},
    ]}


def test_read_turns_range_reads_whole_span_no_extra_window():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D1:2-4"], context=1)
    # 区间整段自成上下文：只 s2,s3,s4，不外扩 s1/s5
    assert "s2" in out and "s3" in out and "s4" in out
    assert "s1" not in out and "s5" not in out


def test_read_turns_single_still_windows():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D1:3"], context=1)
    # 单句仍带前后窗口
    assert "s2" in out and "s3" in out and "s4" in out


def test_read_turns_enumeration_span_no_window():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D1:1,3"], context=1)
    # 列举也算"整段"（同一 ref 多句），不各自补窗口
    assert "s1" in out and "s3" in out
    assert "s2" not in out and "s4" not in out


# ---- 并集守卫在区间形式下正确 ----

def test_merge_guard_union_balanced_with_range_lines():
    # keep 行带区间锚，merge 行独有单句 → 追加后并集相等，合并成功
    content = ["[2023-05-08] a [D1:1-3]", "[2023-05-07] b [D1:9]"]
    out = V8._apply_line_merges(
        json.dumps({"groups": [{"keep": 1, "merge": [2]}]}), content)
    assert out is not None
    joined = "\n".join(out)
    assert dia_ids_in(joined) == dia_ids_in("\n".join(content))
    assert "D1:9" in joined                       # merge 独有 id 被追加


def test_merge_guard_range_subsumed_id_not_duplicated():
    # merge 行的 id 已被 keep 行区间覆盖 → 不重复追加
    content = ["[2023-05-08] a [D1:1-3]", "[2023-05-07] b [D1:2]"]
    out = V8._apply_line_merges(
        json.dumps({"groups": [{"keep": 1, "merge": [2]}]}), content)
    assert out is not None
    assert "· [" not in "\n".join(out)[len("[2023-05-08] a [D1:1-3]"):]  # 无追加块


# ---- refs 行号区间 → dia_id ----

def test_refs_line_number_range_expands():
    line_map = {3: "D2:3", 4: "D2:4", 5: "D2:5"}
    assert _refs_to_dia_ids(["3-5"], line_map) == ["D2:3", "D2:4", "D2:5"]


def test_refs_mixed_line_numbers():
    line_map = {3: "D2:3", 4: "D2:4", 9: "D2:9"}
    assert _refs_to_dia_ids(["3-4", 9], line_map) == ["D2:3", "D2:4", "D2:9"]


# ---- prompt 关键词 ----

def test_distill_prompt_mentions_range():
    from src.v8_memory import _V8_DISTILL_PROMPT
    assert "3-15" in _V8_DISTILL_PROMPT
    assert "虚扩" in _V8_DISTILL_PROMPT


def test_retrieve_prompts_mention_range_passthrough():
    from src.adapters.run_nativemem import _V8_RETRIEVE_PROMPT, _V8_SINGLE_PROMPT
    for p in (_V8_RETRIEVE_PROMPT, _V8_SINGLE_PROMPT):
        assert "区间" in p and "原样" in p
