# NativeMem v8 双视图索引 + 回原文检索 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 记忆库把每件事存成"一行摘要 + [dia_id] 指针"，同时按话题(`topics/`)和时间(`timeline/年/月-英文.md`)两个视图各摆一份；答题时查视图定位 → 顺 dia_id 回原始对话读那几句 → 只用相关原文作答。

**Architecture:** 走独立分支 `NATIVEMEM_PROMPT=v8`，不动 v4/v6/v7。写入 = 模型提炼一次事件行(`distill_events`) + 代码落盘双视图(`write_events`)，杜绝 v7 模型手写丢失。检索 = 模型用 grep/cat 查视图 + 工具 `read_turns` 回原文。原文映射由 `locomo10.json` 直接建。

**Tech Stack:** Python 3，OpenAI SDK(阿里云 endpoint，qwen3.6-flash)，pytest，纯 markdown 记忆库。

## Global Constraints

- 模型固定阿里云 `qwen3.6-flash`：复用 `src/nativemem.py` 的 `client` / `ALIYUN_MODEL`，OpenAI client 已 `trust_env=False`（macOS 代理会 502）。测试全用 fake client，不联网。
- 事件行固定格式：`[YYYY-MM-DD] 一句话摘要 · [dia_id]`（多源 `[D1:3, D1:5]`）。日期写完整，不让模型猜。
- 时间视图路径：`timeline/{YYYY}/{MM}-{Mon}.md`，`Mon` 是英文月份三字母（Jan..Dec）。文件内按 `[YYYY-MM-DD]` 升序。
- 话题视图路径：`topics/{topic}.md`。
- with-original 口径：允许回原文，hide_raw 关掉；`NATIVEMEM_NO_ORIGINAL` 过滤在 v8 不启用。
- 只动 v8 分支，v4/v6/v7 行为不变（`NATIVEMEM_PROMPT != "v8"` 时全部原样）。
- 落盘由代码保证写全，绝不让模型手动 bash 写入记忆内容。

---

### Task 1: 回原文 —— dia_id → 原始对话映射 + read_turns

**Files:**
- Create: `src/v8_memory.py`
- Test: `tests/test_v8_original.py`

**Interfaces:**
- Produces:
  - `build_turn_index(conv) -> dict[str, dict]`：把一段 conversation 的所有 turn 建成 `{dia_id: {"speaker","text","order"}}` 映射。`order` 是全局顺序号（用于取前后上下文）。
  - `read_turns(turn_index, dia_ids, context=1) -> str`：给定若干 dia_id，返回对应原文（每个命中 turn 连同前后 `context` 句），拼成 `speaker: text` 文本。无效/不存在的 dia_id 跳过，不报错。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_original.py
from src.v8_memory import build_turn_index, read_turns

def _conv():
    return {
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "hi"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "hello there"},
            {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"},
        ],
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "researching adoption"},
        ],
    }

def test_build_turn_index_maps_every_dia_id():
    idx = build_turn_index(_conv())
    assert set(idx.keys()) == {"D1:1", "D1:2", "D1:3", "D2:1"}
    assert idx["D1:3"]["text"] == "I went to a LGBTQ support group"
    assert idx["D1:3"]["speaker"] == "Caroline"

def test_read_turns_returns_hit_with_context():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D1:3"], context=1)
    # D1:3 的原文 + 前一句(D1:2)作为上下文
    assert "LGBTQ support group" in out
    assert "hello there" in out  # 前一句上下文

def test_read_turns_skips_invalid_ids():
    idx = build_turn_index(_conv())
    out = read_turns(idx, ["D9:99", "D1:1"], context=0)
    assert "hi" in out          # D1:1 命中
    assert out.strip() != ""    # 无效 id 不导致崩或全空
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_original.py -v`
Expected: FAIL（`No module named 'src.v8_memory'`）

- [ ] **Step 3: 实现**

```python
# src/v8_memory.py
"""v8: 双视图索引 + 回原文检索的核心（纯函数，不含大模型调用的落盘/提炼在别的 task）。"""
import os
import re


def build_turn_index(conv):
    """把 conversation 的所有 turn 建成 {dia_id: {speaker,text,order}} 映射。
    order 是跨 session 的全局顺序号，用于 read_turns 取前后上下文。"""
    index = {}
    order = 0
    i = 1
    while f"session_{i}" in conv:
        for t in conv[f"session_{i}"]:
            if not isinstance(t, dict):
                continue
            did = t.get("dia_id")
            if not did:
                continue
            index[did] = {
                "speaker": t.get("speaker", "user"),
                "text": t.get("text", ""),
                "order": order,
            }
            order += 1
        i += 1
    return index


def read_turns(turn_index, dia_ids, context=1):
    """给定 dia_id 列表，返回对应原文（每个命中连同前后 context 句）。
    无效 id 跳过。返回 'speaker: text' 按原顺序拼接、去重的文本。"""
    by_order = {v["order"]: (k, v) for k, v in turn_index.items()}
    want_orders = set()
    for did in dia_ids:
        hit = turn_index.get(did)
        if hit is None:
            continue
        o = hit["order"]
        for oo in range(o - context, o + context + 1):
            if oo in by_order:
                want_orders.add(oo)
    lines = []
    for o in sorted(want_orders):
        _did, v = by_order[o]
        lines.append(f"{v['speaker']}: {v['text']}")
    return "\n".join(lines)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_original.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add src/v8_memory.py tests/test_v8_original.py
git commit -m "feat(v8): build_turn_index + read_turns 回原文映射"
```

---

### Task 2: 事件行落盘 —— write_events 写双视图

**Files:**
- Modify: `src/v8_memory.py`
- Test: `tests/test_v8_write.py`

**Interfaces:**
- Consumes: 无（纯代码落盘）
- Produces:
  - `event_line(when, summary, dia_ids) -> str`：拼一行 `[YYYY-MM-DD] summary · [dia_id]`；多 id → `[D1:3, D1:5]`。
  - `timeline_path(memory_dir, when) -> str`：由日期算出 `timeline/{YYYY}/{MM}-{Mon}.md` 绝对路径（Mon = 英文三字母月）。
  - `write_events(memory_dir, events)`：events 是 `[{"when","summary","dia_ids","topic"}]`；把每个事件行追加到 timeline 对应月份文件（文件内按日期升序插入）和 `topics/{topic}.md`。文件夹/文件不存在就建。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_write.py
import os
from src.v8_memory import event_line, timeline_path, write_events

def test_event_line_format():
    assert event_line("2023-05-07", "去了支持小组", ["D1:3"]) == "[2023-05-07] 去了支持小组 · [D1:3]"
    assert event_line("2023-05-07", "研究收养", ["D2:5", "D2:6"]) == "[2023-05-07] 研究收养 · [D2:5, D2:6]"

def test_timeline_path_year_month_english(tmp_path):
    p = timeline_path(str(tmp_path), "2023-05-07")
    assert p.endswith(os.path.join("timeline", "2023", "05-May.md"))

def test_write_events_dual_view_and_sorted(tmp_path):
    d = str(tmp_path)
    write_events(d, [
        {"when": "2023-05-20", "summary": "研究收养", "dia_ids": ["D2:5"], "topic": "Caroline-adoption"},
        {"when": "2023-05-07", "summary": "去支持小组", "dia_ids": ["D1:3"], "topic": "Caroline-support"},
    ])
    tl = open(os.path.join(d, "timeline", "2023", "05-May.md")).read()
    # 时间视图：两条都在同一月份文件，且按日期升序（05-07 在 05-20 前）
    assert tl.index("05-07") < tl.index("05-20")
    # 话题视图：各自话题文件
    assert "研究收养" in open(os.path.join(d, "topics", "Caroline-adoption.md")).read()
    assert "去支持小组" in open(os.path.join(d, "topics", "Caroline-support.md")).read()

def test_write_events_appends_not_overwrites(tmp_path):
    d = str(tmp_path)
    write_events(d, [{"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "t"}])
    write_events(d, [{"when": "2023-05-08", "summary": "B", "dia_ids": ["D1:2"], "topic": "t"}])
    t = open(os.path.join(d, "topics", "t.md")).read()
    assert "A" in t and "B" in t
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_write.py -v`
Expected: FAIL（`cannot import name 'write_events'`）

- [ ] **Step 3: 实现（追加到 src/v8_memory.py）**

```python
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_LINE_DATE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]")


def event_line(when, summary, dia_ids):
    src = ", ".join(dia_ids) if dia_ids else ""
    return f"[{when}] {summary} · [{src}]"


def timeline_path(memory_dir, when):
    y, m, _d = when.split("-")
    mon = _MONTHS[int(m) - 1]
    return os.path.join(memory_dir, "timeline", y, f"{m}-{mon}.md")


def _sanitize_topic(topic):
    t = re.sub(r"[^\w\- ]", "", topic).strip().replace(" ", "-")
    return t or "misc"


def _append_sorted(path, line):
    """把 line 插入 path，保持文件内按 [YYYY-MM-DD] 升序。无日期行原样保留在前。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
    lines.append(line)

    def key(l):
        m = _LINE_DATE_RE.match(l)
        return m.group(1) if m else ""
    lines.sort(key=key)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_events(memory_dir, events):
    """把每个事件行落盘到时间视图（按月份文件、文件内按日排序）和话题视图。"""
    for ev in events:
        when = ev["when"]
        line = event_line(when, ev["summary"], ev.get("dia_ids", []))
        _append_sorted(timeline_path(memory_dir, when), line)
        topic = _sanitize_topic(ev.get("topic", "misc"))
        topic_path = os.path.join(memory_dir, "topics", f"{topic}.md")
        _append_sorted(topic_path, line)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_write.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/v8_memory.py tests/test_v8_write.py
git commit -m "feat(v8): write_events 事件行落盘双视图(时间/话题)"
```

---

### Task 3: 提炼事件行 —— distill_events

**Files:**
- Modify: `src/v8_memory.py`
- Test: `tests/test_v8_distill.py`

**Interfaces:**
- Consumes: `src.nativemem.client` / `ALIYUN_MODEL` / `log_usage`（测试用 monkeypatch 替换成 fake）
- Produces:
  - `_V8_DISTILL_PROMPT`：提炼 prompt 常量。要求模型输出 JSON `{"events":[{"when","summary","dia_ids","topic"}]}`，只挑值得记的事件、写一句话摘要、归一个话题名、标 dia_id，不复制原文；喂入已有话题名列表让它优先复用。
  - `distill_events(session_text, obs_date, dia_ids, known_topics=None, max_retry=6) -> list[dict]`：调一次模型，稳健解析出事件行列表。解析失败重试，彻底失败返回 `[]`。每个事件 `when` 缺失则回填 `obs_date`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_distill.py
import json
import src.v8_memory as V8

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()

def _install_fake(monkeypatch, content):
    def fake_create(**kwargs):
        return _FakeResp(content)
    monkeypatch.setattr(V8.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)

def test_distill_events_parses_json(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "去支持小组", "dia_ids": ["D1:3"], "topic": "support"},
    ]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events("Caroline: I went to a support group", "2023-05-07", ["D1:3"])
    assert len(evs) == 1
    assert evs[0]["summary"] == "去支持小组"
    assert evs[0]["dia_ids"] == ["D1:3"]

def test_distill_events_backfills_missing_when(monkeypatch):
    payload = json.dumps({"events": [{"summary": "x", "dia_ids": ["D1:1"], "topic": "t"}]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events("x", "2023-05-07", ["D1:1"])
    assert evs[0]["when"] == "2023-05-07"

def test_distill_events_bad_json_returns_empty(monkeypatch):
    _install_fake(monkeypatch, "not json at all")
    evs = V8.distill_events("x", "2023-05-07", ["D1:1"], max_retry=1)
    assert evs == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_distill.py -v`
Expected: FAIL（`cannot import name 'distill_events'` 或 AttributeError）

- [ ] **Step 3: 实现（追加到 src/v8_memory.py，顶部补 import）**

在 `src/v8_memory.py` 顶部已有 import 下补：
```python
import json
import time
from src.nativemem import client, ALIYUN_MODEL, log_usage
```

函数与 prompt：
```python
_V8_DISTILL_PROMPT = """你从一段对话里挑出值得长期记住的【事件】，输出 JSON，不要复制原文。
观测日期：{obs_date}
已有话题名（尽量复用，别造近义新名）：{known_topics}

每个事件给：
- when: 事件日期 YYYY-MM-DD（能推断就推断，推断不出用观测日期）
- summary: 一句话摘要（是什么事，够检索定位即可，别逐字抄）
- dia_ids: 这条事件相关的 turn id（从对话里给的 id 中选）
- topic: 归一个话题名（同一主体+主题，如 Caroline-adoption）

只输出 JSON：{{"events":[{{"when":"2023-05-07","summary":"...","dia_ids":["D1:3"],"topic":"..."}}]}}
没有值得记的就 {{"events":[]}}。"""


def distill_events(session_text, obs_date, dia_ids, known_topics=None, max_retry=6):
    kt = ", ".join(known_topics) if known_topics else "（暂无）"
    prompt = _V8_DISTILL_PROMPT.format(obs_date=obs_date, known_topics=kt)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": session_text}]
    for retry in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL, messages=messages,
                max_tokens=4000, temperature=0.2)
            break
        except Exception:  # noqa: BLE001
            if retry < max_retry - 1:
                time.sleep(3 * (retry + 1))
            else:
                return []
    log_usage(resp, phase="v8_distill")
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
    events = obj.get("events", []) if isinstance(obj, dict) else obj
    out = []
    for e in events:
        if not isinstance(e, dict) or not e.get("summary"):
            continue
        e.setdefault("when", obs_date)
        e.setdefault("dia_ids", [])
        e.setdefault("topic", "misc")
        out.append(e)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_distill.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add src/v8_memory.py tests/test_v8_distill.py
git commit -m "feat(v8): distill_events 一次提炼事件行(不复制原文)"
```

---

### Task 4: build_memory 的 v8 分支

**Files:**
- Modify: `src/adapters/run_nativemem.py`（build_memory 开头加 v8 分支；顶部 import v8_memory）
- Test: `tests/test_v8_build.py`

**Interfaces:**
- Consumes: `distill_events`、`write_events`、`split_into_chunks(session, 50)`、`normalize_date`
- Produces: build_memory 在 `NATIVEMEM_PROMPT=="v8"` 时走 v8 路径：每个 session 一块 → distill_events → write_events；累积 known_topics 传给下个 session。返回 `(build_time, n_events)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_build.py
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
    assert os.path.exists(os.path.join(d, "timeline", "2023", "05-May.md"))
    assert os.path.exists(os.path.join(d, "topics", "support.md"))
    assert "去支持小组" in open(os.path.join(d, "topics", "support.md")).read()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_build.py -v`
Expected: FAIL（v8 分支不存在，走了旧路径不产生 timeline/）

- [ ] **Step 3: 实现**

在 `src/adapters/run_nativemem.py` 顶部 import 区加：
```python
from src.v8_memory import (build_turn_index, read_turns,  # noqa: E402
                           distill_events, write_events)
```

在 `build_memory` 函数体最前面（`raw_dir = ...` 之前）加 v8 分支：
```python
    if os.environ.get("NATIVEMEM_PROMPT") == "v8":
        import time as _t
        t0 = _t.time()
        os.makedirs(memory_dir, exist_ok=True)
        sessions, dates = [], []
        i = 1
        while f"session_{i}" in conv:
            sessions.append(conv[f"session_{i}"])
            dates.append(conv.get(f"session_{i}_date_time", ""))
            i += 1
        if max_sessions:
            sessions, dates = sessions[:max_sessions], dates[:max_sessions]
        known_topics = set()
        n_events = 0
        for session, date in zip(sessions, dates):
            obs = normalize_date(date)
            for chunk_text, dia_ids in split_into_chunks(session, 50):
                if not chunk_text.strip():
                    continue
                events = distill_events(chunk_text, obs, dia_ids,
                                        known_topics=sorted(known_topics))
                if not events:
                    continue
                write_events(memory_dir, events)
                for e in events:
                    known_topics.add(e.get("topic", "misc"))
                n_events += len(events)
        return _t.time() - t0, n_events
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_build.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add src/adapters/run_nativemem.py tests/test_v8_build.py
git commit -m "feat(v8): build_memory v8 分支(按session提炼+落盘双视图)"
```

---

### Task 5: 检索 —— _collect_v8（查视图 + 回原文）

**Files:**
- Modify: `src/adapters/run_nativemem.py`（collect_memories 加 v8 分支 + `_collect_v8` + `_V8_RETRIEVE_PROMPT`）
- Test: `tests/test_v8_retrieve.py`

**Interfaces:**
- Consumes: `execute_tool`（grep/cat 查视图，v8 下 hide_raw=False）、`build_turn_index`/`read_turns`（回原文）、`client`/`ALIYUN_MODEL`
- Produces: `_collect_v8(question, memory_dir, turn_index, max_rounds=6) -> (memories, steps)`：模型用 bash 查 `topics/`、`timeline/` 定位相关事件行 → 从命中行提取 dia_id → 调 `read_turns` 回原文 → 收集相关原文片段作为 memories 返回。`collect_memories` 在 `NATIVEMEM_PROMPT=="v8"` 时调它（需 turn_index，由 main 传入）。

- [ ] **Step 1: 写失败测试（fake 客户端只 grep 一次就收尾，验证回原文接上）**

```python
# tests/test_v8_retrieve.py
import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8

def test_collect_v8_returns_original_via_dia_id(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    # 预置双视图记忆：一条事件行带 [D1:3]
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    # fake 模型：第一轮吐一个 grep tool_call，第二轮无 tool_call 直接给含 dia_id 的答复文本
    calls = {"n": 0}
    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content; self.tool_calls = tool_calls
    class _Resp:
        def __init__(self, msg): self.choices=[type("C",(),{"message":msg})()]; self.usage=type("U",(),{"prompt_tokens":1,"completion_tokens":1})()
    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            tc = type("T", (), {"id":"1","function":type("F",(),{"name":"bash","arguments":'{"cmd":"grep -r 支持小组 topics/"}'})()})()
            return _Resp(_Msg("", [tc]))
        return _Resp(_Msg("相关记忆：[D1:3]"))  # 模型指认 D1:3
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    joined = " ".join(m.get("text","") for m in mems)
    assert "LGBTQ support group" in joined  # 回原文取到了 D1:3 的原始句子
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_retrieve.py -v`
Expected: FAIL（`_collect_v8` 不存在）

- [ ] **Step 3: 实现**

在 `src/adapters/run_nativemem.py` 加 prompt 与函数：
```python
_V8_RETRIEVE_PROMPT = """你在一个记忆库里找答案。工作目录是记忆库根目录，只能用 bash。
记忆库有两个视图：
- topics/ ：按话题分文件（问"是什么/谁/哪里"查这里）
- timeline/年/月-英文.md ：按时间排的事件（问"什么时候/先后/多久"查这里）
每条事件是一行：[日期] 摘要 · [dia_id]。dia_id 指向原始对话。

问题：{question}

步骤：先 grep/cat 相关视图定位事件行，找到后把相关行里的 [dia_id]（如 D1:3）列在你的最终回答里。
只列最相关的几条，别贪多。找齐 dia_id 后停手，直接输出，别反复翻。"""

_DIA_ID_RE = re.compile(r"D\d+:\d+")


def _collect_v8(question, memory_dir, turn_index, max_rounds=6):
    prompt = _V8_RETRIEVE_PROMPT.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    final_text, steps = "", 0
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL, messages=messages, tools=TOOLS,
            max_tokens=1200, temperature=0.0)
        log_usage(resp, phase="v8_retrieve")
        steps += 1
        msg = resp.choices[0].message
        final_text = msg.content or final_text
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            out = execute_tool(tc.function.name, args, memory_dir, hide_raw=False)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    # 从模型最终回答里抽 dia_id → 回原文
    dia_ids = _DIA_ID_RE.findall(final_text or "")
    if not dia_ids:
        return [], steps
    original = read_turns(turn_index, dia_ids, context=1)
    memories = [{"text": original, "date": ""}] if original.strip() else []
    return memories, steps
```

`collect_memories` 顶部加 v8 分支（turn_index 通过模块级变量或参数传入；这里用模块级 `_V8_TURN_INDEX`，由 main 在 build 后设置）：
```python
def collect_memories(question, memory_dir, max_rounds=10):
    if os.environ.get("NATIVEMEM_PROMPT") == "v8":
        return _collect_v8(question, memory_dir, _V8_TURN_INDEX, max_rounds=6)
    ...  # 原有分支不变
```

在模块顶部（`DATA_PATH = ...` 附近）加：
```python
_V8_TURN_INDEX = {}  # v8 检索回原文用；main 在 build 后设置
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_retrieve.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add src/adapters/run_nativemem.py tests/test_v8_retrieve.py
git commit -m "feat(v8): _collect_v8 查视图+回原文检索"
```

---

### Task 6: main 接线 + 端到端冒烟

**Files:**
- Modify: `src/adapters/run_nativemem.py`（main 里 build 后设置 `_V8_TURN_INDEX`）
- Create: `scripts/smoke_v8.sh`
- Test: 端到端脚本（真实模型，小样本，人工核对）

**Interfaces:**
- Consumes: 全部 v8 路径
- Produces: main 在 v8 下 build 完设置 turn_index 供检索回原文；一个可手跑冒烟脚本。

- [ ] **Step 1: main 里接线**

在 `main()` build 之后、检索循环之前加：
```python
    if os.environ.get("NATIVEMEM_PROMPT") == "v8":
        global _V8_TURN_INDEX
        _V8_TURN_INDEX = build_turn_index(conv)
```
（`conv` 在 main 里已从 sample 取到；`global` 声明放函数内首次赋值前。）

- [ ] **Step 2: 冒烟脚本**

```bash
# scripts/smoke_v8.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export NATIVEMEM_PROMPT=v8
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
OUT="results/smoke-v8"
rm -rf "$OUT"
python src/adapters/run_nativemem.py --sample 0 --max-sessions 3 --questions-limit 3 \
  --output "$OUT/result.json"
echo "=== 记忆库结构 ==="
find "$OUT/memory_sample0" -type f | sort
echo "=== timeline 抽查 ==="
find "$OUT/memory_sample0/timeline" -name '*.md' | head -1 | xargs cat
echo "=== topics 抽查 ==="
find "$OUT/memory_sample0/topics" -name '*.md' | head -1 | xargs cat
echo "=== 检索结果(应含回原文的原始句子) ==="
cat "$OUT/result.json"
```

- [ ] **Step 3: 跑冒烟（真实模型 + 网络）**

Run: `chmod +x scripts/smoke_v8.sh && bash scripts/smoke_v8.sh`
Expected（人工核对）：
- `memory_sample0/` 下有 `timeline/2023/*-*.md` 和 `topics/*.md` 双视图
- 每条是 `[日期] 摘要 · [dia_id]` 一行，**不含逐字原文**
- `result.json` 的 memories 里是**回原文取到的原始对话句子**（证明 dia_id → 原文接通）
- 3 个问题都有检索结果，无大面积空

- [ ] **Step 4: 提交**

```bash
git add src/adapters/run_nativemem.py scripts/smoke_v8.sh
git commit -m "feat(v8): main 接线 turn_index + 端到端冒烟脚本"
```

---

## Self-Review

**Spec coverage：**
- §3 双视图结构（timeline/年/月-英文 + topics）→ Task 2 write_events ✓
- §4 存记忆=提炼一次+代码落盘 → Task 3 distill_events + Task 4 build 分支 ✓
- §5 检索三步（定位→回原文→作答）→ Task 5 _collect_v8 ✓（作答由统一 answerer 做，本 adapter 只收集，符合 collect 契约）
- §6 回原文实现（build_turn_index/read_turns）→ Task 1 + Task 6 接线 ✓
- §7 边界（空事件跳过/话题复用/无效dia_id/时间排序）→ Task 3 空返回、known_topics 复用、Task 1 跳无效、Task 2 _append_sorted ✓
- §8 复用与新增 → split_into_chunks(50)/normalize_date 复用，v8_memory 新增，v7 不动 ✓
- Global：v8 独立分支，v4/v6/v7 不受影响（全部 `if NATIVEMEM_PROMPT=="v8"` 门控）✓

**Placeholder scan：** 无 TBD/TODO，每个 code step 给完整代码。

**Type consistency：** `distill_events(...) -> list[dict{when,summary,dia_ids,topic}]`（Task 3 定义）被 write_events（Task 2 消费同结构）和 build 分支（Task 4）一致使用；`build_turn_index -> dict`、`read_turns(idx, dia_ids, context)`（Task 1）被 _collect_v8（Task 5）一致调用；`write_events(memory_dir, events)` 签名 Task 2 定义、Task 4 调用一致。

**未纳入（YAGNI）：** 答题/judge 属统一 evaluate.py，不在本 adapter；consolidate_topics 合并话题留到全量跑发现话题碎再接（Task 范围外）；timeline 按年/半年再切留到单月文件真的过大再说。
