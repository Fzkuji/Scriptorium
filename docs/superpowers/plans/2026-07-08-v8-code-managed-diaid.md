# v8 优化：dia_id 全程代码托管 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** dia_id 从模型手记/手抄改为代码托管：提炼时喂带行号的 chunk、模型只写行号(refs 必填)、代码转真实 dia_id；检索时给 read_original 工具(dia_id 必填参数)让模型点事件、代码回原文。

**Architecture:** 只改 v8 两处——`src/v8_memory.py` 的提炼(带行号 + refs→dia_id)、`src/adapters/run_nativemem.py` 的检索(read_original 工具)。v4/v6/v7 不动。

**Tech Stack:** Python 3，OpenAI SDK(qwen3.6-flash @ 阿里云)，pytest，fake client 单测。

## Global Constraints

- 只改 v8 分支/函数，v4/v6/v7 行为不变。
- 行号是 chunk 内 1-based 局部号；第 i 句 ↔ 传入 dia_ids[i-1]（严格按 split 顺序 1:1）。
- 提炼：模型输出 `refs`（行号列表，至少一个）；代码转 dia_id；refs 空的事件丢弃。
- 检索：read_original 的 dia_ids 是必填参数；代码用 read_turns 回原文；不再靠正则从最终文本抠 id。
- 测试用 monkeypatch fake client，不联网。run_nativemem.py 调 v8_memory 函数用模块限定 `v8_memory.xxx`（否则 monkeypatch 失效——Task 4 已验证的教训）。

---

### Task 1: 行号 chunk + refs→dia_id 转换（提炼侧）

**Files:**
- Modify: `src/v8_memory.py`（改 `_V8_DISTILL_PROMPT`、`distill_events`，新增 `_number_chunk`、`_refs_to_dia_ids`）
- Test: `tests/test_v8_numbered.py`

**Interfaces:**
- Produces:
  - `_number_chunk(session_text, dia_ids) -> (numbered_text, line_map)`：把 `speaker: text` 每行前加 `[n] `（n 从 1），返回带号文本 + `{n: dia_id}` 映射（n 对应 dia_ids[n-1]）。
  - `_refs_to_dia_ids(refs, line_map) -> list[str]`：把模型给的行号列表转成 dia_id，去重、丢无效号。
  - `distill_events(session_text, obs_date, dia_ids, known_topics=None, max_retry=6)`：签名不变，但内部改为喂带号文本、要模型写 `refs`、代码转 dia_id。事件 refs 转出为空 → 丢弃该事件。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_numbered.py
import json
import src.v8_memory as V8

def test_number_chunk_adds_line_numbers_and_maps():
    text = "Caroline: hi\n\nMelanie: hello\n\nCaroline: went to group\n\n"
    numbered, line_map = V8._number_chunk(text, ["D1:1", "D1:2", "D1:3"])
    assert "[1] Caroline: hi" in numbered
    assert "[3] Caroline: went to group" in numbered
    assert line_map == {1: "D1:1", 2: "D1:2", 3: "D1:3"}

def test_refs_to_dia_ids_maps_and_drops_invalid():
    lm = {1: "D1:1", 2: "D1:2", 3: "D1:3"}
    assert V8._refs_to_dia_ids([1, 3], lm) == ["D1:1", "D1:3"]
    assert V8._refs_to_dia_ids([3, 3, 2], lm) == ["D1:3", "D1:2"]  # 去重、保序
    assert V8._refs_to_dia_ids([99], lm) == []                     # 越界号丢弃
    assert V8._refs_to_dia_ids(["2"], lm) == ["D1:2"]              # 字符串号也接受

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 10})()

def _install_fake(monkeypatch, content):
    monkeypatch.setattr(V8.client.chat.completions, "create", lambda **k: _FakeResp(content))
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)

def test_distill_uses_refs_and_code_fills_dia_ids(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "去支持小组", "refs": [3], "topic": "support"}]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events("Caroline: hi\n\nMelanie: hello\n\nCaroline: went to group\n\n",
                            "2023-05-07", ["D1:1", "D1:2", "D1:3"])
    assert len(evs) == 1
    assert evs[0]["dia_ids"] == ["D1:3"]   # 代码从 refs=[3] 转出

def test_distill_drops_event_with_empty_refs(monkeypatch):
    payload = json.dumps({"events": [
        {"when": "2023-05-07", "summary": "无引用事件", "refs": [], "topic": "t"}]})
    _install_fake(monkeypatch, payload)
    evs = V8.distill_events("Caroline: hi\n\n", "2023-05-07", ["D1:1"])
    assert evs == []   # refs 空 → 丢弃
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_numbered.py -v`
Expected: FAIL（`_number_chunk` 不存在 / distill 还在用 dia_ids）

- [ ] **Step 3: 实现**

在 `src/v8_memory.py` 加两个 helper（放在 distill_events 上方）：
```python
def _number_chunk(session_text, dia_ids):
    """给 chunk 每行加 [n] 行号，返回 (带号文本, {n: dia_id})。
    n 从 1，第 n 行对应 dia_ids[n-1]。空行跳过、不占号。"""
    lines = [ln for ln in session_text.split("\n") if ln.strip()]
    numbered, line_map = [], {}
    for i, ln in enumerate(lines, 1):
        numbered.append(f"[{i}] {ln}")
        if i - 1 < len(dia_ids):
            line_map[i] = dia_ids[i - 1]
    return "\n".join(numbered), line_map


def _refs_to_dia_ids(refs, line_map):
    """把模型给的行号列表（int 或 str）转成 dia_id，去重保序，丢无效号。"""
    if not isinstance(refs, list):
        refs = [refs]
    out, seen = [], set()
    for r in refs:
        try:
            n = int(str(r).strip())
        except (ValueError, TypeError):
            continue
        did = line_map.get(n)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out
```

改 `_V8_DISTILL_PROMPT`（把 dia_ids 那条换成 refs）：
```python
_V8_DISTILL_PROMPT = """你从一段对话里挑出值得长期记住的【事件】，输出 JSON，不要复制原文。
观测日期：{obs_date}
已有话题名（尽量复用，别造近义新名）：{known_topics}

对话每行开头有 [数字] 行号。每个事件给：
- when: 事件日期 YYYY-MM-DD（能推断就推断，推断不出用观测日期）
- summary: 一句话摘要（是什么事，够检索定位即可，别逐字抄）
- refs: 这条事件对应的行号列表（就是行首 [数字] 的那个数字），**至少写一个**
- topic: 归一个话题名（同一主体+主题，如 Caroline-adoption）

只输出 JSON：{{"events":[{{"when":"2023-05-07","summary":"...","refs":[3,5],"topic":"..."}}]}}
没有值得记的就 {{"events":[]}}。"""
```

改 `distill_events`：喂带号文本 + refs 转 dia_id。把函数体里
```python
    prompt = _V8_DISTILL_PROMPT.format(obs_date=obs_date, known_topics=kt)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": session_text}]
```
改成
```python
    prompt = _V8_DISTILL_PROMPT.format(obs_date=obs_date, known_topics=kt)
    numbered_text, line_map = _number_chunk(session_text, dia_ids)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": numbered_text}]
```
把事件处理循环里的
```python
        e.setdefault("dia_ids", [])
        e["dia_ids"] = _normalize_event_dia_ids(e.get("dia_ids", []), dia_ids)
        e.setdefault("topic", "misc")
        out.append(e)
```
改成
```python
        e["dia_ids"] = _refs_to_dia_ids(e.get("refs", []), line_map)
        if not e["dia_ids"]:
            continue                      # refs 转不出任何 dia_id → 丢弃
        e.setdefault("topic", "misc")
        out.append(e)
```
（`_normalize_event_dia_ids` 保留不删，v8 提炼不再调用它——留作兼容/回归参考。）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_numbered.py -v`
Expected: PASS（5 passed）。再跑 `python -m pytest tests/ -q` 确认无回归（旧的 test_v8_distill 里断言 dia_ids 输入的用例可能因改 refs 而需调整——若旧用例失败，说明它测的是旧行为，按新 refs 契约更新该用例的输入 payload，保持断言意图不变）。

- [ ] **Step 5: 提交**

```bash
git add src/v8_memory.py tests/test_v8_numbered.py tests/test_v8_distill.py
git commit -m "feat(v8-opt): 提炼喂带行号chunk，模型写refs行号，代码转dia_id"
```

---

### Task 2: read_original 工具（检索侧，必填 dia_id）

**Files:**
- Modify: `src/adapters/run_nativemem.py`（改 `_collect_v8`、`_V8_RETRIEVE_PROMPT`，新增 read_original 工具定义与分发）
- Test: `tests/test_v8_readtool.py`

**Interfaces:**
- Consumes: `read_turns`（Task1 之前已有）、`execute_tool`（bash grep/cat）
- Produces: `_collect_v8` 除 bash 工具外，多一个 `read_original` 工具（参数 `dia_ids: list[str]` 必填）；模型调它 → 代码用 `read_turns(turn_index, dia_ids)` 回原文并作为 tool 结果返回；`_collect_v8` 收集所有 read_original 取到的原文作为 memories。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_readtool.py
import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8

def test_collect_v8_uses_read_original_tool(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when": "2023-05-07", "summary": "去支持小组",
                         "dia_ids": ["D1:3"], "topic": "support"}])
    conv = {"session_1": [
        {"speaker": "Caroline", "dia_id": "D1:3", "text": "I went to a LGBTQ support group"}]}
    idx = V8.build_turn_index(conv)

    calls = {"n": 0}
    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content; self.tool_calls = tool_calls
    class _Resp:
        def __init__(self, msg):
            self.choices = [type("C", (), {"message": msg})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            # 模型调 read_original，必填 dia_ids
            tc = type("T", (), {"id": "1", "function": type("F", (), {
                "name": "read_original",
                "arguments": '{"dia_ids": ["D1:3"]}'})()})()
            return _Resp(_Msg("", [tc]))
        return _Resp(_Msg("done"))  # 第二轮无 tool_call 收尾
    monkeypatch.setattr(R.client.chat.completions, "create", fake_create)
    monkeypatch.setattr(R, "log_usage", lambda *a, **k: None)

    mems, steps = R._collect_v8("去过支持小组吗", d, idx)
    joined = " ".join(m.get("text", "") for m in mems)
    assert "LGBTQ support group" in joined   # read_original 工具回原文取到 D1:3
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_v8_readtool.py -v`
Expected: FAIL（read_original 未定义/未分发，memories 空）

- [ ] **Step 3: 实现**

在 `src/adapters/run_nativemem.py` 加 read_original 工具定义（与现有 TOOLS 并列，供 v8 检索用）：
```python
_V8_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_original",
        "description": "读事件对应的原始对话。dia_ids 从事件行末尾的 [D1:3, D1:5] 里取。",
        "parameters": {
            "type": "object",
            "properties": {
                "dia_ids": {"type": "array", "items": {"type": "string"},
                            "description": "要回原文的 dia_id 列表，如 [\"D1:3\"]"}
            },
            "required": ["dia_ids"],
        },
    },
}
```

改 `_V8_RETRIEVE_PROMPT`（告诉模型用 read_original）：
```python
_V8_RETRIEVE_PROMPT = """你在一个记忆库里找答案。工作目录是记忆库根目录，只能用 bash。
记忆库有两个视图：
- topics/ ：按话题分文件（问"是什么/谁/哪里"查这里）
- timeline/年/月-英文.md ：按时间排的事件（问"什么时候/先后/多久"查这里）
每条事件是一行：[日期] 摘要 · [dia_id]。dia_id 指向原始对话。

问题：{question}

步骤：
1. 先用 bash grep/cat 相关视图，定位相关的事件行。
2. 找到后，从行末的 [D1:3, D1:5] 取出 dia_id，调 read_original(dia_ids=[...]) 读原始对话。
3. 只取最相关的几条，别贪多。读到原文后停手。"""
```

改 `_collect_v8`：工具列表加 read_original，分发时 read_original 走 read_turns 收集原文：
```python
def _collect_v8(question, memory_dir, turn_index, max_rounds=6):
    prompt = _V8_RETRIEVE_PROMPT.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    tools = TOOLS + [_V8_READ_TOOL]
    collected, steps = [], 0
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL, messages=messages, tools=tools,
            max_tokens=1200, temperature=0.0)
        log_usage(resp, phase="v8_retrieve")
        steps += 1
        msg = resp.choices[0].message
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
            if tc.function.name == "read_original":
                dids = args.get("dia_ids", [])
                if isinstance(dids, str):
                    dids = [dids]
                original = read_turns(turn_index, dids, context=1)
                if original.strip():
                    collected.append(original)
                out = original or "(无对应原文)"
            else:
                out = execute_tool(tc.function.name, args, memory_dir, hide_raw=False)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    memories = [{"text": t, "date": ""} for t in collected]
    return memories, steps
```
（`_DIA_ID_RE` 不再用于收集，可保留；旧的"从 final_text 抠 id"逻辑被 read_original 收集取代。）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_v8_readtool.py -v`
Expected: PASS（1 passed）。再 `python -m pytest tests/ -q`（旧 test_v8_retrieve 可能因改为工具收集而需更新其 fake：把第二轮 `"相关记忆：[D1:3]"` 改为一轮 read_original tool_call——若失败，更新该用例的 fake 使其调 read_original，保持"原文出现在 memories"的断言意图）。

- [ ] **Step 5: 提交**

```bash
git add src/adapters/run_nativemem.py tests/test_v8_readtool.py tests/test_v8_retrieve.py
git commit -m "feat(v8-opt): 检索加 read_original 工具(dia_id必填)，代码回原文收集"
```

---

### Task 3: 冒烟验证（真实模型）

**Files:**
- 无新增（复用 scripts/smoke_v8.sh）

**Interfaces:**
- Consumes: 全部 v8-opt 路径

- [ ] **Step 1: 跑冒烟**

Run: `bash scripts/smoke_v8.sh`
Expected（人工核对）：
- timeline/topics 事件行 dia_id **空 [] 比例大幅下降**（对比优化前 6/9 空 → 目标 <2/9）
- 检索非空题数从 1/3 提升
- q2 那类回原文仍通

- [ ] **Step 2: 核对并记录**

检查 `results/smoke-v8/memory_sample0/` 下 dia_id 空比例、result.json 非空题数，与优化前对比，记录到冒烟报告。

- [ ] **Step 3:（无代码改动则不提交；若冒烟暴露新 bug，报告 controller 定夺）**

---

## Self-Review

**Spec coverage：**
- 改动1 提炼带行号+refs必填+代码转 → Task 1（_number_chunk/_refs_to_dia_ids/prompt/distill 改）✓
- 改动2 检索 read_original 工具必填 id → Task 2 ✓
- 边界：refs 空丢弃（Task1 test）、read_original 无效 id 跳过（read_turns 已有）、行号 1:1 映射（Task1 _number_chunk）✓
- 验收：冒烟 dia_id 空比例↓、非空题数↑ → Task 3 ✓
- Global：只改 v8，v4/v6/v7 不动；模块限定调用 ✓

**Placeholder scan：** 无 TBD/TODO，每 code step 给完整代码。

**Type consistency：** `_number_chunk -> (str, dict)`、`_refs_to_dia_ids(refs, line_map) -> list[str]`（Task1）；`_collect_v8(question, memory_dir, turn_index, max_rounds)` 签名不变（Task2）；read_original 工具参数 `dia_ids: list[str]` 与分发一致。

**注意旧测试：** Task1/2 明确提示旧 test_v8_distill、test_v8_retrieve 的 fake 可能因契约变化需更新——按新 refs/工具契约调整输入，保持断言意图，不是删测试。
