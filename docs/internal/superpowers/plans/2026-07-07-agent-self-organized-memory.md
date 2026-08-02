# Agent 自组织记忆管理（v7）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 v7 写入路径——每个 chunk 交给 agent 看顶层结构、自己用 bash 决定放哪，取代 v6 硬编码 `people/{person}.md` 的 `store_facts_code`；配套三节奏结构维护（即时/每 session 轻量/增长触发全局重构）、dia_id 来源锚、两口径检索。

**Architecture:** 复用现有分层提炼（`distill_chunk`/`verify_details`）产出结构化 facts；新增 `process_chunk_agent` 让 agent 拿顶层视图 + facts + dia_id 自组织存储；结构质量不靠写入保证，靠 `tidy_local`（每 session，代码测量 `inspect_structure` 标 ⚠ → 模型局部修）和 `reorganize_library`（库规模 `measure_library` 增长过阈值 → 模型全局重排）两节奏兜底。来源用 LoCoMo 固有的 `dia_id`（`D1:3`），不用行号、不复制原文。

**Tech Stack:** Python 3，OpenAI SDK（阿里云端点 qwen3.6-flash），bash 工具（`subprocess`），pytest（新建）。设计文档：`docs/superpowers/specs/2026-07-06-agent-self-organized-memory-design.md`。

## Global Constraints

- 落地代码只碰 `src/nativemem.py` 和 `src/adapters/run_nativemem.py`；测试放新建的 `tests/`。
- OpenAI client 复用 `nativemem.py` 现有全局 `client`（`ALIYUN_KEY`/`BUILDER_BASE`/`trust_env=False`/`timeout=180`），不新建 client。
- 模型名走现有解析：`os.environ["MODEL"] or os.environ.get("BUILDER_MODEL","deepseek-v4-flash")`，实验用 `BUILDER_MODEL=qwen3.6-flash`。
- 来源锚 = `dia_id`（LoCoMo turn 固有字段，如 `D1:3`）。`[source](D1:3)` 或多个 `[source](D1:3, D1:5)`。不用行号、不复制原文、不建 `raw/`。
- 一条记忆永远是三行块：`[YYYY-MM-DD] 骨架句` / `  > "原文引用"` / `  [source](dia_id...)`。
- 代码红线：代码给顶层视图 / 工具 / 测量报告失衡 / 保内容完整；代码不写死记忆内部路径、不决定按什么分、不接管"放哪"。阈值只标 ⚠ 提示模型，不强制动作。
- 每条记忆的 `[YYYY-MM-DD]` 由 `normalize_date` 保证格式统一。
- v7 由 `NATIVEMEM_PROMPT=v7` 启用；`v4/v5/v6` 分支保留不动。
- 新增环境变量默认值：`NATIVEMEM_STORE_MODE=oneshot`、`NATIVEMEM_REBALANCE=on`、`NATIVEMEM_REORG_GROWTH=2.0`、`NATIVEMEM_REORG_FILE_DELTA=8`、`NATIVEMEM_BIG_FILE=150`、`NATIVEMEM_SMALL_FILE=8`、`NATIVEMEM_WIDE_DIR=20`、`NATIVEMEM_NO_ORIGINAL`（未设=用原文，`1`=不用原文）。

---

## 文件结构

- **`src/nativemem.py`**（修改）：新增 v7 的纯函数与 agent 函数——`split_into_chunks`、`_top_level_view`、`_AGENT_STORE_PROMPT`、`STRUCTURE_STANDARD`、`process_chunk_agent`、`inspect_structure`、`tidy_local`、`reorganize_library`、`measure_library`、`library_grew_past_threshold`、`_REBALANCE_PROMPT`。复用 `client`/`TOOLS`/`execute_tool`/`distill_chunk`/`verify_details`/`consolidate_topics`/`_find_merge_candidates`/`normalize_date`/`log_usage`。`store_facts_code` 保留但 v7 不走。
- **`src/adapters/run_nativemem.py`**（修改）：`build_memory` 加 v7 分支（走 `process_chunk_agent` + 三节奏）；`collect_memories` 加 v7 分支（看顶层自己翻 + 按 `NATIVEMEM_NO_ORIGINAL` 过滤 `[source]` 行）。旧 grep/nav 分支保留。
- **`tests/`**（新建）：`tests/test_v7_chunks.py`、`tests/test_v7_structure.py`、`tests/test_v7_store.py`、`tests/conftest.py`。纯函数直接单测；agent 函数用 monkeypatch 替换 `client` 或 `execute_tool`。

每个任务结束都能独立测。TDD：先写失败测试，再实现，再验证，再提交。

---

### Task 1: `split_into_chunks` —— 切 chunk 带回 dia_id 列表

**Files:**
- Modify: `src/nativemem.py`（新增函数，加在 `distill_chunk` 之后，约 NM:418 后）
- Create: `tests/conftest.py`
- Create: `tests/test_v7_chunks.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces: `split_into_chunks(session: list[dict], size: int = 10) -> list[tuple[str, list[str]]]`
  返回 `[(chunk_text, dia_ids), ...]`。`chunk_text` = 每 turn `"{speaker}: {text}\n\n"` 拼接（与 RN:279-281 现有格式一致）；`dia_ids` = 该 chunk 覆盖的 `turn["dia_id"]` 列表。跳过非 dict turn 和空 text。

- [ ] **Step 1: 建 `tests/conftest.py`（提供一个真实 session fixture）**

```python
# tests/conftest.py
import json, os, sys
import pytest

# 让 tests 能 import src.nativemem
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

_DATA = os.path.join(_ROOT, "benchmarks", "locomo", "data", "locomo10.json")

@pytest.fixture
def locomo_session():
    """LoCoMo sample0 的 session_1（18 个 turn，均为 dict，有 speaker/dia_id/text）。"""
    with open(_DATA) as f:
        data = json.load(f)
    return data[0]["conversation"]["session_1"]
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_v7_chunks.py
import nativemem

def test_split_returns_text_and_dia_ids(locomo_session):
    chunks = nativemem.split_into_chunks(locomo_session, size=10)
    # 18 turns / 10 → 2 chunks
    assert len(chunks) == 2
    text0, dia0 = chunks[0]
    # 第一个 chunk 覆盖前 10 个 turn 的 dia_id
    assert dia0 == [t["dia_id"] for t in locomo_session[:10]]
    # chunk_text 含 speaker: text 格式
    assert locomo_session[0]["speaker"] + ":" in text0
    assert locomo_session[0]["text"] in text0

def test_split_skips_empty_and_nondict():
    session = [
        {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
        "not a dict",
        {"speaker": "B", "dia_id": "D1:2", "text": ""},
        {"speaker": "A", "dia_id": "D1:3", "text": "bye"},
    ]
    chunks = nativemem.split_into_chunks(session, size=10)
    assert len(chunks) == 1
    text, dia = chunks[0]
    # 只保留有 text 的 dict turn
    assert dia == ["D1:1", "D1:3"]
    assert "hi" in text and "bye" in text
```

- [ ] **Step 3: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_chunks.py -v`
Expected: FAIL —— `AttributeError: module 'nativemem' has no attribute 'split_into_chunks'`

- [ ] **Step 4: 实现 `split_into_chunks`**

加到 `src/nativemem.py`（`distill_chunk` 定义之后）：

```python
def split_into_chunks(session, size=10):
    """把一个 session 的 turn 列表按 size 切 chunk。
    返回 [(chunk_text, dia_ids), ...]：chunk_text 是 'speaker: text' 拼接，
    dia_ids 是该 chunk 覆盖的 turn['dia_id'] 列表。跳过非 dict / 空 text 的 turn。"""
    turns = [t for t in session
             if isinstance(t, dict) and str(t.get("text", "")).strip()]
    chunks = []
    for cs in range(0, len(turns), size):
        group = turns[cs:cs + size]
        text = "".join(f"{t.get('speaker', 'user')}: {t.get('text', '')}\n\n"
                       for t in group)
        dia_ids = [t["dia_id"] for t in group if "dia_id" in t]
        chunks.append((text, dia_ids))
    return chunks
```

- [ ] **Step 5: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_chunks.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add tests/conftest.py tests/test_v7_chunks.py src/nativemem.py
git commit -m "feat(v7): split_into_chunks returns chunk text + dia_id list"
```

---

### Task 2: `_top_level_view` —— 只列记忆根目录直接子项

**Files:**
- Modify: `src/nativemem.py`（新增函数）
- Create: `tests/test_v7_structure.py`

**Interfaces:**
- Consumes: 无
- Produces: `_top_level_view(memory_dir: str) -> str`
  只列根目录直接子项：文件列文件名；目录列 `name/  (N items)`；跳过隐藏文件和 `raw/`。空库返回 `"(empty memory)"`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v7_structure.py
import os
import nativemem

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
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py::test_top_level_view_lists_direct_children -v`
Expected: FAIL —— `AttributeError: ... '_top_level_view'`

- [ ] **Step 3: 实现 `_top_level_view`**

```python
def _top_level_view(memory_dir):
    """只列记忆根目录的直接子项（模型 drill-in 的起点，不展开文件内部）。
    文件列文件名；目录列 'name/  (N items)'；跳过隐藏文件和 raw/。"""
    try:
        names = sorted(os.listdir(memory_dir))
    except FileNotFoundError:
        return "(empty memory)"
    lines = []
    for name in names:
        if name.startswith(".") or name == "raw":
            continue
        full = os.path.join(memory_dir, name)
        if os.path.isdir(full):
            n = len([x for x in os.listdir(full) if not x.startswith(".")])
            lines.append(f"{name}/  ({n} items)")
        else:
            lines.append(name)
    return "\n".join(lines) if lines else "(empty memory)"
```

- [ ] **Step 4: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add tests/test_v7_structure.py src/nativemem.py
git commit -m "feat(v7): _top_level_view lists root-level children only"
```

---

### Task 3: `measure_library` + `library_grew_past_threshold` —— 库规模测量与增长触发

**Files:**
- Modify: `src/nativemem.py`（新增两函数）
- Modify: `tests/test_v7_structure.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `measure_library(memory_dir: str) -> dict` 返回 `{"files": int, "dirs": int, "bytes": int, "entries": int}`。`entries` = 全库 `[YYYY-MM-DD]` 开头行的总数。跳过 `raw/` 和隐藏。
  - `library_grew_past_threshold(old: dict, new: dict, growth_ratio: float = 2.0, file_delta: int = 8) -> bool`：`(new.files+new.dirs) - (old.files+old.dirs) >= file_delta` **或** `old.bytes>0 and new.bytes/old.bytes >= growth_ratio`。任一为真返回 True。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_v7_structure.py
import re

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
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py::test_measure_library_counts -v`
Expected: FAIL —— `AttributeError: ... 'measure_library'`

- [ ] **Step 3: 实现两函数**

```python
_ENTRY_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]")

def measure_library(memory_dir):
    """测量库规模：files/dirs/bytes/entries。跳过 raw/ 和隐藏。entries = 全库
    [YYYY-MM-DD] 开头的行数。是三节奏增长触发的基线。"""
    files = dirs = total_bytes = entries = 0
    for root, dirnames, filenames in os.walk(memory_dir):
        dirnames[:] = [d for d in dirnames if d != "raw" and not d.startswith(".")]
        for d in dirnames:
            dirs += 1
        for fn in filenames:
            if fn.startswith("."):
                continue
            files += 1
            path = os.path.join(root, fn)
            try:
                total_bytes += os.path.getsize(path)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if _ENTRY_RE.match(line):
                            entries += 1
            except OSError:
                pass
    return {"files": files, "dirs": dirs, "bytes": total_bytes, "entries": entries}

def library_grew_past_threshold(old, new, growth_ratio=2.0, file_delta=8):
    """自上次全局重构以来是否增长过阈值：文件+文件夹数增量 >= file_delta，
    或 字节比值 >= growth_ratio，任一为真。"""
    struct_delta = (new["files"] + new["dirs"]) - (old["files"] + old["dirs"])
    if struct_delta >= file_delta:
        return True
    if old["bytes"] > 0 and new["bytes"] / old["bytes"] >= growth_ratio:
        return True
    return False
```

> 注：`re` 已在 nativemem.py 顶部 import（`normalize_date` 用了）；`_ENTRY_RE` 加在模块级即可。

- [ ] **Step 4: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py -v`
Expected: PASS（全部 passed）

- [ ] **Step 5: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add tests/test_v7_structure.py src/nativemem.py
git commit -m "feat(v7): measure_library + library_grew_past_threshold (growth trigger)"
```

---

### Task 4: `inspect_structure` —— 结构体检，标 ⚠ 失衡项

**Files:**
- Modify: `src/nativemem.py`（新增函数）
- Modify: `tests/test_v7_structure.py`

**Interfaces:**
- Consumes: `_find_merge_candidates`（NM:650，复用找近义标题；这里也用来找近义文件名）
- Produces: `inspect_structure(memory_dir: str, big=150, small=8, wide=20) -> dict`
  返回 `{"report": str, "warnings": list[dict]}`。`warnings` 每项 `{"kind": "big"|"small"|"wide"|"dup", "target": str, "detail": str}`。`report` 是给模型看的纯文本体检报告（含全库汇总行）。无失衡时 `warnings=[]`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_v7_structure.py
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
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py -k inspect -v`
Expected: FAIL —— `AttributeError: ... 'inspect_structure'`

- [ ] **Step 3: 实现 `inspect_structure`**

```python
def _count_entries_in_file(path):
    n = 0
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if _ENTRY_RE.match(line):
                    n += 1
    except OSError:
        pass
    return n

def inspect_structure(memory_dir, big=150, small=8, wide=20):
    """结构体检：纯测量，把失衡维度用阈值标 ⚠。返回 {report, warnings}。
    维度：单文件过大(>big)/过小(<small)、目录过宽(>wide)、近义重名文件。
    只测量不动手——修由 tidy_local/reorganize_library 交给模型。"""
    warnings, lines = [], []
    all_md = []  # (relpath, entries)
    for root, dirnames, filenames in os.walk(memory_dir):
        dirnames[:] = [d for d in dirnames if d != "raw" and not d.startswith(".")]
        # 目录过宽
        rel_root = os.path.relpath(root, memory_dir)
        visible = [x for x in filenames if not x.startswith(".")] + dirnames
        if len(visible) > wide:
            tgt = rel_root if rel_root != "." else "(root)"
            warnings.append({"kind": "wide", "target": tgt + "/",
                             "detail": f"{len(visible)} 个直接子项 (>{wide})"})
            lines.append(f"- {tgt}/: {len(visible)} 个直接子项     ⚠ 目录过宽 (>{wide})")
        for fn in filenames:
            if fn.startswith(".") or not fn.endswith(".md"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, memory_dir)
            n = _count_entries_in_file(path)
            all_md.append((rel, fn, n))
            if n > big:
                warnings.append({"kind": "big", "target": rel, "detail": f"{n} 条 (>{big})"})
                lines.append(f"- {rel}: {n} 条     ⚠ 过大 (>{big})")
            elif n < small:
                warnings.append({"kind": "small", "target": rel, "detail": f"{n} 条 (<{small})"})
                lines.append(f"- {rel}: {n} 条     ⚠ 过小 (<{small})")
    # 近义重名文件（复用 _find_merge_candidates，把文件名当 heading 喂进去）
    basenames = [fn for _, fn, _ in all_md]
    for group in _find_merge_candidates(basenames):
        if len(group) > 1:
            warnings.append({"kind": "dup", "target": ", ".join(group),
                             "detail": "疑似近义重名"})
            lines.append(f"- 近义文件名: [{', '.join(group)}]     ⚠ 疑重复")
    total = measure_library(memory_dir)
    summary = (f"全库: {total['files']} 文件 / {total['dirs']} 目录 / "
               f"{total['entries']} 条")
    report = "memory/ 结构报告:\n" + ("\n".join(lines) if lines else "(无失衡)") + "\n" + summary
    return {"report": report, "warnings": warnings}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py -k inspect -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add tests/test_v7_structure.py src/nativemem.py
git commit -m "feat(v7): inspect_structure flags oversized/tiny/wide/dup with warnings"
```

---

### Task 5: `STRUCTURE_STANDARD` + `_AGENT_STORE_PROMPT` —— 写入 prompt 常量

**Files:**
- Modify: `src/nativemem.py`（新增两常量）

**Interfaces:**
- Consumes: 无
- Produces:
  - `STRUCTURE_STANDARD: str`（好结构定义，写入软引导 + 整理目标共用）
  - `_AGENT_STORE_PROMPT: str`（含 `{top_view}`/`{facts}`/`{dia_ids}` 占位，指导 agent 用 bash 自组织存储 + 即时维护三规则 + 输出 `<summary>`）

- [ ] **Step 1: 加常量到 `src/nativemem.py`**（加在 `process_chunk` 之前）

```python
STRUCTURE_STANDARD = """\
好结构的标准（尽量遵守，做不到没关系，整理环节会兜底）：
- 文件别太大：单文件条目过多会让检索捞不准。
- 文件别太碎：一堆只有几条的小文件会让检索跳很多次。
- 目录别太宽：一个文件夹平铺几十个文件难定位。
- 同主题别分散：一个话题散在多个文件会漏检。
- 命名统一：同一实体/主题别造近义重名（LGBTQ_Support_Group vs lgbtq-support-group）。
- 默认浅、按需深：小规模一主体一文件即可；大到超标才拆子目录、留摘要页+链接。
"""

_AGENT_STORE_PROMPT = """\
你在维护一个用文件系统组织的记忆库。工作目录就是记忆库根目录。
你只能用 bash（ls / cat / grep / sed / mkdir / tee / cat >> 等）读写文件。

## 当前记忆库顶层结构
{top_view}

## 这批要存的事实（已提炼好，每条含 skeleton 骨架句 / verbatim 原文 / 相关 dia_id）
{facts}

## 本批对话覆盖的 turn id（写 source 时从中选与该条事实相关的）
{dia_ids}

## 每条记忆固定写成这三行块
[YYYY-MM-DD] 骨架句
  > "原文引用"
  [source](dia_id)          # 来自多句就 [source](D1:3, D1:5)

## 即时维护三规则（局部就能判断，务必遵守）
1. 放最相关的已有文件：先 ls/cat 看顶层和相关文件，把新事实放进语义最近的已有文件/## 小标题下；确实没有对应主体才新建文件。别新开文件堆一起。
2. 见重复就地合并/更新：同一事实的新版本（时间更新/细节补充/状态改变）就地改已有条目或紧挨着补，别无脑追加造重复。
3. 命名沿用惯例：新文件/小标题命名看已有的怎么起就怎么来，别造近义重名。

{structure_standard}

存完后，用一句话总结这批对话讲了什么，放在 <summary></summary> 里（给下一批当上下文）。
"""
```

- [ ] **Step 2: 冒烟验证常量可 import 且占位符齐全**

Run:
```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
python -c "import sys; sys.path.insert(0,'src'); import nativemem as n; \
p=n._AGENT_STORE_PROMPT; assert all(k in p for k in ['{top_view}','{facts}','{dia_ids}','{structure_standard}']); \
assert '三行块' in p and n.STRUCTURE_STANDARD; print('OK')"
```
Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add src/nativemem.py
git commit -m "feat(v7): STRUCTURE_STANDARD + _AGENT_STORE_PROMPT constants"
```

---

### Task 6: `process_chunk_agent` —— agent 自组织写入（oneshot / threecall）

**Files:**
- Modify: `src/nativemem.py`（新增函数）
- Create: `tests/test_v7_store.py`

**Interfaces:**
- Consumes: `distill_chunk`（NM:352）、`_top_level_view`（Task 2）、`STRUCTURE_STANDARD`/`_AGENT_STORE_PROMPT`（Task 5）、`client`/`TOOLS`/`execute_tool`/`log_usage`（现有）
- Produces: `process_chunk_agent(chunk_text: str, chunk_date: str, memory_dir: str, dia_ids: list[str], running_summary: str = "", mode: str = "oneshot") -> str`
  提炼 → agent 用 bash 自组织存 → 返回 `<summary>` 内容（正则抽取，抽不到返回 `""`）。`mode`：`oneshot`（一次会话做完存+总结）/`threecall`（存、总结分两次调；提炼已由 `distill_chunk` 独立完成）。

- [ ] **Step 1: 写失败测试（monkeypatch 掉模型调用，验证编排逻辑）**

```python
# tests/test_v7_store.py
import re
import nativemem

def test_process_chunk_agent_stores_and_returns_summary(tmp_path, monkeypatch):
    # 1) mock distill_chunk：返回一条已提炼 fact
    def fake_distill(chunk_text, chunk_date, max_retry=6):
        return [{"person": "Caroline", "topic": "support group",
                 "skeleton": "Caroline went to an LGBTQ support group and found it powerful.",
                 "keep_verbatim": ["LGBTQ"],
                 "verbatim": "I went to a LGBTQ support group yesterday and it was so powerful.",
                 "distilled": "Caroline went to an LGBTQ support group and found it powerful."}]
    monkeypatch.setattr(nativemem, "distill_chunk", fake_distill)

    # 2) mock 模型 agent 会话：直接落一个文件 + 返回带 <summary> 的最终文本
    calls = {"n": 0}
    class FakeResp:
        def __init__(self, content, tool_calls=None):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": content, "tool_calls": tool_calls})()})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
    def fake_create(**kwargs):
        calls["n"] += 1
        # 第一次就直接给最终答复（无 tool_calls），模拟 agent 说“我存好了”
        target = tmp_path / "Caroline.md"
        target.write_text('## support group\n[2023-05-08] Caroline went to an LGBTQ support group and found it powerful.\n  > "I went to a LGBTQ support group yesterday and it was so powerful."\n  [source](D1:3)\n')
        return FakeResp("Stored. <summary>Caroline shared about an LGBTQ support group.</summary>")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", fake_create)

    summary = nativemem.process_chunk_agent(
        "Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\n\n",
        "2023-05-08", str(tmp_path), ["D1:3"], running_summary="", mode="oneshot")

    assert "LGBTQ support group" in summary
    assert (tmp_path / "Caroline.md").exists()
    assert "[source](D1:3)" in (tmp_path / "Caroline.md").read_text()

def test_process_chunk_agent_empty_facts_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(nativemem, "distill_chunk", lambda *a, **k: [])
    # facts 为空时不该调模型、直接返回 ""
    def boom(**kwargs):
        raise AssertionError("should not call model on empty facts")
    monkeypatch.setattr(nativemem.client.chat.completions, "create", boom)
    out = nativemem.process_chunk_agent("x", "2023-05-08", str(tmp_path), [], mode="oneshot")
    assert out == ""
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_store.py -v`
Expected: FAIL —— `AttributeError: ... 'process_chunk_agent'`

- [ ] **Step 3: 实现 `process_chunk_agent`**

```python
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)

def _format_facts_for_agent(facts):
    lines = []
    for i, f in enumerate(facts, 1):
        skel = f.get("distilled") or f.get("skeleton", "")
        lines.append(f"{i}. skeleton: {skel}\n   verbatim: {f.get('verbatim','')}")
    return "\n".join(lines)

def _run_store_agent(memory_dir, top_view, facts_text, dia_ids, max_rounds=15):
    """跑一轮 bash-agent 存储会话，返回最终文本。"""
    prompt = _AGENT_STORE_PROMPT.format(
        top_view=top_view, facts=facts_text,
        dia_ids=", ".join(dia_ids) if dia_ids else "(none)",
        structure_standard=STRUCTURE_STANDARD)
    messages = [{"role": "user", "content": prompt}]
    final_text = ""
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL, messages=messages, tools=TOOLS,
            max_tokens=2000, temperature=0.2)
        log_usage(resp, phase="v7_store")
        msg = resp.choices[0].message
        final_text = msg.content or final_text
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments)
            out = execute_tool(tc.function.name, args, memory_dir)  # 写入期不 hide_raw（本就无 raw）
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return final_text

def process_chunk_agent(chunk_text, chunk_date, memory_dir, dia_ids,
                        running_summary="", mode="oneshot"):
    """v7 写入主体：提炼 → agent 用 bash 自组织存 → 返回本 chunk 一句总结。
    mode: 'oneshot'（一次会话存+总结）/ 'threecall'（存、总结分调）。"""
    os.makedirs(memory_dir, exist_ok=True)
    facts = distill_chunk(chunk_text, chunk_date)
    if not facts:
        return ""
    top_view = _top_level_view(memory_dir)
    facts_text = _format_facts_for_agent(facts)
    final_text = _run_store_agent(memory_dir, top_view, facts_text, dia_ids)
    m = _SUMMARY_RE.search(final_text or "")
    if m:
        return m.group(1).strip()
    if mode == "threecall":
        # 存储没给 summary，再单独调一次要总结
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL, max_tokens=120, temperature=0.2,
            messages=[{"role": "user",
                       "content": f"用一句话总结这段对话：\n{chunk_text}"}])
        log_usage(resp, phase="v7_summary")
        return (resp.choices[0].message.content or "").strip()
    return ""
```

> 说明：oneshot 与 threecall 共用 `_run_store_agent`；差别仅在拿不到 `<summary>` 时 threecall 补一次总结调用，oneshot 直接返回 ""（总结靠 agent 在存储会话里给）。提炼两模式都由 `distill_chunk` 独立完成（符合 spec §2.4「提炼这一步独立」）。

- [ ] **Step 4: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_store.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add tests/test_v7_store.py src/nativemem.py
git commit -m "feat(v7): process_chunk_agent (oneshot/threecall) distill+agent-store"
```

---

### Task 7: `tidy_local` + `reorganize_library` + `_REBALANCE_PROMPT` —— 节奏2/节奏3

**Files:**
- Modify: `src/nativemem.py`（新增两函数 + 一常量）
- Modify: `tests/test_v7_structure.py`

**Interfaces:**
- Consumes: `inspect_structure`（Task 4）、`consolidate_topics`（NM:686）、`client`/`TOOLS`/`execute_tool`/`log_usage`（现有）
- Produces:
  - `tidy_local(memory_dir: str, big=150, small=8, wide=20) -> int`：节奏2。先 `inspect_structure`，**无 ⚠ 直接返回 0**；有则对每个 `.md` 跑 `consolidate_topics`（文件内并标题）+ 对 big/dup 项让模型局部修，返回处理的 ⚠ 数。
  - `reorganize_library(memory_dir: str, big=150, small=8, wide=20) -> int`：节奏3。给模型 `inspect_structure` 报告 + 顶层视图，模型用 bash 全局重排，返回 bash 轮数。
  - `_REBALANCE_PROMPT: str`（含 `{report}`/`{top_view}`/`{structure_standard}` 占位）

- [ ] **Step 1: 写失败测试（无 ⚠ 时 tidy_local 不调模型）**

```python
# 追加到 tests/test_v7_structure.py
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
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py -k "tidy_local or reorganize" -v`
Expected: FAIL —— `AttributeError: ... 'tidy_local'`

- [ ] **Step 3: 实现两函数 + 常量**

```python
_REBALANCE_PROMPT = """\
你在整理一个用文件系统组织的记忆库。工作目录就是记忆库根目录，只能用 bash 操作。

## 结构体检报告（代码测量，⚠ 是失衡提示，不是强制命令）
{report}

## 顶层结构
{top_view}

{structure_standard}

你有全局视野，把这个库重排成任何时刻都好检索的样子：拆过大文件、合分散主题、
按子主题给过宽目录建子目录、并孤儿小文件、统一命名、必要时留摘要页+链接。
拆合建删随你判断。
红线：别丢任何记忆内容，别改每条的 [source](dia_id) 锚。
"""

def _run_rebalance_agent(memory_dir, report, top_view, max_rounds=15):
    prompt = _REBALANCE_PROMPT.format(report=report, top_view=top_view,
                                      structure_standard=STRUCTURE_STANDARD)
    messages = [{"role": "user", "content": prompt}]
    rounds = 0
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL, messages=messages, tools=TOOLS,
            max_tokens=2000, temperature=0.2)
        log_usage(resp, phase="v7_reorg")
        rounds += 1
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments)
            out = execute_tool(tc.function.name, args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return rounds

def tidy_local(memory_dir, big=150, small=8, wide=20):
    """节奏2：每 session 末轻量再平衡。无 ⚠ 直接返回 0（省 API）。
    有则文件内并标题（consolidate_topics）+ 对 big/dup 项让模型局部修。"""
    res = inspect_structure(memory_dir, big=big, small=small, wide=wide)
    if not res["warnings"]:
        return 0
    handled = 0
    # 先文件内并近义标题（步骤A，已实现，确定性）
    for root, dirnames, filenames in os.walk(memory_dir):
        dirnames[:] = [d for d in dirnames if d != "raw" and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".md"):
                consolidate_topics(os.path.join(root, fn))
    # 再对 big/dup 局部让模型修（用同一 rebalance agent，但只喂当前报告）
    if any(w["kind"] in ("big", "dup") for w in res["warnings"]):
        top_view = _top_level_view(memory_dir)
        _run_rebalance_agent(memory_dir, res["report"], top_view)
        handled = len(res["warnings"])
    return handled

def reorganize_library(memory_dir, big=150, small=8, wide=20):
    """节奏3：库规模增长触发的全局重构（= 记忆初始化）。模型拿全局报告+顶层视图重排。"""
    res = inspect_structure(memory_dir, big=big, small=small, wide=wide)
    top_view = _top_level_view(memory_dir)
    return _run_rebalance_agent(memory_dir, res["report"], top_view)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_structure.py -v`
Expected: PASS（全部 passed）

- [ ] **Step 5: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add tests/test_v7_structure.py src/nativemem.py
git commit -m "feat(v7): tidy_local (rhythm2) + reorganize_library (rhythm3) + _REBALANCE_PROMPT"
```

---

### Task 8: `build_memory` 加 v7 分支 —— 三节奏主循环

**Files:**
- Modify: `src/adapters/run_nativemem.py:241-294`（`build_memory` 加 v7 分支）
- Modify: `src/adapters/run_nativemem.py:39-41`（import 新符号）

**Interfaces:**
- Consumes: `split_into_chunks`/`process_chunk_agent`/`tidy_local`/`reorganize_library`/`measure_library`/`library_grew_past_threshold`（Task 1/3/6/7）、`normalize_date`（NM:335）
- Produces: `build_memory` 在 `NATIVEMEM_PROMPT=="v7"` 时走三节奏；返回签名不变 `(elapsed, chunk_idx)`。

- [ ] **Step 1: 扩展 import**

改 `src/adapters/run_nativemem.py:39-41`，把新符号加进从 `nativemem` 的 import：

```python
from nativemem import (  # noqa: E402  （保持文件现有相对/绝对 import 风格）
    client, ALIYUN_MODEL, TOOLS, execute_tool, log_usage, process_chunk,
    save_to_raw_archive, check_and_reorganize, consolidate_topics,
    normalize_date, split_into_chunks, process_chunk_agent,
    tidy_local, reorganize_library, measure_library, library_grew_past_threshold,
)
```

> 若现有 import 是逐行/不同写法，按现有风格把这 7 个新名（`normalize_date, split_into_chunks, process_chunk_agent, tidy_local, reorganize_library, measure_library, library_grew_past_threshold`）加进去即可。

- [ ] **Step 2: 写失败测试（v7 分支跑通、三节奏被调）**

```python
# tests/test_v7_build.py
import os, sys, json, importlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "adapters"))

def test_build_memory_v7_calls_agent_and_rhythms(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    import nativemem, run_nativemem
    importlib.reload(run_nativemem)

    seen = {"chunks": 0, "tidy": 0, "reorg": 0}
    def fake_pca(chunk_text, chunk_date, memory_dir, dia_ids, running_summary="", mode="oneshot"):
        seen["chunks"] += 1
        assert dia_ids and dia_ids[0].startswith("D")  # dia_id 传进来了
        return f"summary {seen['chunks']}"
    monkeypatch.setattr(run_nativemem, "process_chunk_agent", fake_pca)
    monkeypatch.setattr(run_nativemem, "tidy_local", lambda *a, **k: seen.__setitem__("tidy", seen["tidy"]+1) or 0)
    # 让节奏3 一定触发
    monkeypatch.setattr(run_nativemem, "library_grew_past_threshold", lambda *a, **k: True)
    monkeypatch.setattr(run_nativemem, "reorganize_library", lambda *a, **k: seen.__setitem__("reorg", seen["reorg"]+1) or 0)
    monkeypatch.setattr(run_nativemem, "measure_library", lambda d: {"files":0,"dirs":0,"bytes":0,"entries":0})

    with open(os.path.join(_ROOT, "benchmarks/locomo/data/locomo10.json")) as f:
        conv = json.load(f)[0]["conversation"]
    run_nativemem.build_memory(conv, str(tmp_path / "mem"), max_sessions=2)

    assert seen["chunks"] > 0      # 每 chunk 走 agent
    assert seen["tidy"] >= 2       # 每 session 末 tidy（2 个 session）
    assert seen["reorg"] >= 1      # 增长触发全局重构
```

- [ ] **Step 3: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_build.py -v`
Expected: FAIL —— v7 分支未实现，`process_chunk_agent`/`tidy_local` 未被调（`seen` 全 0），断言失败。

- [ ] **Step 4: 在 `build_memory` 里加 v7 分支**

在 `src/adapters/run_nativemem.py` 的 `build_memory` 开头（收集 sessions/dates 之后、现有切 chunk 循环之前）插入 v7 分支并 return：

```python
    # ---- v7: agent 自组织 + 三节奏 ----
    if os.environ.get("NATIVEMEM_PROMPT") == "v7":
        import time as _time
        t0 = _time.time()
        store_mode = os.environ.get("NATIVEMEM_STORE_MODE", "oneshot")
        rebalance_on = os.environ.get("NATIVEMEM_REBALANCE", "on") != "off"
        growth_env = os.environ.get("NATIVEMEM_REORG_GROWTH", "2.0")
        reorg_on = growth_env != "off"
        growth_ratio = float(growth_env) if reorg_on else 0.0
        file_delta = int(os.environ.get("NATIVEMEM_REORG_FILE_DELTA", "8"))
        big = int(os.environ.get("NATIVEMEM_BIG_FILE", "150"))
        small = int(os.environ.get("NATIVEMEM_SMALL_FILE", "8"))
        wide = int(os.environ.get("NATIVEMEM_WIDE_DIR", "20"))

        os.makedirs(memory_dir, exist_ok=True)
        snapshot = measure_library(memory_dir)
        chunk_idx = 0
        for si, (session, date) in enumerate(zip(sessions, dates)):
            obs = normalize_date(date)
            session_summary = ""
            for chunk_text, dia_ids in split_into_chunks(session, 10):
                if not chunk_text.strip():
                    continue
                chunk_idx += 1
                s = process_chunk_agent(chunk_text, obs, memory_dir, dia_ids,
                                        running_summary=session_summary,
                                        mode=store_mode)
                # 滚动保留最近 5 条 chunk 总结
                session_summary = "\n".join(
                    (session_summary + f"\n- {s}").strip().splitlines()[-5:])
            # 节奏2：每 session 末轻量再平衡
            if rebalance_on:
                tidy_local(memory_dir, big=big, small=small, wide=wide)
            # 节奏3：库规模增长过阈值 → 全局重构
            if reorg_on and library_grew_past_threshold(
                    snapshot, measure_library(memory_dir),
                    growth_ratio=growth_ratio, file_delta=file_delta):
                reorganize_library(memory_dir, big=big, small=small, wide=wide)
                snapshot = measure_library(memory_dir)
        return _time.time() - t0, chunk_idx
```

> 说明：`sessions`/`dates` 变量是 `build_memory` 现有代码在 RN:247-251 收集的；v7 分支放在它们之后。旧路径（save_to_raw_archive + process_chunk + _tidy）完全不动，只在非 v7 时走。

- [ ] **Step 5: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_build.py -v`
Expected: PASS（1 passed）

- [ ] **Step 6: 回归——确认旧路径没被破坏**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/adapters'); import run_nativemem; print('import ok')"`
Expected: 输出 `import ok`（无 import 错误，旧分支语法完好）

- [ ] **Step 7: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add src/adapters/run_nativemem.py tests/test_v7_build.py
git commit -m "feat(v7): build_memory v7 branch with three-rhythm structure maintenance"
```

---

### Task 9: `collect_memories` 加 v7 检索分支 —— 看顶层自己翻 + 两口径

**Files:**
- Modify: `src/adapters/run_nativemem.py`（`collect_memories` 加 v7 分支，约 RN:179-221）
- Create: `tests/test_v7_retrieve.py`

**Interfaces:**
- Consumes: `_top_level_view`（Task 2，需从 nativemem import）、`parse_memories`（RN:224）、`execute_tool`/`client`/`ALIYUN_MODEL`/`TOOLS`/`log_usage`（现有）
- Produces: `collect_memories` 在 `NATIVEMEM_PROMPT=="v7"` 时走"顶层视图 + agent bash 往下翻"；`NATIVEMEM_NO_ORIGINAL=1` 时过滤检索结果里的 `[source](...)` 行。返回签名不变 `(list[dict], int)`。

- [ ] **Step 1: import `_top_level_view` 到 run_nativemem**

在 Task 8 改过的 import 块里追加 `_top_level_view`。

- [ ] **Step 2: 写失败测试（no_original 过滤 source 行）**

```python
# tests/test_v7_retrieve.py
import os, sys, importlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "adapters"))

def _fake_agent_final(monkeypatch, run_nativemem, final_text):
    class R:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": final_text, "tool_calls": None})()})()]
            self.usage = type("U", (), {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2})()
    monkeypatch.setattr(run_nativemem.client.chat.completions, "create", lambda **k: R())

def test_v7_retrieve_filters_source_when_no_original(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    monkeypatch.setenv("NATIVEMEM_NO_ORIGINAL", "1")
    import run_nativemem
    importlib.reload(run_nativemem)
    (tmp_path / "Caroline.md").write_text("## g\n[2023-05-08] went to support group\n  [source](D1:3)\n")
    final = ("<memories>\n[2023-05-08] went to support group\n  [source](D1:3)\n</memories>")
    _fake_agent_final(monkeypatch, run_nativemem, final)
    mems, _ = run_nativemem.collect_memories("when support group?", str(tmp_path))
    joined = " ".join(m["text"] for m in mems)
    assert "support group" in joined
    assert "D1:3" not in joined            # source 行被过滤
    assert "[source]" not in joined

def test_v7_retrieve_keeps_source_when_use_original(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    monkeypatch.delenv("NATIVEMEM_NO_ORIGINAL", raising=False)
    import run_nativemem
    importlib.reload(run_nativemem)
    (tmp_path / "Caroline.md").write_text("## g\n[2023-05-08] went to support group\n  [source](D1:3)\n")
    final = ("<memories>\n[2023-05-08] went to support group [source](D1:3)\n</memories>")
    _fake_agent_final(monkeypatch, run_nativemem, final)
    mems, _ = run_nativemem.collect_memories("when support group?", str(tmp_path))
    joined = " ".join(m["text"] for m in mems)
    assert "support group" in joined       # 用原文口径保留（此处只验证不崩、有结果）
```

- [ ] **Step 3: 运行确认失败**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_retrieve.py -v`
Expected: FAIL —— v7 分支未实现，检索走了旧 grep/nav，`[source]` 未按 no_original 过滤。

- [ ] **Step 4: 加 v7 检索分支 + source 过滤**

在 `collect_memories` 里，模式选择处加 v7 分支（放在现有 `nav`/`grep` 判断之前）：

```python
    if os.environ.get("NATIVEMEM_PROMPT") == "v7":
        return _collect_memories_v7(question, memory_dir, max_rounds=max_rounds)
```

并新增函数（放在 `collect_memories` 附近）：

```python
_V7_RETRIEVE_PROMPT = """\
你在一个用文件系统组织的记忆库里检索，回答问题需要的记忆条目。工作目录是记忆库根目录，
只能用 bash（ls / cat / grep / sed）。

## 顶层结构
{top_view}

## 问题
{question}

从顶层往下翻：grep '^## ' 看某文件主题、ls 进子目录、grep 关键词按内容找、
grep '\\[2023-05' 按时间标签找。把与问题相关的记忆条目（连同它们的行）收集起来，
放进 <memories></memories>。只收集，不要在这里回答问题。
"""

def _filter_source_lines(text):
    """no_original 口径：删掉 [source](...) 定位行/片段，藏掉来源锚。"""
    import re as _re
    # 整行就是 source 的：删行
    kept = [ln for ln in text.splitlines()
            if not _re.match(r"\s*\[source\]\(.*\)\s*$", ln)]
    out = "\n".join(kept)
    # 行内尾随的 [source](...) 也去掉
    out = _re.sub(r"\s*\[source\]\([^)]*\)", "", out)
    return out

def _collect_memories_v7(question, memory_dir, max_rounds=10):
    top_view = nativemem._top_level_view(memory_dir) \
        if hasattr(nativemem, "_top_level_view") else _top_level_view(memory_dir)
    prompt = _V7_RETRIEVE_PROMPT.format(top_view=top_view, question=question)
    messages = [{"role": "user", "content": prompt}]
    final_text, rounds = "", 0
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL, messages=messages, tools=TOOLS,
            max_tokens=1500, temperature=0.0)
        log_usage(resp, phase="v7_retrieve")
        rounds += 1
        msg = resp.choices[0].message
        final_text = msg.content or final_text
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments)
            out = execute_tool(tc.function.name, args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    if os.environ.get("NATIVEMEM_NO_ORIGINAL") == "1":
        final_text = _filter_source_lines(final_text)
    return parse_memories(final_text), rounds
```

> 说明：`_top_level_view` 若已从 nativemem import（Task 9 Step 1），直接用；这里用 `hasattr` 兜底两种引用方式。`execute_tool` 在 v7 检索不传 `hide_raw`——记忆里本就没 raw，无需屏蔽；两口径差别只由 `_filter_source_lines` 控制。

- [ ] **Step 5: 运行确认通过**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/test_v7_retrieve.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 全量回归**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add src/adapters/run_nativemem.py tests/test_v7_retrieve.py
git commit -m "feat(v7): collect_memories v7 branch (top-view drill-in) + no-original source filter"
```

---

### Task 10: 端到端冒烟（真实模型，小样本）

**Files:**
- Create: `scripts/smoke_v7.sh`

**Interfaces:**
- Consumes: 全部 v7 路径
- Produces: 一个可手跑的冒烟脚本，构建 sample0 的前 2 个 session + 检索 3 个问题，人工核对产出。

- [ ] **Step 1: 建冒烟脚本**

```bash
# scripts/smoke_v7.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export NATIVEMEM_PROMPT=v7
export NATIVEMEM_STORE_MODE=oneshot
export BUILDER_MODEL="${BUILDER_MODEL:-qwen3.6-flash}"
OUT="results/smoke-v7"
rm -rf "$OUT"
python src/adapters/run_nativemem.py \
  --sample 0 --max-sessions 2 --questions-limit 3 \
  --output "$OUT/result.json"
echo "=== 生成的记忆库结构 ==="
find "$OUT/memory_sample0" -type f | sort
echo "=== 抽查一个文件 ==="
find "$OUT/memory_sample0" -name '*.md' | head -1 | xargs cat
echo "=== 检索结果 ==="
cat "$OUT/result.json"
```

- [ ] **Step 2: 跑冒烟（需要网络 + 阿里云 key）**

Run: `cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki" && bash scripts/smoke_v7.sh`
Expected（人工核对）：
- `memory_sample0/` 下有模型自建的 `.md`（**不是**硬编码 `people/`）
- 每条记忆是三行块，含 `[source](D1:x)` dia_id 锚
- `result.json` 有 3 个问题的检索结果

- [ ] **Step 3: 提交**

```bash
cd "/Users/fzkuji/Documents/Research-Wiki/Research Ideas/model-aligned-wiki"
git add scripts/smoke_v7.sh
git commit -m "chore(v7): end-to-end smoke script (2 sessions, 3 questions)"
```

---

## Self-Review

**Spec coverage：**
- §1.1 三行块 + dia_id 锚 → Task 5/6（写入格式）、Task 9（检索过滤 source）✓
- §1.2 结构模型自组织 → Task 6（agent 自组织，无硬编码路径）✓
- §1.3 STRUCTURE_STANDARD → Task 5 ✓
- §1.5 三节奏 → 节奏1 Task 6、节奏2 Task 7 `tidy_local`、节奏3 Task 7 `reorganize_library` + Task 8 触发 ✓
- §2 process_chunk_agent（oneshot/threecall） → Task 6 ✓
- §2.3 _top_level_view → Task 2 ✓
- §2.5 即时维护三规则 → Task 5 prompt ✓
- §3 主循环 + split_into_chunks + measure/growth → Task 1/3/8 ✓
- §4.0 inspect_structure → Task 4 ✓
- §4.A tidy_local / §4.B reorganize_library / _REBALANCE_PROMPT → Task 7 ✓
- §0.6 触发机制（阈值、可关） → Task 8 环境变量 ✓
- §5 检索看顶层自己翻 → Task 9 ✓
- §6 两口径（过滤 [source]） → Task 9 ✓
- §8 时间标签检索 → Task 9 prompt 里 `grep '\[2023-05'` ✓
- §9 配置表 → Task 8 全部环境变量 ✓
- §10 废弃 store_facts_code → v7 分支不走它（保留旧版）✓

**Placeholder scan：** 无 TBD/TODO；每个 code step 给了完整代码。

**Type consistency：** `process_chunk_agent(chunk_text, chunk_date, memory_dir, dia_ids, running_summary, mode)` 在 Task 6 定义、Task 8 调用一致；`inspect_structure` 返回 `{report, warnings}` 在 Task 4 定义、Task 7 消费一致；`measure_library` 返回 `{files,dirs,bytes,entries}` 在 Task 3 定义、Task 8 消费一致；`library_grew_past_threshold(old,new,growth_ratio,file_delta)` 一致。

**未纳入（YAGNI）：** LlamaFactory / dia_id→evidence 召回分析属评测分析，不在写入/检索路径，留给后续实验脚本；`store_facts_code` 不删（旧版本回归需要）。
