# NativeMem v8 → spec-3 自组织补齐 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 v8(双视图 markdown:`topics/` + `timeline/`,事件行 `[date] summary · [dia_id]`,方案 B 单模型为主,s0 overall 88.8)补齐到 spec-3 三个短板:①检索地图残缺 ②写入结构硬编码、topic 名无收敛 ③无整理环节。按依赖排序落三块:地图解耦 → 整理环节 → 写入自组织放置。

**核心红线(用户已批准,见 `docs/method/memory_management_design.md` §5.1/§5.2/§5.4 与 spec-3 §5/§6/§9/§12):**
- 结构由模型自组织,代码不写死任何记忆内部路径;地图函数不 hardcode 目录名,walk 整棵树如实呈现;默认目录级+文件计数、不平铺全量文件名(展开策略做成开关)。
- 代码只做确定性活(测量、触发判定、精确去重、给候选),语义决策(合并/命名/放哪)归模型。
- 保留 v8 "提炼一次 + 代码落盘" 的写入效率,不回退到 v7 逐条 bash 手写。
- 硬编码常量提成 `NATIVEMEM_*` env var 为消融做准备。
- 评测口径不变:方案 B、qwen3.6-flash build/answer、gpt-4o-mini judge、`scripts/eval_full.py`。不许用 benchmark 答案调参。

**Architecture:** 只改 v8 分支/函数。落地文件仅两个:`src/v8_memory.py`(写入/落盘/整理数据操作)、`src/adapters/run_nativemem.py`(v8 build 主循环、检索地图、prompt)。v4/v6/v7 一律不动。可复用的 v7 确定性工具:`measure_library`、`library_grew_past_threshold`、`_count_entries_in_file`、`_find_merge_candidates`/`_heading_similarity`(都在 `src/nativemem.py`,且 `_ENTRY_RE = ^\[\d{4}-\d{2}-\d{2}\]` 恰好匹配 v8 事件行)——直接 import,不重写。

**Tech Stack:** Python 3,OpenAI SDK(qwen3.6-flash @ 阿里云),pytest,monkeypatch fake client 单测(不联网)。

## Global Constraints

- 只改 v8:`NATIVEMEM_PROMPT == "v8"` 分支与 v8 专用函数。v4/v6/v7 行为字节级不变。
- `run_nativemem.py` 调 `v8_memory` 的函数一律用模块限定 `v8_memory.xxx`(否则 monkeypatch 失效——历史教训,见 `tests/test_v8_build.py` 注释)。
- 事件行格式不变:`[YYYY-MM-DD] summary · [D1:3, D1:5]`(`event_line`)。任何整理都不许改行内 `[dia_id]` 锚、不许丢内容。
- 地图/整理都跳过隐藏文件;v8 记忆库里没有 `raw/`(原文留在数据集原位),但 walk 逻辑仍统一跳过 `raw`/隐藏,与 `measure_library` 一致。
- 新 env var 全部有默认值,默认行为 = 当前行为或最保守形态,保证"不设任何新 var 时结果可复现 88.8 基线"(除非该块本身就是要改默认,届时显式说明)。
- 每块可独立评测:单块落地后跑 `bash scripts/eval_v8_ab.sh 0`(方案 A/B 都出),拿方案 B overall 与 88.8 对比。

---

# 块一:检索地图解耦(最小改动,先落地,独立评测)

**问题(诊断已确认):** `_v8_topic_list`(`run_nativemem.py:286`)只 `listdir(topics/)`、平铺全部文件名、不含 `timeline/`。实测单样本 topic 文件数 12/72/138 失控,平铺后地图爆且噪声大,导航失效。

**目标行为:** 地图函数递归 walk `memory_dir` 整棵树(含 `timeline/` 与任意模型自建目录),如实呈现**目录树 + 每目录/文件的条目计数**,默认**目录级 + 文件计数、不平铺全量文件名**;展开粒度做成开关。不 hardcode 任何目录名。

### Task 1.1: 结构地图函数 `_v8_structure_map`

**Files:**
- Modify: `src/adapters/run_nativemem.py`(新增 `_v8_structure_map`;`_collect_v8`/`_collect_and_answer_v8` 改调它;`_V8_RETRIEVE_PROMPT`/`_V8_SINGLE_PROMPT` 的 `{topic_list}` 段改成结构树说明)
- Test: `tests/test_v8_map.py`

**新旧行为对比:**

| | 旧 `_v8_topic_list`(:286) | 新 `_v8_structure_map` |
|---|---|---|
| 覆盖 | 只 `topics/` | 递归整棵树,含 `timeline/`、模型自建子目录 |
| 呈现 | 平铺全部 `.md` basename | 目录树缩进 + 每项 `[N 条]` 计数;默认**不**列 `topics/` 下每个文件名(只给 `topics/ (72 files, 640 entries)` 之类摘要),`timeline/` 因是代码固定两层、文件少,列到月份文件 |
| 硬编码 | 写死 `topics` 目录名 | 只知 `memory_dir` 根入口,walk 出什么呈现什么 |
| 上下文 | 138 个文件名炸 prompt | 目录级恒小 |

**展开策略开关 `NATIVEMEM_V8_MAP`(默认 `dir`):**
- `dir`(默认):只到目录级 + 计数,`topics/` 只给一行汇总(`topics/ (N files, M entries)`),`timeline/` 给到月份文件(年目录下每 `MM-Mon.md` 一行带条目数)。地图恒小。
- `files`:`topics/` 也逐文件列(basename + 条目数),复现"平铺"形态但带计数——留作对照实验,验证"目录级 vs 平铺"哪种检索更准。
- (不做 embedding/语义聚类展开——YAGNI;`dir`/`files` 两档够验证假设,语义分组等实测证明目录级不够再加。)

**Interfaces:**
- Produces: `_v8_structure_map(memory_dir, mode=None) -> str`。`mode` 缺省读 `NATIVEMEM_V8_MAP`。返回缩进目录树文本。空库返回 `(empty memory)`。计数复用 `_count_entries_in_file`(import 自 `src.nativemem`)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_v8_map.py
import os
import src.adapters.run_nativemem as R
import src.v8_memory as V8

def _seed(d):
    V8.write_events(d, [
        {"when": "2023-05-07", "summary": "A", "dia_ids": ["D1:1"], "topic": "Caroline-support"},
        {"when": "2023-05-08", "summary": "B", "dia_ids": ["D1:2"], "topic": "Caroline-support"},
        {"when": "2023-06-02", "summary": "C", "dia_ids": ["D2:1"], "topic": "Melanie-art"},
    ])

def test_map_includes_timeline_not_just_topics(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="dir")
    assert "timeline" in m          # 旧地图缺 timeline,这里必须有
    assert "topics" in m

def test_map_dir_mode_does_not_flatten_all_topic_files(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="dir")
    # dir 模式给 topics/ 汇总而非逐个 basename
    assert "Caroline-support" not in m
    assert "2 files" in m or "2 file" in m   # topics 下 2 个文件的汇总

def test_map_files_mode_lists_topic_files_with_counts(tmp_path):
    d = str(tmp_path / "mem"); _seed(d)
    m = R._v8_structure_map(d, mode="files")
    assert "Caroline-support" in m
    assert "2" in m   # 该 topic 2 条

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
```

- [ ] **Step 2: 跑测试确认失败**
Run: `python -m pytest tests/test_v8_map.py -v` — Expected FAIL(`_v8_structure_map` 未定义)。

- [ ] **Step 3: 实现**
- 在 `run_nativemem.py` 新增 `_v8_structure_map`:`os.walk(memory_dir)`,跳过 `raw`/隐藏;对每个目录累计其下 `.md` 文件数与条目数(`_count_entries_in_file`);按缩进层级输出。`dir` 模式对 `topics/`(及任意含 >`NATIVEMEM_V8_MAP_INLINE`(默认 8)个文件的目录)只输出一行汇总 `name/ (K files, N entries)`,其余目录/文件逐项列(带 `[N]`);`files` 模式全部逐文件列。`timeline/` 天然文件少(年/月),照常列到月份文件。
  - ponytail: 用"目录文件数超阈值才折叠成汇总"这一条统一规则,而不是特判 `topics` 名——满足"不 hardcode 目录名"红线。阈值 env `NATIVEMEM_V8_MAP_INLINE` 默认 8。
- `_collect_v8`、`_collect_and_answer_v8` 两处 `.format(... topic_list=_v8_topic_list(memory_dir))` 改成 `structure=_v8_structure_map(memory_dir)`;prompt 里 `{topic_list}` 段落改成 `{structure}`,说明文字从"现有话题文件清单"改为"记忆库当前结构(目录/文件+条目数),先看这张地图再决定 cat/grep 哪里"。
- 保留 `_v8_topic_list` 不删(可能被别处/测试引用),标注 deprecated。

- [ ] **Step 4: 跑测试确认通过**
Run: `python -m pytest tests/test_v8_map.py tests/test_v8_retrieve.py -q` — Expected PASS。若 `test_v8_retrieve` 因 prompt 变量改名失败,更新 fake 用例(断言意图不变:回原文取到 D1:3)。全量 `python -m pytest tests/ -q` 无回归。

- [ ] **Step 5: 提交**
```bash
git checkout -b v8-self-organized-upgrade   # 若尚在默认分支
git add src/adapters/run_nativemem.py tests/test_v8_map.py tests/test_v8_retrieve.py
git commit -m "feat(v8): 检索地图递归 walk 整棵树(含 timeline),默认目录级+计数不平铺"
```

### Task 1.2: 块一独立评测

- [ ] **Step 1: 跑单样本 A/B 对比**
Run: `bash scripts/eval_v8_ab.sh 0`(方案 B 用 `NATIVEMEM_V8_SINGLE=1`)。
Expected(人工核对):方案 B overall 与基线 88.8 对比,导航/召回错题(原 12/15)应减少或持平(地图不再爆、含 timeline 后时间题更易导航)。记录分数、检索平均 steps。**不看 benchmark 答案改 prompt,只看聚合分。**
- [ ] **Step 2:** 若持平或提升 → 保留 `dir` 为默认;顺带 `NATIVEMEM_V8_MAP=files` 跑一次作对照,记录哪档高。若掉分,报告 controller 定夺(可能需调 `NATIVEMEM_V8_MAP_INLINE` 或 timeline 呈现粒度)。

---

# 块二:整理环节(环节 B,spec §5.2 / §9)

**问题(诊断已确认):** v8 纯增量追加 `write_events`,从不合并近义/去重/拆分。topic 名由 LLM 每 chunk 自由起、无收敛 → topic 文件数失控(72/138)。

**目标行为:** 加显式整理环节:**代码触发**(每 session 末 + 结构指标超标)、**代码找候选**(近义 topic 文件名相似度)、**模型定夺**合并/命名。目标 topic 文件数收敛。分工死守 spec §5.4:代码测量/触发/给候选/精确去重,模型只判"这几个近义 topic 文件是否合一、合后叫什么"。

**为什么整理只针对 `topics/`,不动 `timeline/`:** `timeline/` 是代码按 `[date]` 确定性路由的固定结构(`timeline_path`),无碎片问题;碎片全在模型自由起名的 `topics/`。整理专治 topic 文件收敛。

### Task 2.1: 精确去重(纯代码,确定性)

**Files:**
- Modify: `src/v8_memory.py`(新增 `dedup_topic_files`)
- Test: `tests/test_v8_tidy.py`

**Interfaces:**
- Produces: `dedup_topic_files(memory_dir) -> int`。walk `topics/`(及模型自建目录里的 `.md`),每文件内**逐行精确去重**(完全相同的事件行只留一条,保序),返回删除行数。纯确定性,不调模型。(`timeline/` 由 `_append_sorted` 已排序,可选一并去重。)

- [ ] **Step 1: 失败测试**
```python
# tests/test_v8_tidy.py
import os
import src.v8_memory as V8

def test_dedup_removes_exact_duplicate_lines(tmp_path):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when":"2023-05-07","summary":"A","dia_ids":["D1:1"],"topic":"t"}])
    V8.write_events(d, [{"when":"2023-05-07","summary":"A","dia_ids":["D1:1"],"topic":"t"}])
    removed = V8.dedup_topic_files(d)
    body = open(os.path.join(d,"topics","t.md")).read()
    assert body.count("] A ·") == 1
    assert removed >= 1
```
- [ ] **Step 2: 确认失败** — `python -m pytest tests/test_v8_tidy.py -v`
- [ ] **Step 3: 实现** `dedup_topic_files`:读每 `.md`,`seen=set()` 逐行过滤完全相同行,改写文件。
- [ ] **Step 4: 通过** — `python -m pytest tests/test_v8_tidy.py -q`
- [ ] **Step 5: 提交** `git commit -m "feat(v8): topic 文件精确去重(确定性)"`

### Task 2.2: 近义 topic 文件合并(代码给候选 + 模型定夺)

**Files:**
- Modify: `src/v8_memory.py`(新增 `_topic_merge_candidates`、`consolidate_topic_files`、`_V8_MERGE_PROMPT`)
- Test: `tests/test_v8_tidy.py`(追加)

**分工:**
- 代码:列出所有 topic 文件 basename → `_find_merge_candidates`(复用 `src.nativemem`,已能按共享显著词/词重叠聚候选组,spec §13 明确"扩到跨文件文件名")→ 得到候选组。
- 模型:只看候选组,判每组是否合、合后文件名(复用 `_MERGE_PROMPT` 的 JSON 契约思路,但 v8 是**文件名**不是 `## 标题`,写 `_V8_MERGE_PROMPT`)。
- 代码:按模型裁决,把被合并文件的事件行搬进目标文件(`_append_sorted` 保持按日期序)、删空文件、跑 `dedup_topic_files`。**不改任何 `[dia_id]` 锚。**

**Interfaces:**
- `_topic_merge_candidates(memory_dir) -> list[list[str]]`:候选组(topic basename)。
- `consolidate_topic_files(memory_dir) -> int`:跑候选→模型→执行合并,返回合并的文件数。带 `max_retry` 与 fake-client 可注入(同 `distill_events` 写法,用 `client`/`ALIYUN_MODEL`/`log_usage`)。

- [ ] **Step 1: 失败测试**
```python
def test_topic_merge_candidates_finds_near_synonyms(tmp_path):
    d = str(tmp_path / "mem")
    for t in ["mental-health-career","mental-health-work","art"]:
        V8.write_events(d, [{"when":"2023-05-07","summary":"x","dia_ids":["D1:1"],"topic":t}])
    groups = V8._topic_merge_candidates(d)
    flat = [set(g) for g in groups]
    assert any({"mental-health-career","mental-health-work"} <= g for g in flat)

def test_consolidate_merges_per_model_verdict(tmp_path, monkeypatch):
    d = str(tmp_path / "mem")
    V8.write_events(d, [{"when":"2023-05-07","summary":"P","dia_ids":["D1:1"],"topic":"mental-health-career"}])
    V8.write_events(d, [{"when":"2023-05-08","summary":"Q","dia_ids":["D1:2"],"topic":"mental-health-work"}])
    import json as _j
    monkeypatch.setattr(V8.client.chat.completions, "create",
        lambda **k: type("R",(),{"choices":[type("C",(),{"message":type("M",(),{"content":
          _j.dumps({"merges":[{"from":["mental-health-career","mental-health-work"],
                               "into":"mental-health-career","merge":True}]})})()})()],
          "usage":type("U",(),{"prompt_tokens":1,"completion_tokens":1})()})())
    monkeypatch.setattr(V8, "log_usage", lambda *a, **k: None)
    V8.consolidate_topic_files(d)
    assert not os.path.exists(os.path.join(d,"topics","mental-health-work.md"))
    merged = open(os.path.join(d,"topics","mental-health-career.md")).read()
    assert "P" in merged and "Q" in merged
```
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**
  - `_topic_merge_candidates`:`os.listdir` topic 文件 basename(剥 `.md`)→ `from src.nativemem import _find_merge_candidates` → 返回组(≥2 才算候选)。
  - `_V8_MERGE_PROMPT`:给候选组,要 JSON `{"merges":[{"from":[...],"into":"...","merge":true/false}]}`。
  - `consolidate_topic_files`:调模型、解析 JSON(复用 `distill_events` 的 `<think>`/```json 清洗 + `re.search(r"\{.*\}")` 兜底);对 `merge:true` 组,把 `from` 各文件行 append 到 `into.md`(`_append_sorted`),删源文件,最后 `dedup_topic_files`。
- [ ] **Step 4: 通过 + 全量无回归**
- [ ] **Step 5: 提交** `git commit -m "feat(v8): 近义 topic 文件合并(代码给候选/模型定夺/代码执行)"`

### Task 2.3: 触发挂到 build 主循环

**Files:**
- Modify: `src/adapters/run_nativemem.py`(v8 build 分支 :520–555)
- Test: `tests/test_v8_build.py`(追加)

**触发(代码判定,spec §6.1):**
- 每 session 末:`NATIVEMEM_V8_TIDY`(默认 `on`)→ 调 `dedup_topic_files` + 结构指标判定。
- 结构超标才合并(省 API,多数 session 空过):topic 文件数 > `NATIVEMEM_V8_MAX_TOPICS`(默认 30)触发 `consolidate_topic_files`。(用 topic **文件数**作触发量,直击"文件数收敛"目标;复用 `measure_library`/`library_grew_past_threshold` 作全局增长的二级触发可选,先用文件数阈值这一条最直接。)
  - ponytail: 只用一个"topic 文件数超阈值"触发合并 + 每 session 去重,不搬 v7 三节奏全套(节奏3 全局重排对 v8 flat 结构收益未证)。三节奏留作后续消融,YAGNI 到实测证明需要再加。

**新旧行为对比:** 旧 v8 build(:540–554)循环里只 `distill_events`→`write_events`→累积 `known_topics`。新增:内层 chunk 循环后、session 边界处插入去重 + 条件合并。

**Interfaces (build 主循环伪代码):**
```python
tidy_on = os.environ.get("NATIVEMEM_V8_TIDY", "on") != "off"
max_topics = int(os.environ.get("NATIVEMEM_V8_MAX_TOPICS", "30"))
for session, date in zip(sessions, dates):
    obs = normalize_date(date)
    for turns, dia_ids in split_into_chunks_structured(session, v8_chunk):
        ... distill_events / write_events / known_topics ...
    if tidy_on:                                   # 节奏2:每 session 末
        v8_memory.dedup_topic_files(memory_dir)
        n_topics = len([f for f in os.listdir(os.path.join(memory_dir,"topics"))
                        if f.endswith(".md")]) if os.path.isdir(...) else 0
        if n_topics > max_topics:
            v8_memory.consolidate_topic_files(memory_dir)
            known_topics = _reload_known_topics(memory_dir)  # 合并后刷新(见块三)
```

- [ ] **Step 1: 失败测试**(fake distill 造 >阈值个近义 topic,断言 build 后 topic 文件数下降 / 去重生效;把 `NATIVEMEM_V8_MAX_TOPICS` 设小如 2 便于触发)
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**(build v8 分支插入上面逻辑;`v8_memory.` 模块限定调用)
- [ ] **Step 4: 通过 + 全量无回归**
- [ ] **Step 5: 提交** `git commit -m "feat(v8): build 每 session 末去重+超阈值触发 topic 合并"`

### Task 2.4: 块二独立评测
- [ ] `bash scripts/eval_v8_ab.sh 0`;核对方案 B overall vs 88.8,**并 `find memory_sample0/topics -name '*.md' | wc -l` 确认 topic 文件数从 72/138 量级收敛**(这是块二的核心验收指标,独立于分数)。`NATIVEMEM_V8_TIDY=off` 跑一次作消融对照。

---

# 块三:写入自组织放置(spec §7.4 规则1/3,known_topics 升级为结构感知)

**问题(诊断已确认):** `write_events`(`v8_memory.py:103`)机械落盘 `topics/{topic}.md`——topic 名由 LLM 每 chunk 自由起、看不到已有结构,无收敛。`known_topics` 现在只是**平铺的字符串集合**(build:534/553 传给 distill),模型看不到"结构"(哪些文件多大、怎么分布),只看到一堆名字。

**目标行为:** distill 出事件后,把**当前结构地图**给模型看,由模型决定归入已有文件还是新建(spec §7.4 规则1"放最相关的已有文件"、规则3"命名沿用惯例")。保持"提炼一次 + 代码落盘"——模型只**输出 topic 名(复用已有则收敛)**,落盘仍由 `write_events` 做,不回退到逐条 bash 手写。

**关键设计(守住写入效率红线):** 不给模型 bash 工具、不让它逐条搬文件。升级点仅是:`distill_events` 的 `known_topics` 从"平铺名字列表"→"结构感知视图"(沿用块一 `_v8_structure_map` 的 `dir` 摘要 + topic 名清单),让模型起名/复用时看得到"已有哪些 topic、各多大",从而收敛到已有名而非造近义新名。落盘路径仍是 `topics/{model_chosen_topic}.md`。

### Task 3.1: known_topics 升级为结构感知视图

**Files:**
- Modify: `src/v8_memory.py`(`_V8_DISTILL_PROMPT` 的 `known_topics` 段说明升级)、`src/adapters/run_nativemem.py`(build 传入的 `known_topics` 从 `set` 升级为带计数的结构文本;新增 `_reload_known_topics`/`_topics_snapshot`)
- Test: `tests/test_v8_build.py`(追加)、`tests/test_v8_distill.py`(若 prompt 变量契约变)

**新旧对比:**

| | 旧 | 新 |
|---|---|---|
| 传给 distill 的 known | `sorted(known_topics)` 纯名字 | 带**每 topic 条目数**的清单(`Caroline-support (5), Melanie-art (2)`),大 topic 排前,让模型优先复用活跃 topic |
| 收敛机制 | prompt 软要求"别造近义新名" | 同左,但模型看得到已有 topic 的**规模分布**,复用信号更强;合并(块二)兜底 |
| 落盘 | `write_events` 机械落 `topics/{topic}.md` | 不变(守写入效率红线) |

- ponytail: 不改 `write_events` 落盘机制、不给模型 bash——只把 `known_topics` 这个已有入参喂得更结构化。这是"结构感知放置"用最小改动达成的形态;spec §7 的 oneshot/threecall bash-agent 写入是 v7 路线,v8 明确保留"提炼一次+代码落盘",不引入。若实测证明纯计数清单仍不收敛,再考虑给模型看目录树(升级 `known_topics` 文本即可,不动落盘)。

**Interfaces:**
- `_topics_snapshot(memory_dir) -> list[(topic, count)]`(按 count 降序);build 每 session 起用它刷新 `known_topics` 文本传入 `distill_events`。
- `_reload_known_topics(memory_dir) -> set`:块二合并后重建名字集(供 build 循环里合并后同步,避免模型再复用已删名)。
- `distill_events` 签名不变(`known_topics` 仍是最后一个语义位),只是传入内容从名字集变成"名字(计数)"清单字符串——**在 build 侧组装文本传入**,`distill_events` 内 `kt = ", ".join(known_topics)` 逻辑对字符串列表照常工作,改动最小。

- [ ] **Step 1: 失败测试**
```python
def test_distill_receives_known_topics_with_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT","v8")
    seen = []
    def fake_distill(turns, obs, dids, known_topics=None, **k):
        seen.append(list(known_topics) if known_topics else [])
        return [{"when":obs or "2023-01-01","summary":"e","dia_ids":list(dids),"topic":"Caroline-support"}]
    monkeypatch.setattr(V8, "distill_events", fake_distill)
    conv = {"session_1":[{"speaker":"C","dia_id":"D1:1","text":"support group"}],
            "session_1_date_time":"2023-05-07",
            "session_2":[{"speaker":"C","dia_id":"D2:1","text":"more support"}],
            "session_2_date_time":"2023-06-02"}
    R.build_memory(conv, str(tmp_path/"mem"))
    # session_2 传入的 known 应含带计数形态的 Caroline-support
    joined = " ".join(seen[1])
    assert "Caroline-support" in joined and "1" in joined  # session1 落了 1 条
```
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现** build v8 分支:每 session 起 `known_view = _format_topics_snapshot(_topics_snapshot(memory_dir))`,传 `known_topics=known_view`;合并后 `known_topics_set = _reload_known_topics(...)`。`_V8_DISTILL_PROMPT` 的 `known_topics` 行文案改为"已有话题名(含条目数,优先复用规模大的、别造近义新名)"。
- [ ] **Step 4: 通过 + 全量无回归**(`test_v8_build_threads_known_topics_across_sessions` 可能需按新格式微调断言,保持"跨 session 传状态"意图)
- [ ] **Step 5: 提交** `git commit -m "feat(v8): 写入放置结构感知——known_topics 升级为带计数清单,落盘仍代码托管"`

### Task 3.2: 块三独立评测
- [ ] `bash scripts/eval_v8_ab.sh 0`;方案 B overall vs 88.8;`find .../topics -name '*.md' | wc -l` 对比块二结果确认复用增强、文件数进一步收敛。此块开关是"是否喂计数",可临时回退 `sorted(set)` 作消融。

---

# 常量提 env(贯穿三块,spec §12)

诊断点名的硬编码:`_collect_and_answer_v8`/`_collect_v8` 的 `max_rounds=8`、`max_tokens=1200`、`read_turns` 的 `context=1`。

### Task 4: 提取 NATIVEMEM_* 消融开关

**Files:** Modify `src/adapters/run_nativemem.py`、`src/v8_memory.py` — Test: `tests/test_v8_env.py`

| env var | 默认 | 替代 |
|---|---|---|
| `NATIVEMEM_V8_MAX_ROUNDS` | `8` | `_collect_v8`/`_collect_and_answer_v8` 的 `max_rounds` |
| `NATIVEMEM_V8_MAX_TOKENS` | `1200` | 检索 `create(max_tokens=...)` |
| `NATIVEMEM_V8_READ_CONTEXT` | `1` | `read_turns(..., context=...)` 两处调用 |
| `NATIVEMEM_V8_MAP` | `dir` | 块一地图展开策略 |
| `NATIVEMEM_V8_MAP_INLINE` | `8` | 块一目录折叠阈值 |
| `NATIVEMEM_V8_TIDY` | `on` | 块二整理开关 |
| `NATIVEMEM_V8_MAX_TOPICS` | `30` | 块二合并触发阈值 |

- [ ] **Step 1:** 失败测试:`test_v8_env.py` 断言不设 env 时取默认值、设了 env 时被读取(可测 `_collect_v8` 默认 `max_rounds` 来自 env,用小值 + fake client 数轮数)。
- [ ] **Step 2–4:** 实现 `int(os.environ.get(...))` 替换写死常量;跑测试。默认值 = 现值,**保证 88.8 基线可复现**。
- [ ] **Step 5:** `git commit -m "feat(v8): 硬编码常量提成 NATIVEMEM_V8_* env(消融准备)"`

---

# 全量回归 + 收尾评测

- [ ] `python -m pytest tests/ -q` 全绿。
- [ ] `bash scripts/eval_v8_ab.sh 0` 方案 B 终态 vs 88.8;记录 overall、导航/召回错题数、topic 文件数收敛值、检索平均 steps。
- [ ] 若单样本达标,扩到 `scripts/eval_full.py` 全样本口径(qwen3.6-flash build/answer + gpt-4o-mini judge)跑一轮,出最终对比表。**全程不看 benchmark 答案调 prompt/阈值。**
- [ ] 用 superpowers:finishing-a-development-branch 决定合并/PR。

---

## Self-Review

**Spec coverage:**
- 块一 地图解耦(递归 walk 含 timeline、目录级+计数、不 hardcode 目录名、展开开关)→ Task 1.1/1.2,对齐 memory_design §5.1、spec §7.3/§10 ✓
- 块二 整理环节(代码触发/代码找候选/精确去重/模型定夺合并,topic 收敛)→ Task 2.1–2.4,对齐 memory_design §5.2、spec §9 ✓
- 块三 写入自组织放置(known_topics 升级结构感知、保留提炼+代码落盘)→ Task 3.1/3.2,对齐 memory_design §5.1、spec §7.4 ✓
- 常量提 env → Task 4,对齐 spec §12 ✓
- 红线:代码不写死路径(地图折叠按文件数阈值非目录名)、代码只做确定性活+给候选、不回退 bash 手写、每块独立评测、不用 benchmark 调参 ✓

**复用 vs 新写:** 复用 `measure_library`/`_count_entries_in_file`/`_find_merge_candidates`/`_heading_similarity`/`_ENTRY_RE`(v8 事件行天然匹配)、`_append_sorted`/`write_events`/`event_line`(落盘不变)、`distill_events` 的 JSON 清洗。新写 `_v8_structure_map`、`dedup_topic_files`、`_topic_merge_candidates`、`consolidate_topic_files`、`_V8_MERGE_PROMPT`、`_topics_snapshot`/`_reload_known_topics`。不引入 v7 三节奏全套(YAGNI,留作后续消融)。

**Placeholder scan:** 无 TBD/TODO;每 code step 给接口/伪代码。

**Type consistency:** `_v8_structure_map(dir, mode=None)->str`;`dedup_topic_files(dir)->int`;`_topic_merge_candidates(dir)->list[list[str]]`;`consolidate_topic_files(dir)->int`;`_topics_snapshot(dir)->list[(str,int)]`。build 传 `known_topics` 由 set→带计数文本(在 build 侧组装,distill 签名不变)。

**测试惯例:** 沿用现有 `tests/` 的 monkeypatch fake client(`test_v8_build.py`/`test_v8_retrieve.py` 模式)、`tmp_path`、模块限定 patch `V8.distill_events`。旧用例契约变化时更新 fake 而非删测试。
