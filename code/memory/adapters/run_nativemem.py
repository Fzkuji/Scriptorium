"""NativeMem adapter for the unified evaluation protocol.

Contract (src/adapters/README.md): build memory + retrieve top-k memories per
question; NO answering, NO judging. The unified answerer/judge live in
src/evaluation/evaluate.py.

NativeMem specifics
-------------------
- Build: src/nativemem.py process_chunk (bash-tool RAM writer), 10-turn
  chunks per session, raw archive alongside — identical to the v3 full run.
  Model: BUILDER_MODEL/BUILDER_BASE (default deepseek-v4-flash @ Aliyun).
- Retrieve: NativeMem's navigation strategy (ls -lhR / grep '^#' / cat / sed)
  but in COLLECT mode — the model gathers relevant memory entries verbatim
  instead of answering. Entries keep their [YYYY-MM-DD] timestamps, which we
  parse into the standard {text, date} form.
- --memory-dir reuses an existing built memory folder (skips the ~20 min
  build); --sample builds fresh when --memory-dir is absent.

Usage (from project root):
    python3 src/adapters/run_nativemem.py --sample 0 \
        --memory-dir results/nativemem-v3-locomo-s0/memory \
        --output results/nativemem-v3-locomo-s0/questions.json
"""

import argparse
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _usage_tracker import tracker  # noqa: E402
tracker.install()

from src.nativemem import (client, ALIYUN_MODEL, TOOLS, execute_tool,  # noqa: E402
                           log_usage, process_chunk, save_to_raw_archive,
                           check_and_reorganize, consolidate_topics,
                           normalize_date, split_into_chunks,
                           split_into_chunks_structured, process_chunk_agent,
                           tidy_local, reorganize_library, measure_library,
                           library_grew_past_threshold, _top_level_view,
                           _count_entries_in_file)
from src.v8_memory import (build_turn_index, read_turns,  # noqa: E402
                           distill_events, write_events)
import src.v8_memory as v8_memory  # noqa: E402  (module ref: keeps monkeypatch.setattr(V8, ...) live)

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
# Retrieval cap. 20 floods weaker answerers (Qwen answers "Unknown" when
# handed 20 noisy fragments even though the answer is among them). Override
# with NATIVEMEM_TOPK for少而精 retrieval.
TOP_K = int(os.environ.get("NATIVEMEM_TOPK", "20"))
_V8_TURN_INDEX = {}  # v8 检索回原文用；main 在 build 后设置

COLLECT_PROMPT = """You are retrieving relevant memories from a personal memory folder to help answer a question.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

## Strategy — grep by keyword, don't guess file names
The memory may be organized into many nested files/folders. Do NOT try to guess which file holds the answer from its name — instead search the CONTENT of every file:

1. Pick the 1-3 key terms from the question (the specific noun/entity/activity, e.g. "sunrise", "adoption", "support group", "pets").
2. `grep -rin "<term>" .` — recursively grep ALL files for that term. This finds the entry no matter which file or subfolder it lives in. If a term gives nothing, try a synonym or a broader/shorter term (e.g. "paint" instead of "painted a sunrise").
3. Once you locate hits, read the full entry with context: `grep -rin -A2 "<term>" .` to capture the distilled line AND its `> "..."` verbatim quote underneath.
4. If the question mentions two people or spans topics, grep for each and gather from wherever they hit.

## Rules
- Do NOT answer the question. Your job is ONLY to collect the memory entries relevant to it.
- Copy each relevant entry VERBATIM, including its [YYYY-MM-DD] prefix and the `> "..."` verbatim line if present (it holds specific details the distilled line may abbreviate).
- Collect the MOST relevant entries only — quality over quantity. For a single-fact question, 1-3 tightly matching entries is better than many loosely related ones. For a list/aggregation question ("what activities...", "which bands..."), gather every matching entry. Cap: {top_k}.
- When done, output the collected entries wrapped in <memories></memories> tags.
- If truly nothing matches after trying a few keywords, output <memories></memories> empty.
"""


_NAV_PROMPT = """You are finding relevant memories to answer a question. Below is the TABLE OF CONTENTS of a person's memory — a list of topic headings under each person. Use your understanding to pick which topics likely hold the answer (a question about moving/origin → a 'background' or 'hometown'-like topic; about a book → 'reading'/'book' topics). Also give a few keywords to grep as backup.

Question: {question}

Table of contents:
{toc}

Output JSON only:
{{"topics": ["Caroline :: adoption journey", "Melanie :: pets"], "keywords": ["adoption", "Sweden"]}}
Pick the 1-5 most relevant topics by MEANING (not exact word match). Keywords are a backup for details that might not match a topic name."""


def _read_toc(memory_dir):
    """Code builds a compact table of contents: person → its ## topic headings.
    Returns (toc_text, index) where index maps 'Person :: topic' → list of
    entry blocks under that heading."""
    people_dir = os.path.join(memory_dir, "people")
    toc_lines = []
    index = {}
    if not os.path.isdir(people_dir):
        return "", index
    for fn in sorted(os.listdir(people_dir)):
        if not fn.endswith(".md"):
            continue
        person = fn[:-3]
        path = os.path.join(people_dir, fn)
        cur = None
        buf = []
        heads = []
        for ln in open(path):
            if ln.startswith("## "):
                if cur is not None:
                    index[f"{person} :: {cur}"] = buf
                cur = ln.strip()[3:].strip()
                heads.append(cur)
                buf = []
            elif cur is not None and ln.strip():
                buf.append(ln.rstrip("\n"))
        if cur is not None:
            index[f"{person} :: {cur}"] = buf
        if heads:
            toc_lines.append(f"{person}: " + " | ".join(heads))
    return "\n".join(toc_lines), index


def collect_memories_nav(question, memory_dir):
    """Navigation retrieval (v6): code builds a table of contents → model picks
    relevant topics by MEANING (one call) → code extracts those topics' entries,
    plus a keyword-grep backup. Deterministic extraction (no model-issued bash),
    so same question → same result. Uses the tidied topic structure."""
    toc, index = _read_toc(memory_dir)
    if not toc:
        return [], 0
    try:
        resp = client.chat.completions.create(
            model=ALIYUN_MODEL,
            messages=[{"role": "system",
                       "content": _NAV_PROMPT.format(question=question, toc=toc)}],
            max_tokens=500, temperature=0,
            response_format={"type": "json_object"})
        log_usage(resp, phase="collect")
        sel = json.loads(re.sub(r"<think>.*?</think>", "",
                        resp.choices[0].message.content or "", flags=re.DOTALL))
    except Exception:  # noqa: BLE001
        sel = {}
    topic_blocks = []
    kw_blocks = []
    seen = set()
    # (1) entries under the picked topics — the primary, precise signal
    for key in sel.get("topics", [])[:4]:
        key = key.strip()
        for ik in index:
            if ik.lower() == key.lower() or ik.lower().endswith(key.lower().split("::")[-1].strip()):
                for bl in index[ik]:
                    if bl not in seen and not bl.strip().startswith(">"):
                        seen.add(bl)
                        topic_blocks.append(bl)
                break
    # (2) keyword backup — only for entries a topic didn't already catch, and
    # only requiring a distinctive keyword hit. Kept SEPARATE and capped so it
    # doesn't flood precise questions.
    kws = [k for k in sel.get("keywords", []) if k and len(k) > 2][:4]
    if kws:
        for ik, bls in index.items():
            for bl in bls:
                if bl.strip().startswith(">"):
                    continue
                low = bl.lower()
                if any(k.lower() in low for k in kws) and bl not in seen:
                    seen.add(bl)
                    kw_blocks.append(bl)
    # Merge with a cap: topics first (precise), then a few keyword hits.
    # Cap total tighter than TOP_K to avoid drowning the answerer.
    cap = min(TOP_K, 10)
    blocks = topic_blocks[:cap]
    if len(blocks) < cap:
        blocks += kw_blocks[:cap - len(blocks)]
    # parse blocks into {text,date}
    mems = []
    for bl in blocks[:cap]:
        dm = re.match(r"\[(\d{4}-\d{2}-\d{2})\]\s*(.*)", bl.strip())
        if dm:
            mems.append({"text": dm.group(2).strip(), "date": dm.group(1)})
        elif bl.strip().startswith(">"):
            mems.append({"text": bl.strip(), "date": ""})
    return mems, 1


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
    # 整行就是 source 的：删行
    kept = [ln for ln in text.splitlines()
            if not re.match(r"\s*\[source\]\(.*\)\s*$", ln)]
    out = "\n".join(kept)
    # 行内尾随的 [source](...) 也去掉
    out = re.sub(r"\s*\[source\]\([^)]*\)", "", out)
    return out


def _collect_memories_v7(question, memory_dir, max_rounds=6):
    top_view = _top_level_view(memory_dir)
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
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            # hide raw/ during retrieval: same no-original 口径 rule as the
            # grep-collection path below — the retrieval agent must never
            # read the raw transcript, only the memory files.
            out = execute_tool(tc.function.name, args, memory_dir,
                                hide_raw=True)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    if os.environ.get("NATIVEMEM_NO_ORIGINAL") == "1":
        final_text = _filter_source_lines(final_text)
    return parse_memories(final_text), rounds


_V8_RETRIEVE_PROMPT = """你在一个记忆库里找答案。工作目录是记忆库根目录，只能用 bash。
记忆库有两个视图：
- topics/ ：按话题分文件（问"是什么/谁/哪里"查这里）
- timeline/年/月-英文.md ：按时间排的事件（问"什么时候/先后/多久"查这里）
每条事件是一行：[日期] 摘要 · [dia_id]。dia_id 指向原始对话。

## 记忆库当前结构（目录/文件 + 条目数），先看这张地图再决定 cat/grep 哪里
{structure}

问题：{question}

步骤：
1. 对照上面的话题清单，cat 最相关的话题文件；或用 grep 在 topics/ 里搜问题的关键名词（换几个同义词多试）。
2. 找到相关事件行后，从行末的 [D1:3, D1:5] 取出 dia_id，调 read_original(dia_ids=[...]) 读原始对话。
3. 只取最相关的几条，别贪多。读到原文后停手。"""


# 方案B：单模型——同一个模型检索完直接回答（不交给独立 answerer，上下文连贯）。
# 答题要求对齐标准 ANSWER_PROMPT：相对时间按事件行/原文的日期戳换算成绝对日期、
# 答案不超过 5-6 词、包在 <answer></answer> 里。
_V8_SINGLE_PROMPT = """你在一个记忆库里找答案并直接回答问题。工作目录是记忆库根目录，只能用 bash。
记忆库有两个视图：
- topics/ ：按话题分文件（问"是什么/谁/哪里"查这里）
- timeline/年/月-英文.md ：按时间排的事件（问"什么时候/先后/多久"查这里）
每条事件是一行：[日期] 摘要 · [dia_id]。dia_id 指向原始对话。

## 记忆库当前结构（目录/文件 + 条目数），先看这张地图再决定 cat/grep 哪里
{structure}

问题：{question}

步骤：
1. 对照话题清单 cat/grep 相关视图，定位相关事件行。
2. 需要细节时从行末 [D1:3] 取 dia_id，调 read_original 读原文（原文每句带 (日期) 说话时间戳）。
3. 信息够了就回答。相对时间（"yesterday"、"last year"）要按说话日期戳换算成具体日期/年份。
   答案不超过 5-6 个词，包在 <answer></answer> 里。找不到就 <answer>Not mentioned</answer>。"""


def _v8_topic_list(memory_dir):
    """DEPRECATED（v8 块一起被 `_v8_structure_map` 取代，保留仅防旧引用）：
    只 listdir(topics/)、平铺全部文件名、不含 timeline/。新检索一律用 _v8_structure_map。"""
    topics_dir = os.path.join(memory_dir, "topics")
    if not os.path.isdir(topics_dir):
        return "（暂无话题文件）"
    names = sorted(fn[:-3] for fn in os.listdir(topics_dir) if fn.endswith(".md"))
    return "\n".join(f"- {n}" for n in names) if names else "（暂无话题文件）"


def _v8_structure_map(memory_dir, mode=None):
    """递归 walk 整棵记忆树，如实呈现目录树 + 每目录/文件的条目计数，当检索"地图"。
    不 hardcode 任何目录名——walk 出什么呈现什么（含 timeline/ 与模型自建子目录）。

    mode（缺省读 env NATIVEMEM_V8_MAP，默认 dir）：
    - dir  ：每个目录把其直属 .md 文件折叠成一行汇总 `name/ (K files, N entries)`，
             不平铺文件名；子目录继续递归。地图恒小。
    - files：逐文件列 basename + [N 条]；但某目录直属 .md 超 NATIVEMEM_V8_MAP_INLINE
             （默认 8）个时仍折叠成汇总，避免宽目录炸 prompt。
    空库返回 (empty memory)。跳过 raw/ 与隐藏文件（与 measure_library 一致）。
    """
    if mode is None:
        mode = os.environ.get("NATIVEMEM_V8_MAP", "dir")
    inline = int(os.environ.get("NATIVEMEM_V8_MAP_INLINE", "8"))
    if not os.path.isdir(memory_dir):
        return "(empty memory)"

    def _walk(d, depth):
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            return []
        subdirs, mdfiles = [], []
        for name in entries:
            if name.startswith(".") or name == "raw":
                continue
            path = os.path.join(d, name)
            if os.path.isdir(path):
                subdirs.append(name)
            elif name.endswith(".md"):
                mdfiles.append(name)
        pad = "  " * depth
        out = []
        # 直属 .md 文件：dir 模式一律折叠；files 模式超阈值才折叠，否则逐文件列。
        if mdfiles:
            if mode == "dir" or len(mdfiles) > inline:
                total = sum(_count_entries_in_file(os.path.join(d, fn)) for fn in mdfiles)
                unit = "file" if len(mdfiles) == 1 else "files"
                out.append(f"{pad}({len(mdfiles)} {unit}, {total} entries)")
            else:
                for fn in mdfiles:
                    n = _count_entries_in_file(os.path.join(d, fn))
                    out.append(f"{pad}{fn[:-3]} [{n}]")
        for sub in subdirs:
            child = _walk(os.path.join(d, sub), depth + 1)
            sub_total = sum(_count_entries_in_file(os.path.join(r, fn))
                            for r, _ds, fs in os.walk(os.path.join(d, sub))
                            if "raw" not in r.split(os.sep)
                            for fn in fs if fn.endswith(".md"))
            out.append(f"{pad}{sub}/ [{sub_total}]")
            out.extend(child)
        return out

    lines = _walk(memory_dir, 0)
    return "\n".join(lines) if lines else "(empty memory)"

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


_V8_DIA_ID_RE = re.compile(r"D\d+:\d+")
_V8_EVENT_DATE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]")


def _v8_event_lines(memory_dir, dia_ids):
    """反查含这些 dia_id 的事件行，返回整行文本（[日期] 摘要 · [dia_id]）。
    去重保序。给 answerer 的是提炼过的精准摘要，比原始 turn 少噪声。"""
    if not dia_ids:
        return ""
    want = set(dia_ids)
    lines, seen = [], set()
    for root, _dirs, files in os.walk(memory_dir):
        if "raw" in root.split(os.sep):
            continue
        for fn in files:
            if not fn.endswith(".md"):
                continue
            try:
                with open(os.path.join(root, fn)) as f:
                    for line in f:
                        ln = line.rstrip("\n")
                        if not _V8_EVENT_DATE_RE.match(ln):
                            continue
                        if want & set(_V8_DIA_ID_RE.findall(ln)) and ln not in seen:
                            seen.add(ln)
                            lines.append(ln)
            except OSError:
                continue
    return "\n".join(lines)


def _collect_v8(question, memory_dir, turn_index, max_rounds=8):
    prompt = _V8_RETRIEVE_PROMPT.format(
        question=question, structure=_v8_structure_map(memory_dir))
    messages = [{"role": "user", "content": prompt}]
    tools = TOOLS + [_V8_READ_TOOL]
    collected, steps = [], 0
    grep_seen_ids, grep_seen_set = [], set()   # bash 输出里出现过的 dia_id（兜底用）
    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL, messages=messages, tools=tools,
                max_tokens=1200, temperature=0.0)
        except Exception:  # noqa: BLE001
            # API 异常（内容过滤等）：用目前已收集/grep 到的兜底收尾，别整题崩
            break
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
                    # 方案A：先放提炼过的事件行本身（精准，含绝对日期+摘要），
                    # 再接原文补充细节。检索模型 grep 时看到的那条精准事件行
                    # 之前只用来取 dia_id 就丢了、没传给 answerer——现在一起喂。
                    ev_lines = _v8_event_lines(memory_dir, dids)
                    block = (ev_lines + "\n" + original) if ev_lines else original
                    collected.append(block)
                out = original or "(无对应原文)"
            else:
                out = execute_tool(tc.function.name, args, memory_dir, hide_raw=False)
                # 兜底：记下 bash（grep/cat 视图）输出里出现的 dia_id。弱模型可能
                # grep 命中了事件行却忘了调 read_original 就收尾——那时用这些 id
                # 兜底回原文，不让已找到的结果因为漏一步而白丢。
                for did in _V8_DIA_ID_RE.findall(out or ""):
                    if did not in grep_seen_set:
                        grep_seen_set.add(did)
                        grep_seen_ids.append(did)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    # 模型没主动 read_original 收到任何东西，但 grep 输出里有 dia_id → 兜底回原文
    if not collected and grep_seen_ids:
        fallback = read_turns(turn_index, grep_seen_ids, context=1)
        if fallback.strip():
            ev_lines = _v8_event_lines(memory_dir, grep_seen_ids)
            collected.append((ev_lines + "\n" + fallback) if ev_lines else fallback)
    memories = [{"text": t, "date": ""} for t in collected]
    return memories, steps


_V8_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _collect_and_answer_v8(question, memory_dir, turn_index, max_rounds=8):
    """方案B：单模型——同一个模型检索+直接回答，上下文连贯，不交给独立 answerer。
    返回 (memories, steps, answer)。answer 从模型最终 <answer> 抽取。"""
    prompt = _V8_SINGLE_PROMPT.format(
        question=question, structure=_v8_structure_map(memory_dir))
    messages = [{"role": "user", "content": prompt}]
    tools = TOOLS + [_V8_READ_TOOL]
    collected, steps, final_text = [], 0, ""
    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL, messages=messages, tools=tools,
                max_tokens=1200, temperature=0.0)
        except Exception:  # noqa: BLE001
            break
        log_usage(resp, phase="v8_single")
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
            if tc.function.name == "read_original":
                dids = args.get("dia_ids", [])
                if isinstance(dids, str):
                    dids = [dids]
                original = read_turns(turn_index, dids, context=1)
                ev_lines = _v8_event_lines(memory_dir, dids)
                block = (ev_lines + "\n" + original) if ev_lines else original
                if block.strip():
                    collected.append(block)
                out = original or "(无对应原文)"
            else:
                out = execute_tool(tc.function.name, args, memory_dir, hide_raw=False)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    m = _V8_ANSWER_RE.search(final_text or "")
    answer = m.group(1).strip() if m else (final_text or "").strip()
    memories = [{"text": t, "date": ""} for t in collected]
    return memories, steps, answer


def collect_memories(question, memory_dir, max_rounds=10):
    """Navigate the memory folder and collect relevant entries (no answering).

    Returns (memories: [{text, date}], steps: int).
    """
    if os.environ.get("NATIVEMEM_PROMPT") == "v8":
        return _collect_v8(question, memory_dir, _V8_TURN_INDEX, max_rounds=8)
    if os.environ.get("NATIVEMEM_PROMPT") == "v7":
        return _collect_memories_v7(question, memory_dir, max_rounds=max_rounds)
    if os.environ.get("NATIVEMEM_RETRIEVAL", "nav" if os.environ.get("NATIVEMEM_PROMPT") == "v6" else "grep") == "nav":
        return collect_memories_nav(question, memory_dir)
    messages = [
        {"role": "system", "content": COLLECT_PROMPT.format(top_k=TOP_K)},
        {"role": "user", "content": f"Question: {question}"},
    ]
    trace = []
    final_text = ""
    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL, messages=messages,
                tools=TOOLS, max_tokens=2000, temperature=0)
        except Exception:  # noqa: BLE001
            time.sleep(3)
            continue
        log_usage(resp, phase="collect")
        msg = resp.choices[0].message
        if msg.content:
            msg.content = re.sub(r"<think>.*?</think>", "", msg.content,
                                 flags=re.DOTALL).strip()
        if not msg.tool_calls:
            final_text = msg.content or ""
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:  # noqa: BLE001
                args = {}
            trace.append({"tool": tc.function.name, "args": args})
            # hide raw/ during retrieval: the answerer must read MEMORY, never
            # the original transcript (no-original口径 integrity).
            result = execute_tool(tc.function.name, args, memory_dir,
                                  hide_raw=True)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})
    return parse_memories(final_text), len(trace)


def parse_memories(text):
    """<memories> block → [{text, date}]; date from [YYYY-MM-DD] prefix."""
    m = re.search(r"<memories>(.*?)</memories>", text, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else text
    memories = []
    for line in body.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith("<"):
            continue
        dm = re.match(r"\[(\d{4}-\d{2}-\d{2})\]\s*(.*)", line)
        if dm:
            memories.append({"text": dm.group(2).strip(), "date": dm.group(1)})
        else:
            memories.append({"text": line, "date": ""})
    return memories[:TOP_K]


def build_memory(conv, memory_dir, max_sessions=None):
    """Replicates src/nativemem.py main(): 10-turn chunks, raw archive."""
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
        # 小块提炼：默认 6 句/块（而非一整个 session）。块小模型注意力集中、
        # 漏事实少（build 覆盖率是失分主因）；后端是大模型，多几次调用可接受。
        # 模型靠累积的 known_topics 看"以前的记忆"，小块间上下文不断裂。
        v8_chunk = int(os.environ.get("NATIVEMEM_CHUNK_TURNS", "6"))
        for session, date in zip(sessions, dates):
            obs = normalize_date(date)
            # structured per-turn chunks: numbering is 1:1 with dia_ids by
            # construction (a turn's internal blank line can't shift block nums).
            for turns, dia_ids in split_into_chunks_structured(session, v8_chunk):
                if not turns:
                    continue
                events = v8_memory.distill_events(turns, obs, dia_ids,
                                        known_topics=sorted(known_topics))
                if not events:
                    continue
                write_events(memory_dir, events)
                for e in events:
                    known_topics.add(e.get("topic", "misc"))
                n_events += len(events)
        return _t.time() - t0, n_events

    raw_dir = os.path.join(memory_dir, "raw")
    os.makedirs(memory_dir, exist_ok=True)
    # v7 记忆库无原文归档（模型自组织，raw/ 从不写入）；只有 legacy 分支
    # 用 save_to_raw_archive 写 raw/。v7 下不建空 raw/，免得留 phantom 目录
    # 干扰读者与 measure_library。
    if os.environ.get("NATIVEMEM_PROMPT") != "v7":
        os.makedirs(raw_dir, exist_ok=True)
    sessions, dates = [], []
    i = 1
    while f"session_{i}" in conv:
        sessions.append(conv[f"session_{i}"])
        dates.append(conv.get(f"session_{i}_date_time", ""))
        i += 1
    if max_sessions:
        sessions, dates = sessions[:max_sessions], dates[:max_sessions]

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
        # 切块粒度：LoCoMo 每 session 最长仅 47 句 / ~1300 token，默认 50
        # 让「一个 session = 一块」，块数减半、build 调用大幅下降，且模型一次
        # 看到完整 session 上下文再决定怎么存（质量更好）。超长 session（>50 句）
        # 仍会自动切成多块。
        chunk_turns = int(os.environ.get("NATIVEMEM_CHUNK_TURNS", "50"))
        chunk_idx = 0
        for si, (session, date) in enumerate(zip(sessions, dates)):
            obs = normalize_date(date)
            session_summary = ""
            for chunk_text, dia_ids in split_into_chunks(session, chunk_turns):
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

    # Periodic topic consolidation (design §B): merge near-synonym headings
    # every N sessions so structure stays clean as we write ("write a bit,
    # tidy a bit"). NATIVEMEM_CONSOLIDATE=off disables; default every 1 session.
    consol_every = os.environ.get("NATIVEMEM_CONSOLIDATE", "1")
    do_consol = consol_every != "off"
    consol_n = int(consol_every) if consol_every.isdigit() else 1
    total_merges = 0

    def _tidy():
        nonlocal total_merges
        # recurse over every .md (model may nest into subdirs), skip raw archive
        for root, _dirs, files in os.walk(memory_dir):
            if "raw" in root.split(os.sep):
                continue
            for fn in files:
                if fn.endswith(".md"):
                    total_merges += consolidate_topics(os.path.join(root, fn))

    t0 = time.time()
    chunk_idx = 0
    for si, (session, date) in enumerate(zip(sessions, dates)):
        turns = session if isinstance(session, list) else []
        for cs in range(0, len(turns), 10):
            chunk = turns[cs:cs + 10]
            chunk_text = "".join(
                f"{t.get('speaker', 'user')}: {t.get('text', '')}\n\n"
                for t in chunk if isinstance(t, dict))
            if not chunk_text.strip():
                continue
            chunk_idx += 1
            source_file = save_to_raw_archive(chunk_text, date, si, chunk_idx, raw_dir)
            print(f"  build chunk {chunk_idx} (session {si+1}/{len(sessions)})")
            process_chunk(chunk_text, date, memory_dir, f"raw/{source_file}")
        # tidy after every consol_n sessions (write-a-bit, tidy-a-bit)
        if do_consol and (si + 1) % consol_n == 0:
            _tidy()
    if do_consol:
        _tidy()  # final pass
        print(f"  consolidation: {total_merges} heading merges total")
    return time.time() - t0, chunk_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--memory-dir", default=None,
                    help="reuse an existing built memory folder (skips build)")
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--questions-limit", type=int, default=None)
    args = ap.parse_args()

    with open(DATA_PATH) as f:
        data = json.load(f)
    sample = data[args.sample]
    conv = sample["conversation"]
    qas = sample["qa"]
    if args.questions_limit:
        qas = qas[: args.questions_limit]

    build_time, n_chunks = 0.0, 0
    build_snap = None
    if args.memory_dir:
        memory_dir = args.memory_dir
        notes = f"reused existing memory dir: {memory_dir}"
        print(f"[nativemem] {notes}")
    else:
        memory_dir = os.path.join(
            os.path.dirname(args.output) or ".", f"memory_sample{args.sample}")
        print(f"[nativemem] building memory -> {memory_dir}")
        tracker.reset("build")
        build_time, n_chunks = build_memory(conv, memory_dir,
                                            max_sessions=args.max_sessions)
        build_snap = tracker.snapshot("build")
        notes = f"built fresh: {n_chunks} chunks, model={ALIYUN_MODEL}"
        print(f"[nativemem] build done: {build_time:.0f}s")

    if os.environ.get("NATIVEMEM_PROMPT") == "v8":
        global _V8_TURN_INDEX
        _V8_TURN_INDEX = build_turn_index(conv)

    n_files = sum(1 for root, _, files in os.walk(memory_dir)
                  for fn in files if fn.endswith(".md") and "raw" not in root)

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 1),
        "num_memories": n_files,
        "notes": notes,
        "build_calls": build_snap["calls"] if build_snap else None,
        "build_tokens_in": build_snap["tokens_in"] if build_snap else None,
        "build_tokens_out": build_snap["tokens_out"] if build_snap else None,
        "build_llm_time_s": build_snap["llm_time_s"] if build_snap else None,
    }]
    for qi, qa in enumerate(qas):
        q = qa["question"]
        gold = str(qa.get("answer", qa.get("adversarial_answer", "")))
        t0 = time.time()
        tracker.reset("q")
        single_v8 = (os.environ.get("NATIVEMEM_PROMPT") == "v8"
                     and os.environ.get("NATIVEMEM_V8_SINGLE") == "1")
        v8_answer = None
        try:
            if single_v8:
                # 方案B：单模型检索+直接回答
                memories, steps, v8_answer = _collect_and_answer_v8(
                    q, memory_dir, _V8_TURN_INDEX)
            else:
                memories, steps = collect_memories(q, memory_dir)
        except Exception as _e:  # noqa: BLE001
            # 单题遇 API 异常（阿里云内容过滤 data_inspection_failed 等）不能
            # 拖垮整轮 199 题评测——记空、继续下一题。
            print(f"  q{qi}: SKIPPED ({type(_e).__name__}: {str(_e)[:60]})")
            memories, steps = [], 0
        snap = tracker.snapshot("q")
        latency = time.time() - t0
        _rec = {
            "question_id": f"s{args.sample}_q{qi}",
            "question": q,
            "gold": gold,
            "category": qa.get("category"),
            "memories": memories,
            "retrieval": {"latency_s": round(latency, 2), "k": TOP_K,
                          "steps": steps,
                          "calls": snap["calls"],
                          "tokens_in": snap["tokens_in"],
                          "tokens_out": snap["tokens_out"]},
        }
        if v8_answer is not None:
            # 方案B：单模型已直接给答案，写进 answer 字段，eval_full 跳过独立 answerer
            _rec["answer"] = v8_answer
        records.append(_rec)
        print(f"  q{qi}: {len(memories)} memories, {steps} steps, {latency:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[nativemem] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
