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
from concurrent.futures import ThreadPoolExecutor, as_completed


def _v8_concurrency(value=8):
    return max(1, int(value))


def _v8_tidy(
    memory_dir,
    touched,
    sections_on,
    article_on,
    final=False,
    *,
    concurrency=8,
    combined=False,
):
    """节奏3/收尾整理。TIDY_COMBINED=off（默认）走 merge_duplicate_lines +
    organize_topic_sections/rewrite_topic_articles 两次独立调用——v8.6 验尸发现
    合并成一次调用时模型把去重和分节张冠李戴（抽查 10 例合并仅 1 例正确、
    引用编号错位），故默认回退两次调用，合并调用留作消融（=on 走 tidy_topic_file）。
    多文件时用 ThreadPoolExecutor 并发发（每文件 LLM 调用 + 独立文件写，文件间无
    共享可变状态；log_usage 已加锁），并发数 NATIVEMEM_V8_CONCURRENCY（默认 8）。"""
    if not touched:
        return
    def _one(name):
        s = {name}
        if combined:
            v8_memory.tidy_topic_file(memory_dir, s, final=final)
            return
        v8_memory.merge_duplicate_lines(memory_dir, s, final=final)
        if sections_on:
            v8_memory.organize_topic_sections(memory_dir, s, final=final)
        elif article_on:
            v8_memory.rewrite_topic_articles(memory_dir, s, final=final)

    names = sorted(touched)
    workers = min(_v8_concurrency(concurrency), len(names))
    if workers <= 1:
        for name in names:
            _one(name)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, names))

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
                           distill_events, write_events, dia_ids_in)
import src.v8_memory as v8_memory  # noqa: E402  (module ref: keeps monkeypatch.setattr(V8, ...) live)
import src.v10_memory as v10_memory  # noqa: E402
import src.nativemem as nativemem_runtime  # noqa: E402
from src.adapters.question_checkpoint import (  # noqa: E402
    atomic_json,
    checkpoint_identity,
    checkpoint_path,
    completed_answers,
    load_or_create,
    mark_complete,
    save_answer,
)
from src.nativemem_versions import load_version  # noqa: E402
from src.memory_bm25 import MemoryBM25Index, render_search_results  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")


def load_source_build_record(path, memory_dir, sample, version):
    """Load and validate provenance for a reused versioned memory."""
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("source build record must contain exactly one record")
    record = records[0]
    expected_method = f"NativeMem-{version}"
    if (
        record.get("question_id") != "_build_stats"
        or record.get("method") != expected_method
    ):
        raise ValueError(f"source build record is not a {expected_method} build")
    if int(record.get("sample", -1)) != sample:
        raise ValueError("source build record sample does not match --sample")
    source_memory = os.path.realpath(str(record.get("memory_dir", "")))
    requested_memory = os.path.realpath(memory_dir)
    if source_memory != requested_memory:
        raise ValueError("source build record memory_dir does not match --memory-dir")
    config_key = f"{version}_config"
    if not record.get("builder_model") or not record.get(config_key):
        raise ValueError(f"source build record is missing {version} provenance")
    return dict(record)


def load_v10_source_build_record(path, memory_dir, sample):
    """Compatibility wrapper used by existing V10 tests and scripts."""
    return load_source_build_record(path, memory_dir, sample, "v10")
# Retrieval cap. 20 floods weaker answerers (Qwen answers "Unknown" when
# handed 20 noisy fragments even though the answer is among them). Override
# with NATIVEMEM_TOPK for少而精 retrieval.
TOP_K = int(os.environ.get("NATIVEMEM_TOPK", "20"))
_V8_TURN_INDEX = {}  # v8 检索回原文用；main 在 build 后设置


def _uses_linked_memory():
    """Return whether the selected version uses topic/timeline memory views."""
    return os.environ.get("NATIVEMEM_PROMPT") in {"v8", "v10"}

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


_V8_RETRIEVE_PROMPT = """你在一个记忆库里找答案。你是主检索器，可以在同一上下文中自主组合 bash、bm25_search 和 read_original。
记忆库有两个视图：
- topics/ ：按话题分文件、门内可能分子目录（问"是什么/谁/哪里"查这里）。每个文件是一篇**带内联引用的文章**：句子写到哪件事，[Dx:y] 引用就嵌在哪（也可能还有未整理的 [日期] 开头的句子）。
- timeline/年/月/日.md ：一天一文件，路径是纯数字 timeline/YYYY/MM/DD.md（月、日两位零填充，如 2023/07/07.md 对应 2023-07-07），按时间排的原子事件行 `[日期] 摘要 · [dia_id]`（问"什么时候/先后/多久"查这里）。`ls timeline/2023/05/` 能看到那月有记录的是哪几天；**时间类问题别只看单天文件，用 `cat timeline/2023/08/*.md` 一把读整月**（天文件很小，漏读一天就漏一件事），跨月就多 cat 几个月。
[Dx:y] 一律指向原始对话，可用 read_original 取原文。引用锚右半可能是区间 [D2:3-15] 或列举 [D2:2,10] 形式（一件事聊了连续多句就写成区间）——直接把整个字符串原样传给 read_original 即可，无需自己拆开。
topics 文中的日期是可跳转链接（如 [2023-05-25](timeline/2023/05/25.md)）——cat 那个 timeline 天文件能看到那天发生了什么，`ls` 同月文件夹再 cat 邻近天，做时间邻域类问题。
timeline 行尾的 → topics/....md 是这条事件所属完整文章的路径——命中时间线后 cat 它拿上下文。

## 记忆库当前结构（目录/文件 + 条目数）
{structure}

## 工具选择
- 路径明确时用 ls/cat 阅读相关 topic 和兄弟文件。
- 查专名、原句、编号时用 grep；一个关键词没命中时换多种措辞。
- 路径不明确、问题包含多个概念、或 grep 没命中时调用 bm25_search；它只做无 embedding 的关键词排序，可改写 query 后再次调用。
- 问题明确给出日历范围时，可在同一次 bm25_search 调用中传 date_from/date_to；支持 YYYY、YYYY-MM、YYYY-MM-DD。时间含义尚未解析清楚时不要使用硬过滤。
- 时间、先后、持续时间和状态变化问题查 timeline。
- 找到 [Dx:y] 后用 read_original 核验原词、日期、指代和冲突；通用 bash 看不到 sources/。
这些工具没有固定顺序。根据每轮结果决定继续搜索、换工具或停止。

问题：{question}

检索纪律（自适应选择工具，不强制固定流水线）：
1. 如果结构地图已经给出明确方向，按语义判断哪 1-3 个话题文件最可能有答案，把它们整个 cat 读全文。
   摘要的用词往往和问题不同（问"pet"可能写成"guinea pig"，问"accident"可能写成"scared/reassured"），
   所以别指望字面对上——靠语义挑文件、读全文，让答案自己浮出来。
   **同一人物名下有多个 topic 文件时（如 Caroline/adoption、Caroline/LGBTQ_support），先 ls 把该人物目录下所有文件列全，
   把和问题貌似相关的文件全部 cat 读完再选答案，不许读一个就答**（答案常散在兄弟文件里）。
2. 词法工具用于跨文件召回：精确短语优先 grep，路径不明确或多概念问题优先 bm25_search；必要时换多种不同措辞多试，
   别指望一个关键词打中，也不要把 BM25 排名直接当成最终答案。
3. 定位到候选事件后，把所有和问题沾边的行都列出来，凡是可能相关的行末 [D1:3, D1:5] dia_id
   全部取出，一次 read_original(dia_ids=[...]) 传进去，不许只挑一条相邻行就回原文。
4. 锁定候选答案后，**回 timeline 用人物名+日期 grep -r timeline/ 交叉核对一次**（天文件散在月份文件夹里，要递归 grep），以 timeline 原子行和原文轮次的原词为准，
   文章散文只当导航（散文可能改错日期或抹平原词，不足为凭）。核对完把相关内容都带回。"""


# 方案B：单模型——同一个模型检索完直接回答（不交给独立 answerer，上下文连贯）。
# 答题要求对齐标准 ANSWER_PROMPT：相对时间按事件行/原文的日期戳换算成绝对日期、
# 答案不超过 5-6 词、包在 <answer></answer> 里；反弃答——找不到也给最佳猜测，不许 Not mentioned。
_V8_SINGLE_PROMPT = """你在一个记忆库里找答案并直接回答问题。你是主检索器，可以在同一上下文中自主组合 bash、bm25_search 和 read_original。
记忆库有两个视图：
- topics/ ：按话题分文件、门内可能分子目录（问"是什么/谁/哪里"查这里）。每个文件是一篇**带内联引用的文章**：句子写到哪件事，[Dx:y] 引用就嵌在哪（也可能还有未整理的 [日期] 开头的句子）。
- timeline/年/月/日.md ：一天一文件，路径是纯数字 timeline/YYYY/MM/DD.md（月、日两位零填充，如 2023/07/07.md 对应 2023-07-07），按时间排的原子事件行 `[日期] 摘要 · [dia_id]`（问"什么时候/先后/多久"查这里）。`ls timeline/2023/05/` 能看到那月有记录的是哪几天；**时间类问题别只看单天文件，用 `cat timeline/2023/08/*.md` 一把读整月**（天文件很小，漏读一天就漏一件事），跨月就多 cat 几个月。
[Dx:y] 一律指向原始对话，可用 read_original 取原文。引用锚右半可能是区间 [D2:3-15] 或列举 [D2:2,10] 形式（一件事聊了连续多句就写成区间）——直接把整个字符串原样传给 read_original 即可，无需自己拆开。
topics 文中的日期是可跳转链接（如 [2023-05-25](timeline/2023/05/25.md)）——cat 那个 timeline 天文件能看到那天发生了什么，`ls` 同月文件夹再 cat 邻近天，做时间邻域类问题。
timeline 行尾的 → topics/....md 是这条事件所属完整文章的路径——命中时间线后 cat 它拿上下文。

## 记忆库当前结构（目录/文件 + 条目数）
{structure}

## 工具选择
- 路径明确时用 ls/cat 阅读相关 topic 和兄弟文件。
- 查专名、原句、编号时用 grep；一个关键词没命中时换多种措辞。
- 路径不明确、问题包含多个概念、或 grep 没命中时调用 bm25_search；它只做无 embedding 的关键词排序，可改写 query 后再次调用。
- 问题明确给出日历范围时，可在同一次 bm25_search 调用中传 date_from/date_to；支持 YYYY、YYYY-MM、YYYY-MM-DD。时间含义尚未解析清楚时不要使用硬过滤。
- 时间、先后、持续时间和状态变化问题查 timeline。
- 找到 [Dx:y] 后用 read_original 核验原词、日期、指代和冲突；通用 bash 看不到 sources/。
这些工具没有固定顺序。根据每轮结果决定继续搜索、换工具或停止。

问题：{question}

检索纪律（自适应选择工具，不强制固定流水线）：
1. 如果结构地图已经给出明确方向，按语义判断哪 1-3 个话题文件最可能有答案，把它们整个 cat 读全文。
   摘要的用词往往和问题不同（问"pet"可能写成"guinea pig"，问"accident"可能写成"scared/reassured"），
   所以别指望字面对上——靠语义挑文件、读全文，让答案自己浮出来。
   **同一人物名下有多个 topic 文件时（如 Caroline/adoption、Caroline/LGBTQ_support），先 ls 把该人物目录下所有文件列全，
   把和问题貌似相关的文件全部 cat 读完再选答案，不许读一个就答**（答案常散在兄弟文件里）。
2. 词法工具用于跨文件召回：精确短语优先 grep，路径不明确或多概念问题优先 bm25_search；必要时换多种不同措辞多试，
   别指望一个关键词打中，也不要把 BM25 排名直接当成最终答案。
3. 定位到候选事件后，把所有和问题沾边的行都列出来，凡是可能相关的行末 [D1:3] dia_id
   全部取出，一次 read_original(dia_ids=[...]) 传进去读原文（原文每句带 (日期) 说话时间戳），
   不许只挑一条相邻行。答前对照问题逐条核对读到的内容。
4. 锁定候选答案后，**回 timeline 用人物名+日期 grep -r timeline/ 交叉核对一次**（天文件散在月份文件夹里，要递归 grep），以 timeline 原子行和原文轮次的原词为准，
   文章散文只当导航（散文可能改错日期或抹平原词，不足为凭）。
   相对时间（"yesterday"、"last year"）要按说话日期戳换算成具体日期/年份。
   答案不超过 5-6 个词，包在 <answer></answer> 里。
   只有证据充分时才给具体答案；如果记忆中没有相关证据或问题前提不成立，明确回答 "Not mentioned in the conversation"。"""


def _topics_snapshot(memory_dir):
    """topics/ 子树里每个话题文件的 (相对路径名, 条目数)，按条目数降序（大话题
    排前，让 distill 优先复用活跃路径）。递归任意深度；name 相对 topics/，如
    `Caroline/adoption` 或一级的 `adoption`。目录不存在返回 []。"""
    snap = [(name, _count_entries_in_file(
                os.path.join(memory_dir, "topics", *name.split("/")) + ".md"))
            for name in v8_memory._topics_dir_files(memory_dir)]
    snap.sort(key=lambda kv: (-kv[1], kv[0]))
    return snap


def _format_topics_snapshot(snapshot):
    """把 (name, count) 清单格式化成传给 distill 的 known_topics 列表：
    每项 'name (count)'，让模型看得到已有 topic 的规模分布、优先复用。"""
    return [f"{name} ({count})" for name, count in snapshot]


def _reload_known_topics(memory_dir):
    """当前 topics/ 里的 topic 名字集（合并后重建，避免模型复用已删名）。"""
    return set(v8_memory._topics_dir_files(memory_dir))


def _v8_structure_map(memory_dir, mode=None):
    """递归 walk 整棵记忆树，如实呈现目录树 + 每目录/文件的条目计数，当检索"地图"。
    不 hardcode 任何目录名——walk 出什么呈现什么（含 timeline/ 与模型自建子目录）。

    mode（缺省读 env NATIVEMEM_V8_MAP，默认 dir）：
    - dir  ：每个目录把其直属 .md 文件折叠成一行汇总 `name/ (K files, N entries)`，
             不平铺文件名；子目录继续递归。地图恒小。
    - files：逐文件列 basename + [N 条]；但某目录直属 .md 超 NATIVEMEM_V8_MAP_INLINE
             （默认 8）个时仍折叠成汇总，避免宽目录炸 prompt。
    空库返回 (empty memory)。跳过 raw/、sources/ 与隐藏文件。
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
            if name.startswith(".") or name in {"raw", "sources"}:
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
                            if not {"raw", "sources"} & set(r.split(os.sep))
                            for fn in fs if fn.endswith(".md"))
            out.append(f"{pad}{sub}/ [{sub_total}]")
            if sub == "timeline" and depth == 0:
                out.append(f"{pad}  (路径模板 timeline/YYYY/MM/DD.md：三层依次是年/月/日，月日两位数字)")
            out.extend(child)
        return out

    lines = _walk(memory_dir, 0)
    return "\n".join(lines) if lines else "(empty memory)"


# Version-neutral interface for current runners. Historical callers still use
# the original name above.
memory_structure_map = _v8_structure_map

_V8_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_original",
        "description": "读事件对应的原始对话。dia_ids 从事件行末尾的 [D1:3, D1:5] 里取；锚可能是区间/列举形式，原样传即可。",
        "parameters": {
            "type": "object",
            "properties": {
                "dia_ids": {"type": "array", "items": {"type": "string"},
                            "description": "要回原文的 dia_id 列表，元素可含区间/列举写法，如 [\"D1:3\"]、[\"D2:3-15\"]、[\"D2:2,10\"]"}
            },
            "required": ["dia_ids"],
        },
    },
}

_V8_BM25_TOOL = {
    "type": "function",
    "function": {
        "name": "bm25_search",
        "description": (
            "对 topics/ 中的记忆事件做本地关键词排序。路径不明确、问题包含多个概念、"
            "或 grep 没命中时使用；这是无 embedding 的候选召回工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词，可改写后重复调用"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
                "path_prefix": {"type": "string", "description": "可选的 topics/ 相对路径前缀"},
                "date_from": {"type": "string", "description": "可选且包含边界的起始时间：YYYY、YYYY-MM 或 YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "可选且包含边界的结束时间：YYYY、YYYY-MM 或 YYYY-MM-DD"},
            },
            "required": ["query"],
        },
    },
}


def _run_bm25_search(memory_dir, args):
    query = str(args.get("query", "")).strip()
    if not query:
        return "No BM25 matches: query is empty."
    try:
        top_k = max(1, min(int(args.get("top_k", 8)), 50))
    except (TypeError, ValueError):
        top_k = 8
    results = MemoryBM25Index(memory_dir).search(
        query,
        top_k=top_k,
        path_prefix=args.get("path_prefix") or None,
        date_from=args.get("date_from") or None,
        date_to=args.get("date_to") or None,
        rerank=os.environ.get("NATIVEMEM_BM25_RERANK", "on") != "off",
    )
    return render_search_results(results)


def _v8_event_lines(memory_dir, dia_ids):
    """反查含这些 dia_id 的行，返回整行文本。去重保序。
    topics 文章化后引用内联在任意文体的句子里，所以**任何**含该 dia_id 的行都算
    命中（不再要求行首日期前缀）；timeline 原子行天然兜底。行内引用可能是区间/列举
    写法（[D2:3-15]），两边都先展开成单句集合再取交集。"""
    if not dia_ids:
        return ""
    want = set()
    for d in dia_ids:
        want.update(dia_ids_in(d))
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
                        if want & set(dia_ids_in(ln)) and ln not in seen:
                            seen.add(ln)
                            lines.append(ln)
            except OSError:
                continue
    return "\n".join(lines)


memory_event_lines = _v8_event_lines


def _collect_v8(question, memory_dir, turn_index, max_rounds=None):
    if max_rounds is None:
        max_rounds = int(os.environ.get("NATIVEMEM_V8_MAX_ROUNDS", "12"))
    max_tokens = int(os.environ.get("NATIVEMEM_V8_MAX_TOKENS", "1200"))
    read_ctx = int(os.environ.get("NATIVEMEM_V8_READ_CONTEXT", "1"))
    prompt = _V8_RETRIEVE_PROMPT.format(
        question=question, structure=_v8_structure_map(memory_dir))
    messages = [{"role": "user", "content": prompt}]
    tools = TOOLS + [_V8_READ_TOOL]
    if os.environ.get("NATIVEMEM_BM25", "off") == "on":
        tools.append(_V8_BM25_TOOL)
    collected, steps = [], 0
    grep_seen_ids, grep_seen_set = [], set()   # bash 输出里出现过的 dia_id（兜底用）
    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL, messages=messages, tools=tools,
                max_tokens=max_tokens, temperature=0.0)
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
                original = read_turns(turn_index, dids, context=read_ctx)
                if original.strip():
                    # 方案A：先放提炼过的事件行本身（精准，含绝对日期+摘要），
                    # 再接原文补充细节。检索模型 grep 时看到的那条精准事件行
                    # 之前只用来取 dia_id 就丢了、没传给 answerer——现在一起喂。
                    ev_lines = _v8_event_lines(memory_dir, dids)
                    block = (ev_lines + "\n" + original) if ev_lines else original
                    collected.append(block)
                out = original or "(无对应原文)"
            elif tc.function.name == "bm25_search":
                out = _run_bm25_search(memory_dir, args)
            else:
                # 通用 shell 只能查看 Abstract Memory；原文必须经 read_original。
                out = execute_tool(tc.function.name, args, memory_dir, hide_raw=True)
            if tc.function.name != "read_original":
                # 兜底：记下 bash/BM25 输出里出现的 dia_id。弱模型可能
                # 找到事件却忘了调 read_original 就收尾。
                for did in dia_ids_in(out or ""):
                    if did not in grep_seen_set:
                        grep_seen_set.add(did)
                        grep_seen_ids.append(did)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    # 模型没主动 read_original 收到任何东西，但 grep 输出里有 dia_id → 兜底回原文
    if not collected and grep_seen_ids:
        fallback = read_turns(turn_index, grep_seen_ids, context=read_ctx)
        if fallback.strip():
            ev_lines = _v8_event_lines(memory_dir, grep_seen_ids)
            collected.append((ev_lines + "\n" + fallback) if ev_lines else fallback)
    memories = [{"text": t, "date": ""} for t in collected]
    return memories, steps


_V8_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _collect_and_answer_v8(question, memory_dir, turn_index, max_rounds=None):
    """方案B：单模型——同一个模型检索+直接回答，上下文连贯，不交给独立 answerer。
    返回 (memories, steps, answer)。answer 从模型最终 <answer> 抽取。"""
    if max_rounds is None:
        max_rounds = int(os.environ.get("NATIVEMEM_V8_MAX_ROUNDS", "12"))
    max_tokens = int(os.environ.get("NATIVEMEM_V8_MAX_TOKENS", "1200"))
    read_ctx = int(os.environ.get("NATIVEMEM_V8_READ_CONTEXT", "1"))
    prompt = _V8_SINGLE_PROMPT.format(
        question=question, structure=_v8_structure_map(memory_dir))
    messages = [{"role": "user", "content": prompt}]
    tools = TOOLS + [_V8_READ_TOOL]
    if os.environ.get("NATIVEMEM_BM25", "off") == "on":
        tools.append(_V8_BM25_TOOL)
    collected, steps, final_text = [], 0, ""
    for _ in range(max_rounds):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL, messages=messages, tools=tools,
                max_tokens=max_tokens, temperature=0.0)
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
                original = read_turns(turn_index, dids, context=read_ctx)
                ev_lines = _v8_event_lines(memory_dir, dids)
                block = (ev_lines + "\n" + original) if ev_lines else original
                if block.strip():
                    collected.append(block)
                out = original or "(无对应原文)"
            elif tc.function.name == "bm25_search":
                out = _run_bm25_search(memory_dir, args)
            else:
                # 通用 shell 只能查看 Abstract Memory；原文必须经 read_original。
                out = execute_tool(tc.function.name, args, memory_dir, hide_raw=True)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    # Some models spend every allowed round on tools and therefore never emit
    # the final <answer>.  Make one tools-disabled call with the complete trace
    # instead of recording an empty answer or rebuilding the memory sample.
    if not (final_text or "").strip():
        finalize = ({"role": "user", "content":
                    "工具阶段已经结束。不要再调用工具。请只根据上面的检索结果回答原问题："
                    f"{question}\n答案不超过 5-6 个词，严格输出 "
                    "<answer>...</answer>。没有可靠证据时回答 Not mentioned in the conversation。"})
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=ALIYUN_MODEL, messages=messages + [finalize],
                    max_tokens=max_tokens, temperature=0.0)
                log_usage(resp, phase="v8_single_finalize")
                steps += 1
                final_text = resp.choices[0].message.content or ""
                if final_text.strip():
                    break
            except Exception:  # noqa: BLE001
                if retry < 2:
                    time.sleep(retry + 1)
    m = _V8_ANSWER_RE.search(final_text or "")
    answer = m.group(1).strip() if m else (final_text or "").strip()
    memories = [{"text": t, "date": ""} for t in collected]
    return memories, steps, answer


def collect_memories(question, memory_dir, max_rounds=10):
    """Navigate the memory folder and collect relevant entries (no answering).

    Returns (memories: [{text, date}], steps: int).
    """
    if _uses_linked_memory():
        return _collect_v8(question, memory_dir, _V8_TURN_INDEX)
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


def _build_memory_v10(conv, memory_dir, max_sessions=None):
    """Compatibility wrapper for callers of the former private builder."""
    return load_version("v10").build_memory(
        conv, memory_dir, max_sessions=max_sessions
    )


def _build_memory_legacy(conv, memory_dir, max_sessions=None):
    """Replicates src/nativemem.py main(): 10-turn chunks, raw archive."""

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


def _build_memory_v8(conv, memory_dir, max_sessions=None):
    """Compatibility wrapper for callers of the former private builder."""
    return load_version("v8").build_memory(
        conv, memory_dir, max_sessions=max_sessions
    )


def build_memory(conv, memory_dir, max_sessions=None):
    """Dispatch versioned builders while preserving pre-V8 implementations."""
    version = os.environ.get("NATIVEMEM_PROMPT")
    if version in {"v8", "v10"}:
        return load_version(version).build_memory(
            conv, memory_dir, max_sessions=max_sessions
        )
    return _build_memory_legacy(conv, memory_dir, max_sessions=max_sessions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--memory-dir", default=None,
                    help="reuse an existing built memory folder (skips build)")
    ap.add_argument(
        "--source-build-record",
        default=None,
        help="original build-only JSON required when reusing versioned memory",
    )
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--questions-limit", type=int, default=None)
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="build the memory and write only the _build_stats record",
    )
    args = ap.parse_args()
    if args.build_only and args.memory_dir:
        ap.error("--build-only cannot be combined with --memory-dir")
    if args.source_build_record and not args.memory_dir:
        ap.error("--source-build-record requires --memory-dir")
    if (
        os.environ.get("NATIVEMEM_PROMPT") == "v10"
        and args.memory_dir
        and not args.source_build_record
    ):
        ap.error("v10 --memory-dir requires --source-build-record")

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
        build_time, n_units = build_memory(conv, memory_dir,
                                            max_sessions=args.max_sessions)
        build_snap = tracker.snapshot("build")
        # v8 的 build_memory 第二返回值是提炼出的**事件数**（n_events），不是 chunk 数；
        # 其余分支返回 chunk_idx。标注区分，别再把事件数误读成 chunk 数（见 auto-s9
        # "1591 chunks" 实为 1591 events / ~120 chunks）。
        unit = "events" if _uses_linked_memory() else "chunks"
        notes = f"built fresh: {n_units} {unit}, model={ALIYUN_MODEL}"
        print(f"[nativemem] build done: {build_time:.0f}s")

    if _uses_linked_memory():
        global _V8_TURN_INDEX
        _V8_TURN_INDEX = build_turn_index(conv)
    n_files = sum(1 for root, _, files in os.walk(memory_dir)
                  for fn in files if fn.endswith(".md") and "raw" not in root)

    build_record = {
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 1),
        "num_memories": n_files,
        "notes": notes,
        "build_calls": build_snap["calls"] if build_snap else None,
        "build_tokens_in": build_snap["tokens_in"] if build_snap else None,
        "build_tokens_out": build_snap["tokens_out"] if build_snap else None,
        "build_llm_time_s": build_snap["llm_time_s"] if build_snap else None,
    }
    if os.environ.get("NATIVEMEM_PROMPT") == "v10" and args.memory_dir:
        build_record = load_v10_source_build_record(
            args.source_build_record,
            memory_dir,
            args.sample,
        )
        build_record["notes"] = (
            f"reused existing memory dir: {os.path.abspath(memory_dir)}; "
            f"source build record: {os.path.abspath(args.source_build_record)}"
        )
    elif os.environ.get("NATIVEMEM_PROMPT") == "v10":
        build_record["method"] = "NativeMem-v10"
        build_record["sample"] = args.sample
        build_record["max_sessions"] = args.max_sessions
        build_record["builder_model"] = ALIYUN_MODEL
        build_record["memory_dir"] = os.path.abspath(memory_dir)
        build_record["v10_config"] = v10_memory.V10BuildConfig.from_env().to_dict()
        phase_usage = {}
        for entry in nativemem_runtime.CALL_LOG:
            phase = str(entry.get("phase", "unknown"))
            totals = phase_usage.setdefault(
                phase,
                {"calls": 0, "tokens_in": 0, "tokens_out": 0},
            )
            totals["calls"] += 1
            totals["tokens_in"] += int(entry.get("prompt_tokens", 0) or 0)
            totals["tokens_out"] += int(entry.get("completion_tokens", 0) or 0)
        build_record["build_phase_usage"] = phase_usage
    if args.build_only:
        atomic_json(args.output, [build_record])
        print(f"[nativemem] wrote build stats -> {args.output}")
        return
    single_v8 = (_uses_linked_memory()
                 and os.environ.get("NATIVEMEM_V8_SINGLE") == "1")
    direct_answer = single_v8

    checkpoint = checkpoint_path(args.output)
    identity = checkpoint_identity(
        sample=args.sample, qas=qas, memory_dir=memory_dir
    )
    checkpoint_state = load_or_create(
        checkpoint, identity=identity, build_record=build_record
    )
    build_record = checkpoint_state["build_record"]
    completed = completed_answers(
        checkpoint_state,
        sample=args.sample,
        qas=qas,
        require_answer=direct_answer,
    )
    if completed:
        print(
            f"[nativemem] resumed {len(completed)}/{len(qas)} question records "
            f"from {checkpoint}"
        )

    def _answer_one(qi, qa):
        """答一题。全程局部变量（memories/steps/messages 都在 _collect_* 内部局部），
        题间无共享可变状态；每题一个 tracker phase，绑定到本线程做 per-题记账。"""
        q = qa["question"]
        gold = str(qa.get("answer", qa.get("adversarial_answer", "")))
        t0 = time.time()
        phase = f"q{qi}"
        tracker.reset(phase)
        direct_answer_text = None
        tool_trace = None
        with tracker.bind_thread(phase):
            try:
                if single_v8:
                    memories, steps, direct_answer_text = _collect_and_answer_v8(
                        q, memory_dir, _V8_TURN_INDEX)
                else:
                    memories, steps = collect_memories(q, memory_dir)
            except Exception as _e:  # noqa: BLE001
                # 单题遇 API 异常（阿里云内容过滤 data_inspection_failed 等）不能
                # 拖垮整轮评测——记空、继续下一题。
                print(f"  q{qi}: SKIPPED ({type(_e).__name__}: {str(_e)[:60]})")
                memories, steps = [], 0
        snap = tracker.snapshot(phase)
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
        if direct_answer_text is not None:
            _rec["answer"] = direct_answer_text
        if tool_trace is not None:
            _rec["tool_trace"] = tool_trace
        print(f"  q{qi}: {len(memories)} memories, {steps} steps, {latency:.1f}s")
        return _rec

    # 199 题互相独立、只读库。主线程按完成顺序逐题原子持久化，最终结果仍按
    # 原始题号排序。中断后仅重新处理缺失或空答案的题。
    pending = [(qi, qa) for qi, qa in enumerate(qas) if qi not in completed]
    workers = min(_v8_concurrency(), len(pending)) if pending else 1
    if workers <= 1:
        for qi, qa in pending:
            record = _answer_one(qi, qa)
            save_answer(checkpoint, checkpoint_state, qi, record)
            if not direct_answer or str(record.get("answer", "")).strip():
                completed[qi] = record
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_answer_one, qi, qa): qi for qi, qa in pending
            }
            for future in as_completed(futures):
                qi = futures[future]
                record = future.result()
                save_answer(checkpoint, checkpoint_state, qi, record)
                if not direct_answer or str(record.get("answer", "")).strip():
                    completed[qi] = record

    missing = [qi for qi in range(len(qas)) if qi not in completed]
    if missing:
        preview = ",".join(str(qi) for qi in missing[:20])
        raise RuntimeError(
            f"{len(missing)} questions lack complete answers; indexes={preview}"
        )

    records = [build_record]
    records.extend(completed[qi] for qi in range(len(qas)))

    atomic_json(args.output, records)
    mark_complete(checkpoint, checkpoint_state)
    print(f"[nativemem] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
