"""v8: 双视图索引 + 回原文检索的核心（纯函数，不含大模型调用的落盘/提炼在别的 task）。"""
import os
import re
import json
import time
import threading

from src.nativemem import client, ALIYUN_MODEL, log_usage, normalize_date


_DISTILL_TRACE_LOCK = threading.Lock()


def _write_distill_trace(record):
    """Append one first-pass/post-verify record when tracing is enabled."""
    path = os.environ.get("NATIVEMEM_DISTILL_TRACE", "").strip()
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": 1,
        "builder_model": ALIYUN_MODEL,
        **record,
    }
    with _DISTILL_TRACE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_turn_index(conv):
    """把 conversation 的所有 turn 建成 {dia_id: {speaker,text,order,date}} 映射。
    order 是跨 session 的全局顺序号，用于 read_turns 取前后上下文。
    date 是该 turn 所在 session 的对话日期（session_N_date_time），归一化成
    YYYY-MM-DD —— 这是标准 answerer prompt 用来把"yesterday"等相对时间换算成
    绝对日期的基准时间戳（对齐 Mem0/Nemori/LightMem：它们的 memory 都带时间戳）。"""
    index = {}
    order = 0
    i = 1
    while f"session_{i}" in conv:
        sess_date = normalize_date(conv.get(f"session_{i}_date_time", ""))
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
                "date": sess_date,
            }
            order += 1
        i += 1
    return index


# dia_id 引用锚：左半 D<对话号>，右半 <句>，右半可扩展成区间/列举/混合：
#   单句 D2:8 | 区间 D2:3-15（含头含尾）| 散句 D2:2,10 | 混合 D2:3-5,9,12
_DIA_RE = re.compile(r"D\d+:\d+(?:[-,]\d+)*")


def expand_dia_ids(s):
    """把一个引用字符串（可含区间/列举/混合写法）展开成单句 dia_id 列表，去重保序。
    单句是恒等（"D2:8" → ["D2:8"]）。s 可以是一个 ref token 也可以是一整行/整段文本，
    其中所有能识别的引用锚都会被展开。识别不出任何锚的非法输入按原样单 id 返回
    （[s]），绝不抛异常。"""
    if not isinstance(s, str):
        return []
    tokens = _DIA_RE.findall(s)
    if not tokens:                               # 非法输入：按原样单 id 处理，别崩
        s = s.strip()
        return [s] if s else []
    out, seen = [], set()
    for tok in tokens:
        conv, rhs = tok.split(":", 1)            # conv="D2", rhs="3-5,9"
        for part in rhs.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                nums = range(lo, hi + 1) if lo <= hi else range(hi, lo + 1)
            else:
                nums = [int(part)]
            for n in nums:
                did = f"{conv}:{n}"
                if did not in seen:
                    seen.add(did)
                    out.append(did)
    return out


def dia_ids_in(text):
    """从任意文本抽出所有引用锚并展开成单句 dia_id 列表（去重保序）。
    统一"从文本抽 dia_id 集合"的入口——去重指纹、并集守卫、集合相等校验等都先
    经它展开再比对，区间/列举/混合和展开后的单句一视同仁。"""
    return expand_dia_ids(text or "")


def fold_dia_ids(ids):
    """把 dia_id 列表折叠成统一书写 token:一个 token 只装一个对话,句号升序,
    连续段折成 a-b、散句逗号列举(逗号后不带空格),对话按编号升序。
    ["D2:5","D2:3","D2:4","D2:9","D5:1"] → ["D2:3-5,9", "D5:1"]。
    识别不出的元素先经 expand_dia_ids 展开,仍不合形的原样跟在末尾。"""
    groups, passthrough, seen_bad = {}, [], set()
    for ref in ids or []:
        for e in expand_dia_ids(ref):
            m = re.fullmatch(r"D(\d+):(\d+)", e)
            if m:
                groups.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
            elif e not in seen_bad:
                seen_bad.add(e)
                passthrough.append(e)
    out = []
    for conv in sorted(groups):
        ts = sorted(groups[conv])
        parts, i = [], 0
        while i < len(ts):
            j = i
            while j + 1 < len(ts) and ts[j + 1] == ts[j] + 1:
                j += 1
            parts.append(f"{ts[i]}-{ts[j]}" if j > i else str(ts[i]))
            i = j + 1
        out.append(f"D{conv}:{','.join(parts)}")
    return out + passthrough


def format_dia_refs(ids):
    """dia_id 列表 → 统一引用串:每个对话一个方括号,括号间单空格。
    唯一合法形态(写入侧一律经此产出;读取侧新旧写法都兼容):
      [D2:8] / [D2:8,11] / [D2:12-14] / [D2:3-5,9] / [D2:8,11] [D5:1]"""
    return " ".join(f"[{tok}]" for tok in fold_dia_ids(ids))


def read_turns(turn_index, dia_ids, context=1):
    """给定 dia_id 列表（每个元素可含区间/列举/混合写法），返回对应原文。
    区间展开后若相邻 turn 连续，整段本身即上下文、不再另加前后窗口；散句仍按
    context 带前后窗口。无效 id 跳过。返回 'speaker: text' 按原顺序拼接、去重的文本。"""
    by_order = {v["order"]: (k, v) for k, v in turn_index.items()}
    # 逐元素展开：区间/列举 → 单句 id；同一元素展开出的多句视为"连续整段"，
    # 段内取原文不再补前后窗口（整段就是上下文），散句才补窗口。
    want_orders = set()
    for ref in dia_ids:
        ids = expand_dia_ids(ref)
        span = len(ids) > 1                       # 区间/列举展开出多句 → 整段
        for did in ids:
            hit = turn_index.get(did)
            if hit is None:
                continue
            o = hit["order"]
            if span:
                want_orders.add(o)                # 整段：只取该句自身
            else:
                for oo in range(o - context, o + context + 1):
                    if oo in by_order:
                        want_orders.add(oo)
    lines = []
    for o in sorted(want_orders):
        _did, v = by_order[o]
        # 带上说话那天的对话日期戳，标准 answerer 靠它把"yesterday"等相对
        # 时间换算成绝对日期（对齐 Mem0/Nemori 的 timestamped memory 口径）。
        d = v.get("date", "")
        lines.append(f"({d}) {v['speaker']}: {v['text']}" if d
                     else f"{v['speaker']}: {v['text']}")
    return "\n".join(lines)


_LINE_DATE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]")


def event_line(when, summary, dia_ids):
    refs = format_dia_refs(dia_ids)
    return f"[{when}] {summary} · {refs}" if refs else f"[{when}] {summary} ·"


_NORM_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_text(s):
    """事实文本规范化（层1精确去重用）：小写、去所有标点、折叠空白。同一事实被
    distill 两遍常只差标点/大小写/空格，规范化后相等即判为重复。"""
    return " ".join(_NORM_PUNCT_RE.sub(" ", s.lower()).split())


def _timeline_line_fingerprint(line):
    """一条 timeline 行的去重指纹 (frozenset(dia_ids), 规范化摘要文本)。
    行形如 `[when] summary · [D1:1, D1:2] → topics/x.md`：dia_id 集合从 `· [...]`
    段取（→ 反链和 [when] 前缀不进指纹，反链由 topic 确定、集合相同即同事实）。"""
    m = re.search(r"·\s*\[([^\]]*)\]", line)
    ids = frozenset(dia_ids_in(m.group(1))) if m else frozenset()
    body = line.split(" · ", 1)[0]           # 去 [when] 前缀后的摘要
    body = _LINE_DATE_RE.sub("", body)
    return ids, _normalize_text(body)


def _timeline_line_exists(path, line):
    """path（当天 timeline 文件）里是否已有与 line 同指纹的行。"""
    if not os.path.exists(path):
        return False
    fp = _timeline_line_fingerprint(line)
    with open(path) as f:
        return any(_timeline_line_fingerprint(l) == fp
                   for l in f if l.strip())


def timeline_path(memory_dir, when):
    """天级路由：一天一文件 timeline/YYYY/MM/DD.md（纯数字，月份两位零填充，
    如 2023/07/07.md）。when 是 YYYY-MM-DD，三段直接落到年/月/日三层。"""
    y, m, d = when.split("-")
    return os.path.join(memory_dir, "timeline", y, m, f"{d}.md")


def _sanitize_topic(topic):
    """事实在主题库里的存放路径（相对 topics/），深度和形态都由模型自己决定——
    代码只做安全和卫生，不做结构决策：按 `/` 分段、每段单独清洗非法字符（非词
    字符→连字符）、丢弃空段和 `..`、拒绝绝对路径（前导 `/` 产生的空段自然被丢）。
    深度设一个宽松的安全上限（env NATIVEMEM_V8_MAX_DEPTH，默认 4，纯防失控）——
    超限时把多余的尾段折进最后一段。空回退 misc。模型路径被限定在 topics/ 门内，
    和代码专属的 timeline/ 天然不冲突。"""
    def clean(seg):
        return re.sub(r"[^\w]+", "-", seg).strip("-")
    segs = [clean(s) for s in topic.split("/")]
    segs = [s for s in segs if s and s != ".."]   # 丢空段与 ..
    if not segs:
        return "misc"
    max_depth = int(os.environ.get("NATIVEMEM_V8_MAX_DEPTH", "4"))
    if len(segs) > max_depth:
        segs = segs[:max_depth - 1] + ["-".join(segs[max_depth - 1:])]
    return "/".join(segs)


# 待链接的日期：[YYYY-MM-DD]（后面没跟 "(" 的，跟了说明已是链接）或裸 YYYY-MM-DD
# （前面没有 "["——有 "[" 的要么被前一分支处理、要么已在链接文本里）。
_DATE_ANY_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})\](?!\()"
                          r"|(?<!\[)\b(\d{4}-\d{2}-\d{2})\b(?!\])")


def link_dates_to_timeline(text):
    """代码确定性后处理：把 topics 文本里的绝对日期改写成指向对应 timeline 天
    文件的 markdown 链接 [2023-05-25](timeline/2023/05/25.md)。路径相对记忆根
    （检索时 cwd 即记忆根，模型可直接 cat 顺跳），由 timeline_path 统一算，与写入
    路由同源。幂等：已是链接的不重复包；非法月份原样保留。不花 LLM。"""
    def repl(m):
        d = m.group(1) or m.group(2)
        mo = int(d.split("-")[1])
        if not 1 <= mo <= 12:
            return m.group(0)
        rel = timeline_path("", d).replace(os.sep, "/")
        return f"[{d}]({rel})"
    return _DATE_ANY_RE.sub(repl, text)


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


_UNSECTIONED_HEADING = "## 未整理"


def _append_topic_line(path, line):
    """把 topics 行追加进 path。文件已分节（含 `## ` 标题）时，新行进末尾的固定
    `## 未整理` 收集节（不参与 _append_sorted 的全文件按日期重排——重排会把新行
    散进各节、破坏已整理的分节结构，也让"未归节"判定失效）；organize 整理时消掉
    该节。文件尚未分节（无 `## `）时沿用 _append_sorted 老行为（行首 [日期] 升序）。"""
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        if "\n## " in "\n" + content:      # 已有 `## ` 分节
            os.makedirs(os.path.dirname(path), exist_ok=True)
            body = content.rstrip("\n")
            if _UNSECTIONED_HEADING not in body.splitlines():
                body += f"\n\n{_UNSECTIONED_HEADING}"
            body += f"\n{line}"
            with open(path, "w") as f:
                f.write(body + "\n")
            return
    _append_sorted(path, line)


def write_events(memory_dir, events):
    """双视图落盘，两侧格式分家：
    - timeline：原子行 `[when] 纯摘要 · [refs]`（代码索引格式不变）
    - topics：内联引用句 `[when] 摘要句内嵌 [Dx:y]`（session 末分节整理的原料；
      [when] 前缀维持 _append_sorted 排序，整理时由模型只做分节、行文本不动）。
    事件无 summary_inline 时（老调用方）topics 行体回退为与 timeline 相同的老格式。"""
    for ev in events:
        when = ev["when"]
        dia_ids = ev.get("dia_ids", [])
        topic = _sanitize_topic(ev.get("topic", "misc"))
        # timeline 行尾反链到这条事件的完整文章（代码确定性生成）：时间查询命中
        # 原子行后，模型顺反链 cat 文章拿上下文。行首 [日期] 与 dia_id 提取不受影响。
        tl_line = (event_line(when, ev["summary"], dia_ids)
                   + f" → topics/{topic}.md")
        # 层1：同 dia_id 集合 + 规范化摘要相同的行已在同天文件里存在时不重复写
        # （同一事实常被 distill 两遍）。纯代码，写入路径上判，不做事后扫描。
        tl_path = timeline_path(memory_dir, when)
        if not _timeline_line_exists(tl_path, tl_line):
            _append_sorted(tl_path, tl_line)
        # 根目录恒两个视图：timeline/（代码路由）+ topics/（固定入口）。
        # 门内层级由模型决定（Caroline/adoption → topics/Caroline/adoption.md，
        # 一级 adoption → topics/adoption.md）。_append_sorted 已 makedirs。
        topic_path = os.path.join(memory_dir, "topics", *topic.split("/")) + ".md"
        inline = ev.get("summary_inline")
        topic_line = (f"[{when}] {inline}" if inline
                      else event_line(when, ev["summary"], dia_ids))
        # topics 侧日期→timeline 链接（代码确定性生成；链接形态行首仍是
        # [日期]，_append_sorted 排序不破）。timeline 侧不自链。
        _append_topic_line(topic_path, link_dates_to_timeline(topic_line))


def dedup_topic_files(memory_dir):
    """逐文件精确去重：每个 .md 里完全相同的事件行只留一条（保序）。返回删除行数。
    纯确定性，不调模型。跳过 raw/隐藏目录与隐藏文件（与 measure_library 一致）。"""
    removed = 0
    for root, dirs, files in os.walk(memory_dir):
        dirs[:] = [x for x in dirs if x != "raw" and not x.startswith(".")]
        for fn in files:
            if not fn.endswith(".md") or fn.startswith("."):
                continue
            path = os.path.join(root, fn)
            with open(path) as f:
                lines = [l.rstrip("\n") for l in f]
            seen, kept = set(), []
            for l in lines:
                if l.strip() and l in seen:
                    removed += 1
                    continue
                if l.strip():
                    seen.add(l)
                kept.append(l)
            if len(kept) != len(lines):
                with open(path, "w") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))
    return removed


_V8_MERGE_PROMPT = """You are tidying the topic FILES in one person's memory. Each topic is a separate markdown file, created incrementally, so some files are really the SAME thread under different names — not just spelling-close (adopt / adoption) but meaning-close (pets / animals, job / career).

Below is the full list of topic file paths (relative to topics/), each with its line count. Propose which files should be merged into a single canonical file.

{files}

Rules:
- Only merge files that are genuinely the SAME thread / topic. Semantic sameness counts (pets ≈ animals), not just similar spelling.
- Pick the clearest existing path as `into` (prefer the fuller / more-populated one). `into` must be one of the paths above; every path in `from` must be one of the paths above; `into` must not appear in its own `from`.
- When in doubt, do NOT merge. Distinct topics stay separate.
- No merges needed → {{"merges": []}}.

Output JSON only:
{{"merges": [{{"into": "Caroline/adoption", "from": ["Caroline/adopt"]}}]}}"""


def _topics_dir_files(memory_dir):
    """topics/ 子树里所有 .md 的相对路径名（剥 .md，相对 topics/），递归 walk
    任意深度，跳过隐藏目录/文件。返回如 `Caroline/adoption`、一级的 `adoption`。
    老库平铺文件天然兼容。目录不存在返回 []。"""
    tdir = os.path.join(memory_dir, "topics")
    if not os.path.isdir(tdir):
        return []
    names = []
    for root, dirs, files in os.walk(tdir):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            if not f.endswith(".md") or f.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(root, f), tdir)[:-3]
            names.append(rel.replace(os.sep, "/"))
    return sorted(names)


def _topic_merge_candidates(memory_dir):
    """近义 topic 文件名候选组（≥2 才算候选）。复用 nativemem._find_merge_candidates。"""
    from src.nativemem import _find_merge_candidates
    names = _topics_dir_files(memory_dir)
    if len(names) < 2:
        return []
    return [g for g in _find_merge_candidates(names) if len(g) >= 2]


def _retarget_timeline_backlinks(memory_dir, from_topic, into_topic):
    """把 timeline 里所有 `→ topics/{from_topic}.md` 反链改指到 into_topic。
    话题文件合并删源后，指向被删文件的反链会悬空——这里跟着改。纯代码。"""
    src = f"→ topics/{from_topic}.md"
    dst = f"→ topics/{into_topic}.md"
    tl_dir = os.path.join(memory_dir, "timeline")
    for root, dirs, files in os.walk(tl_dir):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for fn in files:
            if not fn.endswith(".md") or fn.startswith("."):
                continue
            path = os.path.join(root, fn)
            with open(path) as f:
                content = f.read()
            if src in content:
                with open(path, "w") as f:
                    f.write(content.replace(src, dst))


def _validate_topic_merges(merges, names):
    """校验模型的话题合并提案，返回 [(into, [from...])]（已过滤非法项）。
    规则：into/from 都必须是现存路径；into≠from；同一 from 不许出现两次；
    任一 into 不许又作为某条 from（防合并链/环——文件不能既是目标又是源）。"""
    nameset = set(names)
    if not isinstance(merges, list):
        return []
    intos, froms = set(), set()
    clean = []
    for mg in merges:
        if not isinstance(mg, dict):
            continue
        into = str(mg.get("into", "")).strip()
        srcs = mg.get("from", [])
        if into not in nameset or not isinstance(srcs, list):
            continue
        valid = []
        for s in srcs:
            s = str(s).strip()
            if s in nameset and s != into and s not in froms:
                froms.add(s)
                valid.append(s)
        if valid:
            intos.add(into)
            clean.append((into, valid))
    # 无环守卫：into 集合与 from 集合不许相交（既当目标又当源 → 拒该 into 项）
    bad = intos & froms
    return [(i, f) for i, f in clean if i not in bad]


def consolidate_topic_files(memory_dir, max_retry=4):
    """层3：把 topics/ 全路径清单（带行数）一次性交模型，让它提**确实指同一线索**
    的语义合并建议 {"merges":[{"into","from":[...]}]}（替代词面重叠候选）。代码验证
    （路径存在、into≠from、无环）后执行：搬行、删源、去重、并改 timeline 反链。
    只搬事件行不改行内容（不动 [dia_id] 锚）。返回被合并（删除）的文件数。"""
    names = _topics_dir_files(memory_dir)
    if len(names) < 2:
        return 0
    tdir = os.path.join(memory_dir, "topics")
    files_txt = "\n".join(
        f"- {n}  ({_count_topic_lines(os.path.join(tdir, *n.split('/')) + '.md')} lines)"
        for n in names)
    for retry in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=ALIYUN_MODEL,
                messages=[{"role": "system",
                           "content": _V8_MERGE_PROMPT.format(files=files_txt)}],
                temperature=0.1)
            break
        except Exception:  # noqa: BLE001
            if retry < max_retry - 1:
                time.sleep(3 * (retry + 1))
            else:
                return 0
    log_usage(resp, phase="v8_consolidate")
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return 0
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return 0
    merges = obj.get("merges", []) if isinstance(obj, dict) else []
    plan = _validate_topic_merges(merges, names)
    merged_files = 0
    for into, srcs in plan:
        into_path = os.path.join(tdir, *into.split("/")) + ".md"
        for src_name in srcs:
            src_path = os.path.join(tdir, *src_name.split("/")) + ".md"
            if not os.path.exists(src_path):
                continue
            with open(src_path) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.strip():
                        _append_sorted(into_path, line)
            os.remove(src_path)
            _retarget_timeline_backlinks(memory_dir, src_name, into)
            merged_files += 1
    if merged_files:
        dedup_topic_files(memory_dir)
    return merged_files


def _count_topic_lines(path):
    """topic 文件里非空、非标题的内容行数（喂模型判合并规模用）。"""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for ln in f if ln.strip() and not ln.lstrip().startswith("#"))

_V8_ARTICLE_PROMPT = """下面是一个人记忆库里的一个话题文件，内容是逐句追加的记忆句（行首 [日期]，句内 [Dx:y] 是指向原始对话的引用锚）。把它重写成一篇**自由文体的小文章**。

要求：
- 结构（标题/段落/列表）你自己定，怎么清楚怎么来。合并讲同一件事的句子，去掉重复。
- **引用一个不许丢、一个不许造**：原文里出现的每个 [Dx:y] 都必须出现在重写后的文章里，且不许出现原文没有的。引用保持内联——写到哪件事，引用就跟在那句话里。
- 日期写进句子文本（如 "On 2023-05-25, ..."），别保留行首 [日期] 前缀格式。
- **日期照抄，禁止推算**：只能用原子行里已经写出的日期，禁止自己推算、换算或新造任何日期。原文里的相对时间表述（"last week"、"two weekends ago"、"yesterday"）必须**原词保留**在句子里，可以和已有日期并存，例："the week before [2023-06-09](timeline/...), 'last week', ..."。绝不把相对时间二次换算成具体日期。
- **关键细节词原词保留，禁止同义替换或泛化**：专有名词、数字、具体名词和形容词（活动名、物品名、"graceful" 这类描述词）必须逐字保留原词，不许换近义词、不许泛化（别把 "nature walk" 写成 "family outing"，别把 "graceful" 写成 "admired"）。宁可句子朴素，不许改写关键细节词。
- 用英文写，保留所有专有名词和数字。

示例（仅示意"带内联引用的文章"长什么样，不是模板，结构随内容自定）：

## Caroline's adoption
On 2023-05-25, Caroline decided to adopt [D2:3], visited an agency [D2:8],
and chose it for its LGBTQ-friendly values [D2:11]. The home visit was
scheduled for July [D5:2].

只输出重写后的文章正文，不要解释。"""

_V8_ARTICLE_INCREMENTAL_HINT = """

**增量整理**：上面的文件已经是"已成文的文章段落 + 尾部若干条新追加的 [日期] 生行"。
把尾部这些新句子按内容融进既有文章的合适位置，**尽量少改动已成文的段落**（保留原有措辞和结构），
不要推倒重写。输出仍是整篇完整文章（含融入后的全部内容）。"""


_RAW_LINE_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]")


def _raw_line_count(path):
    """一个 topic 文件里"自上次文章化以来新追加的生行"数——零状态、按格式判定：
    write_events 追加的记忆行行首恒是 [YYYY-MM-DD]（link_dates_to_timeline 只往
    日期后补 (timeline/...) 链接，不动行首方括号），文章化后的散文以 ## 或
    On [date]... 开头，绝不以行首 [日期] 打头。所以行首匹配 ^\\[日期\\] 的行数就是
    待整理量。文件不存在返回 0。"""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for ln in f if _RAW_LINE_RE.match(ln))


def _should_rewrite(path, raw, total_nonblank, min_raw, final):
    """该文件是否触发文章化。收尾扫描（final）：只要有生行就整理。常规：生行数
    ≥ min_raw，或从未成文（全是生行）且行数 ≥ 3——够小则先攒着，检索仍能读生行。"""
    if raw == 0:
        return False
    if final:
        return True
    if raw >= min_raw:
        return True
    return raw == total_nonblank and total_nonblank >= 3   # 从未成文的小文件


def rewrite_topic_articles(memory_dir, touched, max_retry=4, final=False):
    """按增长触发的文章化整理：只重写"自上次整理以来生行数够多"的 topic 文件，
    而非本 session 碰过的每个文件全文重写（那是 O(n²) 输出——文件越长每次重写
    生成越多）。生行 = 行首 [YYYY-MM-DD] 的未整理记忆行（_raw_line_count）。
    触发规则见 _should_rewrite；final=True 时对任何仍有生行的文件收尾整理。
    硬校验：重写前后文中 dia_id **集合必须相等且非空**（regex 提取比对），否则
    拒绝写回保持原样——落盘永远代码执行，库不因模型失误变坏。返回重写文件数。"""
    min_raw = int(os.environ.get("NATIVEMEM_V8_REWRITE_MIN", "8"))
    rewritten = 0
    for name in sorted(touched):
        path = os.path.join(memory_dir, "topics", *name.split("/")) + ".md"
        if not os.path.exists(path):
            continue                      # 可能已被 consolidate 合并删除
        with open(path) as f:
            original = f.read()
        before = set(dia_ids_in(original))
        if not before:
            continue                      # 无引用锚的文件不动
        raw = sum(1 for ln in original.splitlines() if _RAW_LINE_RE.match(ln))
        total_nonblank = sum(1 for ln in original.splitlines() if ln.strip())
        if not _should_rewrite(path, raw, total_nonblank, min_raw, final):
            continue                      # 生行不够 → 先攒着，生行留在文件尾
        # 增量模式：文件已有成文段（raw < 总行），指示模型把尾部生行融入既有文章、
        # 少动已成文部分，别推倒重写（降低越长越易出错的重写风险）。全新文件全量写。
        sys_prompt = _V8_ARTICLE_PROMPT
        if raw < total_nonblank:
            sys_prompt += _V8_ARTICLE_INCREMENTAL_HINT
        text = _distill_call(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": original}],
            phase="v8_article", max_retry=max_retry)
        if not text:
            continue
        article = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        article = re.sub(r"^```(?:markdown|md)?|```$", "", article,
                         flags=re.MULTILINE).strip()
        if not article or set(dia_ids_in(article)) != before:
            continue                      # 引用集合不等 → 拒绝写回
        with open(path, "w") as f:
            f.write(link_dates_to_timeline(article) + "\n")
        rewritten += 1
    return rewritten


_V8_MERGE_LINES_PROMPT = """下面是一个人记忆库里的一个话题文件的内容行，每行前面有 [编号]。这些是逐句追加的记忆句（行内 [Dx:y] 是指向原始对话的引用锚）。同一件事常被提炼了两遍，于是出现**陈述同一个事实的近重复行**。

你的任务：**只找出陈述同一事实的行组**，让代码把它们合并成一条。**一个字都不许改行的文本**，你只输出行号。

要求：
- 每组给一个 keep（信息最全、细节最多的那行的编号）和一个 merge 列表（与 keep 陈述**同一个事实**、可被它取代的其它行的编号）。
- **只合并真正同义的行**：讲的是同一个人做的同一件事。只要日期不同、地点不同、对象不同、是不同的一件事，就**不要**放进同一组——拿不准就别合。
- 一个编号最多出现在一处（要么是某组的 keep，要么是某组的 merge，不许跨组重复，不许自己 merge 自己）。
- 没有任何近重复就输出空 groups。

只输出 JSON：{{"groups":[{{"keep":3,"merge":[7,12]}}]}}"""


def _line_date(line):
    """内容行行首 [YYYY-MM-DD]（可能是链接 [..](..)) 的日期串，取不到返回 ''。"""
    m = _LINE_DATE_RE.match(line)
    return m.group(1) if m else ""


def _apply_line_merges(text, content):
    """按模型返回的同义行组方案合并 content（编号 1..N 的内容行原文），返回合并后
    的内容行列表；任一硬守卫失败返回 None（记 warning，调用方保留原文件）。
    薄封装 _apply_line_merges_mapped，只取合并后行列表（旧消融路径用）。"""
    r = _apply_line_merges_mapped(text, content)
    return None if r is None else r[0]


def _parse_merge_groups(groups, content):
    """把模型 groups 方案解析成 [(keep_idx, [merge_idx...])]；任一守卫失败返回 None
    （记 warning）。守卫：group 是对象、keep/merge 行号是 1..N 内整数、组间不重叠、
    keep∉merge。合并前后 dia_id 并集校验留给 _apply_line_merges_mapped。"""
    if not isinstance(groups, list):
        print("[v8_merge_lines] groups 字段缺失，放弃合并")
        return None
    n = len(content)
    parsed, touched = [], set()               # parsed: [(keep_idx, [merge_idx...])]
    for g in groups:
        if not isinstance(g, dict):
            print("[v8_merge_lines] group 非对象，放弃合并")
            return None
        try:
            keep = int(str(g.get("keep", "")).strip())
        except (ValueError, TypeError):
            print("[v8_merge_lines] keep 非整数，放弃合并")
            return None
        merge = g.get("merge", [])
        if not isinstance(merge, list):
            print("[v8_merge_lines] merge 非列表，放弃合并")
            return None
        midxs = []
        for r in merge:
            try:
                k = int(str(r).strip())
            except (ValueError, TypeError):
                print("[v8_merge_lines] merge 行号非整数，放弃合并")
                return None
            midxs.append(k)
        for k in [keep] + midxs:
            if not 1 <= k <= n:
                print(f"[v8_merge_lines] 行号 {k} 越界，放弃合并")
                return None
            if k in touched:
                print(f"[v8_merge_lines] 行号 {k} 跨组重叠/自并，放弃合并")
                return None
            touched.add(k)
        parsed.append((keep, midxs))
    return parsed


def _merge_from_parsed(parsed, content):
    """按已解析的 parsed 组合并 content，返回 (合并后行列表, 存活前编号→合并后位置)。
    dia_id 并集守卫不过返回 (None, None)。survivor_pos 只含未被 merge 掉、且落到结果里
    的前编号（1-based → 结果 0-based 位置）。keep 文本逐字保留，merge 行独有 dia_id 追加。"""
    before_ids = set(dia_ids_in("\n".join(content)))
    merged_out = {m for _k, ms in parsed for m in ms}   # 被并掉的行号（1-based）
    keep_of = {k: ms for k, ms in parsed}
    result, survivor_pos = [], {}
    for i, line in enumerate(content, 1):
        if i in merged_out:
            continue
        survivor_pos[i] = len(result)
        result.append(line)
        if i not in keep_of:
            continue
        midxs = keep_of[i]
        # 追加 merge 行独有的 dia_id 到 keep 行尾（文本不动，只加引用）
        keep_ids = set(dia_ids_in(line))         # 展开区间/列举后的单句集合
        extra = []
        earliest = _line_date(line)
        for m in midxs:
            mline = content[m - 1]
            md = _line_date(mline)
            if md and (not earliest or md < earliest):
                earliest = md
            for d in dia_ids_in(mline):          # merge 行独有的单句 id
                if d not in keep_ids and d not in extra:
                    extra.append(d)
        if earliest and _line_date(line) and earliest != _line_date(line):
            line = _LINE_DATE_RE.sub(f"[{earliest}]", line, count=1)
        if extra:
            line = f"{line} · {format_dia_refs(extra)}"
        result[-1] = line
    after_ids = set(dia_ids_in("\n".join(result)))
    if after_ids != before_ids:
        print("[v8_merge_lines] dia_id 并集不等，放弃合并")
        return None, None
    return result, survivor_pos


def _apply_line_merges_mapped(text, content):
    """解析模型 groups 方案并合并 content。返回 (合并后行列表, 存活前编号→合并后位置)；
    任一硬守卫失败返回 None。无实际合并时返回 (content, 恒等映射)（不算失败）。
    守卫：JSON 可解析；行号 1..N 内、组间不重叠、keep∉merge；合并前后全文件
    dia_id 并集完全相等。"""
    obj = _parse_json_obj(text, "v8_merge_lines")
    if obj is None:
        return None
    groups = obj.get("groups") if isinstance(obj, dict) else None
    parsed = _parse_merge_groups(groups, content)
    if parsed is None:
        return None
    if not any(m for _k, m in parsed):
        return content, {i: i - 1 for i in range(1, len(content) + 1)}
    return _merge_from_parsed(parsed, content)


def _parse_json_obj(text, tag):
    """剥 <think>/```围栏后解析 JSON 对象；失败尝试抓第一个 {...}；再失败返回 None。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            print(f"[{tag}] JSON 解析失败，放弃")
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            print(f"[{tag}] JSON 解析失败，放弃")
            return None


def merge_duplicate_lines(memory_dir, touched, final=False):
    """层2：topic 文件内同义行合并——模型只提近重复行组，代码执行合并+硬守卫。
    触发与分节整理同位（session 末 touched + final sweep），且**先于分节做**
    （合并会删行，先合再分节省一次）。开关 NATIVEMEM_V8_MERGE_LINES（默认 on）。
    守卫见 _apply_line_merges；任一违反 → 放弃该文件合并、保留原文件。返回合并文件数。"""
    if os.environ.get("NATIVEMEM_V8_MERGE_LINES", "on") == "off":
        return 0
    min_raw = int(os.environ.get("NATIVEMEM_V8_REWRITE_MIN", "8"))
    merged = 0
    for name in sorted(touched):
        path = os.path.join(memory_dir, "topics", *name.split("/")) + ".md"
        if not os.path.exists(path):
            continue                      # 可能已被 consolidate 合并删除
        with open(path) as f:
            original = f.read()
        # 只在该文件即将被重新分节时才合并（同一阈值）：合并重建会丢弃旧
        # `## ` 标题，必须保证紧随其后的分节把标题补回来；同时把合并调用
        # 频率降到与分节同频，否则每 session 每文件一次全文调用，成本爆炸。
        if not final and _unsectioned_count(original) < min_raw:
            continue
        title_lines, content = _split_topic_file(original)
        if len(content) < 2:
            continue                      # 少于两行无从重复
        numbered = "\n".join(f"[{i}] {ln}" for i, ln in enumerate(content, 1))
        text = _distill_call(
            [{"role": "system", "content": _V8_MERGE_LINES_PROMPT},
             {"role": "user", "content": numbered}],
            phase="v8_merge_lines")
        if not text:
            continue
        new_content = _apply_line_merges(text, content)
        if new_content is None or len(new_content) == len(content):
            continue                      # 守卫未过或无合并 → 保留原文件
        # 合并只在无分节的原始文件上做（触发时机在分节前）；重建保留 # 一级标题。
        rebuilt = "\n".join(title_lines + new_content) + "\n"
        with open(path, "w") as f:
            f.write(rebuilt)
        merged += 1
    return merged


_V8_SECTIONS_PROMPT = """下面是一个人记忆库里的一个话题文件的内容行，每行前面有 [编号]。这些是逐句追加的记忆句（行内 [Dx:y] 是指向原始对话的引用锚，可能含日期链接）。

你的任务：**只做分节整理**——给这些行分组，起小标题，决定每节内的行序。**一个字都不许改行的文本**，你只输出行的编号归属。

要求：
- 把内容相近的行归到同一节，给每节起一个能概括内容的中文或英文小标题（heading）。
- 每节的 lines 按你希望的行内顺序列出编号（可重排，让同一件事的行挨在一起）。
- **每个编号必须恰好出现在一节里，一次**：不许漏任何编号，不许重复，不许新增不存在的编号。
- 不要改写、合并、删除或新造任何行文本——你只给编号。

只输出 JSON：{{"sections":[{{"heading":"Adoption","lines":[3,1]}},{{"heading":"Career","lines":[2]}}]}}"""

_LINE_NO_RE = re.compile(r"^\[(\d+)\]\s(.*)$", re.DOTALL)


def _split_topic_file(text):
    """把 topic 文件拆成 (title_lines, content_lines)：
    - title_lines：文件最前面连续的 `# 一级标题` 行（保留，整理时置顶）
    - content_lines：其余的内容行（非空、非 `## ` 二级标题、非 `# ` 一级标题）
    `## ` 二级标题是上次分节留下的，整理时丢弃重排；空行丢弃。"""
    title_lines, content_lines = [], []
    seen_content = False
    for ln in text.splitlines():
        s = ln.rstrip()
        if not s.strip():
            continue
        if s.startswith("# ") and not seen_content:
            title_lines.append(s)
            continue
        if s.startswith("## "):
            continue                      # 旧分节标题，丢弃重排
        seen_content = True
        content_lines.append(s)
    return title_lines, content_lines


def _unsectioned_count(text):
    """文件里"未归节"的内容行数——触发整理的判据。判定简单可靠：
    - 无任何 `## ` 二级标题 → 所有内容行都算未归节
    - 有 `## ` → 出现在第一个 `## ` 之前、以及 `## 未整理` 收集节之下的内容行算未归节
    （write_events 追加的新行进 `## 未整理` 节，见 _append_topic_line）。"""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    heads = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    if not heads:
        return sum(1 for ln in lines
                   if not ln.startswith("# ") and not ln.startswith("## "))
    n = 0
    in_unsectioned = False
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            in_unsectioned = (ln == _UNSECTIONED_HEADING)
            continue
        if ln.startswith("# "):
            continue
        if i < heads[0] or in_unsectioned:
            n += 1
    return n


def organize_topic_sections(memory_dir, touched, final=False):
    """分节整理（替代全文重写）：模型只出分节方案，代码搬行、行逐字不动。
    触发：文件"未归节"行数 ≥ NATIVEMEM_V8_REWRITE_MIN（final=True 时只要有未归节行）。
    模型给 {"sections":[{"heading","lines":[行号...]}]}；代码按方案重建文件：
    保留原有 `# 一级标题`，然后每节 `## heading` + 归属行原文（可重排）。
    硬校验：重建后内容行多重集合必须与原文逐字相等（排序比对）；JSON 坏/行号越界/
    缺失/重复/多重集不等 → 放弃本次整理、保留原文件、记 warning。返回整理文件数。"""
    min_raw = int(os.environ.get("NATIVEMEM_V8_REWRITE_MIN", "8"))
    organized = 0
    for name in sorted(touched):
        path = os.path.join(memory_dir, "topics", *name.split("/")) + ".md"
        if not os.path.exists(path):
            continue                      # 可能已被 consolidate 合并删除
        with open(path) as f:
            original = f.read()
        title_lines, content = _split_topic_file(original)
        if not content:
            continue
        raw = _unsectioned_count(original)
        if raw == 0:
            continue
        if not final and raw < min_raw:
            continue                      # 未归节行不够 → 先攒着
        numbered = "\n".join(f"[{i}] {ln}" for i, ln in enumerate(content, 1))
        text = _distill_call(
            [{"role": "system", "content": _V8_SECTIONS_PROMPT},
             {"role": "user", "content": numbered}],
            phase="v8_sections")
        if not text:
            continue
        rebuilt = _apply_sections(text, title_lines, content)
        if rebuilt is None:
            continue                      # 校验未过 → 保留原文件（_apply_sections 已 warn）
        with open(path, "w") as f:
            f.write(rebuilt)
        organized += 1
    return organized


def _apply_sections(text, title_lines, content):
    """按模型返回的分节方案重建文件内容字符串；任一校验失败返回 None（记 warning）。
    content 是编号 1..N 的内容行原文列表。校验：JSON 可解析、每编号恰好一次、
    无越界、重建后内容行多重集合与原文逐字相等。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            print("[v8_sections] JSON 解析失败，放弃整理")
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            print("[v8_sections] JSON 解析失败，放弃整理")
            return None
    sections = obj.get("sections") if isinstance(obj, dict) else None
    if not isinstance(sections, list) or not sections:
        print("[v8_sections] sections 字段缺失或为空，放弃整理")
        return None
    n = len(content)
    used, order = set(), []                   # order: [(heading, [idx...])]
    for sec in sections:
        if not isinstance(sec, dict):
            print("[v8_sections] section 非对象，放弃整理")
            return None
        heading = str(sec.get("heading", "")).strip() or "misc"
        idxs = sec.get("lines", [])
        if not isinstance(idxs, list):
            print("[v8_sections] lines 非列表，放弃整理")
            return None
        cur = []
        for r in idxs:
            try:
                k = int(str(r).strip())
            except (ValueError, TypeError):
                print("[v8_sections] 行号非整数，放弃整理")
                return None
            if not 1 <= k <= n:
                print(f"[v8_sections] 行号 {k} 越界，放弃整理")
                return None
            if k in used:
                print(f"[v8_sections] 行号 {k} 重复，放弃整理")
                return None
            used.add(k)
            cur.append(k)
        order.append((heading, cur))
    if len(used) != n:                        # 有漏行
        print(f"[v8_sections] 覆盖 {len(used)}/{n} 行，有遗漏，放弃整理")
        return None
    # 硬校验：重建内容行多重集合 == 原文（排序逐字比对）。编号唯一+全覆盖已保证，
    # 这里再显式核一次守住不变量。
    rebuilt_content = [content[k - 1] for _h, ks in order for k in ks]
    if sorted(rebuilt_content) != sorted(content):
        print("[v8_sections] 内容行多重集合与原文不等，放弃整理")
        return None
    parts = list(title_lines)
    for heading, ks in order:
        if not ks:
            continue                          # 空节丢弃
        parts.append(f"## {heading}")
        parts.extend(content[k - 1] for k in ks)
    return "\n".join(parts) + "\n"


_V8_TIDY_PROMPT = """下面是一个人记忆库里的一个话题文件的内容行，每行前面有 [编号]。这些是逐句追加的记忆句（行内 [Dx:y] 是指向原始对话的引用锚，可能含日期链接）。

你要一次做两件事，**一个字都不许改行的文本**，只输出行的编号：

1. 去重（groups）：找出**陈述同一事实的近重复行**，每组给一个 keep（信息最全的那行编号）和一个 merge 列表（与 keep 陈述同一事实、可被取代的其它行编号）。只合并真正同义的行（同一个人做的同一件事）；日期/地点/对象不同就别合，拿不准别合。一个编号最多出现在一处，不许跨组重复或自并。没有近重复就 groups 为空。

2. 分节（sections）：给这些行分组、起小标题、决定节内行序，把内容相近的行归到一节。行号仍用**上面 [编号] 的原编号**（去重前的编号），被你放进某组 merge 的行不必再写进 sections。每节 lines 按你希望的行内顺序列出编号。

只输出 JSON：{{"groups":[{{"keep":3,"merge":[7,12]}}],"sections":[{{"heading":"Adoption","lines":[3,1]}},{{"heading":"Career","lines":[2]}}]}}"""


def _apply_tidy(text, title_lines, content):
    """一次调用同时做去重+分节。返回重建后的文件字符串；任一硬守卫失败返回 None。
    行号映射选择（spec A 第二方案，实现清晰、守卫好写）：sections 用**合并前**的原编号，
    代码先按 groups 合并、拿到「存活前编号→合并后位置」映射，再按 sections 重排存活行：
    - 引用被 merge 掉的行 / 越界 / 重复引用 → 静默跳过（不算错，spec 允许漏引用被删行）
    - 任何存活行未被任何 section 引用 → 追加进末尾 `## 未整理` 节（保证不丢内容）
    强守卫全在合并那步（dia_id 并集不变、keep 文本逐字不动）；分节只是存活行的重排，
    末尾兜底节保证最终文件内容行多重集合恰等于合并后存活行集合，不可能丢行。"""
    obj = _parse_json_obj(text, "v8_tidy")
    if obj is None:
        return None
    if not isinstance(obj, dict):
        print("[v8_tidy] 顶层非对象，放弃整理")
        return None
    # ---- 1. 合并（全部原守卫）----
    merged = _apply_line_merges_mapped(json.dumps({"groups": obj.get("groups", [])}),
                                       content)
    if merged is None:
        return None                           # 合并守卫未过 → 整个文件放弃
    new_content, survivor_pos = merged        # survivor_pos: 前编号(1-based)→new 位置
    # ---- 2. 分节：把 sections 里的前编号映射到存活行位置，漏引用兜底 ----
    sections = obj.get("sections")
    if not isinstance(sections, list):
        print("[v8_tidy] sections 字段缺失，放弃整理")
        return None
    used_pos, order = set(), []               # order: [(heading, [new_pos...])]
    for sec in sections:
        if not isinstance(sec, dict):
            print("[v8_tidy] section 非对象，放弃整理")
            return None
        heading = str(sec.get("heading", "")).strip() or "misc"
        idxs = sec.get("lines", [])
        if not isinstance(idxs, list):
            print("[v8_tidy] lines 非列表，放弃整理")
            return None
        cur = []
        for r in idxs:
            try:
                k = int(str(r).strip())
            except (ValueError, TypeError):
                continue                      # 非整数行号：静默跳过
            pos = survivor_pos.get(k)         # 前编号→合并后位置
            if pos is None or pos in used_pos:
                continue                      # 被 merge 掉 / 越界 / 重复引用：静默跳过
            used_pos.add(pos)
            cur.append(pos)
        order.append((heading, cur))
    # 兜底：任何存活行没被引用 → 收进末尾 `## 未整理`，保证不丢内容
    leftover = [p for p in range(len(new_content)) if p not in used_pos]
    if leftover:
        order.append((_UNSECTIONED_HEADING[3:], leftover))
    parts = list(title_lines)
    for heading, ps in order:
        if not ps:
            continue                          # 空节丢弃
        parts.append(f"## {heading}")
        parts.extend(new_content[p] for p in ps)
    return "\n".join(parts) + "\n"


def tidy_topic_file(memory_dir, touched, final=False):
    """层2 去重 + 分节整理**合并成一次 LLM 调用**（NATIVEMEM_V8_TIDY_COMBINED=on 走此路，
    默认 on）。触发时机、阈值与拆开的 merge_duplicate_lines + organize_topic_sections
    完全一致（未归节行 ≥ NATIVEMEM_V8_REWRITE_MIN；final=True 只要有未归节行）。
    模型一次返回 {"groups":..., "sections":...}，代码先合并后分节（见 _apply_tidy）；
    任一硬守卫失败 → 放弃该文件、原样保留。返回整理的文件数。"""
    min_raw = int(os.environ.get("NATIVEMEM_V8_REWRITE_MIN", "8"))
    tidied = 0
    for name in sorted(touched):
        path = os.path.join(memory_dir, "topics", *name.split("/")) + ".md"
        if not os.path.exists(path):
            continue                          # 可能已被 consolidate 合并删除
        with open(path) as f:
            original = f.read()
        title_lines, content = _split_topic_file(original)
        if len(content) < 2:
            continue                          # 少于两行无从去重/分节
        raw = _unsectioned_count(original)
        if raw == 0:
            continue
        if not final and raw < min_raw:
            continue                          # 未归节行不够 → 先攒着
        numbered = "\n".join(f"[{i}] {ln}" for i, ln in enumerate(content, 1))
        text = _distill_call(
            [{"role": "system", "content": _V8_TIDY_PROMPT},
             {"role": "user", "content": numbered}],
            phase="v8_tidy")
        if not text:
            continue
        rebuilt = _apply_tidy(text, title_lines, content)
        if rebuilt is None:
            continue                          # 守卫未过 → 保留原文件
        with open(path, "w") as f:
            f.write(rebuilt)
        tidied += 1
    return tidied


_V8_DISTILL_PROMPT = """你把一段对话里的【所有具体事实】都提炼成事件，输出 JSON，不要复制原文。
观测日期：{obs_date}
日历（日期换算的唯一权威）：
{calendar}
已有话题名（含条目数，优先复用规模大的、别造近义新名）：{known_topics}

**目标是覆盖全，不是挑重点。** 这些记忆之后要用来回答各种细节问题，所以：
- **宁多勿漏**：每个提到的具体事实都单独成一条事件——一个人做的一件事、去的一个地方、读的一本书、买的一样东西、一个计划、一段经历、一个偏好，都要记。别把多件事塞进一条，也别因为"不重要"就跳过。
- **保留具体信息**：摘要里**必须保留原文的专有名词**（人名、地名、国家、书名、店名、品牌）、**数字**（年龄、数量、年份）、**相对时间词**（before/after/last/ago/next/the day before…）。这些正是会被追问的细节，丢了就答不出。
- **不要擅自结束持续状态**：把实体是否仍存在/持有，与它过去的用途、内容、角色或状态分开记录。`old`、`previous`、`used to`、`previously` 本身不表示该实体已不存在或不再持有；只有原文明说 sold/discarded/removed/replaced/no longer owned 等变化时，才能写成状态已结束。不确定时保留原始措辞和日期证据，不要替用户下结论。

对话每行开头有 [数字] 行号。每个事件给：
- when: 事件日期 YYYY-MM-DD（能推断就推断，推断不出用观测日期）
- summary: 一句话英文摘要，保留上面说的专有名词/数字。**必须用英文写**，和对话原文同语言，方便后续英文检索命中。两条硬要求：
  1. **内联引用（写到哪引到哪）**：每个事实点在句内**提及处**紧跟它出自的行号标记，如 `decided to adopt [1], visited an agency [3]`。一句话引多行就各自标各自的。行号标记支持三种写法：单行 `[8]`；一件事**连续聊了好几行**就用区间 `[3-15]`（含头含尾，一次覆盖整段，别只挑首尾两行）；不连续的散行用列举 `[2,10]`，也可混合 `[3-5,9]`。**区间要如实覆盖真在聊这件事的行，不许为了省事把没聊到的行也圈进去（不许虚扩）。**
  2. **相对时间换算成绝对日期写进句子**：yesterday/last week/the day before 这类，必须对照上面的日历查出具体日期，直接写进文本（如 "went hiking yesterday (2023-05-06)"）。只许查表，禁止脱离日历心算；查出的日期和原相对时间词必须并存。日历覆盖不到或语义模糊的（如 recently/a while ago）保留原词，禁止造日期。
- refs: 这条事件对应的行号列表（就是行首 [数字] 的那个数字），**至少写一个**。同样支持区间/列举：连续多行写成 `"3-15"`，散行 `[2,10]`，混合 `["3-5",9]` 都行
- topic: 这条事实在主题库里的存放路径（相对 topics/）。按"以后要找这条信息时会去哪翻"来定：可以一级（如 `adoption`），也可以多级（如 `Caroline/adoption`、`Melanie/family/kids`），你自己判断怎么组织最好找。**必须用英文**。
  **复用已有路径，禁止开同义新文件**：事件延续已有线索时，必须直接复用上面「已有话题名」列表里的现成路径，不许另造近义名。开任何新路径前，先把整个列表扫一遍，确认没有近义项——若 `adoption` 已存在，就不许再开 `adopt`/`adopting`/`adoption-plan`；只有列表里确实找不到能装下这条事实的路径时才新建。
  路径的每一级都必须是**内容**（这条事实讲的是什么事），**禁止用言语行为当话题名**：inquiry/feedback/sharing/conversation/compliments/well_wishes/question/response 一类全不许。判断标准：话题名要能回答"这条信息是关于什么的"，不是"这句话是怎么说出来的"。若一句话是 A 对 B 的某件事发表评论/夸奖/回应，归到 **B 的那件事**下（如 Melanie 夸 Caroline 适合做咨询师 → `Caroline/career_plans`，不是 `Melanie/compliments`）。

只输出 JSON：{{"events":[{{"when":"2023-05-07","summary":"Caroline visited an adoption agency [3] and liked its LGBTQ-friendly values [5]","refs":[3,5],"topic":"Caroline/adoption"}}]}}
一段对话通常能提炼出多条事件。真的什么具体事实都没有才 {{"events":[]}}。"""

def _number_chunk(turns, dia_ids):
    """给 chunk 每个 turn 加 [n] 编号，返回 (带号文本, {n: dia_id})。
    turns 是 [(speaker, text), ...] 结构化 turn 列表（split_into_chunks_structured
    产出），n 从 1，第 n 个 turn 对应 dia_ids[n-1]。

    根治「内部空行错位」：按 TURN 编号，而不是重新解析拼接后的自由文本。
    过去把 "speaker: text\\n\\n" 拼接串再按 "\\n\\n+" 切块，无法区分「turn
    之间的分隔空行」与「某个 turn 自身 text 里的空行」——含内部空行的 turn 会
    被切成 2+ 块，导致其后所有块号相对 dia_ids 整体错位（事件映射到错误
    dia_id 或被丢弃）。现在直接拿结构化 turn，一个 turn 恒为一个编号块，turn
    内部的任何换行/空行都留在这一块里，永不产生新编号：块数 == len(turns) ==
    len(dia_ids)，对齐由构造保证。"""
    numbered, line_map = [], {}
    # 防御：turn 数与 dia_ids 数不一致时，只按较短的一方对齐，不让编号漂移。
    n = min(len(turns), len(dia_ids)) if dia_ids else len(turns)
    for i, (speaker, text) in enumerate(turns, 1):
        numbered.append(f"[{i}] {speaker}: {text}")
        if i <= n:
            line_map[i] = dia_ids[i - 1]
    return "\n".join(numbered), line_map


_V8_SEGMENT_PROMPT = """下面是一段对话，每行前有 [编号]。请在**话题转换处**把它切成若干段，一段 = 一个连贯话题（连续聊同一件事的相邻句归为一段）。

严格要求：
- 只按编号切，不许改动、重排、遗漏或重复任何编号。
- 每段是**连续递增**的编号（如 [1,2,3]），段与段首尾相接、覆盖全部 [1]..[n]。
- 只输出 JSON，不要解释：{"segments": [[1,2,3],[4,5,6,7], ...]}"""


def _fixed_segments(turns, dia_ids, size):
    """按 size 固定切 (turns, dia_ids)，返回 [(turns_sub, dia_ids_sub), ...]。"""
    out = []
    for cs in range(0, len(turns), size):
        out.append((turns[cs:cs + size], dia_ids[cs:cs + size]))
    return out


def _cap_segment(seg, cap):
    """一段编号列表超过 cap 句时按 cap 硬切成多段。"""
    return [seg[i:i + cap] for i in range(0, len(seg), cap)]


def segment_session_by_topic(turns, dia_ids, max_retry=6):
    """用一次 LLM 调用按话题切分一个 session，返回 [(turns_sub, dia_ids_sub), ...]，
    与 split_into_chunks_structured 同构（可直接进 distill 循环）。

    turns 是 [(speaker, text), ...]，dia_ids 一一对应。把句子编号 [1]..[n] 发给模型，
    模型在话题转换处切段。守卫：JSON 可解析；每个编号恰出现一次；段内连续递增；
    段间顺序衔接。任一违反或 JSON 坏 → 回退固定 6 句切块。

    段长上限 NATIVEMEM_V8_SEGMENT_MAX（默认 10）：超长段代码再按上限硬切。"""
    fixed_size = int(os.environ.get("NATIVEMEM_CHUNK_TURNS", "6"))
    cap = int(os.environ.get("NATIVEMEM_V8_SEGMENT_MAX", "10"))
    n = min(len(turns), len(dia_ids))
    if n == 0:
        return []
    turns, dia_ids = turns[:n], dia_ids[:n]

    numbered, _ = _number_chunk(turns, dia_ids)
    text = _distill_call(
        [{"role": "system", "content": _V8_SEGMENT_PROMPT},
         {"role": "user", "content": numbered}],
        phase="v8_segment", max_retry=max_retry)
    obj = _parse_json_obj(text, "v8_segment") if text else None
    segments = obj.get("segments") if isinstance(obj, dict) else None

    # 守卫：segments 是 list[list[int]]；每编号 1..n 恰一次；段内连续递增；段间衔接。
    if not isinstance(segments, list) or not segments:
        print("[v8_segment] segments 缺失/为空，回退固定切块")
        return _fixed_segments(turns, dia_ids, fixed_size)
    expected = 1
    for seg in segments:
        if not isinstance(seg, list) or not seg:
            print("[v8_segment] 段非法，回退固定切块")
            return _fixed_segments(turns, dia_ids, fixed_size)
        for k, num in enumerate(seg):
            if not isinstance(num, int) or num != expected:
                print("[v8_segment] 编号漏/重/非连续，回退固定切块")
                return _fixed_segments(turns, dia_ids, fixed_size)
            expected += 1
    if expected != n + 1:
        print("[v8_segment] 覆盖不全，回退固定切块")
        return _fixed_segments(turns, dia_ids, fixed_size)

    out = []
    for seg in segments:
        for piece in _cap_segment(seg, cap):
            idx = [num - 1 for num in piece]
            out.append(([turns[i] for i in idx], [dia_ids[i] for i in idx]))
    return out


def _refs_to_dia_ids(refs, line_map):
    """把模型给的行号列表（int/str，元素可是区间 "3-5" 或列举）转成 dia_id，去重保序，
    丢无效号。距离型区间在这里就地展开：一件事连聊了 3~15 行，模型写 "3-15" 即可，
    不必逐个列。"""
    if not isinstance(refs, list):
        refs = [refs]
    out, seen = [], set()

    def take(n):
        did = line_map.get(n)
        if did and did not in seen:
            seen.add(did)
            out.append(did)

    for r in refs:
        s = str(r).strip()
        if "-" in s.lstrip("-"):                  # 行号区间 "3-15"（负号开头不算区间）
            a, _, b = s.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            for n in range(lo, hi + 1):
                take(n)
            continue
        try:
            take(int(s))
        except ValueError:
            continue
    return out


_WHEN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# 笔记文本里内嵌的绝对日期（v9.0c compose 用它把 when 从观测日回填到事实发生日）。
_DATE_IN_TEXT_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _earliest_date_in_notes(nums, note_by_n):
    """扫这些笔记编号的文本，返回里面出现的最早绝对日期 YYYY-MM-DD（字典序=时间序），
    一个都没有返回 None。只认换算好的绝对日期，"in 2022" 这类只有年份的不管（靠 prompt）。"""
    found = []
    for n in nums:
        for d in _DATE_IN_TEXT_RE.findall(note_by_n.get(n, "")):
            found.append(d)
    return min(found) if found else None

# 句内行号标记：[1] / [1,3] / 区间 [3-15] / 混合 [3-5,9]（distill 在提及处内联引用）
_MARKER_RE = re.compile(r"\[(\d+(?:[-,]\d+)*)\]")


def _tidy_spaces(s):
    return re.sub(r"\s+([,.;!?])", r"\1", re.sub(r"\s{2,}", " ", s)).strip()


def _inline_refs(summary, line_map):
    """把 summary 句内的 [n]/[n,m] 行号标记换成真实 dia_id。
    返回 (plain, inline, marker_dia_ids)：
    - plain：去掉全部标记的纯文本（timeline 摘要、专名校验用）
    - inline：标记替换成 [Dx:y, ...] 的文本（topics 视图用）；无效行号的标记整个移除
    - marker_dia_ids：句内标记解析出的 dia_id（去重保序，供并进 dia_ids）"""
    extra, seen = [], set()

    def repl(m):
        dids = _refs_to_dia_ids(m.group(1).split(","), line_map)
        for d in dids:
            if d not in seen:
                seen.add(d)
                extra.append(d)
        return format_dia_refs(dids) if dids else ""

    inline = _tidy_spaces(_MARKER_RE.sub(repl, summary))
    plain = _tidy_spaces(_MARKER_RE.sub("", summary))
    return plain, inline, extra

# 首词过滤：句首必大写、代词、常见虚词——大写不代表专名，全排除。
_STOPWORDS = {
    "I", "A", "An", "The", "He", "She", "It", "We", "They", "You", "My", "Your",
    "His", "Her", "Its", "Our", "Their", "This", "That", "These", "Those",
    "And", "But", "Or", "So", "If", "When", "While", "Then", "Now", "Here",
    "There", "What", "Who", "How", "Why", "Where", "Yes", "No", "Oh", "Well",
    "Do", "Did", "Does", "Is", "Are", "Was", "Were", "Have", "Has", "Had",
    "Will", "Would", "Can", "Could", "Should", "May", "Might", "Must",
}
# 大写开头连续词组（"Becoming Nicole"、"Paris"）；两侧非字母边界靠 findall 天然处理。
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*")
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,40})[\"'“”‘’]")
_NUMBER_RE = re.compile(r"\b\d+\b")


def verify_event_coverage(turns, events):
    """§2.1 硬校验：代码从 chunk 原文抽候选专名/数字，对照所有 event 的 summary
    （拼接后不区分大小写查找），返回**原文有、摘要全都没覆盖**的候选词列表。
    简单启发式，不追求精确 NER：大写开头词组（滤句首词/说话人名）、引号短语、数字。"""
    speaker_names = {s for s, _ in turns}
    summaries = " ".join(str(e.get("summary", "")) for e in events).lower()

    cands, seen = [], set()

    def add(c):
        c = c.strip()
        if not c or c in seen:
            return
        seen.add(c)
        cands.append(c)

    for _speaker, text in turns:
        for m in _PROPER_RE.findall(text):
            # 多词组整体保留；单个词若是停用词/说话人名则丢
            words = m.split()
            if len(words) == 1 and (words[0] in _STOPWORDS or words[0] in speaker_names):
                continue
            add(m)
        for q in _QUOTED_RE.findall(text):
            add(q)
        for n in _NUMBER_RE.findall(text):
            add(n)

    # 覆盖判定：候选（小写）作为子串出现在任一 summary 里就算覆盖。
    return [c for c in cands if c.lower() not in summaries]


def _parse_distill_response(text, obs_date, line_map, dia_dates=None):
    """把模型返回的 JSON 文本解析成规范化 events（when 校验 + refs→dia_ids）。
    解析失败或无有效事件返回 []。首轮/补抽轮共用这段规范化逻辑。"""
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
    if not isinstance(events, list):
        return []
    out = []
    for e in events:
        if not isinstance(e, dict) or not e.get("summary"):
            continue
        # when 必须是 YYYY-MM-DD；模型有时会把相对时间词（"friday"、"last week"）
        # 塞进 when，那样 timeline_path 的 when.split("-") 会崩。非法一律回填
        # obs_date（相对时间信息本就该留在 summary 里，不进日期字段）。
        dia_ids = _refs_to_dia_ids(e.get("refs", []), line_map)
        w = str(e.get("when", "")).strip()
        fallback_date = obs_date
        if dia_dates:
            fallback_date = next(
                (dia_dates[dia_id] for dia_id in dia_ids if dia_id in dia_dates),
                obs_date,
            )
        e["when"] = w if _WHEN_RE.match(w) else fallback_date
        # 句内标记 → 真实 dia_id；标记引到的 id 并进 dia_ids（refs 是校验字段，
        # 两边取并集，timeline 锚不缺）。
        plain, inline, extra = _inline_refs(str(e["summary"]), line_map)
        for d in extra:
            if d not in dia_ids:
                dia_ids.append(d)
        e["dia_ids"] = dia_ids
        if not e["dia_ids"]:
            continue                      # refs+标记都转不出 dia_id → 丢弃
        e["summary"] = plain
        # 句内无标记：回退为句末追加（与老 topics 行体一致）
        e["summary_inline"] = (inline if extra
                               else f"{plain} · {format_dia_refs(dia_ids)}")
        e.setdefault("topic", "misc")
        out.append(e)
    return out


def _distill_call(messages, phase="v8_distill", max_retry=6, model=None):
    """调模型（重试退避），返回 message.content 字符串；全部失败返回 None。
    model=None 时用全局 ALIYUN_MODEL（老调用方全部走这条，行为一字不变）；
    v9.0 两级建库各自传 NATIVEMEM_V9_SCRIBE_MODEL / _COMPOSER_MODEL 覆盖模型名，
    但仍复用同一个全局 client（v9.0 不做多 client，两级模型须同服务商 base/key）。"""
    model = model or ALIYUN_MODEL
    fail_fast = os.environ.get("NATIVEMEM_FAIL_FAST") == "1"
    for retry in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.2)
            break
        except Exception:  # noqa: BLE001
            if fail_fast:
                raise
            if retry < max_retry - 1:
                time.sleep(3 * (retry + 1))
            else:
                return None
    log_usage(resp, phase=phase)
    return resp.choices[0].message.content or ""


def distill_events(turns, obs_date, dia_ids, known_topics=None, recent=None,
                   max_retry=6):
    """turns 是 [(speaker, text), ...] 结构化 turn 列表（split_into_chunks_structured
    产出）。按 turn 编号，块 n ↔ dia_ids[n-1] 由构造对齐，不再重解析拼接文本。

    recent：本 session 前面 chunk 已提炼句子的尾部（滚动上下文，内存变量不读盘），
    用于防重复提炼、增量沿用同一 topic、指代消解。

    §2.1 硬校验（开关 NATIVEMEM_V8_VERIFY，默认 on）：首轮解析成功后代码对照原文
    专名/数字，若有摘要没覆盖的，连同原 chunk 再问一次补抽，合并两轮结果。补抽只
    做一轮、失败不重试，避免 build 变慢。"""
    kt = ", ".join(known_topics) if known_topics else "（暂无）"
    cal = calendar_strip(obs_date) or _V9_CALENDAR_FALLBACK
    prompt = _V8_DISTILL_PROMPT.format(obs_date=obs_date, calendar=cal,
                                       known_topics=kt)
    if recent:
        prompt += ("\n\n## 本 session 前面已提炼的记忆（滚动上下文）\n"
                   + "\n".join(f"- {s}" for s in recent)
                   + "\n上面已记过的事**别重复**提炼；同一件事的新进展沿用同一 "
                     "topic 路径写增量；原文里的指代（he/she/it/there…）按这些"
                     "上下文**消解**成具体人名/地名再写进摘要。")
    numbered_text, line_map = _number_chunk(turns, dia_ids)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": numbered_text}]
    text = _distill_call(messages, max_retry=max_retry)
    if text is None:
        return []
    out = _parse_distill_response(text, obs_date, line_map)
    first_pass = [dict(event) for event in out]

    if os.environ.get("NATIVEMEM_V8_VERIFY", "on") == "off":
        _write_distill_trace({
            "observation_date": obs_date,
            "turn_count": len(turns),
            "input_dia_ids": list(dia_ids),
            "first_pass_events": first_pass,
            "missing_coverage_points": [],
            "verify_events": [],
            "post_verify_events": [dict(event) for event in out],
            "verify_enabled": False,
        })
        return out
    missing = verify_event_coverage(turns, out)
    if not missing:
        _write_distill_trace({
            "observation_date": obs_date,
            "turn_count": len(turns),
            "input_dia_ids": list(dia_ids),
            "first_pass_events": first_pass,
            "missing_coverage_points": [],
            "verify_events": [],
            "post_verify_events": [dict(event) for event in out],
            "verify_enabled": True,
        })
        return out
    # 补抽一轮：把漏掉的具体信息点出来，连同原 chunk 再问一次（只一轮，不重试）。
    followup = ("以下原文里的具体信息（专有名词/数字）在你上一轮的事件里漏了："
                + "、".join(missing)
                + "。请把每一条都补成对应的事件（同样给 when/summary/refs/topic），"
                  "只输出补充的事件 JSON：{\"events\":[...]}。")
    text2 = _distill_call(messages + [{"role": "assistant", "content": text},
                                      {"role": "user", "content": followup}],
                          phase="v8_distill_verify", max_retry=1)
    verify_events = []
    if text2:
        verify_events = _parse_distill_response(text2, obs_date, line_map)
        out.extend(verify_events)
    _write_distill_trace({
        "observation_date": obs_date,
        "turn_count": len(turns),
        "input_dia_ids": list(dia_ids),
        "first_pass_events": first_pass,
        "missing_coverage_points": list(missing),
        "verify_events": [dict(event) for event in verify_events],
        "post_verify_events": [dict(event) for event in out],
        "verify_enabled": True,
    })
    return out


# ==================== v9.0 两级建库流水线 ====================
# 开关 NATIVEMEM_V9_PIPELINE=two_tier 时 build 走这两级（transcribe → compose）；
# 默认 off，走既有切块 distill（一字节不变）。第一级逐句忠实转写（每句一条自包含
# 笔记），第二级从整 session 笔记流里谱曲成事件。两级各自的模型由
# NATIVEMEM_V9_SCRIBE_MODEL / NATIVEMEM_V9_COMPOSER_MODEL 指定（默认取 BUILDER_MODEL），
# 但共用同一个全局 client（不做多 client，两级模型须同服务商）。

def calendar_strip(obs_date, days_back=21):
    """生成注入转写 prompt 的**纯日历条**——只有日期与星期这类确定性日历事实,
    不含任何相对时间短语(不写 yesterday/last Friday 这种词,那是语言规则,归模型)。
    模型对照日历自己查"上周五是哪天":查表是阅读不是算术,星期推算/跨月跨年进位
    全部由 datetime 保证。非法日期返回空串(调用方注入安全占位)。"""
    from datetime import date, timedelta
    try:
        d = date.fromisoformat(str(obs_date).strip())
    except (ValueError, TypeError):
        return ""
    cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    en = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = [f"对话日期：{d.isoformat()}（{cn[d.weekday()]}/{en[d.weekday()]}）",
             "此前三周的日历（由近到远）："]
    for i in range(1, days_back + 1):
        x = d - timedelta(days=i)
        lines.append(f"{x.isoformat()} {cn[x.weekday()]}/{en[x.weekday()]}")
    # 两行日历事实(仍是日期,不是短语):上一自然周区间、上月与去年
    mon_this = d - timedelta(days=d.weekday())
    prev_mon, prev_sun = mon_this - timedelta(days=7), mon_this - timedelta(days=1)
    lines.append(f"上一个自然周：{prev_mon.isoformat()}（星期一）~ {prev_sun.isoformat()}（星期日）")
    prev_month_last = d.replace(day=1) - timedelta(days=1)
    lines.append(f"上月：{prev_month_last.strftime('%Y-%m')}  去年：{d.year - 1}")
    return "\n".join(lines)


_V9_CALENDAR_FALLBACK = "（本次无日历——相对时间一律保留原词，禁止自行造日期。）"

_V9_TRANSCRIBE_PROMPT = """下面是一段对话，每行开头有 [编号]，格式 `[n] 说话人: 原话`。这段对话发生在 {obs_date}。

**日历（唯一权威）**：
{calendar}

请对**每一个编号句**逐句忠实转写成笔记——**目标是每句都有笔记、每个事实都不漏**，不是挑重点。

**一句话说了几个独立事实，就为它出几条笔记**，一条笔记只装一个事实。同一个编号可以出现多条笔记（同一 `line` 重复即可）。若一句话只讲一件事，就一条笔记。别把一句话里的多个事实揉进一条笔记。

每条笔记只装一个事实，且满足：
- **自包含**：句子里的代词（he/she/it/they/there/this…）按上下文消解成具体的人名或物名；说话人的"我/你/my/your"换成具体说话人名（如 Melanie 说的"my daughter"写成"Melanie's daughter"）。读这条笔记的人没有原文也能懂它在讲谁、讲什么。
- **忠实转写，不解读**：保留原文的专有名词（人名/地名/书名/店名/品牌）、数字、引号内的原话。只把这句话说了什么如实记下来，别推断言外之意、别合并别的句子的信息。

**尤其要保住这些"排位靠后"最容易被丢的事实**（它们往往是一句话里的第二、三个信息，习惯性只写第一个就会漏）：
- **并列的第二/第三个事实**：一句话往往先说一件硬事实，再带出情感、象征意义、人物关系（如 "I adopted a dog, it means a lot to me and reminds me of my late father"——除了"领养了狗"，还有"这对她意义重大""让她想起已故的父亲"两条，各出一条笔记）。这些软事实必须各自成条，不许因为"是附带的"就并掉或丢掉。
- **相对时间/事件锚点词**：句子里的相对时间表达（昨天、上周、上周五、两周前的周末……任何语言的说法）**对照上面的日历查出具体日期**写进笔记——只许查表，**禁止脱离日历心算**（数星期、倒推天数这类算术全靠日历读出来）。日历覆盖不到或语义模糊的（"a while ago"、"recently"、"soon"）**保留原词，禁止造日期**。查出的日期与原相对短语**并存**（如 "went hiking last Friday (2023-07-21)"）。**以事件为锚的相对短语**（"after the road trip"、"before the wedding"…）查不出日期，必须把锚点短语**原样留在某条笔记里**，不许丢。
- **引号内的原话**：说话人引用的原话（"...he said '...'"）照原样保留在笔记里。
- **寒暄句也要有笔记**：问候、道谢、附和、告别这类没什么信息量的句子同样要出一条笔记（如 "Melanie 问候了 Caroline"、"Caroline 谢了 Melanie"），**绝不许因为"不重要"就跳过某个编号**。

只输出 JSON，笔记用英文写，同一 line 可重复出现多条（一条一个事实），每个编号至少一条：
{{"notes":[{{"line":1,"note":"..."}},{{"line":1,"note":"..."}},{{"line":2,"note":"..."}}]}}"""


def _parse_notes(text, valid_lines):
    """解析转写返回的 {"notes":[{"line":n,"note":"..."}]}，返回 {line:int -> [note:str, ...]}，
    只保留 line 落在 valid_lines（本轮发出去的合法编号集）里、note 非空的项。
    v9.0c：一句可出多条笔记（一条一个事实），同一 line 的多条按出现顺序保序累积。
    JSON 坏或 notes 缺失返回 {}。"""
    obj = _parse_json_obj(text, "v9_transcribe")
    notes = obj.get("notes") if isinstance(obj, dict) else None
    if not isinstance(notes, list):
        return {}
    out = {}
    for it in notes:
        if not isinstance(it, dict):
            continue
        try:
            ln = int(str(it.get("line", "")).strip())
        except (ValueError, TypeError):
            continue
        note = str(it.get("note", "")).strip()
        if ln in valid_lines and note:
            out.setdefault(ln, []).append(note)
    return out


def _transcribe_batch(turns, dia_ids, obs_date, model, max_retry):
    """转写一批句子（≤ batch 大小），返回 {行号(1..n) -> [note, ...]}，行号相对本批。
    v9.0c：一句可对应多条笔记（一条一个事实）。覆盖判定是"每号至少一条"——缺行=一条都没有的行。
    首轮全量发；缺行 → 只把缺的句子(带前后 2 句上下文，上下文句仍带原编号)再发一轮
    补转写，最多补 2 轮；仍缺的行由调用方用原句原文兜底（转写层不许丢句）。"""
    numbered, line_map = _number_chunk(turns, dia_ids)   # line_map: 行号 -> dia_id
    n = len(line_map)
    if n == 0:
        return {}
    all_lines = set(range(1, n + 1))
    cal = calendar_strip(obs_date) or _V9_CALENDAR_FALLBACK
    sys_prompt = _V9_TRANSCRIBE_PROMPT.format(obs_date=obs_date, calendar=cal)
    text = _distill_call(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": numbered}],
        phase="v9_transcribe", max_retry=max_retry, model=model)
    notes = _parse_notes(text, all_lines) if text else {}

    numbered_lines = numbered.split("\n")            # 第 i 行即编号 i 的带号原文
    for _ in range(2):                               # 最多补 2 轮
        missing = sorted(all_lines - set(notes))
        if not missing:
            break
        # 缺行连同前后 2 句上下文（带原编号）再发一轮，只要求补齐缺的编号。
        ctx = sorted({j for m in missing
                      for j in range(max(1, m - 2), min(n, m + 2) + 1)})
        ctx_text = "\n".join(numbered_lines[j - 1] for j in ctx)
        want = "、".join(str(m) for m in missing)
        user = (f"下面这些编号句上一轮漏了笔记，请只为这些编号补出笔记"
                f"（编号：{want}），规则同前，其余编号是上下文不用输出：\n{ctx_text}")
        text2 = _distill_call(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user}],
            phase="v9_transcribe", max_retry=1, model=model)
        add = _parse_notes(text2, set(missing)) if text2 else {}
        if not add:
            break                                    # 补不出就别再空转
        notes.update(add)
    return notes


def transcribe_session(turns, dia_ids, obs_date, max_retry=6):
    """v9.0 第一级：把整个 session 逐句忠实转写成自包含笔记。
    turns 是 [(speaker, text), ...]，dia_ids 一一对应，obs_date 是 session 日期。
    一次（或超长时按批多次）LLM 调用，每个编号句产出一条笔记：代词/我你消解成具体名、
    相对时间就地换算成绝对日期、保留专名数字引号、寒暄句也不跳。返回 [(dia_id, note), ...]
    按原顺序，**长度恒等于输入句数**——模型漏的行用原句原文兜底，转写层不丢句。

    v9.0c：一句可产出多条笔记（一条一个事实，象征/情感/关系类软事实、事件锚点短语各自成条），
    同一 dia_id 在返回里对应多个 (dia_id, note)（下游 line_map 按笔记序号编号，天然支持）。

    超长 session（句数 > NATIVEMEM_V9_SCRIBE_BATCH，默认 50）按 50 句机械切批（批间无
    语义依赖，只控输出长度），逐批转写后按原顺序拼回。模型经 NATIVEMEM_V9_SCRIBE_MODEL
    指定（默认 BUILDER_MODEL），走全局 client。"""
    n = min(len(turns), len(dia_ids))
    if n == 0:
        return []
    turns, dia_ids = list(turns[:n]), list(dia_ids[:n])
    model = os.environ.get("NATIVEMEM_V9_SCRIBE_MODEL") or ALIYUN_MODEL
    if os.environ.get("NATIVEMEM_V9_SCRIBE_MODE", "batch") == "per_turn":
        return _transcribe_per_turn(turns, dia_ids, obs_date, model, max_retry)
    batch = int(os.environ.get("NATIVEMEM_V9_SCRIBE_BATCH", "50"))
    out = []
    for cs in range(0, n, batch):
        bt, bd = turns[cs:cs + batch], dia_ids[cs:cs + batch]
        notes = _transcribe_batch(bt, bd, obs_date, model, max_retry)
        for i, (did, (spk, txt)) in enumerate(zip(bd, bt), 1):
            # 一句多条：按序号 i 取该句的所有笔记，同一 did 展开成多条；
            # 兜底：模型一条都没给时，用原句原文当唯一笔记（转写层保真、绝不丢句）。
            for note in notes.get(i) or [f"{spk}: {txt}"]:
                out.append((did, note))
    return out


def _transcribe_per_turn(turns, dia_ids, obs_date, model, max_retry):
    """一句一句读(NATIVEMEM_V9_SCRIBE_MODE=per_turn):每句单独一次调用,只为当前句
    出笔记;前 K 句(NATIVEMEM_V9_TURN_CONTEXT,默认 5)作为只读上下文供代词消解。
    规则与批量转写同一套(_V9_TRANSCRIBE_PROMPT:一句多笔记/日历查表/软事实锚点词)。
    句间无依赖(上下文用原文不用笔记),按序串行即可——session 间已有并行。
    兜底同批量:模型没给笔记的句子用原句原文顶上,不丢句。"""
    k = int(os.environ.get("NATIVEMEM_V9_TURN_CONTEXT", "5"))
    cal = calendar_strip(obs_date) or _V9_CALENDAR_FALLBACK
    sys_prompt = _V9_TRANSCRIBE_PROMPT.format(obs_date=obs_date, calendar=cal)
    out = []
    for i, (did, (spk, txt)) in enumerate(zip(dia_ids, turns), 1):
        ctx = "\n".join(f"（上下文）{s}: {t}" for s, t in turns[max(0, i - 1 - k):i - 1])
        user = ((ctx + "\n") if ctx else "") + \
            f"[1] {spk}: {txt}\n\n只为上面这个编号句 [1] 出笔记(可多条),上下文句不用输出。"
        text = _distill_call(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user}],
            phase="v9_transcribe_turn", max_retry=max_retry, model=model)
        notes = _parse_notes(text, {1}) if text else {}
        for note in notes.get(1) or [f"{spk}: {txt}"]:
            out.append((did, note))
    return out


_V9_COMPOSE_PROMPT = """下面是一个人某天对话的**逐句笔记流**，每行开头有 [编号]，是已经消解好代词、换算好日期的自包含笔记。观测日期：{obs_date}
已有话题名（含条目数，优先复用规模大的、别造近义新名）：{known_topics}

你的任务：给每一条笔记**逐条记账**——每个编号都要显式给一个去向，再把这些编号组织成事件，输出 JSON。这不是"挑重点"：你不做重要性判断，只做"这条笔记属于哪个事件"的归位。

**每一条笔记编号都必须恰好出现一次**（`notes` 数组里）：
- 纯问候/道谢/附和/告别这类**零信息量**的笔记，`do` 写 `"skip"`；
- 其余**一律**指向某个事件 id（`"E1"`/`"E2"`…），一条不许漏。你只需要判断"是不是纯寒暄"，不是"重不重要"；拿不准就别 skip，指给一个事件。

**一个事件 = 一个原子事实**：一个事件只装讲**同一个原子事实**的笔记。合并只限于**同一事实的重复陈述**——两条笔记说的是同一件事的同一个点（同一件事换句话再说一遍、同一句话的补充说明）才能并进一个事件。**禁止把一段连续对话里的多个不同事实并进一个事件**：一个人今天连着讲了三件事，就是三个事件，宁可 10 个小事件，也不要 1 个把不同事实塞满从句的胖事件。象征意义、情感、人物关系这类软事实，只要是**独立的一点**，就各自成事件，不许降级成某个大事件的从句。

每个事件给：
- id: `"E1"`、`"E2"`… 与 `notes` 里 `do` 认领它的编号对应。
- when: 事件日期 YYYY-MM-DD。**若认领这个事件的笔记文本里写了绝对日期（YYYY-MM-DD 或某个年份），when 用那个日期**（事实真正发生的那天）；有多个取最早的；一条绝对日期都没有才用观测日期 {obs_date}。禁止自己推算新日期。
- summary: **一句短句**英文摘要，只讲这一个原子事实，**保留笔记里的专有名词、数字、事件锚点短语（after the road trip 之类）**。**禁止从句堆叠**——不要把别的事实塞进 which/where/and 从句里凑成长句；那些事实该另立事件。在这个事实点的**提及处**内联引用它出自的笔记编号，如 `decided to adopt [1]`；同一事实由连续几条笔记讲可用区间 `[3-6]`，不连续的散条用列举 `[2,10]` 或混合 `[3-5,9]`。**内联引用的编号必须来自本事件在 `notes` 里认领的编号**。
- topic: 这条事实在主题库里的存放路径（相对 topics/），**必须用英文**。按"以后要找这条信息会去哪翻"来定，可一级（如 `adoption`）也可多级（如 `Caroline/adoption`、`Melanie/family/kids`）。
  **复用已有路径，禁止开同义新文件**：事件延续已有线索时，直接复用上面「已有话题名」里的现成路径，不许另造近义名。开新路径前先扫一遍列表确认没有近义项——若 `adoption` 已存在，就不许再开 `adopt`/`adopting`/`adoption-plan`。
  路径每一级都必须是**内容**（这条事实讲的是什么事），**禁止用言语行为当话题名**：inquiry/feedback/sharing/conversation/compliments/well_wishes/question/response 一类全不许。判断标准：话题名要能回答"这条信息是关于什么的"，不是"这句话是怎么说出来的"。若一句话是 A 对 B 的某件事发表评论/夸奖/回应，归到 **B 的那件事**下（如 Melanie 夸 Caroline 适合做咨询师 → `Caroline/career_plans`，不是 `Melanie/compliments`）。

只输出 JSON，两部分：
{{"notes":[{{"n":1,"do":"skip"}},{{"n":2,"do":"E1"}},{{"n":3,"do":"E1"}},{{"n":4,"do":"E2"}}],
 "events":[{{"id":"E1","when":"2023-05-07","summary":"Caroline visited an adoption agency [2] and liked its LGBTQ-friendly values [3]","topic":"Caroline/adoption"}},{{"id":"E2","when":"2023-05-07","summary":"Caroline bought a book [4]","topic":"Caroline/reading"}}]}}
`notes` 必须覆盖 1..{n_notes} 每个编号恰好一次。全是纯寒暄时，`notes` 里每条都 `"skip"`，`events` 为 `[]`。"""


def _compose_parse(text, obs_date, line_map, n_notes, note_by_n=None):
    """把 compose 返回的**逐条记账** JSON 解析成与 distill_events 同结构的 events。
    协议两部分：notes=[{{"n":编号,"do":"skip"|"E<k>"}}] 给每条笔记的去向；
    events=[{{"id":"E<k>","when","summary","topic"}}]。返回 (events_out, missing_numbers)。

    when 回填（v9.0c）：事件 when 解析成 obs_date（模型没给绝对日期）时，若认领它的笔记
    文本里匹配到绝对日期且 ≠obs_date，以其中最早那个改写 when（事实发生日）。note_by_n
    是 {{编号 -> 笔记文本}}，缺省不回填。只年份的（"in 2022"）不代码改，靠 prompt。

    记账校验：1..n_notes 每个编号必须在 notes 里恰好出现一次——缺的、do 指向不存在事件 id 的、
    以及整段 JSON 坏掉时的全部编号，都算"没交代"进 missing_numbers（供调用方补一轮）。
    重复/越界（>n_notes 或 <1）的编号忽略，不影响其它编号的一次性认领。
    每个事件的 dia_ids = **认领它的笔记编号并集**（映射回 dia_id）∪ summary 内联引到的 dia_id
    ——比只靠内联更全。无人认领的事件丢弃；summary 内联引到本事件认领集之外的编号只记 warning。"""
    all_nums = set(range(1, n_notes + 1))
    obj = _parse_json_obj(text, "v9_compose")
    if not isinstance(obj, dict):
        return [], sorted(all_nums)               # JSON 坏 → 全体没交代
    raw_events = obj.get("events") if isinstance(obj.get("events"), list) else []
    events_by_id = {str(e.get("id", "")).strip(): e
                    for e in raw_events if isinstance(e, dict) and e.get("id")}

    # notes 记账：编号 -> 去向；统计每个编号出现次数（判"恰好一次"）。
    claims = {}                                   # event_id -> [认领编号...]（保序）
    seen_counts = {}
    accounted = set()                             # 出现且去向合法（skip 或已存在事件）的编号
    raw_notes = obj.get("notes") if isinstance(obj.get("notes"), list) else []
    for it in raw_notes:
        if not isinstance(it, dict):
            continue
        try:
            n = int(str(it.get("n", "")).strip())
        except (ValueError, TypeError):
            continue
        if n not in all_nums:
            continue                              # 越界编号忽略
        seen_counts[n] = seen_counts.get(n, 0) + 1
        if seen_counts[n] > 1:
            continue                              # 重复认领只认第一次
        do = str(it.get("do", "")).strip()
        if do == "skip":
            accounted.add(n)
        elif do in events_by_id:
            claims.setdefault(do, []).append(n)
            accounted.add(n)
        # do 指向不存在的事件 id：不计入 accounted → 落进 missing

    missing_numbers = sorted(all_nums - accounted)

    out = []
    for eid, e in events_by_id.items():
        nums = claims.get(eid)
        if not nums:
            continue                              # 无人认领 → 丢弃（没出处，timeline 锚缺）
        if not e.get("summary"):
            continue
        w = str(e.get("when", "")).strip()
        w = w if _WHEN_RE.match(w) else obs_date
        # 兜底：when 落到观测日，但认领笔记文本里有绝对日期 → 用最早那个当事实发生日。
        if w == obs_date and note_by_n:
            d = _earliest_date_in_notes(nums, note_by_n)
            if d and d != obs_date:
                w = d
        e["when"] = w
        plain, inline, extra = _inline_refs(str(e["summary"]), line_map)
        claimed_dids = [line_map[n] for n in nums if n in line_map]
        # dia_ids = 认领并集 ∪ 内联引到的，保序去重。
        dia_ids, seen = [], set()
        for d in claimed_dids + extra:
            if d and d not in seen:
                seen.add(d)
                dia_ids.append(d)
        stray = [d for d in extra if d not in set(claimed_dids)]
        if stray:
            print(f"[v9_compose] event {eid} summary 内联引到认领集之外的笔记 {stray}（只警告不拒）")
        e["dia_ids"] = dia_ids
        e["summary"] = plain
        e["summary_inline"] = inline
        e.setdefault("topic", "misc")
        out.append(e)
    return out, missing_numbers


def compose_events(notes, obs_date, known_topics=None, max_retry=6):
    """v9.0 第二级：把一个 session 的逐句笔记谱成事件。
    notes 是 transcribe_session 产出的 [(dia_id, note), ...]（已全 session 见齐，
    故新路径下不需要滚动 recent）。一次 LLM 调用，笔记带 [编号] 发给模型。
    v9.0c 原子纪律：一个事件=一个原子事实，只合并"同一事实的重复陈述"，禁止把连续对话的
    多个不同事实并成胖事件；when 优先取认领笔记里的绝对日期（事实发生日）。
    模型内联引用笔记编号，代码把编号映射回 dia_id，产出与 distill_events 相同
    结构的 events（summary/summary_inline/when/dia_ids/topic），可直接喂 write_events。

    逐条记账：每条笔记编号必须显式给去向（skip 或某事件 id）。首轮后若有编号没交代
    （缺号/去向指向不存在的事件）→ 把这些编号连同原文发回去补一轮（可新开事件）；补后仍缺的
    按 skip 处理并打 warning（原文兜底在转写层，不算丢）。
    覆盖校验（复用 verify_event_coverage 思路）在记账校验之后照跑：对笔记全文做专名/数字覆盖
    检查，缺了再发一轮补谱。模型经 NATIVEMEM_V9_COMPOSER_MODEL 指定（默认 BUILDER_MODEL）。"""
    if not notes:
        return []
    n_notes = len(notes)
    line_map = {i: did for i, (did, _note) in enumerate(notes, 1)}
    note_by_n = {i: note for i, (_did, note) in enumerate(notes, 1)}
    numbered = "\n".join(f"[{i}] {note}" for i, note in note_by_n.items())
    kt = ", ".join(known_topics) if known_topics else "（暂无）"
    sys_prompt = _V9_COMPOSE_PROMPT.format(obs_date=obs_date, known_topics=kt,
                                           n_notes=n_notes)
    model = os.environ.get("NATIVEMEM_V9_COMPOSER_MODEL")
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": numbered}]
    text = _distill_call(messages, phase="v9_compose", max_retry=max_retry, model=model)
    if text is None:
        return []
    out, missing_nums = _compose_parse(text, obs_date, line_map, n_notes, note_by_n)

    # 记账修补：有编号没交代 → 把缺的编号连原文发回去，要求逐条给 do（可新开事件），1 轮。
    if missing_nums:
        miss_lines = "\n".join(f"[{n}] {note_by_n[n]}" for n in missing_nums
                               if n in note_by_n)
        want = "、".join(str(n) for n in missing_nums)
        repair = (f"这些编号你上一轮没交代去向（编号：{want}）。请只为这些编号补出记账，"
                  f"逐条给 do（纯寒暄 skip，否则指向事件 id，可沿用已有 id 或新开 E<k>），"
                  f"新开事件就在 events 里带上：\n{miss_lines}\n"
                  f"只输出 JSON：{{\"notes\":[...],\"events\":[...]}}，notes 只含这些编号。")
        text_r = _distill_call(messages + [{"role": "assistant", "content": text},
                                           {"role": "user", "content": repair}],
                               phase="v9_compose_repair", max_retry=1, model=model)
        if text_r:
            add, _still = _compose_parse(text_r, obs_date, line_map, n_notes, note_by_n)
            # 修补轮只解析这批缺号的去向：只并入认领了缺号的事件，避免误收无关号。
            miss_set = set(missing_nums)
            for e in add:
                if any(line_map.get(n) in set(e.get("dia_ids", [])) for n in miss_set):
                    out.append(e)
            still = _accounted_after(out, line_map, missing_nums)
            if still:
                print(f"[v9_compose] 修补后仍未交代的编号按 skip 处理: {still}（转写层原文兜底）")

    if os.environ.get("NATIVEMEM_V8_VERIFY", "on") == "off":
        return out
    # 覆盖校验：拿笔记全文当"原文"抽专名/数字，看有没有摘要全没覆盖的（谱曲漏选）。
    note_turns = [("", note) for _did, note in notes]
    missing = verify_event_coverage(note_turns, out)
    if not missing:
        return out
    followup = ("以下笔记里的具体信息（专有名词/数字）在你上一轮的事件里漏了："
                + "、".join(missing)
                + "。请把每一条都补成对应的事件（同样逐条记账：给 notes 的 do 指向新事件，"
                  "events 给 id/when/summary/topic，summary 内联引用对应笔记编号），"
                  "只输出 JSON：{\"notes\":[...],\"events\":[...]}。")
    text2 = _distill_call(messages + [{"role": "assistant", "content": text},
                                      {"role": "user", "content": followup}],
                          phase="v9_compose_verify", max_retry=1, model=model)
    if text2:
        add2, _ = _compose_parse(text2, obs_date, line_map, n_notes, note_by_n)
        out.extend(add2)
    return out


def _accounted_after(events, line_map, wanted_nums):
    """wanted_nums 里，dia_id 没落进任何 event.dia_ids 的编号（即仍没交代的），排序返回。"""
    covered = {d for e in events for d in e.get("dia_ids", [])}
    return sorted(n for n in wanted_nums if line_map.get(n) not in covered)
