"""NativeMem V8 adapter."""

import os


def build_memory(conv, memory_dir, max_sessions=None):
    import time as _t
    from src.adapters import run_nativemem as shared

    v8_memory = shared.v8_memory
    write_events = shared.write_events
    split_into_chunks_structured = shared.split_into_chunks_structured
    normalize_date = shared.normalize_date
    _format_topics_snapshot = shared._format_topics_snapshot
    _topics_snapshot = shared._topics_snapshot
    _v8_tidy = shared._v8_tidy
    _v8_concurrency = shared._v8_concurrency
    ThreadPoolExecutor = shared.ThreadPoolExecutor
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
    n_events = 0
    # 小块提炼：默认 6 句/块（而非一整个 session）。块小模型注意力集中、
    # 漏事实少（build 覆盖率是失分主因）；后端是大模型，多几次调用可接受。
    # 模型靠累积的 known_topics 看"以前的记忆"，小块间上下文不断裂。
    v8_chunk = int(os.environ.get("NATIVEMEM_CHUNK_TURNS", "6"))
    # 切块策略：fixed=固定 6 句/块（默认，一字不变）；topic=每 session 先用一次
    # 便宜 LLM 调用按话题切边界再分块。topic 坏了会自动回退固定切块，不崩。
    v8_segment = os.environ.get("NATIVEMEM_V8_SEGMENT", "fixed")
    # 整理环节（spec §6.1）：每 session 末去重（默认 on），topic 文件数超阈值才合并。
    tidy_on = os.environ.get("NATIVEMEM_V8_TIDY", "on") != "off"
    max_topics = int(os.environ.get("NATIVEMEM_V8_MAX_TOPICS", "30"))
    # 分节整理（默认）替代全文重写：模型只出分节方案、代码搬行、行逐字不动。
    # 旧的 NATIVEMEM_V8_ARTICLE（全文重写，默认 off）保留供消融；两者互斥，
    # SECTIONS=on 优先走分节路径。
    sections_on = os.environ.get("NATIVEMEM_V8_SECTIONS", "on") != "off"
    article_on = (not sections_on
                  and os.environ.get("NATIVEMEM_V8_ARTICLE", "off") != "off")
    # v9.0 两级建库（spec）：=two_tier 时每 session 先逐句转写再谱曲成事件，
    # 得到 events 后走既有 write_events/touched/tidy 流程；默认 off 走下面既有
    # 切块 distill，一字节不变。两级都见全 session，故新路径不用滚动 recent。
    v9_two_tier = os.environ.get("NATIVEMEM_V9_PIPELINE") == "two_tier"
    # 转写层无跨 session 状态（只看自己的句子+日期），全部 session 预先并行
    # 转写；谱曲层要看演进中的 known_topics，保持按时间串行。
    v9_notes_by_idx = {}
    if v9_two_tier and sessions:
        def _pre_transcribe(item):
            idx, (sess, dt) = item
            flat = split_into_chunks_structured(sess, 10 ** 9)
            if not flat:
                return idx, []
            ft, fd = flat[0]
            return idx, v8_memory.transcribe_session(ft, fd,
                                                     normalize_date(dt))
        _w = min(_v8_concurrency(), len(sessions))
        with ThreadPoolExecutor(max_workers=_w) as ex:
            for idx, nts in ex.map(_pre_transcribe,
                                   enumerate(zip(sessions, dates))):
                v9_notes_by_idx[idx] = nts
    for s_idx, (session, date) in enumerate(zip(sessions, dates)):
        obs = normalize_date(date)
        # 结构感知放置（spec §7.4）：把带条目数的 topic 清单（大 topic 排前）
        # 喂给 distill，模型看得到已有规模分布、优先复用而非造近义新名。
        known_view = _format_topics_snapshot(_topics_snapshot(memory_dir))
        touched = set()               # 本 session 写过的 topic（文章化整理对象）
        if v9_two_tier:
            notes = v9_notes_by_idx.get(s_idx, [])
            if notes:
                events = v8_memory.compose_events(notes, obs,
                                                  known_topics=known_view)
            else:
                events = []
            if events:
                write_events(memory_dir, events)
                touched.update(
                    v8_memory._sanitize_topic(e.get("topic", "misc"))
                    for e in events)
                n_events += len(events)
            if tidy_on:                           # 节奏2：每 session 末
                v8_memory.dedup_topic_files(memory_dir)
                n_topics = len(v8_memory._topics_dir_files(memory_dir))
                if n_topics > max_topics:
                    v8_memory.consolidate_topic_files(memory_dir)
            if touched:                           # 节奏3：整理（按增长触发）
                _v8_tidy(memory_dir, touched, sections_on, article_on)
            continue
        recent = []                   # 滚动上下文：本 session 已提炼句（内存变量）
        # structured per-turn chunks: numbering is 1:1 with dia_ids by
        # construction (a turn's internal blank line can't shift block nums).
        if v8_segment == "topic":
            # 整个 session 铺平成一个大块（size 足够大→单块），再按话题切分。
            flat = split_into_chunks_structured(session, 10 ** 9)
            if flat:
                ft, fd = flat[0]
                chunks = v8_memory.segment_session_by_topic(ft, fd)
            else:
                chunks = []
        else:
            chunks = split_into_chunks_structured(session, v8_chunk)
        for turns, dia_ids in chunks:
            if not turns:
                continue
            events = v8_memory.distill_events(turns, obs, dia_ids,
                                    known_topics=known_view,
                                    recent=recent[-20:])
            if not events:
                continue
            recent.extend(e["summary"] for e in events)
            write_events(memory_dir, events)
            touched.update(v8_memory._sanitize_topic(e.get("topic", "misc"))
                           for e in events)
            n_events += len(events)
        if tidy_on:                               # 节奏2：每 session 末
            v8_memory.dedup_topic_files(memory_dir)
            n_topics = len(v8_memory._topics_dir_files(memory_dir))
            if n_topics > max_topics:
                v8_memory.consolidate_topic_files(memory_dir)
        if touched:                           # 节奏3：整理（按增长触发）
            _v8_tidy(memory_dir, touched, sections_on, article_on)
    # 收尾：对所有仍有未归节行的文件做一次收尾整理，保证交付态整洁（同样过硬校验）。
    all_topics = set(v8_memory._topics_dir_files(memory_dir))
    _v8_tidy(memory_dir, all_topics, sections_on, article_on, final=True)
    return _t.time() - t0, n_events
