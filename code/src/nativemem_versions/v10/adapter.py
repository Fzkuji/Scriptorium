"""NativeMem V10 adapter."""

import os


def build_memory(conv, memory_dir, max_sessions=None):
    """Build the selected v8.8+calendar design under explicit v10 policies."""
    import time as _t
    from src.adapters import run_nativemem as shared

    v8_memory = shared.v8_memory
    v10_memory = shared.v10_memory
    write_events = shared.write_events
    split_into_chunks_structured = shared.split_into_chunks_structured
    normalize_date = shared.normalize_date
    _format_topics_snapshot = shared._format_topics_snapshot
    _topics_snapshot = shared._topics_snapshot
    _v8_tidy = shared._v8_tidy

    if os.environ.get("NATIVEMEM_V9_PIPELINE") == "two_tier":
        raise ValueError("v10 cannot be combined with NATIVEMEM_V9_PIPELINE")

    config = v10_memory.V10BuildConfig.from_env()
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

    max_topics = int(os.environ.get("NATIVEMEM_V8_MAX_TOPICS", "30"))
    sections_on = os.environ.get("NATIVEMEM_V8_SECTIONS", "on") != "off"
    article_on = (
        not sections_on
        and os.environ.get("NATIVEMEM_V8_ARTICLE", "off") != "off"
    )
    n_events = 0
    pending_touched = set()

    def _session_maintenance(touched, passes):
        for _ in range(passes):
            v8_memory.dedup_topic_files(memory_dir)
            n_topics = len(v8_memory._topics_dir_files(memory_dir))
            if n_topics > max_topics:
                v8_memory.consolidate_topic_files(memory_dir)
            if touched:
                _v8_tidy(memory_dir, touched, sections_on, article_on)

    if config.session_group_size > 1:
        for offset in range(0, len(sessions), config.session_group_size):
            batch = []
            for session, date in zip(
                sessions[offset:offset + config.session_group_size],
                dates[offset:offset + config.session_group_size],
            ):
                prepared_turns = []
                prepared_ids = []
                for chunk_turns, chunk_ids in split_into_chunks_structured(
                    session, max(1, len(session))
                ):
                    prepared_turns.extend(chunk_turns)
                    prepared_ids.extend(chunk_ids)
                batch.append({
                    "observation_date": normalize_date(date),
                    "turns": prepared_turns,
                    "dia_ids": prepared_ids,
                })
            known_view = _format_topics_snapshot(_topics_snapshot(memory_dir))
            events = v10_memory.distill_session_group(
                batch, known_topics=known_view
            )
            touched = set()
            if events:
                write_events(memory_dir, events)
                touched.update(
                    v8_memory._sanitize_topic(event.get("topic", "misc"))
                    for event in events
                )
                n_events += len(events)
            pending_touched.update(touched)
            completed_sessions = min(
                offset + config.session_group_size, len(sessions)
            )
            if config.session_maintenance_due(completed_sessions):
                _session_maintenance(
                    set(pending_touched), config.session_tidy_passes
                )
                pending_touched.clear()

        for _ in range(config.final_tidy_passes):
            all_topics = set(v8_memory._topics_dir_files(memory_dir))
            _v8_tidy(memory_dir, all_topics, sections_on, article_on, final=True)
        return _t.time() - t0, n_events

    for s_idx, (session, date) in enumerate(zip(sessions, dates)):
        obs = normalize_date(date)
        known_view = _format_topics_snapshot(_topics_snapshot(memory_dir))
        event_history = []
        raw_history = []
        rolling_summary = ""
        touched = set()
        chunks = split_into_chunks_structured(
            session, config.chunk_size(len(session))
        )

        for turns, dia_ids in chunks:
            if not turns:
                continue

            if config.context_mode == "events":
                recent = (
                    event_history[-config.context_items :]
                    if config.context_items > 0
                    else None
                )
                # This is the selected v8.8 call path.  With v10 defaults the
                # writer prompt and verification behavior remain unchanged.
                events = v8_memory.distill_events(
                    turns,
                    obs,
                    dia_ids,
                    known_topics=known_view,
                    recent=recent,
                )
            elif config.context_mode == "none":
                events = v8_memory.distill_events(
                    turns, obs, dia_ids, known_topics=known_view, recent=None
                )
            else:
                prior_context = v10_memory.render_prior_context(
                    config,
                    event_history=event_history,
                    raw_history=raw_history,
                    rolling_summary=rolling_summary,
                )
                events, rolling_summary = v10_memory.distill_with_context(
                    turns,
                    obs,
                    dia_ids,
                    known_topics=known_view,
                    prior_context=prior_context,
                    previous_summary=rolling_summary,
                    summary_max_words=(
                        config.summary_max_words
                        if config.context_mode == "summary"
                        else None
                    ),
                )

            # Preceding raw context is updated only after the current writer
            # call, so the current chunk can never appear in its own context.
            raw_history.extend(turns)
            if not events:
                continue
            event_history.extend(event["summary"] for event in events)
            write_events(memory_dir, events)
            touched.update(
                v8_memory._sanitize_topic(event.get("topic", "misc"))
                for event in events
            )
            n_events += len(events)

        pending_touched.update(touched)
        completed_sessions = s_idx + 1
        if config.session_maintenance_due(completed_sessions):
            _session_maintenance(
                set(pending_touched), config.session_tidy_passes
            )
            pending_touched.clear()

    for _ in range(config.final_tidy_passes):
        all_topics = set(v8_memory._topics_dir_files(memory_dir))
        _v8_tidy(memory_dir, all_topics, sections_on, article_on, final=True)

    return _t.time() - t0, n_events
