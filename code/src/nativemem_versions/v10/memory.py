"""NativeMem v10 build policy and context-aware distillation.

v10 keeps the selected v8.8+calendar memory representation and retrieval
path.  It makes the three construction decisions explicit and independently
configurable:

1. how many dialogue turns are processed by one writer call;
2. what preceding session context is visible to the writer;
3. when, and how many times, maintenance runs.

The default values reproduce the selected v8.8+calendar construction policy.
Alternative policies are experiment settings; they are not silently enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Iterable, Sequence

import src.v8_memory as v8_memory


CONTEXT_MODES = frozenset({"events", "raw", "summary", "none"})


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_write_turns() -> int | str:
    raw = os.environ.get("NATIVEMEM_V10_WRITE_TURNS", "6").strip().lower()
    if raw == "session":
        return raw
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "NATIVEMEM_V10_WRITE_TURNS must be a positive integer or "
            f"'session', got {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            f"NATIVEMEM_V10_WRITE_TURNS must be >= 1, got {value}"
        )
    return value


@dataclass(frozen=True)
class V10BuildConfig:
    """The three v10 construction policies.

    ``context_items`` means event count in ``events`` mode and raw turn count
    in ``raw`` mode.  ``summary`` mode carries one rolling summary whose word
    budget is ``summary_max_words``.  The summary is updated in the same writer
    response, so the mode does not add a separate summarization request.

    ``tidy_every_sessions=0`` disables session-boundary maintenance.
    ``session_tidy_passes`` is the pass count at each enabled boundary, while
    ``final_tidy_passes`` controls the all-topics maintenance after the final
    session.
    """

    write_turns: int | str = 6
    session_group_size: int = 1
    context_mode: str = "events"
    context_items: int = 20
    summary_max_words: int = 180
    tidy_every_sessions: int = 1
    session_tidy_passes: int = 1
    final_tidy_passes: int = 1

    def __post_init__(self) -> None:
        if self.write_turns != "session" and (
            not isinstance(self.write_turns, int) or self.write_turns < 1
        ):
            raise ValueError("write_turns must be >= 1 or 'session'")
        if self.session_group_size < 1:
            raise ValueError("session_group_size must be >= 1")
        if self.session_group_size > 1 and self.write_turns != "session":
            raise ValueError(
                "session_group_size > 1 requires write_turns='session'"
            )
        if self.session_group_size > 1 and self.context_mode != "none":
            raise ValueError(
                "session_group_size > 1 requires context_mode='none'"
            )
        if self.context_mode not in CONTEXT_MODES:
            allowed = ", ".join(sorted(CONTEXT_MODES))
            raise ValueError(
                f"context_mode must be one of {allowed}, got {self.context_mode!r}"
            )
        if self.context_items < 0:
            raise ValueError("context_items must be >= 0")
        if self.summary_max_words < 1:
            raise ValueError("summary_max_words must be >= 1")
        if self.tidy_every_sessions < 0:
            raise ValueError("tidy_every_sessions must be >= 0")
        if self.session_tidy_passes < 0:
            raise ValueError("session_tidy_passes must be >= 0")
        if self.final_tidy_passes < 0:
            raise ValueError("final_tidy_passes must be >= 0")

    @classmethod
    def from_env(cls) -> "V10BuildConfig":
        return cls(
            write_turns=_env_write_turns(),
            session_group_size=_env_int(
                "NATIVEMEM_V10_SESSION_GROUP_SIZE", 1, 1
            ),
            context_mode=os.environ.get(
                "NATIVEMEM_V10_CONTEXT_MODE", "events"
            ).strip().lower(),
            context_items=_env_int("NATIVEMEM_V10_CONTEXT_ITEMS", 20),
            summary_max_words=_env_int(
                "NATIVEMEM_V10_SUMMARY_MAX_WORDS", 180, 1
            ),
            tidy_every_sessions=_env_int(
                "NATIVEMEM_V10_TIDY_EVERY_SESSIONS", 1
            ),
            session_tidy_passes=_env_int(
                "NATIVEMEM_V10_SESSION_TIDY_PASSES", 1
            ),
            final_tidy_passes=_env_int(
                "NATIVEMEM_V10_FINAL_TIDY_PASSES", 1
            ),
        )

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def chunk_size(self, session_turns: int) -> int:
        if self.write_turns == "session":
            return max(1, session_turns)
        return self.write_turns

    def session_maintenance_due(self, completed_sessions: int) -> bool:
        return (
            self.tidy_every_sessions > 0
            and self.session_tidy_passes > 0
            and completed_sessions % self.tidy_every_sessions == 0
        )


def distill_session_group(
    sessions,
    *,
    known_topics=None,
    max_retry: int = 6,
):
    """Distill consecutive dated sessions in one writer request.

    ``sessions`` contains ``observation_date``, ``turns``, and ``dia_ids``.
    The ordinary one-session path does not call this function.
    """

    turns = []
    dia_ids = []
    dia_dates = {}
    ranges = []
    start = 1
    for number, session in enumerate(sessions, start=1):
        current_turns = list(session["turns"])
        current_ids = list(session["dia_ids"])
        if len(current_turns) != len(current_ids):
            raise ValueError("session turns and dia_ids must have equal length")
        date = str(session["observation_date"])
        turns.extend(current_turns)
        dia_ids.extend(current_ids)
        dia_dates.update({dia_id: date for dia_id in current_ids})
        end = start + len(current_turns) - 1
        ranges.append(f"Session {number}: lines {start}-{end}, date {date}")
        start = end + 1

    topics = ", ".join(known_topics) if known_topics else "（暂无）"
    unique_dates = list(dict.fromkeys(dia_dates.values()))
    calendars = []
    for date in unique_dates:
        calendar = (
            v8_memory.calendar_strip(date)
            or v8_memory._V9_CALENDAR_FALLBACK
        )
        calendars.append(f"Observation date {date}:\n{calendar}")
    prompt = v8_memory._V8_DISTILL_PROMPT.format(
        obs_date="varies by session; use the line-to-session mapping below",
        calendar="\n\n".join(calendars),
        known_topics=topics,
    )
    prompt += (
        "\n\n## Session boundaries\n"
        + "\n".join(ranges)
        + "\nEach event must use the observation date of its referenced lines "
          "when no more specific event date can be inferred. Do not merge "
          "unrelated facts merely because the sessions are supplied together."
    )
    numbered_text, line_map = v8_memory._number_chunk(turns, dia_ids)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": numbered_text},
    ]
    text = v8_memory._distill_call(
        messages, phase="v10_session_group", max_retry=max_retry
    )
    if text is None:
        return []
    fallback_date = unique_dates[-1] if unique_dates else "1970-01-01"
    events = v8_memory._parse_distill_response(
        text, fallback_date, line_map, dia_dates=dia_dates
    )
    first_pass = [dict(event) for event in events]
    missing = (
        []
        if os.environ.get("NATIVEMEM_V8_VERIFY", "on") == "off"
        else v8_memory.verify_event_coverage(turns, events)
    )
    added = []
    if missing:
        followup = (
            "以下原文里的具体信息（专有名词/数字）在你上一轮的事件里漏了："
            + "、".join(missing)
            + "。请按各行所属 session 的日期补成事件，只输出补充的事件 JSON："
              "{\"events\":[...]}。"
        )
        text2 = v8_memory._distill_call(
            messages
            + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": followup},
            ],
            phase="v10_session_group_verify",
            max_retry=1,
        )
        if text2:
            added = v8_memory._parse_distill_response(
                text2, fallback_date, line_map, dia_dates=dia_dates
            )
            events.extend(added)
    v8_memory._write_distill_trace({
        "observation_dates": unique_dates,
        "turn_count": len(turns),
        "input_dia_ids": list(dia_ids),
        "first_pass_events": first_pass,
        "missing_coverage_points": list(missing),
        "verify_events": [dict(event) for event in added],
        "post_verify_events": [dict(event) for event in events],
        "verify_enabled": os.environ.get("NATIVEMEM_V8_VERIFY", "on") != "off",
        "session_group_size": len(sessions),
    })
    return events


def render_prior_context(
    config: V10BuildConfig,
    *,
    event_history: Sequence[str],
    raw_history: Sequence[tuple[str, str]],
    rolling_summary: str,
) -> str:
    """Render only the preceding information selected by ``context_policy``."""

    if config.context_mode == "none":
        return ""

    if config.context_mode == "events":
        if config.context_items == 0:
            return ""
        items = list(event_history[-config.context_items :])
        if not items:
            return ""
        return (
            "\n\n## 本 session 前面已提炼的记忆（滚动上下文）\n"
            + "\n".join(f"- {item}" for item in items)
            + "\n上面已记过的事**别重复**提炼；同一件事的新进展沿用同一 "
              "topic 路径写增量；原文里的指代（he/she/it/there…）按这些"
              "上下文**消解**成具体人名/地名再写进摘要。"
        )

    if config.context_mode == "raw":
        if config.context_items == 0:
            return ""
        turns = list(raw_history[-config.context_items :])
        if not turns:
            return ""
        rendered = "\n".join(f"- {speaker}: {text}" for speaker, text in turns)
        return (
            "\n\n## 当前片段之前的原始对话（只读上下文）\n"
            + rendered
            + "\n这些旧对话只用于消解当前片段中的人名、代词、时间和话题延续；"
              "不要再次提炼旧对话中的事实。"
        )

    if not rolling_summary.strip():
        return ""
    return (
        "\n\n## 当前 session 的滚动摘要（只读上下文）\n"
        + rolling_summary.strip()
        + "\n摘要只用于消解当前片段中的指代和话题延续；不要再次提炼摘要里的旧事实。"
    )


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(
        r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE
    ).strip()
    try:
        value = json.loads(cleaned)
    except Exception:  # noqa: BLE001
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return {}
    return value if isinstance(value, dict) else {}


def _bounded_words(text: str, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[-max_words:])


def _fallback_summary(
    previous: str, event_summaries: Iterable[str], max_words: int
) -> str:
    combined = " ".join(
        part for part in [previous.strip(), *[str(x).strip() for x in event_summaries]]
        if part
    )
    return _bounded_words(combined, max_words)


def distill_with_context(
    turns,
    obs_date,
    dia_ids,
    *,
    known_topics=None,
    prior_context: str = "",
    previous_summary: str = "",
    summary_max_words: int | None = None,
    max_retry: int = 6,
):
    """Run the v8.8 writer with a v10 prior-context policy.

    When ``summary_max_words`` is set, the same response also updates a rolling
    context summary.  Durable memories remain the validated v8 event objects;
    the rolling summary is only input to later chunks.
    """

    topics = ", ".join(known_topics) if known_topics else "（暂无）"
    calendar = (
        v8_memory.calendar_strip(obs_date) or v8_memory._V9_CALENDAR_FALLBACK
    )
    prompt = v8_memory._V8_DISTILL_PROMPT.format(
        obs_date=obs_date, calendar=calendar, known_topics=topics
    )
    prompt += prior_context
    if summary_max_words is not None:
        prompt += (
            "\n\n## v10 滚动摘要输出\n"
            f"在同一个 JSON 顶层额外输出 context_summary，最多 {summary_max_words} "
            "个英文单词。它要合并已有滚动摘要与当前片段，只保留后续指代消解所需的"
            "人物、关系、地点、时间和进行中的话题。事件仍放在 events 数组中。"
            "输出格式：{\"events\":[...],\"context_summary\":\"...\"}。"
        )

    numbered_text, line_map = v8_memory._number_chunk(turns, dia_ids)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": numbered_text},
    ]
    text = v8_memory._distill_call(
        messages, phase="v10_distill", max_retry=max_retry
    )
    if text is None:
        return [], previous_summary

    events = v8_memory._parse_distill_response(text, obs_date, line_map)
    obj = _json_object(text)
    next_summary = str(obj.get("context_summary", "")).strip()
    if summary_max_words is not None:
        if next_summary:
            next_summary = _bounded_words(next_summary, summary_max_words)
        else:
            next_summary = _fallback_summary(
                previous_summary,
                (event.get("summary", "") for event in events),
                summary_max_words,
            )

    if os.environ.get("NATIVEMEM_V8_VERIFY", "on") == "off":
        return events, next_summary
    missing = v8_memory.verify_event_coverage(turns, events)
    if not missing:
        return events, next_summary

    followup = (
        "以下原文里的具体信息（专有名词/数字）在你上一轮的事件里漏了："
        + "、".join(missing)
        + "。请把每一条都补成对应的事件（同样给 when/summary/refs/topic），"
          "只输出补充的事件 JSON：{\"events\":[...]}。"
    )
    text2 = v8_memory._distill_call(
        messages
        + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": followup},
        ],
        phase="v10_distill_verify",
        max_retry=1,
    )
    if text2:
        added = v8_memory._parse_distill_response(text2, obs_date, line_map)
        events.extend(added)
        if summary_max_words is not None and added:
            next_summary = _fallback_summary(
                next_summary,
                (event.get("summary", "") for event in added),
                summary_max_words,
            )
    return events, next_summary
