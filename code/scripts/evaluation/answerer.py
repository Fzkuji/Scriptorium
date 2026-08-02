"""Unified answerer for the evaluation protocol.

Every system under comparison retrieves its own memories, then THIS module
generates the final short answer with the SAME model and the SAME prompt
(Mem0 original ANSWER_PROMPT, "less than 5-6 words"). This isolates memory
quality from answer-generation style — see protocol §6.3.

Memory presentation (Mem0 convention, avoids anchoring):
- chronological order (oldest first) when dates are available
- date prefix "(<date>) <text>" per memory
- no retrieval scores shown
"""

from .prompts import ANSWER_PROMPT
from .llm_clients import answerer_chat


def format_memories(memories):
    """memories: list of dicts {text, date?} or plain strings → prompt block."""
    if isinstance(memories, str):
        return memories
    entries = []
    for m in memories:
        if isinstance(m, str):
            entries.append({"text": m, "date": ""})
        else:
            entries.append({"text": m.get("text", m.get("memory", "")),
                            "date": str(m.get("date", m.get("created_at", "")) or "")})
    # chronological if dates present (ISO or comparable strings), stable otherwise
    if any(e["date"] for e in entries):
        entries.sort(key=lambda e: e["date"])
    lines = []
    for e in entries:
        lines.append(f"({e['date']}) {e['text']}" if e["date"] else e["text"])
    return "\n".join(lines) if lines else "(No relevant memories found)"


import re


def extract_answer(text):
    """Deterministic extraction: <answer> tags first, then fallbacks."""
    text = text.strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "Answer:" in text:
        return text.rsplit("Answer:", 1)[-1].strip()
    return text


def generate_answer(question, memories):
    """Return (short_answer, usage_dict)."""
    prompt = ANSWER_PROMPT.format(
        memories=format_memories(memories), question=question)
    text, usage = answerer_chat([{"role": "user", "content": prompt}])
    return extract_answer(text), usage
