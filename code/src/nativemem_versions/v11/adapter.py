"""NativeMem V11 adapter."""

import json
import os
import time
from pathlib import Path

from . import memory


def build_memory(conv, memory_dir, max_sessions=None):
    from src.adapters import run_nativemem as shared

    started = time.time()
    os.makedirs(memory_dir, exist_ok=True)
    sessions = []
    index = 1
    while f"session_{index}" in conv:
        turns = []
        refs = []
        session = conv[f"session_{index}"]
        for chunk_turns, chunk_refs in shared.split_into_chunks_structured(
            session, max(1, len(session))
        ):
            turns.extend(chunk_turns)
            refs.extend(chunk_refs)
        sessions.append({
            "observation_date": shared.normalize_date(
                conv.get(f"session_{index}_date_time", "")
            ),
            "turns": turns,
            "refs": refs,
        })
        index += 1
    if max_sessions:
        sessions = sessions[:max_sessions]

    event_count = 0
    usage_logger = lambda response: shared.log_usage(response, phase="v11_agent")
    verification_path = Path(memory_dir) / "verification.jsonl"
    for session in sessions:
        audit = memory.write_sessions(
            memory_dir,
            client=shared.client,
            model=shared.ALIYUN_MODEL,
            sessions=[session],
            usage_logger=usage_logger,
        )
        event_count += sum(
            int(record.get("count", 0))
            for record in audit
            if record.get("tool") == "save_memory"
        )
        verification = memory.verify_session(
            memory_dir,
            client=shared.client,
            model=shared.ALIYUN_MODEL,
            observation_date=session["observation_date"],
            turns=session["turns"],
            refs=session["refs"],
            usage_logger=usage_logger,
        )
        with verification_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(verification, ensure_ascii=False) + "\n")

    memory.manage_memory(
        memory_dir,
        client=shared.client,
        model=shared.ALIYUN_MODEL,
        usage_logger=usage_logger,
    )
    return time.time() - started, event_count
