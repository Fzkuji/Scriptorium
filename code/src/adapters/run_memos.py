"""MemOS (MemTensor/MemOS) baseline adapter for the unified LoCoMo protocol.

Run from project root:
    python3 "src/adapters/run_memos.py" --sample 0 --output results/memos_s0.json
    python3 "src/adapters/run_memos.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_memos.json

Retrieval only: builds MemOS textual memories from LoCoMo sessions and dumps
top-20 retrieved memories per question. Answering and judging are done by
src/evaluation/evaluate.py.

Model support findings (MemOS aka MemoryOS 2.0.20, third_party/memos)
---------------------------------------------------------------------
- LLM: `openai` backend (memos/llms/openai.py) is a bare OpenAI SDK client
  with `api_base` (memos/configs/llm.py:27) -> any OpenAI-compatible endpoint
  works, so BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY plug in directly.
- Embedder: `sentence_transformer` backend runs a LOCAL SentenceTransformer
  (memos/embedders/sentence_transformer.py) with `embedding_dims` -> we use
  all-MiniLM-L6-v2, 384d. No embedding API needed.
- Vector store: qdrant backend supports embedded/local mode via `path`
  (memos/vec_dbs/qdrant.py, config path field) -> no server needed.
- Memory backend used here: `general_text` (GeneralTextMemory), the OSS
  in-process "extract facts via LLM -> embed -> qdrant -> vector search" core.
  NOTE: MemTensor's reported LoCoMo 75.8 used their memos-api SERVER
  (TreeTextMemory), which hard-requires a graph DB server (Neo4j / PolarDB /
  Postgres, see memos/graph_dbs/) plus their API stack; that path is not
  reproducible in-process, so this adapter is the faithful open, serverless
  subset (same extraction prompt SIMPLE_STRUCT_MEM_READER_PROMPT, same
  embed/search machinery).

In-process shim (no third_party modification)
---------------------------------------------
MemOS pins `transformers<5` / `fastapi<0.116` which conflict with this env
(transformers 5.12.1), so we do NOT `pip install -e third_party/memos`.
Instead we `sys.path.insert(0, third_party/memos/src)` and import the package
straight from the source tree — byte-identical code, no file modified.
Extra runtime deps installed into the env: concurrent-log-handler, ollama,
prometheus-client (import-time requirements of memos.log / embedder factory /
mem_scheduler). qdrant-client & sentence-transformers were already present.

Timestamp handling
------------------
GeneralTextMemory.extract() joins messages as "role:content" with no chat_time
field, and stamps metadata.updated_at with *today*. For temporal grounding we
prefix every message content with "[<session_date>]" (mirrors the
"role: [chat_time]: content" layout of their SimpleStructMemReader), and after
extraction we overwrite each item's metadata.updated_at with the session date
(ISO-8601 UTC) so the memory `date` field is meaningful.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

from _usage_tracker import tracker
tracker.install()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "memos", "src"))

from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

from memos.configs.memory import MemoryConfigFactory  # noqa: E402
from memos.memories.factory import MemoryFactory  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MODEL = BUILDER_MODEL
TOP_K = 20


def parse_locomo_date(date_str):
    """Parse LoCoMo date: '1:56 pm on 8 May, 2023' -> aware datetime (UTC)."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def setup_memos(sample_idx):
    """GeneralTextMemory: builder LLM extraction + local MiniLM + local qdrant."""
    workdir = tempfile.mkdtemp(prefix=f"memos_locomo_s{sample_idx}_")
    config = MemoryConfigFactory(
        backend="general_text",
        config={
            "extractor_llm": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": MODEL,
                    "api_key": BUILDER_KEY,
                    "api_base": BUILDER_BASE,
                    "temperature": 0.0,
                    "remove_think_prefix": True,
                    "max_tokens": 8192,
                },
            },
            "vector_db": {
                "backend": "qdrant",
                "config": {
                    "collection_name": f"locomo_memos_s{sample_idx}",
                    "distance_metric": "cosine",
                    "vector_dimension": 384,
                    "path": os.path.join(workdir, "qdrant"),
                },
            },
            "embedder": {
                "backend": "sentence_transformer",
                "config": {
                    "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dims": 384,
                },
            },
        },
    )
    return MemoryFactory.from_config(config)


def session_messages(turns, speaker_a, date_str):
    """LoCoMo turns -> role/content messages, date-prefixed, captions kept."""
    messages = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker", "")
        text = turn.get("text", "") or ""
        blip = turn.get("blip_caption", "") or ""
        query = turn.get("query", "") or ""
        if query and blip:
            photo = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo = f"[Sharing image - query for: {query}]"
        elif blip:
            photo = f"[Sharing image that shows: {blip}]"
        else:
            photo = ""
        if photo:
            text = f"{text} {photo}".strip()
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        prefix = f"[{date_str}] " if date_str else ""
        messages.append({"role": role, "content": f"{prefix}{speaker}: {text}"})
    return messages


def main():
    parser = argparse.ArgumentParser(description="MemOS LoCoMo adapter (retrieval only)")
    parser.add_argument("--sample", type=int, default=0, help="LoCoMo conversation index")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Dev: only ingest first N sessions")
    parser.add_argument("--questions-limit", type=int, default=None,
                        help="Dev: only process first N questions")
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        data = json.load(f)
    sample = data[args.sample]
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Person A")

    session_ids = sorted(
        (int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))),
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[memos] sample={args.sample} sessions={len(session_ids)} model={MODEL}")
    m = setup_memos(args.sample)

    # ---- Phase 1: build memory (extract per session, then add) ----
    tracker.reset("build")
    t0 = time.time()
    failed_sessions = []
    num_memories = 0
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "")
        messages = session_messages(turns if isinstance(turns, list) else [],
                                    speaker_a, date_str)
        if not messages:
            continue
        parsed = parse_locomo_date(date_str)
        iso_date = parsed.isoformat() if parsed else (date_str or None)
        for attempt in range(2):
            try:
                items = m.extract(messages)
                if iso_date:
                    for item in items:
                        # Backdate: extract() stamps today's date (see docstring).
                        item.metadata.updated_at = iso_date
                m.add(items)
                num_memories += len(items)
                print(f"  session {si}: {len(items)} memories "
                      f"({len(messages)} msgs, {time.time()-t0:.0f}s elapsed)")
                break
            except Exception as e:
                if attempt == 1:
                    failed_sessions.append(si)
                    print(f"  session {si} FAILED after retry: {e}")
                else:
                    time.sleep(5)
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")
    print(f"[memos] build done: {build_time:.0f}s, {num_memories} memories")

    # ---- Phase 2: retrieval per question ----
    qas = sample["qa"]
    if args.questions_limit:
        qas = qas[: args.questions_limit]

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 1),
        "num_memories": num_memories,
        "build_calls": build_snap["calls"],
        "build_tokens_in": build_snap["tokens_in"],
        "build_tokens_out": build_snap["tokens_out"],
        "build_llm_time_s": build_snap["llm_time_s"],
        "notes": (
            f"MemOS (MemoryOS 2.0.20) general_text backend, llm={MODEL} via "
            f"openai-compatible base_url, embedder=all-MiniLM-L6-v2 (local, 384d), "
            f"qdrant embedded, sessions={len(session_ids)}, "
            f"failed_sessions={failed_sessions}; session date prefixed to every "
            f"message and written back to metadata.updated_at (extract stamps today); "
            f"tree_text/memos-api path (LoCoMo 75.8) needs graph-DB server, not used"
        ),
    }]

    for qi, qa in enumerate(qas):
        question = qa["question"]
        gold = qa.get("answer")
        if gold is None:
            gold = qa.get("adversarial_answer", "")
        tracker.reset("q")
        t_q = time.time()
        memories = []
        try:
            hits = m.search(question, top_k=TOP_K) or []
            for h in hits:
                memories.append({
                    "text": h.memory,
                    "date": getattr(h.metadata, "updated_at", None),
                })
        except Exception as e:
            print(f"  q{qi} search error: {e}")
        latency = time.time() - t_q
        q_snap = tracker.snapshot("q")
        records.append({
            "question_id": f"s{args.sample}_q{qi}",
            "question": question,
            "gold": str(gold),
            "category": qa.get("category", -1),
            "memories": memories,
            "retrieval": {"latency_s": round(latency, 3), "k": TOP_K,
                          "calls": q_snap["calls"],
                          "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(qas)} questions done")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[memos] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
