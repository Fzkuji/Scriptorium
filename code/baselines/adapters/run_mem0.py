"""Mem0 baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_mem0.py" --sample 0 --output results/mem0_s0.json
    python3 "src/adapters/run_mem0.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_mem0.json

Retrieval only: this adapter builds Mem0 memories from LoCoMo sessions and
dumps top-20 retrieved memories per question. Answering and judging are done
by src/evaluation/evaluate.py.

Model support findings (mem0ai 2.0.10, matches third_party/mem0)
----------------------------------------------------------------
- LLM config: mem0 has its own provider factory (mem0.utils.factory.LlmFactory),
  NOT litellm. Providers: openai, anthropic, gemini, deepseek, ollama, vllm,
  litellm (as one provider among many), etc. The "openai" provider is a bare
  OpenAI SDK client and accepts `openai_base_url` (mem0/llms/openai.py:51),
  so ANY OpenAI-compatible endpoint works -> deepseek-v4-flash via the Aliyun
  compatible-mode URL is fully supported.
- Embedder: providers include openai, huggingface, ollama, gemini, fastembed...
  The "huggingface" provider runs a LOCAL SentenceTransformer (no API needed)
  and takes `embedding_dims` (mem0/embeddings/huggingface.py). No OpenAI
  embedding requirement, no dimension restriction (qdrant collection dims are
  set via vector_store.config.embedding_model_dims). We use
  sentence-transformers/all-MiniLM-L6-v2, 384 dims.
- Vector store: local qdrant (embedded, path-based), no server needed.

Timestamp caveat
----------------
The task spec suggested `add(..., timestamp=epoch)` following
third_party/mem0-benchmarks (which uses the Mem0 *Platform* client). In OSS
mem0ai 2.0.10 `Memory.add(timestamp=...)` unconditionally raises ValueError
("platform-only temporal parameter", mem0/memory/main.py:768-769). Instead we
pass the session date as metadata `created_at` (ISO-8601 UTC): _create_memory
preserves a caller-provided created_at (mem0/memory/main.py:1891) and search
results surface it as the top-level `created_at` field, which we use as the
memory date. Additionally the session date is prefixed to the first message of
each session ("[Conversation on <date>] ...") so the fact-extraction LLM has
temporal grounding — without it, OSS extraction anchors relative dates to
today's date and writes wrong absolute dates into memory text (the platform
grounds extraction with the timestamp param instead).

Patches to third_party
----------------------
None. No file in third_party/ was modified; the pip-installed mem0ai 2.0.10
package is byte-identical to third_party/mem0 for memory/main.py and is used
as-is.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _usage_tracker import tracker  # noqa: E402
tracker.install()

from scripts.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402
from mem0 import Memory  # noqa: E402

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


def setup_mem0(sample_idx):
    """Mem0 with deepseek-v4-flash builder LLM + local MiniLM embeddings."""
    workdir = tempfile.mkdtemp(prefix=f"mem0_locomo_s{sample_idx}_")
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": MODEL,
                "api_key": BUILDER_KEY,
                "openai_base_url": BUILDER_BASE,
                "temperature": 0.3,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"locomo_mem0_s{sample_idx}",
                "embedding_model_dims": 384,
                "path": os.path.join(workdir, "qdrant"),
            },
        },
        "history_db_path": os.path.join(workdir, "history.db"),
        "version": "v1.1",
    }
    return Memory.from_config(config)


def session_messages(turns, speaker_a):
    """LoCoMo turns -> mem0 message list (speaker-prefixed, image captions kept)."""
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
        messages.append({"role": role, "content": f"{speaker}: {text}"})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Mem0 LoCoMo adapter (retrieval only)")
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

    # Sessions in numeric order.
    session_ids = sorted(
        (int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))),
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[mem0] sample={args.sample} sessions={len(session_ids)} model={MODEL}")
    m = setup_mem0(args.sample)
    user_id = f"locomo_s{args.sample}"

    # ---- Phase 1: build memory (one add per session, dated via metadata) ----
    tracker.reset("build")
    t0 = time.time()
    failed_sessions = []
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "")
        messages = session_messages(turns if isinstance(turns, list) else [], speaker_a)
        if not messages:
            continue
        if date_str:
            # Temporal grounding for OSS fact extraction (see module docstring).
            messages[0]["content"] = f"[Conversation on {date_str}] {messages[0]['content']}"
        parsed = parse_locomo_date(date_str)
        metadata = {"session": si, "session_date": date_str}
        if parsed:
            # OSS mem0 rejects add(timestamp=...); created_at metadata is the
            # supported way to backdate memories (see module docstring).
            metadata["created_at"] = parsed.isoformat()
        for attempt in range(2):
            try:
                m.add(messages, user_id=user_id, metadata=metadata)
                break
            except Exception as e:
                if attempt == 1:
                    failed_sessions.append(si)
                    print(f"  session {si} FAILED after retry: {e}")
                else:
                    time.sleep(5)
        print(f"  session {si} ingested ({len(messages)} msgs, {time.time()-t0:.0f}s elapsed)")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    try:
        all_mems = m.get_all(filters={"user_id": user_id})
        results_list = all_mems.get("results", all_mems) if isinstance(all_mems, dict) else all_mems
        num_memories = len(results_list)
    except Exception as e:
        print(f"  get_all failed: {e}")
        num_memories = -1
    print(f"[mem0] build done: {build_time:.0f}s, {num_memories} memories")

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
            f"mem0ai 2.0.10, llm={MODEL}, endpoint={BUILDER_BASE}, "
            f"embedder=all-MiniLM-L6-v2 (local, 384d), qdrant local, "
            f"sessions={len(session_ids)}, failed_sessions={failed_sessions}, "
            f"session date passed as metadata created_at (OSS rejects add timestamp param)"
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
            res = m.search(question, top_k=TOP_K, filters={"user_id": user_id})
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            for h in hits:
                memories.append({
                    "text": h.get("memory", ""),
                    "date": h.get("created_at"),
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
    print(f"[mem0] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
