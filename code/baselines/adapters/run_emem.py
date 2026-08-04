"""E-mem baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_emem.py" --sample 0 --output results/emem_s0.json
    python3 "src/adapters/run_emem.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_emem.json

E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory
(ICML 2026, arXiv 2601.21714). Official repo cloned to third_party/emem
(github.com/dog-last/E-mem). Retrieval only: builds E-mem text-mode memory
blocks from LoCoMo sessions and dumps the per-block evidence returned for each
question. Answering and judging are done by src/evaluation/evaluate.py.

Model support findings (third_party/emem @ main)
------------------------------------------------
- Two storage modes: "kv_cache" (local HF model + GPU KV tensors) and "text"
  (pure OpenAI-compatible API). We use "text" mode - the repo's own mode for
  API-based models; no GPU/local LLM needed.
- All LLM roles take {"api_key","base_url","model"} dicts fed to the bare
  OpenAI SDK (src/agent/base.py), so any OpenAI-compatible endpoint works.
  We point all four roles (memory agent = block summaries + block-local
  evidence extraction, manager, aggregator, router fallback) at
  BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY. Verified: Aliyun deepseek-v4-flash
  accepts the repo's tools=[] and extra_body={"repetition_penalty":1.1} calls.
- Embedder: HybridRouter (multi-pathway routing = summary-embedding +
  chunk-embedding + BM25) supports provider "huggingface" running a LOCAL
  SentenceTransformer; the repo's own default model is
  sentence-transformers/all-MiniLM-L6-v2 (384d) - exactly our protocol
  embedder. BM25 uses jieba tokenizer (installed).
- Tokenizer for block sizing: Qwen/Qwen3-4B (repo config.text.yaml default),
  loaded from local HF cache; used only for token counting.

Retrieval semantics
-------------------
E-mem has no top-k item search. Per question the HybridRouter activates up to
max_blocks=5 inactive memory blocks (repo default, config.text.yaml); each
block's assistant agent then does LLM "local reasoning" extracting
question-relevant evidence from its uncompressed context; the active
(still-filling) block is queried the same way. Each block's evidence string
becomes one memory item (dates are embedded inline in the evidence, per the
repo's own query prompt; "date" field left null). The master-agent
aggregation step (BaseChatManager._aggregate_memory_results) is skipped: it
synthesizes an answer-like summary and the adapter contract forbids
answering. retrieval.k reports the number of memory items returned (<= 6),
not 20 - block-level evidence is E-mem's native retrieval granularity.

Config mirrors evaluation/locomo/config.text.yaml: block_size_ratio=0.125 of a
32768 context window (4096-token blocks), overlap chunk mode ratio 0.1,
max_blocks=5, hybrid router with bm25_boost_threshold=0.8. Memory is built one
turn at a time as "[<session datetime>] <speaker>: <text>" exactly like the
repo's own eval_locomo.py; we additionally append LoCoMo blip image captions
for content parity with the other 14 adapters (the repo's loader drops them).

Patches to third_party
----------------------
None. In-process shims only:
- sys.modules "src" purge: both this project and emem expose a "src" package;
  we import llm_clients first, purge src* from sys.modules, then put
  third_party/emem at sys.path[0] and import emem's src.* modules.
- TEXT_DATA_DIR env var (supported by emem's text_block.py) points block
  storage at a per-run temp dir instead of ./text_data in cwd.
"""

from _usage_tracker import tracker
tracker.install()

import argparse
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMEM_DIR = os.path.join(PROJECT_ROOT, "third_party", "emem")

sys.path.insert(0, PROJECT_ROOT)
from scripts.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

# --- shim: swap project "src" namespace package for emem's "src" package ---
for _mod in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_mod]
sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, EMEM_DIR)

# Block storage goes to a temp dir (emem reads this env var at call time).
os.environ["TEXT_DATA_DIR"] = tempfile.mkdtemp(prefix="emem_locomo_")

from src.conversation_manager.factory import create_chat_manager  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MODEL = BUILDER_MODEL
TOKENIZER_ID = "Qwen/Qwen3-4B"  # repo config.text.yaml default; token counting only
MAX_MEMORIES = 20

# Sentinel strings emem returns when a block has nothing.
_EMPTY_MARKERS = ("No knowledge available.", "No active memory.", "No relevant memory found.")


def build_manager():
    """TextStorageChatManager with builder LLM + local MiniLM hybrid router."""
    cfg = {"api_key": BUILDER_KEY, "base_url": BUILDER_BASE, "model": MODEL}
    return create_chat_manager(
        storage_mode="text",
        model_id=TOKENIZER_ID,
        chat_openai_config=dict(cfg),
        aggregator_openai_config=dict(cfg),
        memory_agent_openai_config=dict(cfg),
        router_openai_config=dict(cfg),
        model_context_window=32768,
        overlap_mode="chunk",
        overlap_ratio=0.1,
        block_size_ratio=0.125,
        max_memory_segments=5,
        max_blocks=5,
        enable_router=True,
        router_type="hybrid",
        hybrid_router_config={
            "embedding_provider": "huggingface",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "summary_weight": 0.3,
            "text_weight": 0.4,
            "bm25_weight": 0.3,
            "use_llm_fallback": False,
            "bm25_use_jieba": True,
            "bm25_boost_threshold": 0.8,
        },
    )


def turn_text(turn):
    """LoCoMo turn -> plain text incl. image captions (parity with other adapters)."""
    if not isinstance(turn, dict):
        return ""
    text = (turn.get("text") or "").strip()
    blip = (turn.get("blip_caption") or "").strip()
    query = (turn.get("query") or "").strip()
    if query and blip:
        photo = f"[Sharing image - query: {query}. The image shows: {blip}]"
    elif query:
        photo = f"[Sharing image - query for: {query}]"
    elif blip:
        photo = f"[Sharing image that shows: {blip}]"
    else:
        photo = ""
    return f"{text} {photo}".strip() if photo else text


def is_evidence(s):
    """True if a block response carries usable evidence."""
    if not s or not s.strip():
        return False
    s = s.strip()
    if s in _EMPTY_MARKERS or s.startswith("[ERROR]"):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="E-mem LoCoMo adapter (retrieval only)")
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

    session_ids = sorted(
        (int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))),
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[emem] sample={args.sample} sessions={len(session_ids)} model={MODEL}")
    manager = build_manager()
    handler = manager.memory_handler

    # ---- Phase 1: build memory (one add per turn, like the repo's eval) ----
    tracker.reset("build")
    t0 = time.time()
    n_turns, failed_turns = 0, 0
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "")
        for turn in (turns if isinstance(turns, list) else []):
            text = turn_text(turn)
            if not text:
                continue
            memory_text = f"[{date_str}] {turn.get('speaker', '')}: {text}"
            n_turns += 1
            for attempt in range(2):
                try:
                    handler.add_memory(memory_text)
                    break
                except Exception as e:
                    if attempt == 1:
                        failed_turns += 1
                        print(f"  turn FAILED after retry: {e}")
                    else:
                        time.sleep(5)
        print(f"  session {si} ingested ({time.time()-t0:.0f}s elapsed)")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    n_inactive = len(handler.inactive_memory_agents)
    active_agent = handler.add_handler.active_memory_agent
    n_blocks = n_inactive + (1 if active_agent is not None else 0)
    n_chunks = sum(len(a.current_block.chunks) for a in handler.inactive_memory_agents)
    if active_agent is not None:
        n_chunks += len(active_agent.current_block.chunks)
    print(f"[emem] build done: {build_time:.0f}s, {n_blocks} blocks ({n_chunks} chunks)")

    # ---- Phase 2: retrieval per question ----
    qas = sample["qa"]
    if args.questions_limit:
        qas = qas[: args.questions_limit]

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 1),
        "num_memories": n_blocks,
        "build_calls": build_snap["calls"],
        "build_tokens_in": build_snap["tokens_in"],
        "build_tokens_out": build_snap["tokens_out"],
        "build_llm_time_s": build_snap["llm_time_s"],
        "notes": (
            f"E-mem (ICML 2026) text mode, official repo third_party/emem; "
            f"llm={MODEL} for block summaries + per-block evidence extraction; "
            f"hybrid router: local all-MiniLM-L6-v2 (384d) + BM25(jieba), "
            f"bm25_boost=0.8, max_blocks=5, block=4096 tok (ratio 0.125 of 32768), "
            f"overlap chunk 0.1; sessions={len(session_ids)}, turns={n_turns}, "
            f"failed_turns={failed_turns}, chunks={n_chunks}; num_memories counts "
            f"memory blocks; k = per-question evidence items (block granularity), "
            f"master aggregation skipped (would synthesize an answer)"
        ),
    }]

    router = handler.query_handler.router
    for qi, qa in enumerate(qas):
        question = qa["question"]
        gold = qa.get("answer")
        if gold is None:
            gold = qa.get("adversarial_answer", "")
        tracker.reset("q")
        t_q = time.time()
        memories = []
        try:
            # Same parallel old/new split as TextMemoryHandler.query_memory,
            # but keeping per-block evidence instead of one joined string.
            with ThreadPoolExecutor(max_workers=2) as ex:
                old_f = ex.submit(router.map_reduce_blocks, question)
                new_f = ex.submit(handler.add_handler.query_new_agent, question)
                old_results = old_f.result()
                new_result = new_f.result()
            for r in old_results or []:
                if is_evidence(r):
                    memories.append({"text": r.strip(), "date": None})
            if is_evidence(new_result):
                memories.append({"text": new_result.strip(), "date": None})
            memories = memories[:MAX_MEMORIES]
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
            "retrieval": {"latency_s": round(latency, 3), "k": len(memories),
                          "calls": q_snap["calls"], "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(qas)} questions done")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[emem] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
