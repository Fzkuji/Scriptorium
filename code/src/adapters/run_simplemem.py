"""SimpleMem baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_simplemem.py" --sample 0 --output results/simplemem_s0.json
    python3 "src/adapters/run_simplemem.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_simplemem.json

Retrieval only: this adapter builds SimpleMem memory entries from LoCoMo
sessions and dumps top-20 retrieved memories per question. Answering and
judging are done by src/evaluation/evaluate.py.

Model support findings (aiming-lab/SimpleMem @ 60a48e8, 2026-06-23)
-------------------------------------------------------------------
- LLM config: simplemem.core.utils.llm_client.LLMClient is a bare OpenAI SDK
  client; SimpleMemSystem(api_key=..., model=..., base_url=...) passes
  straight through, so ANY OpenAI-compatible endpoint works ->
  BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY (deepseek-v4-flash @ Aliyun
  compatible-mode) are fully supported. We disable streaming
  (use_streaming=False) for robustness with the Aliyun endpoint.
- Embedder: simplemem.core.utils.embedding.EmbeddingModel wraps a LOCAL
  SentenceTransformer (no embedding API anywhere). Any model name not
  starting with "qwen3" takes the standard SentenceTransformer path and the
  vector dimension is read from the model at runtime (LanceDB schema uses
  embedding_model.dimension), so EMBEDDING_MODEL=
  sentence-transformers/all-MiniLM-L6-v2 (384d) just works.
- Storage: embedded LanceDB at a local path (+ tantivy FTS index for the
  lexical/BM25 layer). Both are plain pip packages (lancedb, tantivy) --
  no Docker/Neo4j/Postgres/Qdrant server, no cloud API.
- Config plumbing: simplemem.core.settings.Settings resolves each key as
  user config.py > env var > default. There is no config.py on sys.path in
  this project, so we configure via environment variables (set below BEFORE
  importing simplemem; Settings reads os.getenv at attribute access time)
  plus SimpleMemSystem constructor args (db_path/table_name per run).

Retrieval semantics
-------------------
system.hybrid_retriever.retrieve(q) runs SimpleMem's published pipeline
(LLM retrieval planning -> parallel semantic top-25 / keyword top-5 /
structured top-5 -> merge/dedup -> LLM reflection rounds) and returns an
insertion-ordered deduplicated List[MemoryEntry] (semantic hits are
cosine-ranked first). We do NOT call ask()/AnswerGenerator -- no answering.
The merged list can exceed 20 entries; we truncate to the first 20 to meet
the unified top-20 contract (the pre-truncation count is recorded per
question under retrieval.n_retrieved). Each memory's date is the entry's
LLM-normalized ISO timestamp (SimpleMem grounds absolute timestamps at
build time from the session date we attach to every Dialogue, mirroring the
upstream test_locomo10.py protocol).

Patches to third_party
----------------------
No file in third_party/simplemem was modified. One IN-PROCESS SHIM (see
_shim_native_fts_index below): upstream VectorStore._init_fts_index creates
the FTS index with use_tantivy=True for local storage, but lancedb >= 0.30
(we have 0.33; upstream pins 0.25.3) can no longer QUERY tantivy-based
indexes -- table.search("text") fails with "Cannot perform full text search
unless an INVERTED index has been created", silently disabling SimpleMem's
lexical/BM25 layer. The shim replaces the bound _init_fts_index on the
VectorStore INSTANCE so the index is built with use_tantivy=False (LanceDB
native inverted index, same BM25 scoring). Installed pip deps:
lancedb, dateparser, tantivy (all local/in-process).
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _usage_tracker import tracker  # noqa: E402
tracker.install()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "simplemem"))

from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

# Settings resolution is config.py > env > default; no config.py exists here,
# so env vars set before component construction are authoritative.
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["EMBEDDING_DIMENSION"] = "384"
os.environ["USE_STREAMING"] = "false"
os.environ["MAX_PARALLEL_WORKERS"] = "4"      # builder LLM concurrency (be nice to the API)
os.environ["MAX_RETRIEVAL_WORKERS"] = "4"
os.environ["OPENAI_API_KEY"] = BUILDER_KEY

from simplemem.text.system import SimpleMemSystem  # noqa: E402
from simplemem.core.models.memory_entry import Dialogue  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MODEL = BUILDER_MODEL
TOP_K = 20


def _shim_native_fts_index(store):
    """In-process shim: build the FTS index as a native LanceDB inverted index.

    Upstream uses use_tantivy=True locally, which lancedb 0.33 can create but
    not query (needs an INVERTED index) -> keyword_search always returned [].
    Replacing the instance's _init_fts_index (called by add_entries after the
    first insertion) with a use_tantivy=False variant restores the lexical
    layer without touching third_party code.
    """
    def _init_fts_index_native():
        if store._fts_initialized:
            return
        try:
            store.table.create_fts_index(
                "lossless_restatement", use_tantivy=False, replace=True)
            store._fts_initialized = True
            print("FTS index created (native inverted index, adapter shim)")
        except Exception as e:
            print(f"FTS index creation skipped (shim): {e}")
    store._init_fts_index = _init_fts_index_native


def turn_text(turn):
    """LoCoMo turn -> text (image captions folded in, upstream test_locomo10 style)."""
    text = turn.get("text", "") or ""
    blip = turn.get("blip_caption", "") or ""
    if blip:
        caption = f"[Image: {blip}]"
        text = f"{caption} {text}".strip() if text else caption
    return text


def main():
    parser = argparse.ArgumentParser(description="SimpleMem LoCoMo adapter (retrieval only)")
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
        int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[simplemem] sample={args.sample} sessions={len(session_ids)} model={MODEL}")

    workdir = tempfile.mkdtemp(prefix=f"simplemem_locomo_s{args.sample}_")
    system = SimpleMemSystem(
        api_key=BUILDER_KEY,
        model=MODEL,
        base_url=BUILDER_BASE,
        db_path=os.path.join(workdir, "lancedb"),
        table_name=f"locomo_s{args.sample}",
        clear_db=False,
        use_streaming=False,
    )
    _shim_native_fts_index(system.vector_store)

    # ---- Phase 1: build memory (all sessions in order, session date on every turn) ----
    dialogues = []
    dialogue_id = 1
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "") or None
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            text = turn_text(turn)
            if not text:
                continue
            dialogues.append(Dialogue(
                dialogue_id=dialogue_id,
                speaker=turn.get("speaker", "unknown"),
                content=text,
                timestamp=date_str,  # raw LoCoMo date string, as in upstream test_locomo10.py
            ))
            dialogue_id += 1

    tracker.reset("build")
    t0 = time.time()
    build_error = None
    try:
        system.add_dialogues(dialogues)
        system.finalize()
    except Exception as e:
        build_error = str(e)
        print(f"[simplemem] build error (continuing with partial memory): {e}")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    try:
        num_memories = len(system.get_all_memories())
    except Exception as e:
        print(f"  get_all_memories failed: {e}")
        num_memories = -1
    print(f"[simplemem] build done: {build_time:.0f}s, {num_memories} memory entries "
          f"from {len(dialogues)} dialogues")

    # ---- Phase 2: retrieval per question (no answering) ----
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
            f"SimpleMem (aiming-lab @ 60a48e8), llm={MODEL} via OpenAI-compatible base_url, "
            f"embedder=all-MiniLM-L6-v2 (local SentenceTransformer, 384d), LanceDB embedded "
            f"(native inverted FTS via adapter shim), window=40 overlap=2, "
            f"retrieval=planning+semantic25/keyword5/"
            f"structured5+reflection (upstream defaults), merged multi-view result truncated "
            f"to top-{TOP_K}, sessions={len(session_ids)}, dialogues={len(dialogues)}, "
            f"build_error={build_error}"
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
        n_retrieved = 0
        try:
            entries = system.hybrid_retriever.retrieve(question)
            n_retrieved = len(entries)
            for e in entries[:TOP_K]:
                memories.append({
                    "text": e.lossless_restatement,
                    "date": e.timestamp or None,
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
                          "n_retrieved": n_retrieved,
                          "calls": q_snap["calls"],
                          "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        print(f"  q{qi+1}/{len(qas)} done ({len(memories)} memories, {latency:.1f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[simplemem] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
