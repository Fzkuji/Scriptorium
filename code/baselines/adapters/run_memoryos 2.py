"""MemoryOS baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_memoryos.py" --sample 0 --output results/memoryos_s0.json
    python3 "src/adapters/run_memoryos.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_memoryos.json

Retrieval only: builds MemoryOS (EMNLP 2025, BAI-LAB/MemoryOS) memories from
LoCoMo sessions and dumps top-20 retrieved memories per question. Answering
and judging are done by src/evaluation/evaluate.py.

Model support findings (third_party/memoryos, memoryos-pypi variant)
--------------------------------------------------------------------
- Pure in-process Python: JSON files on disk + faiss (faiss-cpu) in-memory
  indexes. No Docker/Neo4j/Qdrant server/cloud API required.
- LLM: `Memoryos(openai_api_key=..., openai_base_url=..., llm_model=...)`
  uses a bare OpenAI SDK client (utils.OpenAIClient), so any OpenAI-compatible
  endpoint works -> BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY are fully supported.
- Embedder: `embedding_model_name="all-MiniLM-L6-v2"` (their default) runs a
  LOCAL SentenceTransformer via utils.get_embedding. 384d, no embedding API.
  NOTE: updater._get_embedding_for_page ignores the configured name and calls
  get_embedding(text) with its own default -- which is also all-MiniLM-L6-v2,
  so with our model choice the behavior is consistent.

In-process shim (no third_party file modified)
----------------------------------------------
The package lives in a hyphenated dir (third_party/memoryos/memoryos-pypi) and
several modules use relative imports (`from .utils import ...`) inside method
bodies, so it must be imported as a real package, not via sys.path flat
imports. We register the directory as package "memoryos" through
importlib.util.spec_from_file_location(submodule_search_locations=...).

Ingestion design (mirrors their official eval/main_loco_parse.py)
-----------------------------------------------------------------
- LoCoMo turns are paired into (user_input=speaker_a turn(s),
  agent_response=speaker_b turn(s)) QA pages; consecutive same-speaker turns
  are merged; a missing side gets a "(said nothing)" placeholder because
  updater.process_short_term_to_mid_term silently DROPS pages with an empty
  side (their official script loses such turns).
- Speaker names are prefixed into the text (their eval injects speaker names
  into the answer prompt instead; our unified answerer only sees memory text).
- blip_caption is appended as an image description, as in their eval script.
- timestamp = raw LoCoMo session date string ("1:56 pm on 8 May, 2023"); it is
  stored verbatim on pages, and mid-term recency only parses internal
  last_visit_time (compute_time_decay falls back gracefully), so this is safe.
- short_term_capacity=1 (their official LoCoMo eval also uses max_capacity=1)
  so every page flows to mid-term; after ingestion we flush the last page via
  updater.process_short_term_to_mid_term() + the heat-triggered profile update,
  otherwise trailing pages would be unretrievable.

Retrieval design
----------------
retriever.retrieve_context(query) with retrieval_queue_capacity=20 returns
mid-term pages + user knowledge + assistant knowledge. Memories are assembled
as [user profile (if any)] + pages + user knowledge + assistant knowledge,
truncated to top-20. Knowledge/profile entries get date=None (their stored
timestamps are build wall-clock time, which would be misleading temporal
signal); pages keep the LoCoMo session date.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import time

from _usage_tracker import tracker
tracker.install()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

MEMORYOS_PKG_DIR = os.path.join(PROJECT_ROOT, "third_party", "memoryos", "memoryos-pypi")


def _load_memoryos_pkg():
    """Register third_party/memoryos/memoryos-pypi as importable package 'memoryos'."""
    spec = importlib.util.spec_from_file_location(
        "memoryos",
        os.path.join(MEMORYOS_PKG_DIR, "__init__.py"),
        submodule_search_locations=[MEMORYOS_PKG_DIR],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["memoryos"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


memoryos_pkg = _load_memoryos_pkg()
Memoryos = memoryos_pkg.Memoryos

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MODEL = BUILDER_MODEL
TOP_K = 20
EMBED_MODEL = "all-MiniLM-L6-v2"


def turn_text(turn, speaker):
    """LoCoMo turn -> speaker-prefixed text with image caption, per their eval."""
    text = (turn.get("text") or "").strip()
    blip = (turn.get("blip_caption") or "").strip()
    if blip:
        text = f"{text} (image description: {blip})".strip()
    if not text:
        return ""
    return f"{speaker}: {text}"


def session_qa_pages(turns, speaker_a, speaker_b):
    """Pair LoCoMo turns into MemoryOS (user_input, agent_response) pages.

    Consecutive same-speaker turns are merged; a missing side gets a
    placeholder so the updater does not drop the page (see module docstring).
    """
    pages = []
    cur_user, cur_agent = [], []

    def flush():
        if not cur_user and not cur_agent:
            return
        pages.append({
            "user_input": "\n".join(cur_user) if cur_user else f"{speaker_a}: (said nothing)",
            "agent_response": "\n".join(cur_agent) if cur_agent else f"{speaker_b}: (said nothing)",
        })
        cur_user.clear()
        cur_agent.clear()

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker", "")
        text = turn_text(turn, speaker)
        if not text:
            continue
        if speaker == speaker_a:
            if cur_agent:  # previous pair complete -> start a new page
                flush()
            cur_user.append(text)
        else:
            cur_agent.append(text)
    flush()
    return pages


def setup_memoryos(sample_idx):
    workdir = tempfile.mkdtemp(prefix=f"memoryos_locomo_s{sample_idx}_")
    return Memoryos(
        user_id=f"locomo_s{sample_idx}",
        openai_api_key=BUILDER_KEY,
        openai_base_url=BUILDER_BASE,
        data_storage_path=workdir,
        llm_model=MODEL,
        embedding_model_name=EMBED_MODEL,
        short_term_capacity=1,       # official LoCoMo eval uses max_capacity=1
        retrieval_queue_capacity=TOP_K,
    )


def count_memories(mos):
    pages = sum(len(s.get("details", [])) for s in mos.mid_term_memory.sessions.values())
    user_kn = len(mos.user_long_term_memory.knowledge_base)
    asst_kn = len(mos.assistant_long_term_memory.assistant_knowledge)
    profile = mos.user_long_term_memory.get_raw_user_profile(mos.user_id)
    has_profile = int(bool(profile and profile.strip().lower() != "none"))
    return pages + user_kn + asst_kn + has_profile, pages, user_kn, asst_kn, has_profile


def main():
    parser = argparse.ArgumentParser(description="MemoryOS LoCoMo adapter (retrieval only)")
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
    speaker_b = conv.get("speaker_b", "Person B")

    session_ids = sorted(
        int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[memoryos] sample={args.sample} sessions={len(session_ids)} model={MODEL}")
    mos = setup_memoryos(args.sample)
    user_id = mos.user_id

    # ---- Phase 1: build memory (QA pages per session, dated with session date) ----
    tracker.reset("build")
    t0 = time.time()
    failed_pages = 0
    total_pages = 0
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "")
        pages = session_qa_pages(turns if isinstance(turns, list) else [], speaker_a, speaker_b)
        for page in pages:
            total_pages += 1
            for attempt in range(2):
                try:
                    mos.add_memory(
                        user_input=page["user_input"],
                        agent_response=page["agent_response"],
                        timestamp=date_str or None,
                    )
                    break
                except Exception as e:
                    if attempt == 1:
                        failed_pages += 1
                        print(f"  session {si} page FAILED after retry: {e}")
                    else:
                        time.sleep(5)
        print(f"  session {si} ingested ({len(pages)} pages, {time.time()-t0:.0f}s elapsed)")

    # Flush the trailing short-term page into mid-term (see module docstring).
    try:
        if mos.short_term_memory.is_full():
            mos.updater.process_short_term_to_mid_term()
            mos._trigger_profile_and_knowledge_update_if_needed()
    except Exception as e:
        print(f"  final flush failed: {e}")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    num_memories, n_pages, n_ukn, n_akn, n_prof = count_memories(mos)
    print(f"[memoryos] build done: {build_time:.0f}s, {num_memories} memories "
          f"(pages={n_pages}, user_kn={n_ukn}, asst_kn={n_akn}, profile={n_prof})")

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
            f"MemoryOS (BAI-LAB memoryos-pypi variant, faiss-cpu), llm={MODEL} via "
            f"OpenAI-compatible base_url, embedder={EMBED_MODEL} (local, 384d), "
            f"short_term_capacity=1 (per official LoCoMo eval), "
            f"retrieval_queue_capacity={TOP_K}, sessions={len(session_ids)}, "
            f"pages_total={total_pages}, pages_failed={failed_pages}, "
            f"memories = mid-term pages {n_pages} + user knowledge {n_ukn} + "
            f"assistant knowledge {n_akn} + profile {n_prof}"
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
            res = mos.retriever.retrieve_context(user_query=question, user_id=user_id)
            profile = mos.user_long_term_memory.get_raw_user_profile(user_id)
            if profile and profile.strip().lower() != "none":
                memories.append({"text": f"[User profile] {profile.strip()}", "date": None})
            for page in res.get("retrieved_pages", []):
                memories.append({
                    "text": f"{page.get('user_input', '')}\n{page.get('agent_response', '')}".strip(),
                    "date": page.get("timestamp"),
                })
            for kn in res.get("retrieved_user_knowledge", []):
                memories.append({"text": f"[User knowledge] {kn.get('knowledge', '')}", "date": None})
            for ak in res.get("retrieved_assistant_knowledge", []):
                memories.append({"text": f"[Assistant knowledge] {ak.get('knowledge', '')}", "date": None})
            memories = memories[:TOP_K]
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
                          "calls": q_snap["calls"], "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(qas)} questions done")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[memoryos] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
