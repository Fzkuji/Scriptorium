"""MemMachine baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_memmachine.py" --sample 0 --output results/memmachine_s0.json
    python3 "src/adapters/run_memmachine.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_memmachine.json

Retrieval only: builds MemMachine episodic memory from LoCoMo sessions and
dumps top-20 retrieved episodes per question. Answering and judging are done
by src/evaluation/evaluate.py.

Feasibility / model support findings (MemMachine main @ 2026-07, packages 0.2.x)
--------------------------------------------------------------------------------
- MemMachine is shipped as a client/server product (FastAPI + Postgres/Neo4j/
  Qdrant via docker-compose), BUT the memory core is an importable library
  (memmachine_server package) and the repo's own LoCoMo evaluation scripts
  (evaluation/episodic_memory/locomo_ingest.py / locomo_search.py) drive it
  fully IN-PROCESS -- no REST server. We follow the same in-process pattern.
- External services are all avoidable via config:
    * episode store + segment store: provider "sqlite" (aiosqlite, local file)
    * vector store: provider "sqlite_vector_store" (local file + usearch ANN)
      -> no Postgres, no Neo4j, no Qdrant server
- Embedder: native "sentence-transformer" provider (EmbedderManager builds a
  local SentenceTransformer). We use all-MiniLM-L6-v2 (384d). No embedding
  API needed.
- LLM: "openai-chat-completions" language-model provider accepts base_url +
  api_key, so BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY work. NOTE: the episodic
  ingest+search path used here is LLM-FREE (embedding search + RRF(identity,
  bm25) rerank). MemMachine's published LoCoMo 91.7 additionally uses an
  LLM-driven retrieval agent (ToolSelectAgent / ChainOfQueryAgent) at query
  time plus a GPT judge; our contract is plain top-20 retrieval, matching the
  other adapters, so the agent is not used.
- Backend choice: the repo's own locomo eval config uses the "declarative"
  long-term-memory backend, which ONLY supports Neo4j/NebulaGraph
  (VectorGraphStore). We instead use the officially supported "event" backend
  (LongTermMemory backend="event": VectorStore + SegmentStore + EpisodeStorage),
  which is the documented local/embedded alternative in
  sample_configs/episodic_memory_config.cpu.sample.

Environment / shim notes (no third_party file modified)
-------------------------------------------------------
- memmachine-server requires Python>=3.12 and heavy pinned deps
  (fastapi>=0.116, ...) that conflict with the system env used by other
  adapters, so it is installed into a dedicated venv:
      third_party/memmachine_venv   (created with --system-site-packages so
      the existing torch/sentence-transformers are reused)
  Install: SETUPTOOLS_SCM_PRETEND_VERSION=0.2.0 \
      third_party/memmachine_venv/bin/pip install \
      ./third_party/memmachine/packages/common \
      ./third_party/memmachine/packages/server
  This adapter re-execs itself into that venv's python at startup, so the
  contract command `python3 src/adapters/run_memmachine.py ...` still works.
- Ingest mirrors evaluation/episodic_memory/locomo_ingest.py: one Episode per
  message, created_at = session datetime + turn-index seconds, speaker as
  producer_id/producer_role. Deviation for cross-baseline fairness: blip
  image captions are appended to the message text (their ingest keeps them in
  metadata only, where the event backend never surfaces them); and
  filterable_metadata is set to {} so free-form metadata is not pushed into
  the vector-store property index (the collection schema does not declare
  user properties).
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ---- re-exec into the dedicated memmachine venv (see module docstring) ----
# NB: compare sys.prefix, not sys.executable -- the venv python is a symlink
# to the system interpreter, so realpath(executable) is identical for both.
VENV_DIR = os.path.join(PROJECT_ROOT, "third_party", "memmachine_venv")
VENV_PY = os.path.join(VENV_DIR, "bin", "python")
if os.path.exists(VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(VENV_DIR):
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

from _usage_tracker import tracker  # noqa: E402
tracker.install()

import asyncio  # noqa: E402

from scripts.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402,F401

from memmachine_server.common.configuration import Configuration  # noqa: E402
from memmachine_server.common.configuration.episodic_config import (  # noqa: E402
    EventLongTermMemoryConf,
)
from memmachine_server.common.episode_store import EpisodeEntry  # noqa: E402
from memmachine_server.common.metrics_factory import PrometheusMetricsFactory  # noqa: E402
from memmachine_server.common.resource_manager.resource_manager import (  # noqa: E402
    ResourceManagerImpl,
)
from memmachine_server.episodic_memory.episodic_memory import (  # noqa: E402
    EpisodicMemory,
    EpisodicMemoryParams,
)
from memmachine_server.episodic_memory.long_term_memory import LongTermMemory  # noqa: E402
from memmachine_server.episodic_memory.long_term_memory.service_locator import (  # noqa: E402
    long_term_memory_params_from_config,
)

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
TOP_K = 20

CONFIG_TEMPLATE = """\
logging:
  level: warning
  path: {workdir}/memmachine.log

episode_store:
  database: db_sqlite

episodic_memory:
  long_term_memory:
    backend: event
    embedder: st_embedder
    reranker: hybrid_reranker
    vector_store: event_vector_store
    segment_store: db_sqlite

semantic_memory:
  enabled: false
  config_database: db_sqlite

session_manager:
  database: db_sqlite

resources:
  databases:
    db_sqlite:
      provider: sqlite
      config:
        path: {workdir}/memmachine.db
    event_vector_store:
      provider: sqlite_vector_store
      config:
        path: {workdir}/event_vectors.db
        vector_search_engine: usearch
  embedders:
    st_embedder:
      provider: sentence-transformer
      config:
        model: sentence-transformers/all-MiniLM-L6-v2
  rerankers:
    ident_reranker:
      provider: identity
    bm25_reranker:
      provider: bm25
    hybrid_reranker:
      provider: rrf-hybrid
      config:
        reranker_ids:
          - ident_reranker
          - bm25_reranker
"""


def parse_locomo_date(date_str):
    """Parse LoCoMo date: '1:56 pm on 8 May, 2023' -> aware datetime (UTC)."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def turn_text(turn):
    """LoCoMo turn -> ingest text (speaker-prefixed by producer fields; captions kept)."""
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
    return text


async def build_memory(workdir, session_key):
    """In-process MemMachine: sqlite episode/segment stores + usearch vectors."""
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG_TEMPLATE.format(workdir=workdir))

    conf = Configuration.load_yml_file(config_path)
    rm = ResourceManagerImpl(conf)
    episode_storage = await rm.get_episode_storage()

    ltm_conf = EventLongTermMemoryConf(
        session_id=session_key,
        vector_store="event_vector_store",
        segment_store="db_sqlite",
        embedder="st_embedder",
        reranker="hybrid_reranker",
    )
    ltm_params = await long_term_memory_params_from_config(ltm_conf, rm)
    ltm = LongTermMemory(ltm_params)
    memory = EpisodicMemory(
        EpisodicMemoryParams(
            session_key=session_key,
            metrics_factory=PrometheusMetricsFactory(),
            long_term_memory=ltm,
            short_term_memory=None,
            enabled=True,
        ),
    )
    return rm, episode_storage, memory


async def run(args):
    with open(DATA_PATH) as f:
        data = json.load(f)
    sample = data[args.sample]
    conv = sample["conversation"]

    session_ids = sorted(
        (int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))),
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    workdir = tempfile.mkdtemp(prefix=f"memmachine_locomo_s{args.sample}_")
    session_key = f"locomo_s{args.sample}"
    print(f"[memmachine] sample={args.sample} sessions={len(session_ids)} workdir={workdir}")

    rm, episode_storage, memory = await build_memory(workdir, session_key)

    # ---- Phase 1: build memory (one episode per message, like their locomo_ingest) ----
    tracker.reset("build")
    t0 = time.time()
    num_episodes = 0
    failed_sessions = []
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "")
        session_dt = parse_locomo_date(date_str) or datetime.now(timezone.utc)
        entries = []
        for mi, turn in enumerate(turns if isinstance(turns, list) else []):
            if not isinstance(turn, dict):
                continue
            text = turn_text(turn)
            if not text:
                continue
            speaker = turn.get("speaker", "unknown")
            entries.append(
                EpisodeEntry(
                    content=f"{speaker}: {text}",
                    producer_id=speaker,
                    producer_role=speaker,
                    created_at=session_dt + mi * timedelta(seconds=1),
                    metadata={"source_timestamp": date_str, "source_speaker": speaker},
                )
            )
        if not entries:
            continue
        try:
            episodes = await episode_storage.add_episodes(session_key, entries)
            for ep in episodes:
                # Keep free-form metadata out of the vector-store property
                # index (collection schema declares no user properties).
                ep.filterable_metadata = {}
            await memory.add_memory_episodes(episodes)
            num_episodes += len(episodes)
            print(f"  session {si} ingested ({len(episodes)} msgs, {time.time()-t0:.0f}s elapsed)")
        except Exception as e:
            failed_sessions.append(si)
            print(f"  session {si} FAILED: {e}")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")
    print(f"[memmachine] build done: {build_time:.0f}s, {num_episodes} episodes")

    # ---- Phase 2: retrieval per question ----
    qas = sample["qa"]
    if args.questions_limit:
        qas = qas[: args.questions_limit]

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 1),
        "num_memories": num_episodes,
        "build_calls": build_snap["calls"],
        "build_tokens_in": build_snap["tokens_in"],
        "build_tokens_out": build_snap["tokens_out"],
        "build_llm_time_s": build_snap["llm_time_s"],
        "notes": (
            "MemMachine main (packages 0.2.x) in-process, event LTM backend "
            "(sqlite episode/segment store + sqlite_vector_store/usearch), "
            "embedder=all-MiniLM-L6-v2 (local, 384d), reranker=rrf(identity,bm25), "
            "LLM-free ingest+search (retrieval agent not used; see docstring), "
            f"sessions={len(session_ids)}, failed_sessions={failed_sessions}"
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
            resp = await memory.query_memory(question, limit=TOP_K)
            episodes = resp.long_term_memory.episodes if resp else []
            for ep in episodes:
                date = ep.created_at.isoformat() if ep.created_at else None
                memories.append({"text": ep.content, "date": date})
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
            "retrieval": {
                "latency_s": round(latency, 3),
                "k": TOP_K,
                "calls": q_snap["calls"],
                "tokens_in": q_snap["tokens_in"],
                "tokens_out": q_snap["tokens_out"],
            },
        })
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(qas)} questions done")

    await memory.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[memmachine] wrote {len(records)-1} question records -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description="MemMachine LoCoMo adapter (retrieval only)")
    parser.add_argument("--sample", type=int, default=0, help="LoCoMo conversation index")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Dev: only ingest first N sessions")
    parser.add_argument("--questions-limit", type=int, default=None,
                        help="Dev: only process first N questions")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
