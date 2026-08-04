"""Zep / Graphiti baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_zep.py" --sample 0 --output results/zep_s0.json
    python3 "src/adapters/run_zep.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_zep.json

Retrieval only: this adapter builds a Graphiti temporal knowledge graph from
LoCoMo sessions and dumps top-20 retrieved facts (entity edges) per question.
Answering and judging are done by src/evaluation/evaluate.py.

Model support findings (graphiti-core 0.29.2, matches third_party/graphiti)
---------------------------------------------------------------------------
- Graph DB: NO Neo4j server needed. graphiti_core ships a KuzuDriver
  (graphiti_core/driver/kuzu_driver.py) backed by the embedded Kuzu database
  (pip wheel, file-based, in-process). The Kuzu Python wheel statically
  bundles the FTS extension, so CREATE_FTS_INDEX / QUERY_FTS_INDEX work
  offline (verified in-process). Upstream marks the Kuzu backend deprecated
  but it is fully functional in 0.29.2. The `neo4j` pip package is a hard
  dependency but only as an import, no server connection is made.
- LLM: OpenAIGenericClient (graphiti_core/llm_client/openai_generic_client.py)
  targets any OpenAI-compatible /chat/completions endpoint via
  LLMConfig(base_url=...). We use structured_output_mode="json_object"
  because DeepSeek-family models do not support the native json_schema
  response format (the client injects the schema into the prompt instead).
  LLMConfig.small_model must also be set to the builder model, otherwise
  graphiti's "small" prompts default to gpt-4.1-nano.
- Embedder: graphiti_core only ships API embedders (openai/azure/gemini/
  voyage), but EmbedderClient is a 2-method ABC, so we provide an in-process
  local shim wrapping sentence-transformers all-MiniLM-L6-v2 (384d). The
  EMBEDDING_DIM env var must be set to 384 BEFORE importing graphiti_core
  (search.py builds a zero query-vector of that size as a fallback).
- Cross-encoder: Graphiti's default reranker (OpenAIRerankerClient) is only
  used by the advanced search_() recipes. The basic graphiti.search() used
  here is EDGE_HYBRID_SEARCH_RRF (semantic + BM25-style FTS fused with RRF),
  which never calls the cross-encoder; we still pass one configured with the
  builder endpoint so no OPENAI_API_KEY is required at construction time.

Ingestion granularity: one episode per LoCoMo session (speaker-prefixed
turns, image captions inlined), reference_time = parsed session date, so
extracted facts get correct valid_at timestamps. This matches the
one-add-per-session granularity of run_mem0.py.

Memory unit dumped per question: EntityEdge.fact, dated by valid_at
(fallback created_at) — the same "facts" surface Zep uses in its own
LoCoMo/LongMemEval reports.

Patches to third_party
----------------------
None. No file in third_party/graphiti was modified. The pip-installed
graphiti-core 0.29.2 (with the [kuzu] extra) matches the shallow clone in
third_party/graphiti and is used as-is. Two in-process shims live in this
file only:
1. LocalMiniLMEmbedder — subclass of the public EmbedderClient ABC.
2. ensure_kuzu_fts_indices() — upstream bug workaround: KuzuDriver overrides
   build_indices_and_constraints() as a no-op (kuzu_driver.py:252) and
   setup_schema() never runs CREATE_FTS_INDEX, so graphiti.search() dies with
   "Table RelatesToNode_ doesn't have an index with name edge_name_and_fact".
   We run get_fulltext_indices(GraphProvider.KUZU) manually on the driver
   (exactly what KuzuGraphMaintenanceOperations.build_indices_and_constraints
   would do if it were wired up). Verified: Kuzu FTS indices auto-update on
   later inserts, so creating them once up front is sufficient.
3. driver._database = group_id — upstream bug workaround: add_episode with an
   explicit group_id compares it against self.driver._database
   (graphiti.py:1079), but KuzuDriver never initializes the _database
   attribute (it is a bare class annotation on GraphDriver), so every
   add_episode raises AttributeError. Pre-setting the instance attribute to
   our group_id makes the check a no-op (equal values -> no clone attempt).
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone

# Must be set before importing graphiti_core (see module docstring).
os.environ["EMBEDDING_DIM"] = "384"
os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from _usage_tracker import tracker  # noqa: E402
tracker.install()

from scripts.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Kuzu backend deprecation

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # noqa: E402
from graphiti_core.driver.driver import GraphProvider  # noqa: E402
from graphiti_core.driver.kuzu_driver import KuzuDriver  # noqa: E402
from graphiti_core.graph_queries import get_fulltext_indices  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MODEL = BUILDER_MODEL
TOP_K = 20


class LocalMiniLMEmbedder(EmbedderClient):
    """In-process sentence-transformers embedder (all-MiniLM-L6-v2, 384d)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def create(self, input_data):
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = [str(t) for t in input_data]
        embs = self.model.encode(texts, normalize_embeddings=True)
        return embs[0].tolist()

    async def create_batch(self, input_data_list):
        embs = self.model.encode(list(input_data_list), normalize_embeddings=True)
        return [e.tolist() for e in embs]


def parse_locomo_date(date_str):
    """Parse LoCoMo date: '1:56 pm on 8 May, 2023' -> aware datetime (UTC)."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def session_episode_body(turns, max_chars_per_turn=2000):
    """LoCoMo turns -> one 'speaker: text' transcript (image captions kept)."""
    lines = []
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
        lines.append(f"{speaker}: {text[:max_chars_per_turn]}")
    return "\n".join(lines)


def make_graphiti(sample_idx):
    """Graphiti with embedded Kuzu + builder LLM + local MiniLM embeddings."""
    workdir = tempfile.mkdtemp(prefix=f"zep_locomo_s{sample_idx}_")
    llm_config = LLMConfig(
        api_key=BUILDER_KEY,
        model=MODEL,
        small_model=MODEL,  # otherwise graphiti falls back to gpt-4.1-nano
        base_url=BUILDER_BASE,
        temperature=0.0,
    )
    llm = OpenAIGenericClient(config=llm_config, structured_output_mode="json_object")
    driver = KuzuDriver(db=os.path.join(workdir, "kuzu_db"))
    return Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=LocalMiniLMEmbedder(),
        cross_encoder=OpenAIRerankerClient(config=llm_config),  # unused by RRF search
        max_coroutines=5,  # be gentle with the Aliyun endpoint
    ), driver, workdir


async def ensure_kuzu_fts_indices(driver):
    """Shim for an upstream no-op (see module docstring): create the FTS
    indices that KuzuDriver.build_indices_and_constraints() forgets to."""
    for q in get_fulltext_indices(GraphProvider.KUZU):
        try:
            await driver.execute_query(q)
        except Exception as e:
            if "already exists" not in str(e):
                raise


async def count_facts(driver, group_id):
    """Count extracted entity edges (facts). Kuzu stores edge props on RelatesToNode_."""
    records, _, _ = await driver.execute_query(
        "MATCH (e:RelatesToNode_) WHERE e.group_id = $gid RETURN count(e) AS c",
        gid=group_id,
    )
    return records[0]["c"] if records else 0


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

    print(f"[zep] sample={args.sample} sessions={len(session_ids)} model={MODEL}")
    graphiti, driver, workdir = make_graphiti(args.sample)
    group_id = f"locomo_s{args.sample}"
    # Shim 3 (see module docstring): KuzuDriver never sets _database, and
    # add_episode(group_id=...) reads it. Equal values -> no clone attempt.
    driver._database = group_id
    await graphiti.build_indices_and_constraints()  # no-op for Kuzu upstream
    await ensure_kuzu_fts_indices(driver)

    # ---- Phase 1: build graph (one episode per session, dated) ----
    tracker.reset("build")
    t0 = time.time()
    failed_sessions = []
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "")
        body = session_episode_body(turns if isinstance(turns, list) else [])
        if not body:
            continue
        ref_time = parse_locomo_date(date_str) or datetime(2023, 1, 1, tzinfo=timezone.utc)
        for attempt in range(2):
            try:
                await graphiti.add_episode(
                    name=f"locomo_s{args.sample}_session_{si}",
                    episode_body=body,
                    source_description=f"LoCoMo conversation session {si} on {date_str}",
                    reference_time=ref_time,
                    source=EpisodeType.message,
                    group_id=group_id,
                )
                break
            except Exception as e:
                if attempt == 1:
                    failed_sessions.append(si)
                    print(f"  session {si} FAILED after retry: {type(e).__name__}: {e}")
                else:
                    time.sleep(5)
        print(f"  session {si} ingested ({time.time()-t0:.0f}s elapsed)")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    try:
        num_memories = await count_facts(driver, group_id)
    except Exception as e:
        print(f"  fact count failed: {e}")
        num_memories = -1
    print(f"[zep] build done: {build_time:.0f}s, {num_memories} facts")

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
            f"graphiti-core 0.29.2 (Graphiti OSS core; not the commercial Zep "
            f"platform), llm={MODEL}, endpoint={BUILDER_BASE}, OpenAIGenericClient "
            f"json_object mode, embedder=all-MiniLM-L6-v2 (local shim, 384d), "
            f"graph=embedded Kuzu (no server), search=EDGE_HYBRID_SEARCH_RRF, "
            f"1 episode per session, sessions={len(session_ids)}, "
            f"failed_sessions={failed_sessions}, workdir={workdir}"
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
            edges = await graphiti.search(question, group_ids=[group_id], num_results=TOP_K)
            for e in edges:
                date = e.valid_at or e.created_at
                memories.append({
                    "text": e.fact,
                    "date": date.isoformat() if date else None,
                })
        except Exception as e:
            print(f"  q{qi} search error: {type(e).__name__}: {e}")
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[zep] wrote {len(records)-1} question records -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Zep/Graphiti LoCoMo adapter (retrieval only)")
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
