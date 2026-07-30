#!/usr/bin/env python3
"""Nemori baseline adapter for the unified LoCoMo evaluation protocol.

Baseline: Nemori (github.com/nemori-ai/nemori, paper arXiv:2508.03341).
Self-organising episodic + semantic memory: LLM segments conversations into
topic-coherent episodes (Event Segmentation Theory), then a predict-calibrate
loop distils semantic facts. Search = vector retrieval over both stores.

Usage (from project root):
    python3 "src/adapters/run_nemori.py" --sample 0 --output results/nemori-locomo-s0/questions.json
    python3 "src/adapters/run_nemori.py" --sample 0 --max-sessions 2 --questions-limit 3 \
        --output /tmp/adapter_smoke_nemori.json

=== Model support survey (2026-07-02, repo @ main, v0.2.0 rewrite) ===

LLM configuration
    - Bare OpenAI SDK (``openai.AsyncOpenAI``), no litellm, no provider
      abstraction beyond a small ``LLMProvider`` Protocol
      (nemori/domain/interfaces.py). ``AsyncLLMClient`` (nemori/llm/client.py)
      accepts arbitrary ``base_url`` + ``api_key``; model name is any string.
    - => any OpenAI-compatible endpoint works. Verified: deepseek-v4-flash via
      Aliyun compatible-mode responds fine, INCLUDING
      ``response_format={"type": "json_object"}`` which Nemori's episode /
      segmentation / semantic generators all rely on.
    - Officially documented models: "openai/gpt-4.1-mini" via OpenRouter
      (README + evaluation/locomo/config.json), default "gpt-4o-mini";
      README recommends OpenRouter or direct OpenAI.

Embedder
    - ``AsyncEmbeddingClient`` (nemori/services/embedding.py) is OpenAI
      embeddings API only — NO built-in local sentence-transformers support.
    - BUT embedding is injected via the ``EmbeddingProvider`` Protocol
      (async embed / embed_batch), and the facade auto-probes the dimension,
      so any dimension works (Qdrant collections are created with the probed
      dim). We inject a local sentence-transformers all-MiniLM-L6-v2 (384-d)
      provider — no third_party code modified.
    - Documented embedding models: "google/gemini-embedding-001" (OpenRouter),
      default "text-embedding-3-small".

Storage / architecture weight
    - v0.2.0 requires PostgreSQL 16 (asyncpg, pg_trgm) + Qdrant, normally via
      ``docker compose up``. Neither Docker nor PostgreSQL exists on this
      machine, so the official ``NemoriMemory`` facade cannot run as-is.
    - Workaround (no server processes needed): all stores are Protocol-typed
      and wired in nemori/factory.py, so this adapter assembles
      ``MemorySystem`` directly with:
        * InMemoryEpisodeStore / InMemorySemanticStore / InMemoryBufferStore
          (pure-dict implementations of the Protocols in
          nemori/domain/interfaces.py; search_by_text = token-overlap, only
          used by text/hybrid mode — we use Nemori's own LoCoMo eval default,
          vector),
        * InProcessQdrantStore: subclass of QdrantVectorStore whose client is
          ``QdrantClient(location=":memory:")`` (qdrant-client local mode) —
          same upsert/query_points code paths, zero servers,
        * LocalSTEmbedding: sentence-transformers all-MiniLM-L6-v2, 384-d.
      The LLM path (AsyncLLMClient -> LLMOrchestrator -> Episode/Semantic
      generators, BatchSegmenter, EpisodeMerger) is 100% upstream Nemori code.

Patches to third_party/nemori: NONE (everything is dependency injection /
subclassing from this file). ``parse_timestamp`` and the message-building
logic are vendored from evaluation/locomo/add.py.

Dependencies installed for this adapter: ``pip install asyncpg`` (imported by
nemori.core.memory_system at import time even though unused here); openai,
qdrant-client, tiktoken, pydantic, Pillow, sentence-transformers were already
present. Repo is used from source via sys.path (no ``pip install -e``).

Adaptation decisions (documented deviations, no upstream behaviour changed):
    - Upstream evaluation/locomo/add.py uses buffer_size_min=1 with
      batch-size-1 pushes, so episode granularity depends on asyncio/LLM-
      latency races. For determinism we push one full session per
      ``add_messages`` call and ``flush`` after each session
      (buffer_size_min/max set high to disable the racy auto-trigger).
      Sessions with >= batch_threshold(20) messages still go through Nemori's
      LLM BatchSegmenter for boundary detection, as in the paper.
    - Upstream LoCoMo eval retrieves top-10 episodes + top-20 semantic and
      answers inline. Our contract needs top-20 unified memories and NO
      answering, so we retrieve top-10 episodes + top-10 semantic
      (k=20 total), method=vector (upstream eval default), and emit
      {"text", "date"} records; answerer/judge run in
      src/evaluation/evaluate.py.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "nemori"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _usage_tracker import tracker
tracker.install()

# BUILDER_* default to deepseek-v4-flash @ Aliyun with ALIYUN_KEY env fallback.
from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY

from qdrant_client import QdrantClient

from nemori.config import MemoryConfig
from nemori.core.memory_system import MemorySystem
from nemori.db.qdrant_store import QdrantVectorStore
from nemori.domain.models import Message, Episode, SemanticMemory
from nemori.llm.client import AsyncLLMClient
from nemori.llm.orchestrator import LLMOrchestrator
from nemori.llm.generators.episode import EpisodeGenerator
from nemori.llm.generators.semantic import SemanticGenerator
from nemori.llm.generators.merger import EpisodeMerger
from nemori.services.event_bus import EventBus
from nemori.search.unified import UnifiedSearch, SearchMethod

DATA_PATH = os.path.join(ROOT, "benchmarks", "locomo", "data", "locomo10.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K_EPISODES = 10
TOP_K_SEMANTIC = 10  # 10 + 10 = unified k=20


# ---------------------------------------------------------------------------
# In-memory implementations of Nemori's storage Protocols
# (nemori/domain/interfaces.py) — replaces PostgreSQL.
# ---------------------------------------------------------------------------

def _text_overlap_score(query: str, doc: str) -> float:
    q = set(query.lower().split())
    d = set(doc.lower().split())
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


class InMemoryEpisodeStore:
    def __init__(self):
        self._data = {}  # (agent_id, user_id) -> {episode_id: Episode}

    def _bucket(self, user_id, agent_id):
        return self._data.setdefault((agent_id, user_id), {})

    async def save(self, episode: Episode) -> None:
        self._bucket(episode.user_id, episode.agent_id)[episode.id] = episode

    async def get(self, episode_id, user_id, agent_id):
        return self._bucket(user_id, agent_id).get(episode_id)

    async def list_by_user(self, user_id, agent_id, limit=100, offset=0):
        eps = sorted(self._bucket(user_id, agent_id).values(),
                     key=lambda e: e.created_at)
        return eps[offset:offset + limit]

    async def delete(self, episode_id, user_id, agent_id) -> None:
        self._bucket(user_id, agent_id).pop(episode_id, None)

    async def delete_by_user(self, user_id, agent_id) -> None:
        self._data.pop((agent_id, user_id), None)

    async def search_by_text(self, user_id, agent_id, query, top_k):
        scored = [(_text_overlap_score(query, f"{e.title} {e.content}"), e)
                  for e in self._bucket(user_id, agent_id).values()]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    async def get_batch(self, episode_ids, user_id, agent_id):
        bucket = self._bucket(user_id, agent_id)
        return [bucket[i] for i in episode_ids if i in bucket]


class InMemorySemanticStore:
    def __init__(self):
        self._data = {}  # (agent_id, user_id) -> {memory_id: SemanticMemory}

    def _bucket(self, user_id, agent_id):
        return self._data.setdefault((agent_id, user_id), {})

    async def save(self, memory: SemanticMemory) -> None:
        self._bucket(memory.user_id, memory.agent_id)[memory.id] = memory

    async def save_batch(self, memories) -> None:
        for m in memories:
            await self.save(m)

    async def get(self, memory_id, user_id, agent_id):
        return self._bucket(user_id, agent_id).get(memory_id)

    async def list_by_user(self, user_id, agent_id, memory_type=None):
        mems = sorted(self._bucket(user_id, agent_id).values(),
                      key=lambda m: m.created_at)
        if memory_type is not None:
            mems = [m for m in mems if m.memory_type == memory_type]
        return mems

    async def delete(self, memory_id, user_id, agent_id) -> None:
        self._bucket(user_id, agent_id).pop(memory_id, None)

    async def delete_by_user(self, user_id, agent_id) -> None:
        self._data.pop((agent_id, user_id), None)

    async def search_by_text(self, user_id, agent_id, query, top_k):
        scored = [(_text_overlap_score(query, m.content), m)
                  for m in self._bucket(user_id, agent_id).values()]
        scored = [(s, m) for s, m in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    async def get_batch(self, memory_ids, user_id, agent_id):
        bucket = self._bucket(user_id, agent_id)
        return [bucket[i] for i in memory_ids if i in bucket]


class InMemoryBufferStore:
    def __init__(self):
        self._buf = {}  # (agent_id, user_id) -> list[(id, Message)]
        self._next_id = 0

    def _bucket(self, user_id, agent_id):
        return self._buf.setdefault((agent_id, user_id), [])

    async def push(self, user_id, agent_id, messages) -> None:
        bucket = self._bucket(user_id, agent_id)
        for m in messages:
            self._next_id += 1
            bucket.append((self._next_id, m))

    async def get_unprocessed(self, user_id, agent_id):
        out = []
        for bid, m in self._bucket(user_id, agent_id):
            out.append(Message(
                role=m.role, content=m.content, timestamp=m.timestamp,
                metadata={**m.metadata, "buffer_id": bid},
                message_id=m.message_id,
            ))
        return out

    async def mark_processed(self, user_id, agent_id, message_ids) -> None:
        done = set(message_ids)
        key = (agent_id, user_id)
        self._buf[key] = [(bid, m) for bid, m in self._buf.get(key, [])
                          if bid not in done]

    async def count_unprocessed(self, user_id, agent_id) -> int:
        return len(self._buf.get((agent_id, user_id), []))


class InProcessQdrantStore(QdrantVectorStore):
    """Upstream QdrantVectorStore, but on qdrant-client's in-process local
    mode (:memory:) instead of a Qdrant server. Only __init__ overridden."""

    def __init__(self, collection_prefix="nemori"):
        self._client = QdrantClient(location=":memory:")
        self._prefix = collection_prefix
        self._episodes_collection = f"{collection_prefix}_episodes"
        self._semantic_collection = f"{collection_prefix}_semantic"


class LocalSTEmbedding:
    """EmbeddingProvider Protocol impl backed by local sentence-transformers
    (all-MiniLM-L6-v2, 384-d). Replaces Nemori's OpenAI-only embedder."""

    def __init__(self, model_name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._lock = asyncio.Lock()

    async def embed(self, text):
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts):
        async with self._lock:  # serialize; ST encode is not re-entrant
            vecs = await asyncio.to_thread(
                self._model.encode, texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# System assembly (mirrors nemori/factory.py with injected components)
# ---------------------------------------------------------------------------

def build_config() -> MemoryConfig:
    return MemoryConfig(
        llm_model=BUILDER_MODEL,
        llm_api_key=BUILDER_KEY,
        llm_base_url=BUILDER_BASE,
        llm_max_concurrent=8,
        embedding_model=EMBED_MODEL,
        embedding_dimension=EMBED_DIM,
        # Disable the racy per-push auto-processing; we flush per session.
        buffer_size_min=100000,
        buffer_size_max=100000,
        # Match evaluation/locomo/config.json where it differs from defaults.
        episode_min_messages=1,
        episode_max_messages=20,
        enable_semantic_memory=True,
        enable_prediction_correction=True,
        search_top_k_episodes=TOP_K_EPISODES,
        search_top_k_semantic=TOP_K_SEMANTIC,
    )


def build_system(config: MemoryConfig):
    episode_store = InMemoryEpisodeStore()
    semantic_store = InMemorySemanticStore()
    buffer_store = InMemoryBufferStore()
    qdrant = InProcessQdrantStore()
    qdrant.ensure_collections(config.embedding_dimension)

    llm_client = AsyncLLMClient(api_key=config.llm_api_key,
                                base_url=config.llm_base_url)
    orchestrator = LLMOrchestrator(
        provider=llm_client,
        default_model=config.llm_model,
        max_concurrent=config.llm_max_concurrent,
    )
    embedding = LocalSTEmbedding()

    episode_gen = EpisodeGenerator(orchestrator=orchestrator, embedding=embedding)
    semantic_gen = SemanticGenerator(
        orchestrator=orchestrator, embedding=embedding,
        enable_prediction_correction=config.enable_prediction_correction,
    )
    merger = EpisodeMerger(
        orchestrator=orchestrator, embedding=embedding,
        episode_store=episode_store, qdrant=qdrant,
        similarity_threshold=config.merge_similarity_threshold,
        merge_top_k=config.merge_top_k,
    ) if config.enable_episode_merging else None
    search = UnifiedSearch(episode_store, semantic_store, embedding, qdrant)

    system = MemorySystem(
        config=config,
        agent_id=config.agent_id,
        db=None,  # unused by MemorySystem; PostgreSQL replaced by stores above
        episode_store=episode_store,
        semantic_store=semantic_store,
        buffer_store=buffer_store,
        orchestrator=orchestrator,
        embedding=embedding,
        episode_generator=episode_gen,
        semantic_generator=semantic_gen,
        event_bus=EventBus(),
        search=search,
        merger=merger,
        qdrant=qdrant,
    )
    return system, episode_store, semantic_store, orchestrator


# ---------------------------------------------------------------------------
# LoCoMo ingestion (vendored from third_party/nemori/evaluation/locomo/add.py)
# ---------------------------------------------------------------------------

def parse_timestamp(value: str) -> datetime:
    """Parse dataset timestamps such as '1:56 pm on 8 May, 2023'."""
    value = " ".join(value.split())
    if " on " not in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now()
    time_part, date_part = value.split(" on ")
    time_part = time_part.lower().strip()
    hour = 0
    minute = 0
    is_pm = "pm" in time_part
    time_part = time_part.replace("pm", "").replace("am", "").strip()
    if ":" in time_part:
        hour_str, minute_str = time_part.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    else:
        hour = int(time_part)
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    parts = date_part.replace(",", "").split()
    day = 1
    month = 1
    year = datetime.now().year
    for part in parts:
        lower = part.lower()
        if lower in months:
            month = months[lower]
        elif part.isdigit():
            num = int(part)
            if num > 31:
                year = num
            else:
                day = num
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute)


def session_keys(conversation):
    keys = [k for k in conversation
            if k.startswith("session_") and not k.endswith("_date_time")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def session_messages(conversation, key):
    """One LoCoMo session -> list of Nemori Messages (add.py conventions)."""
    raw_ts = conversation.get(f"{key}_date_time")
    ts = parse_timestamp(raw_ts) if raw_ts else datetime.now()
    messages = []
    for chat in conversation.get(key) or []:
        speaker = chat.get("speaker", "user")
        parts = [chat.get("text", "")]
        if chat.get("blip_caption"):
            parts.append(f"[Image: {chat['blip_caption']}]")
        if chat.get("query"):
            parts.append(f"[Search: {chat['query']}]")
        messages.append(Message(
            role=speaker,
            content=" ".join(parts),
            timestamp=ts,
            metadata={
                "original_speaker": speaker,
                "dataset_timestamp": raw_ts,
                "dia_id": chat.get("dia_id"),
            },
        ))
    return messages


async def ingest(system, conversation, user_id, max_sessions=None):
    keys = session_keys(conversation)
    if max_sessions is not None:
        keys = keys[:max_sessions]
    n_episodes = 0
    for key in keys:
        msgs = session_messages(conversation, key)
        if not msgs:
            continue
        await system.add_messages(user_id, msgs)
        episodes = await system.flush(user_id)
        n_episodes += len(episodes)
        print(f"  {key}: {len(msgs)} msgs -> {len(episodes)} episodes")
    await system.drain(timeout=60.0)
    return len(keys), n_episodes


# ---------------------------------------------------------------------------
# QA retrieval
# ---------------------------------------------------------------------------

def memory_record(item) -> dict:
    if isinstance(item, Episode):
        return {"text": f"[Episode] {item.title}: {item.content}",
                "date": item.created_at.isoformat(sep=" ")}
    return {"text": f"[Fact] {item.content}",
            "date": item.created_at.isoformat(sep=" ")}


async def answer_questions(system, qa_list, user_id, sample_idx, limit=None):
    if limit is not None:
        qa_list = qa_list[:limit]
    records = []
    for idx, qa in enumerate(qa_list):
        question = qa.get("question", "")
        tracker.reset("q")
        t0 = time.perf_counter()
        result = await system.search(
            user_id, question,
            top_k_episodes=TOP_K_EPISODES,
            top_k_semantic=TOP_K_SEMANTIC,
            method=SearchMethod.VECTOR,
        )
        latency = time.perf_counter() - t0
        q_snap = tracker.snapshot("q")
        memories = ([memory_record(e) for e in result.episodes]
                    + [memory_record(m) for m in result.semantic_memories])
        gold = qa.get("answer", qa.get("adversarial_answer", ""))
        records.append({
            "question_id": f"s{sample_idx}_q{idx}",
            "question": question,
            "gold": str(gold),
            "category": qa.get("category"),
            "memories": memories,
            "retrieval": {"latency_s": round(latency, 4),
                          "k": TOP_K_EPISODES + TOP_K_SEMANTIC,
                          "calls": q_snap["calls"],
                          "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        if (idx + 1) % 20 == 0:
            print(f"  retrieved {idx + 1}/{len(qa_list)}")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    with open(DATA_PATH) as f:
        dataset = json.load(f)
    item = dataset[args.sample]
    conversation = item["conversation"]
    user_id = f"{conversation.get('speaker_a', 'speaker')}_{args.sample}"

    config = build_config()
    system, episode_store, semantic_store, orchestrator = build_system(config)

    print(f"Building Nemori memory for sample {args.sample} (user {user_id})")
    tracker.reset("build")
    t0 = time.perf_counter()
    n_sessions, n_flush_episodes = await ingest(
        system, conversation, user_id, max_sessions=args.max_sessions)
    build_time = time.perf_counter() - t0
    build_snap = tracker.snapshot("build")

    episodes = await episode_store.list_by_user(user_id, config.agent_id,
                                                limit=1000000)
    semantics = await semantic_store.list_by_user(user_id, config.agent_id)
    stats = orchestrator.stats
    print(f"Build done in {build_time:.1f}s: {len(episodes)} episodes, "
          f"{len(semantics)} semantic memories, "
          f"{stats.total_requests} LLM calls, {stats.total_tokens} tokens")

    build_stats = {
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 2),
        "num_memories": len(episodes) + len(semantics),
        "build_calls": build_snap["calls"],
        "build_tokens_in": build_snap["tokens_in"],
        "build_tokens_out": build_snap["tokens_out"],
        "build_llm_time_s": build_snap["llm_time_s"],
        "notes": (
            f"nemori (arXiv:2508.03341) @ main; llm={config.llm_model} via "
            f"{config.llm_base_url}; embed={EMBED_MODEL} local ({EMBED_DIM}d); "
            f"storage=in-memory stores + qdrant-client :memory: (no "
            f"PostgreSQL/Qdrant servers; no third_party patches); "
            f"sessions={n_sessions}, episodes={len(episodes)}, "
            f"semantic={len(semantics)}; search=vector, "
            f"top{TOP_K_EPISODES} episodes + top{TOP_K_SEMANTIC} semantic; "
            f"llm_calls={stats.total_requests}, "
            f"llm_tokens={stats.total_tokens}, "
            f"llm_errors={stats.total_errors}, "
            f"calls_by_phase={stats.requests_by_phase}"
        ),
    }

    print("Retrieving memories per question")
    records = await answer_questions(
        system, item.get("qa", []), user_id, args.sample,
        limit=args.questions_limit)

    out = [build_stats] + records
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} question records -> {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Nemori adapter: LoCoMo -> unified retrieval records")
    parser.add_argument("--sample", type=int, default=0,
                        help="LoCoMo conversation index")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Dev: only ingest first N sessions")
    parser.add_argument("--questions-limit", type=int, default=None,
                        help="Dev: only process first N questions")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
