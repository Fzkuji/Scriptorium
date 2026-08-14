#!/usr/bin/env python3
"""Hindsight baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_hindsight.py" --sample 0 --output results/hindsight_s0.json
    python3 "src/adapters/run_hindsight.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_hindsight.json

Retrieval only: builds Hindsight memories from LoCoMo sessions and dumps
top-20 recalled memories per question. Answering and judging are done by
src/evaluation/evaluate.py.

Feasibility / model support findings (hindsight repo @ 6a479dd, v0.8.4)
-----------------------------------------------------------------------
- Hindsight is a full service system (FastAPI + PostgreSQL/pgvector), BUT the
  core engine (hindsight_api.engine.MemoryEngine) is a plain Python class the
  official benchmark runner itself uses in-process
  (hindsight-dev/benchmarks/common/benchmark_runner.py:create_memory_engine).
  No Docker / server needed.
- Storage: db_url="pg0://<name>" uses pg0-embedded (pip package by the same
  vendor) which manages an embedded PostgreSQL with pgvector automatically
  (binaries auto-downloaded to ~/.pg0 on first run). No external Postgres.
- LLM: provider "openai" goes through OpenAICompatibleLLM and accepts a
  custom base_url -> BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY (Aliyun
  compatible-mode) work. Structured output uses the soft json_object path by
  default (schema embedded in prompt, HINDSIGHT_API_LLM_STRICT_SCHEMA=false),
  which DashScope supports.
- Embedder: default provider is "local" (sentence-transformers). We pass
  LocalSTEmbeddings("sentence-transformers/all-MiniLM-L6-v2") (384d,
  dimension auto-detected). Reranker default is also local
  (cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers).

Protocol fidelity (mirrors the official LoCoMo benchmark defaults)
------------------------------------------------------------------
- Ingestion: one retain_batch_async call; each session is one item with
  content=json.dumps(session_turns), context="Conversation between A and B
  (session_K of ...)", event_date=parsed session date, document_id -- exactly
  what LoComoDataset.prepare_sessions_for_ingestion builds.
- No consolidation wait / no worker poller: official locomo_benchmark.py
  default is wait_consolidation=False.
- Recall: recall_async(query, budget=HIGH (thinking_budget=500 default),
  max_tokens=4096, no fact_type filter) as in benchmark_runner.answer_question.
  We keep the top-20 returned facts ({"text","date"}); the official answer
  prompt additionally sees entities/chunks, which our unified contract drops
  for all baselines alike.
- Memory date = MemoryFact.occurred_start (falls back to mentioned_at).

Patches to third_party / shims
------------------------------
None. No file under third_party/ was modified. hindsight-api-slim[embedded-db]
is pip-installed from the local clone into a dedicated venv
(.venv-hindsight, created with --system-site-packages so the existing local
torch/sentence-transformers are reused). The only shim is the re-exec below:
when invoked with the system python3, the script re-execs itself with
.venv-hindsight/bin/python so the CLI contract (plain `python3 src/adapters/
run_hindsight.py`) still holds.

Setup (one-time, already done):
    python3 -m venv --system-site-packages .venv-hindsight
    .venv-hindsight/bin/pip install "./third_party/hindsight/hindsight-api-slim[embedded-db]"
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- venv re-exec shim (see docstring) ----
# NB: compare sys.prefix (venv root), not sys.executable — the venv python is
# a symlink to the base interpreter, so realpath(executable) is identical.
_VENV_DIR = os.path.join(PROJECT_ROOT, ".venv-hindsight")
_VENV_PY = os.path.join(_VENV_DIR, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

sys.path.insert(0, PROJECT_ROOT)

from _usage_tracker import tracker  # noqa: E402
tracker.install()

from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

# Config env must be set before hindsight_api reads it (get_config caches).
os.environ.setdefault("HINDSIGHT_API_LLM_PROVIDER", "openai")
os.environ.setdefault("HINDSIGHT_API_LLM_MODEL", BUILDER_MODEL)
os.environ.setdefault("HINDSIGHT_API_LLM_API_KEY", BUILDER_KEY)
os.environ.setdefault("HINDSIGHT_API_LLM_BASE_URL", BUILDER_BASE)
os.environ.setdefault("HINDSIGHT_API_SKIP_LLM_VERIFICATION", "true")
os.environ.setdefault("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

from hindsight_api.engine import MemoryEngine, LocalSTEmbeddings  # noqa: E402
from hindsight_api.engine.memory_engine import Budget  # noqa: E402
from hindsight_api.models import RequestContext  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MODEL = BUILDER_MODEL
TOP_K = 20
PG0_URL = "pg0://hindsight-locomo"  # one shared embedded-postgres instance


def parse_locomo_date(date_str):
    """LoCoMo date '1:56 pm on 8 May, 2023' -> aware datetime (UTC).

    Same format string as the official LoComoDataset._parse_date, plus an
    abbreviated-month fallback.
    """
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def fact_date(fact):
    """MemoryFact -> ISO date string (occurred_start, else mentioned_at)."""
    for attr in ("occurred_start", "mentioned_at"):
        val = getattr(fact, attr, None)
        if isinstance(val, datetime):
            return val.date().isoformat()
        if val:
            return str(val)
    return None


async def build_and_retrieve(args):
    with open(DATA_PATH) as f:
        data = json.load(f)
    sample = data[args.sample]
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Person A")
    speaker_b = conv.get("speaker_b", "Person B")
    sample_id = sample.get("sample_id", f"locomo_{args.sample}")

    session_ids = sorted(
        int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k))
    )
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[hindsight] sample={args.sample} sessions={len(session_ids)} model={MODEL}")

    engine = MemoryEngine(
        db_url=PG0_URL,
        memory_llm_provider="openai",
        memory_llm_api_key=BUILDER_KEY,
        memory_llm_model=MODEL,
        memory_llm_base_url=BUILDER_BASE,
        embeddings=LocalSTEmbeddings("sentence-transformers/all-MiniLM-L6-v2"),
        skip_llm_verification=True,
    )
    await engine.initialize()
    bank_id = f"locomo_s{args.sample}_{int(time.time())}"  # fresh bank per run
    ctx = RequestContext()

    # ---- Phase 1: build (one batch retain over all sessions, official style) ----
    contents = []
    for si in session_ids:
        turns = conv.get(f"session_{si}")
        if not isinstance(turns, list) or not turns:
            continue
        event_date = parse_locomo_date(conv.get(f"session_{si}_date_time", ""))
        item = {
            "content": json.dumps(turns),
            "context": f"Conversation between {speaker_a} and {speaker_b} (session_{si} of {sample_id})",
            "document_id": f"{sample_id}_session_{si}",
        }
        if event_date:
            item["event_date"] = event_date
        contents.append(item)

    tracker.reset("build")
    t0 = time.time()
    num_memories = 0
    build_notes_extra = ""
    try:
        unit_ids = await engine.retain_batch_async(bank_id=bank_id, contents=contents, request_context=ctx)
        num_memories = sum(len(u) for u in unit_ids)
    except Exception as e:
        # Fallback: per-session retain so one bad session doesn't kill the build.
        print(f"  batch retain failed ({e}); falling back to per-session retain")
        failed = []
        for item in contents:
            try:
                ids = await engine.retain_batch_async(bank_id=bank_id, contents=[item], request_context=ctx)
                num_memories += sum(len(u) for u in ids)
            except Exception as e2:
                failed.append(item["document_id"])
                print(f"  {item['document_id']} FAILED: {e2}")
        build_notes_extra = f", per-session fallback, failed={failed}"
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")
    print(f"[hindsight] build done: {build_time:.0f}s, {num_memories} memory units, bank={bank_id}")

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
            f"hindsight-api-slim 0.8.4 (repo @6a479dd) in-process MemoryEngine, "
            f"llm={MODEL} via Aliyun openai-compatible (soft json_object structured output), "
            f"embedder=all-MiniLM-L6-v2 (local, 384d), reranker=local ms-marco-MiniLM-L-6-v2, "
            f"storage=pg0 embedded postgres, sessions={len(session_ids)}, "
            f"recall budget=HIGH max_tokens=4096 (official locomo defaults, "
            f"wait_consolidation=False), bank={bank_id}{build_notes_extra}"
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
            res = await engine.recall_async(
                bank_id,
                question,
                budget=Budget.HIGH,
                max_tokens=4096,
                request_context=ctx,
            )
            for fact in (res.results or [])[:TOP_K]:
                memories.append({"text": fact.text, "date": fact_date(fact)})
        except Exception as e:
            print(f"  q{qi} recall error: {e}")
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

    await engine.close()
    return records


def main():
    parser = argparse.ArgumentParser(description="Hindsight LoCoMo adapter (retrieval only)")
    parser.add_argument("--sample", type=int, default=0, help="LoCoMo conversation index")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Dev: only ingest first N sessions")
    parser.add_argument("--questions-limit", type=int, default=None,
                        help="Dev: only process first N questions")
    args = parser.parse_args()

    records = asyncio.run(build_and_retrieve(args))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[hindsight] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
