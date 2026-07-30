"""EverMemOS (EverOS) baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_evermemos.py" --sample 0 --output results/evermemos_s0.json
    python3 "src/adapters/run_evermemos.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_evermemos.json

Retrieval only: builds EverOS episodic memories from LoCoMo sessions and dumps
top-20 retrieved memories per question. Answering and judging are done by
src/evaluation/evaluate.py.

What "EverMemOS" is here
------------------------
github.com/EverMind-AI/EverMemOS (the ~10k-star repo cited by the EverMemOS
paper, arXiv:2601.02163) was RENAMED to EverMind-AI/EverOS; the clone under
third_party/evermemos is that repo at main (v1.1.x). The original
enterprise EverMemOS stack (MongoDB/Elasticsearch/Milvus) is gone upstream;
current EverOS is the local-first successor: Markdown files as source of
truth + embedded SQLite (state/queue) + embedded LanceDB (vector/BM25).
No Docker, no external DB, no external service.

Architecture (why a subprocess server)
--------------------------------------
EverOS explicitly ships no in-process library mode (QUICKSTART.md: "EverOS
runs as a service"); ingestion runs through FastAPI lifespans (SQLite
migrations, LanceDB tables, cascade indexing daemon, OME strategy engine all
live in the server process). Same pattern as run_mirix.py: we spawn the
server as a local subprocess via src/adapters/_evermemos_server_shim.py
(which monkeypatches a LOCAL all-MiniLM-L6-v2 embedding provider in process
— see that file's docstring) and talk plain HTTP. The server runs from a
dedicated venv (default third_party/evermemos_venv, override with
EVERMEMOS_PY=<python>) because everos needs Python 3.12 + lancedb + everalgo.

Model support findings (everos @ main 0341f12, 2026-07)
-------------------------------------------------------
- LLM: [llm] settings are a plain OpenAI-protocol chat endpoint
  (everos/component/llm/openai_provider.py), configured here via
  EVEROS_LLM__MODEL/API_KEY/BASE_URL env -> BUILDER_MODEL/BUILDER_BASE/
  BUILDER_KEY (deepseek-v4-flash @ Aliyun compatible mode). Used for boundary
  detection + episode extraction + OME strategies (atomic facts, profile...).
- Embedder: upstream only ships an OpenAI-compatible HTTP embedding client;
  the shim injects local sentence-transformers all-MiniLM-L6-v2 (384d,
  zero-padded to the hardcoded LanceDB Vector(1024) — cosine-order
  preserving). EVEROS_EMBEDDING__* env get dummy values only to pass the
  settings guard in everos/service/search.py; they are never used.
- Rerank: NOT needed. user-owner HYBRID search (episode hierarchy path:
  episode hybrid recall + atomic-fact MaxSim + RRF + fact eviction) requires
  only the embedding provider (SearchManager._validate_components); the
  rerank provider is only for agent skills / AGENTIC method. No rerank
  endpoint is configured.

Ingestion / retrieval protocol
------------------------------
- mode=chat (EVEROS_MEMORIZE__MODE): user-memory pipeline only.
- One POST /api/v1/memory/add per LoCoMo session (all turns, speaker-prefixed
  "Name: text", blip image captions appended as in run_mem0.py), then
  POST /api/v1/memory/flush to force boundary extraction (OSS-only endpoint).
- Single synthetic owner sender_id="user" for every message: EverOS fans out
  one episode copy per distinct user-role sender (user_memory.py), so using
  the two real speakers would double every episode + OME fact extraction.
  Speaker attribution is preserved in the message text prefix instead.
- Timestamps: session date "1:56 pm on 8 May, 2023" -> epoch ms (UTC),
  +60s per turn, so episodes carry the true conversation dates.
- After ingest, poll GET /debug/drained (shim route) until OME idle and the
  cascade queue is drained, so LanceDB indexing is complete before search.
- Retrieval: POST /api/v1/memory/search {user_id:"user", method:"hybrid",
  top_k:20}. Memory text = episode narrative (`episode` field); when the
  hierarchy's fact-eviction returns nested atomic_facts, their contents are
  appended so the dumped memory keeps the evicted fact detail. date =
  episode timestamp (UTC date).

Patches to third_party
----------------------
None. No file in third_party/evermemos is modified; the local-embedder
monkeypatch and the /debug/drained route live in
src/adapters/_evermemos_server_shim.py (in-process only).
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _usage_tracker import tracker  # noqa: E402
tracker.install()

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import requests

# The server is on 127.0.0.1; macOS system proxies (e.g. Clash on :7890) must
# not intercept these calls — requests honors them by default and the proxy
# answers 502 Bad Gateway for some localhost requests.
HTTP = requests.Session()
HTTP.trust_env = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
SHIM = os.path.join(PROJECT_ROOT, "src", "adapters", "_evermemos_server_shim.py")
VENV_PY = os.environ.get(
    "EVERMEMOS_PY",
    os.path.join(PROJECT_ROOT, "third_party", "evermemos_venv", "bin", "python"),
)
MODEL = BUILDER_MODEL
TOP_K = 20
PORT = int(os.environ.get("EVERMEMOS_PORT", "8377"))
OWNER = "user"
DRAIN_TIMEOUT_S = 1800


def parse_locomo_date_ms(date_str):
    """'1:56 pm on 8 May, 2023' -> Unix epoch milliseconds (UTC)."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            continue
    return None


def turn_text(turn):
    """LoCoMo turn -> 'Speaker: text [image caption]' (same as run_mem0.py)."""
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
        return None
    return f"{speaker}: {text}"


def start_server(root, port):
    """`everos init` into root, then spawn the shim server subprocess."""
    subprocess.run(
        [os.path.join(os.path.dirname(VENV_PY), "everos"), "init", "--root", root],
        check=True, capture_output=True,
    )
    env = dict(os.environ)
    env.update({
        "EVEROS_ROOT": root,
        "EVEROS_LLM__MODEL": MODEL,
        "EVEROS_LLM__API_KEY": BUILDER_KEY,
        "EVEROS_LLM__BASE_URL": BUILDER_BASE,
        # Dummies: guard-passing only; the shim's local MiniLM provider is used.
        "EVEROS_EMBEDDING__MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "EVEROS_EMBEDDING__API_KEY": "local",
        "EVEROS_EMBEDDING__BASE_URL": "http://127.0.0.1:1/unused",
        "EVEROS_MEMORIZE__MODE": "chat",
        "TOKENIZERS_PARALLELISM": "false",
    })
    proc = subprocess.Popen(
        [VENV_PY, SHIM, str(port)],
        env=env,
        stdout=open(os.path.join(root, "server.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(
                f"everos server died at startup; see {root}/server.log")
        try:
            if HTTP.get(f"{base}/health", timeout=2).status_code == 200:
                return proc, base
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"everos server not healthy after 240s; see {root}/server.log")


def wait_drained(base, quiet_polls=2, poll_s=5.0):
    """Poll /debug/drained until OME idle + cascade queue empty, stable twice."""
    t0 = time.time()
    ok_streak = 0
    last = {}
    while time.time() - t0 < DRAIN_TIMEOUT_S:
        try:
            last = HTTP.get(f"{base}/debug/drained", timeout=30).json()
        except requests.RequestException:
            time.sleep(poll_s)
            continue
        idle = (last.get("ome_idle") and last.get("cascade_pending", 1) == 0
                and last.get("cascade_lag", 1) == 0)
        ok_streak = ok_streak + 1 if idle else 0
        if ok_streak >= quiet_polls:
            return last
        time.sleep(poll_s)
    print(f"  WARN: drain timeout after {DRAIN_TIMEOUT_S}s (state={last})")
    return last


def main():
    parser = argparse.ArgumentParser(description="EverMemOS/EverOS LoCoMo adapter (retrieval only)")
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

    root = tempfile.mkdtemp(prefix=f"everos_locomo_s{args.sample}_")
    print(f"[evermemos] sample={args.sample} sessions={len(session_ids)} "
          f"model={MODEL} root={root}")
    proc, base = start_server(root, PORT)
    print(f"[evermemos] server up at {base}")

    ok = False
    try:
        # ---- Phase 1: build memory (one add + flush per session) ----
        tracker.reset("build")
        t0 = time.time()
        failed_sessions = []
        for si in session_ids:
            turns = conv.get(f"session_{si}", [])
            date_str = conv.get(f"session_{si}_date_time", "")
            base_ms = parse_locomo_date_ms(date_str) or int(time.time() * 1000)
            messages = []
            for ti, turn in enumerate(turns if isinstance(turns, list) else []):
                if not isinstance(turn, dict):
                    continue
                text = turn_text(turn)
                if not text:
                    continue
                messages.append({
                    "sender_id": OWNER,
                    "role": "user",
                    "timestamp": base_ms + ti * 60_000,
                    "content": text,
                })
            if not messages:
                continue
            sid = f"locomo_s{args.sample}_session_{si}"
            try:
                r = HTTP.post(f"{base}/api/v1/memory/add", json={
                    "session_id": sid, "messages": messages}, timeout=600)
                r.raise_for_status()
                r = HTTP.post(f"{base}/api/v1/memory/flush", json={
                    "session_id": sid}, timeout=600)
                r.raise_for_status()
                status = r.json().get("data", {}).get("status")
            except requests.RequestException as e:
                failed_sessions.append(si)
                print(f"  session {si} FAILED: {e}")
                continue
            print(f"  session {si} ingested ({len(messages)} msgs, "
                  f"flush={status}, {time.time()-t0:.0f}s elapsed)")

        print("[evermemos] waiting for OME + cascade drain...")
        drain = wait_drained(base)
        build_time = time.time() - t0
        build_snap = tracker.snapshot("build")
        num_memories = drain.get("episodes", -1)
        print(f"[evermemos] build done: {build_time:.0f}s, "
              f"{num_memories} episodes / {drain.get('atomic_facts')} facts "
              f"(failed={drain.get('failed_retryable', 0) + drain.get('failed_permanent', 0)})")

        # ---- Phase 2: retrieval per question ----
        qas = sample["qa"]
        if args.questions_limit:
            qas = qas[: args.questions_limit]

        records = [{
            "question_id": "_build_stats",
            "build_time_s": round(build_time, 1),
            "build_calls": build_snap["calls"],
            "build_tokens_in": build_snap["tokens_in"],
            "build_tokens_out": build_snap["tokens_out"],
            "build_llm_time_s": build_snap["llm_time_s"],
            "num_memories": num_memories,
            "notes": (
                f"EverMemOS->EverOS main 0341f12, llm={MODEL}, endpoint={BUILDER_BASE}, "
                f"embedder=all-MiniLM-L6-v2 (local shim, "
                f"384d zero-padded to 1024), storage=md+sqlite+lancedb "
                f"(embedded), mode=chat, search=hybrid (episode hierarchy, "
                f"no rerank), atomic_facts={drain.get('atomic_facts')}, "
                f"cascade_failed={drain.get('failed_retryable', 0) + drain.get('failed_permanent', 0)}, "
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
                r = HTTP.post(f"{base}/api/v1/memory/search", json={
                    "user_id": OWNER,
                    "query": str(question),
                    "method": "hybrid",
                    "top_k": TOP_K,
                }, timeout=120)
                r.raise_for_status()
                episodes = r.json().get("data", {}).get("episodes", [])
                for ep in episodes:
                    text = ep.get("episode") or ep.get("summary") or ""
                    facts = [f.get("content", "") for f in ep.get("atomic_facts", [])]
                    facts = [f for f in facts if f and f not in text]
                    if facts:
                        text = text + " [Facts: " + " | ".join(facts) + "]"
                    ts = ep.get("timestamp")
                    date = str(ts)[:10] if ts else None
                    memories.append({"text": text, "date": date})
            except requests.RequestException as e:
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
        print(f"[evermemos] wrote {len(records)-1} question records -> {args.output}")
        ok = True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        diagnostics_dir = os.environ.get("LOCOMO_DIAGNOSTICS_DIR")
        server_log = os.path.join(root, "server.log")
        if diagnostics_dir and os.path.isfile(server_log):
            os.makedirs(diagnostics_dir, exist_ok=True)
            shutil.copy2(server_log, os.path.join(diagnostics_dir, "evermemos_server.log"))
        if ok:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"[evermemos] kept root for debugging: {root}")


if __name__ == "__main__":
    main()
