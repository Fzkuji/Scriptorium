"""MIRIX baseline adapter for the unified LoCoMo evaluation protocol.

Run from project root:
    python3 "src/adapters/run_mirix.py" --sample 0 --output results/mirix_s0.json
    python3 "src/adapters/run_mirix.py" --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_mirix.json

Retrieval only: builds MIRIX multi-agent memories from LoCoMo sessions and
dumps top-20 retrieved memories per question. Answering and judging are done
by src/evaluation/evaluate.py.

Architecture (why a subprocess server)
--------------------------------------
MIRIX's build/retrieve core is only exposed through its FastAPI REST server
(mirix/server/rest_api.py); the official evals (third_party/mirix/evals/)
drive it the same way via MirixClient -> http://127.0.0.1:8531. This adapter
spawns that server as a local subprocess (uvicorn, SQLite backend) and talks
plain HTTP with `requests`, so the main env needs no mirix imports. The server
runs from a dedicated venv (default: third_party/mirix-venv, override with
MIRIX_PY=<python>) because mirix pins heavy deps (llama-index-core, mcp,
opentelemetry, ...).

Model support findings (MIRIX @ github master, 2026-07)
-------------------------------------------------------
- LLM: llm_config accepts model_endpoint_type="openai" + custom model_endpoint
  + api_key (mirix/schemas/llm_config.py), so any OpenAI-compatible endpoint
  works -> BUILDER_MODEL/BUILDER_BASE/BUILDER_KEY (deepseek-v4-flash @ Aliyun).
  Function calling is required (memory agents write via tool calls).
- Embedder: NOT needed. Storage-side embeddings are gated by env
  BUILD_EMBEDDINGS_FOR_MEMORY (mirix/constants.py:261) which we set to false;
  retrieval (`/memory/retrieve/conversation`) extracts topics with the LLM and
  searches with BM25 (rest_api.py: search_method="bm25"), never touching the
  embedding endpoint. A dummy openai embedding_config is passed only to satisfy
  schema validation; it is never called.
- Database: server falls back to SQLite (~/.mirix/sqlite.db, aiosqlite) when
  no MIRIX_PG_URI/pg env is set (mirix/server/server.py:343). No Postgres, no
  Docker, no Redis (optional, degrades gracefully).

Ingestion / retrieval protocol (mirrors third_party/mirix/evals/)
-----------------------------------------------------------------
- One /memory/add_sync per session: the whole session as a single user message
  (plain string, as evals/mirix_memory_system.py does), prefixed with the
  LoCoMo timestamp + the same extraction instructions as evals/main_eval.py,
  occurred_at=<parsed session date> for episodic anchoring.
  Unlike the official eval we also append blip image captions to turns, for
  parity with our other adapters (run_mem0.py etc.).
- Request chaining=False (official-eval parity), but the server env gets
  CHAINING_FOR_MEMORY_UPDATE=true (mirix/constants.py:249, default false):
  inner memory agents' chaining is controlled ONLY by that env var, and with
  deepseek-v4-flash a non-chained memory agent spends its single round on
  conversation_search and never reaches *_memory_insert (verified: 0 episodic
  memories after 2 sessions). With inner chaining the agents continue after
  tool calls and insert normally. temperature=0.3 for run-to-run stability
  (MIRIX default is 0.7).
- Per question: POST /memory/retrieve/conversation (content in list form --
  string content would fail the server's has_content check and skip topic
  extraction), limit=20 per memory type. Results from the six memory types are
  round-robin interleaved (relevant-episodic, semantic, knowledge-vault,
  procedural, resource, core) then padded with recent-episodic, capped at 20.

Patches to third_party
----------------------
None. No file in third_party/mirix was modified. Config (agents list + system
prompts) is read from third_party/mirix/evals/{configs/0201c.yaml-equivalent,
prompts/0201a/} conventions; prompts are inlined client-side exactly like
MirixClient._load_system_prompts does.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _usage_tracker import tracker
tracker.install()

import argparse
import atexit
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "locomo", "data", "locomo10.json")
MIRIX_ROOT = os.path.join(PROJECT_ROOT, "third_party", "mirix")
MIRIX_PY = os.environ.get(
    "MIRIX_PY", os.path.join(PROJECT_ROOT, "third_party", "mirix-venv", "bin", "python"))
PROMPTS_DIR = os.path.join(MIRIX_ROOT, "evals", "prompts", "0201a")
TOP_K = 20
ADD_TIMEOUT = 1800
RETRIEVE_TIMEOUT = 300

# Same extraction instructions the official MIRIX LoCoMo eval prepends to each
# session chunk (third_party/mirix/evals/main_eval.py).
INSTRUCTIONS = """Instructions:

1. Carefully analyze all utterances from both speakers.
2. The conversation has a timestamp, but the events mentioned in the conversation may have different timestamps. You have to extract the exact date of the mentioned events. Remember that "mentioned at" is not the same as "occurred at" so this has to be noted in the memories.
3. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
4. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp.
5. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
6. You are supposed to extract the event/fact/semantic knowledge from the conversation. For example, if the conversation happens at 2023 and the conversation says that "John went to India last year", then you should save the fact that "John went to India in 2022". Similarly for all other kinds of memories.
7. Make sure to extract the facts about the characters, such as their name, age, gender, occupation, hometown, etc."""

AGENTS = [
    "core_memory_agent",
    "resource_memory_agent",
    "semantic_memory_agent",
    "episodic_memory_agent",
    "procedural_memory_agent",
    "knowledge_vault_memory_agent",
]


def parse_locomo_date(date_str):
    """'1:56 pm on 8 May, 2023' -> naive datetime (server stores naive)."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(port):
    if not os.path.exists(MIRIX_PY):
        sys.exit(f"MIRIX venv python not found: {MIRIX_PY}\n"
                 f"Create it: python3 -m venv third_party/mirix-venv && "
                 f"third_party/mirix-venv/bin/pip install -r <trimmed reqs>")
    env = dict(os.environ)
    # No embeddings at build time; retrieval is topic-LLM + BM25 (see docstring).
    env["BUILD_EMBEDDINGS_FOR_MEMORY"] = "false"
    # Let inner memory agents keep stepping after tool calls (search -> insert).
    env["CHAINING_FOR_MEMORY_UPDATE"] = "true"
    # Force the SQLite fallback: strip any Postgres config.
    for k in list(env):
        if k.upper().startswith(("MIRIX_PG", "PG")):
            env.pop(k)
    env["PYTHONUNBUFFERED"] = "1"
    diagnostics_dir = os.environ.get("LOCOMO_DIAGNOSTICS_DIR")
    if diagnostics_dir:
        os.makedirs(diagnostics_dir, exist_ok=True)
        log_path = os.path.join(diagnostics_dir, "mirix_server.log")
    else:
        log_path = f"/tmp/mirix_server_{port}.log"
    log_f = open(log_path, "w")
    # _mirix_server_shim.py = stock server + in-process patch that adds
    # extra_body {"enable_thinking": false} to LLM requests (Aliyun rejects
    # tool_choice=required in thinking mode). See the shim's docstring.
    shim = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_mirix_server_shim.py")
    proc = subprocess.Popen(
        [MIRIX_PY, shim, str(port)],
        cwd=MIRIX_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log_f.flush()
        log_f.close()
    atexit.register(_cleanup)

    base = f"http://127.0.0.1:{port}"
    for _ in range(180):
        if proc.poll() is not None:
            sys.exit(f"MIRIX server died on startup; see {log_path}")
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.ok:
                print(f"[mirix] server up on {base} (log: {log_path})")
                return proc, base
        except requests.RequestException:
            pass
        time.sleep(1)
    sys.exit(f"MIRIX server did not come up in 180s; see {log_path}")


class MirixHTTP:
    """Minimal stand-in for mirix.client.MirixClient (same REST protocol)."""

    def __init__(self, base, client_id, org_id):
        self.base = base
        self.headers = {"Content-Type": "application/json",
                        "X-Client-ID": client_id, "X-Org-ID": org_id}
        self.client_id = client_id
        self.org_id = org_id
        self.meta_agent_id = None

    def _post(self, path, payload, timeout=120):
        r = requests.post(self.base + path, json=payload, headers=self.headers,
                          timeout=timeout)
        r.raise_for_status()
        return r.json()

    def bootstrap(self, user_id):
        self._post("/organizations/create_or_get",
                   {"org_id": self.org_id, "name": self.org_id})
        self._post("/clients/create_or_get",
                   {"client_id": self.client_id, "name": self.client_id,
                    "org_id": self.org_id, "write_scope": "read_write",
                    "read_scopes": ["read_write"], "status": "active"})
        self._post("/users/create_or_get", {"user_id": user_id, "name": user_id})

    def initialize_meta_agent(self):
        prompts = {}
        for fn in os.listdir(PROMPTS_DIR):
            if fn.endswith(".txt"):
                with open(os.path.join(PROMPTS_DIR, fn), encoding="utf-8") as f:
                    prompts[fn[:-4]] = f.read()
        config = {
            "llm_config": {
                "model": BUILDER_MODEL,
                "model_endpoint_type": "openai",
                "model_endpoint": BUILDER_BASE,
                "api_key": BUILDER_KEY,
                "context_window": 128000,
                "temperature": 0.3,
            },
            # Never called: BUILD_EMBEDDINGS_FOR_MEMORY=false + BM25 retrieval.
            "embedding_config": {
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": BUILDER_BASE,
                "embedding_model": "text-embedding-3-small",
                "embedding_dim": 1536,
            },
            "meta_agent_config": {"agents": AGENTS, "system_prompts": prompts},
        }
        data = self._post("/agents/meta/initialize",
                          {"config": config, "update_agents": False})
        self.meta_agent_id = data["id"]
        return data

    def add_sync(self, chunk, occurred_at=None):
        payload = {
            "user_id": self.user_id,
            "meta_agent_id": self.meta_agent_id,
            # Plain-string content: /memory/add_sync prefixes "[USER]" itself;
            # list-form dict content would crash its normalizer.
            "messages": [{"role": "user", "content": chunk}],
            # Official-eval parity; inner-agent chaining comes from the env
            # var CHAINING_FOR_MEMORY_UPDATE instead (see docstring).
            "chaining": False,
            "filter_tags": {"kind": "conversation_session"},
        }
        if occurred_at:
            payload["occurred_at"] = occurred_at
        return self._post("/memory/add_sync", payload, timeout=ADD_TIMEOUT)

    def retrieve(self, question, limit=TOP_K):
        # List-form content is required: the server's has_content check only
        # recognizes [{"type": "text", "text": ...}] and otherwise skips topic
        # extraction entirely (rest_api.py retrieve_memory_with_conversation).
        payload = {
            "user_id": self.user_id,
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": question}]}],
            "limit": limit,
        }
        return self._post("/memory/retrieve/conversation", payload,
                          timeout=RETRIEVE_TIMEOUT)


def flatten_memories(resp, top_k=TOP_K):
    """Round-robin over memory types -> [{'text','date'}], capped at top_k."""
    mems = resp.get("memories", {}) or {}

    def ep_text(it):
        s, d = it.get("summary") or "", it.get("details") or ""
        return f"{s} — {d}".strip(" —") if d else s

    episodic = mems.get("episodic", {}) or {}
    relevant = [{"text": ep_text(it), "date": it.get("timestamp"), "_id": it.get("id")}
                for it in episodic.get("relevant", []) if ep_text(it)]
    recent = [{"text": ep_text(it), "date": it.get("timestamp"), "_id": it.get("id")}
              for it in episodic.get("recent", []) if ep_text(it)]

    def items_of(t, fmt):
        out = []
        for it in (mems.get(t, {}) or {}).get("items", []) or []:
            text = fmt(it)
            if text:
                out.append({"text": text, "date": None, "_id": it.get("id")})
        return out

    semantic = items_of("semantic", lambda it: ": ".join(
        x for x in [it.get("name"), it.get("details") or it.get("summary")] if x))
    vault = items_of("knowledge_vault", lambda it: it.get("caption") or "")
    procedural = items_of("procedural", lambda it: it.get("summary") or "")
    resource = items_of("resource", lambda it: ": ".join(
        x for x in [it.get("title"), it.get("summary")] if x))

    core = []
    for scope_data in ((mems.get("core", {}) or {}).get("scopes", {}) or {}).values():
        for it in scope_data.get("items", []) or []:
            if it.get("value"):
                core.append({"text": f"{it.get('label', '')}: {it['value']}".strip(": "),
                             "date": None, "_id": it.get("id")})

    pools = [relevant, semantic, vault, procedural, resource, core]
    out, seen = [], set()
    i = 0
    while len(out) < top_k and any(i < len(p) for p in pools):
        for p in pools:
            if i < len(p) and len(out) < top_k:
                it = p[i]
                key = it.get("_id") or it["text"]
                if key not in seen:
                    seen.add(key)
                    out.append({"text": it["text"], "date": it["date"]})
        i += 1
    for it in recent:  # pad with recent episodic
        if len(out) >= top_k:
            break
        key = it.get("_id") or it["text"]
        if key not in seen:
            seen.add(key)
            out.append({"text": it["text"], "date": it["date"]})
    return out


def total_memories(resp):
    """-> (total, {type: count}) from a retrieve response's total_count fields."""
    mems = resp.get("memories", {}) or {}
    per_type = {}
    for t in ("episodic", "semantic", "resource", "procedural", "knowledge_vault", "core"):
        per_type[t] = (mems.get(t, {}) or {}).get("total_count", 0) or 0
    return sum(per_type.values()), per_type


def session_chunk(session_no, turns, date_str):
    lines = [f"You have access to the conversation between two speakers. "
             f"The conversation is timestamped at {date_str}.\n", INSTRUCTIONS,
             f"Session {session_no} ({date_str})"]
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = (turn.get("speaker") or "").strip()
        text = (turn.get("text") or "").strip()
        blip = (turn.get("blip_caption") or "").strip()
        query = (turn.get("query") or "").strip()
        if query and blip:
            text = f"{text} [Sharing image - query: {query}. The image shows: {blip}]".strip()
        elif blip:
            text = f"{text} [Sharing image that shows: {blip}]".strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MIRIX LoCoMo adapter (retrieval only)")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--questions-limit", type=int, default=None)
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        data = json.load(f)
    sample = data[args.sample]
    conv = sample["conversation"]

    session_ids = sorted(
        int(m.group(1)) for k in conv if (m := re.match(r"^session_(\d+)$", k)))
    if args.max_sessions:
        session_ids = session_ids[: args.max_sessions]

    print(f"[mirix] sample={args.sample} sessions={len(session_ids)} model={BUILDER_MODEL}")

    port = free_port()
    proc, base = start_server(port)

    run_tag = uuid.uuid4().hex[:8]
    client = MirixHTTP(base, client_id=f"locomo-s{args.sample}-{run_tag}",
                       org_id="org-locomo")
    client.user_id = f"user-locomo-s{args.sample}-{run_tag}"
    client.bootstrap(client.user_id)
    client.initialize_meta_agent()
    print(f"[mirix] meta agent {client.meta_agent_id} "
          f"(client {client.client_id})")

    # ---- Phase 1: build memory (one add_sync per session) ----
    tracker.reset("build")
    t0 = time.time()
    failed_sessions = []
    for si in session_ids:
        turns = conv.get(f"session_{si}", [])
        date_str = conv.get(f"session_{si}_date_time", "") or ""
        chunk = session_chunk(si, turns if isinstance(turns, list) else [], date_str)
        parsed = parse_locomo_date(date_str)
        occurred_at = parsed.isoformat() if parsed else None
        for attempt in range(2):
            try:
                client.add_sync(chunk, occurred_at=occurred_at)
                break
            except Exception as e:
                if attempt == 1:
                    failed_sessions.append(si)
                    print(f"  session {si} FAILED after retry: {e}")
                else:
                    time.sleep(5)
        print(f"  session {si} ingested ({time.time()-t0:.0f}s elapsed)")
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    # Content-less probe: returns totals without an LLM topic-extraction call.
    try:
        probe = client.retrieve("", limit=1)
        num_memories, per_type = total_memories(probe)
    except Exception as e:
        print(f"  probe failed: {e}")
        num_memories, per_type = -1, {}
    print(f"[mirix] build done: {build_time:.0f}s, {num_memories} memories {per_type}")

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
            f"MIRIX (github master) via local REST server (SQLite backend, "
            f"BUILD_EMBEDDINGS_FOR_MEMORY=false, BM25+topic-LLM retrieval), "
            f"llm={BUILDER_MODEL} openai-compatible, agents={len(AGENTS)}, "
            f"prompts=evals/prompts/0201a, inner_chaining=env temp=0.3, "
            f"sessions={len(session_ids)}, failed_sessions={failed_sessions}, "
            f"per_type={per_type}"
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
            resp = client.retrieve(question, limit=TOP_K)
            memories = flatten_memories(resp, top_k=TOP_K)
        except Exception as e:
            print(f"  q{qi} retrieve error: {e}")
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
    print(f"[mirix] wrote {len(records)-1} question records -> {args.output}")


if __name__ == "__main__":
    main()
