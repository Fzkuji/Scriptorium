"""Naive lower-bound baseline adapter for the unified LoCoMo evaluation protocol.

Two methods, selected with --method:

1. full_context : no retrieval at all. The entire conversation (all sessions,
   chronological, each session prefixed with its date header, one
   "speaker: text" line per turn) is emitted as a single huge memory
   {"text": <full transcript>, "date": ""}. If the transcript exceeds the
   answerer context budget (~50K tokens; we cap at 160K characters) it is
   truncated to the MOST RECENT content and the truncation is recorded in
   _build_stats.notes.

2. bm25 : lexical-retrieval lower bound. Within each session, every 3
   consecutive turns form one chunk (chunk text carries the session date
   header); chunks are indexed with rank_bm25.BM25Okapi (simple lowercase
   alphanumeric tokenization) and the top-20 chunks are returned per question.

Model-support survey (this baseline is self-built, no third_party framework):
- LLM configuration: NONE. Neither method calls an LLM to build memory
  (full_context is pure concatenation, bm25 is pure lexical indexing), so
  there is no provider abstraction / litellm / OpenAI SDK involved and
  build cost is zero. deepseek-v4-flash compatibility is therefore trivially
  a non-issue at build time; the shared answerer/judge in
  src/evaluation/evaluate.py handle all LLM calls downstream.
- Embedder: NONE required. bm25 is lexical (rank_bm25, pip install
  rank-bm25); no OpenAI embedding, no sentence-transformers, no dimension
  constraints.
- Officially supported models: N/A (no third-party memory framework).

Patches to third_party: none (no third-party code is used).

Output record schema (per QA):
  {"question_id": "s{sample}_q{idx}", "question": ..., "gold": ...,
   "category": ..., "memories": [{"text": ..., "date": ...}],
   "retrieval": {"latency_s": ..., "k": 20}}
plus a "_build_stats" head record:
  {"question_id": "_build_stats", "build_time_s", "num_memories", "notes"}

The adapter never answers questions and never judges — that is done by
src/evaluation/evaluate.py.

Usage (from project root):
  python3 src/adapters/run_naive.py --sample 0 --method full_context \
      --output results/naive_full_s0.json
  python3 src/adapters/run_naive.py --sample 0 --method bm25 \
      --output results/naive_bm25_s0.json
"""

from _usage_tracker import tracker
tracker.install()

import argparse
import json
import re
import time

LOCOMO_PATH = "benchmarks/locomo/data/locomo10.json"
TOP_K = 20
CHUNK_TURNS = 3
# ~50K-token answerer budget; assume >=3.2 chars/token for English dialogue.
FULL_CONTEXT_CHAR_CAP = 160_000


def load_conversation(sample_idx, max_sessions=None):
    """Return (sessions, qa_list). sessions = [(date_str, [turn, ...]), ...]."""
    with open(LOCOMO_PATH, "r") as f:
        data = json.load(f)
    item = data[sample_idx]
    conv = item["conversation"]
    idxs = sorted(
        int(m.group(1))
        for k in conv
        if (m := re.fullmatch(r"session_(\d+)", k)) and isinstance(conv[k], list)
    )
    if max_sessions is not None:
        idxs = idxs[:max_sessions]
    sessions = [(conv.get(f"session_{i}_date_time", ""), conv[f"session_{i}"]) for i in idxs]
    return sessions, item["qa"]


def turn_line(turn):
    text = (turn.get("text") or "").strip()
    cap = (turn.get("blip_caption") or "").strip()
    if cap:
        text = (text + " " if text else "") + f"[shared image: {cap}]"
    return f"{turn.get('speaker', '?')}: {text}"


def build_full_context(sessions):
    parts = []
    for date, turns in sessions:
        lines = [f"=== Session on {date} ==="]
        lines.extend(turn_line(t) for t in turns)
        parts.append("\n".join(lines))
    full = "\n\n".join(parts)
    truncated = False
    if len(full) > FULL_CONTEXT_CHAR_CAP:
        truncated = True
        full = full[-FULL_CONTEXT_CHAR_CAP:]
        # start at a clean line boundary
        nl = full.find("\n")
        if 0 <= nl < 500:
            full = full[nl + 1:]
        full = "[...earlier conversation truncated...]\n" + full
    return full, truncated


def build_bm25_chunks(sessions):
    chunks = []  # [(text, date), ...]
    for date, turns in sessions:
        for i in range(0, len(turns), CHUNK_TURNS):
            group = turns[i:i + CHUNK_TURNS]
            body = "\n".join(turn_line(t) for t in group)
            chunks.append((f"(Session on {date})\n{body}", date))
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    ap = argparse.ArgumentParser(description="Naive lower-bound LoCoMo adapter")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--questions-limit", type=int, default=None)
    ap.add_argument("--method", required=True, choices=["full_context", "bm25"])
    args = ap.parse_args()

    sessions, qa_list = load_conversation(args.sample, args.max_sessions)
    if args.questions_limit is not None:
        qa_list = qa_list[:args.questions_limit]

    tracker.reset("build")
    t0 = time.time()
    notes = f"method={args.method}; sessions={len(sessions)}; no LLM build cost"
    if args.method == "full_context":
        full_text, truncated = build_full_context(sessions)
        num_memories = 1
        if truncated:
            notes += (f"; transcript exceeded {FULL_CONTEXT_CHAR_CAP} chars, "
                      "truncated to most recent content")
        retrieve = lambda q: [{"text": full_text, "date": ""}]  # noqa: E731
    else:  # bm25
        from rank_bm25 import BM25Okapi
        chunks = build_bm25_chunks(sessions)
        bm25 = BM25Okapi([tokenize(text) for text, _ in chunks])
        num_memories = len(chunks)

        def retrieve(q):
            scores = bm25.get_scores(tokenize(q))
            order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:TOP_K]
            return [{"text": chunks[i][0], "date": chunks[i][1]} for i in order]
    build_time = time.time() - t0
    snap = tracker.snapshot("build")

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 3),
        "num_memories": num_memories,
        "build_calls": snap["calls"],
        "build_tokens_in": snap["tokens_in"],
        "build_tokens_out": snap["tokens_out"],
        "build_llm_time_s": snap["llm_time_s"],
        "notes": notes,
    }]

    for idx, qa in enumerate(qa_list):
        gold = qa.get("answer", qa.get("adversarial_answer", ""))
        tracker.reset("q")
        t1 = time.time()
        memories = retrieve(qa["question"])
        latency = time.time() - t1
        qsnap = tracker.snapshot("q")
        records.append({
            "question_id": f"s{args.sample}_q{idx}",
            "question": qa["question"],
            "gold": str(gold),
            "category": qa.get("category"),
            "memories": memories,
            "retrieval": {"latency_s": round(latency, 4), "k": TOP_K,
                          "calls": qsnap["calls"],
                          "tokens_in": qsnap["tokens_in"],
                          "tokens_out": qsnap["tokens_out"]},
        })

    with open(args.output, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"[run_naive:{args.method}] wrote {len(records) - 1} QA records "
          f"(+_build_stats) to {args.output}; num_memories={num_memories}, "
          f"build_time_s={build_time:.3f}")


if __name__ == "__main__":
    main()
