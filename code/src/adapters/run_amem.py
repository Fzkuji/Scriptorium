"""A-Mem (AgenticMemory) adapter for the unified LoCoMo evaluation protocol.

Baseline repo: third_party/amem (WujiangXu/AgenticMemory — the A-Mem paper's
experiment repo; its ``AgenticMemorySystem`` lives in ``memory_layer.py``,
not ``memory_system.py`` like the agiresearch/A-mem library repo).

Pipeline (retrieval only — NO answering, NO judging here):
  1. Ingest one LoCoMo conversation turn-by-turn via
     ``AgenticMemorySystem.add_note(content, time=<session date>)``, exactly
     mirroring the official eval script third_party/amem/test_advanced.py:
     content = "Speaker <speaker>says : <text>", image turns get the
     "[Image: <blip_caption>]" prefix like the official load_dataset.py.
     Every add_note makes 2 LLM calls (note construction/analyze_content +
     link generation/evolution in process_memory) — this is slow by design.
  2. For each QA, retrieve top-20 notes with the baseline's own retriever
     (SimpleEmbeddingRetriever, cosine over sentence-transformer embeddings
     of "content:... context:... keywords:... tags:..." docs). This local
     repo has no ``search_agentic``; ``_search_top_k`` below is its exact
     equivalent (retriever.search(query, k) -> indices -> MemoryNote list),
     matching what search_agentic does in the agiresearch/A-mem repo.
     Note: the official test_advanced.py additionally rewrites the question
     into LLM-generated keywords before retrieval; the unified protocol
     retrieves with the raw question for cross-system comparability.
  3. ``date`` per memory = ``MemoryNote.timestamp`` (the session date_time
     string that was passed to add_note, e.g. "1:56 pm on 8 May, 2023").

Model support survey (A-Mem framework)
---------------------------------------
* LLM config: hand-rolled provider abstraction ``LLMController`` with
  backends "openai" | "ollama" | "sglang" (memory_layer.py). "openai" is
  the bare OpenAI SDK; "ollama" is routed through litellm
  (``LiteLLMController``); "sglang" is raw HTTP to an SGLang server. The
  robust variant (memory_layer_robust.py) adds vLLM and a LiteLLM backend.
  README declares gpt-4o-mini / vLLM Qwen2.5 / Ollama qwen2.5:3b etc.
* Arbitrary OpenAI-compatible base_url: YES. ``OpenAIController`` reads
  OPENAI_API_BASE (our pre-existing repo patch, memory_layer.py:37-43) and
  passes it as base_url. deepseek-v4-flash via the Aliyun compatible-mode
  endpoint verified working, INCLUDING the strict ``json_schema``
  response_format that A-Mem sends on every call (tested directly).
* Embedder: fully local sentence-transformers, default 'all-MiniLM-L6-v2'
  (384-d) in ``SimpleEmbeddingRetriever``. No OpenAI embedding anywhere in
  this repo; no dimension constraint (in-memory numpy cosine). So the
  unified protocol's local-embedding requirement is A-Mem's native setup.

Patches / shims (no third_party file was modified by this adapter)
------------------------------------------------------------------
* Pre-existing repo patch (kept): memory_layer.py lines 37-43 —
  OpenAIController falls back to the OPENAI_API_BASE env var for base_url.
* Runtime shim 1: ``memory_layer.re = re``. memory_layer.py line 385 calls
  ``re.sub`` but the module never imports ``re``; without this shim
  MemoryNote.analyze_content ALWAYS falls into its exception path and every
  note gets empty keywords/tags (confirmed in the earlier cached run:
  cached notes all have keywords=[]). Injecting the module attribute fixes
  the bug without editing the file.
* Runtime shim 2: wrap ``OpenAIController.get_completion`` with 3 retries +
  a schema-complete fallback JSON, because memory_layer.process_memory has
  no try/except around its LLM call — one transient API error would
  otherwise crash a build of hundreds of notes.

Usage (from project root):
    python3 src/adapters/run_amem.py --sample 0 --output results/amem/questions.json
    python3 src/adapters/run_amem.py --sample 0 --max-sessions 2 \
        --questions-limit 3 --output /tmp/adapter_smoke_amem.json
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

from _usage_tracker import tracker
tracker.install()

ROOT =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "amem"))

from src.evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY

# A-Mem's OpenAIController reads these env vars (base_url via our repo patch).
os.environ["OPENAI_API_KEY"] = BUILDER_KEY
os.environ["OPENAI_API_BASE"] = BUILDER_BASE

import memory_layer  # noqa: E402

# Shim 1: memory_layer.py uses re.sub (line 385) without importing re.
memory_layer.re = re

from memory_layer import AgenticMemorySystem  # noqa: E402

LOCOMO_PATH = os.path.join(ROOT, "benchmarks", "locomo", "data", "locomo10.json")
BUILD_MODEL = BUILDER_MODEL
EMBED_MODEL = "all-MiniLM-L6-v2"  # local sentence-transformers, 384-d

# Satisfies both response schemas used by memory_layer (analyze_content and
# process_memory) so a total LLM failure degrades gracefully.
_FALLBACK_JSON = json.dumps({
    "keywords": [], "context": "General", "tags": [],
    "should_evolve": False, "actions": [], "suggested_connections": [],
    "tags_to_update": [], "new_context_neighborhood": [],
    "new_tags_neighborhood": [],
})


def _make_robust(llm, retries=3, wait=3):
    """Shim 2: retry wrapper around OpenAIController.get_completion."""
    orig = llm.get_completion

    def robust_get_completion(prompt, response_format, temperature=0.7):
        last_err = None
        for attempt in range(retries):
            try:
                return orig(prompt, response_format, temperature)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(wait * (attempt + 1))
        print(f"[amem-adapter] LLM call failed after {retries} retries, "
              f"using fallback: {last_err}", file=sys.stderr)
        return _FALLBACK_JSON

    llm.get_completion = robust_get_completion


def _turn_text(turn):
    """Mirror third_party/amem/load_dataset.py image-turn handling."""
    text = turn.get("text", "")
    if "img_url" in turn and "blip_caption" in turn:
        caption = f"[Image: {turn['blip_caption']}]"
        text = f"{caption} {text}" if text else caption
    return text


def _session_keys(conversation, max_sessions=None):
    keys = [k for k in conversation
            if re.fullmatch(r"session_\d+", k) and isinstance(conversation[k], list)]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    if max_sessions is not None:
        keys = keys[:max_sessions]
    return keys


def build_memory(conversation, max_sessions=None):
    ms = AgenticMemorySystem(
        model_name=EMBED_MODEL,
        llm_backend="openai",
        llm_model=BUILD_MODEL,
    )
    _make_robust(ms.llm_controller.llm)

    t0 = time.time()
    n_turns = 0
    keys = _session_keys(conversation, max_sessions)
    for key in keys:
        date_time = conversation.get(f"{key}_date_time")
        turns = conversation[key]
        print(f"[amem-adapter] ingesting {key} ({len(turns)} turns, {date_time})",
              file=sys.stderr)
        for turn in turns:
            text = _turn_text(turn)
            if not text:
                continue
            # Exact official ingest format (test_advanced.py, incl. glued "says :").
            content = "Speaker " + turn["speaker"] + "says : " + text
            sink = io.StringIO()  # memory_layer prints full prompts; silence them
            with contextlib.redirect_stdout(sink):
                ms.add_note(content, time=date_time)
            n_turns += 1
    build_time = time.time() - t0
    print(f"[amem-adapter] build done: {n_turns} turns -> {len(ms.memories)} notes "
          f"in {build_time:.1f}s", file=sys.stderr)
    return ms, build_time, keys


def _search_top_k(ms, query, k=20):
    """Equivalent of agiresearch/A-mem's search_agentic for this repo:
    retriever.search -> corpus indices -> MemoryNote list (insertion-aligned)."""
    if not ms.memories:
        return []
    indices = ms.retriever.search(query, k)
    notes = list(ms.memories.values())
    out = []
    for i in indices:
        note = notes[int(i)]
        out.append({"text": note.content, "date": note.timestamp})
    return out


def main():
    ap = argparse.ArgumentParser(description="A-Mem LoCoMo adapter")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--questions-limit", type=int, default=None)
    args = ap.parse_args()

    with open(LOCOMO_PATH) as f:
        data = json.load(f)
    sample = data[args.sample]
    conversation = sample["conversation"]
    qa_list = sample["qa"]
    if args.questions_limit is not None:
        qa_list = qa_list[:args.questions_limit]

    tracker.reset("build")
    ms, build_time, used_sessions = build_memory(conversation, args.max_sessions)
    build_snap = tracker.snapshot("build")

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 2),
        "num_memories": len(ms.memories),
        "build_calls": build_snap["calls"],
        "build_tokens_in": build_snap["tokens_in"],
        "build_tokens_out": build_snap["tokens_out"],
        "build_llm_time_s": build_snap["llm_time_s"],
        "notes": (f"A-Mem AgenticMemorySystem (third_party/amem/memory_layer.py); "
                  f"build LLM={BUILD_MODEL}, endpoint={BUILDER_BASE}; "
                  f"embedder={EMBED_MODEL} (local sentence-transformers, 384-d); "
                  f"per-turn add_note (note construction + evolution LLM calls); "
                  f"retrieval=SimpleEmbeddingRetriever cosine top-20 with raw "
                  f"question; sessions={used_sessions}"),
    }]

    for idx, qa in enumerate(qa_list):
        gold = qa.get("answer", qa.get("adversarial_answer"))
        tracker.reset("q")
        t0 = time.time()
        memories = _search_top_k(ms, qa["question"], k=20)
        latency = time.time() - t0
        q_snap = tracker.snapshot("q")
        records.append({
            "question_id": f"s{args.sample}_q{idx}",
            "question": qa["question"],
            "gold": str(gold),
            "category": qa["category"],
            "memories": memories,
            "retrieval": {"latency_s": round(latency, 4), "k": 20,
                          "calls": q_snap["calls"],
                          "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        if (idx + 1) % 20 == 0:
            print(f"[amem-adapter] retrieved {idx + 1}/{len(qa_list)}",
                  file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[amem-adapter] wrote {len(records) - 1} QA records + _build_stats "
          f"to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
