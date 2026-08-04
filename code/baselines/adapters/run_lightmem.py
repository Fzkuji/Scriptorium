#!/usr/bin/env python3
"""LightMem (zjunlp/LightMem, ICLR 2026) adapter for the unified LoCoMo protocol.

Build memories with LightMem's own pipeline (topic segmentation -> LLM fact
extraction -> Qdrant embedding index -> offline update), then retrieve top-k
per question. NO answer generation, NO judging here (src/evaluation/evaluate.py
does both).

Usage (from project root):
    python3 "src/adapters/run_lightmem.py" --sample 0 --output out.json \
        [--max-sessions N] [--questions-limit N] [--k 20] [--qdrant-dir DIR]

Model-support findings (2026-07 survey of third_party/lightmem):
  * LLM config: no litellm; per-provider manager factory
    (src/lightmem/factory/memory_manager/factory.py) with providers
    "openai", "deepseek", "ollama", "vllm", "vllm_offline", "transformers".
    The "openai" manager is a bare OpenAI SDK client that honors
    configs.openai_base_url + configs.api_key, so ANY OpenAI-compatible
    endpoint works. deepseek-v4-flash via the Aliyun compatible-mode URL is
    directly usable (verified; it also needs response_format
    {"type":"json_object"}, which the Aliyun endpoint supports). README even
    advertises native deepseek-v4-flash support (2026-04-24 news).
  * Embedder: TextEmbedderFactory supports "huggingface" (local
    sentence-transformers; default all-MiniLM-L6-v2) and "openai". Local
    sentence-transformers with 384 dims is a first-class official config
    (their LoCoMo scripts use exactly huggingface/384). No OpenAI embedding
    required. Dimension is free (must match qdrant embedding_model_dims).
  * Other heavy deps of the OFFICIAL LoCoMo script that we bypass:
      - llmlingua-2 pre-compressor (llmlingua pip pkg + GPU model): we set
        pre_compress=False. Compression only shortens text; skipping it keeps
        the method semantics (segment -> extract -> index -> offline update).
      - llmlingua-2 BERT as topic segmenter: the segmenter only uses BERT
        self-attentions to propose topic cuts
        (factory/topic_segmenter/llmlingua_2.py). We substitute the locally
        cached "bert-base-uncased" (same 12-layer BERT family, layers 8-11,
        attn_implementation=eager, CPU). Functionally identical mechanism.
  * Declared model list: model_name_context_windows in
    factory/memory_manager/openai.py mentions gpt-4o-mini / qwen3-30b /
    glm-4.6 with a DEFAULT=128k fallback; README quickstart uses gpt-4o-mini,
    deepseek-v4-flash/pro, and local Ollama/vLLM/Transformers.

Patches to third_party: NONE (repo untouched). Two in-process workarounds:
  1. `LightMemory.compressor = None` class attribute is set before
     instantiation: with pre_compress=False, LightMemory.__init__
     (src/lightmem/memory/lightmem.py line 169) still reads self.compressor
     when topic_segment=True and would raise AttributeError. The class
     attribute satisfies the lookup; the segmenter is built with shared=False
     so the compressor is never used.
  2. OPENROUTER_API_KEY is popped from the environment because OpenaiManager
     silently reroutes every call to OpenRouter when that variable exists.

Ingestion mirrors experiments/locomo/add_locomo.py: one add_memory() call per
dialog turn ({user text (+blip caption), empty assistant msg}, session
timestamp), force_segment/force_extract on the final turn, then
construct_update_queue_all_entries() + offline_update_all_entries(0.9)
(LightMem's offline "sleep-time" update). Retrieval = LightMemory.retrieve()'s
exact mechanism (embed query -> qdrant cosine top-k) called through
text_embedder/embedding_retriever directly so we keep text and date as
separate fields.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "lightmem" / "src"))

from _usage_tracker import tracker  # noqa: E402
tracker.install()

from evaluation.llm_clients import BUILDER_MODEL, BUILDER_BASE, BUILDER_KEY  # noqa: E402

# OpenaiManager reroutes to OpenRouter if this env var exists (see docstring).
os.environ.pop("OPENROUTER_API_KEY", None)

from lightmem.memory.lightmem import LightMemory  # noqa: E402

# Workaround #1 (see docstring): survive pre_compress=False + topic_segment=True.
LightMemory.compressor = None

LOCOMO_PATH = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
SEGMENTER_BERT = "bert-base-uncased"  # official: llmlingua-2 BERT (see docstring)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LoCoMo extraction prompt from the official reproduction script.
_prompts_file = ROOT / "third_party" / "lightmem" / "experiments" / "locomo" / "prompts.py"


def load_locomo_prompt():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lightmem_locomo_prompts", _prompts_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.METADATA_GENERATE_PROMPT_locomo


def parse_locomo_timestamp(ts):
    """'1:56 pm on 8 May, 2023' -> '2023-05-08 13:56:00' (fromisoformat-safe)."""
    ts = ts.strip().strip("()")
    for fmt in ("%I:%M %p on %d %B, %Y", "%H:%M on %d %B, %Y"):
        try:
            return datetime.strptime(ts, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise ValueError(f"Unparseable LoCoMo session timestamp: {ts!r}")


def build_lightmem(collection_name, qdrant_dir):
    config = {
        "pre_compress": False,           # official uses llmlingua-2; skipped (docstring)
        "topic_segment": True,
        "precomp_topic_shared": False,
        "topic_segmenter": {
            "model_name": "llmlingua-2",  # factory key; actual BERT set below
            "configs": {
                "model_name": SEGMENTER_BERT,
                "buffer_len": 512,
                "model_config": {"attn_implementation": "eager"},
            },
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",       # bare OpenAI SDK, custom base_url OK
            "configs": {
                "model": BUILDER_MODEL,
                "api_key": BUILDER_KEY,
                "openai_base_url": BUILDER_BASE,
                "max_tokens": 8000,
                "temperature": 0.1,
            },
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",  # local sentence-transformers
            "configs": {
                "model": EMBED_MODEL,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cpu"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": os.path.join(qdrant_dir, collection_name),
                "on_disk": True,
            },
        },
        "update": "offline",
        "extraction_mode": "flat",
    }
    return LightMemory.from_config(config)


def extract_sessions(conversation):
    """conversation dict -> ordered [(iso_ts, [turns])]."""
    nums = sorted(
        int(k.split("_")[1])
        for k in conversation
        if k.startswith("session_") and not k.endswith("_date_time") and k.split("_")[1].isdigit()
    )
    sessions = []
    for n in nums:
        turns = conversation.get(f"session_{n}")
        if not turns:
            continue
        ts = parse_locomo_timestamp(conversation.get(f"session_{n}_date_time", ""))
        sessions.append((ts, turns))
    return sessions


def ingest(lm, sessions, prompt):
    """One add_memory per dialog turn, mirroring add_locomo.py."""
    n_sessions = len(sessions)
    for s_idx, (ts, turns) in enumerate(sessions):
        for t_idx, turn in enumerate(turns):
            content = turn["text"]
            if turn.get("blip_caption"):
                content = f"{content} (image description: {turn['blip_caption']})"
            speaker = turn["speaker"]
            messages = [
                {"role": "user", "content": content, "speaker_id": speaker,
                 "speaker_name": speaker, "time_stamp": ts},
                {"role": "assistant", "content": "", "speaker_id": speaker,
                 "speaker_name": speaker, "time_stamp": ts},
            ]
            is_last = (s_idx == n_sessions - 1) and (t_idx == len(turns) - 1)
            lm.add_memory(
                messages=messages,
                METADATA_GENERATE_PROMPT=prompt,
                force_segment=is_last,
                force_extract=is_last,
            )
        print(f"[build] session {s_idx + 1}/{n_sessions} ingested ({len(turns)} turns)",
              flush=True)


def search(lm, question, k):
    """LightMemory.retrieve()'s mechanism, but keeping structured payloads."""
    qvec = lm.text_embedder.embed(question)
    hits = lm.embedding_retriever.search(query_vector=qvec, limit=k, return_full=True)
    memories = []
    for h in hits:
        p = h.get("payload", {})
        memories.append({"text": p.get("memory", ""), "date": p.get("time_stamp", "")})
    return memories


def main():
    ap = argparse.ArgumentParser(description="LightMem adapter (unified protocol)")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--questions-limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--qdrant-dir", type=str, default=None,
                    help="Persist qdrant store here (default: fresh temp dir)")
    args = ap.parse_args()

    data = json.loads(LOCOMO_PATH.read_text())
    conv = data[args.sample]
    sessions = extract_sessions(conv["conversation"])
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    qa_list = conv["qa"]
    if args.questions_limit:
        qa_list = qa_list[: args.questions_limit]

    qdrant_dir = args.qdrant_dir or tempfile.mkdtemp(prefix="lightmem_qdrant_")
    collection = f"lightmem_s{args.sample}"
    prompt = load_locomo_prompt()

    print(f"[build] sample={args.sample} sessions={len(sessions)} "
          f"questions={len(qa_list)} qdrant={qdrant_dir}", flush=True)

    tracker.reset("build")
    t0 = time.time()
    lm = build_lightmem(collection, qdrant_dir)
    ingest(lm, sessions, prompt)
    # LightMem offline ("sleep-time") update phase, same params as add_locomo.py.
    lm.construct_update_queue_all_entries()
    lm.offline_update_all_entries(score_threshold=0.9)
    build_time = time.time() - t0
    build_snap = tracker.snapshot("build")

    entries = lm.embedding_retriever.get_all()
    num_memories = len(entries)
    tok = lm.get_token_statistics()
    print(f"[build] done in {build_time:.1f}s, {num_memories} memories, "
          f"{tok['summary']['total_llm_calls']} LLM calls", flush=True)

    records = [{
        "question_id": "_build_stats",
        "build_time_s": round(build_time, 2),
        "num_memories": num_memories,
        "build_calls": build_snap["calls"],
        "build_tokens_in": build_snap["tokens_in"],
        "build_tokens_out": build_snap["tokens_out"],
        "build_llm_time_s": build_snap["llm_time_s"],
        "notes": (
            f"LightMem flat mode, builder={BUILDER_MODEL}, endpoint={BUILDER_BASE}, "
            f"json_object mode, embedder={EMBED_MODEL} (384d, local ST), "
            f"segmenter={SEGMENTER_BERT} (sub for llmlingua-2 BERT), "
            f"pre_compress=off, offline update(0.9); "
            f"llm_calls={tok['summary']['total_llm_calls']}, "
            f"llm_tokens={tok['summary']['total_llm_tokens']}, "
            f"sessions={len(sessions)}"
        ),
    }]

    for idx, qa in enumerate(qa_list):
        question = qa["question"]
        gold = qa.get("answer")
        if gold is None:
            gold = qa.get("adversarial_answer", "")
        tracker.reset("q")
        t0 = time.time()
        memories = search(lm, question, args.k)
        latency = time.time() - t0
        q_snap = tracker.snapshot("q")
        records.append({
            "question_id": f"s{args.sample}_q{idx}",
            "question": question,
            "gold": str(gold),
            "category": qa.get("category"),
            "memories": memories,
            "retrieval": {"latency_s": round(latency, 4), "k": args.k,
                          "calls": q_snap["calls"],
                          "tokens_in": q_snap["tokens_in"],
                          "tokens_out": q_snap["tokens_out"]},
        })
        print(f"[qa] {idx + 1}/{len(qa_list)} retrieved {len(memories)} "
              f"in {latency:.3f}s", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"[done] wrote {len(records) - 1} QA records -> {out}", flush=True)


if __name__ == "__main__":
    main()
