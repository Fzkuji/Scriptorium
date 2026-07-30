"""Quick clean-口径 test: N questions from a sample, no-original (raw masked +
'>' lines dropped), against a built memory dir. Answerer + judge configurable.

Usage:
  python3 scripts/quick_test.py <memory_dir> <sample_idx> <n> <answer_model> <answer_base> <answer_key> <or_key>
Answerer uses answer_model @ answer_base; judge = gpt-4o-mini @ OpenRouter.
"""
import sys
import os
import json
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEM_DIR = sys.argv[1]
SAMPLE = int(sys.argv[2])
N = int(sys.argv[3])
ANS_MODEL = sys.argv[4]
ANS_BASE = sys.argv[5]
ANS_KEY = sys.argv[6]
OR_KEY = sys.argv[7]

os.environ["NATIVEMEM_PROMPT"] = os.environ.get("NATIVEMEM_PROMPT", "v6")
os.environ["NATIVEMEM_RETRIEVAL"] = os.environ.get("NATIVEMEM_RETRIEVAL", "grep")
os.environ["BUILDER_MODEL"] = ANS_MODEL
os.environ["BUILDER_BASE"] = ANS_BASE
os.environ["ALIYUN_KEY"] = ANS_KEY

import importlib.util
spec = importlib.util.spec_from_file_location(
    "rn", os.path.join(os.path.dirname(__file__), "..", "src", "adapters", "run_nativemem.py"))
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)

import requests
from concurrent.futures import ThreadPoolExecutor

data = json.load(open("benchmarks/locomo/data/locomo10.json"))
qas = [q for q in data[SAMPLE]["qa"] if q.get("category") in (1, 2, 3, 4)][:N]

empty = [0]
leak = [0]


def retrieve(q):
    mems, _ = rn.collect_memories(q["question"], MEM_DIR)
    mems = [m for m in mems if not m["text"].strip().startswith(">")]
    if not mems:
        empty[0] += 1
    for m in mems:
        if "raw/" in m["text"]:
            leak[0] += 1
    return q, "\n".join(m["text"] for m in mems)


with ThreadPoolExecutor(max_workers=4) as ex:
    ret = list(ex.map(retrieve, qas))
print(f"{ANS_MODEL} no-original {len(qas)}q: empty={empty[0]} raw_leak={leak[0]}")


def post(url, payload, hdr=None):
    for _ in range(4):
        try:
            return requests.post(url, headers=hdr or {}, json=payload, timeout=90).json()
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return None


def qa(qc):
    q, ctx = qc
    gold = q.get("answer", q.get("adversarial_answer", ""))
    prompt = f"Memories:\n{ctx}\n\nQuestion: {q['question']}\nAnswer concisely:"
    r = post(ANS_BASE.rstrip("/") + "/chat/completions",
             {"model": ANS_MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0},
             {"Authorization": f"Bearer {ANS_KEY}"})
    if not r or "choices" not in r:
        return None
    a = re.sub(r"<think>.*?</think>", "", r["choices"][0]["message"]["content"] or "",
               flags=re.DOTALL).strip()
    jp = (f"Judge if the answer matches the gold answer. Treat different date "
          f"formats for the same date as equal. Be generous on topic. "
          f"Question: {q['question']} Gold: {gold} Answer: {a}. "
          f'JSON only: {{"label": "CORRECT" or "WRONG"}}')
    jr = post("https://openrouter.ai/api/v1/chat/completions",
              {"model": "openai/gpt-4o-mini",
               "messages": [{"role": "user", "content": jp}],
               "response_format": {"type": "json_object"}, "temperature": 0},
              {"Authorization": f"Bearer {OR_KEY}"})
    if not jr or "choices" not in jr:
        return None
    return 1 if "CORRECT" in jr["choices"][0]["message"]["content"].upper() else 0


with ThreadPoolExecutor(max_workers=4) as ex:
    sc = [x for x in ex.map(qa, ret) if x is not None]

if sc:
    print(f"=== {ANS_MODEL} no-original: J={sum(sc)/len(sc)*100:.0f} ({len(sc)} valid) ===")
else:
    print(f"=== {ANS_MODEL}: all answers failed (answerer unreachable) ===")
