#!/usr/bin/env python3
"""一键分析一个 results/<run>/ 目录:分数(双口径+分项)、库统计、build 成本。
用法: python3 scripts/analyze_run.py results/v90b-b1 [results/对照组 ...]
研究协议(docs/v9_research_protocol.md)的证据三件套之 (a)(b)。"""
import json, re, sys, glob, os

ABSTAIN = re.compile(r"not mentioned|no information|cannot be determined|not specified|unknown", re.I)
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
DIA = re.compile(r"D(\d+):(\d+)")

def analyze(run_dir):
    print(f"===== {run_dir}")
    p = os.path.join(run_dir, "eval_full.json")
    if os.path.exists(p):
        rows = json.load(open(p))["records"]
        n = len(rows)
        std = sum(1 for r in rows if r["judge_score"] >= 1)
        strict = sum(1 for r in rows if r["judge_score"] >= 1 and not ABSTAIN.search(str(r.get("answer") or "")))
        ab = sum(1 for r in rows if ABSTAIN.search(str(r.get("answer") or "")))
        cats = {}
        for r in rows:
            c = r["category"]; ok = r["judge_score"] >= 1 and not ABSTAIN.search(str(r.get("answer") or ""))
            cats.setdefault(c, [0, 0]); cats[c][0] += ok; cats[c][1] += 1
        print(f"  分数: n={n} 标准={std/n*100:.1f} 严格={strict/n*100:.1f} 弃答={ab}")
        print("  分项:", {CAT.get(c, c): f"{a/b*100:.1f}" for c, (a, b) in sorted(cats.items())})
    for qf in sorted(glob.glob(os.path.join(run_dir, "sample*_questions.json"))):
        recs = json.load(open(qf))
        b = recs[0] if recs and recs[0].get("question_id") == "_build_stats" else None
        if b:
            print(f"  build[{os.path.basename(qf)}]: {b['build_time_s']/60:.0f}min "
                  f"{b['build_calls']}调用 out={b['build_tokens_out']/1e6:.2f}M | {b.get('notes','')}")
    for md in sorted(glob.glob(os.path.join(run_dir, "memory_sample*"))):
        files = glob.glob(md + "/**/*.md", recursive=True)
        lines = [l for f in files for l in open(f) if l.strip() and not l.strip().startswith("#")]
        multi = sum(1 for l in lines if len(DIA.findall(l)) >= 2)
        tl = sum(1 for f in files if "/timeline/" in f)
        print(f"  库[{os.path.basename(md)}]: {len(files)}文件(timeline {tl}) {len(lines)}行 多引用行{multi}")

for d in sys.argv[1:]:
    analyze(d.rstrip("/"))
