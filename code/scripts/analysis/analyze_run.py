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
        payload = json.load(open(p))
        # The locked evaluator writes {"records"}; the unified one writes
        # {"meta", "results"} and carries a build-stats row with no score.
        rows = payload.get("records") or payload.get("results") or []
        rows = [r for r in rows if r.get("judge_score") is not None]
        n = len(rows)
        std = sum(1 for r in rows if r["judge_score"] >= 1)
        strict = sum(1 for r in rows if r["judge_score"] >= 1 and not ABSTAIN.search(str(r.get("answer") or "")))
        ab = sum(1 for r in rows if ABSTAIN.search(str(r.get("answer") or "")))
        cats = {}
        for r in rows:
            c = r.get("beam_category", r.get("category")); ok = r["judge_score"] >= 1 and not ABSTAIN.search(str(r.get("answer") or ""))
            cats.setdefault(c, [0, 0]); cats[c][0] += ok; cats[c][1] += 1
        print(f"  分数: n={n} 标准={std/n*100:.1f} 严格={strict/n*100:.1f} 弃答={ab}")
        print("  分项:", {CAT.get(c, c): f"{a/b*100:.1f}" for c, (a, b) in sorted(cats.items())})
    # LongMemEval 用 avg_score + results,分项按 question_type 而非数字类别。
    for lf in sorted(glob.glob(os.path.join(run_dir, "eval_*.json"))):
        if os.path.basename(lf) == "eval_full.json":
            continue
        try:
            d = json.load(open(lf))
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict) or "avg_score" not in d:
            continue
        rows = d.get("results") or []
        types = {}
        for r in rows:
            t = r.get("question_type") or r.get("type") or "unknown"
            ok = float(r.get("score") or r.get("judge_score") or 0) >= 1
            types.setdefault(t, [0, 0])
            types[t][0] += ok
            types[t][1] += 1
        print(f"  [{os.path.basename(lf)}] avg={d['avg_score']:.1f} "
              f"n={len(rows)} found={d.get('found', '-')}")
        if len(types) > 1:
            print("  分项:", {t: f"{a/b*100:.1f}" for t, (a, b) in sorted(types.items())})
    # 旧格式:build 统计塞在 sample*_questions.json 的 _build_stats 记录里。
    for qf in sorted(glob.glob(os.path.join(run_dir, "sample*_questions.json"))):
        recs = json.load(open(qf))
        b = recs[0] if recs and isinstance(recs, list) and isinstance(recs[0], dict) \
            and recs[0].get("question_id") == "_build_stats" else None
        if b and "build_time_s" in b:
            print(f"  build[{os.path.basename(qf)}]: {b['build_time_s']/60:.0f}min "
                  f"{b['build_calls']}调用 out={b['build_tokens_out']/1e6:.2f}M | {b.get('notes','')}")
    # 当前格式:build.json 独立成文件,键名也换了。
    bp = os.path.join(run_dir, "build.json")
    if os.path.exists(bp):
        b = json.load(open(bp))
        secs = b.get("wall_time_s") or 0
        out = b.get("output_tokens") or 0
        line = (f"  build: {secs/60:.0f}min {b.get('calls', 0)}调用 "
                f"out={out/1e6:.2f}M events={b.get('event_count', 0)}")
        mem = b.get("memory") or {}
        if mem:
            line += (f" | 库 {mem.get('files', 0)}文件"
                     f"(topic {mem.get('topic_files', 0)}"
                     f"/timeline {mem.get('timeline_files', 0)}"
                     f"/source {mem.get('source_files', 0)})")
        print(line)
    pp = os.path.join(run_dir, "performance.json")
    if os.path.exists(pp):
        u = (json.load(open(pp)).get("usage") or {}).get("totals") or {}
        if u:
            # estimated 用你给的单价;SDK 那个按 Anthropic 价折算,不是账单。
            print(f"  成本: estimated=${u.get('estimated_cost_usd', 0):.4f} "
                  f"in={u.get('input_tokens', 0)/1e6:.2f}M "
                  f"out={u.get('output_tokens', 0)/1e6:.2f}M "
                  f"cache_read={u.get('cache_read_tokens', 0)/1e6:.2f}M")
    for md in sorted(glob.glob(os.path.join(run_dir, "memory_sample*"))):
        files = glob.glob(md + "/**/*.md", recursive=True)
        lines = [l for f in files for l in open(f) if l.strip() and not l.strip().startswith("#")]
        multi = sum(1 for l in lines if len(DIA.findall(l)) >= 2)
        tl = sum(1 for f in files if "/timeline/" in f)
        print(f"  库[{os.path.basename(md)}]: {len(files)}文件(timeline {tl}) {len(lines)}行 多引用行{multi}")

for d in sys.argv[1:]:
    analyze(d.rstrip("/"))
