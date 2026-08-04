#!/usr/bin/env python3
"""决胜局裁决:v8.8 vs v9.0c+日历,按预注册规则(docs/v9_research_protocol.md)机械判定。
- 每臂 4 个 run(s0/s1 × r1/r2)的严格分 → 均值与合并σ;|Δ|≤2σ_pooled 平局→按成本定 v9
- 机制校验:s1 日期题 q26/q30/q32/q38 在日历臂的对错(应转对)
用法: python3 scripts/judge_showdown.py"""
import json, re, glob, os, statistics

ABSTAIN = re.compile(r"not mentioned|no information|cannot be determined|not specified|unknown", re.I)
DATE_QS = ["s1_q26", "s1_q30", "s1_q32", "s1_q38"]

def strict(run_dir):
    p = os.path.join(run_dir, "eval_full.json")
    if not os.path.exists(p):
        return None, {}
    rows = json.load(open(p))["records"]
    ok = sum(1 for r in rows if r["judge_score"] >= 1 and not ABSTAIN.search(str(r.get("answer") or "")))
    dq = {r["question_id"]: (r["judge_score"] >= 1) for r in rows if r["question_id"] in DATE_QS}
    return ok / len(rows) * 100, dq

arms = {"v88": [], "v9c": []}
date_check = {"v88": {}, "v9c": {}}
for d in sorted(glob.glob("results/sd-*")):
    arm = "v88" if "-v88-" in d else "v9c"
    s, dq = strict(d)
    if s is None:
        print(f"{d}: 无判分结果,跳过"); continue
    arms[arm].append((os.path.basename(d), s))
    for q, correct in dq.items():
        date_check[arm].setdefault(q, []).append(correct)

print("== run 级严格分")
for arm, runs in arms.items():
    for name, s in runs:
        print(f"  {name}: {s:.1f}")
means, sds = {}, {}
for arm, runs in arms.items():
    vals = [s for _, s in runs]
    means[arm] = statistics.mean(vals) if vals else float("nan")
    sds[arm] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{arm}: mean={means[arm]:.2f} sd={sds[arm]:.2f} n={len(vals)}")

pooled = ((sds["v88"] ** 2 + sds["v9c"] ** 2) / 2) ** 0.5
delta = means["v9c"] - means["v88"]
print(f"\nΔ(v9c−v88)={delta:+.2f}  合并σ={pooled:.2f}  阈值±{2*pooled:.2f}")
if abs(delta) <= 2 * pooled:
    print("裁决:平局 → 按预注册规则,以成本定 **v9.0c+日历** 为定稿配置")
elif delta > 0:
    print("裁决:**v9.0c+日历 胜**,定稿")
else:
    print("裁决:**v8.8 胜**,定稿(v9 保留为成本档)")

print("\n== 机制校验(s1 日期题,日历臂应转对)")
for q in DATE_QS:
    a = date_check["v88"].get(q, []); b = date_check["v9c"].get(q, [])
    print(f"  {q}: v88 {sum(a)}/{len(a)} 对   v9c {sum(b)}/{len(b)} 对")
