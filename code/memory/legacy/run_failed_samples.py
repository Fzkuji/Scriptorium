"""Re-run only the timed-out samples (1,2,3,4,7,8) with extended timeout."""
import subprocess, json, sys, time
sys.stdout.reconfigure(line_buffering=True)

FAILED = [1, 2, 3, 4, 7, 8]
results = []

for i in FAILED:
    print(f"\n{'='*60}\nSAMPLE {i}\n{'='*60}")
    t0 = time.time()
    proc = subprocess.run(
        ["python3", "-u", "memory_builder.py",
         "--sample", str(i), "--sessions", "0", "--verify", "20",
         "--model", "qwen3.6-flash", "--outdir", "memory_test_v2"],
        capture_output=True, text=True, timeout=1800
    )
    elapsed = time.time() - t0
    output = proc.stdout
    print(output[-800:] if len(output) > 800 else output)

    found = output.count("[FOUND]")
    miss = output.count("[MISS]")
    total = found + miss
    results.append({"sample": i, "found": found, "miss": miss, "total": total,
                    "accuracy": found/total if total > 0 else 0, "time": elapsed})
    print(f"  Sample {i}: {found}/{total} = {found/total:.1%}" if total > 0 else f"  Sample {i}: no results")

print(f"\n{'='*60}\nFAILED SAMPLES RESULTS\n{'='*60}")
for r in results:
    print(f"Sample {r['sample']}: {r['accuracy']:.1%} ({r['found']}/{r['total']}) in {r['time']:.0f}s")

# Merge with existing results
try:
    with open("memory_test_v2/all_results.json") as f:
        existing = json.load(f)
    existing_results = existing["results"]
except:
    existing_results = []

# Update
for r in results:
    for i, er in enumerate(existing_results):
        if er["sample"] == r["sample"]:
            existing_results[i] = r
            break
    else:
        existing_results.append(r)

existing_results.sort(key=lambda x: x["sample"])
tf = sum(r["found"] for r in existing_results if r["total"] > 0)
tt = sum(r["total"] for r in existing_results if r["total"] > 0)
overall = tf/tt if tt else 0

with open("memory_test_v2/all_results.json", "w") as f:
    json.dump({"results": existing_results, "overall": overall}, f, indent=2)
print(f"\nOverall (all 10): {tf}/{tt} = {overall:.1%}")
