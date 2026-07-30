"""Run all 10 samples with optimized settings (5W1H + copy + FC verify)."""
import subprocess, json, sys, time
sys.stdout.reconfigure(line_buffering=True)

results = []
for i in range(10):
    print(f"\n{'='*60}\nSAMPLE {i}/10\n{'='*60}")
    t0 = time.time()
    proc = subprocess.run(
        ["python3", "-u", "memory_builder.py",
         "--sample", str(i), "--sessions", "0", "--verify", "20",
         "--model", "qwen3.6-flash", "--outdir", "memory_test_v2"],
        capture_output=True, text=True, timeout=900
    )
    elapsed = time.time() - t0
    output = proc.stdout
    print(output[-500:] if len(output) > 500 else output)
    
    found = output.count("[FOUND]")
    miss = output.count("[MISS]")
    total = found + miss
    results.append({"sample": i, "found": found, "miss": miss, "total": total,
                    "accuracy": found/total if total > 0 else 0, "time": elapsed})
    print(f"  Sample {i}: {found}/{total} = {found/total:.1%}" if total > 0 else f"  Sample {i}: no results")

print(f"\n{'='*60}\nOVERALL\n{'='*60}")
tf = sum(r["found"] for r in results)
tt = sum(r["total"] for r in results)
print(f"{'Sample':<8} {'Accuracy':<10} {'Time':<8}")
for r in results:
    print(f"{r['sample']:<8} {r['accuracy']:<10.1%} {r['time']:<8.0f}s")
print(f"\nOverall: {tf}/{tt} = {tf/tt:.1%}")

with open("memory_test_v2/all_results.json", "w") as f:
    json.dump({"results": results, "overall": tf/tt if tt else 0}, f, indent=2)
