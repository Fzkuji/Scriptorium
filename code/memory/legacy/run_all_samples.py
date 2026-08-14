"""
Run memory_builder on all 10 LoCoMo samples, collect stats.
Usage: python3 run_all_samples.py
"""
import subprocess
import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

TOTAL_SAMPLES = 10
VERIFY = 20
MODEL = "qwen3.6-flash"

results = []

for i in range(TOTAL_SAMPLES):
    print(f"\n{'='*60}")
    print(f"SAMPLE {i}/{TOTAL_SAMPLES}")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(
        ["python3", "-u", "memory_builder.py",
         "--sample", str(i),
         "--sessions", "0",
         "--verify", str(VERIFY),
         "--model", MODEL],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0

    output = proc.stdout
    print(output)

    # Parse results
    found = output.count("[FOUND]")
    miss = output.count("[MISS]")
    total = found + miss

    # Parse wiki stats
    file_count = 0
    for line in output.splitlines():
        if "files," in line and "bytes" in line:
            parts = line.strip().split()
            for j, p in enumerate(parts):
                if p == "files,":
                    file_count = int(parts[j-1])

    results.append({
        "sample": i,
        "found": found,
        "miss": miss,
        "total": total,
        "accuracy": found / total if total > 0 else 0,
        "file_count": file_count,
        "time": elapsed
    })

    print(f"\n  Sample {i}: {found}/{total} = {found/total:.1%}" if total > 0 else f"\n  Sample {i}: no results")

# Summary
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
print(f"{'Sample':<8} {'Found':<8} {'Miss':<8} {'Total':<8} {'Accuracy':<10} {'Files':<8} {'Time':<8}")
total_found = 0
total_tested = 0
for r in results:
    print(f"{r['sample']:<8} {r['found']:<8} {r['miss']:<8} {r['total']:<8} {r['accuracy']:<10.1%} {r['file_count']:<8} {r['time']:<8.0f}s")
    total_found += r['found']
    total_tested += r['total']

overall = total_found / total_tested if total_tested > 0 else 0
print(f"\nOverall: {total_found}/{total_tested} = {overall:.1%}")

# Save
with open("memory_test/all_samples_results.json", "w") as f:
    json.dump({"results": results, "overall_accuracy": overall, "model": MODEL}, f, indent=2)
print(f"\nSaved to memory_test/all_samples_results.json")
