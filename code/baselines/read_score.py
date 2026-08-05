#!/usr/bin/env python3
"""Read the judge accuracy out of one evaluation file.

Prints "<percent> <count>". Exits non-zero when the file holds no judged
record, which is how a sweep tells a run that scored something from one that
died before judging — an evaluator writes its output file as it goes, so the
file existing proves nothing. The count says how much was judged; a run that
died halfway reports its real n rather than passing as complete.

    python baselines/read_score.py results/comparison-s0/mem0/eval.json
"""

import json
import sys


def read_score(path):
    """Return (percent, judged_count) for an evaluation file."""
    with open(path) as handle:
        data = json.load(handle)
    records = data.get("records") or data.get("results") or []
    judged = [r for r in records if r.get("judge_score") is not None]
    if not judged:
        raise ValueError(f"no judged records in {path}")
    return 100 * sum(r["judge_score"] for r in judged) / len(judged), len(judged)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    try:
        percent, count = read_score(sys.argv[1])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"{percent:.1f} {count}")


if __name__ == "__main__":
    main()
