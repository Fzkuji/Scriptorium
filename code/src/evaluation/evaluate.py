"""Unified evaluation CLI — one entry point for all benchmarks and metrics.

Input: per-question JSON (list or {"results": [...]}), each record:
    {
      "question_id": str,            # optional
      "question": str,
      "gold": str,                   # ground-truth answer (or rubric/explanation)
      "category": int,               # LoCoMo: 1-5
      "question_type": str,          # LongMemEval: 6 types
      "abstention": bool,            # LongMemEval: _abs questions
      "memories": [{text, date}]     # retrieved memories → unified answerer
        OR "answer": str             # pre-generated answer (skips answerer)
    }

Pipeline: (answerer if needed) → lexical metrics → LLM judge → aggregate.
Every stage writes back into the per-question records and saves to disk, so
runs are resumable and re-judgeable (protocol §6.8).

Usage (from project root):
    python3 -m src.evaluation.evaluate --benchmark locomo \
        --input results/<run>/questions.json --output results/<run>/eval.json \
        --metrics judge f1 bleu rouge em f1_official
    python3 -m src.evaluation.evaluate --benchmark longmemeval \
        --input ... --output ... --metrics judge f1 bleu
Flags: --skip-judge (lexical only), --skip-answerer (use existing answers),
       --limit N (dev slice).
"""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.metrics import compute_lexical
from src.evaluation.answerer import generate_answer
from src.evaluation.judges import judge_locomo, judge_longmemeval
from src.evaluation.llm_clients import (ANSWERER_BASE, ANSWERER_MODEL,
                                        JUDGE_BASE, JUDGE_MODEL)

LOCOMO_CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                    4: "single-hop", 5: "adversarial"}
LME_TYPES = ("single-session-user", "single-session-assistant",
             "single-session-preference", "multi-session",
             "temporal-reasoning", "knowledge-update")


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("results", "questions", "individual_results"):
            if key in data:
                return data[key]
        raise ValueError(f"Cannot find question list in {path}")
    return data


def save(records, meta, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"meta": meta, "results": records}, handle,
                indent=2, ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ensure_distinct_paths(paths):
    entries = [(label, Path(value).expanduser().absolute())
               for label, value in paths.items()]
    for index, (left_label, left) in enumerate(entries):
        for right_label, right in entries[index + 1:]:
            aliases = left.resolve() == right.resolve()
            if left.exists() and right.exists():
                try:
                    aliases = aliases or os.path.samefile(left, right)
                except OSError:
                    pass
            aliases = aliases or (
                left.parent.resolve() == right.parent.resolve()
                and unicodedata.normalize("NFC", left.name).casefold()
                == unicodedata.normalize("NFC", right.name).casefold()
            )
            if aliases:
                raise ValueError(
                    f"artifact path collision: {left_label}={left}, "
                    f"{right_label}={right}"
                )


@contextmanager
def exclusive_output_lock(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{output}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"evaluation output is already locked: {output}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _resume_records(output, expected_meta):
    with Path(output).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("existing evaluation output is not an object")
    meta = payload.get("meta")
    records = payload.get("results")
    if not isinstance(meta, dict) or not isinstance(records, list):
        raise ValueError("existing evaluation output lacks meta/results")
    for key in (
        "benchmark", "input", "input_sha256", "answerer_model", "judge_model",
        "answerer_base", "judge_base", "metrics", "evaluator_sha256", "limit",
        "skip_answerer", "skip_judge",
    ):
        if meta.get(key) != expected_meta.get(key):
            raise ValueError(f"existing evaluation metadata differs at {key}")
    return records, meta


def run_answerer(records, out_path, meta, skip=False):
    if skip:
        return
    pending = [r for r in records if "answer" not in r and "memories" in r]
    for i, r in enumerate(pending):
        answer, usage = generate_answer(r["question"], r["memories"])
        r["answer"] = answer
        r["answerer_usage"] = usage
        if (i + 1) % 20 == 0:
            print(f"  answerer {i+1}/{len(pending)}")
            save(records, meta, out_path)
    if pending:
        save(records, meta, out_path)


def run_lexical(records, metrics):
    lexical = [m for m in metrics if m != "judge"]
    if not lexical:
        return
    for r in records:
        if "answer" not in r:
            continue
        r["lexical"] = compute_lexical(
            r["answer"], r.get("gold", ""), category=r.get("category"),
            metrics=lexical)


def run_judge(records, benchmark, out_path, meta):
    pending = [r for r in records if "answer" in r and "judge_score" not in r]
    for i, r in enumerate(pending):
        if benchmark == "locomo":
            score, raw, usage = judge_locomo(
                r["question"], r.get("gold", ""), r["answer"],
                category=r.get("category"))
        else:
            score, raw, usage = judge_longmemeval(
                r.get("question_type", "multi-session"), r["question"],
                r.get("gold", ""), r["answer"],
                abstention=bool(r.get("abstention")))
        r["judge_score"] = score
        r["judge_raw"] = raw
        r["judge_usage"] = usage
        if (i + 1) % 20 == 0:
            print(f"  judge {i+1}/{len(pending)}")
            save(records, meta, out_path)
    if pending:
        save(records, meta, out_path)


def aggregate_efficiency(records):
    """LightMem-style efficiency accounting (tokens in thousands, calls, time).

    Sources:
    - build: the adapter's _build_stats record (build_time_s, build_calls,
      build_tokens_in/out via _usage_tracker)
    - retrieval: per-question record["retrieval"] (latency_s, calls,
      tokens_in/out) — search-time LLM cost, zero for embedding-only systems
    - answer/judge: per-question usage recorded by this script
    Judge cost is reported separately (evaluation overhead, not system cost).
    """
    builds = [r for r in records if r.get("question_id") == "_build_stats"]
    qs = [r for r in records if r.get("question_id") != "_build_stats"]

    def _sum(getter):
        return sum(getter(r) or 0 for r in qs)

    retr = {
        "calls": _sum(lambda r: r.get("retrieval", {}).get("calls")),
        "tokens_in": _sum(lambda r: r.get("retrieval", {}).get("tokens_in")),
        "tokens_out": _sum(lambda r: r.get("retrieval", {}).get("tokens_out")),
        "wall_time_s": round(_sum(lambda r: r.get("retrieval", {}).get("latency_s")), 1),
    }
    ans = {
        "calls": sum(1 for r in qs if "answerer_usage" in r),
        "tokens_in": _sum(lambda r: r.get("answerer_usage", {}).get("prompt_tokens")),
        "tokens_out": _sum(lambda r: r.get("answerer_usage", {}).get("completion_tokens")),
    }
    judge = {
        "calls": _sum(
            lambda r: r.get("judge_usage", {}).get("request_attempts")
        ),
        "tokens_in": _sum(lambda r: r.get("judge_usage", {}).get("prompt_tokens")),
        "tokens_out": _sum(lambda r: r.get("judge_usage", {}).get("completion_tokens")),
    }
    build_tokens = sum(
        (r.get("build_tokens_in", 0) or 0)
        + (r.get("build_tokens_out", 0) or 0)
        for r in builds)
    system_total_tokens = (
        build_tokens
        + retr["tokens_in"] + retr["tokens_out"]
        + ans["tokens_in"] + ans["tokens_out"])
    return {
        "build": {
            "units": len(builds),
            "time_s": round(sum((r.get("build_time_s", 0) or 0)
                                for r in builds), 1),
            "calls": sum((r.get("build_calls", 0) or 0) for r in builds),
            "tokens_in": sum((r.get("build_tokens_in", 0) or 0) for r in builds),
            "tokens_out": sum((r.get("build_tokens_out", 0) or 0) for r in builds),
            "num_memories": sum((r.get("num_memories", 0) or 0) for r in builds),
        },
        "retrieval": retr,
        "answer": ans,
        "system_total_tokens_k": round(system_total_tokens / 1000, 2),
        "judge_overhead": judge,
    }


def _mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_locomo(records):
    """Main table: cat 1-4 (protocol §6.6); cat 5 reported separately."""
    main = [r for r in records if r.get("category") in (1, 2, 3, 4)]
    adv = [r for r in records if r.get("category") == 5]
    out = {"n_main": len(main), "n_adversarial": len(adv), "by_category": {}}

    def block(rs):
        b = {}
        if any("judge_score" in r for r in rs):
            b["judge"] = _mean(r["judge_score"] for r in rs if "judge_score" in r)
        lex_keys = set()
        for r in rs:
            lex_keys.update(r.get("lexical", {}).keys())
        for k in sorted(lex_keys):
            b[k] = _mean(r["lexical"][k] for r in rs if k in r.get("lexical", {}))
        return b

    for cat in (1, 2, 3, 4, 5):
        rs = [r for r in records if r.get("category") == cat]
        if rs:
            out["by_category"][LOCOMO_CAT_NAMES[cat]] = {"n": len(rs), **block(rs)}
    out["overall_cat1_4"] = block(main)
    if adv:
        out["adversarial_separate"] = block(adv)
    return out


def aggregate_longmemeval(records):
    """6 types + task-avg (macro) + overall (micro) + abstention separate."""
    abs_qs = [r for r in records if r.get("abstention")]
    out = {"n": len(records), "n_abstention": len(abs_qs), "by_type": {}}

    def judge_mean(rs):
        scored = [r["judge_score"] for r in rs if "judge_score" in r]
        return _mean(scored) if scored else None

    type_means = []
    for t in LME_TYPES:
        # Official LongMemEval accounting includes abstention questions in
        # their original task type and in overall accuracy.  Abstention
        # accuracy is an additional view over the same 30 questions.
        rs = [r for r in records if r.get("question_type") == t]
        if rs:
            m = judge_mean(rs)
            out["by_type"][t] = {"n": len(rs), "judge": m}
            if m is not None:
                type_means.append(m)
    out["task_averaged_acc"] = _mean(type_means) if type_means else None
    out["overall_acc"] = judge_mean(records)
    out["abstention_acc"] = judge_mean(abs_qs)
    lex_keys = set()
    for r in records:
        lex_keys.update(r.get("lexical", {}).keys())
    for k in sorted(lex_keys):
        out[f"overall_{k}"] = _mean(
            r["lexical"][k] for r in records if k in r.get("lexical", {}))
    return out


def print_report(agg, benchmark, meta):
    print(f"\n{'='*64}")
    print(f"  {meta.get('label', 'run')} — {benchmark} unified evaluation")
    print(f"  answerer={meta['answerer_model']}  judge={meta['judge_model']}")
    print(f"{'='*64}")
    print(json.dumps(agg, indent=2, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["locomo", "longmemeval"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--metrics", nargs="+",
                    default=["judge", "f1", "bleu", "rouge", "em"],
                    help="any of: judge f1 f1_official bleu rouge em")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--skip-answerer", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().absolute()
    lock_path = Path(f"{output_path}.lock")
    _ensure_distinct_paths({
        "input": input_path,
        "output": output_path,
        "output_lock": lock_path,
    })
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    fresh_meta = {
        "label": args.label,
        "benchmark": args.benchmark,
        "answerer_model": ANSWERER_MODEL,
        "judge_model": JUDGE_MODEL,
        "answerer_base": ANSWERER_BASE,
        "judge_base": JUDGE_BASE,
        "metrics": args.metrics,
        "limit": args.limit,
        "skip_answerer": args.skip_answerer,
        "skip_judge": args.skip_judge,
        "input": str(input_path),
        "input_sha256": input_sha256,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with exclusive_output_lock(output_path):
        if output_path.exists():
            records, meta = _resume_records(output_path, fresh_meta)
        else:
            records = load_records(input_path)
            if args.limit:
                records = records[: args.limit]
            meta = fresh_meta
        meta["status"] = "running"
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        save(records, meta, output_path)
        try:
            run_answerer(records, output_path, meta, skip=args.skip_answerer)
            run_lexical(records, args.metrics)
            if "judge" in args.metrics and not args.skip_judge:
                run_judge(records, args.benchmark, output_path, meta)

            agg = (aggregate_locomo(records) if args.benchmark == "locomo"
                   else aggregate_longmemeval(records))
            agg["efficiency"] = aggregate_efficiency(records)
            meta["aggregate"] = agg
            meta["status"] = "complete"
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            meta["finished_at"] = meta["updated_at"]
            meta.pop("last_error", None)
            save(records, meta, output_path)
        except BaseException as exc:
            meta["status"] = "partial"
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            meta["last_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "at": meta["updated_at"],
            }
            save(records, meta, output_path)
            raise
    print_report(agg, args.benchmark, meta)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
