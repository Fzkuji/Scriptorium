#!/usr/bin/env python3
"""Aggregate audited session-group QA results by actual source length."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


class SessionGroupResultError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SessionGroupResultError(f"JSON object required: {path}")
    return value


def load_quality(
    score_root: Path,
    benchmark: str,
    tier: str,
    session_group_size: int,
) -> dict[str, Any]:
    cell = score_root / benchmark / tier / f"s{session_group_size}"
    if benchmark == "locomo":
        report = _read_json(cell / "locomo" / "eval_full.json")
        records = report.get("records")
        if not isinstance(records, list) or len(records) != 314:
            raise SessionGroupResultError("LoCoMo score inventory differs")
        scores = [row.get("judge_score") for row in records if isinstance(row, Mapping)]
        if len(scores) != 314 or any(score not in (0, 1) for score in scores):
            raise SessionGroupResultError("LoCoMo judge scores are invalid")
        score = statistics.mean(scores)
        if report.get("n") != 314 or report.get("overall") != score:
            raise SessionGroupResultError("LoCoMo aggregate differs")
        return {"metric": "locked_locomo_lj", "questions": 314, "score": score}
    if benchmark == "longmemeval-s":
        path = cell / "hypotheses.jsonl.eval-results-gpt-4o-mini"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        labels = [
            row.get("autoeval_label", {}).get("label")
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("autoeval_label"), Mapping)
        ]
        ids = [str(row.get("question_id", "")) for row in rows]
        if len(rows) != 2 or len(set(ids)) != 2 or any(not isinstance(x, bool) for x in labels):
            raise SessionGroupResultError("LongMemEval score inventory differs")
        correct = sum(labels)
        return {
            "metric": "official_longmemeval_qa_boolean",
            "questions": 2,
            "correct": correct,
            "score": correct / 2,
        }
    if benchmark == "beam-100k":
        report = _read_json(cell / "beam-semantic" / tier / "metrics.json")
        overall = report.get("overall")
        if report.get("metric_scope") != "rubric_nugget_only" or not isinstance(
            overall, Mapping
        ):
            raise SessionGroupResultError("BEAM semantic metric scope differs")
        score = overall.get("avg_score")
        pass_rate = overall.get("pass_rate")
        if (
            overall.get("questions") != 40
            or not isinstance(score, (int, float))
            or not isinstance(pass_rate, (int, float))
            or not 0 <= score <= 1
            or not 0 <= pass_rate <= 1
        ):
            raise SessionGroupResultError("BEAM semantic metrics are invalid")
        return {
            "metric": "beam_screening_rubric_nugget_mean",
            "questions": 40,
            "score": float(score),
            "pass_rate": float(pass_rate),
        }
    raise SessionGroupResultError(f"unsupported benchmark: {benchmark}")


def load_qa_efficiency(cell: Path) -> dict[str, int | float]:
    completion_path = cell / "completion.json"
    completion = _read_json(completion_path)
    audit = _read_json(cell / "audit.json")
    digest = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    if (
        completion.get("status") != "complete"
        or audit.get("status") != "verified_complete"
        or audit.get("completion_sha256") != digest
        or completion.get("run_ids") != audit.get("run_ids")
        or completion.get("question_count") != audit.get("question_count")
    ):
        raise SessionGroupResultError("QA completion and audit differ")
    questions = audit.get("question_count")
    visible = audit.get("visible_tokens")
    retrieval = audit.get("retrieval_model_calls")
    answers = audit.get("answer_model_calls")
    proxy = audit.get("proxy")
    chains = proxy.get("attempt_chains") if isinstance(proxy, Mapping) else None
    retries = chains.get("authorized_retries") if isinstance(chains, Mapping) else None
    if (
        not isinstance(questions, int)
        or questions <= 0
        or any(
            not isinstance(value, int) or value < 0
            for value in (visible, retrieval, answers, retries)
        )
    ):
        raise SessionGroupResultError("QA efficiency metrics are invalid")
    return {
        "questions": questions,
        "visible_tokens": visible,
        "visible_tokens_per_question": visible / questions,
        "retrieval_model_calls": retrieval,
        "retrieval_model_calls_per_question": retrieval / questions,
        "answer_model_calls": answers,
        "authorized_retries": retries,
    }


def analyze_campaign(
    *,
    sizes_report: Path,
    qa_root: Path,
    score_root: Path,
    allow_partial: bool,
) -> dict[str, Any]:
    sizes = _read_json(sizes_report)
    planned_runs = sizes.get("planned_runs")
    cells = sizes.get("cells")
    if (
        not isinstance(planned_runs, int)
        or planned_runs <= 0
        or planned_runs % 2
        or not isinstance(cells, list)
    ):
        raise SessionGroupResultError("actual-size report scope is invalid")
    analyzed: list[dict[str, Any]] = []
    pending: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise SessionGroupResultError("actual-size cell is invalid")
        benchmark = str(cell.get("benchmark"))
        tier = str(cell.get("tier"))
        size = cell.get("nominal_session_group_size")
        if not isinstance(size, int) or size <= 0:
            raise SessionGroupResultError("session-group size is invalid")
        label = f"{benchmark}/{tier}/s{size}"
        if cell.get("runs") != 2:
            pending.append(label)
            continue
        qa_cell = qa_root / benchmark / tier / f"s{size}"
        try:
            qa = load_qa_efficiency(qa_cell)
            quality = load_quality(score_root, benchmark, tier, size)
        except FileNotFoundError:
            pending.append(label)
            continue
        analyzed.append({**cell, "quality": quality, "qa_efficiency": qa})
    planned_cells = planned_runs // 2
    status = "complete" if len(analyzed) == planned_cells else "partial"
    if status == "partial" and not allow_partial:
        raise SessionGroupResultError(
            f"campaign has {planned_cells - len(analyzed)} pending cells"
        )
    return {
        "status": status,
        "planned_cells": planned_cells,
        "analyzed_cells": len(analyzed),
        "pending_cell_count": planned_cells - len(analyzed),
        "pending_cells": sorted(pending),
        "cells": analyzed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-report", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = analyze_campaign(
        sizes_report=args.sizes_report,
        qa_root=args.qa_root,
        score_root=args.score_root,
        allow_partial=args.allow_partial,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"{report['status']}: {report['analyzed_cells']}/"
        f"{report['planned_cells']} cells -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
