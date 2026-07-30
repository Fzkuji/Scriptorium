#!/usr/bin/env python3
"""Plot first-pass and post-verification coverage over actual group size."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOW_SOURCE = ROOT / "experiments/gpt56-chunk-curve/analysis/aggregate_by_benchmark_window.csv"
SESSION_SOURCE = ROOT / "results/formal/gpt56-session-groups/analysis/session-group-results.json"
OUT = Path(__file__).with_name("verifier_recovery.svg")
BENCHMARKS = (("locomo", "LoCoMo"), ("longmemeval-s", "LongMemEval-S"), ("beam-100k", "BEAM"))
X_MIN, X_MAX = 3.2, 400


def load_points() -> dict[str, list[dict]]:
    points = {benchmark: [] for benchmark, _ in BENCHMARKS}
    with WINDOW_SOURCE.open() as handle:
        for row in csv.DictReader(handle):
            points[row["benchmark"]].append({
                "actual": float(row["messages"]) / float(row["writer_calls"]),
                "first": 100 * float(row["first_pass_source_coverage"]),
                "final": 100 * float(row["final_source_coverage"]),
                "kind": "window",
            })
    cells = json.loads(SESSION_SOURCE.read_text(encoding="utf-8"))["cells"]
    for benchmark, _ in BENCHMARKS:
        sizes = sorted({cell["nominal_session_group_size"] for cell in cells if cell["benchmark"] == benchmark})
        for size in sizes:
            subset = [cell for cell in cells if cell["benchmark"] == benchmark and cell["nominal_session_group_size"] == size]
            points[benchmark].append({
                "actual": sum(cell["construction"]["messages"] for cell in subset) / sum(cell["writer_calls"] for cell in subset),
                "first": 100 * sum(cell["construction"]["first_pass_source_coverage"] for cell in subset) / len(subset),
                "final": 100 * sum(cell["construction"]["post_verify_source_coverage"] for cell in subset) / len(subset),
                "kind": "session",
            })
    return points


def marker(x: float, y: float, color: str, kind: str) -> str:
    if kind == "window":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="white" stroke="{color}" stroke-width="2"/>'
    return f'<path d="M{x:.1f},{y-4:.1f} L{x+4:.1f},{y:.1f} L{x:.1f},{y+4:.1f} L{x-4:.1f},{y:.1f} Z" fill="white" stroke="{color}" stroke-width="2"/>'


def main() -> None:
    points = load_points()
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 350">',
           '<style>text{font-family:Times New Roman,serif;font-size:12px;fill:#222}.small{font-size:10px}.title{font-size:14px;font-weight:600}</style>']
    panel_width, plot_width, plot_height, top = 375, 292, 220, 42
    log_min, log_span = math.log2(X_MIN), math.log2(X_MAX) - math.log2(X_MIN)
    for panel, (benchmark, title) in enumerate(BENCHMARKS):
        left = 52 + panel * panel_width
        x = lambda value: left + plot_width * (math.log2(value) - log_min) / log_span
        y = lambda value: top + plot_height * (102 - value) / 62
        rows = sorted(points[benchmark], key=lambda row: row["actual"])
        svg.append(f'<text class="title" x="{left+plot_width/2:.1f}" y="19" text-anchor="middle">{title}</text>')
        for tick in (40, 60, 80, 100):
            py = y(tick)
            svg.extend([f'<line x1="{left}" y1="{py:.1f}" x2="{left+plot_width}" y2="{py:.1f}" stroke="#ddd"/>',
                        f'<text class="small" x="{left-7}" y="{py+4:.1f}" text-anchor="end">{tick}%</text>'])
        for tick in (4, 8, 16, 32, 64, 128, 256):
            px = x(tick)
            svg.extend([f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+plot_height}" stroke="#eee"/>',
                        f'<text class="small" x="{px:.1f}" y="{top+plot_height+16}" text-anchor="middle">{tick}</text>'])
        svg.extend([f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333"/>',
                    f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
                    f'<text class="small" x="{left+plot_width/2:.1f}" y="{top+plot_height+34}" text-anchor="middle">Actual messages / writer call (log₂)</text>'])
        for row in rows:
            svg.append(f'<line x1="{x(row["actual"]):.1f}" y1="{y(row["first"]):.1f}" x2="{x(row["actual"]):.1f}" y2="{y(row["final"]):.1f}" stroke="#9ecae1" stroke-width="2"/>')
        for field, color in (("first", "#D55E00"), ("final", "#0072B2")):
            for kind in ("window", "session"):
                subset = [row for row in rows if row["kind"] == kind]
                coords = " ".join(f'{x(row["actual"]):.1f},{y(row[field]):.1f}' for row in subset)
                svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2"/>')
            windows = [row for row in rows if row["kind"] == "window"]
            sessions = [row for row in rows if row["kind"] == "session"]
            svg.append(f'<line x1="{x(windows[-1]["actual"]):.1f}" y1="{y(windows[-1][field]):.1f}" x2="{x(sessions[0]["actual"]):.1f}" y2="{y(sessions[0][field]):.1f}" stroke="{color}" stroke-width="1.4"/>')
            for row in rows:
                svg.append(marker(x(row["actual"]), y(row[field]), color, row["kind"]))
    svg.extend(['<line x1="360" y1="334" x2="386" y2="334" stroke="#D55E00" stroke-width="2.2"/><text class="small" x="394" y="338">First pass</text>',
                '<line x1="490" y1="334" x2="516" y2="334" stroke="#0072B2" stroke-width="2.2"/><text class="small" x="524" y="338">After verifier</text>',
                '<circle cx="650" cy="334" r="3.6" fill="white" stroke="#444" stroke-width="1.7"/><text class="small" x="660" y="338">W</text>',
                marker(705, 334, "#444", "session"), '<text class="small" x="715" y="338">Session group</text>', '</svg>'])
    OUT.write_text("".join(svg), encoding="utf-8")
    assert round(points["beam-100k"][-1]["final"], 1) == 88.5


if __name__ == "__main__":
    main()
