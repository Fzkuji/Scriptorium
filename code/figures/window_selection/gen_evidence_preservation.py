#!/usr/bin/env python3
"""Plot LoCoMo source and gold-evidence coverage over actual group size."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOW_SOURCE = ROOT / "experiments/gpt56-chunk-curve/analysis/aggregate_by_benchmark_window.csv"
SESSION_SOURCE = ROOT / "results/formal/gpt56-session-groups/analysis/session-group-results.json"
OUT = Path(__file__).with_name("evidence_preservation.svg")
X_MIN, X_MAX = 3.2, 400
SERIES = (("source", "All source turns referenced", "#D55E00", ""),
          ("gold", "Questions with all gold evidence referenced", "#0072B2", ""))


def load_points() -> list[dict]:
    points = []
    with WINDOW_SOURCE.open() as handle:
        for row in csv.DictReader(handle):
            if row["benchmark"] == "locomo":
                points.append({
                    "actual": float(row["messages"]) / float(row["writer_calls"]),
                    "source": 100 * float(row["final_source_coverage"]),
                    "gold": 100 * float(row["gold_question_all_coverage"]),
                    "kind": "window",
                })
    cells = json.loads(SESSION_SOURCE.read_text(encoding="utf-8"))["cells"]
    sizes = sorted({cell["nominal_session_group_size"] for cell in cells if cell["benchmark"] == "locomo"})
    for size in sizes:
        subset = [cell for cell in cells if cell["benchmark"] == "locomo" and cell["nominal_session_group_size"] == size]
        points.append({
            "actual": sum(cell["construction"]["messages"] for cell in subset) / sum(cell["writer_calls"] for cell in subset),
            "source": 100 * sum(cell["evidence"]["source_coverage"] for cell in subset) / len(subset),
            "gold": 100 * sum(cell["evidence"]["gold_all_coverage"] for cell in subset) / len(subset),
            "kind": "session",
        })
    return sorted(points, key=lambda row: row["actual"])


def marker(x: float, y: float, color: str, kind: str) -> str:
    if kind == "window":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" stroke="{color}" stroke-width="2.1"/>'
    return f'<path d="M{x:.1f},{y-4.5:.1f} L{x+4.5:.1f},{y:.1f} L{x:.1f},{y+4.5:.1f} L{x-4.5:.1f},{y:.1f} Z" fill="white" stroke="{color}" stroke-width="2.1"/>'


def main() -> None:
    rows = load_points()
    x0, y0, width, height = 82, 48, 650, 225
    log_min, log_span = math.log2(X_MIN), math.log2(X_MAX) - math.log2(X_MIN)
    x = lambda value: x0 + width * (math.log2(value) - log_min) / log_span
    y = lambda value: y0 + height * (101 - value) / 51
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 350">',
           '<style>text{font-family:Times New Roman,serif;font-size:13px;fill:#222}.small{font-size:11px}</style>']
    for tick in (50, 60, 70, 80, 90, 100):
        py = y(tick)
        svg.extend([f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+width}" y2="{py:.1f}" stroke="#ddd"/>',
                    f'<text class="small" x="{x0-9}" y="{py+4:.1f}" text-anchor="end">{tick}%</text>'])
    for tick in (4, 8, 16, 32, 64, 128, 256):
        px = x(tick)
        svg.extend([f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+height}" stroke="#eee"/>',
                    f'<text class="small" x="{px:.1f}" y="{y0+height+18}" text-anchor="middle">{tick}</text>'])
    svg.extend([f'<line x1="{x0}" y1="{y0+height}" x2="{x0+width}" y2="{y0+height}" stroke="#333"/>',
                f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+height}" stroke="#333"/>',
                f'<text transform="translate(20,{y0+height/2}) rotate(-90)" text-anchor="middle">Coverage on LoCoMo</text>',
                f'<text class="small" x="{x0+width/2}" y="{y0+height+38}" text-anchor="middle">Actual messages / writer call (log₂)</text>'])
    for field, label, color, dash in SERIES:
        for kind in ("window", "session"):
            subset = [row for row in rows if row["kind"] == kind]
            coords = " ".join(f'{x(row["actual"]):.1f},{y(row[field]):.1f}' for row in subset)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        windows = [row for row in rows if row["kind"] == "window"]
        sessions = [row for row in rows if row["kind"] == "session"]
        svg.append(f'<line x1="{x(windows[-1]["actual"]):.1f}" y1="{y(windows[-1][field]):.1f}" x2="{x(sessions[0]["actual"]):.1f}" y2="{y(sessions[0][field]):.1f}" stroke="{color}" stroke-width="1.5"/>')
        for row in rows:
            svg.append(marker(x(row["actual"]), y(row[field]), color, row["kind"]))
        lx = 115 if field == "source" else 455
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.extend([f'<line x1="{lx}" y1="20" x2="{lx+28}" y2="20" stroke="{color}" stroke-width="2.6"{dash_attr}/>',
                    f'<text class="small" x="{lx+35}" y="24">{label}</text>'])
    svg.extend(['<circle cx="300" cy="335" r="3.7" fill="white" stroke="#444" stroke-width="1.7"/><text class="small" x="310" y="339">W</text>',
                marker(360, 335, "#444", "session"), '<text class="small" x="370" y="339">Session group</text>', '</svg>'])
    OUT.write_text("".join(svg), encoding="utf-8")
    assert round(rows[-1]["gold"], 2) == 84.39


if __name__ == "__main__":
    main()
