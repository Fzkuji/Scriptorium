#!/usr/bin/env python3
"""Plot construction cost over the unified actual group-size sweep."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOW_SOURCE = ROOT / "experiments/gpt56-chunk-curve/analysis/aggregate_by_benchmark_window.csv"
SESSION_SOURCE = ROOT / "results/formal/gpt56-session-groups/analysis/session-group-results.json"
OUT = Path(__file__).with_name("construction_efficiency.svg")
BENCHMARKS = (("locomo", "LoCoMo"), ("longmemeval-s", "LongMemEval-S"), ("beam-100k", "BEAM"))
METRICS = (
    ("Calls", "calls_per_100_messages", "#0072B2", ""),
    ("Input tokens", "input_tokens_per_1000_source_tokens", "#D55E00", ""),
    ("Output tokens", "output_tokens_per_1000_source_tokens", "#009E73", ""),
    ("Wall time", "wall_minutes_per_100_messages", "#CC79A7", ""),
)
X_MIN, X_MAX = 3.2, 400


def load_points() -> dict[str, list[dict]]:
    points = {benchmark: [] for benchmark, _ in BENCHMARKS}
    with WINDOW_SOURCE.open() as handle:
        for row in csv.DictReader(handle):
            point = {field: float(row[field]) for _, field, _, _ in METRICS}
            point.update({
                "actual": float(row["messages"]) / float(row["writer_calls"]),
                "label": f'W{int(row["write_turns"])}',
                "kind": "window",
            })
            points[row["benchmark"]].append(point)
    cells = json.loads(SESSION_SOURCE.read_text(encoding="utf-8"))["cells"]
    for benchmark, _ in BENCHMARKS:
        sizes = sorted({cell["nominal_session_group_size"] for cell in cells if cell["benchmark"] == benchmark})
        for size in sizes:
            subset = [cell for cell in cells if cell["benchmark"] == benchmark and cell["nominal_session_group_size"] == size]
            row = {
                field: sum(cell["construction"][field] for cell in subset) / len(subset)
                for _, field, _, _ in METRICS
            }
            row.update({
                "actual": sum(cell["construction"]["messages"] for cell in subset) / sum(cell["writer_calls"] for cell in subset),
                "label": f"S{size}",
                "kind": "session",
            })
            points[benchmark].append(row)
    return points


def diamond(x: float, y: float, color: str) -> str:
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
        y = lambda value: top + plot_height * (110 - value) / 110
        rows = sorted(points[benchmark], key=lambda row: row["actual"])
        baseline = next(row for row in rows if row["label"] == "W4")
        svg.append(f'<text class="title" x="{left + plot_width / 2:.1f}" y="19" text-anchor="middle">{title}</text>')
        for tick in (0, 25, 50, 75, 100):
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
        for _name, field, color, dash in METRICS:
            values = [(row["actual"], 100 * row[field] / baseline[field], row["kind"]) for row in rows]
            for kind in ("window", "session"):
                subset = [(actual, value) for actual, value, point_kind in values if point_kind == kind]
                coords = " ".join(f"{x(actual):.1f},{y(value):.1f}" for actual, value in subset)
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
                svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')
            windows = [(actual, value) for actual, value, kind in values if kind == "window"]
            sessions = [(actual, value) for actual, value, kind in values if kind == "session"]
            svg.append(f'<line x1="{x(windows[-1][0]):.1f}" y1="{y(windows[-1][1]):.1f}" x2="{x(sessions[0][0]):.1f}" y2="{y(sessions[0][1]):.1f}" stroke="{color}" stroke-width="1.4"/>')
            for actual, value in windows:
                svg.append(f'<circle cx="{x(actual):.1f}" cy="{y(value):.1f}" r="3.5" fill="white" stroke="{color}" stroke-width="1.8"/>')
            for actual, value in sessions:
                svg.append(diamond(x(actual), y(value), color))
    for index, (name, _field, color, dash) in enumerate(METRICS):
        lx = 155 + index * 185
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.extend([f'<line x1="{lx}" y1="334" x2="{lx+24}" y2="334" stroke="{color}" stroke-width="2"{dash_attr}/>',
                    f'<text class="small" x="{lx+30}" y="338">{name}</text>'])
    svg.extend(['<circle cx="900" cy="334" r="3.5" fill="white" stroke="#444" stroke-width="1.7"/><text class="small" x="910" y="338">W</text>',
                diamond(950, 334, "#444"), '<text class="small" x="960" y="338">Session group</text>', '</svg>'])
    OUT.write_text("".join(svg), encoding="utf-8")
    assert [row["label"] for row in sorted(points["locomo"], key=lambda row: row["actual"])] == ["W4", "W8", "W16", "W32", "S2", "S4", "S8", "S16"]


if __name__ == "__main__":
    main()
