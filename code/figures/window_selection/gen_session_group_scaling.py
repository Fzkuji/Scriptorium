#!/usr/bin/env python3
"""Plot window and session-group QA on one actual group-size axis."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WINDOW_STATS = ROOT / "experiments/gpt56-chunk-curve/analysis/aggregate_by_benchmark_window.csv"
SESSION_RESULTS = ROOT / "results/formal/gpt56-session-groups/analysis/session-group-results.json"
OUT = Path(__file__).with_name("session_group_scaling.svg")
COLORS = {"luna": "#0072B2", "terra": "#D55E00", "sol": "#009E73"}
LABELS = {"luna": "Luna", "terra": "Terra", "sol": "Sol"}
SPECS = [
    ("locomo", "LoCoMo (n=314 per point)", "Judge accuracy (%)", 70, 95),
    ("longmemeval-s", "LongMemEval-S (n=2 per point)", "Correctness (%)", 0, 100),
    ("beam-100k", "BEAM screening (n=40 per point)", "Rubric mean (%)", 45, 70),
]
TIERS = ("luna", "terra", "sol")
WINDOWS = (4, 8, 16, 32)
X_MIN, X_MAX = 3.2, 400


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_window_quality() -> dict[tuple[str, str, int], float]:
    quality: dict[tuple[str, str, int], float] = {}
    base = ROOT / "results/formal/gpt56-window-qa-scores-20260721"
    for path in base.glob("*-w*/locomo/eval_full.json"):
        tier, window = path.parent.parent.name.split("-w")
        quality["locomo", tier, int(window)] = 100 * json.loads(path.read_text())["overall"]
    for path in base.glob("*-w*/longmemeval-s/hypotheses.jsonl.eval-results-gpt-4o-mini"):
        tier, window = path.parent.parent.name.split("-w")
        rows = read_jsonl(path)
        quality["longmemeval-s", tier, int(window)] = 100 * sum(
            row["autoeval_label"]["label"] for row in rows
        ) / len(rows)
    for path in base.glob("*-w*/beam-semantic/*/metrics.json"):
        tier, window = path.parents[2].name.split("-w")
        quality["beam-100k", tier, int(window)] = 100 * json.loads(path.read_text())["overall"]["avg_score"]

    w32 = ROOT / "results/formal/gpt56-w32-screening-scores-20260719-r1"
    for tier in ("terra", "sol"):
        quality["locomo", tier, 32] = 100 * json.loads(
            (w32 / tier / "locomo/eval_full.json").read_text()
        )["overall"]
        rows = read_jsonl(w32 / tier / "longmemeval-s/hypotheses.jsonl.eval-results-gpt-4o-mini")
        quality["longmemeval-s", tier, 32] = 100 * sum(
            row["autoeval_label"]["label"] for row in rows
        ) / len(rows)
        quality["beam-100k", tier, 32] = 100 * json.loads(
            (w32 / "beam-semantic" / tier / "metrics.json").read_text()
        )["overall"]["avg_score"]
    return quality


def load_actual_sizes(cells: list[dict]) -> tuple[dict, dict]:
    window_sizes = {}
    with WINDOW_STATS.open() as handle:
        for row in csv.DictReader(handle):
            window_sizes[row["benchmark"], int(row["write_turns"])] = (
                float(row["messages"]) / float(row["writer_calls"])
            )
    session_sizes = {}
    for benchmark, *_ in SPECS:
        sizes = sorted({cell["nominal_session_group_size"] for cell in cells if cell["benchmark"] == benchmark})
        for size in sizes:
            subset = [cell for cell in cells if cell["benchmark"] == benchmark and cell["nominal_session_group_size"] == size]
            session_sizes[benchmark, size] = (
                sum(cell["construction"]["messages"] for cell in subset)
                / sum(cell["writer_calls"] for cell in subset)
            )
    return window_sizes, session_sizes


def diamond(x: float, y: float, color: str) -> str:
    return f'<path d="M{x:.1f},{y-4:.1f} L{x+4:.1f},{y:.1f} L{x:.1f},{y+4:.1f} L{x-4:.1f},{y:.1f} Z" fill="white" stroke="{color}" stroke-width="2"/>'


def main() -> None:
    cells = json.loads(SESSION_RESULTS.read_text(encoding="utf-8"))["cells"]
    window_quality = load_window_quality()
    window_sizes, session_sizes = load_actual_sizes(cells)
    session_quality = {
        (cell["benchmark"], cell["tier"], cell["nominal_session_group_size"]): 100 * cell["quality"]["score"]
        for cell in cells
        if isinstance(cell["quality"].get("score"), (int, float))
    }

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 400">',
        '<style>text{font-family:Times New Roman,serif;font-size:12px;fill:#222}'
        '.small{font-size:10px}.title{font-size:14px;font-weight:600}</style>',
    ]
    panel_width, plot_width, plot_height, top = 375, 292, 210, 45
    log_min, log_span = math.log2(X_MIN), math.log2(X_MAX) - math.log2(X_MIN)

    for panel, (benchmark, title, ylabel, ymin, ymax) in enumerate(SPECS):
        left = 52 + panel * panel_width
        x = lambda value: left + plot_width * (math.log2(value) - log_min) / log_span
        y = lambda value: top + plot_height * (ymax - value) / (ymax - ymin)
        svg.append(f'<text class="title" x="{left + plot_width / 2:.1f}" y="20" text-anchor="middle">{title}</text>')

        for tick_index in range(5):
            value = ymin + (ymax - ymin) * tick_index / 4
            py = y(value)
            svg.extend([
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" stroke="#ddd"/>',
                f'<text class="small" x="{left - 7}" y="{py + 4:.1f}" text-anchor="end">{value:.0f}</text>',
            ])
        for tick in (4, 8, 16, 32, 64, 128, 256):
            px = x(tick)
            svg.extend([
                f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_height}" stroke="#eee"/>',
                f'<text class="small" x="{px:.1f}" y="{top + plot_height + 16}" text-anchor="middle">{tick}</text>',
            ])
        svg.extend([
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
            f'<text class="small" transform="translate({left - 37},{top + plot_height / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>',
        ])

        configurations = [
            (f"W{window}", window_sizes[benchmark, window]) for window in WINDOWS
        ] + [
            (f"S{size}", session_sizes[benchmark, size])
            for size in sorted(size for b, size in session_sizes if b == benchmark)
        ]
        for index, (label, actual) in enumerate(configurations):
            px = x(actual)
            row_y = top + plot_height + 32 + 24 * (index % 2)
            svg.extend([
                f'<line x1="{px:.1f}" y1="{top + plot_height}" x2="{px:.1f}" y2="{row_y - 13}" stroke="#999" stroke-width=".7"/>',
                f'<text class="small" x="{px:.1f}" y="{row_y}" text-anchor="middle">{label}</text>',
                f'<text class="small" x="{px:.1f}" y="{row_y + 11}" text-anchor="middle">{actual:.1f}</text>',
            ])

        for tier in TIERS:
            w_points = [
                (window_sizes[benchmark, window], window_quality[benchmark, tier, window])
                for window in WINDOWS if (benchmark, tier, window) in window_quality
            ]
            s_points = sorted([
                (session_sizes[benchmark, size], score)
                for (b, t, size), score in session_quality.items() if (b, t) == (benchmark, tier)
            ])
            for points in (w_points, s_points):
                if points:
                    coords = " ".join(f"{x(px):.1f},{y(py):.1f}" for px, py in points)
                    svg.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[tier]}" stroke-width="2.1"/>')
            if w_points and s_points:
                svg.append(
                    f'<line x1="{x(w_points[-1][0]):.1f}" y1="{y(w_points[-1][1]):.1f}" '
                    f'x2="{x(s_points[0][0]):.1f}" y2="{y(s_points[0][1]):.1f}" '
                    f'stroke="{COLORS[tier]}" stroke-width="1.5"/>'
                )
            for px, py in w_points:
                svg.append(f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="3.7" fill="white" stroke="{COLORS[tier]}" stroke-width="2"/>')
            for px, py in s_points:
                svg.append(diamond(x(px), y(py), COLORS[tier]))

        svg.append(
            f'<text class="small" x="{left + plot_width / 2:.1f}" y="{top + plot_height + 76}" text-anchor="middle">Actual source messages per writer call (log₂ scale)</text>'
        )

    legend_y = 382
    for index, tier in enumerate(TIERS):
        lx = 330 + index * 92
        svg.extend([
            f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 20}" y2="{legend_y}" stroke="{COLORS[tier]}" stroke-width="2.1"/>',
            f'<text class="small" x="{lx + 26}" y="{legend_y + 4}">{LABELS[tier]}</text>',
        ])
    svg.extend([
        f'<circle cx="{650}" cy="{legend_y}" r="3.7" fill="white" stroke="#444" stroke-width="1.7"/>',
        f'<text class="small" x="660" y="{legend_y + 4}">Writer window</text>',
        diamond(755, legend_y, "#444"),
        f'<text class="small" x="765" y="{legend_y + 4}">Session group</text>',
        '</svg>',
    ])
    OUT.write_text("".join(svg), encoding="utf-8")

    assert window_sizes["locomo", 32] == 4068 / 192
    assert round(session_sizes["beam-100k", 1], 2) == 64.67
    assert len(window_quality) == 36  # three tiers × four windows × three benchmarks
    assert len(session_quality) == 30  # 12 LoCoMo + 12 LongMemEval-S + 6 BEAM


if __name__ == "__main__":
    main()
