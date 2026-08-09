#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/gpt56-chunk-curve/analysis/aggregate_by_window.csv"
OUT = Path(__file__).with_name("memory_granularity.svg")


def panel(x0, label, values, ymin, ymax, color, suffix=""):
    y0, width, height = 45, 300, 220
    xs = [x0 + i * width / 3 for i in range(4)]
    y = lambda value: y0 + height * (ymax - value) / (ymax - ymin)
    parts = [f'<text x="{x0-45}" y="25">{label}</text>']
    for tick in range(5):
        value = ymin + tick * (ymax - ymin) / 4
        py = y(value)
        parts += [f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+width}" y2="{py:.1f}" stroke="#ddd"/>',
                  f'<text x="{x0-9}" y="{py+5:.1f}" text-anchor="end">{value:.1f}{suffix}</text>']
    parts += [f'<line x1="{x0}" y1="{y0+height}" x2="{x0+width}" y2="{y0+height}" stroke="#333"/>',
              f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+height}" stroke="#333"/>']
    for i, (window, value) in enumerate(zip((4, 8, 16, 32), values)):
        x, py = xs[i], y(value)
        parts += [f'<text x="{x:.1f}" y="{y0+height+25}" text-anchor="middle">W{window}</text>',
                  f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="white" stroke="{color}" stroke-width="2.4"/>']
    points = " ".join(f"{x:.1f},{y(value):.1f}" for x, value in zip(xs, values))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8"/>')
    parts += [f'<text x="{xs[0]:.1f}" y="{y(values[0])-12:.1f}" text-anchor="middle" fill="{color}">{values[0]:.2f}{suffix}</text>',
              f'<text x="{xs[2]:.1f}" y="{y(values[2])-12:.1f}" text-anchor="middle" fill="{color}">{values[2]:.2f}{suffix}</text>']
    return "".join(parts)


def main():
    with SOURCE.open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    assert [int(row["write_turns"]) for row in rows] == [4, 8, 16, 32]
    entries = [row["events_per_message"] for row in rows]
    coverage = [100 * row["final_source_coverage"] for row in rows]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 320">',
           '<style>text{font-family:Times New Roman,serif;font-size:14px;fill:#222}</style>',
           panel(70, "Entries per source message", entries, 3, 7, "#0072B2"),
           panel(485, "Final source coverage", coverage, 86, 94, "#D55E00", "%"),
           '</svg>']
    OUT.write_text("".join(svg))


if __name__ == "__main__":
    main()
