#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/gpt56-chunk-curve/analysis/aggregate_by_window.csv"
OUT = Path(__file__).with_name("build_statistics.svg")
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def load_rows():
    with SOURCE.open() as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def panel(x0, y0, width, height, ylabel, ymin, ymax, series, percent=False):
    left, top, right, bottom = 53, 15, 12, 38
    px0, py0 = x0 + left, y0 + top
    pw, ph = width - left - right, height - top - bottom
    parts = []
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = py0 + ph * (1 - tick / 4)
        label = f"{value:.0f}%" if percent else f"{value:.1f}"
        parts += [f'<line x1="{px0}" y1="{y:.1f}" x2="{px0+pw}" y2="{y:.1f}" stroke="#dddddd"/>',
                  f'<text x="{px0-7}" y="{y+4:.1f}" text-anchor="end">{label}</text>']
    for i, window in enumerate((4, 8, 16, 32)):
        x = px0 + pw * i / 3
        parts.append(f'<text x="{x:.1f}" y="{py0+ph+20}" text-anchor="middle">{window}</text>')
    parts += [f'<line x1="{px0}" y1="{py0+ph}" x2="{px0+pw}" y2="{py0+ph}" stroke="#333"/>',
              f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0+ph}" stroke="#333"/>',
              f'<text x="{px0+pw/2:.1f}" y="{y0+height-3}" text-anchor="middle">Writer window</text>',
              f'<text transform="translate({x0+13},{py0+ph/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>']
    for index, (name, values) in enumerate(series.items()):
        points = []
        for i, value in enumerate(values):
            x = px0 + pw * i / 3
            y = py0 + ph * (ymax - value) / (ymax - ymin)
            points.append((x, y))
        dash = ('', ' stroke-dasharray="6 3"', ' stroke-dasharray="2 3"')[index]
        parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2"{} />'.format(
            " ".join(f"{x:.1f},{y:.1f}" for x, y in points), COLORS[index], dash))
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="white" stroke="{COLORS[index]}" stroke-width="1.8"/>')
    return "".join(parts)


def legend(x, y, names):
    parts = []
    for i, name in enumerate(names):
        dx = x + i * 112
        dash = ('', ' stroke-dasharray="6 3"', ' stroke-dasharray="2 3"')[i]
        parts += [f'<line x1="{dx}" y1="{y}" x2="{dx+22}" y2="{y}" stroke="{COLORS[i]}" stroke-width="2"{dash}/>',
                  f'<text x="{dx+27}" y="{y+4}">{name}</text>']
    return "".join(parts)


def main():
    rows = load_rows()
    base = rows[0]
    normalized = {
        "Calls": [100 * r["calls_per_100_messages"] / base["calls_per_100_messages"] for r in rows],
        "Input tokens": [100 * r["input_tokens_per_1000_source_tokens"] / base["input_tokens_per_1000_source_tokens"] for r in rows],
        "Wall time": [100 * r["wall_minutes_per_100_messages"] / base["wall_minutes_per_100_messages"] for r in rows],
    }
    coverage = {
        "First pass": [100 * r["first_pass_source_coverage"] for r in rows],
        "After verify": [100 * r["final_source_coverage"] for r in rows],
        "Verify gain": [100 * r["verify_source_coverage_gain"] for r in rows],
    }
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 610">',
           '<style>text{font-family:Times New Roman,serif;font-size:13px;fill:#222}</style>',
           legend(260, 20, normalized.keys()),
           panel(5, 35, 440, 265, "Relative construction cost", 0, 110, normalized, True),
           legend(555, 20, coverage.keys()),
           panel(455, 35, 440, 265, "Source coverage (%)", 0, 100, coverage, True),
           panel(5, 330, 440, 265, "Entries per source message", 0, 7, {"Entries/message": [r["events_per_message"] for r in rows]}),
           panel(455, 330, 440, 265, "Abstract/source byte ratio", 0, 2.5, {"Byte ratio": [r["abstract_to_source_byte_ratio"] for r in rows]}),
           '</svg>']
    OUT.write_text("".join(svg))
    assert [int(r["write_turns"]) for r in rows] == [4, 8, 16, 32]


if __name__ == "__main__":
    main()
