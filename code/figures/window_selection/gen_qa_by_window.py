#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("qa_by_window.svg")
COLORS = {"luna": "#0072B2", "terra": "#D55E00", "sol": "#009E73"}
WINDOWS = [4, 8, 16, 32]


def load_data():
    base = ROOT / "results/formal/gpt56-window-qa-scores-20260721"
    locomo, beam = {}, {}
    for path in base.glob("*-w*/locomo/eval_full.json"):
        tier, window = path.parent.parent.name.split("-w")
        locomo[tier, int(window)] = 100 * json.loads(path.read_text())["overall"]
    for path in base.glob("*-w*/beam-semantic/*/metrics.json"):
        tier, window = path.parents[2].name.split("-w")
        beam[tier, int(window)] = 100 * json.loads(path.read_text())["overall"]["avg_score"]
    w32 = ROOT / "results/formal/gpt56-w32-screening-scores-20260719-r1/beam-semantic"
    for path in w32.glob("*/metrics.json"):
        tier = path.parent.name
        beam[tier, 32] = 100 * json.loads(path.read_text())["overall"]["avg_score"]
    w32_root = ROOT / "results/formal/gpt56-w32-screening-scores-20260719-r1"
    for path in w32_root.glob("*/locomo/eval_full.json"):
        locomo[path.parent.parent.name, 32] = 100 * json.loads(path.read_text())["overall"]
    return locomo, beam


def panel(x0, y0, width, height, ylabel, ymin, ymax, series):
    left, top, right, bottom = 52, 18, 12, 42
    px0, py0 = x0 + left, y0 + top
    pw, ph = width - left - right, height - top - bottom
    parts = []
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = py0 + ph * (1 - tick / 4)
        parts += [f'<line x1="{px0}" y1="{y:.1f}" x2="{px0+pw}" y2="{y:.1f}" stroke="#dddddd"/>',
                  f'<text x="{px0-7}" y="{y+4:.1f}" text-anchor="end">{value:.0f}</text>']
    for i, window in enumerate(WINDOWS):
        x = px0 + pw * i / 3
        parts += [f'<text x="{x:.1f}" y="{py0+ph+22}" text-anchor="middle">{window}</text>']
    parts += [f'<line x1="{px0}" y1="{py0+ph}" x2="{px0+pw}" y2="{py0+ph}" stroke="#333"/>',
              f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0+ph}" stroke="#333"/>',
              f'<text x="{px0+pw/2:.1f}" y="{y0+height-5}" text-anchor="middle">Writer window (turns)</text>',
              f'<text transform="translate({x0+14},{py0+ph/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>']
    for name, values in series.items():
        points = []
        for i, window in enumerate(WINDOWS):
            if window not in values:
                continue
            x = px0 + pw * i / 3
            y = py0 + ph * (ymax - values[window]) / (ymax - ymin)
            points.append((x, y))
        dash = "" if name != "terra" else ' stroke-dasharray="6 3"'
        parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2.2"{} />'.format(
            " ".join(f"{x:.1f},{y:.1f}" for x, y in points), COLORS[name], dash))
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.7" fill="white" stroke="{COLORS[name]}" stroke-width="2"/>')
    return "".join(parts)


def main():
    locomo, beam = load_data()
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 350">',
           '<style>text{font-family:Times New Roman,serif;font-size:13px;fill:#222}</style>',
           panel(5, 38, 440, 300, "LoCoMo judge accuracy (%)", 87, 94,
                 {t: {w: locomo[t, w] for w in WINDOWS} for t in ("luna", "terra", "sol")}),
           panel(455, 38, 440, 300, "BEAM rubric mean (%)", 48, 64,
                 {t: {w: beam[t, w] for w in WINDOWS} for t in ("terra", "sol")})]
    for i, name in enumerate(("Luna", "Terra", "Sol")):
        x = 265 + i * 110
        dash = " stroke-dasharray=\"6 3\"" if name.lower() == "terra" else ""
        svg += [f'<line x1="{x}" y1="18" x2="{x+25}" y2="18" stroke="{COLORS[name.lower()]}" stroke-width="2.2"{dash}/>',
                f'<text x="{x+31}" y="22">{name}</text>']
    svg.append('</svg>')
    OUT.write_text("".join(svg))
    assert all((tier, window) in locomo for tier in ("luna", "terra", "sol") for window in WINDOWS)
    assert all((tier, window) in beam for tier in ("terra", "sol") for window in WINDOWS)


if __name__ == "__main__":
    main()
