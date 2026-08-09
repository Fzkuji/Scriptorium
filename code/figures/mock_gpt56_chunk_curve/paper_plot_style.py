from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR / "mock_results.json"

mpl.rcParams.update({
    "font.size": 9.5,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("status") != "synthetic_hypothetical":
        raise ValueError("mock figure scripts require explicitly synthetic data")
    return data


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D7DEE5", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", facecolor="white")
    plt.close(fig)
