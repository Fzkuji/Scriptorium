from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from paper_plot_style import load_data, save_figure, style_axis


data = load_data()
windows = data["windows"]
x = np.arange(len(windows))
tick_labels = ["sess." if value == "session" else value for value in windows]
benchmarks = list(data["benchmarks"])

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))
for panel, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
    ci = data["benchmarks"][benchmark]["ci95_half_width"]
    all_values = []
    for model_name, model in data["models"].items():
        values = np.asarray(model["quality"][benchmark], dtype=float)
        all_values.extend(values.tolist())
        ax.fill_between(
            x, values - ci, values + ci,
            color=model["color"], alpha=0.055, linewidth=0,
        )
        ax.plot(
            x, values, color=model["color"], marker=model["marker"],
            label=model_name,
        )
        selected_index = windows.index(model["selected_window"])
        ax.scatter(
            [selected_index], [values[selected_index]], s=58,
            facecolor="white", edgecolor=model["color"], linewidth=1.5,
            marker=model["marker"], zorder=5,
        )
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Writer window $W$")
    ax.set_ylabel(data["benchmarks"][benchmark]["metric"])
    ax.set_ylim(min(all_values) - 1.5, max(all_values) + 1.3)
    ax.text(
        0.02, 0.96, f"({chr(97 + panel)}) {benchmark}",
        transform=ax.transAxes, va="top", ha="left", fontweight="semibold",
    )
    style_axis(ax)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", ncol=3, frameon=False,
    bbox_to_anchor=(0.5, 1.03), handlelength=2.1,
)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.19, top=0.86, wspace=0.30)
save_figure(fig, "fig1_quality_curves")
