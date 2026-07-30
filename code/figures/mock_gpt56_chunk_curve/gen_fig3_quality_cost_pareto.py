from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from paper_plot_style import load_data, save_figure, style_axis


data = load_data()
windows = data["windows"]
baseline_index = windows.index("6")

fig, ax = plt.subplots(figsize=(5.15, 3.45))
annotation_offsets = {
    "Sol": (-10, 10),
    "Terra": (3, -18),
    "Luna": (10, 9),
}
for model_name, model in data["models"].items():
    quality = np.mean(
        np.asarray([model["quality"][name] for name in data["benchmarks"]]),
        axis=0,
    )
    quality_delta = quality - quality[baseline_index]
    tokens = np.asarray(
        model["cost"]["input_tokens_k_per_100_messages"], dtype=float
    )
    relative_cost = 100.0 * tokens / tokens[baseline_index]
    ax.plot(
        relative_cost, quality_delta, color=model["color"],
        marker=model["marker"], label=model_name,
    )
    selected_index = windows.index(model["selected_window"])
    ax.scatter(
        [relative_cost[selected_index]], [quality_delta[selected_index]],
        s=76, facecolor="white", edgecolor=model["color"], linewidth=1.7,
        marker=model["marker"], zorder=6,
    )
    ax.annotate(
        f"$W^*={model['selected_window']}$",
        (relative_cost[selected_index], quality_delta[selected_index]),
        xytext=annotation_offsets[model_name], textcoords="offset points",
        color=model["color"], fontsize=8.2,
    )
    ax.scatter(
        [relative_cost[baseline_index]], [quality_delta[baseline_index]],
        s=48, facecolor=model["color"], edgecolor="white", linewidth=0.8,
        marker=model["marker"], zorder=5,
    )

ax.axhline(-1.5, color="#8C2D35", linestyle="--", linewidth=1.1)
ax.text(
    141, -1.34, "non-inferiority margin: -1.5 pp",
    color="#8C2D35", fontsize=8.2, ha="right",
)
ax.set_xlabel("Relative build input tokens (% of $W=6$)")
ax.set_ylabel("Mean benchmark score change vs. $W=6$ (pp)")
ax.set_xlim(25, 145)
ax.set_ylim(-7.8, 0.8)
ax.legend(frameon=False, loc="lower right")
style_axis(ax)
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.98)
save_figure(fig, "fig3_quality_cost_pareto")
