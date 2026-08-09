from __future__ import annotations

from pathlib import Path

from paper_plot_style import FIG_DIR, load_data


data = load_data()


def bold_if_selected(row: dict, value: str) -> str:
    if row["configuration"] == "Selected":
        return rf"\textbf{{{value}}}"
    return value


main_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Hypothetical full-benchmark results after selecting the Writer window on a disjoint development subset. All values are synthetic placeholders. Calls, tokens, cost, and latency cover memory construction only.}",
    r"\label{tab:mock_chunk_main}",
    r"\small",
    r"\begin{tabular}{llrrrrrrr}",
    r"\toprule",
    r"Writer & $W$ & LoCoMo & LongMemEval-S & BEAM & Calls/100 & Tokens/100 (K) & \$/1K & Min/1K \\",
    r"\midrule",
]

previous_model = None
for row in data["formal_results"]:
    if previous_model is not None and row["model"] != previous_model:
        main_lines.append(r"\midrule")
    values = [
        row["model"],
        str(row["window"]),
        f"{row['locomo']:.1f}",
        f"{row['longmemeval']:.1f}",
        f"{row['beam']:.1f}",
        f"{row['calls_per_100']:.1f}",
        f"{row['tokens_k_per_100']:.1f}",
        f"{row['cost_per_1000']:.2f}",
        f"{row['latency_min_per_1000']:.1f}",
    ]
    values = [bold_if_selected(row, value) for value in values]
    main_lines.append(" & ".join(values) + r" \\")
    previous_model = row["model"]

main_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
    "",
])
(FIG_DIR / "TABLE_main_results.tex").write_text(
    "\n".join(main_lines), encoding="utf-8"
)

selection_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Hypothetical model-specific window selection. A configuration passes when the lower 95\% confidence bound exceeds $-1.5$ points and construction cost falls by at least 20\%.}",
    r"\label{tab:mock_chunk_selection}",
    r"\small",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Writer & $W^*$ & $\Delta$Score & CI lower & Calls $\downarrow$ & Tokens $\downarrow$ & Cost $\downarrow$ \\",
    r"\midrule",
]
for row in data["selection_summary"]:
    selection_lines.append(
        f"{row['model']} & {row['selected_window']} & "
        f"{row['mean_quality_delta_pp']:.1f} & {row['ci95_lower_pp']:.1f} & "
        f"{row['call_reduction_pct']:.1f}\\% & "
        f"{row['token_reduction_pct']:.1f}\\% & "
        f"{row['cost_reduction_pct']:.1f}\\% \\\\"
    )
selection_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
])
(FIG_DIR / "TABLE_selection_summary.tex").write_text(
    "\n".join(selection_lines), encoding="utf-8"
)

latex_includes = r"""% Synthetic preview only. Replace mock_results.json with measured results.
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/mock_gpt56_chunk_curve/fig1_quality_curves.pdf}
  \caption{Hypothetical effect of Writer input granularity on downstream memory quality. Open markers identify the selected model-specific window. Shaded regions denote illustrative 95\% confidence intervals.}
  \label{fig:mock_quality_curves}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/mock_gpt56_chunk_curve/fig2_cost_curves.pdf}
  \caption{Hypothetical memory-construction cost as the Writer window increases. Costs include Writer, Verify, and Tidy calls. Dollar values use API-reference pricing and are not subscription charges.}
  \label{fig:mock_cost_curves}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.49\textwidth]{figures/mock_gpt56_chunk_curve/fig3_quality_cost_pareto.pdf}
  \caption{Hypothetical quality--cost trade-off. Each open marker is the largest model-specific window satisfying the pre-registered non-inferiority criterion.}
  \label{fig:mock_quality_cost}
\end{figure}

\input{figures/mock_gpt56_chunk_curve/TABLE_main_results.tex}
\input{figures/mock_gpt56_chunk_curve/TABLE_selection_summary.tex}
"""
(FIG_DIR / "latex_includes.tex").write_text(latex_includes, encoding="utf-8")
