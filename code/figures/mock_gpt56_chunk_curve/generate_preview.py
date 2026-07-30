from __future__ import annotations

import html
from pathlib import Path

from paper_plot_style import FIG_DIR, load_data


data = load_data()


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


main_rows = []
previous_model = None
for row in data["formal_results"]:
    classes = ["selected"] if row["configuration"] == "Selected" else []
    if previous_model is not None and row["model"] != previous_model:
        classes.append("group-start")
    badge = (
        '<span class="choice">selected</span>'
        if row["configuration"] == "Selected" else ""
    )
    main_rows.append(f"""
      <tr class="{' '.join(classes)}">
        <td><strong>{html.escape(row['model'])}</strong></td>
        <td><code>{html.escape(str(row['window']))}</code>{badge}</td>
        <td>{fmt(row['locomo'])}</td>
        <td>{fmt(row['longmemeval'])}</td>
        <td>{fmt(row['beam'])}</td>
        <td>{fmt(row['calls_per_100'])}</td>
        <td>{fmt(row['tokens_k_per_100'])}K</td>
        <td>${fmt(row['cost_per_1000'], 2)}</td>
        <td>{fmt(row['latency_min_per_1000'])} min</td>
      </tr>""")
    previous_model = row["model"]

selection_rows = []
for row in data["selection_summary"]:
    selection_rows.append(f"""
      <tr>
        <td><strong>{html.escape(row['model'])}</strong></td>
        <td><code>{html.escape(str(row['selected_window']))}</code></td>
        <td>{fmt(row['mean_quality_delta_pp'])} pp</td>
        <td>{fmt(row['ci95_lower_pp'])} pp</td>
        <td>−{fmt(row['call_reduction_pct'])}%</td>
        <td>−{fmt(row['token_reduction_pct'])}%</td>
        <td>−{fmt(row['cost_reduction_pct'])}%</td>
        <td><span class="pass">Pass</span></td>
      </tr>""")

document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPT-5.6 Writer Window — Mock Paper Results</title>
  <style>
    :root {{
      --ink: #10212b;
      --muted: #5d6c75;
      --line: #dce3e7;
      --paper: #ffffff;
      --canvas: #edf1f2;
      --navy: #102f35;
      --cyan: #0b7182;
      --gold: #c68417;
      --green: #15765c;
      --rose: #b83a61;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--canvas);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      line-height: 1.55;
    }}
    .page {{
      width: min(1180px, calc(100% - 40px));
      margin: 24px auto 64px;
      background: var(--paper);
      box-shadow: 0 18px 54px rgba(19, 43, 50, .12);
      border-radius: 18px;
      overflow: hidden;
    }}
    header {{
      padding: 54px 64px 46px;
      color: white;
      background: var(--navy);
    }}
    .warning {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      padding: 7px 12px;
      border: 1px solid rgba(255,255,255,.35);
      border-radius: 999px;
      color: #ffd8e4;
      background: rgba(184,58,97,.22);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .04em;
    }}
    h1 {{
      max-width: 820px;
      margin: 22px 0 12px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 1.06;
      letter-spacing: -.035em;
      font-weight: 600;
    }}
    header p {{ max-width: 760px; color: #c9d9dc; font-size: 17px; margin: 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-bottom: 1px solid var(--line);
    }}
    .summary article {{ padding: 26px 34px; border-right: 1px solid var(--line); }}
    .summary article:last-child {{ border-right: 0; }}
    .summary strong {{ display: block; color: var(--cyan); font: 600 28px/1 Georgia, serif; margin-bottom: 6px; }}
    .summary span {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 28px 64px 68px; }}
    section {{ padding-top: 38px; }}
    .section-head {{
      display: grid;
      grid-template-columns: 70px 1fr;
      gap: 18px;
      align-items: start;
      margin-bottom: 20px;
    }}
    .number {{
      color: var(--rose);
      font: italic 500 30px/1 Georgia, serif;
      border-top: 2px solid var(--rose);
      padding-top: 8px;
    }}
    h2 {{ margin: 0 0 5px; font: 600 27px/1.2 Georgia, "Times New Roman", serif; }}
    .section-head p {{ color: var(--muted); margin: 0; max-width: 820px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 14px; padding: 24px 20px 16px; background: #fff; }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin: 14px 10px 0; }}
    .insights {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 16px; }}
    .insight {{ padding: 18px 18px 17px; border-left: 3px solid var(--cyan); background: #f5f8f8; border-radius: 4px 12px 12px 4px; }}
    .insight b {{ display: block; margin-bottom: 4px; }}
    .insight span {{ color: var(--muted); font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 13px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; font-size: 13.5px; }}
    th {{ text-align: right; padding: 13px 12px; background: #f1f5f5; color: #46565e; font-size: 12px; letter-spacing: .015em; white-space: nowrap; }}
    th:first-child, th:nth-child(2), td:first-child, td:nth-child(2) {{ text-align: left; }}
    td {{ text-align: right; padding: 12px; border-top: 1px solid #e8edef; white-space: nowrap; }}
    tr.group-start td {{ border-top: 2px solid #bfcbd0; }}
    tr.selected {{ background: #edf8f4; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; color: #23434b; }}
    .choice {{ margin-left: 7px; padding: 2px 6px; border-radius: 999px; color: var(--green); background: #d9efe7; font-size: 10px; font-weight: 700; text-transform: uppercase; }}
    .pass {{ display: inline-block; padding: 3px 9px; border-radius: 999px; background: #d9efe7; color: var(--green); font-weight: 700; }}
    .note {{
      margin-top: 16px;
      padding: 16px 19px;
      border-radius: 10px;
      background: #fff8e9;
      color: #654d1f;
      font-size: 13px;
    }}
    .roles {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
    .role {{ padding: 16px; border-top: 3px solid var(--navy); background: #f5f7f7; min-height: 132px; }}
    .role b {{ display: block; font-family: Georgia, serif; margin-bottom: 8px; }}
    .role p {{ color: var(--muted); font-size: 13px; margin: 0; }}
    footer {{ padding: 22px 64px; background: #f2f5f5; color: var(--muted); font-size: 12px; }}
    @media (max-width: 820px) {{
      .page {{ width: 100%; margin: 0; border-radius: 0; }}
      header, main {{ padding-left: 24px; padding-right: 24px; }}
      .summary, .insights, .roles {{ grid-template-columns: 1fr; }}
      .summary article {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .section-head {{ grid-template-columns: 46px 1fr; }}
    }}
    @media print {{
      body {{ background: white; }}
      .page {{ width: 100%; margin: 0; box-shadow: none; border-radius: 0; }}
      section {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<div class="page">
  <header>
    <div class="warning">SYNTHETIC PREVIEW · 模拟结果，非真实实验</div>
    <h1>Writer Input Granularity:<br>Quality–Cost Trade-off</h1>
    <p>假设实验达到理想但合理的结果时，论文主文建议使用的 3 张图与 2 张表。所有数字仅用于检查叙事、版式和结果是否足以支持论文结论。</p>
  </header>

  <div class="summary">
    <article><strong>3 figures</strong><span>质量曲线、成本曲线、质量—成本关系</span></article>
    <article><strong>2 tables</strong><span>正式 benchmark 结果、参数选择与置信区间</span></article>
    <article><strong>W=6 baseline</strong><span>与当前每 6 条消息保存一次的系统直接比较</span></article>
  </div>

  <main>
    <section>
      <div class="section-head"><div class="number">01</div><div><h2>不同读取长度下的最终性能</h2><p>回答“读取窗口增大后，记忆是否仍能支持正确问答”。空心点表示每个模型根据预注册规则选出的最大窗口。</p></div></div>
      <figure>
        <img src="fig1_quality_curves.png" alt="三个 benchmark 上的模拟质量曲线">
        <figcaption><b>Figure 1.</b> 模拟的完整 benchmark 质量曲线。理想结果表现为：质量先保持稳定，超过模型各自的处理范围后下降。</figcaption>
      </figure>
      <div class="insights">
        <div class="insight"><b>Sol: W*=24</b><span>平均性能仅下降 0.4 pp，允许最大的固定读取窗口。</span></div>
        <div class="insight"><b>Terra: W*=16</b><span>在 16 条消息处满足性能与成本的共同约束。</span></div>
        <div class="insight"><b>Luna: W*=12</b><span>更大窗口开始出现跨 benchmark 的一致退化。</span></div>
      </div>
    </section>

    <section>
      <div class="section-head"><div class="number">02</div><div><h2>构建成本随读取长度的变化</h2><p>同时报告调用数、输入 token 和 API 参考成本。Writer、Verify、Tidy 全部计入，不能只统计第一次提炼。</p></div></div>
      <figure>
        <img src="fig2_cost_curves.png" alt="模拟的构建成本曲线">
        <figcaption><b>Figure 2.</b> 模拟的记忆构建成本。三个指标都随窗口增大下降，但不同模型的绝对美元成本不同。</figcaption>
      </figure>
    </section>

    <section>
      <div class="section-head"><div class="number">03</div><div><h2>质量—成本关系与参数选择</h2><p>使用非劣效边界决定参数，而不是直接选择最便宜的配置。横轴越小表示构建 token 越少，纵轴越高表示性能保持越好。</p></div></div>
      <figure style="max-width: 680px; margin: 0 auto;">
        <img src="fig3_quality_cost_pareto.png" alt="模拟的质量成本关系">
        <figcaption><b>Figure 3.</b> 模拟的质量—成本关系。虚线是预注册的 −1.5 pp 非劣效边界，空心点是最终选定配置。</figcaption>
      </figure>
    </section>

    <section>
      <div class="section-head"><div class="number">T1</div><div><h2>完整 benchmark 主结果</h2><p>每个模型只保留当前默认 W=6、选定 W* 和整个 session 三种配置。表内性能来自冻结参数后的正式测试。</p></div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Writer</th><th>W</th><th>LoCoMo ↑</th><th>LongMemEval-S ↑</th><th>BEAM ↑</th><th>Calls/100 ↓</th><th>Tokens/100 ↓</th><th>$/1K ↓</th><th>Latency/1K ↓</th></tr></thead>
          <tbody>{''.join(main_rows)}</tbody>
        </table>
      </div>
      <div class="note">表中美元值是公开 API 单价换算的参考成本，不代表订阅额度的实际收费。真实论文还需要同时报告 provider 返回的 token 与额度消耗。</div>
    </section>

    <section>
      <div class="section-head"><div class="number">T2</div><div><h2>预注册选择规则是否通过</h2><p>相对 W=6，质量差值的 95% CI 下界必须高于 −1.5 pp，同时构建成本至少下降 20%。</p></div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Writer</th><th>Selected W*</th><th>Mean ΔScore ↑</th><th>95% CI lower ↑</th><th>Calls ↓</th><th>Tokens ↓</th><th>Cost ↓</th><th>Decision</th></tr></thead>
          <tbody>{''.join(selection_rows)}</tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head"><div class="number">Use</div><div><h2>五个结果组件各自证明什么</h2><p>主文不需要展示全部 162 个配置，完整曲线和中间统计可以放附录。</p></div></div>
      <div class="roles">
        <div class="role"><b>Figure 1</b><p>证明窗口大小确实影响最终记忆质量，并展示不同模型的临界位置。</p></div>
        <div class="role"><b>Figure 2</b><p>证明成本降低来自调用数和 token 的实际减少。</p></div>
        <div class="role"><b>Figure 3</b><p>展示性能约束下的有效参数，而不是只比较最低成本。</p></div>
        <div class="role"><b>Table 1</b><p>给出冻结参数后的三个完整 benchmark 正式结果。</p></div>
        <div class="role"><b>Table 2</b><p>给出置信区间、选择标准和模型能力差异。</p></div>
      </div>
    </section>
  </main>
  <footer>Generated from mock_results.json · Values are synthetic placeholders · 2026-07-18</footer>
</div>
</body>
</html>
"""

(FIG_DIR / "preview.html").write_text(document, encoding="utf-8")
