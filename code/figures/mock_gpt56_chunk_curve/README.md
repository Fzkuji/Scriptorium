# GPT-5.6 Writer-window mock figures

本目录仅用于预览论文图表形式。`mock_results.json` 中的数值全部是人为构造的假设值，不是实验结果，不能进入论文结果或项目结果索引。

重新生成：

```bash
python3 figures/mock_gpt56_chunk_curve/gen_fig1_quality_curves.py
python3 figures/mock_gpt56_chunk_curve/gen_fig2_cost_curves.py
python3 figures/mock_gpt56_chunk_curve/gen_fig3_quality_cost_pareto.py
python3 figures/mock_gpt56_chunk_curve/gen_tables.py
python3 figures/mock_gpt56_chunk_curve/generate_preview.py
```

论文候选组件为 3 张图、2 张表。PDF 是论文矢量版本，PNG 用于浏览器预览，`latex_includes.tex` 提供 LaTeX 引用模板。
