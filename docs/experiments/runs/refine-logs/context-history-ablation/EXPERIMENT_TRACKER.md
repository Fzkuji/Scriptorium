# v10 前序上下文实验跟踪

更新时间：2026-07-17

| Milestone | 状态 | 产物 | 备注 |
|---|---|---|---|
| M0 配置冻结 | completed | `EXPERIMENT_PLAN.md` | 7 context × 2 writers |
| M0 runner dry-run | pending | `experiment_manifest.json` | 不调用模型 |
| M0 GPT-5.5 / MiniMax smoke | completed | `results/v10-context-history-ablation-smoke-20260717/` | 14/14 complete，0 failed |
| M0 DeepSeek V4 Flash smoke | completed | `results/v10-context-history-ablation-smoke-deepseek-v4-flash-20260717/` | 7/7 complete，0 failed；OpenRouter exact model；raw20 为成本/时间最佳 |
| M1 s0+s1 build-only | pending | 独立 pilot 目录 | smoke 通过后执行 |
| M2 build 分析 | pending | `analysis.json`、paper table | 不调用模型 |
| M3 固定下游 QA | pending | answer records、locked eval | 复用 M1 memory |
| M4 full confirmation | pending | full result table | 只扩展预选配置 |

固定 evaluator SHA256：`f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`

最终报告要求：质量、build/QA 成本、分阶段成本、wall/LLM time、吞吐、storage、Pareto frontier，以及 100/500/1000-message projection；所有 projection 与实测结果分列。

当前 writer：

- GPT-5.5：subscription proxy，thinking off。
- MiniMax：`minimax/minimax-m2.7`，OpenRouter OpenAI-compatible API。
