# v10 写入间隔阶段性结果

日期：2026-07-16。

## 受控范围

- LoCoMo sample 0、session 1，共 18 条消息。
- writer：GPT-5.5（thinking off）和 GPT-4o-mini。
- `W={2,6,10,20,session}`；其余 v10 配置完全相同。
- build-only 覆盖指标只统计 gold evidence 全部位于 session 1 的 4 个问题。
- QA 复用已建 memory，retriever/answerer 固定为 GPT-5.5，judge 固定为锁定的
  `scripts/eval_full.py` 中 GPT-4o-mini；只报告 q0、q1、q2、q4。
- session 1 只有 18 条消息，因此本次 `W=20` 与 `W=session` 是相同处理条件，
  两次运行之间的差异属于重复运行波动。

## 结果

| Writer | W | Evidence-all | QA | Build calls | Extract+verify | Maintenance | Input tokens | Output tokens | Topics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 | 2 | 4/4 | 未运行 | 43 | 18 | 25 | 45,678 | 2,639 | 14 |
| GPT-5.5 | 6 | 4/4 | 4/4 | 17 | 6 | 11 | 16,211 | 1,066 | 8 |
| GPT-5.5 | 10 | 4/4 | 4/4 | 15 | 4 | 11 | 12,504 | 970 | 8 |
| GPT-5.5 | 20 | 4/4 | 3/4 | 12 | 2 | 10 | 8,853 | 958 | 7 |
| GPT-5.5 | session | 4/4 | 4/4 | 12 | 2 | 10 | 8,700 | 904 | 7 |
| GPT-4o-mini | 2 | 4/4 | 未运行 | 41 | 18 | 23 | 43,014 | 1,827 | 14 |
| GPT-4o-mini | 6 | 4/4 | 4/4 | 29 | 6 | 23 | 19,695 | 1,419 | 16 |
| GPT-4o-mini | 10 | 3/4 | 4/4 | 21 | 4 | 17 | 13,975 | 1,096 | 12 |
| GPT-4o-mini | 20 | 3/4 | 4/4 | 17 | 2 | 15 | 9,368 | 860 | 11 |
| GPT-4o-mini | session | 3/4 | 4/4 | 18 | 2 | 16 | 9,575 | 887 | 13 |

GPT-5.5 的 `W=20` 唯一错误是把 gold `2022` 回答成 `2022-05-08`，而相同
18-turn 处理条件的 `session` 重复运行回答 `2022` 并判对。因此不能把 3/4 与
4/4 的差异归因于窗口。

GPT-4o-mini 在 `W>=10` 时没有保留 `D1:12`，但保留了 `D1:14` 中“去年画了
湖上日出”的事实，四个问题仍全部答对。这说明 `dia_id` 覆盖适合排查完全漏写，
不适合单独决定窗口；最终选择仍必须经过固定下游 QA。

## 当前可支持的判断

1. `W=2` 不进入下一轮候选。相对 `W=6`，GPT-5.5 的 build calls 增加
   153%，input tokens 增加 182%；GPT-4o-mini 的 calls 增加 41%，input tokens
   增加 118%，但本 session 没有 evidence 收益，因此未进入 QA 候选。
2. `W=10` 是当前最稳妥的正式候选。相对 `W=6`，GPT-5.5 的 input tokens
   减少 22.9%，GPT-4o-mini 减少 29.0%，两者 QA 都是 4/4。
3. `W=20` 有更大的成本优势，但当前 session 长度不足以测试 20-turn 与整
   session 的差异；不能据此替换默认值。
4. 模型影响的不只是单次窗口提取。GPT-4o-mini 在 `W=6` 生成 16 个 topic，
   GPT-5.5 生成 8 个，导致 maintenance calls 为 23 对 11。topic 路由碎片化会
   放大 session 后整理成本。
5. 当前 coverage verifier 在每个 chunk 后都触发一次补抽，因此每个窗口实际
   包含一次 distill 和一次 verify。session 整理又占 GPT-5.5 总调用的 58%–83%，
   占 GPT-4o-mini 的 56%–89%。仅增大 `W` 不能消除整理成本。

## Verify on/off 对照

在同一 session、`W=10` 上关闭 `NATIVEMEM_V8_VERIFY`，其余配置不变：

| Writer | Verify | Evidence-all | QA | Events | Build calls | Input tokens |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.5 | on | 4/4 | 4/4 | 17 | 15 | 12,504 |
| GPT-5.5 | off | 4/4 | 4/4 | 17 | 14 | 8,251 |
| GPT-4o-mini | on | 3/4 | 4/4 | 19 | 21 | 13,975 |
| GPT-4o-mini | off | 3/4 | 4/4 | 7 | 9 | 6,203 |

GPT-5.5 的 verify 没有增加最终事件或 QA。GPT-4o-mini 的 verify 增加了 12 条
事件，但新增内容大量是问候、感谢、感叹和肯定表达，例如 `Thanks`、`Wow`、
`Yeah`；本次四题 QA 没有收益。这个单-session 结果支持把 verify 作为独立消融项，
不支持直接声称它改善最终性能，也不足以在全量实验前永久关闭。

## 未完成证据

- Qwen3.6-flash 同代码 smoke test 被 token-plan 的 `429 insufficient_quota`
  阻塞；失败产物均为 0 events，已排除，不能与 GPT 结果比较。
- 尚无 s0+s1 独立重复，不能估计 `writer model × W` 交互项。
- 尚未覆盖长度大于 20 的 session；LoCoMo 最长 session 为 47 条消息。
- v10 默认仍保持 `W=6`，直到 `W=10/20` 通过 s0+s1 重复和全量确认。

## 产物

- GPT-5.5 build analysis：
  `results/v10-write-interval-m0-gpt55-s0-session1-20260716/analysis.json`
- GPT-4o-mini build analysis：
  `results/v10-write-interval-m0-gpt4omini-s0-session1-20260716/analysis.json`
- 固定下游 QA 与锁定 judge：
  `results/v10-write-interval-m0-qa-20260716/`
- Verify on/off：
  `results/v10-verify-ablation-m0-20260716/`
- evaluator SHA256：
  `f8265ae58153b532bdb70a786699a4a711389088bdbc6eb103a943070d4509cd`
