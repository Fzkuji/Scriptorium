# 实验结果分析

本次可分析的完整构建矩阵包含 72 个 run：Luna、Terra、Sol 三个 builder 档位，W=4、8、16、32 四个 writer window，以及来自 LoCoMo、LongMemEval-S 和 BEAM-100K 的 6 个 history。总构建量为 25,901 次模型调用、65.97M input tokens、7.61M output tokens 和 52.02 小时累计 wall time。

下表按 window 聚合全部 18 个配对构建。`Source coverage` 表示最终 abstract memory 引用的唯一 source turn 比例；`Gold-all` 仅在具有 gold evidence 的 LoCoMo 和 LongMemEval-S 上计算；`Abs./source` 是 abstract memory 与 source text 的字节比。

| W | Calls / 100 msg | Input / 1K source tok. | Output / 1K source tok. | Wall min. / 100 msg | Source coverage | Gold-all | Abs./source |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 109.60 | 14,388.6 | 1,551.4 | 12.37 | 92.16% | 99.84% | 2.36 |
| 8 | 79.79 | 10,713.0 | 1,269.8 | 9.73 | 90.64% | 99.52% | 2.05 |
| 16 | 60.37 | 8,556.8 | 1,025.2 | 7.63 | 88.85% | 99.62% | 1.75 |
| 32 | 55.10 | 7,897.0 | 946.4 | 7.00 | 88.92% | 99.45% | 1.62 |

`Gold-all` 是构建诊断指标，不是 benchmark accuracy；其中 LoCoMo 使用 turn-level evidence，LongMemEval-S 使用 answer-session coverage，两者的证据粒度不同。论文表格应按数据集分别报告，不能把这一聚合值当成统一评测指标。

增大 window 对构建成本的影响具有一致方向。从 W4 到 W8，18/18 个配对 run 的 calls、input tokens 和 wall time 都下降；从 W8 到 W16，除一个配对 run 外，input tokens 和 wall time也都下降。W16 到 W32 的平均变化为 calls -9.30%、input tokens -9.48%、output tokens -10.22%、wall time -11.02%。前三次翻倍的平均 wall-time 降幅依次为 22.23%、20.61% 和 11.02%，说明 W32 仍节省成本，但增量收益已经减小。

Window 增大主要减少了 memory 的全量覆盖和体积，没有相同比例地损失 gold evidence。W4 到 W32 的最终 source coverage 从 92.16% 降至 88.92%，abstract/source ratio 从 2.36 降至 1.62；同一范围内 Gold-all 只从 99.84% 变为 99.45%。W16 与 W32 的配对差异更小：source coverage 平均变化 +0.07 percentage points，95% bootstrap interval 为 [-1.34, 1.62]；Gold-all 平均变化 -0.17 points，interval 为 [-0.57, 0.09]。这些指标支持将 W32 作为后续正式实验的默认构建设置，但不证明 W32 的 downstream accuracy 优于 W16。

Verify 对大 window 的覆盖恢复量更高。First-pass source coverage 在 W4、W8、W16、W32 下分别为 83.19%、79.79%、74.35% 和 71.94%，verify 增加的 coverage 分别为 8.97、10.85、14.50 和 16.98 percentage points。最终 reference precision 在全部 window 下均为 100%，表示生成的 source IDs 在语法和索引层面有效；该指标不判断 memory entry 的语义正确性。

不同数据集的曲线不相同。LoCoMo 的最终 source coverage 从 W4 的 84.31% 降到 W32 的 74.26%，但 Gold-all 仍从 99.69% 保持到 98.90%，说明较大 window 主要减少了未被当前 gold annotations 使用的 turns。LongMemEval-S 的 source coverage 在 92.08% 到 93.69% 之间，所选两个问题的 gold answer sessions 在所有设置中均被覆盖。BEAM 的 source coverage 在 97.91% 到 99.66% 之间，但数据集没有 turn-level gold evidence，不能计算相同的 gold-coverage 指标。

模型档位体现的是构建规模与成本差异，不是单调的语义质量排序。在四个 window 上平均，Luna 的 source coverage 为 94.60%，但 abstract/source ratio 为 2.30，input 为 11,735 tokens / 1K source tokens，wall time 为 10.64 min. / 100 messages。Terra 对应为 86.93%、1.54、9,176 和 7.26；Sol 为 88.90%、1.98、10,255 和 9.66。Luna 更高的覆盖同时伴随更大的 memory 和更高成本，且没有 Luna QA，因此不能将更高覆盖写成更高回答质量。

在 W32 下，Sol 相比 Terra 的 source coverage 平均高 2.06 percentage points，但 calls 高 14.92%、input tokens 高 13.38%、wall time 高 37.15%，abstract/source ratio 高 40.39%。已有 QA 只支持描述性比较：LoCoMo 两个 conversation 上 Terra 为 89.81%，Sol 为 91.08%；BEAM 两个 conversation 的 rubric-nugget mean 分别为 55.77% 和 55.52%，question pass rate 为 65% 和 60%；LongMemEval-S 两个问题均为 2/2。Sol 没有在三个 screening benchmark 上呈现一致优势。若后续实验需要固定一个低成本 builder，Terra + W32 具有当前最直接的成本与筛选质量依据；若论文最终采用 Sol，需要将理由限定为模型档位选择，而不是现有结果已经证明其更优。

对预先列出的六项主张，当前证据判断如下：

| 主张 | 判断 | 证据边界 |
|---|---|---|
| Window 增大稳定降低构建成本，且收益递减 | Supported | 72 个构建；成本指标，不涉及 QA |
| W32 可作为后续正式实验的默认构建设置 | Weakly supported | W16→W32 成本继续下降且构建 coverage 基本稳定；缺少 W16/W32 paired QA |
| Verify 对大 window 的遗漏补偿更大 | Supported | 支持 coverage gain 随 W 增长；不支持语义质量的因果结论 |
| Terra 比 Sol 更高效，Sol 没有一致 QA 优势 | Supported descriptively | 仅 W32 的两个 conversation / 两个问题 screening |
| Luna 的高 coverage 不能解释为更高 QA | Supported | 没有 Luna QA，且其 memory 更大、成本更高 |
| 已经确定 W4/8/16/32 的 downstream QA 排序 | Unsupported | W4、W8、W16 没有 QA |

当前证据可以写入论文的英文表述：

> Across 72 paired construction runs, increasing the writer window from 4 to 32 reduced model calls, input tokens, output tokens, and wall-clock time, while macro-averaged all-evidence coverage remained between 99.4% and 99.8% on the selected LoCoMo and LongMemEval histories.

> Verification increasingly compensated for information omitted by the first-pass writer: its mean source-coverage gain rose from 9.0 points at W=4 to 17.0 points at W=32.

> Moving from W=16 to W=32 reduced calls by 9.3% and wall time by 11.0% on average, with no clear change in aggregate final source coverage (+0.07 points; 95% bootstrap interval [-1.34, 1.62]). We therefore use W=32 as the default construction setting in subsequent experiments.

> At W=32, Sol produced larger memories and covered 2.1 points more source turns than Terra, but required 37.2% more wall time and showed no consistent advantage across the paired QA screening subsets.

当前不能写的结论包括：W32 的 downstream accuracy 优于 W4、W8 或 W16；现有数值代表完整 LoCoMo、LongMemEval 或 BEAM benchmark；Luna 的回答质量优于其他档位；Source coverage 或 reference precision 等同于 memory 的语义正确性；两段 conversation 的差异具有 benchmark-level statistical significance。

若论文需要把 W32 从“构建默认值”提升为“质量最优或质量无损的默认值”，至少还需在同一批问题上完成 W16 与 W32 的 paired QA。正式主结果仍需在选定的 builder 和 W32 下覆盖完整 benchmark；Verify 的因果作用需要独立的 on/off ablation，而不是只依赖其在日志中增加的 source coverage。
