# GPT-5.6 单次读取长度实验跟踪表

状态：subscription smoke 已通过；误启动的 Luna 54 配置筛选阶段已停止。

| ID | 内容 | 规模 | 状态 | 模型请求 |
|---|---|---:|---|---:|
| P000 | 固定模型、`W` 网格、六个数据单元与评测协议 | 1 份计划 | DONE | 0 |
| P001 | 生成并校验 manifest 与 run matrix | 162 配置 | DONE | 0 |
| P002 | 为生成器添加本地数据、hash、矩阵测试 | 3 tests passed | DONE | 0 |
| P003 | 建立现有结果索引与新目录规则 | 文档 | DONE | 0 |
| X001 | `luna` + BEAM-100K + `W=session` pilot | 1 attempted / 0 completed；83,096 input / 17,365 output tokens | INTERRUPTED | 5 requests |
| X002 | `luna` + BEAM-100K + `W=32` pilot | 1 attempted / 0 completed；231,133 input / 32,215 output tokens | INTERRUPTED | 40 requests |
| X003 | frontier `luna` 文本及 tool-call 合同预检 | 2 requests | DONE | 模型回显一致且单次 reasoning tokens=0；不足以验证完整构建 |
| X004 | frontier `luna` + LoCoMo `conv-48` + `W=32` smoke | 1 attempted / 0 completed | FAILED | 完整构建累计 125,687 reasoning tokens；无完成标记 |
| Q001 | frontier 持久构建编排器 | 0 builds started | NOT RUNNING | 当前无 tmux 或 orchestrator 进程 |
| X005 | subscription `luna` + BEAM-100K `conv-1` + `W=32` smoke | 1 completed | DONE | 133 calls；516,452 input / 63,060 output；reasoning tokens=0 |
| Q002 | subscription Luna 持久执行器 | 1 attempted / 0 new completed | INTERRUPTED | 因错误理解授权而启动；20 requests 后停止；tmux 与 port 8204 已关闭 |
| E100 | `gpt-5.6-luna` 筛选曲线 | 1 validated smoke only | NOT STARTED | 54 配置正式筛选未获授权，不再自动执行 |
| E200 | `gpt-5.6-terra` 筛选曲线 | 54 builds | WAITING | Luna 筛选完成后启动 |
| E300 | `gpt-5.6-sol` 筛选曲线 | 54 builds | WAITING | Terra 筛选完成后启动 |
| E400 | 候选参数重复与完整 benchmark | 待筛选后确定 | WAITING | 根据筛选曲线冻结 W* 后执行 |

P000–P003 是准备工作。X001–X002 只记录已中断的调用与额度消耗，不属于正式结果。X004 位于 `results/formal/gpt56-chunk-curve-frontier-smoke-20260718/`，其失败 memory 位于该配置的 `memory.failed.*`，不能用于曲线或论文表格。X005 及 E100 位于 `results/formal/gpt56-chunk-curve-subscription-smoke-20260718/`。只有同时存在 `build.json` 和 `memory/_SUCCESS.json` 的配置才计为完成。

历史的 context-history smoke、write-interval smoke 和 GPT-5.5 正式结果保留在原目录，不能作为本矩阵中任一配置的替代结果。
