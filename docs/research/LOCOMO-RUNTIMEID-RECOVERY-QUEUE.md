# LoCoMo Runtime-ID 恢复队列

更新时间：2026-08-08

本文件记录 `runtimeid-r1` build-only 扩展实验中未完成的任务，以及后续运行顺序。不得将失败任务误计为完整样本，也不得绕过 LoCoMo evaluator lock 启动评分。

## 当前队列

### 1. conv-26：加入下一组五路并发

- 运行目录：`/home/qi2/scriptorium-runs/locomo-5way-lane1-conv26-token15k-t120-runtimeids-r1`
- checkpoint：2/2 batches、19/19 sessions、419/419 source turns。
- 未完成阶段：`final_management=pending`。
- 失败原因：PackyAPI 调用连接被重置，`ECONNRESET`。
- 判断：属于临时 API/网络连接失败，不是 WSL、ext4、Shell 或文件权限错误。
- 后续动作：下一次启动五路并发时，将本任务作为其中一路以 `resume=true` 恢复；其余四路可运行新样本。
- 恢复前检查：sample ID、model、writer protocol hash、batch-plan hash 和 checkpoint schema 必须全部匹配。禁止删除原 checkpoint 或在原目录执行全新构建。
- 完成条件：`status=complete` 且 `final_management=complete`，随后核验组件与引用完整性。

### 2. conv-43：五路结束后单独诊断

- 运行目录：`/home/qi2/scriptorium-runs/locomo-5way-lane4-conv43-token15k-t120-runtimeids-r1`
- checkpoint：2/3 batches、19/29 sessions、433/680 source turns。
- 未提交阶段：batch 3，输入约 14,241 tokens。
- 失败原因：`TopicFormatError: duplicate footnote definition: e-2992cc900e`。
- 现场：已生成 writer failure snapshot；失败批次没有提交 checkpoint。
- 判断：属于模型输出的 Topic 脚注定义冲突及修复未通过校验，不是 WSL、ext4、Shell 或权限错误。
- 后续动作：不要混入下一组五路直接盲目重试。等并发组结束后，单独只读审计 failure snapshot、batch-3 live audit 和 Topic 格式校验路径；确认修复策略后再从 2/3 checkpoint 恢复。
- 注意：恢复时预计会重做尚未提交的 batch 3；已提交的前两批不得重做。

#### 2026-08-08 诊断结论

- failure audit 显示第一次提交因重复稳定脚注 `e-a58f5fbb2e` 被拒绝；事务已正确回滚。
- 自动修复随后再次提交，但又产生重复稳定脚注 `e-2992cc900e`，因此 repair 被拒绝并保存 snapshot。
- 根因位于 Runtime-ID 归一化：模型在同一 trajectory 的多个文件编辑中会重新从 `e1` 编号，旧实现按裸标签全局映射，可能把不同证据归一成同一个稳定脚注 ID。
- 已修改 `code/src/management/topic_normalization.py`：本地脚注标签现在按文件内定义出现次序分配稳定 ID；同一文件重复使用 `e1`、以及不同文件各自使用 `e1`，均不会相互碰撞。
- 已在 `code/tests/management/test_memory.py` 增加同文件重复标签和跨文件重复标签两条回归测试。
- WSL 验证结果：管理定向测试 56 passed；合并远端 orphan-citation/on-disk 修复并补齐 `posix-bash` 后，Markdown、management 和 current runtime 相关回归共 106 passed。
- 本次修改没有改变 prompt/tool 定义，因此 writer protocol hash 不变；原 checkpoint 仍满足协议匹配要求。

## 下一次推荐调度

1. 启动前先确认当前没有旧 wrapper 或重复恢复进程。
2. 若采用六路验证：一路恢复 `conv-26` final management，一路从 2/3 checkpoint 恢复 `conv-43`，另外四路分配 `conv-47/48/49/50`。
   四个新配置已创建，所有恢复与新建配置均显式使用 `shell_backend: posix-bash`。
3. 监控必须区分“新增 batch/session/source turns”和“final management 活动”；`conv-26` 不会再增加 source turns，不能据此误报停滞。
4. 五路全部结束后汇总完成样本、成本、agent turns 和异常。
5. 六路启动后的前 5 分钟重点检查 `conv-43` 是否再次出现重复脚注，以及全局是否出现 429、排队或持续 timeout；若异常，只停靠异常 lane，不影响其他五路。

## 已完成且无需重跑

- `conv-41`：3/3 batches、32/32 sessions、663/663 source turns，complete。
- `conv-42`：3/3 batches、29/29 sessions、629/629 source turns，complete。
- `conv-44`：3/3 batches、28/28 sessions、675/675 source turns，complete。

这些结果属于同一 `runtimeid-r1` cohort，可以保留使用；恢复失败 lane 不要求重跑已完成 lane。
