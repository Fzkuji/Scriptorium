# v7 Agent Self-Organized Memory — Progress Ledger

Plan: docs/superpowers/plans/2026-07-07-agent-self-organized-memory.md
Branch: feature/v7-agent-memory
Base commit: b64b434

## Tasks
- Task 1: complete (commits b64b434..2b2b4e5, review clean)
- Task 2: complete (commits 2b2b4e5..86397b2, review clean)
- Task 3: complete (commits 86397b2..8332a90, review clean)
- Task 4: complete (commits 8332a90..5b60374, 3 commits incl 2 review fixes: .md dup bug + regression test validity; review clean)
- Task 5: complete (commits 5b60374..9e73f98, review clean)
- Task 6: complete (commits 9e73f98..e8b0ec2, review clean)
- Task 7: complete (commits e8b0ec2..0da9cfd, 2 commits incl assert hardening; SUMMARY_PAGE on/off switch implemented+tested; review clean)
- Task 8: complete (commits 0da9cfd..5e8684f, review clean, no findings)
- Task 9: complete (commits 5e8684f..e436011, 2 commits incl raw-leak fix hide_raw=True + real-bash regression test; review clean)
- Task 10: complete (commit 4155b27..95a9574 smoke script; 冒烟三条硬验收全过：模型自建 Caroline.md/Melanie.md、三行块带 [source](D1:x)、3 问检索有结果)
  - 冒烟暴露真 bug + 已修（commit b17eb2c）：_run_store_agent 15 轮循环无终止收敛，弱模型 qwen3.6-flash 每轮重复 cat>> 同一批 facts，导致 D1:12/13/18 在 Melanie.md 逐字重复 4 次。修法：写过一次后注入"去重+输出 summary 结束"提醒 + max_rounds 15→6。test_v7_store + 全套 23 测试通过。重跑冒烟确定性验证：17 条 source 零重复（uniq -d=0），修前是 3 条各重复 4 次。内容覆盖两 session（D1×6 + D2×15），模型自组织出 2 个 ## 主题分节，无内容丢失。DONE。

## 最终全分支 review（opus 4.8，b64b434..b17eb2c）：READY-WITH-MINORS，无 Critical
核心约束全 clean：no-original 完整（每条检索路径都 hide_raw + 输出过滤，双层防护）、模型自组织（无硬编码 people/）、dia_id 锚（10 样本 5882 turn 0 缺失）、去重修复正确、v5/v6 无回归。
review 的 4 项发现已全部修复（commit 4cb4113，26 passed）：
1. [Important] _collect_memories_v7 的 json.loads 无守卫 → 加 try/except（弱模型畸形 tool-call 参数不再让整题检索崩）+ 回归测试
2. [Important] running_summary 死参数 → 接进 _AGENT_STORE_PROMPT 的「## 上文」段（跨 chunk 滚动上文），空则填「（无）」+ 测试
3. [Minor] _top_level_view 加 NotADirectoryError 兜底
4. [Minor] with-original 测试补 dia_id 保留断言
遗留（安全、可延后，最终 review 判定不阻塞合并）：v7 build 仍建空 raw/ 目录（内容为空，无泄漏；建议后续用 NATIVEMEM_PROMPT!=v7 守卫 makedirs）。

## Minor findings (for final review)
- Task 2: _top_level_view 不处理 memory_dir 是文件的情况（NotADirectoryError 未捕获）— brief 范围外，实际 memory_dir 恒为目录，最终 review 定夺

## 后续实验（实现全通后跑）
- SUMMARY_PAGE on/off：Task 7 的 reorganize_library 必须实现 NATIVEMEM_SUMMARY_PAGE=on|off 分支（prompt 里加不加"拆子目录后留同名摘要页+链接"）。等 Task 10 冒烟通过后，同批数据 build 两次比 LoCoMo 分数 + 检索跳数 + 弱模型摘要同步质量。用户明确要求做这个对比实验。
- Task 6 Minor: _run_store_agent 无-tool_calls 分支的 messages.append 写法与有-tool_calls 分支不统一（不影响行为，reviewer 已披露）
- Task 6 观察: running_summary 参数签名有但函数体未用（brief 参考代码即如此）→ 意味着跨 chunk 滚动总结当前没注入 agent prompt。Task 8 或最终 review 确认：要么确定不需要，要么补 prompt 注入

## 设计/实现不一致（最终 review 定夺）
- 设计文档 §13 说 v7 "废弃 save_to_raw_archive、去掉 hide_raw（记忆里没原文）"，但实际 build_memory 对所有版本（含 v7）仍无条件建 memory_dir/raw/ 归档原文。→ Task 9 检索因此必须 hide_raw=True 才能保 no-original 口径（已在 Task 9 fix 修）。根本问题：要么 v7 build_memory 真的不建 raw/（对齐设计），要么承认 raw/ 存在、所有检索路径都 hide_raw。当前是后者的补丁。最终 review 决定是否要让 v7 build 彻底不建 raw/。
