# HR-4E-5S S5 最终收口（2026-09-23）

**权威状态：** `HR-4E-5S-S5 = CLOSED / RESTART_RECOVERY_EXACT_PASS / AUTOMATION_CLOSEOUT_WARNING_RETAINED`。S5 scientific qualification 为 `PASS / RESTART_RECOVERY_EXACT_QUALIFIED`；历史 clean job `252545` 仍为 `FAILED / 1:0`，其 post-run audit 缺陷未被追溯改写。formal E5 为 `READY_FOR_DESIGN / NOT_STARTED`，本次未授权也未启动 formal E5。

## 范围与结论

S5 验证的是受控 worker-loss 后，在既定持久化状态语义内确认旧 writer 静默、由新进程完成 durable-state reconstruction、optical 与两个 hydro consumer 继续执行，并使最终状态与 fault-OFF clean reference 严格等价。它不声称已验证整个 allocation 消失后的重启、长时间耐久运行或 formal 多脉冲生产；这些属于后续 formal E5 设计与独立准入。

本次只修复本地 post-run 审计工具和整理既有证据，未重跑 GPU、clean、worker-loss、recovery 或 optical/hydro science。没有修改 `advance_hr4_single_screen`、HR-2/3/4 科学算子、Streaming lifecycle 生产语义、restart/bootstrap、ownership、barrier/promotion、exact comparator、冻结参数、source 或 LUT。

## 关键证据

| 层 | 已确认的事实 | 证据入口 |
| --- | --- | --- |
| 真实故障与安全边界 | `249485` 的 worker-loss receipt、initial workers stopped、fresh recovery bootstrap-ready 和 final lifecycle audit 已保留；recovery receipt audit 为 PASS | [S5 证据索引](S5_EVIDENCE_INDEX.json) 中 `249485` 项 |
| clean 结果 | `252545` 的科学产物已生成；修复后的本地只读终审 PASS，独立 supplemental audit r2 为 34/34 PASS | 证据索引中的 clean audit 与 supplemental r2 |
| 文件身份 | 新 clean reference 178/178 原始字节哈希通过；`249485` candidate 339/339 复核通过 | 证据索引中的 transfer verification |
| strict exact | 432/432 POST、NEXT、DEPOSITION 字段相等，`mismatch_count=0` | 证据索引中的 `exact_comparison.json` |
| 全局与生命周期 | final optical 数组/哈希相等；9/9 ledger exact；authoritative manifest、completion map、ownership、barrier、promotion、artifact inventory exact | 同一 exact comparison 报告 |
| recovery provenance | 4/4 检查 PASS；依冻结 expected-effects 合同，retry delta 与 recovery attempts 一致，终态 queue/backlog 为空 | 证据索引中的 `recovery_provenance.json` |

最终 exact 报告 SHA-256：`80a74891bb66a0324195176fa02493eb19a39393ff2af41a5e32823629a21600`。新 clean source 执行 SHA：`1751064871b6be5fd262e53b23ddb349155e5c15`。本地 comparator 版本差异已核查：科学推进、`compare_exact`、`validate_recovery_provenance` 与 compare CLI 路径未改；新增的是输入/reference 前门校验。详细身份和原始文件哈希见索引及原报告。

## 历史工程缺陷保留

- `244700`：worker-loss 可观测性缺陷。
- `245133`：Phase 0 shell quoting 缺陷。
- `249184`：bootstrap immutable snapshot 与 live queue 混用引发启动竞争。
- `249485`：后续比较所需 reference 路径/包装无效；真实恢复产物保留。
- `251660`：`FINAL_RUNNER` 未导出到 launcher 子 shell。
- `252545`：post-run final audit 调用不存在的 `StreamingLifecycle._load_artifact`；Slurm 原终态为 **`FAILED / 1:0`**。
- supplemental audit r1：错误地比较 canonical array hash 与 raw `.npy` file hash；独立 r2 改用执行版本 `sha256_array` 后 34/34 PASS，r1 原报告保留。

这些缺陷是工程历史，不再阻塞已由真实故障、fresh-process recovery 与 strict exact 证据构成的 S5 科学收口。特别地，本地审计修复只改进未来 audit workflow；不会使 `252545` 变为 `COMPLETED / 0:0`，也不能以此宣称历史 Slurm 终态 PASS。

## 移交边界

`No GPU rerun required`：candidate 与 clean 的科学计算和严格比较已完成，剩余问题是 post-run audit automation。S5 科学门已关闭；formal E5 仅进入设计准备，仍须单独审查其输入绑定、资源与存储门，不因本报告自动启动。future preflight 规则见 [S5 工程检查表](S5_ENGINEERING_PREFLIGHT_CHECKLIST.md)。
