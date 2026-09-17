# S5-FINAL 244700 修复审阅摘要

状态：`READY_FOR_SINGLE_ALLOCATION_REPAIR_REVIEW`。

本包实现并验证了最小的单 allocation、两轮独立科学进程方案。controller 在 allocation 内、无 GPU 运行；initial hydro worker 只在真实 claim 后进入 arming；controller 对精确目标 step 做一次节点本地 PID 证明并写 intent/receipt；initial writers 静默后才允许 snapshot/freeze/一次 bootstrap；recovery 使用新的 step/PID/epoch，最后仍由原 strict exact/provenance 条件裁决。

已完成的是本地软件与 batch 静态验证，不是 HPC 科学验收。244700 保持只读历史失败证据；没有创建 run root、没有提交 Slurm、没有申请 GPU、没有推送或合并。

网页端需裁决：

1. 是否接受该单 allocation 合同修订及其未覆盖跨 allocation/node failure 的边界；
2. 是否接受 optional scheduling callback 不改变 HR-4 科学算子或 fault-off 默认路径的 diff 审计；
3. 是否授权新 SHA 的单一、非覆盖 3-GPU allocation 与现场 node-local listpids 前门；
4. 是否同意任何前门失败都停在缺陷审阅，不自动重提。
