# 244700 最小根因证据

已证实的是场景控制失败，而不是科学 mismatch：三个 scientific step 均 `0:0`，两个 consumer 都正常完成，随后 batch 写出 `worker_loss_not_triggered`。控制器拒绝信号时的原因是 `TARGET_STEP_PID_UNPROVEN`，且 `signal_sent=false`。

原实现由 allocation 外持久轮询器先扫描 lifecycle，再事后执行 `scontrol listpids job.step`。它没有在真实 hydro claim 与 PID 仍存活之间建立原子或事件驱动的保持关系。244700 的原始逐次命令时间线未持久化，所以“控制器仅仅轮询过慢”仍是推测，而不是已证实结论。

修复将控制器移入同一 allocation 的 batch 节点；initial hydro worker 在真实、已持久化 `HYDRO_RUNNING` claim 后、进入未修改的 HR-4 solver 前写 arming receipt 并有限等待。controller 记录节点本地 `scontrol listpids` argv/hostname/stdout/stderr/return code，只有两个不同 actor 的 receipt、claim 和 PID 同时成立时才持久化一次 intent 并向精确 `job.step` 发信号。无法证明时仍 fail closed。

未验证：本站 Slurm 的 `listpids` 行为、Proctrack 细节和同 allocation 内 step 清退语义；见 `slurm_observability_evidence/NOT_RUN.md`。
