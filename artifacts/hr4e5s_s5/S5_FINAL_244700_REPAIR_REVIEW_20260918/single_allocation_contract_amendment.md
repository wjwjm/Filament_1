# S5-FINAL 单 allocation 合同修订草案

`execution_mode = SINGLE_ALLOCATION_TWO_EXECUTION_EPOCHS`。

初始与 recovery 不再分别申请 batch job；它们必须是同一 allocation 内、不同 Slurm step、不同 PID、不同 execution epoch 的科学进程。`recovery_is_new_allocation=false`，`recovery_is_fresh_process_restart=true`。不得伪造新的 recovery job ID。

初始 epoch 的唯一故障信号前必须存在：controller READY、三个 identity、两个不同 hydro actor 的真实 arming claim、target PID 的节点本地 `scontrol listpids` 证据、唯一 signal intent。初始所有 scientific writer 退出且 listpids 证明静默后，才允许冻结 inventory/effects、重建一次并启动 recovery epoch。

最终 PASS 仍要求 recovery scientific steps 正常、48/48 生命周期完整、ownership/barrier/promotion 通过、432/432 exact comparisons、零 mismatch 和 strict recovery provenance PASS。该模式不覆盖 allocation/node 故障或跨 allocation 重启；`cross_allocation_restart_tested=false`。
