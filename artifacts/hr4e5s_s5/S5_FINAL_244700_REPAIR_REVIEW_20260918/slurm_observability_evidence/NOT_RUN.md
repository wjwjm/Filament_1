# NOT_RUN：站点 Slurm 可观测性

本轮没有提交 CPU-only 或 GPU Slurm 作业。因此未验证 `scontrol listpids job.step` 是否在 batch node 正确列出该节点 worker PID、空 step 的 return code、Proctrack 配置或 step 清退时序。

代码保留这些字段并 fail closed；修复审阅通过后，首次新 allocation 的 controller preflight 必须持久化实测证据。若节点本地 PID 前门失败，不得注入故障、不得进入 recovery、不得自动重提。
