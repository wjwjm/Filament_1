# 2026-08-24 Hybrid 0.60 m 执行效率复盘

## 结论

Hybrid 验证最终完成，但执行路径把一个“两个算例、一次提交、完成后比较”的任务
扩大成了专用 campaign framework；本地编排、第一次 Slurm 技术失败和后处理
错误重试造成了主要可避免成本。超算端 job `222025` 的 7:35:00 主要是真实的
串行计算时间，不是排队时间。

本复盘只约束后续工程执行，不改变 Hybrid 的物理/数值实现、机械分类、人类最终
结论、生产配置或既有结果 provenance。

## 量化时间线

- 首次实现相对基线新增了传播模式、测试和一套 prepare/lock/submit/batch/
  postprocess/compare 基础设施；最终分支相对基线涉及 44 个文件。
- job `222014` 在传播开始前 2 秒失败：batch 在 conda 激活前调用裸
  `python`，reference、hybrid 和 LUT warm-up 均未启动。
- job `222025` 排队约 30 秒后运行，调度器总时长 7:35:00；reference
  `16500.02 s`、hybrid `10739.61 s`，两者在一个 allocation 中严格串行。
- rollout 审计发现后处理阶段出现 70 次同签名 `apply_patch` 格式失败；两轮主要
  后处理交互约 37.8 分钟和 49.2 分钟。
- 其他重复失败包括 PowerShell 中使用 `head`、多层 SSH/Bash 变量丢失、
  Python `-c` 转义、papp token/数据库问题、SCP 解析失败，以及本地缺少
  PIL/matplotlib。

权威结果证据：

- 以下文件保存在 Git 标签 `hybrid-0p60-validation-2026-08-24`
  （commit `ebfe5c3afbeef4583f983e8dc9258c25c8d6c980`），不复制到 `main`；
  轻量索引见 `2026-08-24_hybrid_0p60_result_archive.md`。
- `Filament_python/results/hybrid_propagation_validation/technical_failure_222014.json`
- `Filament_python/results/hybrid_propagation_validation/scheduler_terminal_evidence_222025.json`
- `Filament_python/results/hybrid_propagation_validation/postprocess_222025/performance.csv`
- `Filament_python/results/hybrid_propagation_validation/FINAL_REPORT_222025.md`

## 根因与永久控制

| 根因 | 永久控制 |
| --- | --- |
| L0 任务升级为专用框架开发 | 全局和项目 AGENTS 增加 L0/L1/L2 分级；L0 优先复用已有 runner/wrapper/postprocess |
| 本地测试未覆盖真实 batch 前导 | `audit_batch_entry.py` 在 run/lock/`sbatch` 前执行 `bash -n` 并拒绝激活前裸 Python |
| 同一 patch/tool 错误被重复提交 | 同一工具、阶段和错误签名第二次触发硬熔断 |
| PowerShell/WSL/SSH/Bash 多层转义脆弱 | 第一次转义/变量丢失后立即改用固定脚本和参数数组/manifest |
| 用 SSH/SCP 重试诊断认证 | 只做一次权威 `acct` 检查，再进入登录/数据库恢复 |
| 已有后处理未被优先复用 | postprocess-only 固定为终态、远端已有脚本、产物检查、comparison、整目录同步 |
| 严格配对真实成本未提前说明 | 提交前报告 reference/candidate 串并行关系和历史 wall-time 估算 |

## 后续验收边界

- 简单双算例执行不得因为“可能以后复用”而新建框架。
- 新生产 batch 未通过 batch-entry audit 时不得产生远端副作用。
- postprocess-only 不修改传播核心、生产配置或冻结结果。
- 软件门禁通过不等同于数值/物理正确；科学结论仍由 Sol 和用户验收。
- 本次改造不重跑 Hybrid、不连接 HPC、不更改或覆盖已有结果。
