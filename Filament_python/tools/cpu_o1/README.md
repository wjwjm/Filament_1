# CPU-O1 联合本地诊断

本分支在已接受电离原型上开发 B/I/R/IR 四身份实验入口。参见 [联合合同](CPU_O1_IR_JOINT_CONTRACT.md) 和 [未提交 HPC 草案](CPU_O1_NEXT_HPC_PROPOSAL_NOT_SUBMITTED.md)。

通过 run_cpu_o1_local.ps1 -EvidenceRoot <新目录> 使用已有隔离解释器依次运行 backend、sanity、compileall、targeted。实际实现、数值差异和测试结果以独立审阅包为准。本工具不授予生产资格，无 HPC 自动提交。

原 cpu_o1_operator_contract.md 和 cpu_o1_execution_workspace_contract.md 保留第一轮身份；联合合同扩展它们。formal_api_UNAPPLIED.patch 从未应用。旧报告/ZIP保持原样。
