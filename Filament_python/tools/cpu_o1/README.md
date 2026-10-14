# CPU-O1 本地实现

本目录仅为 CPU 小数组原型及合同。科学父提交 c8f3ada；工具父提交3d4cbb5。详见 operator/execution 合同及最终审阅报告。不得作为正式科学执行入口。

参考函数保持原样；compiled adapter 显式调用。Numba依赖仅在新原型路径需要。测试必须经仓库run_local_tests.ps1 -PythonExe <隔离环境>执行backend、sanity、targeted。当前完整场、HPC性能与Linux可部署性均未验证。

目录中的正式API建议patch为UNAPPLIED_DESIGN_ONLY；不得自动apply。新的接口接缝需后续独立审核，不修改当前GPU默认路径。
