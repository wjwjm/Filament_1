# CPU 优化统一开发入口

新对话请选择分支 `codex/cpu-optimization`。
推荐直接打开本地工作树 `D:/Filament_1/.worktrees/cpu_optimization`。

## 已整合范围

P1、P1R、P1R2、O1、O1/IR、Q1、Q2、Q3 的全部已提交历史均为
`66801ad435d6ec1967338cee4eb82af2a152389c` 的祖先，无需重复合并。
本分支在该提交上保存 S4 当前 17 个 Python 工具/测试文件，原始字节来自
`D:/Filament_1/.worktrees/cpu_s4/Filament_python/tools/cpu_o1/cpu_s4`。
旧分支及工作树保留。O1/IR 的未提交旧版本已与47c993b相同，不覆盖后续Q1/Q3修补。

## 新对话如何继续

先读本文件及 `Filament_python/tools/cpu_o1/CPU_Q3_README.md`。
当前工作为 S4-1 同一候选 I 的 OFF/PROFILE 性能归因，不是新增算法或S4-2。
科学基线 HEAD 为66801ad435d6ec1967338cee4eb82af2a152389c，科学树身份
821fec5bdacd832b0347e1f2e84285cb54172e57。本次整合不改变科学代码。

运行进度与历史证据仍在旧工作树，首先读取：
`D:/Filament_1/.worktrees/cpu_s4/reports/CPU_S4_1_LAYOUT_RETRY_20261018/CONTINUE_S4.md`。
该文档包含追加更新，后面的时间戳记录优先。Job8823835是本次整合前最后确认的作业；
此处不是实时调度状态。先查看现有监控/账本，禁止重复提交或建立第二套监控。
用户授权诊断并修复基础设施后再次提交；科学非exact/非有限必须停止人工审阅。
旧作业的源码、Git HEAD、overlay身份和路径均不可用新分支身份替代。

## 历史证据入口

- S4及失败/提交账本：D:/Filament_1/.worktrees/cpu_s4/reports
- Q3：D:/Filament_1/.worktrees/cpu_q3/reports
- Q2：D:/Filament_1/.worktrees/cpu_q2/reports
- Q1：D:/Filament_1/.worktrees/cpu_q1/reports 及 D:/Filament_1/reports

大型输出、缓存、旧审阅ZIP与部署脚本不批量复制入Git；原位置保留。
本分支为开发接续入口，不自动获得新的HPC运行资格。冻结admission仍绑定原基线，
未来部署必须遵循原运行合同核验，不能直接将本分支HEAD替换到旧授权配置。
本次未push，未合并main，未改变GPU工作树，也不授予CPU N5生产资格。
